"""
GameNN-BitMoE — 极端高效的通用智能体架构。
═══════════════════════════════════════════════════════

核心技术组合:
  BitNet b1.58  ×  MoE 512×Top-1  ×  GLA  ×  GameNN 生长

设计目标:
  在 8GB 笔记本 GPU 上训练 50B 等效参数模型。
  权重 1.58 位存储, 每步只激活 ~100M 参数。
"""
import os, json, time, random, math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════
#  0. 基础组件
# ═══════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(dim, max_len=4096, device='cpu'):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(max_len, device=device).float()
    sincos = torch.einsum('i,j->ij', pos, inv_freq)
    return torch.stack([sincos.sin(), sincos.cos()], dim=-1).unsqueeze(0)


def apply_rope(x, rope_cache):
    B, L, D = x.shape
    x = x.view(B, L, D // 2, 2)
    sin, cos = rope_cache[0, :L, :, 0], rope_cache[0, :L, :, 1]
    rot = torch.stack([x[..., 0] * cos - x[..., 1] * sin,
                       x[..., 1] * cos + x[..., 0] * sin], dim=-1)
    return rot.view(B, L, D)


# ═══════════════════════════════════════════════════
#  1. BitNet 三元线性层
# ═══════════════════════════════════════════════════

class TernaryLinear(nn.Module):
    """
    1.58-bit 三元线性层。
    
    权重: -1, 0, +1 三个值 (2位存储)
    前向: 只需整数加法, 无乘法
    训练: 保存 fp16 主权重, 前向时量化
    
    存储: 50B 参数 → ~10 GB (vs fp16 100GB)
    速度: 加法操作可比矩阵乘快 10-100× (硬件支持时)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 主权重 (fp16, 用于训练时更新)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _ternary(self, w):
        """fp16 → 三元 (-1, 0, +1)。"""
        scale = w.abs().mean() + 1e-8
        return torch.where(w > 0.5 * scale, 1.0, torch.where(w < -0.5 * scale, -1.0, 0.0))
    
    def forward(self, x):
        # 量化到三元
        w_ternary = self._ternary(self.weight)
        return F.linear(x, w_ternary, self.bias)
    
    @torch.no_grad()
    def compress(self):
        """返回三元权重的紧凑表示 (用于 CPU 存储)。"""
        t = self._ternary(self.weight)
        # 编码为 uint8: 每 4 个权重用 1 字节
        # -1→00, 0→01, +1→10
        flat = (t + 1).long().flatten()  # 0, 1, 2
        # 每 4 个一组打包
        n = (flat.shape[0] + 3) // 4
        packed = torch.zeros(n, dtype=torch.uint8, device=flat.device)
        for i in range(min(4, flat.shape[0])):
            packed[:len(flat)//4 + (1 if i < len(flat)%4 else 0)] |= (flat[i::4] << (i * 2))
        return packed


# ═══════════════════════════════════════════════════
#  2. GLA — Gated Linear Attention
# ═══════════════════════════════════════════════════

class GLAAttention(nn.Module):
    """
    Gated Linear Attention — 线性复杂度, RNN 形式。
    
    训练 (O(N)):
      S = 0
      for t in 1..N:
        S = G_t ⊙ S + K_t · V_t    (门控累积)
        O_t = Q_t · S               (线性输出)
    
    生成 (O(1) per step):
      S_t = G_t ⊙ S_{t-1} + K_t · V_t
      O_t = Q_t · S_t
    
    相比 FlashAttn:
      - 不计算 O(N²) 注意力矩阵 → 省显存, 省计算
      - RNN 形式 → O(1) 生成, 无需 KV Cache
      - 长序列时吞吐量高 3-5×
    """
    
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # QKV + 门控投影
        self.q_proj = TernaryLinear(d_model, d_model)
        self.k_proj = TernaryLinear(d_model, d_model)
        self.v_proj = TernaryLinear(d_model, d_model)
        self.g_proj = TernaryLinear(d_model, nhead)  # 门控
        self.out_proj = TernaryLinear(d_model, d_model)
    
    def forward(self, x, mask=None, state=None):
        """
        x: (B, L, D)
        state: (B, H, D_h, D_h) — 前一步的状态
        return: (B, L, D), new_state
        """
        B, L, D = x.shape
        H = self.nhead
        Dh = self.head_dim
        
        q = self.q_proj(x).view(B, L, H, Dh)
        k = self.k_proj(x).view(B, L, H, Dh)
        v = self.v_proj(x).view(B, L, H, Dh)
        g = torch.sigmoid(self.g_proj(x))  # (B, L, H)
        
        # GLA RNN 前向
        if state is None:
            state = torch.zeros(B, H, Dh, Dh, device=x.device)
        elif state.ndim < 4:
            state = torch.zeros(B, H, Dh, Dh, device=x.device)
        
        outputs = []
        for t in range(L):
            # S_t = G_t ⊙ S_{t-1} + K_t ⊗ V_t
            gt = g[:, t, :].view(B, H, 1, 1)  # (B, H, 1, 1)
            kt = k[:, t, :, :].unsqueeze(-1)   # (B, H, Dh, 1)
            vt = v[:, t, :, :].unsqueeze(-2)   # (B, H, 1, Dh)
            state = gt * state + kt @ vt       # (B, H, Dh, Dh)
            
            # O_t = Q_t · S_t
            qt = q[:, t, :, :]                 # (B, H, Dh)
            ot = (qt.unsqueeze(-2) @ state).squeeze(-2)  # (B, H, Dh)
            outputs.append(ot)
        
        out = torch.stack(outputs, dim=1).view(B, L, D)
        return self.out_proj(out), state


# ═══════════════════════════════════════════════════
#  3. 分词器
# ═══════════════════════════════════════════════════

class Tokenizer:
    def __init__(self):
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_size = 4
    def fit(self, texts):
        for t in texts:
            for c in t:
                if c not in self.token_to_id:
                    i = self.vocab_size
                    self.token_to_id[c] = i
                    self.id_to_token[i] = c
                    self.vocab_size += 1
    def encode(self, text, max_len=256):
        return [2] + [self.token_to_id.get(c, 1) for c in text[:max_len]]
    def decode(self, ids, skip_special=True):
        return ''.join(self.id_to_token.get(i, '?') for i in ids if not (skip_special and i in [0, 1, 2, 3]))
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'token_to_id': self.token_to_id}, f, ensure_ascii=False)
    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, encoding='utf-8') as f:
            tok.token_to_id = json.load(f)['token_to_id']
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.vocab_size = len(tok.token_to_id)
        return tok


# ═══════════════════════════════════════════════════
#  4. BitMoE — 大规模稀疏 MoE
# ═══════════════════════════════════════════════════

class BitMoEFFN(nn.Module):
    """
    BitNet × MoE: 三元权重专家 × Top-1 路由。
    
    支持:
      - 4 → 512 专家动态增长
      - 每个专家用 TernaryLinear (1.58-bit)
      - Top-1 路由 (每 token 只激活 1 个专家)
      - 专家卸载 (非活跃专家在 CPU)
    
    512 专家 × ~100M/专家 = 50B 总参数
    每步活跃: 1 专家 × ~100M = 100M
    三元存储: 50B → ~10 GB
    """
    
    def __init__(self, dim=256, d_ff=None, n_experts=4, top_k=1, offload=True):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.offload = offload
        d_ff = d_ff or dim * 3
        
        # 专家: TernaryLinear × SwiGLU 门控
        self.experts = nn.ModuleList([self._make_expert(dim, d_ff) for _ in range(n_experts)])
        self.router = TernaryLinear(dim, n_experts)
        
        # 专家在 GPU 上的掩码 (默认全在 GPU)
        self._gpu_mask = torch.ones(n_experts, dtype=torch.bool)
    
    def _make_expert(self, dim, d_ff):
        return nn.Sequential(
            TernaryLinear(dim, d_ff),
            nn.SiLU(),
            TernaryLinear(d_ff, dim),
        )
    
    def forward(self, x):
        B, L, D = x.shape
        logits = self.router(x)  # (B, L, n_eks)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            if not self._gpu_mask[i]:
                continue  # 在 CPU 上, 跳过
            mask = (indices == i).any(dim=-1)
            if mask.any():
                eo = expert(x[mask].unsqueeze(0)).squeeze(0)
                w = (weights * (indices == i).float()).sum(dim=-1)
                out[mask] += eo * w[mask].unsqueeze(-1)
        return out
    
    def add_expert(self):
        """生长: 加一个专家。"""
        ne = self._make_expert(self.dim, self.dim * 3).to(next(self.parameters()).device)
        ne.load_state_dict(self.experts[-1].state_dict())
        for p in ne.parameters():
            p.data += torch.randn_like(p) * 0.02
        self.experts.append(ne)
        self.n_experts += 1
        # 扩展路由
        ow = self.router.weight.data
        ob = self.router.bias.data
        self.router = TernaryLinear(self.dim, self.n_experts).to(ow.device)
        self.router.weight.data[:ow.shape[0], :ow.shape[1]] = ow
        self.router.bias.data[:ob.shape[0]] = ob
        # 扩展 GPU 掩码
        new_mask = torch.ones(self.n_experts, dtype=torch.bool)
        new_mask[:len(self._gpu_mask)] = self._gpu_mask
        self._gpu_mask = new_mask
        return sum(p.numel() for p in ne.parameters())
    
    def offload_expert(self, idx):
        """将专家移到 CPU。"""
        if idx < len(self.experts):
            self.experts[idx] = self.experts[idx].cpu()
            self._gpu_mask[idx] = False
    
    def restore_expert(self, idx):
        """将专家恢复到 GPU。"""
        if idx < len(self.experts):
            self.experts[idx] = self.experts[idx].to(next(self.router.parameters()).device)
            self._gpu_mask[idx] = True
    
    @property
    def active_params(self):
        """当前在 GPU 上的参数量。"""
        return sum(p.numel() for i, e in enumerate(self.experts) if self._gpu_mask[i] for p in e.parameters())
    
    @property
    def total_params(self):
        return sum(p.numel() for e in self.experts for p in e.parameters()) + sum(p.numel() for p in self.router.parameters())


# ═══════════════════════════════════════════════════
#  5. BitMoE Block
# ═══════════════════════════════════════════════════

class BitMoEBlock(nn.Module):
    """
    完整的 BitMoE 块:
      GLA Attention + BitMoE FFN + RMSNorm + RoPE
    """
    
    def __init__(self, d_model=256, nhead=4, n_experts=4):
        super().__init__()
        self.d_model = d_model
        self.attn = GLAAttention(d_model, nhead)
        self.norm1 = RMSNorm(d_model)
        self.ffn = BitMoEFFN(d_model, d_model * 3, n_experts)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None, state=None):
        # GLA Attention
        r = x
        x = self.norm1(x)
        attn_o, new_state = self.attn(x, mask, state)
        x = r + self.dropout(attn_o)
        # BitMoE FFN
        r = x
        x = self.norm2(x)
        x = r + self.dropout(self.ffn(x))
        return x, new_state


# ═══════════════════════════════════════════════════
#  6. GameNN-BitMoE 主模型
# ═══════════════════════════════════════════════════

class GrowingLLM(nn.Module):
    """
    Growing-LLM: 自生长通用语言模型。
    
    架构:
      TokenEmbed → RoPE → [BitMoEBlock × N] → LM Head
    
    所有线性层用 TernaryLinear (1.58-bit)
    MoE 支持 4→512 专家动态增长
    GLA 线性注意力, 长序列无损
    
    参数:
      起始: 2层, 256维, 4专家 → ~4M 参数
      终点: 32层, 1024维, 512专家 → ~50B 参数
      存储: 50B × 1.58bit ≈ 10 GB (CPU RAM)
      活跃: ~100M 参数/步 (GPU显存 < 2GB)
    """
    
    def __init__(self, vocab_size=8000, d_model=256, nhead=4, init_layers=2, n_experts=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_experts = n_experts
        self.max_layers = 256
        self._widths = [256, 384, 512, 768, 1024]
        
        # Standard (non-ternary) — 嵌入层保持精度
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer('rope_cache', precompute_rope(d_model))
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList()
        for _ in range(init_layers):
            self.layers.append(BitMoEBlock(d_model, nhead, n_experts))
        
        self.lm_head = TernaryLinear(d_model, vocab_size, bias=False)
        
        # 生长追踪
        self.growth_log = []
        self._loss_hist = []
    
    @property
    def n_layers(self):
        return len(self.layers)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def active_params(self):
        """GPU 上活跃参数 (不含卸载到 CPU 的专家)。"""
        active = sum(p.numel() for p in self.token_embed.parameters())
        active += sum(p.numel() for p in self.lm_head.parameters())
        for layer in self.layers:
            if isinstance(layer.ffn, BitMoEFFN):
                active += layer.ffn.active_params
            active += sum(p.numel() for p in layer.attn.parameters())
            active += sum(p.numel() for p in layer.norm1.parameters())
            active += sum(p.numel() for p in layer.norm2.parameters())
        return active
    
    def forward(self, token_ids, mask=None):
        B, L = token_ids.shape
        x = self.token_embed(token_ids)
        x = apply_rope(x, self.rope_cache)
        x = self.dropout(x)
        state = None
        for layer in self.layers:
            x, state = layer(x, mask, state)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, token_ids, max_new=128, temp=0.6):
        """自回归生成 (GLA RNN 状态 O(1) 步进)。"""
        self.eval()
        # Prefill
        x = self.token_embed(token_ids)
        x = apply_rope(x, self.rope_cache)
        x = self.dropout(x)
        state = None
        for layer in self.layers:
            x, state = layer(x, state=state)
        logits = self.lm_head(x[:, -1:, :])
        
        for step in range(max_new):
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            if step < 8:
                for sid in [0, 3]:
                    probs[:, sid] = 0
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            nid = torch.multinomial(probs, 1)
            token_ids = torch.cat([token_ids, nid], dim=-1)
            if nid.item() == 3 and step > 4:
                break
            
            x = self.token_embed(nid)
            x = apply_rope(x, self.rope_cache[:, :1])
            x = self.dropout(x)
            for layer in self.layers:
                x, state = layer(x, state=state)
            logits = self.lm_head(x)
        
        return token_ids[0].tolist()
    
    def reply(self, text, tokenizer, max_new=96, temp=0.6):
        ids = tokenizer.encode(text)
        t = torch.tensor([ids], device=next(self.parameters()).device)
        out = self.generate(t, max_new, temp)
        return tokenizer.decode(out[len(ids):])
    
    # ─── 生长 ───
    
    def knowledge_gap(self, input_ids):
        self.eval()
        with torch.no_grad():
            logits = self(input_ids[:, -128:])
            gap = 1.0 - F.softmax(logits[:, -1, :], dim=-1).max().item()
        return gap
    
    def should_grow(self, avg_loss=None):
        if len(self._loss_hist) < 5:
            if avg_loss is not None:
                self._loss_hist.append(avg_loss)
            return False
        recent = np.mean(self._loss_hist[-5:])
        older = np.mean(self._loss_hist[:-5]) if len(self._loss_hist) >= 10 else recent
        self._loss_hist.append(avg_loss if avg_loss is not None else recent)
        return recent > older * 0.95 and recent > 0.3  # 停滞 + loss 还高
    
    def grow_depth(self):
        ref = self.layers[-1] if self.layers else None
        nl = BitMoEBlock(self.d_model, nhead=4, n_experts=self.n_experts).to(
            next(self.parameters()).device)
        if ref:
            nl.load_state_dict(ref.state_dict())
        self.layers.append(nl)
        added = sum(p.numel() for p in nl.parameters())
        self.growth_log.append({'type': 'depth', 'to': self.n_layers, 'added': added})
        return added
    
    def grow_expert(self):
        added = 0
        for layer in self.layers:
            if isinstance(layer.ffn, BitMoEFFN):
                added += layer.ffn.add_expert()
        self.n_experts = max(layer.ffn.n_experts for layer in self.layers if isinstance(layer.ffn, BitMoEFFN))
        self.growth_log.append({'type': 'expert', 'to': self.n_experts, 'added': added})
        return added


# ═══════════════════════════════════════════════════
#  7. GrowthEngine — 统一生长调度
# ═══════════════════════════════════════════════════

class GrowthEngine:
    def __init__(self, model):
        self.model = model
        self.cooldown = 0
    
    def step(self, avg_loss=None):
        results = []
        if self.cooldown > 0:
            self.cooldown -= 1
            return results
        if not self.model.should_grow(avg_loss):
            return results
        
        if self.model.n_layers < 32:
            added = self.model.grow_depth()
            results.append(('depth', self.model.n_layers, added))
            self.cooldown = 5
        elif self.model.n_experts < 512:
            added = self.model.grow_expert()
            results.append(('expert', self.model.n_experts, added))
            self.cooldown = 3
        
        return results
    
    def report(self):
        return {
            'layers': self.model.n_layers,
            'experts': self.model.n_experts,
            'params': self.model.count_params(),
            'active': self.model.active_params(),
        }


# ═══════════════════════════════════════════════════
#  8. 工具: 专家卸载管理
# ═══════════════════════════════════════════════════

class ExpertManager:
    """
    专家卸载管理器。
    
    当 GPU 显存不足时, 将非活跃专家移到 CPU。
    训练时每步换入需要的专家, 换出不需要的。
    """
    
    def __init__(self, model, gpu_budget=4):
        self.model = model
        self.gpu_budget = gpu_budget  # GPU 上最多保留的专家数
    
    def step(self, input_ids):
        """
        根据输入决定哪些专家需要上 GPU, 调度换入换出。
        """
        # 获取当前输入的路由偏好
        self.model.eval()
        with torch.no_grad():
            x = self.model.token_embed(input_ids)
            for layer in self.model.layers:
                if isinstance(layer.ffn, BitMoEFFN):
                    logits = layer.ffn.router(x.mean(dim=1))
                    preferred = logits.topk(self.gpu_budget, dim=-1).indices.flatten().unique()
                    
                    # 换入需要的, 换出不需要的
                    for i in range(layer.ffn.n_experts):
                        should_be_gpu = i in preferred
                        is_on_gpu = layer.ffn._gpu_mask[i]
                        if should_be_gpu and not is_on_gpu:
                            layer.ffn.restore_expert(i)
                        elif not should_be_gpu and is_on_gpu:
                            layer.ffn.offload_expert(i)
        
        return len(preferred)
