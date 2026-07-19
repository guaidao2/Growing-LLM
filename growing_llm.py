"""
GameNN-Growing — 统一通用智能体框架。
═══════════════════════════════════════════

GameNN 系列核心架构,整合:
  ├── GrowingLLM   : 自生长语言模型 (深度+宽度双维度)
  ├── MoEFFN       : 多专家前馈 (动态增减专家)
  ├── GrowthEngine : 统一生长调度器 (知识盲区检测)
  ├── DomainEvolver: 多领域顺序学习 (经验回放)
  └── CrossDomain  : 跨领域范式迁移 (注意力模式复用)

设计哲学:
  不要一开始就定死参数量,让模型自己决定什么时候长大。
"""
import os, json, time, random, math
from collections import deque, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ═══════════════════════════════════════════════════
#  0. 基础工具
# ═══════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS LayerNorm — 去均值, 速度+15%。"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
    def expand(self, new_dim):
        w = self.weight.data
        nw = torch.ones(new_dim)
        nw[:len(w)] = w
        self.weight = nn.Parameter(nw)


def precompute_rope(dim, max_len=4096, base=10000.0, device='cpu'):
    """RoPE 旋转编码 — 比 Learned PE 外推性更好。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
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
#  1. 分词器
# ═══════════════════════════════════════════════════

class Tokenizer:
    """字符级分词器 (中文用字符级最优, 无需BPE)。"""
    def __init__(self, vocab=None):
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.vocab_size = 4
        if vocab:
            for c in vocab:
                self._add(c)
    
    def _add(self, c):
        if c not in self.token_to_id:
            i = self.vocab_size
            self.token_to_id[c] = i
            self.id_to_token[i] = c
            self.vocab_size += 1
    
    def fit(self, texts):
        for t in texts:
            for c in t:
                self._add(c)
    
    def encode(self, text, max_len=256, add_eos=False):
        ids = [self.token_to_id.get(c, 1) for c in text[:max_len]]
        if add_eos:
            return [2] + ids + [3]
        return [2] + ids
    
    def decode(self, ids, skip_special=True):
        chars = []
        for i in ids:
            if skip_special and i in [0, 1, 2, 3]:
                continue
            chars.append(self.id_to_token.get(i, '?'))
        return ''.join(chars)
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'token_to_id': self.token_to_id}, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        tok = cls()
        tok.token_to_id = d['token_to_id']
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.vocab_size = len(tok.token_to_id)
        return tok


# ═══════════════════════════════════════════════════
#  2. 核心模块
# ═══════════════════════════════════════════════════

class FlashAttn(nn.Module):
    """Flash Attention — PyTorch SDPA 封装。"""
    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0, f'{d_model} % {nhead} != 0'
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None, kv_cache=None):
        B, L, D = x.shape
        qkv = self.in_proj(x).view(B, L, 3, self.nhead, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if kv_cache is not None:
            ck, cv = kv_cache
            if ck is not None:
                k = torch.cat([ck, k], dim=2)
                v = torch.cat([cv, v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=0.0 if not self.training else 0.0)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        return self.out_proj(out), (k, v)
    
    def expand(self, old_dim, new_dim):
        self.head_dim = new_dim // self.nhead
        self.in_proj = nn.Linear(new_dim, new_dim * 3).to(self.in_proj.weight.device)
        self.out_proj = nn.Linear(new_dim, new_dim).to(self.out_proj.weight.device)


class SwiGLU(nn.Module):
    """SwiGLU 前馈 — LLaMA 级。"""
    def __init__(self, dim, hidden=None):
        super().__init__()
        h = hidden or dim * 3
        self.w1 = nn.Linear(dim, h)
        self.w2 = nn.Linear(dim, h)
        self.w3 = nn.Linear(h, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    def expand(self, old_dim, new_dim):
        for n in ['w1', 'w2', 'w3']:
            w = getattr(self, n)
            is_out = 'w3' in n
            nw = torch.zeros(new_dim if is_out else w.weight.shape[0],
                           new_dim if not is_out else w.weight.shape[1])
            nb = torch.zeros(new_dim if is_out else w.bias.shape[0])
            md0, md1 = min(w.weight.shape[0], nw.shape[0]), min(w.weight.shape[1], nw.shape[1])
            nw[:md0, :md1] = w.weight.data[:md0, :md1]
            nb[:md0] = w.bias.data[:md0]
            setattr(self, n, nn.Linear(nw.shape[1], nw.shape[0]))
            getattr(self, n).weight.data = nw
            getattr(self, n).bias.data = nb


class MoEFFN(nn.Module):
    """MoE 前馈 — 多专家 + Router。"""
    def __init__(self, dim=192, expert_dim=None, n_experts=4, top_k=2):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        ed = expert_dim or dim * 2
        self.experts = nn.ModuleList([SwiGLU(dim, ed) for _ in range(n_experts)])
        self.router = nn.Linear(dim, n_experts)
    
    def forward(self, x):
        B, L, D = x.shape
        logits = self.router(x)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                eo = exp(x[mask].unsqueeze(0)).squeeze(0)
                w = (weights * (indices == i).float()).sum(dim=-1)
                out[mask] += eo * w[mask].unsqueeze(-1)
        return out
    
    def add_expert(self):
        ne = SwiGLU(self.dim, self.dim * 2).to(next(self.parameters()).device)
        ne.load_state_dict(self.experts[-1].state_dict())
        for p in ne.parameters(): p.data += torch.randn_like(p) * 0.02
        self.experts.append(ne)
        self.n_experts += 1
        ow, ob = self.router.weight.data, self.router.bias.data
        self.router = nn.Linear(self.dim, self.n_experts).to(ow.device)
        self.router.weight.data[:ow.shape[0], :ow.shape[1]] = ow
        self.router.bias.data[:ob.shape[0]] = ob
        return sum(p.numel() for p in ne.parameters())
    
    @property
    def n_active(self):
        return self.n_experts


class TransformerBlock(nn.Module):
    """Transformer 块 — FlashAttn + SwiGLU/MoEFFN + RoPE。"""
    def __init__(self, d_model=256, nhead=4, use_moe=False, n_experts=4):
        super().__init__()
        self.d_model = d_model
        self.attn = FlashAttn(d_model, nhead)
        self.norm1 = RMSNorm(d_model)
        self.ffn = MoEFFN(d_model, expert_dim=d_model*2, n_experts=n_experts) if use_moe else SwiGLU(d_model, d_model*3)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None, kv_cache=None):
        r = x
        x = self.norm1(x)
        attn_o, nkv = self.attn(x, mask, kv_cache)
        x = r + self.dropout(attn_o)
        r = x
        x = self.norm2(x)
        x = r + self.dropout(self.ffn(x))
        return (x, nkv) if kv_cache is not None else x


# ═══════════════════════════════════════════════════
#  3. GrowingLLM — 自生长语言模型
# ═══════════════════════════════════════════════════

class GrowingLLM(nn.Module):
    """
    自生长语言模型。
    
    起始: 2层, 256维, 4头 → ~3M 参数
    生长: 深度(2→N层) + 宽度(256→512维) + 专家(4→N)
    终点: ~40M 参数 (单卡上限)
    
    架构:
      TokenEmbed → [TransformerBlock × N] → LM Head
      RoPE 编码, RMSNorm, MoEFFN
    """
    
    def __init__(self, vocab_size=8000, d_model=256, nhead=4, init_layers=2,
                 use_moe=False, n_experts=4):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_moe = use_moe
        self.n_experts = n_experts
        self.max_layers = 256
        self._width_levels = [192, 256, 384, 512, 768, 1024]
        self.width_idx = self._width_levels.index(d_model) if d_model in self._width_levels else 1
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer('rope_cache', precompute_rope(d_model))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        for _ in range(init_layers):
            self.layers.append(TransformerBlock(d_model, nhead, use_moe, n_experts))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        
        # 生长追踪
        self.growth_log = []
        self._gap_window = deque(maxlen=20)
        self._loss_history = []
    
    @property
    def n_layers(self):
        return len(self.layers)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    # ─── 前向 ───
    
    def forward(self, token_ids, mask=None):
        B, L = token_ids.shape
        x = self.token_embed(token_ids)
        x = apply_rope(x, self.rope_cache)
        x = self.dropout(x)
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=token_ids.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, token_ids, max_new=128, temp=0.6):
        """自回归生成 (带 KV Cache + EOS/PAD 抑制)。"""
        self.eval()
        device = token_ids.device
        
        # Prefill
        x = self.token_embed(token_ids)
        x = apply_rope(x, self.rope_cache)
        x = self.dropout(x)
        L = token_ids.shape[1]
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        
        kvs = []
        for layer in self.layers:
            out = layer(x, causal_mask)
            kvs.append(out[1] if isinstance(out, tuple) else None)
            x = out[0] if isinstance(out, tuple) else out
        
        logits = self.lm_head(x[:, -1:, :])
        
        for step in range(max_new):
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            # 前 8 步抑制 EOS(3) 和 PAD(0)
            if step < 8:
                for suppress_id in [0, 3]:
                    probs[:, suppress_id] = 0
                probs = probs / probs.sum()
            nid = torch.multinomial(probs, 1)
            token_ids = torch.cat([token_ids, nid], dim=-1)
            if nid.item() == 3 and step > 4:
                break
            
            # 增量解码
            x = self.token_embed(nid)
            x = apply_rope(x, self.rope_cache)
            x = self.dropout(x)
            new_kvs = []
            for i, layer in enumerate(self.layers):
                out = layer(x, kv_cache=kvs[i] if i < len(kvs) else None)
                if isinstance(out, tuple):
                    x, kv = out
                    new_kvs.append(kv)
                else:
                    x = out
                    new_kvs.append(None)
            kvs = new_kvs
            logits = self.lm_head(x)
        
        return token_ids[0].tolist()
    
    def reply(self, text, tokenizer, max_new=96, temp=0.6):
        """对话接口。"""
        ids = tokenizer.encode(text)
        t = torch.tensor([ids], device=next(self.parameters()).device)
        out = self.generate(t, max_new, temp)
        return tokenizer.decode(out[len(ids):])
    
    # ─── 生长 ───
    
    def knowledge_gap(self, input_ids):
        """知识盲区检测。"""
        self.eval()
        with torch.no_grad():
            logits = self(input_ids[:, -128:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            gap = 1.0 - probs.max().item()
        self._gap_window.append(gap)
        return gap
    
    def should_grow(self, avg_loss=None):
        """判定是否需要生长。"""
        if len(self._gap_window) < 10:
            return False
        recent = np.mean(list(self._gap_window)[-5:])
        old = np.mean(list(self._gap_window)[:-5]) if len(self._gap_window) >= 10 else recent
        gap_up = recent > old * 1.2 and recent > 0.5
        loss_stuck = avg_loss is not None and avg_loss > 0.3
        return gap_up or loss_stuck
    
    def grow_depth(self):
        """加一层 (+~1.4M 参数)。"""
        ref = self.layers[-1] if self.layers else None
        new_layer = TransformerBlock(
            self.d_model, use_moe=self.use_moe, n_experts=self.n_experts
        ).to(next(self.parameters()).device)
        if ref:
            new_layer.load_state_dict(ref.state_dict())
        self.layers.append(new_layer)
        added = sum(p.numel() for p in new_layer.parameters())
        self.growth_log.append({'type': 'depth', 'to': self.n_layers, 'added': added})
        return added
    
    def grow_width(self):
        """加宽度 (256→384→512→768→1024)。"""
        if self.width_idx >= len(self._width_levels) - 1:
            return 0
        old_dim = self.d_model
        self.width_idx += 1
        new_dim = self._width_levels[self.width_idx]
        
        old_params = self.count_params()
        
        # Embed
        ow = self.token_embed.weight.data
        nw = torch.zeros(self.vocab_size, new_dim)
        nw[:, :old_dim] = ow
        self.token_embed = nn.Embedding(self.vocab_size, new_dim)
        self.token_embed.weight.data = nw
        
        # RoPE
        self.rope_cache = precompute_rope(new_dim, device=self.rope_cache.device)
        
        # Layers
        for layer in self.layers:
            layer.attn.expand(old_dim, new_dim)
            layer.norm1 = RMSNorm(new_dim).to(layer.norm1.weight.device)
            layer.norm2 = RMSNorm(new_dim).to(layer.norm2.weight.device)
            if isinstance(layer.ffn, MoEFFN):
                for e in layer.ffn.experts:
                    e.expand(old_dim, new_dim)
                layer.ffn.router = nn.Linear(new_dim, layer.ffn.n_experts)
            else:
                layer.ffn.expand(old_dim, new_dim)
        
        # LM Head
        ov = self.lm_head.weight.data
        nv = torch.zeros(self.vocab_size, new_dim)
        nv[:, :old_dim] = ov
        self.lm_head = nn.Linear(new_dim, self.vocab_size, bias=False)
        self.lm_head.weight.data = nv
        self.lm_head.weight = self.token_embed.weight
        
        self.d_model = new_dim
        added = self.count_params() - old_params
        self.growth_log.append({'type': 'width', 'to': new_dim, 'added': added})
        return added
    
    def grow_expert(self):
        """加一个 MoE 专家。"""
        if not self.use_moe:
            return 0
        added = 0
        for layer in self.layers:
            if isinstance(layer.ffn, MoEFFN):
                added += layer.ffn.add_expert()
        self.n_experts = max(layer.ffn.n_experts for layer in self.layers if isinstance(layer.ffn, MoEFFN))
        self.growth_log.append({'type': 'expert', 'to': self.n_experts, 'added': added})
        return added


# ═══════════════════════════════════════════════════
#  4. GrowthEngine — 统一生长调度
# ═══════════════════════════════════════════════════

class GrowthEngine:
    """
    统一生长调度器。
    监视模型的知识盲区, 决定"这次该长哪里"。
    """
    
    def __init__(self, model):
        self.model = model
        self.cooldown = 0
        self._cooldown_depth = 0
        self._cooldown_width = 0
    
    def step(self, avg_loss=None, force_grow=False):
        """每步调用。返回: [(组件, 动作, 新增参数量), ...]。"""
        results = []
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return results
        
        if not self.model.should_grow(avg_loss) and not force_grow:
            return results
        
        # 决策: 长什么
        if self.model.n_layers < 16 and self._cooldown_depth <= 0:
            added = self.model.grow_depth()
            results.append(('depth', self.model.n_layers, added))
            self._cooldown_depth = 8
            self.cooldown = 3
        elif self.model.width_idx < len(self.model._width_levels) - 1 and self._cooldown_width <= 0:
            added = self.model.grow_width()
            if added > 0:
                results.append(('width', self.model.d_model, added))
                self._cooldown_width = 15
                self.cooldown = 5
        elif self.model.use_moe:
            added = self.model.grow_expert()
            results.append(('expert', self.model.n_experts, added))
            self.cooldown = 3
        
        if self._cooldown_depth > 0: self._cooldown_depth -= 1
        if self._cooldown_width > 0: self._cooldown_width -= 1
        
        return results
    
    def report(self):
        print(f'\n=== GrowthEngine ===')
        print(f'  Layers: {self.model.n_layers}')
        print(f'  Dim: {self.model.d_model}')
        print(f'  Experts: {self.model.n_experts}')
        print(f'  Params: {self.model.count_params():,}')
        print(f'  Growth events: {len(self.model.growth_log)}')
        return {
            'layers': self.model.n_layers,
            'dim': self.model.d_model,
            'experts': self.model.n_experts,
            'params': self.model.count_params(),
        }


# ═══════════════════════════════════════════════════
#  5. DomainEvolver — 多领域学习
# ═══════════════════════════════════════════════════

class DomainEvolver:
    """
    多领域演进器。
    顺序学习不同领域, 自动生长, 经验回放防遗忘。
    """
    
    def __init__(self, model, engine, tokenizer, memory_size=10000):
        self.model = model
        self.engine = engine
        self.tokenizer = tokenizer
        self.replay = deque(maxlen=memory_size)
        self.history = []
        self.opt = None  # 在 learn 中创建
    
    def learn(self, domain_name, texts, epochs=30, batch_size=16, lr=3e-4):
        device = next(self.model.parameters()).device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        print(f'\n{"="*50}')
        print(f'  Learn: {domain_name}')
        print(f'  {self.model.n_layers}layers | {self.model.count_params():,}params')
        print(f'  Data: {len(texts)} samples')
        
        for ep in range(epochs):
            self.model.train()
            total_loss = 0
            n = 0
            for text in texts[:200]:  # subsample for speed
                ids = self.tokenizer.encode(text, add_eos=True)
                t = torch.tensor([ids], device=device)
                if t.size(1) < 4: continue
                logits = self.model(t[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), t[:, 1:].reshape(-1))
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss += loss.item()
                n += 1
            
            # Replay
            if len(self.replay) > 0:
                for text in random.sample(list(self.replay), min(batch_size, len(self.replay))):
                    ids = self.tokenizer.encode(text, add_eos=True)
                    t = torch.tensor([ids], device=device)
                    if t.size(1) < 4: continue
                    logits = self.model(t[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, self.model.vocab_size), t[:, 1:].reshape(-1))
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    total_loss += loss.item()
                    n += 1
            
            avg = total_loss / max(1, n)
            
            # 生长
            decisions = self.engine.step(avg_loss=avg)
            if decisions and (ep + 1) % 5 == 0:
                for comp, to, added in decisions:
                    print(f'    Grow: {comp} → {to} (+{added:,})')
            
            if (ep + 1) % 10 == 0:
                print(f'  Ep {ep+1:3d}/{epochs} | loss={avg:.4f} | {self.model.n_layers}layers')
        
        for t in texts:
            self.replay.append(t)
        self.history.append({'domain': domain_name, 'loss': avg, 'layers': self.model.n_layers})
        print(f'  Done: {domain_name}\n')
        return avg


# ═══════════════════════════════════════════════════
#  6. CrossDomainTransfer — 范式迁移
# ═══════════════════════════════════════════════════

class CrossDomainTransfer:
    """
    跨领域范式迁移。
    一个领域学到的注意力模式可迁移到另一个领域。
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pattern_bank = {}
        self.log = []
    
    def extract(self, domain_name, samples=None):
        """提取范式指纹。"""
        device = next(self.model.parameters()).device
        patterns = {'domain': domain_name, 'n_layers': self.model.n_layers, 'd_model': self.model.d_model}
        
        self.model.eval()
        with torch.no_grad():
            # 注意力头重要性
            attn_imp = []
            for layer in self.model.layers:
                dummy = torch.randn(1, 8, self.model.d_model, device=device)
                try:
                    _, w = layer.attn.in_proj(dummy).view(1, 8, 3, layer.attn.nhead, layer.attn.head_dim).transpose(2, 3).mean(0)
                    attn_imp.append(w.mean(dim=[0, 1, 2]).cpu().numpy().tolist())
                except:
                    attn_imp.append([1.0 / layer.attn.nhead] * layer.attn.nhead)
            patterns['attention'] = attn_imp
        
        self.pattern_bank[domain_name] = patterns
        return patterns
    
    def find_best_match(self, samples=None):
        """找最相似领域。"""
        if not self.pattern_bank:
            return None, 0.0
        device = next(self.model.parameters()).device
        results = {}
        for dom, pat in self.pattern_bank.items():
            if not pat.get('attention'):
                continue
            score = 0
            for imp in pat['attention']:
                if imp and len(imp) > 1:
                    score += abs(imp[0] - (1.0 / len(imp) if imp else 0.5))
            results[dom] = 1.0 - min(1.0, score / len(pat['attention']))
        if not results:
            return None, 0.0
        best = max(results, key=results.get)
        return best, results[best]
    
    def transfer_info(self, target_domain, samples=None):
        """查询可迁移的领域。"""
        src, sim = self.find_best_match(samples)
        info = f'transfer_from_{src}(sim={sim:.2f})' if src and sim > 0.3 else 'no_transfer'
        self.log.append({'target': target_domain, 'source': src, 'action': info})
        return info
    
    def report(self):
        print(f'\n=== CrossDomainTransfer ===')
        print(f'  Patterns: {list(self.pattern_bank.keys())}')
        print(f'  Transfers: {len(self.log)}')
        for t in self.log[-3:]:
            print(f'    {t["target"]} <- {t["source"]}')
