"""
GrowthEngine — 统一生长控制器。
管理GrowingLLM、MoE-GameNN、GNN三者的动态参数增长。
每个组件实现GrowthInterface, GrowthEngine统一调度。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════
#  统一生长接口
# ═══════════════════════════════════════════════════

class GrowthInterface:
    """所有可生长模型必须实现的接口。"""
    
    def grow_depth(self):
        """增加深度(层/专家)。返回: 新增参数量。"""
        raise NotImplementedError
    
    def grow_width(self, new_dim):
        """扩展宽度(隐藏维度)。返回: 新增参数量。"""
        raise NotImplementedError
    
    def knowledge_gap(self, input_tensor):
        """评估对当前输入的"不确定度"。返回: float 0-1。"""
        raise NotImplementedError
    
    def count_params(self):
        """当前总参数量。"""
        raise NotImplementedError


# ═══════════════════════════════════════════════════
#  BPE 分词器
# ═══════════════════════════════════════════════════

class BPETokenizer:
    """
    Byte-Pair Encoding 分词器,支持中英文混合。
    中文单字保留,英文按BPE合并,数字和符号保留。
    """
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3, '<MASK>': 4}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>', 4: '<MASK>'}
        self.merges = {}  # (token_a, token_b) → merged_token
    
    def _pre_tokenize(self, text):
        """初始分词: 中文单字,英文按字母拆分,数字和符号保留。"""
        import re
        tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文
                tokens.append(char)
            elif char.isascii() and char.isalpha():  # 英文
                tokens.append(char.lower())
            elif char.isdigit():  # 数字
                tokens.append(char)
            else:  # 符号
                tokens.append(char)
        return tokens
    
    def fit(self, texts):
        """从文本中学习BPE合并规则。"""
        from collections import Counter
        
        # 统计初始token频率
        token_freqs = Counter()
        for text in texts:
            tokens = self._pre_tokenize(text)
            for t in tokens:
                token_freqs[t] += 1
        
        # 初始词表
        next_id = len(self.token_to_id)
        for token in token_freqs:
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1
        
        # BPE合并
        for step in range(self.vocab_size - len(self.token_to_id)):
            # 统计相邻pair频率
            pair_freqs = Counter()
            for text in texts:
                tokens = self._pre_tokenize(text)
                ids = [self.token_to_id.get(t, 1) for t in tokens]
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    pair_freqs[pair] += 1
            
            if not pair_freqs:
                break
            
            # 找最频繁的pair
            best_pair = pair_freqs.most_common(1)[0][0]
            
            # 创建新token
            merged = self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            if merged in self.token_to_id:
                continue
            
            new_id = len(self.token_to_id)
            self.token_to_id[merged] = new_id
            self.id_to_token[new_id] = merged
            self.merges[best_pair] = new_id
    
    def encode(self, text, max_len=128):
        """编码: 文本 → token IDs。"""
        tokens = self._pre_tokenize(text)
        ids = [self.token_to_id.get(t, 1) for t in tokens]
        
        # 应用BPE合并
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(ids) - 1:
                pair = (ids[i], ids[i + 1])
                if pair in self.merges:
                    ids[i] = self.merges[pair]
                    del ids[i + 1]
                    changed = True
                i += 1
        
        ids = [2] + ids[:max_len] + [3]  # <BOS>...<EOS>
        return ids
    
    def decode(self, ids):
        """解码: token IDs → 文本。"""
        chars = []
        for i in ids:
            if i in [0, 2, 3, 4]:
                continue
            chars.append(self.id_to_token.get(i, '?'))
        return ''.join(chars)


# ═══════════════════════════════════════════════════
#  可生长的 Transformer 块
# ═══════════════════════════════════════════════════

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        h = hidden_dim or dim * 3
        self.w1 = nn.Linear(dim, h)
        self.w2 = nn.Linear(dim, h)
        self.w3 = nn.Linear(h, dim)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    
    def expand_width(self, old_dim, new_dim):
        """扩展宽度: old_dim → new_dim。"""
        for name, p in [('w1', self.w1), ('w2', self.w2), ('w3', self.w3)]:
            w, b = p.weight.data, p.bias.data
            new_w = torch.zeros(new_dim if 'w3' not in name else w.shape[0], 
                              new_dim if 'w3' in name else w.shape[1])
            new_b = torch.zeros(new_dim if 'w3' not in name else b.shape[0])
            # 复制旧权重
            min_d0 = min(w.shape[0], new_w.shape[0])
            min_d1 = min(w.shape[1], new_w.shape[1])
            new_w[:min_d0, :min_d1] = w[:min_d0, :min_d1]
            new_b[:min_d0] = b[:min_d0]
            p.data = new_w
            p.bias = nn.Parameter(new_b)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=192, nhead=4, d_ff=None):
        super().__init__()
        d_ff = d_ff or d_model * 3
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None, kv_cache=None):
        if kv_cache is not None:
            # 增量解码
            cached_k, cached_v = kv_cache
            q = self.norm1(x)
            k = v = q
            if cached_k is not None:
                k = torch.cat([cached_k, k], dim=1)
                v = torch.cat([cached_v, v], dim=1)
            attn_out, _ = self.self_attn(q, k, v, attn_mask=None if mask is None else mask[-1:, :k.shape[1]])
            x = x + self.dropout(attn_out)
            return self.norm2(x + self.dropout(self.ffn(x))), (k, v)
        else:
            x2 = self.self_attn(x, x, x, attn_mask=mask)[0]
            x = self.norm1(x + self.dropout(x2))
            x = self.norm2(x + self.dropout(self.ffn(x)))
            return x
    
    def clone_weights(self, src, noise=0.02):
        self.load_state_dict(src.state_dict())
        for p in self.parameters():
            p.data += torch.randn_like(p) * noise
    
    def expand_width(self, old_dim, new_dim):
        """宽度增长: 所有子模块扩展维度。"""
        self.d_model = new_dim
        # 注意力
        w = self.self_attn.in_proj_weight.data
        new_w = torch.zeros(new_dim * 3, new_dim)
        new_w[:w.shape[0], :w.shape[1]] = w
        self.self_attn.in_proj_weight.data = new_w
        b = self.self_attn.in_proj_bias.data
        new_b = torch.zeros(new_dim * 3)
        new_b[:b.shape[0]] = b
        self.self_attn.in_proj_bias.data = new_b
        self.self_attn.out_proj = nn.Linear(new_dim, new_dim)
        # LayerNorm
        self.norm1 = nn.LayerNorm(new_dim)
        self.norm2 = nn.LayerNorm(new_dim)
        # FFN
        self.ffn.expand_width(old_dim, new_dim)


# ═══════════════════════════════════════════════════
#  GrowingLLM v2 (深度+宽度生长, BPE, KV Cache)
# ═══════════════════════════════════════════════════

class GrowingLLMv2(nn.Module, GrowthInterface):
    """
    v2 改进:
      - BPE分词器
      - 深度+宽度双维度生长
      - KV Cache增量解码
      - WorldModel知识盲区检测
    """
    def __init__(self, vocab_size=8000, d_model=192, nhead=4, init_layers=2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList()
        for _ in range(init_layers):
            self.layers.append(TransformerBlock(d_model, nhead))
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
        
        # 生长状态
        self.growth_log = []
        self.knowledge_gaps = deque(maxlen=20)
        self.width_level = 0  # 0:192, 1:256, 2:384, 3:512
        self.widths = [192, 256, 384, 512]
    
    @property
    def n_layers(self):
        return len(self.layers)
    
    # ─── GrowthInterface ───
    
    def grow_depth(self):
        """加一层 (深度增长, +~980K参数)。"""
        new_layer = TransformerBlock(self.d_model).to(next(self.parameters()).device)
        if self.layers:
            new_layer.clone_weights(self.layers[-1])
        self.layers.append(new_layer)
        added = sum(p.numel() for p in new_layer.parameters())
        self.growth_log.append({'type': 'depth', 'to': self.n_layers, 'added': added})
        return added
    
    def grow_width(self, new_dim=None):
        """扩展宽度 (维度增长, +~1.5M参数)。"""
        if new_dim is None and self.width_level < len(self.widths) - 1:
            self.width_level += 1
            new_dim = self.widths[self.width_level]
        if new_dim is None or new_dim <= self.d_model:
            return 0
        
        old_dim = self.d_model
        print(f'    [WidthGrow] {old_dim} -> {new_dim}')
        
        # 嵌入层
        old_w = self.token_embed.weight.data
        new_w = torch.zeros(self.vocab_size, new_dim)
        new_w[:, :old_dim] = old_w
        self.token_embed = nn.Embedding(self.vocab_size, new_dim)
        self.token_embed.weight.data = new_w
        
        # 位置编码
        old_pos = self.pos_embed.data
        new_pos = torch.randn(1, 256, new_dim) * 0.02
        new_pos[:, :, :old_dim] = old_pos
        self.pos_embed = nn.Parameter(new_pos)
        
        # 每层Transformer块
        for layer in self.layers:
            layer.expand_width(old_dim, new_dim)
        
        # LM头
        old_w = self.lm_head.weight.data
        new_w = torch.zeros(self.vocab_size, new_dim)
        new_w[:, :old_dim] = old_w
        self.lm_head = nn.Linear(new_dim, self.vocab_size, bias=False)
        self.lm_head.weight.data = new_w
        self.lm_head.weight = self.token_embed.weight  # 重新绑定
        
        old_total = self.count_params()
        self.d_model = new_dim
        added = self.count_params() - old_total
        self.growth_log.append({'type': 'width', 'to': new_dim, 'added': added})
        return added
    
    def knowledge_gap(self, input_tensor):
        """
        知识盲区检测。
        用模型自身的预测分布评估"不确定度":
          gap = 1 - max(softmax(logits))
        
        值越高 → 模型越不确定 → 越需要生长
        """
        self.eval()
        with torch.no_grad():
            logits = self(input_tensor[:, -64:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            confidence = probs.max().item()
            gap = 1.0 - confidence
        self.knowledge_gaps.append(gap)
        return gap
    
    def should_grow(self, avg_loss=None):
        """综合判定是否需要生长。(知识盲区 + loss停滞)"""
        if len(self.knowledge_gaps) < 10:
            return False
        
        recent_gap = np.mean(list(self.knowledge_gaps)[-5:])
        old_gap = np.mean(list(self.knowledge_gaps)[:-5]) if len(self.knowledge_gaps) > 10 else recent_gap
        
        # 知识盲区上升 → 需要生长
        gap_increasing = recent_gap > old_gap * 1.2
        gap_high = recent_gap > 0.6
        
        # loss停滞
        loss_stuck = avg_loss is not None and avg_loss > 0.3
        
        need_depth = (gap_increasing and gap_high) or loss_stuck
        need_width = gap_high and self.width_level < len(self.widths) - 1
        
        if need_width and self.n_layers >= 4:
            return 'width'
        if need_depth and self.n_layers < 16:
            return 'depth'
        return False
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())
    
    # ─── 前向 + KV Cache ───
    
    def forward(self, token_ids, mask=None):
        B, L = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :L, :]
        x = self.dropout(x)
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=token_ids.device), diagonal=1)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, token_ids, max_new=64, temp=0.6):
        """带KV Cache的自回归生成。"""
        self.eval()
        device = next(self.parameters()).device
        
        # 预填充
        x = self.token_embed(token_ids) + self.pos_embed[:, :token_ids.shape[1], :]
        x = self.dropout(x)
        L = token_ids.shape[1]
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        
        kv_caches = []
        for layer in self.layers:
            out = layer(x, causal_mask)
            if isinstance(out, tuple):
                x, kv = out
                kv_caches.append(kv)
            else:
                x = out
                kv_caches.append(None)
        
        logits = self.lm_head(x[:, -1:, :])
        
        for step in range(max_new):
            probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
            next_id = torch.multinomial(probs, 1)
            token_ids = torch.cat([token_ids, next_id], dim=-1)
            
            if next_id.item() == 3:
                break
            
            # 增量解码 (KV Cache)
            x = self.token_embed(next_id) + self.pos_embed[:, :1, :]
            x = self.dropout(x)
            new_kv = []
            for i, layer in enumerate(self.layers):
                out = layer(x, kv_cache=kv_caches[i] if i < len(kv_caches) else None)
                if isinstance(out, tuple):
                    x, kv = out
                    new_kv.append(kv)
                else:
                    x = out
                    new_kv.append(None)
            kv_caches = new_kv
            
            logits = self.lm_head(x)
        
        return token_ids[0].tolist()
    
    def reply(self, text, tokenizer, max_new=64, temp=0.6):
        """对外接口: 文本 → 回复。"""
        ids = tokenizer.encode(text)
        t = torch.tensor([ids], device=next(self.parameters()).device)
        out = self.generate(t, max_new=max_new, temp=temp)
        return tokenizer.decode(out[len(ids):])


# ═══════════════════════════════════════════════════
#  GrowthEngine — 统一调度器
# ═══════════════════════════════════════════════════

class GrowthEngine:
    """
    统一生长引擎。
    
    管理:
      - GrowingLLMv2:  Transformer层深度 + 宽度
      - MoE-GameNN:    ExpertGRU数量
      - GNNEncoder:    图编码维度
    
    生长策略:
      step() → 检查所有模型的知识盲区
            → 决定"这次该长哪里"
            → 执行生长
    """
    
    def __init__(self, llm=None, moe=None, gnn=None):
        self.llm = llm      # GrowingLLMv2
        self.moe = moe      # MoE-GameNN
        self.gnn = gnn      # GNNEncoder
        
        self.history = []
        self.cooldown = 0  # 生长冷却步数
        self.growth_budget = {'depth': 0, 'width': 0, 'experts': 0}
    
    def step(self, input_tensor=None, avg_loss=None):
        """每步调用,检查是否需要生长。"""
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        
        decisions = []
        
        # 检查LLM
        if self.llm is not None:
            decision = self.llm.should_grow(avg_loss)
            if decision == 'depth':
                added = self.llm.grow_depth()
                decisions.append(f'LLM+层({added:,})')
                self.growth_budget['depth'] += 1
                self.cooldown = 5
            elif decision == 'width':
                added = self.llm.grow_width()
                if added > 0:
                    decisions.append(f'LLM+宽({added:,})')
                    self.growth_budget['width'] += 1
                    self.cooldown = 10
        
        # 检查MoE
        if self.moe is not None and hasattr(self.moe, 'should_grow'):
            if self.moe.should_grow():
                added = self.moe.grow_expert()
                decisions.append(f'MoE+专家({added:,})')
                self.growth_budget['experts'] += 1
                self.cooldown = 3
        
        if decisions:
            self.history.append({
                'step': len(self.history),
                'actions': decisions,
                'budget': dict(self.growth_budget),
            })
            print("  [GrowthEngine] " + ", ".join(decisions))
        
        return decisions
    
    def report(self):
        """生长报告。"""
        print('\n=== GrowthEngine Report ===')
        if self.llm:
            llm_p = self.llm.count_params()
            print(f'  LLM: {self.llm.n_layers}层, {llm_p:,}参数')
            if hasattr(self.llm, 'token_embed'):
                print(f'  LLM维度: {self.llm.d_model}')
        print(f'  总生长次数: {len(self.history)}')
        print(f'  总预算: {self.growth_budget}')
        return self.growth_budget

# ═══════════════════════════════════════════════════
#  领域进化 + 网络卸载
# ═══════════════════════════════════════════════════

class DomainEvolver:
    """
    多领域进化管理器。
    
    能力:
      - 在不同领域数据上顺序训练
      - 知识盲区检测 → 自动生长
      - 经验回放防遗忘
      - 领域级checkpoint
    
    使用:
      evolver = DomainEvolver(llm, engine)
      evolver.learn('网络安全', security_data)
      evolver.learn('数学推理', math_data)  
      evolver.learn('代码生成', code_data)
    """
    
    def __init__(self, llm, growth_engine=None, memory_size=10000):
        self.llm = llm
        self.engine = growth_engine
        self.replay_buffer = deque(maxlen=memory_size)
        self.domain_history = []
        self.opt = torch.optim.AdamW(llm.parameters(), lr=3e-4, weight_decay=1e-5)
    
    def learn(self, domain_name, texts, epochs=30, batch_size=4):
        """
        学习一个新领域。
        
        流程:
          1. 测量当前模型在这领域上的知识盲区
          2. 需要时自动生长 (深度/宽度)
          3. 训练 + 回放旧领域数据
          4. 保存领域checkpoint
        """
        device = next(self.llm.parameters()).device
        print(f'\n{"="*50}')
        print('  学习新领域: ' + domain_name)
        print('  ' + '='*50)
        print('  当前: ' + str(self.llm.n_layers) + '层, ' + str(self.llm.d_model) + '维, ' + str(self.llm.count_params()) + '参数')
        
        # 1. 测量知识盲区
        sample_text = texts[0] if texts else ''
        sample_ids = torch.tensor([self.tokenizer.encode(sample_text)], device=device) if hasattr(self, 'tokenizer') else None
        if sample_ids is not None:
            gap = self.llm.knowledge_gap(sample_ids)
            print(f'  知识盲区: {gap:.1%} ', end='')
            if gap > 0.5:
                print(f'(高 → 需要增长容量)')
            else:
                print(f'(低 → 现有容量足够)')
        
        # 2. 生长决策
        if self.engine:
            for _ in range(3):  # 最多触发3次生长
                decision = self.engine.step(avg_loss=1.0)
                if not decision:
                    break
        
        # 3. 训练
        for epoch in range(epochs):
            self.llm.train()
            total_loss = 0
            
            # 新领域数据
            for text in texts:
                ids = self.tokenizer.encode(text) if hasattr(self, 'tokenizer') else [0]
                t = torch.tensor([ids], device=device)
                if t.size(1) < 3: continue
                logits = self.llm(t[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, self.llm.vocab_size), t[:, 1:].reshape(-1))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                total_loss += loss.item()
            
            # 经验回放 (防遗忘)
            if len(self.replay_buffer) > 0:
                replay_texts = random.sample(list(self.replay_buffer), min(batch_size, len(self.replay_buffer)))
                for text in replay_texts:
                    ids = self.tokenizer.encode(text)
                    t = torch.tensor([ids], device=device)
                    if t.size(1) < 3: continue
                    logits = self.llm(t[:, :-1])
                    loss = F.cross_entropy(logits.reshape(-1, self.llm.vocab_size), t[:, 1:].reshape(-1))
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(texts))
            
            # 训练中生长
            if self.engine and (epoch+1) % 10 == 0:
                self.engine.step(avg_loss=avg_loss)
            
            if (epoch+1) % 10 == 0:
                print('  Epoch ' + str(epoch+1).rjust(3) + '/' + str(epochs) + ' | loss=' + f'{avg_loss:.4f}' + ' | ' + str(self.llm.n_layers) + '层 | ' + str(self.llm.count_params()) + '参数')
        
        # 4. 保存领域记忆
        for text in texts:
            self.replay_buffer.append(text)
        self.domain_history.append({
            'domain': domain_name,
            'epochs': epochs,
            'final_loss': avg_loss,
            'layers': self.llm.n_layers,
            'params': self.llm.count_params(),
        })
        
        # 保存领域checkpoint (可选加载)
        torch.save(self.llm.state_dict(), f'models/growth/domain_{domain_name}.pth')
        print(f'  领域[{domain_name}] 学习完成! checkpoint已保存')
        return avg_loss


class NetworkShard:
    """
    网络分片 — 当模型太大跑不动时,自动卸载到网络。
    
    工作方式:
      1. 检测本地GPU显存
      2. 如果不足 → 自动分割模型
      3. 部分层留在本地,部分层转发到网络服务
      4. 透明切换,调用方无感
    """
    
    def __init__(self, llm, local_device='cuda', remote_url=None):
        self.llm = llm
        self.local_device = local_device
        self.remote_url = remote_url
        self.shard_point = None  # 第几层之后走网络
        self.enabled = False
        self.memory_threshold = 0.8  # 显存使用率超过80%时启用
    
    def check_memory(self):
        """检查显存状态。"""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated(0)
        return used / total
    
    def auto_configure(self):
        """自动配置分片策略。"""
        usage = self.check_memory()
        model_size = self.llm.count_params() * 4  # fp32 bytes
        
        if usage > self.memory_threshold and self.remote_url:
            self.enabled = True
            # 决定切分点: 留前1/3在本地,后2/3走网络
            self.shard_point = max(1, self.llm.n_layers // 3)
            print(f'\n  [NetworkShard] 显存{usage:.0%}, 启用网络分片')
            print(f'    {self.shard_point}/{self.llm.n_layers}层留在本地, 其余走网络')
            return True
        return False
    
    def forward(self, x, kv_caches=None):
        """透明前向: 自动决定本地/网络。"""
        if not self.enabled or self.shard_point is None:
            # 全本地
            return self._forward_local(x, kv_caches)
        
        # 部分本地,部分网络
        h = x
        local_kv = []
        for i, layer in enumerate(self.llm.layers):
            if i < self.shard_point:
                # 本地
                out = layer(h, kv_cache=kv_caches[i] if kv_caches else None)
                if isinstance(out, tuple):
                    h, kv = out
                    local_kv.append(kv)
                else:
                    h = out
                    local_kv.append(None)
            else:
                # 网络 (模拟发送)
                import json, urllib.request
                data = json.dumps({
                    'tensor': h.cpu().numpy().tolist(),
                    'layer_idx': i,
                    'total_layers': self.llm.n_layers,
                }).encode()
                try:
                    req = urllib.request.Request(self.remote_url, data=data, 
                        headers={'Content-Type': 'application/json'})
                    resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
                    h = torch.tensor(resp['output'], device=self.local_device)
                except:
                    # 网络失败,回退到本地 (CPU)
                    layer = layer.to('cpu')
                    h = layer(h.cpu())
                    layer.to(self.local_device)
                    h = h.to(self.local_device)
                local_kv.append(None)
        
        return self.llm.lm_head(h), local_kv
    
    def _forward_local(self, x, kv_caches=None):
        """全本地前向。"""
        for i, layer in enumerate(self.llm.layers):
            out = layer(x, kv_cache=kv_caches[i] if kv_caches else None)
            if isinstance(out, tuple):
                x, kv = out
            else:
                x = out
        return self.llm.lm_head(x), None


# 补上 import random
import random
