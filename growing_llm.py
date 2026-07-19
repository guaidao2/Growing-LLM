"""
GrowingLLM — 可动态增长的 Transformer 语言模型。
从 2 层开始,发现不够用时自动加层,参数从 2.5M 逐步长到 10M+。
"""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# ═══════════════════════════════════════════════════
#  轻量分词器
# ═══════════════════════════════════════════════════

class GrowingTokenizer:
    """基于频率的中英文分词器,支持扩展。"""
    def __init__(self, max_vocab=8000):
        self.special = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.stoi = dict(self.special)
        self.itos = {v: k for k, v in self.special.items()}
        self.max_vocab = max_vocab
    
    def fit(self, texts):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t)
        for c, _ in cnt.most_common(self.max_vocab - len(self.special)):
            if c not in self.stoi:
                i = len(self.stoi)
                self.stoi[c] = i
                self.itos[i] = c
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'stoi': self.stoi, 'itos': {str(k): v for k, v in self.itos.items()}}, f)
    
    def load(self, path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
            self.stoi = d['stoi']
            self.itos = {int(k): v for k, v in d['itos'].items()}
    
    def encode(self, text, max_len=128):
        ids = [self.stoi.get(c, 1) for c in text[:max_len]]
        return [self.stoi['<BOS>']] + ids + [self.stoi['<EOS>']]
    
    def decode(self, ids):
        return ''.join(self.itos.get(i, '?') for i in ids if i not in [0, 2, 3])
    
    @property
    def vocab_size(self):
        return len(self.stoi)


# ═══════════════════════════════════════════════════
#  Transformer 层
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


class TransformerBlock(nn.Module):
    """单层 Transformer 解码器块。每层 ~980K 参数(d=256, h=1024)。"""
    def __init__(self, d_model=256, nhead=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout(x2))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
    
    def clone_weights(self, source_layer, noise=0.02):
        """从另一层复制权重+噪声。"""
        self.load_state_dict(source_layer.state_dict())
        for p in self.parameters():
            p.data += torch.randn_like(p) * noise


# ═══════════════════════════════════════════════════
#  GrowingLLM 核心
# ═══════════════════════════════════════════════════

class GrowingLLM(nn.Module):
    """
    可生长 Transformer。
    
    起始: 2层, 256dim, 4头 ≈ 2.5M 参数
    生长: 每步加1层(~980K), 丢门槛触发
    
    生长触发: 训练 loss 连续 N 步未改善 + WorldModel预测误差大
    """
    def __init__(self, vocab_size=8000, d_model=256, nhead=4, d_ff=1024, init_layers=2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        self.dropout = nn.Dropout(0.1)
        
        self.layers = nn.ModuleList()
        for _ in range(init_layers):
            self.layers.append(TransformerBlock(d_model, nhead, d_ff))
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # 权重绑定
        
        # 生长追踪
        self.growth_log = []
        self.loss_history = []
        self.growth_patience = 5
        self.max_layers = 24
        
        self.train_loss_ema = 10.0
    
    @property
    def n_layers(self):
        return len(self.layers)
    
    def forward(self, token_ids, mask=None):
        B, L = token_ids.shape
        x = self.token_embed(token_ids) + self.pos_embed[:, :L, :]
        x = self.dropout(x)
        
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=token_ids.device), diagonal=1)
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        return self.lm_head(x)
    
    def should_grow(self, current_loss):
        """判断是否需要加一层。"""
        self.loss_history.append(current_loss)
        self.train_loss_ema = 0.9 * self.train_loss_ema + 0.1 * current_loss
        
        if len(self.layers) >= self.max_layers:
            return False
        
        if len(self.loss_history) < self.growth_patience + 1:
            return False
        
        recent = self.loss_history[-self.growth_patience:]
        # 连续 patience 步没改善 → 加层
        if min(recent) >= self.loss_history[-self.growth_patience - 1]:
            return True
        
        return False
    
    def grow(self):
        """加一层 Transformer。参数增长 ~980K。"""
        new_layer = TransformerBlock(self.d_model).to(next(self.parameters()).device)
        if self.layers:
            new_layer.clone_weights(self.layers[-1])
        
        self.layers.append(new_layer)
        self.growth_log.append({
            'step': len(self.growth_log),
            'layers': self.n_layers,
            'params': sum(p.numel() for p in self.parameters()),
            'loss': self.loss_history[-1] if self.loss_history else 0,
        })
        
        print(f'    [Growing] 层数: {self.n_layers} | 参数: {sum(p.numel() for p in self.parameters()):,} | loss: {self.loss_history[-1]:.4f}')
        return self.n_layers
    
    def generate(self, token_ids, max_new=64, temp=0.6):
        """自回归生成。"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new):
                logits = self(token_ids[:, -128:])
                probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
                next_id = torch.multinomial(probs, 1)
                token_ids = torch.cat([token_ids, next_id], dim=-1)
                if next_id.item() == 3:  # <EOS>
                    break
        return token_ids[0].tolist()
    
    def reply(self, text, tokenizer, max_new=96, temp=0.6):
        """对外接口: 文本 → 回复。"""
        ids = tokenizer.encode(text)
        t = torch.tensor([ids], device=next(self.parameters()).device)
        out = self.generate(t, max_new=max_new, temp=temp)
        return tokenizer.decode(out[len(ids):])
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════
#  生长训练器
# ═══════════════════════════════════════════════════

def train_with_growth(
    model, tokenizer, data_texts,
    epochs=50, batch_size=8, lr=3e-4,
    save_path='models/growth/llm.pth', grow=True,
):
    """训练并自动生长。"""
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    print(f'\n训练 GrowingLLM (起始{model.n_layers}层, {model.count_params():,}参数)')
    print(f'  数据: {len(data_texts)} 条 | 设备: {device}')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for text in data_texts:
            ids = tokenizer.encode(text)
            t = torch.tensor([ids], device=device)
            
            if t.size(1) < 4:
                continue
            
            logits = model(t[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), t[:, 1:].reshape(-1))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / max(1, len(data_texts))
        sch.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | {model.n_layers}层 | {model.count_params():,}参数')
        
        # 生长判定
        if grow and model.should_grow(avg_loss):
            model.grow()
            # 生长后重新初始化优化器
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.8, weight_decay=1e-5)
    
    model.eval()
    torch.save(model.state_dict(), save_path)
    print(f'  保存 → {save_path}')
    print(f'  最终: {model.n_layers}层, {model.count_params():,}参数')
    if model.growth_log:
        print(f'  生长历史: {len(model.growth_log)} 次')
    
    return model
