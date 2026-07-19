"""GrowingLLM 训练脚本 — R1数据集基础问答训练。"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from growing_llm import GrowingLLMv2, BPETokenizer, GrowthEngine

# ─── 配置 ───
CONFIG = {
    'data_path': 'dataset/processed_basic.jsonl',
    'vocab_size': 12000,
    'd_model': 256,
    'nhead': 4,
    'init_layers': 2,
    'use_moe': True,
    'n_moe_experts': 4,
    'batch_size': 128,
    'epochs': 200,
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'max_seq_len': 128,
    'save_path': 'models/growth/llm_r1.pth',
    'save_every': 5,  # epoch
    'log_every': 1,
    'growth_patience': 8,
    'fp16': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}


class TextDataset(Dataset):
    """从JSONLines加载文本数据集。"""
    def __init__(self, path, tokenizer, max_len=192):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                self.data.append(d['text'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.data[idx], max_len=self.max_len)
        # 固定长度padding,避免collate_fn动态padding的CPU开销
        padded = torch.full((self.max_len + 2,), 0, dtype=torch.long)  # +2 for BOS/EOS
        length = min(len(ids), self.max_len + 2)
        padded[:length] = torch.tensor(ids[:length], dtype=torch.long)
        return padded


def build_tokenizer(data_path, vocab_size=12000):
    from collections import Counter
    cnt = Counter()
    n = 0
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            d = __import__("json").loads(line)
            cnt.update(d["text"])
            n += 1

    chars = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"] + [c for c in cnt]
    tokenizer = BPETokenizer(vocab_size=len(chars))
    tokenizer.token_to_id = {c: i for i, c in enumerate(chars)}
    tokenizer.id_to_token = {i: c for i, c in enumerate(chars)}
    tokenizer.stoi = tokenizer.token_to_id
    tokenizer.itos = tokenizer.id_to_token
    def _save(p):
        import json
        d2 = {"token_to_id": tokenizer.token_to_id, "id_to_token": {str(k): v for k, v in tokenizer.id_to_token.items()}}
        with open(p, "w", encoding="utf-8") as f2:
            json.dump(d2, f2, ensure_ascii=False)
    tokenizer.save = _save
    print(f"  Vocab: {tokenizer.vocab_size} tokens (from {n} texts)")
    return tokenizer



def collate_fn(batch):
    """动态padding到batch内最大长度。"""
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


def train():
    cfg = CONFIG
    device = torch.device(cfg['device'])
    os.makedirs('models/growth', exist_ok=True)
    
    print(f'GrowingLLM Training')
    print(f'  Device: {device}')
    print(f'  Data: {cfg["data_path"]}')
    print(f'  Starting: {cfg["init_layers"]} layers, {cfg["d_model"]} dim')
    print()
    
    # 1. Tokenizer
    print('[1/4] Building tokenizer...')
    tokenizer = build_tokenizer(cfg['data_path'], cfg['vocab_size'])
    tokenizer.save('models/growth/tokenizer_r1.json')
    
    # 2. Dataset
    print('\n[2/4] Loading dataset...')
    ds = TextDataset(cfg['data_path'], tokenizer, cfg['max_seq_len'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True)
    print(f'  Samples: {len(ds):,}')
    print(f'  Batches: {len(dl):,}')
    
    # 3. Model
    print('\n[3/4] Initializing model...')
    model = GrowingLLMv2(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg['d_model'],
        nhead=cfg['nhead'],
        init_layers=cfg['init_layers'],
        use_moe=cfg['use_moe'],
        n_moe_experts=cfg['n_moe_experts'],
    ).to(device)
    
    engine = GrowthEngine(llm=model)
    
    # 尝试加载已有权重 (支持不同层数)
    if os.path.exists(cfg['save_path']):
        try:
            sd = torch.load(cfg['save_path'], map_location=device)
            # 从权重的key数量推算实际层数
            layer_keys = [k for k in sd if k.startswith('layers.')]
            loaded_layers = max(int(k.split('.')[1]) for k in layer_keys) + 1 if layer_keys else cfg['init_layers']
            if loaded_layers > cfg['init_layers']:
                print(f'  Detected {loaded_layers} layers in checkpoint, adjusting model...')
                # 生长到对应层数
                while model.n_layers < loaded_layers:
                    model.grow_depth()
            model.load_state_dict(sd)
            print(f'  Resumed from: {cfg["save_path"]} ({loaded_layers} layers)')
        except Exception as e:
            print(f'  Fresh start (resume failed: {e})')
    
    print(f'  Params: {model.count_params():,}')
    print(f'  Layers: {model.n_layers}')
    
    # 4. Training
    print('\n[4/4] Training...')
    print(f'  Epochs: {cfg["epochs"]}')
    print(f'  Batch: {cfg["batch_size"]}')
    print(f'  Max seq: {cfg["max_seq_len"]}')
    print()
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['fp16'])
    last_grow = -20
    best_loss = float('inf')
    t0 = time.time()
    growth_events = []
    
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        t_epoch = time.time()
        
        for batch_idx, batch in enumerate(dl):
            batch = batch.to(device)
            if batch.size(1) < 4: continue
            
            with torch.cuda.amp.autocast(enabled=cfg['fp16']):
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                batch[:, 1:].reshape(-1)
            )
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
            
            # 实时进度 (每10个batch)
            if batch_idx % 10 == 0:
                elapsed = time.time() - t_epoch
                pct = batch_idx / len(dl) * 100
                avg = total_loss / max(1, n_batches)
                print(f'  Ep {epoch:3d} [{pct:3.0f}%] loss={avg:.4f} | {batch_idx}/{len(dl)} | {elapsed:.0f}s', end='\r')
        
        avg_loss = total_loss / max(1, n_batches)
        
        # 生长判定
        can_grow = (epoch - last_grow) >= cfg['growth_patience']
        loss_stuck = avg_loss > 0.3 and can_grow and epoch > 5
        high_loss = avg_loss > 2.0 and can_grow and epoch > 3
        
        if loss_stuck or high_loss:
            model.grow_depth()
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'] * 0.9, weight_decay=cfg['weight_decay'])
            scaler = torch.cuda.amp.GradScaler(enabled=cfg['fp16'])
            last_grow = epoch
            growth_events.append(epoch)
        
        # 日志
        if epoch % cfg['log_every'] == 0 or epoch == 1:
            dt = time.time() - t0
            lr_now = opt.param_groups[0]['lr']
            growth_mark = f' +layer->{model.n_layers}' if growth_events and growth_events[-1] == epoch else ''
            print(f'  Ep {epoch:3d}/{cfg["epochs"]} | loss={avg_loss:.4f} | {model.n_layers}layers | {model.count_params():,}params | lr={lr_now:.2e} | {dt:.0f}s{growth_mark}')
        
        # 保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg['save_path'])
        
        if epoch % cfg['save_every'] == 0:
            torch.save(model.state_dict(), cfg['save_path'].replace('.pth', f'_ep{epoch}.pth'))
    
    # 最终保存
    model.eval()
    torch.save(model.state_dict(), cfg['save_path'])
    
    print(f'\nTraining complete!')
    print(f'  Final: {model.n_layers} layers, {model.count_params():,} params')
    print(f'  Best loss: {best_loss:.4f}')
    print(f'  Growth events: {len(growth_events)} (at epochs {growth_events})')
    print(f'  Saved: {cfg["save_path"]}')
    print(f'  Time: {time.time() - t0:.0f}s')
    
    # 快速测试
    print(f'\nQuick test:')
    for q in ['445端口怎么利用', '什么是神经网络', 'Python是什么']:
        r = model.reply(q, tokenizer, max_new=32, temp=0.5)
        print(f'  Q: {q}')
        print(f'  A: {r or "(empty)"}')
        print()
    
    return model, tokenizer


if __name__ == '__main__':
    train()
