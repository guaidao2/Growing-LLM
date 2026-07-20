"""
GrowingLLM 训练脚本 (BPE + ChatML)。
用法: python train_r1.py
"""
import os, sys, json, time, torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from growing_llm import GrowingLLM, Tokenizer, GrowthEngine

CONFIG = {
    'data_path': 'dataset/processed_basic.jsonl',
    'vocab_size': 12000,  # 字符级
    'd_model': 384,
    'nhead': 6,
    'init_layers': 2,
    'batch_size': 128,
    'epochs': 50,
    'lr': 5e-4,
    'lr_min': 1e-5,
    'weight_decay': 1e-5,
    'ternary': False,  # 三元留给推理, 训练用fp32
    'max_seq_len': 128,
    'save_path': 'models/growth/llm_r1.pth',
    'save_every': 5,
    'growth_patience': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

class TextDataset:
    def __init__(self, path, tok, max_len=128):
        self.tok = tok
        self.max_len = max_len
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line)['text'])
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        ids = self.tok.encode(self.data[idx], max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    return pad_sequence(batch, batch_first=True, padding_value=0)

def train():
    cfg = CONFIG
    device = torch.device(cfg['device'])
    print(f'GrowingLLM Training | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    
    # 1. Build/Load BPE tokenizer
    print('\n[1/3] BPE tokenizer...')
    tok_path = 'models/growth/tokenizer.json'
    if os.path.exists(tok_path):
        tok = Tokenizer.load(tok_path)
        print(f'  Loaded: {tok.vocab_size} tokens')
    else:
        texts = []
        with open(cfg['data_path'], 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(json.loads(line)['text'])
        tok = Tokenizer()
        tok.fit(texts, target_size=cfg['vocab_size'])
        tok.save(tok_path)
        print(f'  Built: {tok.vocab_size} tokens')
    
    # 2. Data + Model
    print('\n[2/3] Loading data + model...')
    ds = TextDataset(cfg['data_path'], tok, max_len=cfg['max_seq_len'])
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, 
                                      collate_fn=collate_fn, num_workers=0, pin_memory=True)
    print(f'  Samples: {len(ds):,}, Batches: {len(dl):,}')
    
    model = GrowingLLM(
        vocab_size=tok.vocab_size, d_model=cfg['d_model'],
        nhead=cfg['nhead'], init_layers=cfg['init_layers'],
        n_experts=4,
    ).to(device)
    model.lm_head.weight = model.token_embed.weight
    # 三元量化: 续训时开启
    if cfg.get('ternary'):
        for mod in model.modules():
            if hasattr(mod, 'use_ternary'):
                mod.use_ternary = True
        print(f'  Ternary: ON')
    print(f'  Model: {model.n_layers}layers, {model.count_params():,}params')
    
    # Load checkpoint
    if os.path.exists(cfg['save_path']):
        try:
            sd = torch.load(cfg['save_path'], map_location=device)
            old_vb = sd['token_embed.weight'].shape[0]
            if old_vb < tok.vocab_size:
                print(f'  Vocab: {old_vb} -> {tok.vocab_size}')
                for k in ['token_embed.weight', 'lm_head.weight']:
                    w = sd[k]; nw = torch.zeros(tok.vocab_size, w.shape[1])
                    nw[:old_vb] = w.cpu(); sd[k] = nw
            n_layers = max(int(k.split('.')[1]) for k in sd if k.startswith('layers.')) + 1
            while model.n_layers < n_layers:
                model.grow_depth()
            model.load_state_dict(sd, strict=False)
            print(f'  Resumed: {n_layers}layers')
        except Exception as e:
            print(f'  Fresh start ({e})')
    else:
        print(f'  Fresh start (no checkpoint)')
    
    engine = GrowthEngine(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'], eta_min=cfg['lr_min'])
    best_loss = float('inf')
    t0 = time.time()
    
    # 3. Training
    print('\n[3/3] Training...')
    for ep in range(1, cfg['epochs'] + 1):
        model.train()
        total, n = 0, 0
        
        for bi, batch in enumerate(dl):
            batch = batch.to(device)
            if batch.size(1) < 4:
                continue
            
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), batch[:, 1:].reshape(-1))
            
            if torch.isnan(loss):
                print(f'  NaN at batch {bi}, skip'); opt.zero_grad(); continue
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
            
            if bi % 80 == 0:
                avg = total / max(1, n)
                pct = bi / len(dl) * 100
                lr = opt.param_groups[0]['lr']
                mem = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                tok_s = bi * cfg['batch_size'] * cfg['max_seq_len'] / max(1, time.time() - t0)
                print(f'  Ep {ep:3d} [{pct:3.0f}%] loss={avg:.4f} lr={lr:.2e} | {mem:.0f}MB | {tok_s:.0f}tok/s', end='\r')
        
        avg = total / max(1, n)
        dt = time.time() - t0
        lr = opt.param_groups[0]['lr']
        mins = f'{dt//60:.0f}m{dt%60:.0f}s' if dt > 120 else f'{dt:.0f}s'
        best = ' save' if avg < best_loss else ''
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), cfg['save_path'])
        
        params_str = f'{model.count_params()/1e6:.1f}M'
        print(f'  Ep {ep:3d}/{cfg["epochs"]} | loss={avg:.4f} | lr={lr:.2e} | {model.n_layers}layers/{params_str} | {mins}{best}')
        
        # Growth + LR
        results = engine.step(avg_loss=avg)
        for typ, val, added in results:
            print(f'    +grow {typ} -> {val} (+{added:,}params)')
        scheduler.step()
        
        if ep % cfg['save_every'] == 0:
            torch.save(model.state_dict(), cfg['save_path'])
            print(f'  Saved: {cfg["save_path"]}')

if __name__ == '__main__':
    train()
