"""用 MiniMind 训练 GrowingLLM 的基础对话能力。"""
import os, sys, json, time, random, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gamenn-hacker'))
import torch, torch.nn.functional as F
from collections import Counter
from growing_llm import GrowingLLMv2, BPETokenizer, GrowthEngine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('=' * 55)

# ─── 1. MiniMind 生成数据 ───
print('[1/3] Generating data from MiniMind...')
from core.security_lm import load as lm_load, reply as lm_reply
teacher, tt = lm_load('dir')

topics = [
    '445端口怎么利用','拿到shell后怎么提权','SQL注入怎么判断',
    '内网渗透怎么打','横向移动有哪些方法','免杀怎么做',
    'WAF怎么绕过','nmap常用参数','弱口令怎么爆破',
    'MS17010怎么用','Linux提权方法','渗透测试标准流程',
    '信息收集做什么','文件上传绕过','Redis未授权利用',
    'Python装饰器是什么','Git怎么合并分支','Docker怎么用',
    'TCP和UDP区别','什么是RESTful API','Linux常用命令',
    '数据库索引原理','什么是微服务架构',
    '什么是人工智能','机器学习分类方法','神经网络怎么工作',
    '区块链是什么','云计算三种模式','什么是大数据',
]

data = []
for i, q in enumerate(topics):
    try:
        ans = lm_reply(f'请用中文回答: {q}', teacher, tt, max_new=96, temp=0.7)
        if ans and len(ans) > 10:
            if '<think>' in ans:
                if '</think>' in ans:
                    ans = re.sub(r'<think>.*?</think>', '', ans, flags=re.DOTALL).strip()
                else:
                    ans = ans[:ans.index('<think>')].strip()
            if ans:
                data.append(f'问:{q}答:{ans}')
    except: pass
    if (i+1)%5==0: print(f'  {i+1}/{len(topics)}')

# Augment
aug = list(data)
for item in data:
    if '答:' in item:
        qp = item[item.index('问:')+2:item.index('答:')]
        ap = item[item.index('答:')+2:]
        aug.append(f'问:如何{ap[:6]}答:{qp}')
data = aug
random.shuffle(data)
print(f'  Total: {len(data)} samples')

del teacher, tt; torch.cuda.empty_cache()

# ─── 2. Tokenizer ───
print('\n[2/3] Building tokenizer...')
cnt = Counter(''.join(data))
chars = ['<PAD>','<UNK>','<BOS>','<EOS>'] + [c for c in cnt]
tokenizer = BPETokenizer(len(chars))
tokenizer.stoi = {c:i for i,c in enumerate(chars)}
tokenizer.itos = {i:c for i,c in enumerate(chars)}
print(f'  Vocab: {tokenizer.vocab_size}')

# ─── 3. Training ───
print('\n[3/3] Training GrowingLLM...')
model = GrowingLLMv2(tokenizer.vocab_size, d_model=256, nhead=4, init_layers=2, use_moe=True, n_moe_experts=4).to(device)
print(f'  Start: {model.n_layers}layers, {model.count_params():,}params')

opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
last_grow = -20
t0 = time.time()

for ep in range(100):
    model.train()
    random.shuffle(data)
    total = 0; n = 0
    for text in data:
        ids = tokenizer.encode(text, max_len=128)
        t = torch.tensor([ids], device=device)
        if t.size(1) < 4: continue
        lo = model(t[:, :-1])
        loss = F.cross_entropy(lo.reshape(-1, tokenizer.vocab_size), t[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        total += loss.item(); n += 1
    avg = total / max(1, n)
    
    if (ep-last_grow)>=8 and avg>0.5 and ep>5:
        model.grow_depth()
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-5)
        last_grow = ep
    
    if (ep+1)%10==0:
        dt = time.time()-t0
        print(f'  Ep {ep+1:3d}/100 | loss={avg:.4f} | {model.n_layers}layers | {model.count_params():,}params | {dt:.0f}s')

model.eval()
torch.save(model.state_dict(), 'models/growth/llm_trained.pth')

# ─── 4. Test ───
print(f'\n{"="*55}')
print('Test:')
for q in ['445端口怎么利用','Python装饰器是什么','什么是人工智能']:
    r = model.reply(q, tokenizer, max_new=48, temp=0.5)
    print(f'Q: {q}')
    print(f'A: {r or "(empty)"}\n')

print(f'Final: {model.n_layers}layers, {model.count_params():,}params')
