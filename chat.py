"""GrowingLLM 交互式对话。"""
import sys, torch, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from growing_llm import GrowingLLM, Tokenizer

# 加载
print('Loading...')
cnt = Counter()
with open('dataset/processed_basic.jsonl', encoding='utf-8') as f:
    for line in f: cnt.update(json.loads(line)['text'])
tok = Tokenizer()
tok.fit([''.join(k for k in cnt)])

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = GrowingLLM(tok.vocab_size, d_model=256, init_layers=2, use_moe=True).to(dev)
sd = torch.load('models/growth/llm_r1.pth', map_location=dev)
m.load_state_dict(sd, strict=False); m.eval()
print(f'  {n} layers, {m.count_params():,} params, {dev}\n')

def gen(model, prompt, max_new=96, temp=0.6):
    return model.reply(prompt, tok, max_new, temp)

print('输入 exit 退出\n')

while True:
    q = input('你: ').strip()
    if not q: continue
    if q.lower() in ['exit','quit','退出']: break
    
    t = encode_prompt('问:'+q+'答:')
    out = gen(m, t, max_new=96, temp=0.7)
    raw = ''.join(tok.id_to_token.get(i,'?') for i in out)
    ans = raw.split('答:',1)[1] if '答:' in raw else raw
    ans = ans.replace('<UNK>','').replace('<PAD>','').strip()
    print(f'AI: {ans}\n')
