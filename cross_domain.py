"""
CrossDomainTransfer — 跨领域范式迁移。
学到的抽象范式可在领域间复用,实现"触类旁通"。
"""
import torch
import numpy as np


class CrossDomainTransfer:
    """
    跨领域范式迁移。
    
    一个领域学到的抽象范式可以迁移到另一个领域。
    例如: 网安"扫描端口" = 数学"穷举可能解"
          提权"逐步提升" = 编程"渐增复杂度"
    
    用法:
      transfer = CrossDomainTransfer(llm)
      transfer.extract_patterns('security', samples, tokenizer)
      transfer.extract_patterns('math', samples, tokenizer)
      transfer.transfer('code', samples, tokenizer)  # 自动寻找最适合的迁移源
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.pattern_bank = {}
        self.transfer_log = []
    
    def extract_patterns(self, domain_name, samples=None, tokenizer=None):
        """提取范式指纹: 注意力模式 + 神经元重要性 + 置信度分布。"""
        device = next(self.llm.parameters()).device
        patterns = {
            'domain': domain_name,
            'n_layers': self.llm.n_layers,
            'd_model': self.llm.d_model,
            'attention_patterns': [],
            'confidence_profile': [],
        }
        
        self.llm.eval()
        with torch.no_grad():
            for layer in self.llm.layers:
                dummy = torch.randn(1, 8, self.llm.d_model, device=device)
                try:
                    _, attn_w = layer.self_attn(dummy, dummy, dummy, need_weights=True)
                    if attn_w is not None:
                        imp = attn_w.mean(dim=[0, 2, 3]).cpu().numpy()
                        patterns['attention_patterns'].append(imp.tolist())
                except:
                    h = layer.self_attn.num_heads
                    patterns['attention_patterns'].append([1.0 / h] * h)
        
        if samples and tokenizer:
            for t in samples[:10]:
                ids = tokenizer.encode(t)
                ti = torch.tensor([ids], device=device)
                if ti.size(1) > 2:
                    patterns['confidence_profile'].append(1.0 - self.llm.knowledge_gap(ti))
        
        self.pattern_bank[domain_name] = patterns
        return patterns
    
    def find_most_similar(self, new_samples, tokenizer=None):
        """找最相似的已学领域。用置信度分布相关度。"""
        if not self.pattern_bank:
            return None, 0.0
        
        device = next(self.llm.parameters()).device
        results = {}
        
        for dom, pat in self.pattern_bank.items():
            if tokenizer and pat.get('confidence_profile'):
                new_conf = []
                for t in new_samples[:10]:
                    ids = tokenizer.encode(t)
                    ti = torch.tensor([ids], device=device)
                    if ti.size(1) > 2:
                        new_conf.append(1.0 - self.llm.knowledge_gap(ti))
                
                old_conf = pat['confidence_profile']
                if len(new_conf) > 2 and len(old_conf) > 2:
                    n = min(len(new_conf), len(old_conf))
                    corr = np.corrcoef(new_conf[:n], old_conf[:n])[0, 1]
                    results[dom] = max(0, corr)
        
        if not results:
            return None, 0.0
        best = max(results, key=results.get)
        return best, results[best]
    
    def transfer(self, target_domain, samples=None, tokenizer=None):
        """
        为学习新领域准备迁移。
        
        返回:
          source: 迁移来源领域名
          similarity: 相似度 
          info: 迁移详情列表
        """
        source, sim = self.find_most_similar(samples or [], tokenizer)
        info = []
        
        if source and sim > 0.3:
            src = self.pattern_bank[source]
            if src['n_layers'] <= self.llm.n_layers:
                info.append(f'weight_init_from_{source}')
            if src.get('attention_patterns'):
                info.append('attention_prior_transferred')
            action = f'transfer_from_{source}(sim={sim:.2f})'
        else:
            action = f'no_transfer(best_sim={sim:.2f})'
        
        self.transfer_log.append({
            'target': target_domain,
            'source': source,
            'similarity': sim,
            'action': action,
        })
        return source, sim, info
    
    def report(self):
        """输出迁移学习报告。"""
        print('\n=== CrossDomainTransfer ===')
        print(f'  已学领域: {list(self.pattern_bank.keys())}')
        print(f'  迁移次数: {len(self.transfer_log)}')
        for t in self.transfer_log[-3:]:
            src = t['source'] or '(none)'
            print(f'    {t["target"]} <- {src} ({t["action"]})')
