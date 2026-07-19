# Growing-LLM

GameNN-BitMoE — 50B 等效参数通用智能体框架，在 8GB 笔记本 GPU 上运行。

## 核心技术组合

- **BitNet b1.58** — 1.58-bit 三元权重,50B 参数仅需 ~10GB
- **MoE 512 专家 × Top-1 路由** — 每步只激活 ~100M 参数
- **GLA 线性注意力** — O(N) 训练, O(1) 生成, 长序列无损
- **GameNN 生长机制** — 从 2 层 4 专家开始, 知识盲区触发生长

## 架构

```
TokenEmbed → RoPE → [BitMoEBlock × N] → LM Head

BitMoEBlock:
  RMSNorm → GLA(线性注意力) → residual
  RMSNorm → BitMoEFFN(Ternary × 512专家 Top-1) → residual
```

## 快速开始

```python
from growing_llm import GameNNBitMoE, Tokenizer, GrowthEngine

model = GameNNBitMoE(vocab_size=8000, d_model=256, init_layers=2, n_experts=4)
engine = GrowthEngine(model)
# 训练循环中: engine.step(avg_loss=loss)  # 自动生长
```

## 论文

详见 [PAPER.md](PAPER.md)

## GameNN 系列

- [Game-nn-O](https://github.com/guaidao2/Game-nn-O) — GameNN 系列起源
- [GameNN-WorldModel](https://github.com/guaidao2/GameNN-WorldModel) — GameNN 世界模型
- [MuLun-Mind](https://github.com/guaidao2/MuLun-Mind) — GameNN 侧枝决策模块
- [MuLun-Waf](https://github.com/guaidao2/MuLun-Waf) — GameNN WAF 应用

## 作者

guaidao2 (玄幕安全团队)
