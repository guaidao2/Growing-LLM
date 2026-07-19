# Growing-LLM

动态参数增长的通用语言模型架构。在 8GB 笔记本 GPU 上从 2 层种子生长至 50B 等效参数。

## 核心创新

- **自生长机制** — 从极小种子开始,知识盲区触发网络深度/宽度/专家数自动增长
- **BitNet b1.58** — 1.58-bit 三元权重,50B 参数仅需 ~10GB 存储
- **MoE 512 专家 × Top-1 路由** — 每步仅激活 ~100M 参数  
- **GLA 线性注意力** — O(N) 训练复杂度, O(1) 推理复杂度
- **GameNN 兼容** — 与 GameNN 系列共享生长引擎

## 架构

```
TokenEmbed → RoPE → [GrowingBlock × N] → LM Head

GrowingBlock:
  RMSNorm → GLA Attention → Residual
  RMSNorm → BitMoE FFN (512专家 Top-1) → Residual
```

## 快速开始

```python
from growing_llm import GrowingLLM, Tokenizer, GrowthEngine

model = GrowingLLM(vocab_size=8000, d_model=256, init_layers=2, n_experts=4)
engine = GrowthEngine(model)
# 训练中自动生长: engine.step(avg_loss=loss)
```

## 论文

详见 [PAPER.md](PAPER.md)

## GameNN 系列

- [Game-nn-O](https://github.com/guaidao2/Game-nn-O)
- [GameNN-WorldModel](https://github.com/guaidao2/GameNN-WorldModel)
- [MuLun-Mind](https://github.com/guaidao2/MuLun-Mind)
- [MuLun-Waf](https://github.com/guaidao2/MuLun-Waf)

## 作者

guaidao2 (玄幕安全团队)
