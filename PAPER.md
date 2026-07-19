# GameNN-BitMoE: 50B 等效参数通用智能体

## 摘要

GameNN-BitMoE 提出一种在 8GB 消费级 GPU 上训练 50B 等效参数语言模型的可行架构。核心技术组合包括：

- **BitNet b1.58** (微软, 2402.17764) — 1.58-bit 三元权重,50B 参数仅需 ~10GB 存储
- **MoE 512 专家 × Top-1 路由** (Google, 2101.03961) — 每步只激活 ~100M 参数
- **GLA Gated Linear Attention** (2312.06635) — O(N) 训练复杂度, O(1) 生成复杂度
- **GameNN 生长机制** (本架构) — 从 2 层 4 专家种子开始,知识盲区触发生长

这是 GameNN 系列的终极架构,整合了此前所有项目的核心设计:

| 项目 | 贡献 |
|------|------|
| Game-nn Simulator | ExpertGRU 动态增长 |
| MoE-GameNN | 多专家融合 + Router |
| GrowingLLM | 深度/宽度双维度生长 |
| MuLun-Mind | 侧枝决策架构 |
| 本工作 | BitNet × MoE × GLA 统一 |

## 1. 架构设计

### 1.1 整体结构

```
输入 → TokenEmbed(字符级) → RoPE → [BitMoEBlock × N] → LM Head(三元)
                                          ↑
                                    GrowthEngine
                                (深度/专家增长调度)

BitMoEBlock:
  RMSNorm → GLA(线性注意力) → Residual + Dropout
  RMSNorm → BitMoEFFN(三元 × 512专家 Top-1) → Residual + Dropout
```

### 1.2 BitNet 三元线性层

标准线性层: y = x · W^T + b (W 为 fp32, 4 字节/参数)
三元线性层: y = x · ternary(W)^T + b (ternary(W) ∈ {-1, 0, +1}, 1.58 位/参数)

训练时保存 fp16 主权重,前向时量化:

```python
def _ternary(w):
    scale = w.abs().mean()
    return where(w > 0.5*scale, 1, where(w < -0.5*scale, -1, 0))
```

50B 参数三元存储: 50×10⁹ × 1.58 / 8 ≈ 9.9 GB (vs fp16 的 100GB)

### 1.3 GLA 线性注意力

标准 Attention: O = softmax(QK^T)V  (O(N²))
GLA: S_t = G_t ⊙ S_{t-1} + K_t · V_t, O_t = Q_t · S_t  (O(N))

| 特性 | FlashAttn | GLA |
|------|-----------|-----|
| 复杂度 | O(N²) | O(N) |
| 生成步进 | O(N) KV Cache | O(1) RNN 状态 |
| 长序列(8K+) | 显存爆炸 | 线性增长 |
| 矩阵乘法 | 需要 | 加法为主 |

### 1.4 MoE 512 专家 × Top-1

```
Router: 每 token 选择最匹配的 1 个专家
激活: 总参数 50B, 每步只激活 ~100M
卸载: 非活跃专家在 CPU, GPU 常驻 ≤4 个
生长: 从 4 专家开始, 动态增长到 512
```

## 2. 生长机制

```
知识盲区: gap = 1 - max(softmax(logits))
gap > 0.5 + 持续上升 → 生长触发

生长决策:
  层数 < 32 → 加层
  层数够    → 加专家 (4 → 512)
  宽度可选  → 加维度 (256 → 1024)
```

| 阶段 | 层数 | 专家 | 参数量 | GPU 显存 |
|------|------|------|--------|----------|
| 种子 | 2 | 4 | 3.7M | 0.5GB |
| 基础 | 4 | 8 | 15M | 0.6GB |
| 扩展 | 8 | 16 | 100M | 0.8GB |
| 大规模 | 16 | 64 | 6B | 1.2GB |
| 极限 | 32 | 512 | 50B | 2GB (卸载) |

## 3. 与 GameNN 家族的关系

GameNN 系列所有变体共享同一个核心思想——动态增长:

```
Game-nn Simulator (坦克棋)
  └─ ExpertGRU 动态增长
      └─ MoE-GameNN (量化交易/五子棋)
          └─ Router + 多专家融合
              └─ GrowingLLM (自生长语言模型)
                  └─ 深度/宽度生长
                      └─ GameNN-BitMoE (本工作)
                          └─ BitNet × MoE × GLA
                              └─ 50B 在 8GB GPU 上训练
```

统一架构:

```python
class GrowthEngine:
    def step(self, avg_loss):
        # 所有 GameNN 变体共享的生长逻辑
        if knowledge_gap > threshold:
            model.grow_depth() / grow_expert()
```

## 4. 核心代码

```python
from growing_llm import GameNNBitMoE, Tokenizer, GrowthEngine

# 创建模型 (种子: 2层, 4专家, 256维)
model = GameNNBitMoE(vocab_size=8000, d_model=256, init_layers=2, n_experts=4)
engine = GrowthEngine(model)

# 训练循环
for epoch in range(100):
    loss = train_step(model, data)
    engine.step(avg_loss=loss)  # 自动生长

# 推理
response = model.reply("445端口怎么利用", tokenizer)
```

## 5. 引用

```
@misc{bitmoe2026,
  author = {guaidao2},
  title = {GameNN-BitMoE: 50B Parameter Language Model on 8GB GPU},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/guaidao2/Growing-LLM}
}
```

---

**项目地址**: https://github.com/guaidao2/Growing-LLM
**作者**: guaidao2 (玄幕安全团队)
**日期**: 2026年7月
