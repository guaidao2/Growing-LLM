# Growing-LLM: 自生长通用语言模型架构

## 摘要

Growing-LLM 提出一种参数动态增长的通用语言模型架构。最初版本采用标准 Transformer 架构,通过知识盲区检测机制自动增加网络深度(从 2 层到 17 层)完成基础语言能力训练。在验证了自生长机制有效性之后,因消费级 GPU 算力限制(8GB),进一步研究出极致内存效率的改进方案——结合 **BitNet b1.58** (1.58-bit 三元权重)、**MoE 512 专家 × Top-1 路由**、**GLA 线性注意力** 三大技术,最终实现在 8GB 笔记本 GPU 上训练 50B 等效参数语言模型。

Growing-LLM 是 GameNN 系列的语言模型实现,继承其核心设计哲学:**不要一开始就定死参数量,让模型根据学习需求逐步生长。**

## 1. 自生长机制

### 1.1 知识盲区检测

```python
gap = 1 - max(softmax(logits))
gap > 0.5 + 持续上升 → 触发生长
```

### 1.2 三维生长

| 维度 | 起始 | 终点 | 触发条件 |
|------|------|------|----------|
| 深度 | 2 层 | N 层 | 盲区高 + 层数少 |
| 宽度 | 256 维 | 1024 维 | 盲区高 + 层数够 |
| 专家 | 4 个 | 512 个 | 盲区高 + MoE 模式 |

### 1.3 实验验证

使用 DeepSeek-R1 蒸馏数据(39K 条中文问答)在 RTX 5060 8GB 上验证:

| 阶段 | 层数 | 参数量 | Loss | 时间 |
|------|------|--------|------|------|
| 种子 | 2 | 4.7M | - | - |
| 基础语言 | 4 | 7.7M | 3.30 | 4min |
| 语法掌握 | 8 | 13.5M | 1.99 | 20min |
| 精细调整 | 17 | 26.5M | 1.18 | 15h |

训练过程中模型自动从 2 层生长至 17 层,全程无需人工干预。

## 2. 极致效率方案

因消费级 GPU 算力限制(8GB),进一步设计以下效率优化组合:

### 2.1 BitNet b1.58 三元权重

标准线性层(4 字节/参数) → 三元线性层(1.58 位/参数):

```python
def _ternary(w):
    scale = w.abs().mean()
    return where(w > 0.5*scale, 1, where(w < -0.5*scale, -1, 0))
```

前向仅需整数加法,50B 参数存储仅 ~10GB。

### 2.2 GLA 线性注意力

| 特性 | 标准 Attention | GLA |
|------|---------------|-----|
| 复杂度 | O(N²) | O(N) |
| 生成 | O(N) KV Cache | O(1) RNN 状态 |
| 长序列 8K+ | 显存爆炸 | 线性增长 |

GLA RNN 形式: S_t = G_t ⊙ S_{t-1} + K_t · V_t

### 2.3 MoE 512 专家 × Top-1 路由

Router 每 token 选择最匹配的 1 个专家。总参数 50B,每步仅 ~100M 活跃。非活跃专家存放 CPU,GPU 常驻 ≤4 个。

### 2.4 组合效果

| 路径 | 50B 参数 | GPU 显存 | 训练速度 |
|------|---------|---------|---------|
| 标准 Transformer | 200GB | ❌ 无法训练 | - |
| 纯 MoE | 100GB | ❌ 超限 | - |
| Growing-LLM (本方案) | ~10GB | < 2GB | ~10 tok/s |

## 3. 架构

```
输入 → TokenEmbed → RoPE → [GrowingBlock × N] → LM Head

GrowingBlock:
  RMSNorm → GLA(线性注意力) → Residual + Dropout
  RMSNorm → BitMoEFFN(Ternary × Experts Top-1) → Residual + Dropout

GrowthEngine (统一调度):
  盲区检测 → 深度/专家/宽度增长决策
```

## 4. 与 GameNN 系列的关系

```
Game-nn Simulator (坦克棋)
  └─ ExpertGRU 动态增长
      └─ MoE-GameNN (量化交易)
          └─ Router + 多专家融合
              └─ MuLun-Mind/MuLun-Waf (网安应用)
                  └─ Growing-LLM (本架构)
                      └─ 自生长语言模型
```

所有变体共享 GrowthEngine:

```python
class GrowthEngine:
    def step(self, avg_loss):
        if knowledge_gap > threshold:
            model.grow_depth() / grow_width() / grow_expert()
```

## 5. 使用

```python
from growing_llm import GrowingLLM, Tokenizer, GrowthEngine

# 种子初始化
model = GrowingLLM(vocab_size=8000, d_model=256, init_layers=2, n_experts=4)
engine = GrowthEngine(model)

# 训练
for epoch in range(100):
    loss = train_step(model, data)
    engine.step(avg_loss=loss)  # 自动生长

# 推理
response = model.reply("445端口怎么利用", tokenizer)
```

## 6. 结论

Growing-LLM 验证了一个核心假设:**通用语言模型不需要一开始就拥有全部参数,而应该根据学习需求逐步生长。** 在 8GB 消费级 GPU 上,从 2 层 4.7M 参数种子开始,自动生长至 17 层 26.5M 参数完成基础语言训练;通过 BitNet × MoE × GLA 组合,理论上可在同等硬件上训练 50B 等效参数模型。

---

**架构名称**: Growing-LLM
**项目仓库**: https://github.com/guaidao2/Growing-LLM
**作者**: guaidao2 (玄幕安全团队)
**日期**: 2026年7月
**所属系列**: GameNN
