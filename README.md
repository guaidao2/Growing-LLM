# GrowingLLM

动态参数增长的 Transformer 语言模型。从 2 层种子(2.2M 参数)开始,根据学习进度自动增加网络深度。

## 核心特性

- **动态生长**: 训练过程中 loss 停滞时自动添加 Transformer 层
- **知识蒸馏**: 由 MiniMind(64M)教师模型生成训练数据
- **极小种子**: 起始仅 2.2M 参数,按需增长至 11M+
- **GameNN 兼容**: 与 GameNN 系列共享"按需增长"设计理念

## 架构

```
输入 → Token嵌入(256d) → [Transformer块 × N] → LM头 → 输出

N = 2 → 3 → 4 → ... (自动增长)
```

## 快速开始

```bash
# 安装依赖
pip install torch numpy

# 训练 (含自动生长)
python train.py

# 推理
python -c "
from growing_llm import GrowingLLM, GrowingTokenizer
m = GrowingLLM(...)
m.load_state_dict(torch.load('models/llm.pth'))
print(m.reply('445端口怎么利用', tokenizer))
"
```

## 论文

详见 [PAPER.md](PAPER.md)

## 相关项目

- [Game-nn-O](https://github.com/guaidao2/Game-nn-O) — GameNN 系列起源
- [GameNN-WorldModel](https://github.com/guaidao2/GameNN-WorldModel) — GameNN 世界模型
- [MuLun-Mind](https://github.com/guaidao2/MuLun-Mind) — GameNN 侧枝决策模块
- [MuLun-Waf](https://github.com/guaidao2/MuLun-Waf) — GameNN WAF 应用

## 作者

guaidao2 (玄幕安全团队)
