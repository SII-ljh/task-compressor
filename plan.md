# QC-ICAE: Query-Conditioned In-Context Autoencoder

## 项目构建计划

---

## 1. 核心思想

借鉴 ICAE (ICLR 2024) 的 LoRA encoder + frozen decoder 架构，构建 query-conditioned 的上下文有损压缩系统：LoRA Encoder 编码长文本得到完整 hidden states，Query-Conditioned Perceiver 以用户 prompt 为引导从中提取 task-relevant 信息，压缩为 k=64 个 soft prompt 送入 frozen decoder 生成。

Base model: Qwen2.5-7B-Instruct (hidden_dim=3584, num_heads=28)。Encoder 和 Decoder 共享同一份 Qwen 权重，Encoder 额外带 LoRA adapter，Decoder forward 时关闭 LoRA 等效于原始 Qwen。

---

## 2. 模型架构

### 2.1 LoRA Encoder

Qwen + LoRA (rank=128, target: q_proj & v_proj)。输入长文本 token ids (B, L)，输出所有 token 位置的 last hidden states (B, L, 3584)。不加 memory token，所有压缩交给 Perceiver。支持 gradient checkpointing。

### 2.2 Prompt Encoder

将变长用户 prompt 压缩为固定 n_p=16 个向量。使用 frozen Qwen embedding 层得到 prompt embeddings，然后用 n_p 个可学习 latent token 通过单层 cross-attention attend prompt embeddings，输出 (B, 16, 3584)。

### 2.3 Query-Conditioned Perceiver

标准 Perceiver，不魔改。Query 由两部分拼接：Prompt Encoder 输出的 n_p=16 个 prompt-conditioned token + n_c=48 个纯可学习 context token，共 k=64 个。KV 是 LoRA Encoder 的完整 hidden states。堆叠 2 层 Perceiver block（每层: cross-attention → self-attention → FFN，均带 Pre-LN + residual）。所有维度与 Qwen 一致 (d=3584, heads=28, ffn=14336)，无需任何投影层。输出 (B, 64, 3584)。

### 2.4 Decoder Wrapper

Frozen Qwen（关闭 LoRA）。输入拼接顺序：[64个soft prompt] + [1个可学习separator token] + [prompt token embeddings]。训练时后接 target token embeddings 做 teacher-forcing；推理时 autoregressive 生成。

### 2.5 完整模型

组装以上四个组件。提供 encode() 和 compress() 接口用于推理时分步执行和缓存。

---

## 3. 训练流程

单阶段联合训练。LoRA Encoder + Perceiver + Prompt Encoder + separator token 全部从头联合训练。

理由：LoRA 初始化时 delta≈0，Perceiver 看到的就是原始 Qwen 的 hidden states——已经是高质量的语义表示，不存在冷启动问题。随训练推进，LoRA 和 Perceiver 自然协同优化。

训练目标：L = L_QA + α·L_distill，α=0.5。

- L_QA：response generation 的 cross-entropy。Perceiver 输出的 64 个 soft prompt + separator + prompt → frozen decoder → 生成 response，loss 仅算在 response token 上。
- L_distill：蒸馏 loss。Teacher 是 frozen Qwen（关闭 LoRA）直接吃未压缩原文 + prompt 的 logits；Student 是吃 soft prompt + prompt 的 logits。KL divergence (T=2.0) 对齐。Teacher forward 不需要梯度。

数据：(context, prompt, response) 三元组，使用已有的数据下载脚本。

差异化学习率：Perceiver / PromptEncoder / separator 从零开始，用 5e-5；LoRA 在强预训练基础上微调，用 1e-5。通过 optimizer param_groups 实现。

超参：batch=128, per_gpu_batch=2, gradient_accumulation=8, warmup=500, total_steps=50k, bf16, max_grad_norm=2.0, weight_decay=0.01。

---

## 4. 损失函数汇总

| Loss | 描述 | 权重 |
|------|------|------|
| L_QA | CE(decoder 从 soft prompt + prompt 生成 response) | 1.0 |
| L_distill | KL(student_logits ∥ teacher_logits) * T² | α=0.5 |

蒸馏 loss 支持两种模式可切换：KL on logits（默认）和 MSE on hidden states。

Teacher forward 的流程：关闭 LoRA → 原文 context token embeddings + prompt token embeddings 直接送 Qwen → 取 response 位置的 logits。与 student 共享同一份 Qwen 权重，仅输入不同。

---

## 5. 推理流程

离线：长文本过 LoRA Encoder → hidden states → 缓存（内存/磁盘，safetensors 格式）。

在线：用户 prompt 过 Prompt Encoder → prompt latents → Perceiver 从缓存的 encoder hidden states 中压缩 → 64 个 soft prompt → Decoder 生成 response。

实现一个 EncoderCacheManager 管理缓存的 put/get/evict。实现 benchmark 脚本测延迟和吞吐。

---

## 6. 消融实验

| 实验 | 变量 | 默认值 | 消融值 |
|------|------|--------|--------|
| A1 | k (soft prompt数) | 64 | 16, 32, 128 |
| A2 | n_p/n_c 比例 | 16/48 | 0/64, 8/56, 32/32 |
| A3 | Perceiver层数 | 2 | 1, 3, 4 |
| A4 | 蒸馏权重α | 0.5 | 0, 0.3, 1.0 |
| A5 | 蒸馏方法 | KL | MSE |
| A6 | Prompt注入方式 | 方案3(query融合) | 方案2(KV拼接) |
| A7 | LoRA rank | 128 | 64, 256 |
| A8 | LoRA学习率 | 1e-5 | 5e-5(与Perceiver统一), 0(冻结) |
| A9 | 加入ICAE式AE预训练阶段 | 无 | 有：先用AE+LM loss训LoRA+memory tokens，再接Perceiver联合训练 |

A2 中 n_p=0 意味着纯 learnable query 不注入 prompt，验证 prompt conditioning 的价值。A8 中 LoRA lr=0 等价于冻结原始 Qwen encoder，验证 LoRA 微调是否有帮助。A9 验证 ICAE 式预训练是否优于直接联合训练。

通过 yaml 配置覆盖实现，提供一键运行脚本。所有实验接入 wandb。

---

## 7. 评估指标

QA 质量：Response BLEU/ROUGE；GPT-4 pairwise judge（win/lose/tie）。对比 baseline 包括：原始 Qwen 吃完整 context（upper bound）、128-token GPT-4 summary、ICAE 原版方案（同等 k 值）。

效率：端到端延迟、缓存命中延迟、GPU显存占用、吞吐量。

压缩质量：不同 k 值下的 QA 准确率曲线。

---

## 8. 多卡训练

默认 PyTorch DDP，大显存需求时切 DeepSpeed ZeRO-2。训练脚本通过 torchrun 启动。Checkpoint 仅 rank 0 保存，日志仅 rank 0 写 wandb。bf16 训练。

---

## 9. 单元测试

每个模块一个测试文件，覆盖：

- **lora_encoder**: 输出形状、LoRA参数量（~1%总参数）、梯度只流过LoRA、enable/disable toggle行为
- **perceiver**: 输出形状 (B,k,D)、变长context输入兼容、attention mask正确性、无NaN
- **prompt_encoder**: 输出形状 (B,n_p,D)、frozen embedding无梯度、变长prompt兼容
- **qcicae**: forward+backward、generate输出非空、encoder-decoder权重共享验证、encode/compress分步执行
- **losses**: teacher=student时KL≈0、权重正确应用、梯度不流过teacher
- **data**: tokenization正确、collator padding正确、数据加载速度
- **inference**: pipeline端到端、cache hit加速、batch推理
- **distributed**: 梯度同步一致性、checkpoint save/load

---

## 10. 关键设计备忘

1. **Encoder-Decoder权重共享**：只加载一份Qwen，通过peft的enable/disable adapter切换encoder和decoder模式。Teacher forward也是disable adapter后的原始Qwen吃完整context。

2. **单阶段联合训练**：LoRA初始delta≈0，Perceiver起点就是原始Qwen的高质量hidden states，无冷启动问题。差异化学习率（LoRA 1e-5、Perceiver 5e-5）替代多阶段训练的复杂度。

3. **Perceiver维度对齐**：所有维度与Qwen一致，零投影层。

4. **显存预估**：Qwen 7B bf16 ~14GB + activations + teacher forward ≈ 33GB/卡，A100 80GB可行，per_gpu_batch=2。扩展到长文本(8K+)需gradient checkpointing + DeepSpeed。
