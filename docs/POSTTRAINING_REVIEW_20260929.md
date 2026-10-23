# 后训练检查与改进（2026-09-29）

这一轮先检查数据监督目标与评估划分，再讨论训练加速。已修复原生 DPO 的长提示标签边界，以及原生 SFT 的多轮对话验证集泄漏风险。未重新训练用户的大模型，也不由单元测试推断下游能力提升。

## 已修复：DPO 把提示尾部当作答案

原先 `_preference_features()` 先截断完整 prompt+answer，再只将 `max_prompt_length` 个前缀 token 掩掉。例如提示实际 1600 token、max_length=64、max_prompt_length=48，答案根本没有进入输入，但第 49–64 个提示 token 会被作为 chosen/rejected 的答案监督。若整个输入都被掩掉，旧逻辑还会强制把最后一个提示 token 设为标签。

现在先用完整联合 tokenization 找到答案边界；chosen/rejected 共用同一提示前缀，真正按预算保留其末尾，再填入回答并生成 labels。至少保留一个提示 token 和一个回答 token；拒绝空白回答、无共同上下文和无有效回答，校验长度参数。补入的分隔空白若是独立 token，归入提示掩码；若与答案合并，则随边界 token 参与监督。短提示联合编码保持不变，标签边界更严格。

使用本机 `D:\huggingface` 中 Qwen3-0.6B 的真实 tokenizer 离线验证：1614 token 中文提示和 1809 token 英文提示各保留 48 token 上下文，只有对应答案 `40/41`、`Paris/London` 被监督，chosen/rejected 的上下文相同。还检查短问题、BOS/EOS、跨边界 token 合并、空白回答和因果 shift。结果见 `artifacts/posttraining_review_20260929/real_tokenizer.json`。

这会改变旧错误样本的训练目标和截断后的有效上下文。应从原始数据重新生成特征；旧训练的 loss 与修复后的 loss 不能直接当作同一目标比较。不是自动修复已经训练出的 checkpoint。

## 已修复：SFT 同一对话出现在训练集与验证集

原生 SFT 会按 assistant 轮次展开多条带历史上下文的样本。过去在展开后随机分样本，一段对话的早期轮可能进入验证集，而后续训练轮已经包含这些文本。

现在 `validation_split` 按可用的原始记录分组，同一记录的所有轮次留在同一侧；分割种子仍固定可复现。只有一条多轮对话时明确报错，不能把多轮当作独立验证记录。极小数据集也至少保留一条训练记录；非法或 NaN 比例报错，避免悄悄关闭验证。

比例现在表示源记录比例，展开后的 token/样本比例可能不同。显式提供独立验证文件的路径不变；本轮没有跨文件去重，也没有按用户/来源去重，重复记录仍需数据审计。

## 后续最值得投入的工作

| 优先级 | 代码证据与问题 | 改进与验收 |
| --- | --- | --- |
| 高：原生 DPO 的参考模型计算 | `NativeDPOTrainer.compute_loss` 每步分别计算 chosen/rejected 的 policy 与 reference；参考模型冻结但仍重复前向 | 预计算固定参考 log-prob，按参考权重、tokenizer、模板、截断、数据内容构造缓存键；比较 loss/梯度、显存和整轮时间。不能只缓存字符串 prompt |
| 高：DPO 词表中间张量 | `_sequence_log_probs` 为所有位置展开 FP32 log-softmax，再取目标 token；policy/reference 四次前向同时保留 logits 变量 | 只处理被监督位置，按 token 分块或融合词表投影与 log-prob；必须验证 backward 和低精度，不先承诺倍数 |
| 中：原生训练恢复与结果记录 | HF/TRL 路径已有 resume 校验、末步评估与 latest loss；原生 SFT/DPO 配置和 `trainer.train()` 尚未接入同等恢复入口 | 统一 checkpoint 恢复、优化器/RNG/调度器校验、最终 eval 与数据特征统计；用不中断训练对照恢复后的参数和步数 |
| 中：SFT 多轮计算重复 | 原生每个 assistant 轮次重新带入全部历史；HF SFT 配置目前关闭 packing | 评估单记录多回答 mask 和长度分桶/packing；保证回答掩码、位置与文档间 attention 隔离，比较有效监督 token/s |
| 中：数据与质量闭环 | 现有小样本训练通过只能证明流程可运行 | 数据重复/来源隔离、丢弃原因、回答被截比例、留出集任务指标；用同样数据和训练预算验证改动收益 |

这些是依据当前源码列出的后续候选，尚未全部实现或测出收益。优先顺序建议：数据目标正确 → 独立评估可信 → 参考缓存/词表内存 → 恢复及吞吐。原生数据修复不等于修改所有 HF/TRL 后端。

## 验证与交付

测试覆盖 DPO 边界、SFT 分组、参数校验、既有后训练兼容、原生生命周期与分布式配置；包含小模型实际 SFT→DPO→保存→重载。完整结果与安装包见 [交付记录](../artifacts/posttraining_review_20260929/REPORT.md)。首轮边界测试复现 16 个失败，原始 XML 保留；随后新增了分组和入口接线检查。

修改集中在 [NativePostTraining.py](../saddlellm/training/NativePostTraining.py)，没有改 DPO 损失公式、优化器、用户权重或训练数据文件。
