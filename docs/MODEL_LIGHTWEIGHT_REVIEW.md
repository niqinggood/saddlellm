# 模型轻量化：现有能力与下一步

2026-09-27 对当前源码的只读审查。本次没有改写、量化或剪枝用户的模型权重。
轻量化不等于推理加速：文件大小、加载后显存、首 token 等待、生成吞吐和质量要分别验收。

## 当前能做什么，缺什么

| 能力 | 代码依据与现状 | 需要完善的部分 |
| --- | --- | --- |
| CPU 动态 INT8 | [quantizer.py](../saddlellm/compression/quantizer.py) 将 Linear 转为 PyTorch 动态 INT8；CLI `quantize` 保存量化 state 与 manifest，服务识别此格式 | 不是 GPU INT8 路径；当前 CLI 从 HF 加载，原生 Saddle 检查点的量化、保存、重载还需单独打通并验证。 |
| HF GPU 4/8bit | [InferenceServer.py](../saddlellm/runtime/InferenceServer.py) 的 `serve --quantization 4bit/8bit` 在加载时使用 bitsandbytes | 原生 Saddle 检查点明确拒绝这条路径。需要按模型/硬件测加载、显存、速度和任务质量，不能宣传所有模型都支持。 |
| 掩码剪枝 | [pruner.py](../saddlellm/compression/pruner.py) 支持非结构化和按行结构化置零；保存时物化掩码 | 矩阵 shape 没变，参数槽位/普通 dense Linear 的工作量没变；不能当作真正删掉神经元或默认加速。head/layer 物理裁剪尚未实现。 |
| 小模型蒸馏 | [DistillationTrainer.py](../saddlellm/distillation/DistillationTrainer.py) 与 [distill.py](../saddlellm/distillation/distill.py) 已有蒸馏入口 | 需要用真实较小学生、独立验证集、模型体积和推理耗时共同验收；仅对同大小学生训练不等于模型更小。 |
| KV 量化 | [lightweight.py](../saddlellm/compression/lightweight.py) 的 `_configure_kv_cache` 仅设置 `_kv_cache_bits` 并打印提示 | 当前原生 [InferenceLayerCache](../saddlellm/models/InferenceCache.py) 仍保存原 dtype tensor，不读取此属性；不能据此声称已减少 KV 显存。 |
| 一键组合 | `Lightweight.optimize` 调用剪枝、量化、KV 配置，再直接 `model.save_pretrained` | 需要统一到量化器的持久化格式，检查重新加载；成功、配置提示和跳过/失败必须分开报告。 |

`Lightweight` 的默认 4bit 路径传入已加载模型，但 bnb 分支要求模型名并重新加载，因此可能
被跳过；CPU 动态 INT8 转换后也不能假定通用 `save_pretrained` 能保存可重载量化模型。
这些是需要验收的组合接口缺口，不是已有 `quantize` CLI 保存 manifest 的能力不存在。
该模块打印“移除参数”和理想位数内存比例，仍需改为真实形状、实际字节和已应用结果。
`_configure_kv_cache` 注释里的固定质量损失比例没有本项目测量依据，不应作为质量承诺。

## 建议顺序

1. **先保证结果真实、产物能重载**：统一 capability/result/manifest；配置、已应用、跳过、
   失败分别呈现；保存后从新进程加载并生成。来源检查点保持不变，产物写入新目录。
2. **做压缩验收报告**：用同一任务数据比较原模型和轻量模型，报告参数/实际文件字节、
   加载后内存、prefill/生成延迟、PPL 或任务指标和质量门禁。量化输出无需 token 完全相同，
   但必须评估真实质量，不能仅靠随机 token benchmark 宣称无损。
3. **原生 weight-only 量化**：在现有 Linear 后端扩展边界内提供可选实现；明确设备、层类型、
   加载和保存格式，保留原浮点路径。先证明本机收益，再考虑额外算子和依赖。
4. **蒸馏成真正更小的学生**：可减少层数、hidden/intermediate，但结构和 checkpoint 必须一致；
   带教师监督恢复任务质量，比只把 dense 权重置零更符合“模型更小”的目标。
5. **KV 量化和物理剪枝后置**：只有确认长上下文/并发 KV 或模型宽度是瓶颈时再做；
   KV 量化涉及保存、读取时反量化及精度检查，物理裁剪涉及成组投影、GQA、残差和配置一致性。

当前 28M 模型词表 32768、hidden=384、embedding/head tied；共享词表矩阵包含
12,582,912 个参数，约占总参数的 44.4%。只量化普通 Linear 不等于整个模型缩为四分之一，
量化 head 还需明确是否保持共享，最终以实际保存与加载后的字节为准。

这些工作仍应通过原来的 CLI/import 暴露，后端依赖按需启用。第一阶段宜把已有接口的
正确性、兼容性和验收闭环做完整，再扩展量化算法数量。本文件是审查与建议，以上缺口未在
本次前缀缓存开发中修复，不应将其当作已实现功能。
