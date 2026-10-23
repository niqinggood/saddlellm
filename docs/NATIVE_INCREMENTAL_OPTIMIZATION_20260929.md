# 原生推理：KV 合组复用与生成路径计算裁剪

本轮落实[调研](INFERENCE_NEXT_ACCELERATION_RESEARCH_20260929.md)中的前三项增量优化：保留有效 KV 合组、省掉稠密生成的辅助损失零张量、裁剪 prefill 末尾归一化。运行库修改集中在 `NativeBatchDecode.py`、`NativeGeneration.py` 和 `SaddleModeling.py`。

**行为变化**

- B3/B1 或 B2/B2 交替推进时，不涉及本次请求的有效 cohort 可以保留；成员单独推进或换组时使重叠组失效，复用前仍检查 buffer/长度。结束、关闭、失败和 `forget()` 会释放相应引用。
- 保留组的总行数不超过 `max_batch_size`，各组按最大层容量计算的 KV token 槽位总和不超过 `max_batch_tokens`；超限按最久未使用顺序淘汰。淘汰不关闭调用方 session，后续可以重新合组。它是执行器引用的保留上限，不是整个服务的显存准入保证。
- 原生 session 使用内部生成专用 forward：普通稠密层无需创建未使用的辅助损失零张量；只建 KV 的中间 prefill 块可以跳过末尾 RMSNorm；需要 logits 时，末尾 RMSNorm 只计算选中的位置。
- 普通 `forward()` 继续返回完整 hidden_states，训练损失与反向传播不变。自定义 forward、相关 hook、定制 norm 保留相应完整计算路径。MTP、MoE 和 AttnRes 的专用语义不被当作普通稠密层裁剪。
- BF16 批量 Linear 的逐行数值保护保留。这轮没有新增不同长度统一 attention、采样合批、分页 KV 或 CUDA Graph。

原有 `--native-stream-batching` 和分块 prefill 开关继续决定是否使用相应服务路径。无需为本轮三项改动新增 CLI 参数。`NativeBatchDecodeRunner.snapshot()` 新增 `retained_cohort_token_slots`，用于观察当前保留组的槽位占用；请求最终清理后应为零。

**本机 GPU A/B**

同一台 Windows / RTX 4090 Laptop、Torch 2.6.0+cu124、本地已有 28M 检查点、BF16。基线加载本轮修改前保存的三个源文件；候选加载当前源码，其余模型模块相同。两个模型使用同一个检查点，均驻留 GPU；未开启额外融合开关，现有 BF16 保护保持一致。

每个场景每个变体预热两次，随机顺序配对七轮。固定合成输入，greedy、EOS 关闭、每请求生成 48 token；前缀缓存关闭。计时包含 session 创建、prefill/decode、每 token 的 CPU 交付、输出复制和清理，不含模型加载、分词、HTTP。桌面 GPU 非独占，不能代表 L20 或生产服务容量。

| 场景 | 总输出 token/s：修改前 → 后 | 配对耗时比中位数 | 更快的配对 | 结论 |
| --- | ---: | ---: | ---: | --- |
| C1，128-token 输入 | 22.64 → 22.90 | 1.0066× | 4/7 | 未证明稳定提速 |
| C4 同组，128-token 输入 | 45.88 → 44.55 | 1.0082× | 4/7 | 未证明稳定提速，总吞吐略降 |
| C4，固定 B3/B1 交替 | 29.75 → 32.29 | 1.0844× | 7/7 | 此受控场景观察到收益 |
| C1，256-token prompt / 64-token 分块 | 20.80 → 20.92 | 0.9983× | 3/7 | 未证明稳定提速 |

总 token/s 为各轮真实输出之和除以耗时之和；配对比是逐对耗时比的中位数，两者不能混算。四组全部输出 token ID 相同，峰值 PyTorch allocated 没有增加。allocated 包含两个驻留模型，不是单模型部署显存；这里也没有 HTTP 首字或稳定 P95 的结论。

B3/B1 场景每轮 cohort 创建次数为 **47 → 1**，合组历史复制字节为 **109,009,920 → 1,966,080**，约减少 **98.2%**。这是合组复制计数，不包含全部 KV 扩容或其他 copy。7 对合计输出完全相同。表中速度是三项合并改动的结果，没有单独消融来精确分配各项贡献。

该场景另启进程独立复测七对，得到 **30.08 → 32.61 token/s**，配对中位 **1.0839×**，仍为 7/7 更快、全部 token 相同，复制量及建组次数与首轮相同。首轮 token 全部交付的中位时间为 193.63 → 197.83 ms，约增加 2.2%，不能说每个指标均提升。两轮都没有 allocated 峰值增长。见[独立复测记录](../artifacts/native_incremental_optimization_20260929/local_bf16_interleaved_repeat.json)。

原始记录和脚本：[本轮证据目录](../artifacts/native_incremental_optimization_20260929/)、[首次 GPU 记录](../artifacts/native_incremental_optimization_20260929/local_bf16_run1.json)。报告保留逐轮耗时、输出 token、复制计数和运行源码 SHA256。

**验证范围**

最终 **597 项相关回归全部通过，无跳过**，包括 CUDA FP32/BF16/FP16 readout 与公共 forward 的 logits/KV 逐元素相等、CUDA BF16 交替组、训练反向传播、普通 hidden_states 合约、hook/自定义 norm 回退、合组保留/淘汰/进出、断连/慢客户端/异常清理、融合算子、前缀缓存、模型兼容与分块 HTTP 路由。机器记录见证据目录 `regression.xml`。Ruff 选定错误检查、发布元数据和生成函数索引检查也通过；函数索引已按当前源码刷新。

小型 CPU 算子诊断确认：两层稠密模型的生成专用路径不再执行原有两次 `aten::new_zeros`；中间 prompt 块少一次最终 norm 的归约。计数变化不等于完整模型速度承诺。

本轮是在当前源码上交付和验证；没有替换服务器安装包。大模型、真实流量及 Linux/L20 的性能需要在相同版本与明确负载下另行验证。
