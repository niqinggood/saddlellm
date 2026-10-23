# 原生多模型与推理优化验证记录（2026-09-28）

本轮完成架构扩展、两个性能候选的真实模型 A/B、自动验收门槛和服务验证。
**没有把未通过独立复测的候选默认开启。** 验证过程与失败记录均保留，不只挑最好看的数字。

## 环境与口径

Windows、RTX 4090 Laptop 16 GB、PyTorch 2.6.0+cu124、Transformers 4.57.6。
真实 Qwen3-0.6B、DeepSeek-R1-Distill-Qwen-1.5B，CUDA BF16，贪心，固定提示输出 96 token。
每个变体预热 2 次，7 对 AB/BA；一份模型、同一精度与投影策略。已有服务驻留显存，测试请求不并行压测。
设备并非隔离生产环境，速度有波动。吞吐包括 prefill、decode、streamer，不含加载和 HTTP。

## 根据实测做的取舍

| 候选 / 模型 | 基线 → 候选输出 token/s（中位数） | 更快的配对 | 结论 |
| --- | --- | --- | --- |
| 单请求控制 / Qwen3-0.6B | 39.61 → 39.48 | 3/7 | 无收益，默认禁用 |
| 单请求控制 / DeepSeek 蒸馏 1.5B | 42.37 → 41.00 | 1/7 | 回退趋势，默认禁用 |
| RoPE 合并 / Qwen3 探索轮 | 41.14 → 43.11 | 7/7 | 初步正向，不能独立作发布结论 |
| RoPE 合并 / Qwen3 复测 | 39.62 → 40.89 | 4/7 | +3.2%，稳定性门槛未通过 |
| RoPE 合并 / DeepSeek 第一轮 | 41.40 → 43.20 | 7/7 | +4.3%，本轮初筛通过 |
| RoPE 合并 / DeepSeek 独立复测 | 41.94 → 43.91 | 4/7 | +4.7%，稳定性门槛未通过 |

所有上述 A/B 的输出 token ID、文本哈希和结束原因完全一致。两项候选均不涉及量化或权重改动。
两模型进程 PyTorch allocated 峰值分别约 1.13 GiB 和 3.32 GiB，候选与基线相同；不是整卡占用。
RoPE 合并虽有正向中位数，但尚不足以保证普遍稳定加速，故 `pack_decode_rotary` 默认 false，仅保留显式实验能力。
单请求控制候选同样默认关闭，且不适用于带自定义停止条件的 HTTP 路径；不能声称它加速了网页聊天。

探索轮运行期间测量脚本增加了报告字段，因此该轮末尾记录的脚本哈希不作为可复现代码快照；后续 release 报告改为启动时捕获哈希。
全部原始报告位于 [实验目录](../artifacts/native_multi_model_20260928/)，包括失败和探索轮：

- [Qwen3 单请求候选](../artifacts/native_multi_model_20260928/qwen3_single_row.json)、[DeepSeek 单请求候选](../artifacts/native_multi_model_20260928/deepseek_single_row.json)
- [Qwen3 RoPE 复测](../artifacts/native_multi_model_20260928/qwen3_rotary_release1.json)
- [DeepSeek RoPE 第一轮](../artifacts/native_multi_model_20260928/deepseek_rotary_release1.json)、[独立复测](../artifacts/native_multi_model_20260928/deepseek_rotary_release2.json)
- [算子剖析](../artifacts/native_multi_model_20260928/qwen3_profile.json)：只用于定位调用密集点，profiler 自身开销不可作为吞吐。

## 正确性、服务与交付

19 个相关测试文件共 **383 项通过**，包含新 Qwen2/Llama 同权重验证、CUDA FP16/BF16、缺失 bias 拒绝、分片与保存重载、RoPE 逐元素一致性、生成、停止词、断连、前缀缓存、投影与合批回归。
不是全仓库测试，也不是完整版 Kimi/MiniMax 验收。[JUnit 记录](../artifacts/native_multi_model_20260928/regression.xml)。

新增本机 `http://127.0.0.1:8130/`：DeepSeek-R1-Distill-Qwen-1.5B，`backend=native`、`model_type=qwen2`、`streaming_backend=native_step`。
保留 8127/8128/8129 原服务，没有修改它们的模型和历史记录。
官方 OpenAI SDK 已验证模型列表、聊天、文本补全、SSE/usage、参数别名、拒绝不支持 tools、客户端断开后继续请求；[结果](../artifacts/native_multi_model_20260928/deepseek_native_sdk_smoke.json)。
这是 API 功能验证，不是医学或通用答题质量认证；32 token 冒烟回答可能停在思考中。

### HTTP 对照发现的问题与继续验证

HF 8128 与 native 8130，同一真实模型、BF16、原始 completion 提示、流式 C1，预热一轮、每轮 3 个请求、重复 3 轮：

| 后端 | 总输出 token/s | P50 总延迟 | P95 总延迟 | P50 首个可见文本 |
| --- | --- | --- | --- | --- |
| HF local | 34.63 | 2769.67 ms | 3008.50 ms | 37.48 ms |
| Saddle native | 39.47 | 2423.61 ms | 2671.53 ms | 29.05 ms |

**这不是质量等价的加速结论**：两者 96 token 用量相同，但回答不同，native 有明显重复。
对照脚本因此以失败退出；不把这组表当作宣传优势。[HTTP 原始记录](../artifacts/native_multi_model_20260928/deepseek_http_c1.json)。

按方案继续排查：关闭全部投影合并后分叉仍存在；只把 native 改为 eager、HF 仍为 SDPA，也没有消除分叉。
随后添加顺序加载的同前缀验证脚本（不同时在 GPU 上放两份权重），双方都设 eager、不合并投影：
真实 1.5B 的 16 个前缀步骤 logits **最大绝对误差全部为 0**，argmax 全部相同。
双方都设 SDPA 时，最大绝对差约 0.109–0.188，第 5 步 argmax 分叉；同前缀检查未通过。
这将问题定位到 SDPA 执行路径的数值差异，而非这份权重的 bias 映射或投影合并。
本机 HF 可使用 compact GQA 的数学回退，Saddle 则优先有效的 fused SDPA/单步 GQA folding，两者调用路径并不相同。

[eager 同前缀报告](../artifacts/native_multi_model_20260928/deepseek_eager_logits.json)、[SDPA 同前缀报告](../artifacts/native_multi_model_20260928/deepseek_sdpa_logits.json)。
上述 16 步使用模型直接前向的动态 KV，对定位有用，但不代表所有提示、长输出与 HTTP 缓存路径的完整等价证明。
因此 DeepSeek 原生适配仍标记实验性；正式质量需求优先使用已经验收的模型/后端组合。

最终 native 服务还完成流式 C4 短测：输出上限 32、每轮 4 请求、预热 1 轮、计时 2 轮，8 个请求均完成、输出一致，结束后活跃请求回到 0。
总输出约 49.34 token/s，P50/P95 总延迟 1618.51/2596.63 ms，P50/P95 首字 995.28/1968.47 ms。
此负载与上面的 96-token C1 不同，**不能据此计算并发加速比**；它暴露了排队导致的首字延迟，并不是连续批处理成果或长期稳定性证明。
[C4 短测报告](../artifacts/native_multi_model_20260928/deepseek_native_http_c4_smoke.json)。最终 8127–8130 均 ready、active=0。

```powershell
python tools/validate_native_dense_logits.py --model /models/DeepSeek-R1-Distill-Qwen-1.5B --attention eager --output artifacts/my_dense_logits.json
```

本地 [2.40 wheel](../artifacts/native_multi_model_20260928/wheel/saddlellm-2.40-py3-none-any.whl) 已构建并通过包合约检查，6 个本轮核心模块与源码逐字节一致；未发布 PyPI、未上传服务器。
SHA-256：`37ab9c9d84979e0fac7ab64cdda438a45c3d3349d0200c0aff43310c01104638`。
全局旧 setuptools 的无隔离构建失败；使用项目声明的 `setuptools>=77` 隔离构建成功，没有修改全局依赖。

## 下一轮优先级

1. 使用隔离负载扩大提示集合及配对样本，确认 RoPE 候选的稳定收益；失败则继续默认关闭。
2. 根据 profiler 拆分归一化、RoPE、缓存写入和 Python 调度耗时，优先减少每层重复的小算子；避免改变 BF16 舍入顺序。
   对 DeepSeek 还必须扩展同前缀及任务质量集，验证 SDPA 路径的差异，不能只增加吞吐而放任答案重复。
3. HTTP 并发测排队、TTFT 和 P95，不能只看内核计时；当前原生固定微批不是连续批处理。
4. 长上下文、并发 8、30 分钟稳定性、实际 8B 与 Linux/MPS 尚未在本轮完成，按 [验证矩阵](INFERENCE_VALIDATION_PLAN.md) 逐项补齐。
