# 原生 CUDA 解码小算子融合

本轮为 Saddle 原生推理增加显式开关 `--native-fused-kernels`，融合单 token 解码的 RMSNorm 后半段与 Q/K RoPE，并缓存已编译的 Triton 调用入口。默认关闭，权重、精度、attention、KV 布局及 BF16 合批的逐行 Linear 保护均保持原有设置。

## 实际接口收益

同一台 L20、同一份 Qwen3-4B 原始权重、BF16、Torch 2.6.0+cu124、Transformers 4.57.6、Triton 3.2.0。两个临时本机 SSE 接口共用同一模型，原生流式合批均开启，仅在完整请求组排空后切换融合开关。每组预热 2 轮、随机次序 7 对；实际输出每请求 48 token，前缀缓存关闭。

| 工作负载 | 融合关闭 → 开启，输出 token/s | 配对耗时比中位数 | 首个可见文本 P95，ms |
|---|---:|---:|---:|
| 单请求 | 48.85 → 56.45 | 1.153× | 48.57 → 45.89 |
| 4 路同长度 | 69.57 → 75.35 | 1.084× | 156.48 → 154.47 |
| 4 路混合长度 | 53.75 → 62.14 | 1.129× | 241.80 → 227.38 |
| 4 路同长度独立复测 | 69.42 → 75.57 | 1.090× | 159.08 → 153.69 |

token/s 由全部实测输出除以总耗时；配对比为逐轮耗时比的中位数，两个统计量不应混算。每组均 7/7 对更快，文本、usage、finish reason 一致，P95 首字与 allocated 峰值通过不恶化超过 5% 的检查，收尾活跃请求和 KV 合组引用为零。服务测试是经过聊天模板处理的 `/v1/completions` SSE；不是重新测过所有聊天 API 行为。

机器有原有业务负载，这些属于开发测试，不是独占 GPU 容量验收。混合长度仍由现有等上下文合批策略处理，本轮没有实现任意长度的统一批量 attention，也没有更改 BF16 Linear 的数值保护。

## 为什么保留一部分 eager 运算

直接用 Triton 重写整个 RMSNorm 的实验出现少量舍入差异，因此未采用。正式实现继续调用 PyTorch 的 FP32 方差归约，仅融合后续加 epsilon、rsqrt、乘法和类型转换。Qwen3 在乘权重前转回激活精度的边界保持不变。

RoPE 保留两个乘积各自转回 BF16/FP16 的舍入，再执行加法；关闭乘加融合。完整 GEMM、激活函数及 attention kernel 均未替换。

最初普通 Triton 调度版本虽然输出一致，但完整模型慢约 8%–10%，同样未采用。正式实现缓存编译后的调用入口，按设备、dtype、形状、步长和指针对齐分开，缓存有界且不持有请求张量。Triton 3.2 与 3.3 的调用参数约定差异由实际编译签名处理。

## 使用

在有兼容 Triton 的 CUDA 环境中，给已有原生服务命令增加一个参数即可：

```bash
saddlellm serve /home/medmodel/models/Qwen3-4B --backend native --device cuda --dtype bfloat16 --local-files-only --native-fused-kernels
```

如需本次测试的 4 路并发策略，再加：

```text
--native-stream-batching --max-concurrency 4 --batch-size 4 --batch-wait-ms 5
```

Python 模型入口：

```python
from saddlellm import optimize_native_inference

model.eval()  # 已加载到 CUDA 的原生模型
report = optimize_native_inference(model, pack_qkv=False, fused_decode_kernels=True)
```

服务加载时预热常见 batch 1..8 范围内的形状，编译时间不计入表中稳态吞吐；Python 入口第一次调用会编译。更大或特殊形状可能首次使用时编译。`/health` 的 `native_projection_packing` 中记录 `fused_decode_norms` 与 `fused_decode_rotary` 配置数量。

训练、有梯度、autocast、CPU、非当前 CUDA 设备、prefill 及不支持的 dtype/布局走原有算子。缺少 Triton 时，显式启用会报错，不会显示已加速。该服务选项不支持量化权重；它是加载选项，注入模型应在创建服务前自行优化。不要在请求执行过程中切换开关。

没有对所有模型、GPU 或依赖版本作逐位一致保证。迁移其他检查点后仍需验证输出。后续显著缩小与成熟服务引擎的差距，需要 CUDA Graph、混合长度调度及批量 Linear 的独立精度验收。

## 证据

完整记录、原始失败实验、最终 wheel、校验脚本与摘要见 [本轮目录](../artifacts/native_acceleration_20260929/)。本地相关回归 202 项通过，Linux/L20 专项回归 42 项通过；真实 L20 的算子、模型与接口结果以该目录 `verified_summary.json` 和 `REPORT.md` 为准。新包只在隔离测试目录运行，未替换或重启原有业务服务。
