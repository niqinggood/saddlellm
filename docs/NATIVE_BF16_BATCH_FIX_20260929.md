# BF16 合批分叉修复与合组显存优化

这是 [原生逐步合批执行器](NATIVE_BATCH_DECODE_20260929.md) 的后续迭代。
本轮解决已复现的 28M CUDA BF16 合批数值分叉，并减少创建合组 KV 时的临时显存。
**仍是实验性 Python 执行器，不是默认 HTTP/SSE 服务上线，也不是历史 8B W8A16 固定微批问题已修复。**

后续 P1 已加入 [显式启用的原生 SSE 合批](NATIVE_STREAM_BATCHING_20260929.md)。本文保留 P0 交付时的实验与边界，
不能把下面模型调用吞吐与后续 HTTP 吞吐混用。P0 的相关回归为 862 项通过，记录在
`artifacts/native_batch_bf16_fix_20260929/regression.xml`。

## 1. 原因如何定位

上一轮同一 28M 模型的 B1 与 B4 BF16 解码，第 25 个生成 token 发生首次 argmax 分叉。
即便强制两条路径每步使用相同历史 token，192 个请求步仍有 3 处 argmax 不同。
更换 `sdpa_reference` 未解决，不能只把问题归因于 attention folding。

本轮在第 2、25 个生成 token 跟踪 embedding、Linear、norm、attention、MLP 和 decoder layer 的输入/输出。
首次观察到“相同输入、不同输出”的模块是第 0 层 attention 的 `o_proj`，发生在第 2 个生成 token；
最大绝对差约 `5.96e-8`。同一步 `up_proj` 也出现独立差异，之后逐层累积。
这是 **Linear 的 B1 与 B4 执行形状相关数值差异**；未进一步宣称已锁定 cuBLAS 内部的某个 kernel。

控制实验：只让 Linear 保持单请求形状，其他操作仍合批。
结果原先 3 处 argmax 分叉变为 0；最终原生实现的两步跟踪共 246 条模块记录，输入/输出均相同。
这支持修复方案，但不等于对所有模型、硬件和依赖版本的位级一致性证明。

证据保留在 [本轮目录](../artifacts/native_batch_bf16_fix_20260929)：

- `trace_baseline.json`：原始分叉及逐模块记录。
- `trace_rowwise_all.json`：仅修改 Linear 形状的诊断对照。
- `trace_native_compat.json`：正式代码路径，无运行时 monkey-patch 的复验。

## 2. 怎样修，为什么仍能提速

在 [SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 新增内部 `_rowwise_decode_linear` 和逐次 forward 的执行标记。
[NativeBatchDecodeRunner](../saddlellm/models/NativeBatchDecode.py) 对合格 CUDA BF16 组选择此路径：

1. prefill 仍然逐请求执行，与基线相同。
2. Q/K/V 或已合并 QKV、attention output、gate/up 或已合并 gate-up、down、lm_head 的 Linear 各按 B1 行形状执行。
3. attention、norm、激活、残差和 KV 更新仍处理整批，请求仍共用一次模型 forward。
4. 各请求沿用自己的生成预算、重复惩罚、EOS、stop 和输出状态。

这里有意放弃 BF16 Linear 的完整批量 GEMM 来保留被验证的单请求数值路线。
因此不能说所有算子都由四次变为一次，也不能承诺接近裸合批的峰值。
仍有收益，是因为其余部分减少了逐请求重复调度和算子执行；各部分收益比例尚未做 profiler 归因。

没有全模型转 FP32，没有重新量化，没有修改模型权重、参数对象、磁盘检查点或全局 CUDA matmul 开关。
正式实现不临时替换 module.forward；执行标记沿本次调用传递，不给其他线程留下全局模式。
普通 `model.generate()`、训练及默认 HTTP 路径不自动使用这个内部标记。

当前边界：

- FP32 合格组继续使用原来的批量 Linear。
- CUDA BF16、等上下文、greedy、无 padding、标准稠密 MHA/MQA/GQA、几何缓存可进入兼容路径。
- CPU BF16、FP16、autocast 混合精度、INT8、MoE、MLA/KDA、窗口、AttnRes 和不兼容模块仍回退。
- 未证明任意 checkpoint 都逐 token 一致，仍需对实际模型、输入和环境验收。

## 3. 同时解决初次合组的临时显存问题

B8、输入 128、输出 64 的首轮 BF16 兼容测试，输出一致，但 PyTorch 峰值 allocated 增长约 7.06%，
未通过本轮 5% 门槛。合组构造期间保留了所有层的原始缓存，同时分配所有层的新缓存。

现在采用 **逐层完成 K/V 复制与发布**：一层完整复制成功后，各请求改持有该层的新行视图，再处理下一层。
对于首次从独立缓存合组的场景，可以提前释放已替换层的原始存储，避免同时保留整套新旧历史。
这不是 KV 内存池或分页分配器；旧合组或其他请求仍持有共享存储时，内存不会凭空释放。

安全性：

- 构造期间已持有所有相关 session 锁，还没有推进模型 forward。
- 每层先准备所有完整 K/V 视图对，再发布，不让任何行出现新 K 配旧 V。
- 后续层分配失败时，已发布层和未动层都仍是相同 token 长度、相同内容的有效历史；可关闭或重试请求。
- forward 已开始后若部分层失败，仍按原有策略使整组失效，禁止继续使用半更新 KV。
- 故障注入测试在第二层分配时抛 OOM，逐层核对历史不变，并成功恢复生成。

## 4. 实测：精度、吞吐、显存分别验收

Windows / RTX 4090 Laptop，本地已有 28M 原生检查点，非独占 GPU。
同权重、同 BF16、同输入/输出预算、greedy；EOS/前缀缓存关闭。
计时含 session 建立、全部 prefill/decode、合组复制、采样、结果复制与清理，不含模型加载、HTTP 和 tokenizer。
每轮预热后 7 对 AB/BA；以下均为聚合输出吞吐，不是单用户速度，也不是 8B 成绩。

| 工作负载 | 逐请求 → 兼容合批（输出 token/s） | 配对耗时比 | 输出一致性 / 门槛 |
| --- | --- | --- | --- |
| B4，输入 32/输出 48，第 1 轮 | 137.8 → 278.8 | 2.03× | token 全一致，7/7 更快，开发门槛通过 |
| B4，同条件独立第 2 轮 | 127.7 → 256.9 | 2.02× | token 全一致，7/7 更快，开发门槛通过 |
| B8，输入 128/输出 64，原合组构造 | 112.7 → 301.1 | 2.65× | token 全一致，但显存增长 7.06%，门槛失败 |
| B8，同条件逐层构造，第 1 轮 | 121.3 → 323.1 | 2.67× | token 全一致，7/7 更快，显存门槛也通过 |
| B8，逐层构造独立第 2 轮 | 129.9 → 342.7 | 2.66× | token 全一致，7/7 更快，开发门槛通过 |

B8 峰值 allocated：逐请求 73,214,464 B；原合组 78,383,104 B；逐层合组 74,206,720 B。
因此相对逐请求基线的增幅从约 7.06% 降为约 1.36%。这些不是 reserved、驱动整卡显存或其他进程占用。
不同轮次负载有波动，不能把 301.1 → 323.1 单独当作逐层分配的速度收益。
B4 的两轮属于逐层显存修复之前；保留各报告代码哈希，不能声称全部报告来自同一源码快照。

对应原始报告：`native_bf16_b4_run1.json`、`native_bf16_b4_run2.json`、
`native_bf16_b8_p128.json`、`native_bf16_b8_layerwise_run1.json`、`native_bf16_b8_layerwise_run2.json`。

## 5. 复现和使用

正常入口不需要新参数：继续 `from saddlellm import NativeBatchDecodeRunner`。
模型显式加载为 `device="cuda", dtype="bfloat16"` 后，执行器对合格的解码组自动使用逐行 Linear 兼容路径。
`snapshot().rowwise_linear_forwards` 表示实际走过该路径的模型前向次数；不是每个 Linear 算子的调用次数。
完整 session/关闭示例见 [原生执行器用法](NATIVE_BATCH_DECODE_20260929.md)。

```powershell
python -m pytest -q tests/test_native_batch_decode.py tests/test_native_generation_session.py
python tools/benchmark_native_batch_decode.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --device cuda --dtype bfloat16 --require-speedup --output artifacts/my_bf16_compat.json
python tools/diagnose_native_batch_decode.py --report artifacts/native_batch_decode_20260929/cuda_bf16_b4_run1.json --trace-steps 2 25 --native-linear-path --output artifacts/my_bf16_trace.json
```

诊断工具不加 `--native-linear-path` 时，故意复现不安全的裸批量 Linear；它不是正常运行库策略。
基准工具 `--probe-low-precision` 同样是研究覆盖，会关闭 BF16 保护，不应当作部署开关。
两种工具拒绝覆盖旧报告。逐模块 hook/teacher forcing 用于定位，不作为计时基准。

本轮增加 CUDA BF16 的三种 attention 后端 × 六种结构配置测试，包括 MHA/MQA/GQA、QK Norm、QKV bias、投影合并。
还检查成员进出/不同重复惩罚、autocast 回退、参数对象和全局 matmul 设置不变、CPU 非法使用不推进 KV、合组 OOM 恢复。
CPU-only 环境会跳过 CUDA 用例，不能把跳过当作 GPU 验证通过。

## 6. 没有冒进上线的部分

这次没有重启现有服务、切换 GPU 模型或把执行器接入默认 `serve`。
历史完整 Qwen3-8B W8A16 固定微批的输出差异尚未单独定位，本补丁不能被宣传为已修复它。
真实 Qwen3/DeepSeek 大权重、多提示质量集、不同 CUDA 版本、长时压力和混合长度仍需逐项验收。

下一步才是原生流式服务调度接入：请求何时进入/退出批、慢客户端背压、断连关闭、模型许可和真实 SSE/usage 必须一起验证。
当前补丁为它提供了更可靠的 BF16 执行基础，但不能把模型层结果写成服务层已完成。
