# 2.40 一键服务部署与完整 8B HTTP 验收

本轮目标：使用者用 `saddlellm serve MODEL` 或 `from saddlellm import serve` 即可部署，
复用 OpenAI 文本客户端，并能明确选择高吞吐路线。使用手册见 [OpenAI 服务](OPENAI_SERVING.md)。
本页区分接口/部署验收、模型吞吐观测与质量验收，不能把三者混为一谈。

## 1. 完成了什么

- CLI/Python `serve()` 默认 auto：兼容 Linux CUDA + vLLM 0.30 时选异步连续批处理；
  其他环境明确选 local；显式原生 INT8/attention/CPU 选项保留相应路线。
- `--require-high-throughput` 不允许误走普通本地执行；`--dry-run` 不加载权重。
- 官方 OpenAI SDK 1.109.1 通过文本协议检查：models、chat、completion、SSE、真实 usage、
  max_completion_tokens、developer、纯 text parts、错误模型名和不支持参数。
- 默认保留模型模板思考策略，不自动量化/CPU offload、不默认静默截断输入。
- local/native 加有界 HTTP 接入，队列满载 503 + Retry-After；vLLM 保留自己的有界预算、
  取消 abort、readiness 与清理。不是为每个请求复制 GPU 模型。
- 修复 Linux 立即重启的端口预检查：允许 TIME_WAIT，但仍拒绝正在监听的占用端口；
  Windows 不启用会产生端口复用风险的 SO_REUSEADDR。
- 相关回归 618 项通过、73 个依赖/异步取消弃用等警告，0 失败/错误/跳过；不是全仓测试声明。

核心代码为 `runtime/ServingPlan.py`、`InferenceServer.py`、`VLLMServing.py`、
`OpenAIProtocol.py`、`InferenceAdmission.py`，并接入 CLI 与惰性 Python 导出。
没有在本轮改写模型/量化 GPU 算子，也没有把 vLLM 的内部优化说成自研内核。

## 2. 实机条件和操作边界

Ubuntu 22.04、NVIDIA L20 49,140 MiB；既有七个业务服务保持运行。
实验前观察：GPU 已用 35,632 MiB、空闲 12,880 MiB。
主业务 Python 环境为 Torch 2.6.0+cu124、Transformers 4.57.6、Triton 3.2.0；
vLLM 在原有独立环境中为 0.30.0、Torch 2.13.0+cu130。

完整原始 Qwen3-8B：8,190,735,360 参数，五个 safetensors 分片；
目录 `/home/medmodel/models/Qwen3-8B`，原始 revision 为
`b968826d9c46dd6066d109eabc6255188de91218`。
收尾重新对原始权重做 SHA256，与此前保存的官方来源记录比较，不用小 fixture 冒充 8B。

BF16 权重本身约 16.38 GB，加上 KV/运行开销超过当前空闲显存，
所以本轮**没有另启完整 BF16 8B vLLM 引擎，也没有新的 2.40 vLLM 8B 吞吐对比**。
采用显式 W8A16 INT8 的完整 8B 验证自有路线；它是有损量化，不是默认精度或无损效果认证。

安装 wheel 使用新的独立 `site` 覆盖目录，客户端 SDK 放独立 client target，
不升级/覆盖业务环境依赖。临时服务只绑定 loopback 8112，有内存、CPU 和最长运行时间限制。
不暂停原业务服务，不替换生产模型，不开公网端口、不上传 PyPI。

## 3. 完整 8B 的 CLI 与 SDK 检查

真实启动命令省略 backend，以验收自动路由；INT8 是显式用户选择：

```bash
saddlellm serve /home/medmodel/models/Qwen3-8B --quantization int8 \
  --device cuda --host 127.0.0.1 --port 8112 --local-files-only \
  --max-input-tokens 1024 --max-new-tokens 256 --default-max-new-tokens 128 \
  --no-web-ui --model-name qwen3-8b
```

实际为 auto → native；health 显示 `execution_backend=native`、`saddle-w8a16`、
144 个 CUDA INT8 线性层、BF16 激活、`triton_w8a16` 和模型模板原思考默认值。
驻留参数/buffer 9,441,315,840 字节，不是整个进程或峰值 GPU 占用。

官方 SDK 对完整 8B 实测通过：模型发现；developer + text parts 对话；
max_completion_tokens=2 确实约束生成；流式增量和末尾 usage；普通 completion；
错误模型名 404；tools 参数 422；提前关闭流后下一请求成功；最后 HTTP 接入 active=0。
这些是功能检查，不因回答包含文本就宣称业务题集质量通过。

首次安装候选包和包含端口重启修复的最终包分别做了 SDK 检查；失败重启日志保留。

## 4. 原生单请求执行：并发增加，不等于吞吐增加

`tools/benchmark_serving.py` 真实 HTTP SSE completion，loopback，8 请求/轮，
预热 1 轮、计时 2 轮，每个设置 16 个计时请求。
同一公开 raw prompt、temperature=0、top_p=1、64 个输出 token 上限；
实际每个计时响应都是 64 个 token。未应用 chat template，没有隐藏关闭 chat thinking；
前缀快照缓存关闭，batch_size=1，max_concurrency=1。

| 客户端并发 | 实际输出 token/s | P95 响应 | P95 首个可见文本 |
| --- | ---: | ---: | ---: |
| 1 | 32.33 | 2.004 s | 0.038 s |
| 4 | 32.04 | 8.043 s | 6.068 s |
| 8 | 31.76 | 16.054 s | 14.084 s |

这说明有界队列改善的是资源/失败行为，不会让串行生成自动获得连续批处理能力。
TTFT 按客户端首个非空文本统计，含 tokenizer 缓冲，不是 GPU 内部单步 token 时间。
有限的重复提示闭环测量不是到达率压力测试、全业务分布或长期容量认证。

## 5. 原生 8B 固定微批：真实增益与尚未通过的文本等价门槛

另做同协议的非流式 C8 对照：同一模型/量化精度、提示词、64-token 预算、
请求数、预热和计时轮数；两次服务按顺序启动，不让两个实验模型争同一张卡。
唯一执行设置差异为 `--batch-size 8 --batch-token-budget 2048`。
最终重启修复只影响启动前的端口探测，不修改模型执行代码。

| 非流式 C8 设置 | 实际输出 token/s | P95 响应 |
| --- | ---: | ---: |
| 逐请求执行，batch_size=1 | 32.43 | 15.754 s |
| 原生固定微批，batch_size=8 | 252.78 | 2.039 s |

这次聚合吞吐约 7.79×；**返回文本哈希与串行路径不一致**。
没有把不同输出删去或重新挑选 prompt；报告记录这一事实。
尚未完成该差异的数值定位与业务质量验收，因此不称为“质量无损 7.79×”，
不据此开启默认微批，不与其他精度/后端的 8B 数字混比。
同样的字节数或 token 数也不能证明回答质量相同。

该结果是单一重复公开原始补全提示、贪心、非流式的观察；不是对采样聊天、
混合长度、长上下文、流式或在线连续批处理的速度承诺。
微批保持显式选项；生产应先用业务题集验收该执行配置。

## 6. vLLM 自动路由与真实 GPU 接口检查的边界

真实独立 vLLM 环境的 dry-run 选中 vllm，continuous_batching=true，
无自动量化/启动降级；这是依赖/设备路线检查，不是显存或完整 8B 启动通过。

为验证新版协议与真实异步引擎连接，复用 78,835,712 参数、两层、hidden=256、
context=128 的**未训练 Qwen3 fixture**，只复用原始 tokenizer，不是原始 8B 权重。
配置 BF16、4 个调度序列、GPU 比例 0.04、eager、无前缀缓存。
验收包括四个并发 GPU completion、chat/completion SSE 的 usage/[DONE]，
以及官方 SDK 的完整文本协议检查；不能由此推断 8B 吞吐或回答能力。

这次后台 unit 起初未带虚拟环境 bin 的 PATH，FlashInfer 因找不到已安装的 ninja 而失败。
仅补正临时 unit 的 PATH 后重测，不升级业务包，不隐式换 HF 路线，保留首次失败日志。
使用手册中的虚拟环境 activate 已处理通常的 CLI PATH；后台管理器需要显式传正确环境。

## 7. 证据、收尾与安装包

最终相关测试：[618 项 JUnit](../artifacts/simple_serving_20260927/related_tests_restartfix.xml)。
工具 `smoke_openai_service.py`、`capture_serving_evidence.py`、`verify_simple_serving_evidence.py`
用于独立保存协议、环境与代码来源证据，不修改模型或业务服务。

| 记录 | 内容 |
| --- | --- |
| [完整 8B 最终 SDK](../artifacts/simple_serving_20260927/server/sdk_smoke_8b_restartfix.json) | 实际请求/响应、预算、流式、断开后的继续执行。 |
| [C1 SSE](../artifacts/simple_serving_20260927/server/http_c1.json)、[C4 SSE](../artifacts/simple_serving_20260927/server/http_c4.json)、[C8 SSE](../artifacts/simple_serving_20260927/server/http_c8.json) | 实际 usage、输出摘要、延迟和首文本时间。 |
| [C8 非流式逐请求](../artifacts/simple_serving_20260927/server/http_nonstream_c8_unbatched.json)、[固定微批](../artifacts/simple_serving_20260927/server/http_nonstream_c8_batch8.json) | 保留不同文本摘要，不用它作无损提速证明。 |
| [vLLM GPU fixture](../artifacts/simple_serving_20260927/server/vllm_fixture_probe/probe.json)、[fixture SDK](../artifacts/simple_serving_20260927/server/sdk_smoke_vllm_fixture.json) | 只证明真实异步接口，非 8B 成绩。 |
| [端口立即重启](../artifacts/simple_serving_20260927/server/port_restart_probe.json)、[在用端口拒绝](../artifacts/simple_serving_20260927/server/port_busy_probe.json) | 真实 Linux TCP 检查。 |
| [最终服务/资源状态](../artifacts/simple_serving_20260927/server/server_final_state.json) | 既有业务存活、临时作业停止、权重与安装哈希。 |
| [一致性核验](../artifacts/simple_serving_20260927/evidence_verification.json) | 运行时来源、测试、原始权重、API 验收和不同输出的明确标记。 |

测速编排初版存在一个日志文件名的 systemd 变量展开问题：`http_c.json` 被依次复用。
每一轮完整 JSON 仍保存在只追加的 `logs/http-bench-v2.log`；核验脚本逐个解码恢复 C1/C4/C8，
不重算或编造数字。初次压测客户端未配置自己的 httpx target 的错误日志也保留。

收尾只停止自有临时 unit，七个原业务服务仍 active，原 Web/两个模型的 HTTP 检查 200，
GPU 恢复至 35,632 MiB 已用、12,880 MiB 空闲。这是收尾观察，不是峰值采样。

交付 [saddlellm-2.40-py3-none-any.whl](../artifacts/simple_serving_20260927/wheel_verified/saddlellm-2.40-py3-none-any.whl)，
SHA256：`2532b9c35bab6f3a99b2ff9169d98f0c281e806c635e90f7c7015c3242de6e09`。
186 个包内 Python 文件与当前源码和服务器最终安装逐字节一致。
候选吞吐基线与最终包仅 ServingPlan.py 的端口预检查有差异，其他运行时文件相同。
最终包实际执行完整 8B 微批/SDK 和 vLLM fixture 检查；旧候选包不作为交付包。
wheel 合约、版本一致性、编译、函数参考和新运行时/测试 Ruff 检查通过。
全仓 Ruff 未作为验收：既有 `_exports.py` 的 RolloutSample 重复键不在本轮服务改动范围，未顺带改动其旧语义。

下一步应在独占/足够空闲的 GPU 上测未量化 8B 的新版 vLLM HTTP 路线，
固定精度、业务提示分布、thinking、采样、上下文和缓存，联合验证吞吐、P95/TTFT、失败率与业务答案。
这比只提高并发参数、缩短输出或复制 GPU 进程更能确定实际部署容量。
