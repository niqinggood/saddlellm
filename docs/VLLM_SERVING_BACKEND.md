# SaddleLLM 可选 vLLM 推理服务

2.37 新增 `saddlellm serve MODEL --backend vllm`。用户继续使用 Saddle 的 CLI、Python factory、
Web 对话页和两个 completion API；GPU 调度由一个异步 vLLM 引擎承担，不再走 HF `model.generate()`。
2.40 的 CLI `serve` 和 Python `serve()` 默认 `backend=auto`，兼容 Linux CUDA 环境可自动选中该引擎。
`--require-high-throughput` 可禁止本地降级，`--dry-run` 先检查路由；显式 local/native 用法保留。
低层 `InferenceServerSettings` 的默认 local 为旧代码兼容保留。
新版一条命令与官方 SDK 使用见 [OpenAI 服务手册](OPENAI_SERVING.md)。

## 最简单的用法

在 Linux + NVIDIA CUDA 的独立环境中安装：

2.40 尚未发布到包仓库时，下面的包名安装应替换为本地 wheel 的 `[serve-vllm]` extra，
或在源码目录安装 `'.[serve-vllm]'`；本轮没有上传 PyPI。

```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install '.[serve-vllm]'
saddlellm serve Qwen/Qwen3-8B --require-high-throughput --dtype bfloat16
```

开发源码未发布时，安装本地项目 `'.[serve-vllm]'` 或本轮 wheel 的 `[serve-vllm]` extra。
首次使用远端模型 ID 需要下载权重；已有目录加 `--local-files-only` 可避免模型目录解析的在线探测。
API 默认监听 `127.0.0.1:8000`，Web 页面与原服务相同。不要用多 uvicorn worker 复制 GPU 模型。

此 extra 固定在 **vLLM >=0.30,<0.31** 的已核对接口范围，安装 vLLM 的兼容 Torch/Transformers。
不要把 `[serve]`、`[posttrain]` 与 `[serve-vllm]` 混装到同一个环境：原 HF 训练/服务约束 Transformers 4，
vLLM 0.30 使用 Transformers 5。Windows/macOS 本地用默认后端；这次没有实现其本地 vLLM 内核。
没有可用 CUDA、版本不匹配或引擎初始化失败会报错，不自动降级为串行 HF。

## 共享 GPU：显式控制预算

```bash
saddlellm serve /models/Qwen3-8B --backend vllm --model-name qwen3-8b \
  --dtype bfloat16 --local-files-only --no-web-ui \
  --max-model-len 4096 --max-input-tokens 3840 \
  --max-new-tokens 256 --default-max-new-tokens 256 \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.47
```

`0.47` 是前次 L20 实验中的特定预算，**不是可以与任意旧服务共存的安全常数**。
必须先看当前可用显存和权重大小。当前业务模型未停止时，不要照抄命令加载另一个 BF16 8B。
引擎比例按整个 GPU 总显存计算，不是按剩余显存计算；引擎会预留 KV 空间。
本后端显式 `cpu_offload_gb=0`，不会为了启动成功暗中把层搬到 CPU。

| 参数 | 实际效果 |
| --- | --- |
| `--backend vllm` | 延迟到 FastAPI startup 创建异步引擎，只有一个权重持有者。 |
| `--gpu-memory-utilization` | vLLM GPU 预算比例，默认 0.9；共享 GPU 需要实算。 |
| `--max-model-len` | 单个请求的输入 + 输出上下文上限；默认输入上限 + 最大输出。 |
| `--max-num-seqs` | 引擎调度的序列上限，默认 16；不是一个同步生成线程池。 |
| `--max-num-batched-tokens` | vLLM 每次调度的 token 预算，默认交给引擎。 |
| `--tensor-parallel-size` | 张量并行 GPU 数，默认 1；当前没有多 GPU 真机验收。 |
| `--vllm-prefix-caching` | 可选自动前缀缓存；默认关闭，测速必须记录缓存条件。 |
| `--enforce-eager` | 关闭 CUDA Graph 捕获，适合排错；通常不是提高速度的选项。 |
| `--max-pending-requests` | HTTP 接入总槽位是 max-num-seqs + 此值，默认额外 64。不是 GPU 内部队列长度的实时读数。 |
| `--max-queued-tokens` | 可选 HTTP 总 token 预留上限：输入 token + 请求输出预算；不是实际 KV 占用测量。 |

`--max-concurrency`、`--batch-size`、原生分桶/微批 token 预算、`--prefix-cache-mb` 属于 local 后端，
在 vLLM 下设置非默认值会明确拒绝，不用它们把异步引擎重新串行化。
`--quantization 4bit/8bit` 是 local 的 bitsandbytes 加载开关，不转成 vLLM 量化参数。
vLLM 支持的预量化 checkpoint 可由模型配置解析；具体格式仍受 vLLM/设备限制，需单独验收。
原生 Saddle checkpoint、独立 LoRA adapter 与 Saddle CPU INT8 包不直接加载；adapter 先合并导出。

## Python import 同样简单

```python
import uvicorn
from saddlellm import InferenceServerSettings, VLLMServingConfig, create_inference_app

if __name__ == "__main__":
    settings = InferenceServerSettings(
        model_path="/models/Qwen3-8B",
        model_name="qwen3-8b",
        backend="vllm",
        dtype="bfloat16",
        max_input_tokens=3840,
        max_new_tokens=256,
        vllm=VLLMServingConfig(
            max_model_len=4096,
            max_num_seqs=8,
            gpu_memory_utilization=0.47,
        ),
    )
    app = create_inference_app(settings)
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

factory 不先加载 HF 权重；进入 ASGI lifespan 才初始化 vLLM。测试/嵌入运行也必须执行 lifespan。
GPU worker 明确采用 `spawn`，不修改 Python 全局 multiprocessing start method。
脚本必须保留 `if __name__ == "__main__"` 保护，避免子进程重复启动父服务。
若外部设置了 `VLLM_WORKER_MULTIPROC_METHOD=fork`，会明确拒绝；不要在初始化 CUDA 后 fork。
默认 `default_max_new_tokens=256`，不能大于 `max_new_tokens`。

## API、错误与取消

`/v1/completions` 和 `/v1/chat/completions` 支持文本、n=1、temperature/top_p、repetition_penalty、
max_tokens、至多四个非空 stop 字符串和 stream。completion 还支持 echo；echo 不计为生成 token。
chat 使用模型 chat template；2.40 默认保留模板的 thinking 设置，`--disable-thinking` 才明确关闭。
chat 同时接受 `max_completion_tokens`、developer 文本角色和纯 text content parts；
未知参数/未实现工具调用等明确拒绝，不猜测缺失的 chat template。
指定 model 时必须与该实例的别名一致。不是所有 OpenAI 功能都已支持：没有工具调用、图像、
logprobs、结构化输出或多候选；请勿依据“兼容”二字推断完整功能。

```json
{
  "model": "qwen3-8b",
  "messages": [{"role": "user", "content": "简单介绍一下KV缓存。"}],
  "temperature": 0,
  "max_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

流式 SSE 输出内容增量、finish_reason、可选 usage chunk、`[DONE]`。
用量来自实际送入引擎的 token ID 与引擎返回的累计生成 token ID，不按中文字数估算。
过长输入或输入 + 输出预算超限，启动生成之前 HTTP 400；不默默截断用户输入。
`max_tokens` 超过服务输出上限时仍按现有 Saddle 约定取服务上限。
接入请求数或总 token 预留满时 HTTP 503 + `Retry-After: 1`，而不是无限挂起。

非流式有显式断开监听，流式由 ASGI 响应监听断开。取消调用 vLLM abort，确认后归还接入槽位；
若 abort 失败，关闭新的接入并使 readiness 失败，不假装 GPU/KV 已释放。
客户端异常取消、生成异常、未返回最终结果或响应尚未开始就断开，均有清理路径。
服务退出调用引擎 shutdown 清理其工作进程/IPC。生成异常返回通用错误，不向客户端回显内部异常/提示词。

`--api-key-env VAR_NAME` 沿用原 bearer 鉴权，密钥不放命令行；有鉴权时两个生成入口与 `/v1/models` 都受保护。
`/livez` 只检查进程存活；`/readyz` 检查引擎和接入状态，不要求队列为空。
`/health` 展示实际后端、版本、精度、上下文、缓存设置和有界接入计数/最近请求指标，不展示提示词或密钥。
其中最近请求的 ttft_ms 是 adapter 收到首个非空累计文本的时间，response_ms 计至请求清理，
都不是客户端 HTTP 延迟测量。被取消请求的 reported_completion_tokens 只计已收到的 token ID，
不声称统计了引擎尚未回传的全部计算量。
公网生产部署还需要 TLS/反向代理、请求体限制、访问控制和业务质量验收。

## 源码逻辑：抓住这一条链

`cli.py` → `InferenceServerSettings(backend="vllm")` → `create_inference_app()` 分派
→ `VLLMServing.create_vllm_inference_app()` → lifespan 创建 `AsyncLLMEngine`
→ HTTP 接入槽位与预算 → tokenizer + `SamplingParams`
→ `engine.generate(TokensPrompt, params, request_id)` 异步迭代
→ 累计输出转内容增量和真实 usage → 完成/取消释放槽位。

没有 `asyncio.Semaphore(max_concurrency=1)` 包住该引擎，也没有 HTTP 到 HTTP 的代理跳转，
Saddle 本身不重写 vLLM 的分页 KV 或连续批处理。
vLLM 0.30 的 AsyncLLMEngine 是 AsyncLLM 的公开别名，
接口核对见 [官方 alias](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/engine/async_llm_engine.py)、
[异步引擎](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py)。
本轮按这些官方接口限定版本，未依赖猜测的旧版关闭/取消方法。

## 验证状态与性能边界

本地新后端 CPU 替身覆盖协议、并行接入、预算、鉴权、用量、取消清理、错误脱敏与生命周期；
与原生/HF 服务、release、benchmark 相关的回归以及额外 generation/cache/WebUI 回归分组执行。
189 + 128 + 12，共 **329 项相关测试通过**；这是相关测试集合，不是重新跑整个项目的全部测试。
新后端自身有 42 项 CPU 测试；新增模块/测试的 Ruff E/F/I、涉及 CLI/原服务的 E/F、
编译、函数参考与版本一致性检查通过。既有不相关区域的导入格式没有批量重排。
其中一个既有合批测试依赖“CPU tokenize 线程一定在 5 ms 内全部抵达”，在并发构建时出现偶发失败；
现在控制请求 cohort 到达时机，仍断言三个请求合为一批及输出顺序，不以放宽正确性断言掩盖失败。
也修复了非流式断开监听的取消 scope 可能吞掉 Task.cancel、留下轮询任务的问题。

### 真实 GPU 接入已验收

Ubuntu 22.04 + NVIDIA L20，vLLM 0.30.0，安装的是新打包 wheel，不依赖源码目录。
用 Qwen3 架构的未训练 fixture：78,835,712 参数、2 层、hidden 256、context 128，
只复用原始 Qwen3 tokenizer，**不是 Qwen3-8B 权重**。设置 BF16、4 个调度序列、
GPU 比例 0.04、eager 模式；启动前检查至少 8 GiB 空闲，未停止任何旧业务服务。

结果：4 个并发非流式请求、1 个 completion SSE、1 个 chat SSE 共 6 个真实 GPU 请求全部成功，
生成计数合计 48 token。最后 health 的 completed=6、in_flight=0、reserved_tokens=0。
测试进程显存一次观察为 2228 MiB（不是峰值）；停止测试后 GPU 使用恢复至原观察的 35632 MiB。
旧 Web、4B、8B 的 HTTP 检查均为 200，旧服务名称/8bit 配置未改，实验 unit 已停止。

真机第一次启动发现 CUDA 子进程 fork 初始化失败；修复为明确 spawn 后，在相同环境重测通过。
这项修复已进入最终 wheel 与回归测试，不要求用户每次额外设置临时环境变量。
当前日志也提示 TokensPrompt 原始输入接口未来会弃用；0.30 测试通过，后续版本需适配 Renderer 再验收，
不能随意扩大版本范围宣称兼容。

交付 wheel：`artifacts/vllm_backend_20260927/wheel-spawn/saddlellm-2.37-py3-none-any.whl`，SHA256：
`1a88f6b72ab00a8ca7ca2b9b64d7e1a20b9ac3013f9fa72f1367c2ed94273d89`。
本地摘要、服务器摘要与源码文件逐字核对通过，安装在独立 `site-spawn/` 覆盖层，未改已有依赖环境。

证据：

- [真实 GPU/HTTP 请求与健康计数](../artifacts/vllm_backend_20260927/evidence/probe-spawn/probe.json)
- [最终 wheel 摘要与安装位置](../artifacts/vllm_backend_20260927/evidence/environment_spawn.json)
- [旧服务不变及收尾检查](../artifacts/vllm_backend_20260927/evidence/business_unchanged.json)
- [首次失败与成功重测日志](../artifacts/vllm_backend_20260927/evidence/logs/)

服务器保留 `/root/saddlellm-vllm-adapter-20260927.kO7wiM` 供复查。
归档 adapter-evidence.tar.gz 本地/服务器摘要匹配：
`d8ce13f571cf1ff71fde9f9031c58ef90188c82c29a2133d4ac98cca0b353ed4`。

### 不把接入测试当作 8B 速度或能力结论

此前 [Qwen3-8B 对照](QWEN3_8B_SERVING_EXPERIMENT.md)中 vLLM 是独立启动的，
不能把其 6.33 倍数字直接当成新 Saddle adapter 已经测出的加速。
本轮 tiny GPU 测试使用 eager，不验证新 adapter 下 CUDA Graph 的收益；
实际新 adapter 的 8B 吞吐、GPU 取消压力、长请求、缓存未命中与多 GPU 仍需另开受控实验。
