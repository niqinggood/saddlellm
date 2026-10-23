# 一条命令部署 SaddleLLM：OpenAI 兼容服务与高吞吐路线

本指南对应 **2.40**。目标是让使用者只记住 `saddlellm serve MODEL`，
由框架负责后端选择、模型生命周期、文本协议、流式输出、用量统计和有界接入。

首次使用见 [部署教程](quickstart/SERVE.md)。只检查推理依赖用 `saddlellm doctor --profile serve`（原生为 `native`，vLLM 为 `vllm`），不用安装 TRL/datasets 来通过推理检查。
`--no-chat-history` 或 Python `chat_history=False` 禁止内置历史的读取与写入，不删除旧数据库；`/health.chat_history_enabled` 显示实际状态。
默认历史是同一模型的共享工作区，不提供用户隔离；多人使用时应关闭历史或由业务层独立隔离。
“兼容 OpenAI”指下文的文本 API 子集，不是 OpenAI 的全部产品/API。
推理优化改善运行效率，不会自动把一个模型训练得更聪明。

## 1. 先安装，再用一条命令启动

### Linux + NVIDIA GPU：高吞吐部署

在本项目源码目录创建独立环境。安装需要网络及足够磁盘空间；CUDA 驱动必须与依赖兼容。

```bash
python3 -m venv .venv-serving
source .venv-serving/bin/activate
python -m pip install '.[serve-vllm]'
saddlellm serve /models/Qwen3-8B --require-high-throughput
```

模型目录需要包含完整配置、权重与 tokenizer。若尚未下载，也可把目录替换成
`Qwen/Qwen3-8B`，首次启动会下载权重；已有完整目录可以加 `--local-files-only`。

**2.40 本轮没有上传 PyPI。** 若从 wheel 安装，将上面的安装命令替换为
`python -m pip install '/path/saddlellm-2.40-py3-none-any.whl[serve-vllm]'`。
不能安装包仓库的旧版本后就假定拥有新版命令行为。

`serve-vllm` 使用已核对的 vLLM `>=0.30,<0.31`。请与 `serve`/`posttrain` 分环境：
本地 HF 服务约束 Transformers 4，vLLM 0.30 使用 Transformers 5。
不要在已有业务/训练环境里为试验强制升级这些包。

用 systemd 等后台管理器启动时，不要只填写虚拟环境 Python 的绝对路径而忽略 PATH；
也需让该服务的 PATH 包含此虚拟环境的 bin 目录，以找到已安装的 ninja 等底层工具。
本轮实机曾因 PATH 未包含 ninja 而在 FlashInfer 初始化失败，补正临时服务的 PATH 后重测。
这与要求使用者自己编写 C++ 算子不同；vLLM/FlashInfer 底层的 JIT 依赖仍必须正确安装和可发现。

### Windows、CPU 或本地 HF/原生服务

在源码目录、已有 Python 虚拟环境中安装：

```powershell
python -m pip install ".[serve]"
saddlellm serve "D:\models\Qwen3-8B"
```

没有兼容的 Linux/CUDA vLLM 时，`auto` 会选择 `local`，并提示该路径没有 vLLM 连续批处理。
普通 HF 检查点走 HF，Saddle 原生检查点走原生模型；显式设备不可用时报错，不悄悄转 CPU。
CPU/Apple GPU 的内存、模型支持与实际速度需单独核对，不能期待同等 NVIDIA GPU 吞吐。

安装依赖是一次性准备。之后启动仍只需 `saddlellm serve 模型路径`。
框架不会在启动时自动安装大型包、量化模型、把层搬到 CPU 或停止其他服务。

## 2. 默认发生什么

默认监听 `127.0.0.1:8000`，启动成功后提供：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:8000/` | 内置对话网页，不额外加载一份模型；`--no-web-ui` 可关闭。 |
| `/v1/models` | 查询当前实例的模型别名。 |
| `/v1/chat/completions` | 文本对话，支持非流式与 SSE。 |
| `/v1/completions` | 原始文本补全，支持非流式与 SSE。 |
| `/livez` | 进程存活，不执行模型前向。 |
| `/readyz` | 模型/引擎就绪；启动或关闭阶段为 503。 |
| `/health` | 实际后端、精度、思考策略、接口范围及队列/引擎状态。 |

启动提示“选择后端”不等于已就绪，应以 `/readyz` 返回 200 为准。
按 `Ctrl+C` 正常退出；服务生命周期负责清理自有引擎资源。

默认 `--backend auto` 的规则：

| 情况 | 选择 |
| --- | --- |
| 普通完整 HF 模型，Linux、CUDA 可用、已安装兼容 vLLM，设备为 auto/cuda | `vllm`：异步连续批处理。 |
| 显式 `--quantization int8`、原生 attention 开关或 CPU 预排布 | `native`：Saddle 自有路线；仍检查设备和模型资格。 |
| Saddle 检查点、独立 adapter、Saddle CPU INT8 包，或显式本地微批/量化/前缀快照选项 | `local`：兼容相应加载方式。 |
| 其他平台/设备或没有兼容 vLLM | `local`，明确提示能力差异。 |

互相冲突的本地与 vLLM 设置会报错；不任意猜测一条路线。
`--backend local/native/vllm` 始终可以显式指定。
选中的引擎初始化失败、显存不足或模型不受支持时，**直接失败，不偷偷换慢后端或改变精度**。

部署前先看路线，不下载/加载权重、不启动 HTTP：

```bash
saddlellm serve /models/Qwen3-8B --require-high-throughput --dry-run
```

这会输出 `selected`、`continuous_batching`、环境探测、思考/输入策略等 JSON。
它不是完整资源或模型架构验收，也不保证随后一定能启动；显式后端还会在实际 startup 校验依赖。
`--require-high-throughput` 要求选中连续批处理 vLLM，否则立即报错；
它保证执行路线，不承诺某个固定 token/s 数字或业务质量。

## 3. 用官方 OpenAI SDK 调用

客户端可在另一台电脑上运行。安装 `python -m pip install openai`，然后：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused-local-key",  # 仅适用于未开启鉴权的本机服务
)
model = client.models.list().data[0].id
result = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "用三句话解释 KV 缓存。"}],
    max_completion_tokens=256,
    temperature=0.2,
)
print(result.choices[0].message.content)
print(result.usage)
```

`base_url` 带 `/v1`，不是 `/v1/chat/completions`。
默认模型别名来自服务设置；最好服务启动时加 `--model-name qwen3-8b`，
调用方就能固定使用 `model="qwen3-8b"`。错误别名返回 404，而不是悄悄调用其他模型。

流式调用：

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "解释一下连续批处理。"}],
    max_completion_tokens=256,
    stream=True,
    stream_options={"include_usage": True},
)
try:
    for chunk in stream:
        for choice in chunk.choices:
            print(choice.delta.content or "", end="", flush=True)
        if chunk.usage is not None:
            print("\nusage:", chunk.usage)
finally:
    stream.close()
```

使用 `include_usage` 时，末尾有 `choices=[]` 的用量块；不要无条件索引 `choices[0]`。
SSE 最后发送 `[DONE]`。用量来自 tokenizer/实际生成 ID，不按中文字数估算。
客户端提前关闭流时触发取消清理；本地执行必须等已运行的 GPU 操作结束后释放许可，
vLLM 路径使用引擎 abort，不假装已经执行的计算瞬间消失。

也可以用 `client.completions.create(model=model, prompt="Hello", max_tokens=32)`。
它是原始文本补全，不自动应用对话模板，不能把其质量/速度直接等同于 chat。

## 4. Python 一次 import 启动服务

```python
from saddlellm import serve

if __name__ == "__main__":
    serve("/models/Qwen3-8B", model_name="qwen3-8b", require_high_throughput=True)
```

`serve()` 与 CLI 一样默认 `backend="auto"`，是阻塞式启动，默认 host/port 也相同。
需要嵌入已有 ASGI 应用时可使用 `InferenceServerSettings` + `create_inference_app()`；
这些低层 settings 为旧代码兼容保留 `backend="local"`，要自动选择需显式填 `backend="auto"`。
不要在模块顶层启动 CUDA 服务，保留 main guard，避免 spawn 子进程重复启动。
一个进程管理一份模型，不用多个 Uvicorn worker 复制同一 GPU 模型。

## 5. 参数怎么选，起什么作用

常用参数：

| 参数 | 默认/效果 |
| --- | --- |
| `--model-name NAME` | 固定对外模型别名；调用方的 model 应与之匹配。 |
| `--host` / `--port` | 默认本机 127.0.0.1:8000；地址/端口冲突尽量在权重分配前报告。 |
| `--backend` | auto/local/native/vllm；自动与显式路线见前文。 |
| `--require-high-throughput` | 不接受普通本地路线；不是性能 SLA。 |
| `--dry-run` | 只生成部署计划，不加载模型/启动 HTTP。 |
| `--device` | auto/cpu/cuda/mps；本地 auto 优先 CUDA，其次 MPS，再 CPU。vLLM 首版限定 Linux CUDA。 |
| `--dtype` | auto/float32/float16/bfloat16；auto 由执行后端解析，不能把字符串 auto 当作实际精度。vLLM 看 health.dtype，原生 INT8 看 health.native_int8.compute_dtype，其他本地模型检查加载设置/日志。完全 FP32 需求应显式指定；CPU/MPS 本地默认 FP32。 |
| `--max-input-tokens` | 默认 4096，包含 chat template 的输入 token；过长默认 400。 |
| `--default-max-new-tokens` | 客户端未指定输出预算时默认 256。 |
| `--max-new-tokens` | 服务端输出上限默认 512；客户端正值超限按既有约定取此上限。 |
| `--enable-thinking` / `--disable-thinking` | 明确改变支持该开关的模型模板；不写时保留模型模板原默认。 |
| `--truncate-input` | 仅 local/native 显式启用旧截断行为，可能丢失用户信息；默认关闭。 |
| `--local-files-only` | 使用已有完整模型目录，不隐式联网补齐缺失文件。 |
| `--api-key-env VAR` | 从环境变量读取 bearer 密钥；传变量名，不传密钥值。 |
| `--no-web-ui` | 只保留 API/健康入口；不改变模型执行性能。 |

Qwen3 等模型默认可能输出思考过程。默认 256/最大 512 个新 token 不一定够完成较长推理；
业务需要时应显式调大输出预算，同时确保输入 + 输出符合模型上下文与显存预算。
`--disable-thinking` 可减小回答延迟，但它是用户选择的质量/行为权衡，不能作为隐藏测速开关。
单纯提高 temperature 也不会提高模型能力。

高吞吐 vLLM 参数：

| 参数 | 默认/效果 |
| --- | --- |
| `--max-num-seqs` | 默认 16，引擎同时调度的序列上限，不是同步 generate 线程数。 |
| `--max-num-batched-tokens` | 每次引擎调度的 token 预算，默认交给引擎。影响 prefill/解码的资源分配。 |
| `--gpu-memory-utilization` | 默认 0.9，按整张 GPU 总显存的比例计算，不是剩余显存比例。共享卡必须先实算。 |
| `--max-model-len` | 总上下文上限；默认输入上限 + 最大输出。不是只限制输出。 |
| `--tensor-parallel-size` | 默认 1，张量并行 GPU 数；本轮未做多 GPU 真机验收。 |
| `--vllm-prefix-caching` | 显式开启前缀缓存，默认关闭；测重复提示收益必须标注。 |
| `--enforce-eager` | 调试时关闭 CUDA Graph 捕获，通常不是加速开关。 |
| `--max-pending-requests` | 默认额外 64 个 HTTP 槽位，总接入上限为 max-num-seqs + 此值。 |
| `--max-queued-tokens` | 可选接入 token 预留总上限，按输入 + 请求输出预算计；不是测量实际 KV 字节。 |

一个可调整的专用 GPU 配置示例（不是通用安全内存常数）：

```bash
saddlellm serve /models/Qwen3-8B --model-name qwen3-8b --require-high-throughput \
  --local-files-only --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --max-model-len 4096 --max-input-tokens 3072 \
  --max-new-tokens 1024 --default-max-new-tokens 512 --no-web-ui
```

不要照抄到显存已被其他模型占满的卡上。完整原始 Qwen3-8B 约 8.19B 参数，
BF16 权重约 16.38 GB，尚未计 KV、CUDA Graph、激活和框架开销。
显存不足应减少并发/上下文、选择更合适的设备或显式选择已验收量化格式，
不是无限增加队列或指望框架偷偷卸载业务模型。Saddle vLLM 配置显式关闭 CPU 层 offload。

本地/原生参数与 vLLM 不是一套机制：

- `--max-concurrency` 默认 1；原生流式服务保持请求许可加默认线程池的已有策略。增加并发会让更多 KV 同时驻留，仍需实测显存、首字和尾延迟，不等于连续批处理。新增有界线程池和按 token 轮转候选没有通过稳定性能验收，只供内部实验，不会自动启用。实现和实测见 [原生服务调度迭代](NATIVE_SERVING_ITERATION_20260928.md)。
- `--batch-size`、`--batch-token-budget`、`--batch-length-bucket-size` 是原生非流式、贪心请求微批选项，不覆盖所有 HF/采样/流式场景。
- `--max-pending-requests` 也限制 local/native HTTP 接入，总槽位为 max-concurrency + 此值；超载 503 + Retry-After，不无界挂起。它是背压，不是 GPU 动态调度。
- `--prefix-cache-mb`、`--prefix-cache-partial` 是原生 KV 快照能力，不是分页 KV；量化 `int8` 是实验性自有 W8A16 路线。
- `--quantization 4bit/8bit` 属于本地 bitsandbytes 路线，需要相应 CUDA 依赖；不自动转换为 vLLM 量化参数。

本地详细边界见 [原生优化](NATIVE_INFERENCE_ENGINE.md)、
[完整 8B W8A16](NATIVE_INT8_8B.md)；高吞吐引擎生命周期见 [vLLM 服务后端](VLLM_SERVING_BACKEND.md)。

## 6. 接口范围和错误：不忽略不支持的功能

支持文本 chat/completion、单候选 n=1、temperature/top_p、stop、repetition_penalty、SSE 与真实 usage。
chat 支持 `max_tokens` 和 `max_completion_tokens`；二者同时填写且不同会拒绝。
`developer` 文本指令映射为模板的 system 指令；content 支持字符串及纯 text parts。
completion 使用 `max_tokens`，也支持 echo；回显 prompt 不计作生成 token。

暂不支持工具/函数调用、多模态输入、Responses API、结构化输出、logprobs、seed 或多个候选。
未知参数明确拒绝，不返回“成功”后忽略这些要求。没有声明 reasoning_content/tool_call 的独立协议。
支持哪些字段可看 `/health.api_capabilities`。

错误使用 `{"error":{"message":...,"type":...,"param":...,"code":...}}` 格式：

- 401：缺少/错误 bearer 密钥；404：模型别名错误。
- 400：输入/上下文超限等生成前校验失败；默认不截断，流式也在 SSE headers 前校验。
- 422：请求结构错误或不支持的参数。
- 503 + `Retry-After: 1`：接入满载；客户端需限制重试次数和并发，不能无界重试。
- 500：受控的通用生成失败；不向客户端回显提示词、密钥或内部异常。

官方 SDK 默认可能重试部分错误；压测应显式关闭重试，避免把重试等待混入吞吐统计。
浏览器 UI 共用这个文本接口，不能因为保留 UI 就推断所有 OpenAI 客户端功能已经支持。

## 7. 公网部署与质量验收

要被其他机器访问，需要显式绑定 `--host 0.0.0.0`，但请先配置鉴权、
网络访问控制及 TLS/反向代理。不要把无鉴权测试端口直接开放公网。

在进程环境/受保护的服务管理配置中设置 `SADDLE_API_KEY`，然后运行：

```bash
saddlellm serve /models/Qwen3-8B --model-name qwen3-8b \
  --require-high-throughput --host 0.0.0.0 --api-key-env SADDLE_API_KEY
```

调用方 SDK 的 api_key 改为实际密钥；密钥不放命令行、不提交仓库、不写测速报告。
`/v1/models` 和生成 API 都受鉴权保护；健康入口不显示密钥，但也应按网络权限限制访问。
当前内置服务没有替代网关的请求体限制、跨实例限流、TLS 或多租户隔离。

`serve` 不重新训练或自动执行评估。提高效果需要独立题集/业务指标：
`evaluate` → `compare` → `export --require-gate` → `serve`。
SSE 正确、返回 200 或量化模型成功启动，都不等于业务答案质量通过。

## 8. 后面专业实现的代码主线

```text
CLI serve / Python serve()
  → InferenceServerSettings 校验
  → ServingPlan.resolve_serving_settings 选择并记录路线
  → 地址/端口预检查 → create_inference_app
      ├─ local/native：加载一份模型 → LocalGenerationEngine
      │    HTTP 有界接入 → tokenizer/输入预算 → 请求许可或原生微批
      │    → native step / HF generate → 文本、真实 usage、流式清理
      └─ vllm：ASGI startup 创建一个异步引擎
           有界接入与 token 预留 → tokenizer + SamplingParams
           → engine.generate 异步调度 → 增量文本/真实 usage
           → 完成/断连 abort → 回收接入 → shutdown 清理 worker
```

关键文件：`cli.py`、`runtime/ServingPlan.py`、`runtime/InferenceServer.py`、
`runtime/VLLMServing.py`、`runtime/OpenAIProtocol.py`、`runtime/InferenceAdmission.py`。
`_exports.py` 给 `from saddlellm import serve` 提供惰性导出，import 本身不加载 Torch/vLLM。

Saddle 自己负责统一入口、API 协议、质量/行为保护、接入背压及后端生命周期；
vLLM 负责成熟的连续批处理和 GPU/KV 调度。不是再次代理一个外部 HTTP 服务，
也不把 vLLM 的分页 KV/内核说成 Saddle 自己实现。
本地原生模型则使用 Saddle 的前向、KV、采样/生成循环和可选 Triton W8A16 算子。
高吞吐主线不再给每个请求串行调用 HF `generate()`，但普通本地路线仍有它的并发限制。

## 9. 如何验证“好用、效果和吞吐高”

local/native 非流式接口已补上客户端断连取消：排队请求退出、原生 token 步间停止、HF stopping criteria 停止，
并在模型线程退出后释放模型并发许可。无需新参数。真实 TCP 验证、适用边界与未通过的提速候选见
[2026-09-29 推理迭代记录](INFERENCE_LIFECYCLE_20260929.md)。这项改进减少废弃计算，不代表普通请求吞吐倍增。

先用官方 SDK 做功能验收：

```bash
python tools/smoke_openai_service.py --base-url http://127.0.0.1:8000 \
  --output artifacts/my-sdk-smoke.json
```

脚本只访问 loopback，不启动/停止服务，拒绝覆盖已有报告。
验证真实模型别名、chat/text parts/developer、输出预算、流式 usage、completion、
错误类型和提前断连后的继续服务；认证测试可加 `--api-key-env`。
这不是质量或吞吐对比。客户端需单独安装 OpenAI SDK。

之后在固定环境测真实 HTTP 吞吐，并发 1/4/8/16 分别记录：

```bash
python tools/benchmark_serving.py --target saddle=http://127.0.0.1:8000/v1 \
  --model qwen3-8b --requests 8 --concurrency 8 --max-tokens 64 \
  --warmup 1 --repeats 2 --stream --cache-policy 'prefix cache disabled' \
  --output artifacts/my-http-c8.json
```

不要覆盖以前的测速文件。该脚本是原始 completion、闭环并发客户端，按实际 usage 统计输出 token/s，
流式还统计首个可见文本的 TTFT/P95；不是打开网页凭感觉计时，也不是直接 forward 的微基准。
必须保持模型/精度、硬件、输入、输出预算、EOS/采样、thinking 与缓存条件一致；
成功率、错误和尾延迟同样重要。工具本身不证明远端权重/硬件，需配合环境及模型来源记录。
增加并发后吞吐不涨而尾延迟增加，说明单模型执行路线仍是瓶颈，不能拿更大队列当作优化。

2.40 的相关回归 **618 项通过**，包括官方 SDK 1.109.1 的本地/异步引擎替身协议检查
及端口重启预检查；实机发现的 Linux TIME_WAIT 误判已修复，立即重启的真实 TCP 检查通过。
这些替身测试不是完整 8B 性能证据；本轮服务实机记录与历史 2.39 模型调用微基准分开保存。
详见 [2.40 实机部署验收](SIMPLE_SERVING_VALIDATION_20260927.md)。

接口设计按 [OpenAI Chat Completions 参数](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
与 [官方流式说明](https://developers.openai.com/api/docs/guides/streaming-responses)核对，
支持 SDK 的字段和明确拒绝的字段均按本指南范围说明；不是全量 API 兼容认证。
