# 跨平台服务部署与推理优化

2.40 的 CLI/Python `serve()` 默认自动路由；Linux CUDA 且 vLLM 版本兼容时选中连续批处理，
其他环境明确提示 local。`--require-high-throughput` 可要求这条路线，否则启动前失败。
安装、OpenAI SDK 与路由规则见 [一条命令部署手册](OPENAI_SERVING.md)。

SaddleLLM 的 Python/原生 PyTorch 推理入口按跨平台组织；**接口适配不等于所有设备已验收**。
本轮在 Windows 实际测试 CPU、NVIDIA CUDA、流式服务、取消回收和真实 loopback HTTP。
新增三平台 CPU CI 和 Linux CPU 容器构建检查，但本轮未在远程 CI、macOS、Linux GPU 上执行。

后续实机补验：[Qwen3-8B 服务实验](QWEN3_8B_SERVING_EXPERIMENT.md)已经在 Ubuntu 22.04、
NVIDIA L20 上部署本次新 wheel，并实际测量 BF16 HF Qwen3 的非流式与流式请求。
这补充了一个 Linux CUDA/HF 模型用例，不替代原生模型、Docker、Linux CPU 或 Mac 真机验收。

## 平台边界

| 环境 | 服务入口 | 本轮验证状态与限制 |
| --- | --- | --- |
| Windows + CPU | `--device cpu` | 本机实际测试；无需 WSL/C++ 编译。 |
| Windows + NVIDIA | `--device cuda` | 本机实际测试 FP32/FP16/BF16 原生推理；需要兼容的 PyTorch 和驱动。 |
| Linux + CPU | 同样的 CLI/import，或本文 Dockerfile | 加入 CPU CI/容器入口，远程结果尚未验证；不宣称镜像已构建成功。 |
| Linux + NVIDIA | CUDA CLI，或 CUDA wheel 版容器 | 后续 L20 实机已测 Qwen3-8B BF16 CLI/HTTP/流式；原生模型与 GPU 容器仍待真机验收。 |
| macOS + CPU | `--device cpu` | 加入 CPU CI；本轮没有 Mac 真机。 |
| macOS + Apple GPU | 新增 `--device mps` | 已测试设备选择与加载传参替身；**没有执行 Apple GPU 内核测试**。默认 FP32，具体模型/算子仍需真机检查。 |

`--device auto` 的选择顺序是 CUDA → 可用的 MPS → CPU。
显式指定不可用的 CUDA/MPS 会报错，不会悄悄换成 CPU。MPS 遇到不支持的算子时，
先使用 `--device cpu` 验证流程；不能把 CUDA 的速度、量化或融合内核能力照搬到 Apple GPU。
本轮 MPS 路由限定文本 `serve` 的模型加载入口，不代表所有训练/embedding 命令都已支持 MPS。

高级后端分开看：Megatron/SGLang/vLLM 本地后端的部署要求不能由 SaddleLLM 消除。
2.37 新增 `serve --backend vllm`，首版限定 Linux CUDA 与 vLLM 0.30，
使用独立 `serve-vllm` extra；2.40 的 auto 在 Windows/macOS/CPU/MPS 仍选择 local；
见 [vLLM 服务接入指南](VLLM_SERVING_BACKEND.md)。
Windows/macOS 客户端可以走已有 HTTP rollout 后端调用远端服务；这不等于能本地运行其所有内核。
2.38 新增 [Qwen3 原生后端](NATIVE_QWEN3_BACKEND.md)：`serve --backend native`，
使用现有 PyTorch 与 safetensors，不要求安装 vLLM。本机已做 tiny Qwen3 CPU/CUDA 真计算检查；
不把这组 Windows 测试当成 macOS/MPS、完整 8B 或所有设备的认证。
本轮没有把这些重依赖纳入默认安装，也没有验证完整异步在线 RL 或多 GPU 生产引擎。
动态 INT8 打包模型仍限定 CPU；运行时 bitsandbytes 4/8 bit 路径仍要求 CUDA，
并受对应安装包/模型兼容性约束，不是全平台量化保证。

## 本机部署：仍是一条命令

先安装服务可选依赖，在仓库内可用 `python -m pip install '.[serve]'`；
已发布安装包可用 `python -m pip install 'saddlellm[serve]'`。
Python 要求见 pyproject；下面的容器/CI 使用 Python 3.11。

```powershell
saddlellm serve MODEL --local-files-only
```

默认只监听本机 `127.0.0.1:8000`。Windows/Linux CPU 指定 `--device cpu`，NVIDIA 指定
`--device cuda`；Apple GPU 可尝试 `--device mps --dtype float32`。
模型目录必须包含权重、配置和 tokenizer；adapter 还必须能找到其基础模型。
`--local-files-only` 会禁止隐式联网补齐缺失权重，但不会替你打包 adapter 的基础模型。

原生非流式贪心请求需要合批时：

```powershell
saddlellm serve MODEL --local-files-only --batch-size 4 --batch-length-bucket-size 64 --batch-token-budget 4096 --max-queued-tokens 16384
```

客户端设 `temperature=0`、`stream=false`。默认不启用长度分桶；具体预算需结合实际模型内存测量。
合批不覆盖 HF/custom、采样、流式路径。`max_concurrency` 默认 1；
多个 Uvicorn worker 会重复加载权重，不是免费的多 GPU 调度。
建议一个进程管理一份模型，使用多个实例/外部路由分流，避免在同一卡上盲目增加 worker。

## 健康、就绪与安全

- `/livez`：进程能响应请求；不运行 GPU forward。
- `/readyz`：模型已加载且服务生命周期已启动时返回 200；启动前/关闭后返回 503。
- `/health`：保留原接口，新增 `ready` 和实际输入设备；批队列还显示最近 1024 个样本的等待/
  批执行 P50/P95/P99 与 padding 统计。不是 Prometheus 全周期直方图。

排队或 GPU 正在工作不代表不健康，不能以“队列不为空”作为重启理由。
客户端断连仍等待已执行的模型线程真正结束，再释放许可和请求缓存。
这里的批执行时间不是用户 TTFT；后者需要客户端流式测量。

对外部署应在受控网关后使用 TLS、鉴权和连接/速率上限。本工具不是完整网关。
通过 `--api-key-env SADDLELLM_API_KEY` 读取已设置的环境变量，别把密钥写进命令或镜像。
不需要页面可加 `--no-web-ui`；默认不启用 `--trust-remote-code`。
流式异常只返回通用错误，不将模型异常里的提示词/密钥原文发给客户端。

## 容器：CPU 与 CUDA 共用一个 Dockerfile

新增 [Dockerfile](../Dockerfile)、[.dockerignore](../.dockerignore)、
[模型加载依赖约束](../deploy/constraints-serve.txt)。默认 CPU wheel；模型、数据、artifacts、
本地密钥不进入构建上下文。运行用户不是 root，权重建议只读挂载。
约束固定了模型加载依赖家族，不是完整的传递依赖锁文件；正式发布还需锁定镜像 digest、
记录完整环境，并在实际部署平台执行 smoke。

构建 CPU 镜像：

```powershell
docker build -t saddlellm-inference:cpu .
```

Windows PowerShell 示例（换成自己真实的绝对模型目录）：

```powershell
docker run --rm --stop-timeout 60 -p 127.0.0.1:8000:8000 --mount "type=bind,source=E:/models/my-model,target=/models,readonly" saddlellm-inference:cpu
```

Linux/macOS Docker 的 source 换成如 `/srv/models/my-model`。容器内是 Linux；
macOS Docker 不会由此获得 Apple MPS。Windows 的容器 GPU 环境也不能当作原生 Windows CUDA 等同处理。
默认 CMD 已配置 `/models`、`0.0.0.0`、离线加载和仅 API。额外传命令参数时会替换 Docker CMD，
所以要一起写出模型路径和监听设置：

```powershell
docker run --rm --stop-timeout 60 -p 127.0.0.1:8000:8000 --mount "type=bind,source=E:/models/my-model,target=/models,readonly" saddlellm-inference:cpu /models --host 0.0.0.0 --port 8000 --local-files-only --no-web-ui --batch-size 4 --batch-length-bucket-size 64
```

NVIDIA GPU 容器示例使用同一套 Python 代码，只改 PyTorch wheel 源：

```powershell
docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t saddlellm-inference:cuda .
docker run --rm --gpus all --stop-timeout 60 -p 127.0.0.1:8000:8000 --mount "type=bind,source=/srv/models/my-model,target=/models,readonly" saddlellm-inference:cuda
```

需要兼容的 NVIDIA 驱动和容器 GPU 支持；镜像本身不能安装宿主驱动。
本轮无法连接本机 Docker daemon，因此仅完成配置/结构检查，**未执行镜像构建或容器 GPU 测试**。
`--stop-timeout 60` 只是例子；若实际模型正在执行的 forward 更长，需要按实测调整，
超时强杀可能中断在途响应。容器 HEALTHCHECK 使用 `/readyz`，启动宽限期也应按模型加载耗时调整。

## 流式补全与真实首文本计时

两个入口现在都支持 `stream=true`：`/v1/chat/completions` 和 `/v1/completions`。
设置 `stream_options: {"include_usage": true}`，结束时会在 `[DONE]` 前返回真实 token 用量，
usage 事件的 choices 为空；普通文本事件的 usage 为 null。默认不发送额外 usage 事件。
非流式接口和 Python `complete_batch` 保持兼容；Python 还可用
`LocalGenerationEngine.stream_complete(request, stream_state)` 拉取补全文本。

Python 无需启动 HTTP 服务；中途停止迭代时也应关闭生成器，以释放请求缓存：

```python
from contextlib import closing
from saddlellm.runtime.InferenceServer import (
    CompletionRequest, InferenceServerSettings, LocalGenerationEngine, load_inference_model,
)

settings = InferenceServerSettings(model_path="MODEL", device="auto", local_files_only=True)
model, tokenizer = load_inference_model(settings)
engine = LocalGenerationEngine(model, tokenizer, settings)
request = CompletionRequest(prompt="介绍 KV 缓存", temperature=0, max_tokens=64)
state = {}
with closing(engine.stream_complete(request, state)) as chunks:
    for piece in chunks:
        print(piece, end="", flush=True)
print(state)  # 完成后可查看真实 token 数和 finish_reason，不含提示词内容
```

`echo=true` 的提示词回显不是模型生成内容，不能把回显当作模型首 token。

已经运行的服务可用：

```powershell
python tools/benchmark_serving.py --target saddle=http://127.0.0.1:8000/v1 --model MODEL_NAME --prompts-file prompts.txt --requests 64 --concurrency 8 --max-tokens 32 --stream
```

`MODEL_NAME` 用 `/v1/models` 返回的服务别名；prompts.txt 是 UTF-8、每行一个 prompt。
去掉 `--stream` 测非流式合批。与其他引擎对比可追加 `--target other=.../v1`，
但必须自行确认同一权重、tokenizer、硬件和缓存策略。

TTFT 在此定义为客户端收到第一个非空文本 SSE 的时间，包含排队、网络和 tokenizer 缓冲；
HTTP 头、空文本、角色事件不算。文本 chunk 可能有多个 token，chunk 间隔不是 TPOT。
空文本输出的 TTFT 为 null，缺失用量、错误或截断的流直接判失败，不估算或补零。
P99 同时报告样本数；几十个样本只能用来发现问题，不能充当生产 SLA。
工具是固定并发客户端的闭环负载，不是独立请求到达的开放式 trace。

## 本轮调度改动

凑批不再固定睡眠：新请求到达、取消或关闭都能唤醒；批次达到人数/预算上限立即执行。
首次凑批截止时间以最老请求的入队时间为准，不因后续到达反复延长。
默认原生 HTTP 队列选择 `shortest_work`，使用 `prompt + max_new_tokens` 作为粗略工作量，
兼顾短 prompt 但超长输出的任务；不是准确的 GPU FLOPs 或真实生成长度预测。
默认 1000 ms 老化阈值仍保留，在批边界优先最老请求，不抢占已执行 forward。
Python settings 可以设置 `batch_scheduling_policy="fifo"` 或 `"shortest_prompt"` 作对照。
Windows 上等待/执行统计改用高精度单调性能计时器，避免几毫秒任务被粗粒度时钟记成 0。

连续批处理、流式请求合批、分页 KV、MTP 生产级提交/回滚、跨实例缓存感知路由仍未实现。
这轮没有把新的微批行为称为完整 continuous batching。

### 源码入口

- [InferenceDevice.py](../saddlellm/runtime/InferenceDevice.py)：设备可用性、auto 选择和 dtype 默认值；不在配置导入时加载 torch。
- [InferenceServer.py](../saddlellm/runtime/InferenceServer.py)：模型加载与设备迁移、HTTP 许可、两个 SSE 入口共用的清理/用量流程、生命周期探针。
- [InferenceBatcher.py](../saddlellm/runtime/InferenceBatcher.py)：提交唤醒、凑批截止时间、工作量/老化选择、线程结束后才释放许可、有界统计。
- [benchmark_serving.py](../tools/benchmark_serving.py)：真实客户端时间、跨网络块 SSE 解析、缺失用量/截断判失败。
- [benchmark_native_serving.py](../tools/benchmark_native_serving.py)：同权重 loopback 服务对照，按顺序测量，退出时关闭监听和服务线程。

## 验证与性能记录

设备选择替身覆盖 auto 优先级、显式不可用设备、MPS FP32 默认与 native/adapter 加载传参。
服务检查覆盖启动/关闭 readiness、两个 SSE 入口、真实 usage、echo、异常脱敏、双入口断连
许可保留、凑批唤醒、预算饱和、输出预算优先级、防饥饿与有界指标。

```powershell
python -m pytest -q tests/test_inference_platforms.py tests/test_inference_stream_metrics.py tests/test_native_streaming.py tests/test_serving_benchmark.py
```

新增 [跨平台 CI](../.github/workflows/inference-platforms.yml) 包括三平台 CPU 用例和 Linux CPU
镜像构建/原生生成 smoke；添加 CI 配置并不证明它已在远端运行成功。

不下载模型的真实 HTTP 基准：

```powershell
python tools/benchmark_native_serving.py --device cuda --dtype float32 --requests 16 --concurrency 8 --decode-tokens 32 --repeats 3
python tools/benchmark_native_serving.py --device cuda --dtype float32 --requests 16 --concurrency 8 --decode-tokens 32 --repeats 3 --stream
```

使用相同随机未训练模型、数值 tokenizer、混合长度的合成输入，仅用于推理流程和排队对照。
多个 loopback 服务共用同一个模型，按顺序测试；没有多个模型争用 GPU。
首次 BF16 探测发现批形状改变后部分贪心输出不同，不能据此宣称严格等价加速。
低精度测试小样例通过不保证所有模型/输入都不变；正式使用需要自己的质量与一致性验证。
下面的正式记录以独立重测结果为准，不混用 CPU smoke、无预热或与测试重叠的探测数据。

### 独立 FP32 HTTP 结果

RTX 4090 Laptop GPU、Torch 2.6.0+cu124，4 层 GQA，hidden=128、词表 1024；
本机已有 FastAPI 0.110.2、Uvicorn 0.29.0、Pydantic 2.5.2，没有为本轮测量升级依赖。
容器/CI 按 pyproject 安装的范围不同，不能把这组宿主性能当作新镜像已验收的结果。
prompt 长度为 33/48/57/62/129/144/153/158，输出预算 32，客户端并发 8，
每轮 16 请求，预热 1 轮后测 3 轮，每方案 48 个计时样本。权重相同、前缀缓存关闭，
测试过程没有与本轮回归重叠；分桶宽度 64、最大批 4。
原始健康快照的计数包含预热；表中客户端延迟/吞吐只包含正式计时轮。

| 非流式方案 | 实际输出 token/s | 端到端 P50 | 端到端 P99 | 最大实际批 |
| --- | ---: | ---: | ---: | ---: |
| 单请求 B=1 | 137.57 | 1814.17 ms | 2328.48 ms | 1 |
| 严格等长微批 | 193.41 | 1021.84 ms | 2342.19 ms | 2 |
| 相近长度分桶 | 372.46 | 515.36 ms | 1307.18 ms | 4 |

三种方案实际 token 用量与输出文本 SHA256 均相同。该负载的分桶吞吐是严格等长微批的
约 **1.93×**、B=1 的约 **2.71×**；不能推广为所有模型/请求分布，也不是 vLLM/SGLang 对比。
严格等长方案的 P99 在此甚至略高于 B=1，说明不能用平均吞吐替代尾延迟。
原始逐请求记录：[native_serving_20260927_fp32.json](../artifacts/native_serving_20260927_fp32.json)。

同样负载的单请求流式路径（**不使用合批队列**），48 个样本：
首非空文本 P50 **1804.12 ms**、P99 **2000.78 ms**，完整响应 P99 **2271.17 ms**，
实际输出吞吐 **118.22 token/s**。这里大量首文本等待来自请求串行占有模型许可；
并发 8 不代表 8 路并行 GPU 解码。此测量是现状基线，不是已优化流式 TTFT 的证据。
下一步应针对流式的公平/连续调度，而不是仅把 `max_concurrency` 改大。
原始记录：[native_serving_20260927_stream_fp32.json](../artifacts/native_serving_20260927_stream_fp32.json)。

### BF16 非等价结果也保留

独立 BF16 重测中用量一致，但分桶有 12/48 个请求的文本 SHA256 与 B=1 不同；
严格等长微批这次未发现文本差异。这是固定 8 类合成输入重复后的结果，不能解读为真实业务错误率。
进一步逐 token 检查：输入索引 1 在输出 offset=13、索引 6 在 offset=19 首次分歧（索引从 0 起）；
参考 top-1/top-2 margin 都为 0.00390625，分桶前后最大 logit 差也为 0.00390625。
该观测支持“低精度舍入足以改变相近候选的贪心选择”，但不是对所有差异原因或模型质量的证明。
本轮不报这组的严格等价加速倍数，也没有为了让结果好看而删掉它。
记录：[native_serving_20260927_bf16_non_equivalent.json](../artifacts/native_serving_20260927_bf16_non_equivalent.json)。

### 最终回归结果

本轮在上一批窗口/预算回归的基础上新增 41 项验证用例，最终同一次选定范围组合运行：
**800 passed，1 deselected，29 warnings，83.75 秒**。不是全仓库测试或跨平台真机认证。
唯一主动排除的是独立 HF/TRL 的 `test_pinned_runtime_completes_real_post_training_smoke`；
原生 scratch、SFT/DPO、重载和服务路径仍在实际选定测试中。
MPS 是设备路由替身测试；Megatron/SGLang 是配置/协议替身，不是生产多 GPU 测试。
警告主要来自已有依赖弃用、PEFT Conv1D 和 Windows 文件时间戳回退，本轮没有升级依赖来隐藏它们。

复现组合范围：使用 [上一批完整命令](INFERENCE_WINDOW_AND_SCHEDULING.md#已验证与尚未完成)，
在最后的 `-k` 过滤条件之前追加 `tests/test_inference_stream_metrics.py tests/test_inference_platforms.py`。
新文件 E/F/I、既有修改文件语法/致命 lint、compileall、生成函数索引检查和 `git diff --check` 通过。
首次回归暴露了 socket smoke 的不合理假设（两请求必须在 5 ms 内到齐）；
修正为批大小是上限、允许真实网络超过凑批期限，确定性 B=2/唤醒用例仍保留。
