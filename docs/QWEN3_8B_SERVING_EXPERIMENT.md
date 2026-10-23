# Qwen3-8B 服务部署与后端对照实验

这份文档用于复现实验，不是网上榜单的转述。测试代码位于 `tools/`，真实测量保存到
`artifacts/qwen3_serving_20260927/`。正式结果与未完成事项在文末更新。

版本说明：本文是 2.36 的原始 8B 部署对照。随后 2.37 已接入
[可选 vLLM 服务后端](VLLM_SERVING_BACKEND.md)，但下列数字不是新 adapter 的 8B 性能验收。

## 比什么，不能混淆什么

选择 `Qwen/Qwen3-8B`，不是随意把它称作 7B。服务器已有原始 BF16 权重，
5 个 safetensors 分片合计 16,381,516,776 字节。架构检查为 36 层、hidden size 4096、
32 个 Q head、8 个 KV head。官方模型信息见 [Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)。

实验分两组：

| 组别 | 后端 | 权重与精度 | 能解释什么 |
| --- | --- | --- | --- |
| 同权重对照 | 原始 Transformers、SaddleLLM HF 路径、vLLM | 同一模型目录、BF16，无 adapter、无量化、无 CPU offload | 这些完整部署栈在本机测试负载下的延迟和吞吐；vLLM 自带不同版本依赖，不是单一调度算法的因果实验。 |
| 实用部署对照 | Ollama `qwen3:8b` | 独立 GGUF，由 `/api/show` 记录实际量化类型和模型摘要 | 用户部署体验、显存与回答差异。不能把量化收益归因于 SaddleLLM 或说它与 BF16 同精度。 |

当前 SaddleLLM 加载 HF Qwen3 时，底层仍是 Transformers 的 `model.generate()`。
原生 `SaddleForCausalLM` 的环形窗口 KV、投影打包、原生微批与前缀快照不会自动移植到 Qwen3。
因此原始 Transformers 是必要的控制组，不能用此前随机 tiny 原生模型的倍数替代这次结果。

## 服务器与隔离

实机核对：Ubuntu 22.04.5、x86_64、16 个逻辑 CPU、123 GiB RAM、NVIDIA L20 48 GB、
驱动 580.126.09。最初 GPU 已使用 40,096 MiB，剩余空间不足以加载 BF16 8B。
经用户授权，暂停旧 Web 和两个旧生成服务；保留 embedding、reranker 与其他应用。
暂停后 GPU 仍约使用 20,829 MiB，**不是完全独占 GPU 的实验**。

采用新建实验目录，不改旧服务的模型目录或依赖，不更新驱动、系统 CUDA 或全局 Python。
Saddle wheel 安装到独立 `saddle-site/`，复用只读的现有兼容依赖；这不是完全独立的依赖虚拟环境。
benchmark 和 vLLM 使用自己的虚拟环境，Ollama 二进制和模型放在实验目录。
测试服务仅监听 `127.0.0.1`，不开放公网端口；逐个加载推理模型，不同时争抢 GPU。
旧服务有恢复指令，并设置实验后端停止、旧服务启动的兜底定时器。

## 1. 打包并部署当前 SaddleLLM

在源码目录执行：

```powershell
python -m build --wheel --outdir artifacts/qwen3_serving_20260927/wheel
Get-FileHash artifacts/qwen3_serving_20260927/wheel/saddlellm-2.36-py3-none-any.whl -Algorithm SHA256
scp artifacts/qwen3_serving_20260927/wheel/saddlellm-2.36-py3-none-any.whl root@SERVER_IP:/YOUR_EXPERIMENT/wheel/
```

这是 Python wheel 打包，不要求自行编译 C/C++。GPU 算子由 PyTorch 等依赖提供。
实验使用的 wheel SHA256：
`13f583b81716af2192018d97510ffaf2ee1b724fb031fa8015c2620cd7839eeb`。
同样的版本号不代表同样的源码，必须保留这个摘要。

独立新环境的通常安装方式：

```bash
python3 -m venv /YOUR_EXPERIMENT/.venv-saddle
/YOUR_EXPERIMENT/.venv-saddle/bin/python -m pip install '/YOUR_EXPERIMENT/wheel/saddlellm-2.36-py3-none-any.whl[serve]'
```

本次服务器已具备兼容的服务依赖，为避免重复大包与修改旧环境，实际采用只读依赖加包覆盖层：

```bash
/home/medmodel/.venv/bin/python -m pip install --no-deps --target /YOUR_EXPERIMENT/saddle-site /YOUR_EXPERIMENT/wheel/saddlellm-2.36-py3-none-any.whl
cd /YOUR_EXPERIMENT
PYTHONPATH=/YOUR_EXPERIMENT/saddle-site /home/medmodel/.venv/bin/python -m saddlellm.cli --help
```

实际依赖核对：Python 3.10.12、Torch 2.6.0+cu124、Transformers 4.57.6、Accelerate 1.12.0、
PEFT 0.18.1、Pydantic 2.13.5、FastAPI 0.141.1、Uvicorn 0.53.0。
不把源码目录加入运行路径；检查 `saddlellm.__file__` 确实来自 `saddle-site/`。

## 2. 只加载 tokenizer，准备同一批原始提示词

上传 `prepare_qwen3_experiment.py`、`benchmark_serving.py`、`run_qwen3_serving_experiment.py`、
`serve_transformers_baseline.py`、`summarize_qwen3_experiment.py` 到实验的 `tools/` 目录。

```bash
python tools/prepare_qwen3_experiment.py --model-path /home/medmodel/models/Qwen3-8B --output-dir prepared --hash-weights
```

输出目录必须不存在，以免覆盖前一份证据。`--hash-weights` 会读取全部权重分片，
否则 manifest 明确标记权重哈希尚未计算。配置、tokenizer、索引与各权重都有 SHA256。
准备工具只计算摘要，默认 `official_repository_verified=false`；来源验证是下一步的独立检查。

计算哈希不等于验证来源。此次另外与官方仓库 `b968826d9c46dd6066d109eabc6255188de91218`
的 LFS SHA256/Git blob ID 比较：五份权重、tokenizer、配置与索引全部匹配，
证据在 `prepared/hf_provenance.json`；不是仅凭模型文件夹名字判断“原始模型”。

`performance.json`：16 个有不同编号的中文数据校验任务，输入长度为 110、344、824、1304 token，
各 4 个。所有提示词统一通过原始 HF tokenizer 的 chat template 渲染，`enable_thinking=False`，
以 raw completion 发送，避免每个后端再次套不同模板。
`quality.json`：8 个算术、JSON、抽取、排序、逻辑、翻译、代码和摘要 smoke 任务。
参考答案只是示例，不按字符串相等给翻译/代码打能力分，也不证明业务或医疗安全。

贪心解码用于部署一致性对照：temperature=0、top_p=1、repetition_penalty=1。
这是关闭 thinking 的固定实验，不是推荐所有任务一律贪心。

## 3. 每次只启动一个 GPU 生成后端

以下命令应在完成 GPU 维护授权、检查端口及显存后执行。不要对其他应用使用 `pkill python`。
多个 uvicorn worker 会复制模型，不能当成免费加速。

### SaddleLLM

```bash
PYTHONPATH=/YOUR_EXPERIMENT/saddle-site /home/medmodel/.venv/bin/python -m saddlellm.cli serve /home/medmodel/models/Qwen3-8B --model-name qwen3-8b --host 127.0.0.1 --port 8211 --device cuda --dtype bfloat16 --quantization none --local-files-only --no-web-ui --max-concurrency 1 --batch-size 1 --max-input-tokens 3840 --max-new-tokens 256
```

`--max-concurrency 1` 是基准配置，多个客户端请求排队共享同一模型。
另行测试增加该值的影响；这只是 HF 请求线程并行，不是连续批处理或分页 KV。
`--batch-size 4` 对这个 HF Qwen3 不会启用原生微批，不能用它声称 Qwen 合批加速。

从源码理解这条路径：`cli.py` 的 serve 参数构成 `InferenceServerSettings`，
`runtime/InferenceServer.py::load_inference_model()` 经 `models/ModelLoader.py::load_causal_lm()`
选择 HF 模型类并加载权重，然后 `create_inference_app()` 创建 FastAPI 和请求并发 semaphore。
completion 请求进入 `LocalGenerationEngine._generate()`：校验预算、tokenizer 编码、
输入移到 CUDA、在 `torch.inference_mode()` 中调用 HF `model.generate()`，最后解码并报告实际 token。
流式走 `_stream_generate()`，工作线程执行 generate，文本 streamer 转成 HTTP SSE。
`/health` 的 `streaming_backend=generate_worker`、`batching.enabled=false`、
`native_projection_packing=null` 与关闭的 prefix cache 是这次 HF 路径的实机证据。
增加 semaphore 许可数只是允许更多请求同时进入 generate，并不自动建立跨请求 KV 调度器。

### 原始 Transformers

```bash
/home/medmodel/.venv/bin/python tools/serve_transformers_baseline.py --model-path /home/medmodel/models/Qwen3-8B --port 8210 --alias qwen3-8b --context-limit 4096
```

这个很小的 FastAPI 服务不使用 Saddle 生成引擎；直接调用同版本 Transformers，
BF16、SDPA、`torch.inference_mode()`，只允许一个同时执行的 generate。
它只测非流式延迟/吞吐，不支持流式 TTFT；收到 stream=true 会报错，不伪造指标。
它是实验控制组，不是具有完整鉴权、背压与取消能力的生产服务器。

### vLLM

vLLM 0.30.0 安装在自己的虚拟环境，不装进旧服务环境。先检查兼容性，再启动：

```bash
source .venv-vllm/bin/activate
.venv-vllm/bin/vllm serve /home/medmodel/models/Qwen3-8B --served-model-name qwen3-8b --host 127.0.0.1 --port 8212 --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.47 --no-enable-prefix-caching --generation-config vllm
```

显存比例是本机共享 GPU 下的实验预算，不是任何机器的推荐常数。
GPU 配置与前缀缓存参数见 [vLLM 引擎参数文档](https://docs.vllm.ai/en/latest/configuration/engine_args/)。
若版本与驱动不兼容，记录启动错误并选择明确兼容的版本重测；不能擅自升级正在承载其他服务的驱动。

实际部署遇到并修复两个启动问题：系统默认镜像尚无 vLLM 0.30.0，且官方下载到服务器很慢，
因此从官方 PyPI 下载 wheel、核对 SHA256 后上传，再在实验 venv 内安装其依赖。
只运行 venv 中的绝对路径程序并不自动激活子进程工具搜索路径；首次 FlashInfer 预热找不到 `ninja`。
激活 venv 后启动成功，无需升级全局依赖或驱动。实际 vLLM 环境为 Torch 2.13.0、Transformers 5.17.0。

### Ollama

使用独立解压的官方 Linux 二进制，不运行会修改全局环境的安装脚本：

```bash
OLLAMA_HOST=127.0.0.1:8213 OLLAMA_MODELS=/YOUR_EXPERIMENT/ollama/models OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_FLASH_ATTENTION=true OLLAMA_KV_CACHE_TYPE=f16 /YOUR_EXPERIMENT/ollama/bin/ollama serve
OLLAMA_HOST=127.0.0.1:8213 /YOUR_EXPERIMENT/ollama/bin/ollama pull qwen3:8b
```

测速脚本使用 Ollama 原生 `/api/generate`，`raw=true`、`think=false`、`num_ctx=4096`、
`repeat_penalty=1`、输出预算一致，解析实际 `eval_count` 与服务端纳秒耗时，
并保留返回的 `prompt_eval_cached_count`（版本提供时）。参数与计数见
[Ollama Generate API](https://docs.ollama.com/api/generate)。
缓存行为仍受 Ollama 控制，不能把它声称为和关闭前缀缓存的 BF16 后端严格相同。

通过 systemd 临时 unit 启动 Ollama 时，应明确 `User=`。未指定用户的临时服务可能没有登录环境，
本次第一次启动就出现 `HOME is not defined`；声明运行用户后成功启动，没有用实验目录替换用户 HOME。

## 4. 执行测试矩阵

独立 benchmark 环境只需要 `httpx==0.28.1`。后端启动且 ready 后运行，例如：

```bash
.venv-bench/bin/python tools/run_qwen3_serving_experiment.py --url http://127.0.0.1:8211 --label saddle --model qwen3-8b --prepared-dir prepared --output-dir results/saddle --precision bf16 --cache-policy 'no cross-request prefix reuse; serial HF generation' --stream
```

其他后端分别替换 URL、label；原始 Transformers 不加 `--stream`。
Ollama 另外加 `--protocol ollama --model qwen3:8b`，precision 必须填实际 GGUF 类型。
这个脚本只访问已经启动的本机后端，不安装软件、不停止任何服务。
它会先检查就绪，最多等待约 120 秒（加上最后一次 HTTP 检查耗时），然后才创建输出目录并测速；
模型还在加载时的拒绝连接不是“吞吐为零”的结果。

固定矩阵：每种并发 1/4/8，16 请求一轮、64 输出 token 上限，1 轮预热、2 轮记录；
流式另测并发 1/4、8 请求、32 输出上限、1 轮预热和 1 轮记录。
质量 smoke 为 8 请求、256 输出上限、不预热、1 轮。输出会保留完整文本，只使用非敏感测试题。

指标：客户端整段延迟 P50/P95/P99、实际生成 token/整轮墙钟时间、固定预算命中率，
流式首个非空文本的 TTFT。TTFT 包括排队、prefill、tokenizer 文本缓冲和 loopback 通信；
数据 chunk 的间隔不是逐 token TPOT。
请求使用闭环、有限客户端 worker；不代表开放式生产到达率、持续峰值容量或 SLA。
只有 32 个非流式测量样本/配置，P99 尾延迟也不是大样本可靠性结论。

单后端报告中的 equality 标记只是自比较。跨后端必须比较相同 phase、prompt SHA、
输出 SHA/文本与实际 token 数。EOS 计数方式也可能不同，不能把所有差异都判为模型变差。
冷热启动、显存峰值和下载时间不在上述热态吞吐指标中；未测量的项目不得补估计数。

所有后端顺序跑完后，汇总相同负载的指标、文本摘要和真实回答：

```bash
.venv-bench/bin/python tools/summarize_qwen3_experiment.py --results-dir results --output summary.json
```

`summary.json` 必须不存在。它分别标注是否同负载、是否同一声明精度和权重，
不会把 Ollama GGUF 的速度差异叫作同精度引擎收益。原始 JSON 保留，方便复算与复查。

## 实测结果与验收状态

### 同一原始 BF16 权重：非流式热态结果

单位为实际生成 tokens/s。每个配置记录 32 请求；不含权重下载、模型加载或编译预热。

| 部署栈 | 客户端并发 1 | 客户端并发 4 | 客户端并发 8 | 并发 8 响应 P50 / P95 |
| --- | ---: | ---: | ---: | --- |
| 原始 Transformers，串行 | 39.66 | 39.75 | 39.75 | 12.71 / 13.00 秒 |
| Saddle HF，`max-concurrency=1` | 39.73 | 39.72 | 39.80 | 12.69 / 12.98 秒 |
| Saddle HF，`max-concurrency=4` | 39.76 | 40.06 | 39.83 | 12.86 / 12.98 秒 |
| vLLM 0.30.0 | 49.05 | 154.47 | 251.77 | 2.03 / 2.04 秒 |

本轮 Saddle HF 的总吞吐基本与原始 Transformers 持平，没有证据表明原生引擎优化已经加速 HF Qwen3。
vLLM 在并发 1/4/8 下相对原始 Transformers 约为 1.24/3.89/6.33 倍；
这是特定共享 L20、输入长度和短输出负载下的完整部署栈观察，不是所有模型/机器的固定收益。
vLLM 使用不同 Torch、Transformers 与推理内核，不能把全部差异仅归因于一项 KV 优化。

### 流式与 Saddle 配置调优

| 部署栈 | 并发 1 首字 P50 | 并发 4 首字 P50 | 并发 4 完整响应 P50 |
| --- | ---: | ---: | ---: |
| Saddle HF，`max-concurrency=1` | 107.88 ms | 2551.15 ms | 3296.72 ms |
| Saddle HF，`max-concurrency=4` | 108.04 ms | 462.52 ms | 3477.96 ms |
| vLLM | 90.22 ms | 366.41 ms | 1000.60 ms |

增加 HF 请求线程并行，本轮 4 并发首字中位数降低约 81.9%，但总吞吐基本不变，
完整响应中位数还略变慢。这是交互响应的配置取舍，不是免费提高吞吐；
也没有因此把全局默认并发改为 4。资源较紧时还应检查并行生成的显存与尾延迟。
原始控制组没有实现流式，所以没有拿非流式耗时伪造它的 TTFT。

### 输出一致不等于模型答对

同权重、同精度、同原始提示词也不保证不同推理实现逐字一致。
Saddle 默认及并发 4 的 HF 路径，各自 96 个性能请求和 8 个质量请求均与原始控制组逐字、token 数一致。
vLLM 的并发 1/4/8 性能请求分别有 16/14/12 个文本不同（每种 32 个），
这些请求的实际 token 数一致；8 道质量小题的文本和计数均相同。
内核数值计算及依赖版本差异可能影响贪心选择，但本实验没有定位 logits，不能断言差异的具体原因。

人工检查发现三种实现都把 `9、2、11、4` 升序排序回答为 **`4,9,11,2`**，
正确应为 `2,4,9,11`。另有“只回答是/否”题输出带标点，代码题带 Markdown 围栏，
若严格要求输出格式也不完全合规。其余小题没有明显语义错误，但只有 8 个通用 smoke 请求，
不能由此声称总体正确率提高、临床可用或量化无损。实际回答与 SHA 均保留供复查。

### Ollama：单列 Q4_K_M 实用部署结果

官方 Linux 二进制版本 0.34.4，`/api/show` 确认为 qwen3、8.2B、GGUF、Q4_K_M。
下载的模型主 blob 为 5,225,374,496 字节，完整 SHA256：
`a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f`。
官方分片下载后段变慢，本次只在独立实验缓存中补齐缺失 byte range，
核对完整摘要后才登记为模型 blob，未改原始 HF 权重。下载恢复证据保留在 `recovery/`。

| 指标 | 客户端并发 1 | 客户端并发 4 | 客户端并发 8 |
| --- | ---: | ---: | ---: |
| 非流式生成 tokens/s | 119.33 | 303.66 | 384.78 |
| 非流式完整响应 P50 | 0.54 秒 | 0.84 秒 | 1.32 秒 |
| 非流式完整响应 P95 | 0.57 秒 | 0.85 秒 | 1.35 秒 |
| 流式首字 P50（独立 32-token 负载） | 12.88 ms | 26.35 ms | 未测 |

这组热态观察混合了量化精度、推理实现与 Ollama 控制的提示词缓存差异。
不能用它证明 Ollama 在同 BF16 精度下比 vLLM 快，也不能把它称为 Saddle 的优化收益。
实际缓存计数证明：每种非流式配置的 32 个样本合计输入 20,656 token，
`prompt_eval_cached_count` 合计 20,624；每个请求都缓存了除最后一个 token 以外的输入。
两种流式配置也全部返回非零缓存计数。因此这应明确称为**提示词缓存命中的热态部署结果**，
不是未命中缓存的新请求性能；不能把低首字延迟单独归因于量化或更快的 prefill 内核。
若要比较未命中请求，应另设每轮唯一的提示词或验证可关闭缓存的版本/配置后重测，不能从这组数推算。

Ollama 的 96 个性能回答均与原始 BF16 文本不同，但实际输入/输出 token 计数相同；
8 道质量题只有摘要措辞不同，排序题仍同样答错。没有证据证明量化提高了准确率或完全无损。

实机日志记录 `offloaded 37/37 layers to GPU`；`/api/ps` 的 size 与 size_vram 相等，
context length 为 4096，说明此次模型没有 CPU 层回退。8 并行 slot 下，模型 buffer 约 4644 MiB，
FP16 KV buffer 约 4608 MiB；nvidia-smi 最后一次观察该进程为 9814 MiB。
这只是选定并行参数下的一次驻留观察，**不是显存峰值测量**，也不是 8B Q4 必须占这么多显存。

### 恢复与验收

2026-09-27 本轮验收完成：原始 Transformers、Saddle 默认/并发 4、vLLM、Ollama 的上述矩阵均已完成。
实验 GPU 后端全部停止；旧 4B、8B 生成与 Web 服务恢复为 active，两个 `/health` 与 Web 首页均 HTTP 200。
旧生成模型名称和 8bit 配置未变，embedding、reranker、RAG、其他应用仍 active。
健康恢复确认后才解除自动恢复定时器。新版 wheel 没有替换旧业务服务，
不把原始 Qwen3 模型擅自切换成护理业务模型；服务恢复记录不等于完整业务问答回归。

服务器保留实验目录 `/root/saddlellm-qwen3-8b-20260927.cYbU3l` 中的 wheel、环境、工具与证据，
方便再次安排维护窗口测试；不会在公网额外留下模型端口。

本地关键证据：

- [汇总指标与全部质量回答](../artifacts/qwen3_serving_20260927/summary.json)
- [原始逐请求 JSON](../artifacts/qwen3_serving_20260927/evidence/results/)
- [HF 官方文件来源核对](../artifacts/qwen3_serving_20260927/evidence/prepared/hf_provenance.json)
- [Ollama 模型格式与量化信息](../artifacts/qwen3_serving_20260927/evidence/prepared/ollama_provenance.json)
- [旧服务恢复检查](../artifacts/qwen3_serving_20260927/evidence/recovery/services_restored.json)
- [实验 wheel](../artifacts/qwen3_serving_20260927/wheel/saddlellm-2.36-py3-none-any.whl)

归档 `final-evidence.tar.gz` 的本地/服务器 SHA256 一致：
`a3b0900cdefad958ce36f15b6cbf5726b0128026a3c8c2aa9ef15ae90556c187`。
本轮新增基准工具与相关服务回归共 103 项通过，Ruff 检查、Python 编译检查、函数参考一致性检查通过；
这不是重新跑过整个项目全部测试的声明。

## 接下来怎么提升 Saddle 的真实 Qwen 服务性能

2.36 这轮新增的是可复现对照工具与部署验收，**当时没有把 vLLM 实现成 `saddlellm serve` 的新后端**，
也没有修改全局并发默认值。保持外部命令/import 简单的下一步，可以增加可选专业服务后端，
统一模型配置、API、鉴权、就绪检查、取消/排队与监控，让高并发请求由 vLLM/SGLang 调度；
原始 HF 路径继续作为依赖少、兼容性高的保底实现。

后续状态：2.37 已实现上述可选 vLLM 后端，并用 tiny fixture 通过真实 GPU/HTTP 接入验收；
具体见新后端指南，不能据此替换本文 8B 表格中的实测范围。

不需要先自行写 C++ 才能得到这类收益。先解决请求合批、KV 预算和后端接入，
再根据实际 profiler 判断是否有值得自写算子的瓶颈。
原生 Saddle 模型的优化应继续单独测试；HF Qwen、量化模型和原生模型不能混在一个加速数字里。
下一轮还应覆盖唯一新提示词、长输出、更长上下文、显存峰值、失败/取消与更大样本的质量集。
