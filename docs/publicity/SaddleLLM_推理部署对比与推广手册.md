# SaddleLLM 推理部署对比与推广手册

适用版本：SaddleLLM 2.40。资料核对日期：2026 年 9 月 27 日。

Saddle 的优势是把训练后的模型更容易地部署成文本服务，同时提供自有原生优化路线和可选 vLLM 后端。当前证据不足以证明其自有引擎全面领先其他引擎。本文帮助使用者选择路线、实际启动服务，并准确理解可对外宣传的结果。

## 一 先区分模型能力和执行效率

模型能力决定答案是否正确、指令是否遵循以及推理是否可靠，通常受模型结构、权重、训练数据和训练方法影响。服务优化主要改变执行效率，包括显存占用、请求吞吐和延迟。把模型部署得更快，不等于把 Qwen3-8B 训练成了更强的模型。

吞吐是所有请求实际输出 token 数除以整个测量区间的墙钟时间；单用户速度是某个请求自身的生成速度；TTFT 是从发出请求到看到首个文本的时间；P95 响应时间表示约 95% 的记录请求在这一时间内完成。增加并发可能提高聚合吞吐，也可能只增加排队，不能只看并发数。

## 二 各方案的定位和选择

以下是功能与使用场景的对比，不是同精度速度排行榜。“适合”是基于功能定位的判断，不是实验得出的普遍性能结论。

| 方案 | 部署入口 | 主要适用场景 | 与 Saddle 的关系和边界 |
| --- | --- | --- | --- |
| 手写 Transformers 基准服务 | 自己的服务脚本 | 兼容原始 HF 模型及调试 | 我们旧实验的控制组，不代表 Transformers 的全部最新功能 |
| Saddle 本地 HF | saddlellm serve MODEL | 原始 HF 模型与适配器的本地接入 | 统一服务与校验，但调用 HF generate 本身不会自动获得高并发引擎加速 |
| Saddle 原生 W8A16 | serve 加 native 或 int8 选项 | 已适配的完整 Qwen3 与受控量化试验 | 自有前向和 Triton 算子；微批有场景限制，尚无自有生产级连续批处理和分页 KV |
| Saddle 接入 vLLM | serve 加高吞吐要求 | 兼容 Linux CUDA 的并发文本服务 | 统一入口使用 vLLM 执行；不能把 vLLM 的内核归为自研 |
| 直接部署 vLLM | vllm serve MODEL | 专业推理与高并发调度 | 自带连续批处理、分页 KV 及多种优化，直接启动也很简洁 |
| 直接部署 SGLang | python -m sglang.launch_server | 专业推理与前缀复用 | 当前 Saddle 的 SGLang 接入用于 rollout 采样，不是 serve 后端选项 |
| Ollama | ollama serve 和模型管理命令 | 本地使用与模型管理 | 平台体验便捷，也支持兼容接口；旧实验使用的 Q4 权重不同于 BF16 |
| llama.cpp | llama serve 或 llama-server | GGUF 量化与 CPU GPU 等多种硬件 | 有现成服务、连续批处理和多种量化能力；没有本项目同机速度数据 |

vLLM 的连续批处理、分页 KV、图执行等能力见 [官方介绍](https://docs.vllm.ai/en/latest/)，服务命令见 [快速上手](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。SGLang 的 RadixAttention、前缀复用和硬件支持见 [官方介绍](https://docs.sglang.io/)，启动与客户端示例见 [请求教程](https://docs.sglang.io/docs/basic_usage/send_request)。Ollama 的便捷平台入口与兼容文本接口见 [快速上手](https://docs.ollama.com/quickstart) 和 [接口说明](https://docs.ollama.com/api/openai-compatibility)。llama.cpp 的量化与硬件路线见 [官方仓库](https://github.com/ggml-org/llama.cpp)，服务能力见 [服务文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

Megatron 是训练执行后端，不应放进推理速度排行榜。Saddle 已接通 Megatron Bridge 的预训练和 SFT 配置入口，以及 SGLang 采样适配；没有接通 Megatron 的 DPO、KTO、PPO、GRPO，也没有完成持续重叠的异步在线 RL。详见 [可选运行时指南](../OPTIONAL_RUNTIME_BACKENDS.md)。

## 三 我们确实具备的优势

### 统一入口减少服务拼接

CLI 的 `saddlellm serve MODEL` 和 Python 的 `serve()` 都默认使用 auto 路由。普通完整 HF 检查点在兼容 Linux CUDA 且已安装 vLLM 0.30 时选择 vLLM；本地检查点与显式原生设置走相应路线。auto 根据环境和检查点类别选择，不是对每个模型都做了性能自动调优。

这对反复训练、评估和部署模型的团队更有价值：接口和调用习惯保持一致，底层模型与后端可以按需要选择。对于只运行现成模型的个人用户，不能宣称比 Ollama 更简单；对于已经熟悉 vLLM 的部署团队，直接 vLLM 命令也足够方便。

### 自有优化有实际执行证据

完整 Qwen3-8B 原生路线覆盖权重映射、前向、KV 处理和逐 token 生成，并实现显式 W8A16、QKV 与 gate/up 投影合并、专用小批 GEMV 与较大输入 GEMM。其驻留参数及 buffer 为 9,441,315,840 字节，原始 BF16 参数约 16,381,470,720 字节，减少约 42.4%。它是有损量化；不等于整个服务显存下降同样比例。

### 部署状态能够被核对

服务提供 liveness、readiness 和实际后端健康状态；错误模型名、不支持参数和过长输入会明确返回错误。接入队列有边界，取消与引擎清理由服务生命周期处理。默认不自动量化、不默默改成 CPU，也不为启动服务停止其他业务。这些是工程完整性的改进，并非其他工具均不存在的独占能力。

### 高吞吐路线与本地路线分开管理

`--require-high-throughput` 要求选择兼容的连续批处理 vLLM 路线。选中引擎后若初始化失败，则直接报告，不悄悄换成 HF。使用者可以在同一个入口下选择原生试验或专业引擎，但仍需安装对应依赖、准备模型并满足资源要求。

## 四 实测数据不能合成一个排名

### 最新原生路线内部对照

完整 Qwen3-8B，8,190,735,360 参数，NVIDIA L20，W8A16 权重与 BF16 激活。非流式 HTTP raw completion，C8，同一 10-token 提示词，每个请求实际输出 64 token，temperature 为 0，前缀缓存关闭。每个配置预热 1 轮、计时 2 轮，16 个计时请求。两个配置顺序执行。

| 执行设置 | 总输出吞吐 | P95 响应 |
| --- | ---: | ---: |
| 固定微批关闭 | 32.43 token/s | 15.754 秒 |
| 固定微批大小为 8 | 252.78 token/s | 2.039 秒 |

聚合吞吐约 7.79 倍。串行与批处理的文本哈希不同，差异原因尚未定位，业务质量尚未验收。因此不能写“无损 7.79 倍”，不能说单用户达到 252.8 token/s，也不能把这个短提示成绩承诺给所有聊天与长上下文请求。详见 [本轮验收](../SIMPLE_SERVING_VALIDATION_20260927.md)。

同轮原生流式路径在单模型执行许可、未合批的配置下，C1 为 32.33 token/s，C8 为 31.76 token/s；C8 的 P95 首文本时间约 14.08 秒。这个结果说明：队列能够管控资源，但不会自动提升流式吞吐。

### 早期外部后端对照

下面是 2.36 时期的 L20 测试，不是新版 Saddle 2.40 vLLM 接入的完整 8B 性能验收。C8 非流式，16 个不同任务提示词，长度 110 至 1304 token，64-token 输出预算，1 轮预热、2 轮计时，共 32 个记录请求。模板统一关闭 thinking。设备并非完全独占，各后端顺序启动。

| 当时部署的后端 | 精度 | 总输出吞吐 | P95 响应 |
| --- | --- | ---: | ---: |
| 手写 Transformers 基准服务 | BF16 | 39.75 token/s | 13.000 秒 |
| Saddle HF 服务 | BF16 | 39.80 token/s | 12.983 秒 |
| 独立部署 vLLM | BF16 | 251.77 token/s | 2.035 秒 |
| Ollama | Q4_K_M | 384.78 token/s | 1.349 秒 |

同一 BF16 实验中，Saddle HF 与基准结果接近，输出文本一致，没有证据证明服务包装本身加速模型；vLLM 观测聚合吞吐约为基准的 6.33 倍，不过 32 个性能请求中有 12 个文本与基准不一致。相同精度也不保证逐字输出相同，需另做质量判断。

Ollama 使用独立四比特 GGUF，缓存行为也由其管理，32 个性能请求文本均与 BF16 基准不同。该数字反映当时实际部署配置，不是同精度引擎胜负。SGLang 和 llama.cpp 没有本项目对应的同机完整 8B 数据，不填写估算成绩。详见 [早期实验说明](../QWEN3_8B_SERVING_EXPERIMENT.md) 及 `artifacts/qwen3_serving_20260927/summary.json`。

最新原生 252.78 与旧 vLLM 251.77 数值相近，但精度、提示分布、GPU 可用状态、实验版本及轮次不同，不能把两者相除后宣布胜负。

### 新版接口验收的范围

本轮 618 项相关回归测试通过，不是全仓所有测试声明。完整 8B 已做原生 HTTP 和 OpenAI SDK 文本接口验证。新版 vLLM 接入已通过真实 GPU 上约 78.8M 参数未训练 fixture 的异步接口检查；该 fixture 不是 8B，不能证明 8B 吞吐或回答质量。由于当时空闲显存不足以另放完整 BF16 8B 与 KV，未做新版 vLLM 完整 8B 测速。

## 五 新手部署步骤

### 先选环境和模型

Python 最低要求为 3.10。建议新建独立虚拟环境，并取得 2.40 源码或对应 wheel。本轮 2.40 没有上传 PyPI，不能用旧版安装结果代表本文功能。完整 Qwen3-8B 的 BF16 参数约 16.38 GB，还需要 KV、激活和框架显存；可用显存不足时，不要直接照抄大并发配置。

完整本地 HF 模型目录应包含配置、tokenizer 和全部权重。已有目录可以加 `--local-files-only`，缺少文件时应补齐，而不是期待框架忽略。首次使用在线模型标识可能下载大文件，需要网络和磁盘空间。

### Windows 本地路线

在源码目录操作，模型路径替换为自己的实际路径：

```powershell
python -m venv .venv-serving
.\.venv-serving\Scripts\Activate.ps1
python -m pip install ".[serve]"
saddlellm serve "D:\models\Qwen3-8B" --model-name qwen3-8b
```

如果 PowerShell 阻止激活，不必修改全局执行策略；可直接使用新环境程序：

```powershell
.\.venv-serving\Scripts\python.exe -m pip install ".[serve]"
.\.venv-serving\Scripts\saddlellm.exe serve "D:\models\Qwen3-8B"
```

Windows 本地可用，不等于 Linux vLLM 的高吞吐路线在 Windows 原生环境也通过。macOS MPS 路线仍需真实设备验收，不宣传各平台拥有相同速度。

### Linux CUDA 的 vLLM 路线

在源码目录另建环境，与 serve 和 posttrain 依赖分开安装：

```bash
python3 -m venv .venv-vllm-serving
source .venv-vllm-serving/bin/activate
python -m pip install '.[serve-vllm]'
saddlellm serve /models/Qwen3-8B --model-name qwen3-8b --require-high-throughput --dry-run
saddlellm serve /models/Qwen3-8B --model-name qwen3-8b --require-high-throughput
```

dry-run 只显示计划，不加载权重，不证明显存足够或模型可启动。当前适配 vLLM 的约束为 0.30 系列；训练及本地服务的 Transformers 4 与该 vLLM 路线的 Transformers 5 不能随意混装。后台管理器还需把虚拟环境 bin 放入 PATH，以找到 ninja 等底层工具。原生 INT8 GPU 使用也需要兼容的 Triton 与 CUDA；无需用户手写 C++，但依赖和首次 JIT 编译仍存在。

### 显式试验原生 W8A16 微批

以下是有损量化和限定场景的试验配置，不是默认生产建议。已安装本地服务依赖与兼容 CUDA Triton 后使用：

```bash
saddlellm serve /models/Qwen3-8B --backend native --device cuda --dtype bfloat16 --quantization int8 --batch-size 8 --batch-token-budget 2048 --model-name qwen3-8b --local-files-only
```

适用的请求需为非流式贪心配置，例如 `stream=False`、`temperature=0`。不满足微批资格的请求不能期待同样合批收益；先检查 health 中的 batching 与 native_int8，再用自己的业务题集验收。此路线从原始权重加载并量化，不修改原始磁盘权重；首次加载与 JIT 不计入上述稳态吞吐。

### 确认服务就绪

服务默认只在本机 `127.0.0.1:8000` 监听。Windows 使用 curl.exe，Linux 可用 curl：

```bash
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

readyz 返回 200 后再发送业务请求。health 用于核对实际后端与执行设置，不因为看到 serve 命令就假定正在使用 vLLM 或原生微批。Ctrl+C 退出；不要通过多个 Uvicorn worker 为同一张卡复制模型。

## 六 Python 客户端调用

客户端可以使用独立环境，安装 OpenAI Python SDK 后调用本机服务。下例仅适用于未开启鉴权的本机服务，模型别名应与启动参数一致：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused-local-key",
)
result = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "解释 KV 缓存。"}],
    max_completion_tokens=256,
    temperature=0,
)
print(result.choices[0].message.content)
print(result.usage)
```

服务默认保留模型模板的思考策略。Qwen3 等模型可能先生成思考过程，256 个输出 token 不一定足够完成答案；业务需要时增加输出上限。显式关闭 thinking 是质量与延迟选择，不应作为隐藏的加速手段。

流式输出设置 `stream=True`，需要用量时加入 `stream_options={"include_usage": True}`。最后的 usage 块可能是空 choices，遍历时不要无条件读取 choices[0]。客户端提前结束时关闭流，让服务取消对应请求。完整示例见 [服务使用手册](../OPENAI_SERVING.md)。

## 七 常用参数及其效果

| 参数 | 效果 | 使用注意 |
| --- | --- | --- |
| --backend auto local native vllm | 选择实际执行路线 | 当前没有 serve 的 sglang 选项 |
| --require-high-throughput | 要求连续批处理 vLLM 路线 | 不是 token/s 或质量保证 |
| --dry-run | 不加载模型地显示计划 | 不是显存及架构验收 |
| --model-name | 固定 API 的模型名称 | 客户端传错名称返回 404 |
| --max-input-tokens | 限制含模板的输入长度 | 默认过长返回 400，不静默截断 |
| --max-new-tokens | 服务端输出上限 | 默认 512；长思考任务可能需调高 |
| --default-max-new-tokens | 客户端未提供时的预算 | 默认 256 |
| --max-num-seqs | vLLM 的调度序列上限 | 增大需要更多 KV 与资源 |
| --gpu-memory-utilization | vLLM 显存预算比例 | 基于整卡总显存，不是空闲比例 |
| --batch-size | 原生固定微批上限 | 不是所有 HF 采样和流式请求都支持 |
| --max-pending-requests | 限制额外接入槽位 | 背压不是 GPU 调度加速 |
| --api-key-env | 从指定环境变量读取密钥 | 不把真实密钥写进命令或文章 |

OpenAI 兼容接口当前是文本子集，包括 models、chat、completion、SSE 与真实 usage。工具调用、多模态、JSON Schema、Responses、logprobs、seed 及多候选结果等不在当前服务支持范围。对于不支持的参数，返回 422，不会静默忽略。不得因为 vLLM 底层拥有某功能就声称 Saddle API 已暴露该功能。

## 八 背后代码怎样协作

使用者先经过 `cli.py` 或导出的 `serve()`，由 `runtime/ServingPlan.py` 读取检查点类别、探测环境并选择路线。`runtime/InferenceServer.py` 创建 HTTP 应用和服务生命周期；vLLM 路线由 `runtime/VLLMServing.py` 管理异步引擎，`OpenAIProtocol.py` 与 `InferenceAdmission.py` 负责文本契约与接入预算。

原生 Qwen3 路线由 `models/NativeQwen3.py` 核对架构并映射权重，`InferenceOptimizer.py` 打包投影，`CUDAInt8.py` 压缩权重并保持 scale 的精度。`Int8WeightKernels.py` 执行 Triton GEMV 或 GEMM；`SaddleModeling.py` 和 `NativeGeneration.py` 管理前向、KV 与增量生成，`runtime/InferenceBatcher.py` 在符合资格时组织固定微批。

本地 HF 路线仍由 Transformers 执行 generate。vLLM 路线由 vLLM 执行模型与调度。这样的分工使我们既可以继续优化自有路线，也可以接入成熟引擎；不把所有后端混称为同一套自研内核。

## 九 生产使用和下一轮验收

对外部署前应启用 API 鉴权、TLS 反向代理、请求大小限制、限流与超时，并通过业务题集验收实际精度和解码配置。先在独立端口和独立环境验证，再接入业务流量；不直接升级共享生产环境的大型依赖。

当前下一步最有价值的工作，是定位微批与串行的文本差异，并做业务质量对照；随后在显存充足的设备上，对新版 Saddle vLLM 路线进行完整 8B 的同精度验收。负载应覆盖混合长度、长上下文、流式与非流式、缓存冷热以及持续到达请求，并同时记录吞吐、TTFT、P95、失败率和资源峰值。

原生生产级连续批处理、分页 KV、图执行和多 GPU 张量并行应被视为后续能力建设，不作为当前已经实现的卖点。配置中的 tensor-parallel-size 不等于本轮已经完成多卡真机验收。

## 十 对外宣传的准确表述

可以说，SaddleLLM 提供一条命令与 Python import 的文本服务入口，支持明确选择的自有原生路线与 vLLM 后端；完整 Qwen3-8B 的 W8A16 八请求非流式微批，在本次有限测试中观测到约 252.8 输出 token/s；参数及 buffer 驻留量约 9.44 GB；本轮 618 项相关回归通过。

不能说，Saddle 自研引擎全面超过 vLLM、Ollama 或 SGLang；单用户默认每秒输出 252.8 token；量化和合批保持无损答案；所有平台吞吐一致；具备完整 OpenAI API；已经完成新版 vLLM 完整 8B 对照或持续异步在线 RL。

公众号正文见同目录 `SaddleLLM_公众号文章.md`，可按其主线发布。本手册中的仓库相对链接指向项目资料，放入公众号时应转换成附件或已发布的实际链接，不虚构项目官网、安装渠道或社区规模。完整源码逻辑手册仍可参考项目原有 Word 手册，本册专注当前服务与推理宣传。
