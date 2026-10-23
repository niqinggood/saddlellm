# 推理性能比较、优势与优化路线

## 结论先看

2.40 补充了 [一条命令 OpenAI 服务](OPENAI_SERVING.md)：自动后端选择、禁止高吞吐降级、
官方 SDK 协议、有界 HTTP 接入和质量/模板保护。它不是又一项 8B 内核加速成绩。
2.40 的真实服务验收见 [部署与 HTTP 记录](SIMPLE_SERVING_VALIDATION_20260927.md)，
与下文 2.39 的单请求模型调用计时分开说明。

2.39 最新完整 8B 结果：自有 W8A16/Triton 路线在 Linux L20 上，BF16 激活/eager、
固定随机输入 128/输出 64、单请求模型调用复测，Saddle 未合并投影 29.45 → 合并后 34.84
输出 token/s（+18.3%）；同一套 Saddle 算子的 HF 架构对照为 27.20。
参数/buffer 驻留约 9.44 GB，较原始 BF16 权重约减少 42%；293 个位置的自编文本抽查
困惑度约上升 0.82%。这是量化优化，不是普通 BF16 无损、HTTP 吞吐或 vLLM 领先结论。
完整条件、代码、wheel 和原始证据见 [完整 8B W8A16 指南](NATIVE_INT8_8B.md)。

历史版本补充：本文保留早先 28M 原生模型的历史测量。2.37 已接入
[vLLM 服务后端](VLLM_SERVING_BACKEND.md)，2.38 又补了
[Qwen3 自研原生适配](NATIVE_QWEN3_BACKEND.md)。不同模型、硬件与轮次的数字不混算。

已有可复现的小模型 GPU 优势，但不能得出“整体超过 vLLM/SGLang”的结论。
优势是简单的 CLI/import 接口、对原生检查点的轻量优化，以及可选接入成熟后端。
不应把普通单机服务包装成已经具备高并发连续批处理的生产引擎。

本轮完成三项模型执行优化：

1. 服务加载时合并 Q/K/V、SwiGLU gate/up；旧权重不需重训。
2. 符合条件的单 token GQA 解码不重复展开 K/V。
3. 固定残差系数 1.0 跳过恒等乘法，可训练系数仍保留梯度。

使用说明和代码入口见 [原生推理说明](NATIVE_INFERENCE_ENGINE.md)。
`serve` 默认启用投影合并，`--no-fuse-projections` 可关闭；其他自动快速路径按资格选择。
通用训练加载器不合并参数对象。Megatron/SGLang 配置见
[可选运行时指南](OPTIONAL_RUNTIME_BACKENDS.md)，这与原生优化是不同层次的能力。

2026-09-27 补充了 [vLLM 公开数据和能力对照](#vllm-public-comparison)。
用户当前电脑不安装 vLLM，不为对比更换系统、升级 PyTorch 或新增大型依赖。
公开成绩用于分析定位与优化方向，不是 SaddleLLM 与 vLLM 的同机实测。

## 固定条件，比较同一份权重

本机：RTX 4090 Laptop GPU 16GB，Windows，Torch 2.6.0+cu124。
模型：`outputs/zh_28m_pretrain_v3/phase2_100m/final_model`，28,319,616 参数，
10 层、hidden=384、词表 32768、6 个 Q 头/2 个 KV 头。
权重文件 SHA256：`5b5b58eeb3be28d6e438907de2b49ad245115f052370db0f8e9ae79026bc3ef0`。

同一组固定随机输入 token，prompt=128、生成=32，greedy、禁用 EOS、无 stop、无重复惩罚。
每条路径预热 2 次，最新对照交错打乱顺序测量 7 轮，边界同步 CUDA，使用中位数。
计时是完整 model.generate 调用，包括 prompt 前向；吞吐只数实际生成 token，不数输入 token。
不包含 HTTP、分词、排队、文本解码或任务质量评估，不称为线上 TTFT/TPOT。
固定预算不代表真实请求的 EOS 分布，也不代表采样时的性能。

Transformers 参考模型不是下载的其他模型，而是同一检查点逻辑权重在内存中的
Llama 兼容结构。仅接受统一的 dense GQA/MHA、普通 RoPE、标准串行残差等受限子集，
严格检查 state_dict；不支持的结构会拒绝，不偷偷换权重。
它只用于测试，不是新的通用 HF 导出功能。

## 最新整段生成结果

### 单请求，BF16

| 路径 | 生成 32 token 中位耗时 | 生成吞吐 | 与原生参考的输出 token |
| --- | ---: | ---: | --- |
| 原生参考：现有缓存/采样优化，未合并投影、旧 KV 展开 | 554.66 ms | 57.7 tokens/s | 一致 |
| 原生：合并投影，旧 KV 展开 | 510.65 ms | 62.7 tokens/s | 一致 |
| 原生：合并投影 + 紧凑 GQA 解码 | 499.50 ms | 64.1 tokens/s | 一致 |
| Transformers eager | 702.36 ms | 45.6 tokens/s | 一致 |
| Transformers SDPA | 627.23 ms | 51.0 tokens/s | 一致 |

本次完整原生路径较未合并参考约 +11.0% 吞吐，较 Transformers SDPA 约 +25.6%。
GQA 折叠本身较“已合并、旧 KV 展开”约 +2.2%，不把全部收益都归因于它。
原始数据：[最新单请求 JSON](../artifacts/inference_compare_20260927_b1_final.json)。

### 四请求固定批次，FP32

| 路径 | 四请求总耗时中位数 | 聚合生成吞吐 | 与原生参考的输出 token |
| --- | ---: | ---: | --- |
| 原生参考：未合并投影、旧 KV 展开 | 508.75 ms | 251.6 tokens/s | 一致 |
| 原生：合并投影，旧 KV 展开 | 471.49 ms | 271.5 tokens/s | 一致 |
| 原生：合并投影 + 紧凑 GQA 解码 | 466.06 ms | 274.6 tokens/s | 一致 |
| Transformers eager | 674.34 ms | 189.8 tokens/s | 一致 |
| Transformers SDPA | 577.82 ms | 221.5 tokens/s | 一致 |

本次完整原生路径较未合并参考约 +9.2% 吞吐，较 Transformers SDPA 约 +24.0%。
GQA 折叠单独约 +1.2%；FP32 prefill 最大 logits 差异约 1.1e-5，完整生成 token 一致。
每轮每条路径都会重新生成；这些不是只测一段 attention kernel 的成绩。
原始数据：[最新四请求 FP32 JSON](../artifacts/inference_compare_20260927_b4_fp32_final.json)。

### 不回避失败与限制

- BF16 四请求跨框架测试出现 token 差异，报告输出 `all_outputs_equal=false`，脚本非零退出。
  原生合并前后仍一致，但不能把该组 HF 结果当作严格等价的速度胜负。
  [原始 BF16 差异记录](../artifacts/inference_compare_20260927_b4.json)。
- BF16 单请求输出虽一致，HF 与原生 prefill logits 并非位级相同，最大差异约 0.22。
  不同归一化/内核的舍入会累积，输出一致不等于所有任务质量已验证。
- 动态 KV + 完整 prefill logits 的旧式参考，在该短上下文小模型上有时反而略快。
  新缓存/最后位置投影的主要收益可能是减少复制和内存，不保证每个短请求都更快。
- 普通桌面机不是隔离压测机；GPU 时钟、桌面负载与系统状态明显影响绝对值。
  早先单请求曾测到约 144 tokens/s，最新约 64；其他参考路径也一起变慢。
  因此只比较同一轮的交错对照，不把两轮绝对数字相减称为优化收益，不挑最高成绩。
- 同时驻留多个参考模型会占额外显存；记录的 `max_extra_cuda_bytes` 是每次生成额外
  的 PyTorch 分配峰值，不包含全部权重、分配器保留、其他进程或所有 CUDA 工作区。
- 原生四请求 BF16 另一轮约 529→576 tokens/s，输出一致；这轮没有 HF 对照，
  不能与上面不同精度/不同系统状态的数据直接相除。
  [原生 GQA 分项记录](../artifacts/inference_compare_20260927_b4_folded.json)。

## 原生固定合批能带来什么

更早一组同模型 BF16、B=4、128/32 token 的独立测试，逐条调用约 56.6 tokens/s，
单固定批次约 221.6 tokens/s，约 3.92 倍聚合吞吐，输出 token 一致。
这比较四个请求完成的总时间，不是单请求延迟降为四分之一，也不是 HTTP 高并发成绩。
代码：[批量生成基准](../tools/benchmark_native_batching.py)。

服务端通过 `--batch-size 4` 合并兼容的同长度、同预算、非流式、temperature=0 请求。
流式、采样、混合长度或自定义/HF 生成不自动进入这条合批路径。
已有可选的单请求跨请求前缀快照，但固定合批不使用它。连续批处理、分页 KV 和
CUDA Graph 仍未实现；前缀快照不等于这些能力。

<a id="prefix-cache-results"></a>

## 跨请求前缀快照：已实现，但不默认开启

2026-09-27，使用相同的 28M 权重、RTX 4090 Laptop GPU、PyTorch 2.6.0+cu124，
固定 B=1、384 输入 / 32 输出、贪心生成、无 EOS 截断。每轮交错四种路径，预热 2 次、
测量 7 次，表中为中位数。部分命中复用前 320 个 token，基准显式开启该高级选项。

| 精度 | 路径 | 模型 prefill（ms） | 整段生成（ms） | 实际输出 tokens/s |
| --- | --- | ---: | ---: | ---: |
| FP32 | 无前缀缓存 | 6.165 | 182.923 | 174.94 |
| FP32 | 冷缓存：计算并保存快照 | 10.338 | 185.784 | 172.24 |
| FP32 | 完整提示词命中 | 2.340 | 183.715 | 174.18 |
| FP32 | 部分前缀命中 | 11.095 | 193.713 | 165.19 |
| BF16 | 无前缀缓存 | 15.592 | 479.186 | 66.78 |
| BF16 | 冷缓存：计算并保存快照 | 23.368 | 484.452 | 66.05 |
| BF16 | 完整提示词命中 | 4.100 | 461.639 | 69.32 |
| BF16 | 部分前缀命中 | 26.947 | 524.838 | 60.97 |

原始记录：[FP32](../artifacts/prefix_cache_20260927_fp32_verified.json)、
[BF16](../artifacts/prefix_cache_20260927_bf16_verified.json)。两份记录均保存权重 SHA256、
输入哈希、每次耗时、实际复用 token 数和误差；所有生成 token 与无缓存路径一致。
FP32 部分命中的 prefill logits 最大绝对差约 `5.25e-6`，其他路径本轮为 0。
这不是所有模型、精度、输入和采样的位级一致或任务质量承诺。

必须一起读的结论：

- 完整命中确实跳过 prompt 前向，模型 prefill 在这两轮分别约减少至原来的
  1/2.63、1/3.80；它不消除新增 token 的解码工作，也不是 HTTP TTFT 测试。
- FP32 整段生成没有变快，BF16 本轮吞吐约 +3.8%；另一轮
  [BF16 记录](../artifacts/prefix_cache_20260927_b1_bf16_final.json) 的完整命中反而约 -1.4%。
  桌面机时钟/负载有明显波动，不能挑一轮宣布通用收益，也不能横比 FP32/BF16 绝对值。
- 部分命中在这两轮都更慢。状态校验、KV 复制、后缀带历史注意力和快照保存有成本；
  减少计算 token 数不自动等于降低请求耗时。
- 热命中的预先填充不计入计时，冷路径包含快照保存。测试固定重复输入，
  不能代表真实服务的命中率、平均吞吐、HTTP 延迟或多租户能力。

因此服务默认 `--prefix-cache-mb 0`。按需启用时默认只复用完全相同的提示词：

```powershell
saddlellm serve outputs/zh_28m_pretrain_v3/phase2_100m/final_model --prefix-cache-mb 64
```

部分复用需要额外 `--prefix-cache-partial`，先用目标模型和负载验证再开启。缓存是
有限字节预算、最多 128 条、LRU 淘汰的不可变 prompt KV/原始 logits 快照；命中后
复制进请求私有 KV，不缓存答案。只支持普通原生 dense、无 padding 的单请求路径，
HF/custom、MLA、MoE、固定合批等不会使用它。详见
[原生推理说明](NATIVE_INFERENCE_ENGINE.md)。

本轮快照驻留 FP32 单条约 3.88 MiB、BF16 单条约 1.94 MiB；部分命中基准保留两条
完整快照，公共 KV 仍重复存储。预算不包含模型、活动请求、分配器保留和全部 CUDA 工作区，
不是整体 GPU 显存上限。模型状态、设备、精度或执行条件变化在下次符合条件的 lookup 时
使快照失效；`.data` 等未追踪改动需要显式清理。没有分页/零复制共享，也不支持并发换权重。

复现方法：

```powershell
python tools/benchmark_prefix_cache.py --model outputs/zh_28m_pretrain_v3/phase2_100m/final_model --device cuda --dtype float32 --prompt-tokens 384 --shared-tokens 320 --decode-tokens 32 --warmup 2 --repeats 7
```

## 注意力局部基准，不冒充模型成绩

[GQA 局部基准](../tools/benchmark_gqa_decode.py) 对比每次重复展开 K/V 与折叠查询行，
用随机 Q/K/V、固定历史、现有 SDPA，并检查误差容限；它不运行 Transformer、
LM head、采样或 HTTP。长历史上的局部收益可能明显高于当前 28M 模型整段约 1–3%，
尤其是 KV 复制占比高时；不可宣称模型因此整体提升数倍。

本机 BF16、32 个 Q 头/8 个 KV 头、head_dim=128、历史 4096，预热后每组 100 次、
交替测量 5 组的局部实测如下；两组注意力结果最大绝对误差均为 0：

| 局部输入 | 每次重复 KV / 折叠查询耗时 | 局部比值 | PyTorch 额外分配峰值 |
| --- | --- | ---: | --- |
| B=1 | 0.456 / 0.262 ms | 1.74 倍 | 约 64 MiB / 24 KiB |
| B=8 | 3.311 / 0.378 ms | 8.76 倍 | 约 512 MiB / 192 KiB |

原始记录：[B=1 注意力 JSON](../artifacts/gqa_decode_20260927_b1_4k.json)、
[B=8 注意力 JSON](../artifacts/gqa_decode_20260927_b8_4k.json)。额外峰值不包括输入 K/V，
也不是整模型显存；上述比值尤其不能用于声称“模型整体快 8.76 倍”。

```powershell
python tools/benchmark_gqa_decode.py --device cuda --dtype bfloat16 --batch-size 1 --history 4096
python tools/benchmark_gqa_decode.py --device cuda --dtype bfloat16 --batch-size 8 --history 4096
```

注意：当前 28M 检查点 context=512，4k 局部基准不能证明该模型已支持 4k 上下文。

## 怎样复现同权重对照

```powershell
python tools/benchmark_inference_compare.py `
  --model outputs/zh_28m_pretrain_v3/phase2_100m/final_model `
  --device cuda --dtype bfloat16 --batch-size 1 `
  --prompt-tokens 128 --decode-tokens 32 --repeats 7 --hf-reference

python tools/benchmark_inference_compare.py `
  --model outputs/zh_28m_pretrain_v3/phase2_100m/final_model `
  --device cuda --dtype float32 --batch-size 4 `
  --prompt-tokens 128 --decode-tokens 32 --repeats 7 --hf-reference
```

不加 `--hf-reference` 可比较原生内部路径。工具同时驻留多份模型，适合能容纳
参考副本的测试模型；不是大模型低显存加载方案。`--output PATH.json` 可保存完整样本、
权重/输入哈希、精度、误差与资源统计。token 不一致会非零退出，不隐藏失败记录。

## vLLM/SGLang：提供入口，但没有伪造实测

本机未安装真实 vLLM/SGLang GPU 运行时，不能给出本机比较成绩。
SGLang 已接通的是可选采样后端，协议/lifecycle 测试通过不等于 GPU 性能验收。
原生 Saddle 检查点不能直接假装成外部引擎可加载的 HF 模型。

[服务比较工具](../tools/benchmark_serving.py) 可以对已启动的 OpenAI-compatible
`/v1/completions` 服务使用完全相同的文本、预算与并发，测客户端总响应 P50/P95、
实际输出 tokens/s 和 requests/s，并核对 token 计数与输出文本。缺 usage 时拒绝估算。
目前是非流式比较，不测 TTFT；不要把它的总响应时间叫首 token 延迟。

先在目标 GPU 环境部署相同的受支持 HF 权重与 tokenizer，统一 model alias，记录
精度/量化、硬件、版本、prefix-cache 策略、并发与 EOS 行为，再运行：

```powershell
python tools/benchmark_serving.py `
  --target saddle=http://127.0.0.1:8000/v1 `
  --target sglang=http://127.0.0.1:30000/v1 `
  --target vllm=http://127.0.0.1:8001/v1 `
  --model same-model --requests 16 --concurrency 4 --max-tokens 32 --repeats 5
```

工具不会自动启动服务、下载/转换权重、测远端 GPU 或核验远端权重。
`--checkpoint-sha256` 是操作者提供的追溯信息，不是已远端验证的事实。
默认重复 prompt 可能受前缀缓存影响，缓存策略必须明确记录，不能混淆热/冷请求。
不同服务常驻同一 GPU 也会影响显存和调度，正式验收应控制部署与背景负载。
同时还应测真实提示词、混合长度、采样、流式 TTFT、任务质量与取消/失败恢复。

<a id="vllm-public-comparison"></a>

## vLLM 公开数据和能力对照

资料核对日期：2026-09-27。以下只引用官方文章和文档，发布日期不等于本机测试日期。
没有找到可核验的“相同 28M 检查点、相同笔记本 GPU、相同请求”的 vLLM 成绩。
不能把别人的 7B/20B 模型或服务器 GPU 吞吐，除以我们的 28M 成绩计算胜负。

### 公开成绩说明了什么

| 公开测试 | 原文条件 | 原文报告的效果 | 对我们有用的结论 |
| --- | --- | --- | --- |
| 2023-06-20，vLLM 首发测试 | LLaMA-7B / A10G；LLaMA-13B / A100 40GB；请求长度取自 ShareGPT；每请求一份输出 | 服务吞吐相对当时 HF 为 14–24 倍，相对 TGI 为 2.2–2.5 倍 | 高并发服务的调度与 KV 内存管理可以比单个算子优化更重要；不是“任意模型单请求快 24 倍”。[原文](https://vllm.ai/blog/2023-06-20-vllm) |
| 2025-11-11，Intel vLLM 测试 | GPT-OSS-20B，MXFP4，B60 八卡测试机上的 TP=1 配置，输入/输出 1024/1024，并发 75 | 聚合输出 1210.74 tokens/s；TTFT 7.614 s；TPOT 53.96 ms | 高聚合吞吐和较长首 token 等待可以同时出现；不能只挑 tokens/s 判断用户体验。[原文表 1](https://vllm.ai/blog/2025-11-11-intel-arc-pro-b) |
| 2026-01-08，KV CPU offloading 测试 | Llama-3.1-8B-Instruct / H100；已有 CPU KV 缓存的加载与重算对照；另测 10000 个 512-token prefill 请求的缓存命中率 | 首 token 延迟随 prompt 长度降低约 2–22 倍；该 prefill 吞吐最高约 9 倍 | 收益依赖已有缓存及命中率，不能解释成新 token 解码通用加速 9 倍。[原文](https://vllm.ai/blog/2026-01-08-kv-offloading-connector) |

第一行是历史基线，不是当前版本 vLLM 与当前 Transformers 的必然差距；原文该段未完整标明
所有版本、精度与并发参数，不补猜测。第二行表格未逐项绑定精确软件版本和启动参数；
文中另给出 `0.10.2-xpu`、`--enforce-eager` 等部署示例，不能据此认定该成绩来自 CUDA Graph。
第三行使用 H100 80GB、Flash Attention，关闭 GPU prefix caching 以隔离 CPU 缓存命中，
且不计分词/反分词；它包含 prefill 微基准，不是完整聊天服务压测。
这些都属于作者公开报告，本项目没有独立复跑确认。

### 与 SaddleLLM 已实现代码逐项比较

以下“尚未实现”指本项目原生引擎，不是否定外部后端本身的能力。

| 能力 | SaddleLLM 当前实现 | vLLM 的实现方向 | 差距如何理解 |
| --- | --- | --- | --- |
| KV 存储 | 请求独立、连续存储、几何扩容；避免逐 token 拼接历史 | 分页块、块表、按需分配和回收 | 我们解决重复复制；分页还解决多请求内存利用与共享，不能称为同一种缓存。 |
| 请求合批 | 同长度、同预算的原生非流式贪心固定微批；运行中不插入新请求 | 每步重新调度，新请求可加入，完成请求退出 | 高并发、混合长度和流式场景是当前最主要的能力缺口。 |
| 长 prompt 调度 | 一次完成 prompt 前向；仅减少最后位置的词表投影 | 分块 prefill，与 decode 请求按 token 预算调度 | 最后位置 logits 优化不等于分块 prefill；长请求的公平性尚需实现。 |
| 跨请求前缀复用 | 可选单请求 dense 快照；默认关闭，启用后默认仅完整命中，部分命中需额外开启；KV 复制到私有请求 | 对相同 token 前缀复用 KV | 已有初始能力，但不是分页共享；本机部分复用更慢，收益必须看目标负载，不直接加快答案 decode。 |
| 计算执行 | PyTorch Linear/SDPA；投影合并、紧凑 GQA、恒等残差快速路径 | 专用执行路径、算子优化、编译与 CUDA Graph 捕获/重放 | 我们已有减少工作量的优化，但尚未具备图重放；硬件和形状不兼容时仍须回退。 |
| 使用入口 | CLI/import，训练、评估与原生服务复用项目配置 | 也有 Python API 和命令行服务 | “命令简单”并非独占优势；我们的定位是训练到部署的统一入口和原生模型兼容。 |

vLLM 调度与 prefill 的依据是其 [V1 引擎解析](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
（文章基于 2025-08-09 的代码，不据此推断当前每个默认参数）。前缀缓存的作用和限制见
[当前官方前缀缓存文档](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)；
图捕获、形状分派及不兼容回退见
[当前官方 CUDA Graph 文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/)。

项目侧可追踪的代码：

- [InferenceCache.py](../saddlellm/models/InferenceCache.py)：每个请求持有 KV、token/mask 缓冲区。
- [NativeGeneration.py](../saddlellm/models/NativeGeneration.py)：prefill/step、采样、停止和会话生命周期。
- [NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py)：有预算的跨请求快照、失效、淘汰和私有恢复。
- [InferenceBatcher.py](../saddlellm/runtime/InferenceBatcher.py)：有限队列、固定兼容批次、取消与许可持有。
- [InferenceOptimizer.py](../saddlellm/models/InferenceOptimizer.py)：eval 模型投影合并及兼容性保护。
- [OperatorBackends.py](../saddlellm/models/OperatorBackends.py)：SDPA 路径与紧凑 GQA 选择。

### 我们是否有优势

现在能证明的是上面的同权重测试：FP32 固定 B=4 时，相对 HF SDPA 约 +24.0% 吞吐，
完整生成 token 一致。优势适用于这组模型与负载，不能推广为所有模型都更快。
此外，当前原生路径不要求安装 vLLM/SGLang，适合本机验证、原生检查点和低并发使用。
这不是声称安装简单就代表生产性能更强。

基于公开架构与本项目代码，推断 vLLM 在大模型、多请求、混合长度等场景具备更完整的
调度与内存管理基础；当前不能声称我们已经追平或超过它。实际差距多少仍须远端同条件测试。
接入外部引擎不意味着必须让普通用户维护两套系统：本地默认路径不变，外部 GPU 后端按需选择。
目前 `backends.rollout: sglang` 是采样适配器，不是 `saddlellm serve` 的通用推理后端；
此前尚无 `serve --backend vllm`；2.37 后已实现，见上述后端指南。
2.38 的 `--backend native` 扩展到已支持的普通 Qwen3，但不等于自研了 vLLM 的连续调度/分页 KV。

### 后续实施优先级（未完成的部分仍是规划）

2026-09-27 后续已落地窗口环形 KV、可选长度分桶与 token 预算调度，详见
[新实现与独立测量](INFERENCE_WINDOW_AND_SCHEDULING.md)。这组随机混合模型测试不与前文
28M 训练权重成绩拼接比较；生产混合请求的排队/TTFT/P99 测量仍未完成。

1. **先完善测量**：增加流式 TTFT/TPOT、排队时间、混合长度与冷/热前缀负载，同时检查
   输出、取消和显存上限。现有服务比较脚本只有非流式总响应时间，不能冒充这些指标。
2. **本机低并发**：继续 profiling RMSNorm、RoPE、采样和 CPU 调度；确认瓶颈后再试固定形状
   编译/图复用，并保留无新增依赖的普通路径。不把单个微基准收益直接承诺为整模型加速。
3. **重复长前缀**：有预算、淘汰和模型状态隔离的初始快照已实现，下一步基于真实负载
   改善状态校验/复制成本和适用范围。仅相同 token 前缀可复用，内容相似不等于共享。
   当前 28M context=512，不能凭缓存功能宣传长上下文模型能力。
4. **高并发生产服务**：优先远端成熟引擎适配和真实负载验收，再决定自研连续调度的范围；
   不为当前 Windows 用户强制安装 Linux 推理栈，也不先重写一整套 C++ 引擎。

对外可以准确表述为：SaddleLLM 已具备实测有效的原生推理优化，用户接口保持简单；
高并发调度能力仍在建设中，公开 vLLM 成绩用于确定方向，不作为本项目胜负宣传。

## 要不要单独写 C/C++ 推理引擎

现阶段不需要先重写一套 C++ 引擎。Python 管理请求、KV、停止规则、版本和资源；
Linear/SDPA 等计算由 PyTorch 的 C++/CUDA 路径执行，不是 Python 逐元素计算。
[PyTorch 2.6 SDPA 官方说明](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)
列出了 C++、Memory-Efficient Attention、FlashAttention 的执行与自动选择边界。
代码中的 `attention_backend=flash` 不保证本机实际用了 Flash 内核。

本轮没有新增 C++/CUDA 源码，不增加编译器或 Triton 安装要求。先减少不必要的工作，
对真实模型 profiling 再决定是否有必要写小型专用算子。例如融合 RMSNorm、RoPE 或
采样可以作为候选，但不能只因“C++ 听起来更专业”就承诺速度。
若确实需要，采用可选的算子扩展、正确性测试与稳定的 PyTorch 路径，用户入口不变；
不让普通使用者承担整套引擎开发或编译维护成本。

## 下一步怎么选

- 单请求解码：基于 profiling 继续减少小算子启动/CPU 调度，目标环境验证固定形状
  编译或图复用；动态 KV、取消和缓冲区生命周期必须同时正确，不先承诺 CUDA Graph 已实现。
- 高并发：优先完成 SGLang 真 GPU 联调及混合长度/流式压测，借用成熟调度和缓存能力。
- 轻量化：量化以质量、显存和真实延迟共同验收；位数变少不必然在这台小模型/Windows
  环境更快，不与本轮无损布局优化混为一谈。

合理定位是“简洁入口、原生可用、后端可替换的训练与推理框架”，而不是未经大模型
高并发实测就宣称“已经重造并超过 vLLM”。

## 本轮回归验收

2026-09-27，前一轮原生推理优化运行以下选定回归，487 项通过（含本机 CUDA 的 GQA 和前缀缓存
FP32/FP16/BF16 检查，以及微型原生 MLA/MoE/MTP 训练与 scratch 恢复验证）。
这是推理、服务、原生模型/训练兼容与可选后端范围，不是全仓库测试或模型能力评测。
Megatron/SGLang 使用协议/运行时测试替身，仍没有真实 GPU 后端验收成绩。

```powershell
python -m pytest -q `
  tests/test_generation_sampling.py tests/test_generation_stop.py `
  tests/test_native_generation_session.py tests/test_native_streaming.py `
  tests/test_streaming_text.py tests/test_inference_batching.py `
  tests/test_generation_fastpaths.py tests/test_inference_cache.py `
  tests/test_native_model_lifecycle.py tests/test_release_pipeline.py `
  tests/test_model_evaluation.py tests/test_completion_evaluation.py `
  tests/test_model_network_optimizations.py tests/test_model_memory_fastpaths.py `
  tests/test_chunked_lm_loss.py tests/test_rope_cache_growth.py `
  tests/test_operator_backends.py tests/test_cli_output.py tests/test_webui.py `
  tests/test_public_api.py tests/test_packed_projection_checkpoint_edges.py `
  tests/test_runtime_backends.py tests/test_inference_optimizer.py `
  tests/test_serving_benchmark.py tests/test_gqa_decode_folding.py `
  tests/test_native_prefix_cache.py tests/test_deepseek_style_training.py `
  tests/test_model_composition.py tests/test_model_distributed_runtime.py `
  tests/test_scratch_pretrain_e2e.py
```

新增文件严格通过 E/F/I lint，已有修改文件通过语法/致命问题检查。
配置示例的 validate-config 与 dry-run 已验证，不自动联网更新权重或启动 GPU worker。

后续混合注意力/AttnRes 与使用流程排错的最新组合验收为 673 项通过，详见
[修复清单与复现命令](BUGFIX_VALIDATION_20260927.md)。该轮没有重新发布吞吐比较成绩。

## 2.38：外部 Qwen3 也可进入 Saddle 自有后端

`--backend native` / `load_native_causal_lm()` 已实现标准 dense Qwen3 safetensors
到 Saddle 前向、增量 KV、生成会话和服务路径的严格加载；默认 HF 路径没有切换。
本轮 34.6M 未训练 fixture 的同权重 CUDA FP32 测量，批量 1 为 HF 156.64 / native 204.45
输出 token/s，批量 4 为 670.06 / 890.72；两项同前缀误差和生成 ID 检查通过。
不是原来的 28M 训练模型，也不是 8B 或 HTTP 吞吐成绩。

真实 Qwen3-8B 已在 Linux CPU FP32 eager 跑通 8 步同权重校验与生成，最大 logits 差
7.6294e-6；单次单核心速率约 HF 0.657 / native 0.654 token/s，基本持平，没有加速结论。
BF16 SDPA 在 34.6M fixture 上仍未通过 logits 门槛，失败报告保留；原生后端因此保持实验性。
本轮相关回归 521 项通过，完整 8B GPU 吞吐和质量仍待独立验收。
完整命令、支持范围、代码流程、最终 wheel 和原始证据见
[Qwen3 原生后端指南](NATIVE_QWEN3_BACKEND.md)。
