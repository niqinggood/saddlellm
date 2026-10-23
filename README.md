# SaddleLLM

当前版本：`2.42`

SaddleLLM 是面向大模型训练、评测与推理部署的工具。对外保持简单的 CLI / Python import，底层按后端与模型分别验证。

第一次使用请从 [安装与三条上手路线](docs/GETTING_STARTED.md) 开始：
[部署聊天](docs/quickstart/SERVE.md) · [LoRA 微调](docs/quickstart/TRAIN.md) · [评测后部署](docs/quickstart/EVALUATE_DEPLOY.md)。
使用范围见 [平台/模型矩阵](docs/SUPPORTED_ENVIRONMENTS.md)，遇到问题见 [排错指南](docs/TROUBLESHOOTING.md)。
维护者交付前按 [发布验收清单](docs/RELEASE_CHECKLIST.md) 检查；2.40 本地候选包尚未因此自动发布到 PyPI。

`doctor --profile serve|native|vllm|train` 按用途自检；CPU 可以用于小模型验证，显存按驱动逐卡查询。
`serve --no-chat-history` / Python `serve(..., chat_history=False)` 关闭内置聊天历史的读写，旧记录不删除。
默认历史属于共享工作区，不是多租户隔离。详细变更与验收见 [本轮发布准备](docs/RELEASE_READINESS_20260929.md)。

## 开发进展与能力边界

新增 [研究项目命令行入口](docs/EXPERIMENT_PROJECTS.md)：`saddlellm experiment --project <目录>` 统一启动任务项目的自检、测试、后训练对比与分析报告。相邻公共政策项目提供 Qwen3-0.6B 的 SFT/DPO/GRPO/SFT+GRPO 对照，保留逐例预测、训练问题与负面结果。

新增 [推理性能预设与实测](docs/INFERENCE_SPEED_20261002.md)：`serve --performance latency|throughput` 简化重复完整长提示缓存和原生流式合批配置，Web/API 调用不变；按模型分别记录收益及未通过的候选。[远程 L20 部署比较](docs/REMOTE_INFERENCE_COMPARISON_20261002.md) 单独记录原生、Transformers 与 vLLM 的表现。

想快速理解“过去有什么问题、做了哪些优化、现在能做到什么、后续如何推进”，先看 [推理优化复盘与后续路线](docs/INFERENCE_OPTIMIZATION_REVIEW_AND_ROADMAP.md)。材料区分默认能力、显式选项、实验路径与未完成项，不混算 28M 和 8B 的成绩。

后续推理开发可参考 [DeepInfer 学习笔记与开发计划](docs/DEEPINFER_LEARNING_NOTES_20261022.md)：固定上游源码快照，对照现有实现，记录 CUDA Graph、分页 KV、流水调度和增量投机解码的学习入口、开发依赖及验收口径。当前为学习与规划，开发待后续触发。

本地开发更新：原生浮点后端新增受限的 Qwen2/Llama 适配，真实 DeepSeek-R1-Distill-Qwen-1.5B 已进入原生生成验证。
国内完整版 DeepSeek/Kimi/MiniMax 不等同于蒸馏版支持。范围见 [多模型原生推理](docs/NATIVE_MULTI_MODEL.md)；
推理优化按 [正确性与性能验证方案](docs/INFERENCE_VALIDATION_PLAN.md) 做同模型交替 A/B，未通过的候选不宣传为加速成果。

新增 [非流式断连取消与实测记录](docs/INFERENCE_LIFECYCLE_20260929.md)：减少客户端离开后的无效生成，保护其他请求；命令不变。微批控制的试验性提速未达整模型门槛，未默认启用。
新增 [原生合批解码执行器与实测](docs/NATIVE_BATCH_DECODE_20260929.md)：实验性 import API，复用合组 KV、合并真实模型前向。[BF16 兼容修复](docs/NATIVE_BF16_BATCH_FIX_20260929.md) 解决所测 28M 的 Linear 形状相关分叉并降低合组临时显存。后续已接入 [原生 SSE 流式合批](docs/NATIVE_STREAM_BATCHING_20260929.md)：`serve --backend native --native-stream-batching --max-concurrency 4 --batch-size 4` 显式启用，默认关闭；含单请求直通、稳定 KV 行复用、有限步数对齐和断连清理。最新支持 [默认采样、混合长度合批与显式 BF16 吞吐模式](docs/NATIVE_SAMPLING_THROUGHPUT_20260929.md)：`--native-decode-mode throughput` 可使用真正批量 Linear，并明确可能的数值/输出差异。真实 28M HTTP 测量与失败案例均有记录，不外推到 8B 或任意长度负载。

[最后一轮调度修复](docs/NATIVE_FINAL_DISPATCH_20260930.md)修正繁忙执行期间收集窗口过期导致的持续拆批，保留修改前后 SSE 对照、慢客户端与取消回归。

[远程 L20 / Qwen3-4B 验证](docs/REMOTE_QWEN3_4B_SSE_VALIDATION_20260929.md)：BF16、四客户端、48-token 上限，相对原生串行许可配置总输出吞吐约 50.54 → 70.13 token/s（约 +39%）；相对原四并发配置约 1.9 倍，文本和 token 数一致。单请求未通过性能提升门槛，混合负载只有同长度子组能够合批；不是 vLLM/Ollama 胜负或 8B 结论。
后续已修复所测 DeepSeek 1.5B 的 SDPA 输出分叉；新增线程池和逐 token 调度未通过性能验收，保持关闭。
实现、调度候选的取舍与真实验证见 [原生服务迭代记录](docs/NATIVE_SERVING_ITERATION_20260928.md)。

`saddlellm serve MODEL` 默认自动选择部署后端：兼容的 Linux CUDA + vLLM 环境走异步连续批处理，
其他环境明确提示使用 local；显式 `--backend local/native/vllm` 保留。
`--require-high-throughput` 禁止降级到普通本地服务，`--dry-run` 可先检查路由而不加载/下载权重。
同一入口提供 `/v1/models`、`/v1/chat/completions`、`/v1/completions` 及 SSE/真实 token usage。
默认保留模型 chat template 的思考设置、不自动量化、不静默截断长输入。
兼容的是文本 API 子集，不含工具调用、多模态或 Responses API；未知参数明确拒绝。

```bash
# Linux CUDA 独立环境，在源码目录安装（2.40 本轮未发布 PyPI）
python -m pip install '.[serve-vllm]'
saddlellm serve /models/Qwen3-8B --require-high-throughput --model-name qwen3-8b
```

客户端把 OpenAI SDK 的 `base_url` 设为 `http://127.0.0.1:8000/v1` 即可。
Python 部署支持 `from saddlellm import serve; serve(path)`，CUDA 脚本需 main guard。
安装选择、完整调用、参数效果、质量/吞吐验收和源码链路见
[一条命令部署与 OpenAI 接口手册](docs/OPENAI_SERVING.md)。

下面完整 8B 的性能数字是 **2.39 冻结版**的已保留实验，不是 2.40 的新并发 HTTP 吞吐成绩。

新增实验性自有 W8A16 推理路线：`saddlellm serve /models/Qwen3-8B --backend native --device cuda --quantization int8`。
隐藏层权重在 CPU 上量化后才进入 GPU，激活/词嵌入/输出层/归一化保留浮点精度。
采用小型 Triton GPU 算子，不需要 bitsandbytes 或 C++ 扩展构建；首次运行会 JIT 编译。
完整 Qwen3-8B、L20、BF16 激活/eager、固定输入 128/输出 64、单请求复测：
同量化 Saddle 未合并投影 29.45 → 合并后 34.84 输出 token/s，提升 18.3%。
这是固定随机 token 提示的模型调用计时，不是 HTTP 吞吐或普通 BF16/vLLM 对比；
小样本困惑度约上升 0.82%，不代表模型更聪明或质量无损。2.39 当时默认精度与默认 `local` 后端均不变；
2.40 改为上文的自动部署路由，INT8 仍需显式选择。
命令、Python import、实现链路、原始记录及资源限制见
[完整 8B W8A16 推理指南](docs/NATIVE_INT8_8B.md)。

新增实验性 FP32 CPU 解码权重预排布：原生 CPU 服务加 `--cpu-prepack`，或调用
`optimize_native_inference(model, cpu_prepack=True)`。使用 PyTorch 内置 oneDNN，
不改变权重精度，不需要额外编译；占用额外 RAM，保持默认关闭。
BF16 的真实 8B 对照未通过数值/输出验收，已禁用其预排布路径；不宣称此功能改善 8B 吞吐。实测记录见
[Qwen3 原生后端指南](docs/NATIVE_QWEN3_BACKEND.md)。

新增实验性的自有 Qwen3 推理后端：`saddlellm serve /models/Qwen3-8B --backend native --dtype bfloat16`。
直接读取标准非量化 safetensors，使用 Saddle 的模型前向、KV 和生成循环，不调用 HF `model.generate()`。
显式 `local` 仍保持兼容；范围、Python import 与真实验证见 [Qwen3 原生后端指南](docs/NATIVE_QWEN3_BACKEND.md)。
BF16 SDPA 已发现数值差异，不能承诺逐 token 相同；共享 GPU 上加载额外 8B 前必须先核对显存预算。
真实 Qwen3-0.6B BF16 服务对照还发现，合并 Q/K/V 投影会改变部分贪心回答；
因此原生 BF16 Qwen3 现默认只合并 gate/up，保留 Q/K/V 独立计算。此策略缩小了已观察到的输出偏差，
但不保证与 HF 逐 token 一致；`--no-fuse-projections` 可关闭全部投影合并。

新增可选异步推理后端：Linux CUDA 独立环境安装 `saddlellm[serve-vllm]`，
使用 `saddlellm serve Qwen/Qwen3-8B --backend vllm`；2.40 的 auto 已可自动选中兼容环境的这条路线。
配置、Python import、取消/预算和验证边界见 [vLLM 服务指南](docs/VLLM_SERVING_BACKEND.md)。

SaddleLLM 是一个配置驱动、与行业无关的大模型全生命周期平台。用户准备数据、模型和一份训练配置，系统负责检查输入、安排训练阶段、调用对应训练器，并保存 checkpoint、评测结果、发布产物和运行状态。医疗、护理、教育等是外部领域项目，通过数据、配置或插件接入，不能成为核心包中的硬编码分支。

平台能力边界、成熟度和开发优先级见 [通用大模型平台边界](docs/GENERAL_LLM_PLATFORM.md)；完整命令与参数见 [简明命令手册](docs/CLI_COMMAND_GUIDE.md)；本轮主链代码审查及后续缺口见 [代码完善审查](docs/CODE_IMPROVEMENT_REVIEW.md)。

自有结构化决策模型已接入统一命令。`saddlellm train configs/decision_choice.yaml` 保留冻结骨干的 Choice 评分头基线；`saddlellm train configs/decision_typed.yaml` 训练 Choice、Score、Noul 多类型决策网络，支持骨干联合训练和实验性 Brier 奖励策略梯度。两种模型都通过 `saddlellm serve-decision MODEL_DIRECTORY` 提供 `/v1/decisions` 数值接口；多类型模型将训练后的骨干一起保存。L20 已验证 Qwen3-Embedding-0.6B 全骨干更新、保存重载和真实 HTTP 调用。当前测试使用合成数据，未证明通用决策能力，也不声称复现原版 RLCD。数据契约、训练命令和实验结果见 [决策模型与 RLCD 笔记](docs/DECISION_MODEL_NOTES.md)。

中长期目标是让使用者通过简单的 CLI/import，训练自己的高水平大模型，而不只调用已有模型。
这是研发目标，不是当前模型已达到头部水平的声明。DeepSeek 风格、Kimi K3 和 Jev 决策路线的
实际基础、缺口与分阶段验收见 [头部模型研发路线](docs/FRONTIER_MODEL_ROADMAP.md)。

本文只讲当前实现，按三个问题展开：

1. 这个系统是什么。
2. 怎么使用，命令会返回什么。
3. 代码和训练过程为什么这样运行。

## 1. 这个系统是什么

### 1.1 系统定位

SaddleLLM 不是一个单独的模型，也不是重新实现 PyTorch、Transformers、PEFT 或 TRL。它位于这些训练库的上层，把一次模型实验需要的内容组织起来：

```text
数据 + 基础模型 + 训练参数 + 执行阶段
                    ↓
              SaddleLLM
                    ↓
训练计划 + checkpoint + 评测 + 发布文件 + 运行记录
```

它主要解决五件事：

1. 用一份 YAML/JSON 明确本次运行要做哪些阶段。
2. 在加载大模型前检查配置、数据和运行环境。
3. 按顺序调用预训练、SFT、偏好训练、评测和导出等实现。
4. 把前一个阶段产生的模型或数据交给后一个阶段。
5. 保存计划、状态和结果，让失败可定位、运行可恢复、产物可追踪。

### 1.2 输入、处理和输出

| 类型 | 具体内容 |
|---|---|
| 输入 | YAML/JSON 配置、文本或多模态数据、基础模型或 checkpoint |
| 处理 | 数据检查、模型加载、阶段排序、训练、评测、导出 |
| 输出 | 阶段计划、训练 checkpoint、最终模型、指标、发布门禁、运行摘要 |

用于 `train`、`validate-config` 和 `plan` 的训练配置只有一种结构：顶层必须有非空的 `stages` 列表。

例如：

```yaml
stages: [sft, preference, eval]
```

这表示依次执行：

```text
监督微调 → 偏好训练 → 评测
```

每个阶段的参数放在同名配置块里：

```yaml
stages: [sft, preference]

sft:
  enabled: true
  data_path: data/sft.jsonl

preference:
  enabled: true
  method: dpo
  data_path: data/preference.jsonl
```

### 1.3 当前支持的能力

| 领域 | 阶段或入口 | 当前实际行为 |
|---|---|---|
| Tokenizer | `tokenizer` | 训练新 tokenizer；未提供训练参数时加载已有 tokenizer |
| 从零预训练 | `pretrain` + `pretrain_mode: scratch` | 创建模型、处理语料、执行 causal LM 训练并保存最终模型 |
| 继续预训练 | `pretrain` + `pretrain_mode: continue` | 加载已有权重，用新训练任务继续优化 |
| 监督微调 | `sft` | 执行全参数、LoRA 或 QLoRA SFT |
| 偏好训练 | `preference` | 执行 DPO、ORPO 或 KTO |
| RLHF | `rlhf` | DPO 可执行；PPO dry-run可规划，实际训练开始前明确阻断 |
| 多教师蒸馏 | `mopd` | 生成计划，或调用教师收集 on-policy 数据供后续 SFT |
| VLA | `vla_sft` | 检查并规范化机器人轨迹；`vla.train: true` 时执行行为克隆训练 |
| 通用图文训练 | `mllm_sft`、`vision_alignment` | 当前负责数据规范化和训练计划，还没有执行通用 VLM 参数优化 |
| 图像生成 | `media_cache` + `image_generation` | 编码图像与文本条件，训练潜变量生成模型 |
| 音乐生成 | `media_cache` + `music_generation` | 编码音频码本与文本条件，训练多码本自回归模型 |
| 视频生成 | `media_cache` + `video_generation` | 编码视频潜变量与文本条件，训练时空生成模型 |
| 世界模型 | `world_model` | 用离线轨迹训练 RSSM 或 categorical RSSM |
| 眼动控制 | `eye_control` | 在安全软件仿真器中评估 PID 或世界模型规划控制 |
| 评测 | `eval` | 计算指标，并可按规则生成发布门禁结果 |
| 导出 | `export` | 打包 HF 或 Saddle 模型；可要求评测门禁先通过 |
| 外部算子 | `operator` | 调用已配置的 agent/大模型算子并登记产物 |

### 1.4 两种模型后端

`model.backend` 决定文本模型由哪套实现负责：

| 后端 | 用途 | 当前边界 |
|---|---|---|
| `hf` | Hugging Face 模型加载、SFT、LoRA、QLoRA、DPO、ORPO、KTO | 最适合已有开源模型的后训练 |
| `saddle` | 项目原生模块化语言模型、原生 checkpoint、预训练和 DPO | 不支持 LoRA/QLoRA；偏好方法当前只支持 DPO |

并行能力按阶段和后端显式开放：原生 `saddle` 预训练可选择 DDP、FSDP 或 DeepSpeed；SFT 和偏好训练目前只开放单机 DDP，且仍属于实验能力。完整边界见下文“自动并行训练”。

原生模型新增可配置的 **KDA + GQA/MLA 混合注意力**和 **Full/Block Attention Residuals**。
Python 使用 `ModelBlueprintLab.hybrid_kda(attention_residuals="block")` 后调用 `build_model()`；
CLI 可运行 `saddlellm train .\configs\hybrid_kda_attnres_scratch.yaml`（先准备自己的语料）。
训练、梯度检查点、保存重载和请求私有递归缓存已接通；默认旧模型仍使用普通残差。
KDA 当前是顺序 PyTorch 参考路径，**不是生产级加速内核或 Kimi 权重复现**。
参数、完整示例、源码流程与验证边界见 [混合注意力与 AttnRes 指南](docs/HYBRID_ATTENTION_AND_ATTNRES.md)。
AMP、打包缓存、dry-run 与配置误用等问题的修复清单及 673 项选定回归，见
[使用流程排错与验收](docs/BUGFIX_VALIDATION_20260927.md)（不是全仓库或模型质量验收）。

### 1.5 代码目录

| 路径 | 作用 |
|---|---|
| `saddlellm/cli.py` | 命令行参数和命令分发 |
| `saddlellm/training/TrainingOrchestrator.py` | 解析配置、校验并执行阶段 |
| `saddlellm/framework/stages.py` | 阶段注册表和阶段能力声明 |
| `saddlellm/training/` | 预训练、SFT、偏好训练和分布式运行 |
| `saddlellm/data/` | 数据采集、清洗、规范化和检查 |
| `saddlellm/models/` | 模型结构、构建、加载和 tokenizer |
| `saddlellm/multimodal/` | 图像、音频、视频、VLM 和 VLA |
| `saddlellm/world_models/` | RSSM 数据、模型、训练和推理 |
| `saddlellm/spatial/` | 空间感知、路径规划、眼动控制和 WorldAgent |
| `saddlellm/evaluation/` | 评测、冒烟测试和发布门禁 |
| `saddlellm/runtime/` | 模型导出、服务和部署 |
| `configs/` | 可以直接参考的配置文件 |
| `tests/` | 行为和接口测试 |

## 2. 怎么使用、命令做什么、返回什么

### 2.1 安装

建议使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[posttrain,dev]"
```

按功能安装可选依赖：

| 需求 | 安装命令 |
|---|---|
| 通用训练 | `python -m pip install -e ".[train]"` |
| SFT、PEFT、TRL | `python -m pip install -e ".[posttrain]"` |
| QLoRA | `python -m pip install -e ".[posttrain,qlora]"` |
| 推理 API | `python -m pip install -e ".[serve]"`；运行时4/8bit量化再加`qlora`：`.[serve,qlora]` |
| 空间智能与眼动控制 | `python -m pip install -e ".[spatial]"` |
| 开发和测试 | `python -m pip install -e ".[dev]"` |

如果要严格使用仓库验证过的后训练版本：

```powershell
python -m pip install -r requirements-posttrain.txt
```

安装完成后确认命令可见：

```powershell
saddlellm --help
```

也可以不用 console script：

```powershell
python -m saddlellm.cli --help
```

### 2.2 第一次运行

创建一个 10 步 LoRA SFT 小项目：

```powershell
saddlellm quickstart ".\my-training"
```

生成的目录：

```text
my-training/
├─ config.yaml
├─ README.md
└─ data/
   └─ sft.jsonl
```

命令返回 JSON，主要字段如下：

```json
{
  "workspace": "绝对路径/my-training",
  "config": "绝对路径/my-training/config.yaml",
  "stages": ["sft"],
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "data_files": {"sft": "绝对路径/my-training/data/sft.jsonl"},
  "created_files": ["..."],
  "next_steps": ["..."],
  "note": "..."
}
```

进入目录后依次执行：

```powershell
Set-Location ".\my-training"
saddlellm doctor
saddlellm inspect-data data/sft.jsonl --task sft
saddlellm validate-config config.yaml
saddlellm train config.yaml --dry-run
saddlellm train config.yaml
```

默认生成的阶段只有：
  stages:
    - sft

  而且默认采用 LoRA SFT，最多训练 10 步。注意，quickstart 只是生成配置和示例数据，不会立即训练。
  quickstart 目前只支持两种阶段组合：
  ### 仅 SFT（默认）
  saddlellm quickstart ".\my-training"
  ### SFT 后接偏好优化
  saddlellm quickstart ".\my-training" --stages sft,preference --preference-method dpo
  其中偏好优化方法可以是 dpo 或 kto。它不支持通过 quickstart 直接创建 pretrain、scratch 等预训练配置；从零预训练需要使用
  其他训练配置或 train 命令。

基础模型 → SFT（学会回答）→ 偏好优化（学会回答得更好）

  ### SFT 做什么
  SFT 数据通常是标准答案：
  {
    "instruction": "如何学习 Python？",
    "output": "先学习基础语法，再通过项目练习。"
  }
  它让模型模仿目标回答。

  ### DPO 是什么
  DPO（Direct Preference Optimization，直接偏好优化）需要一组成对答案：
  {
    "prompt": "如何学习 Python？",
    "chosen": "先学习基础语法，再通过项目练习。",
    "rejected": "随便看看就行，不用练习。"
  }
  含义是：
  - chosen：更好的回答
  - rejected：较差的回答
  - 模型学习提高 chosen 的概率，降低 rejected 的概率
  DPO 适合你能为同一个问题准备“好答案和坏答案”的场景。它不需要先单独训练一个奖励模型。
  ### KTO 是什么
  KTO（Kahneman-Tversky Optimization）不要求好坏回答严格成对，只需要给每个回答打一个标签：
  {
    "prompt": "如何学习 Python？",
    "completion": "先学习基础语法，再通过项目练习。",
    "label": true
  }
  或者：
  {
    "prompt": "如何学习 Python？",
    "completion": "随便看看就行。",
    "label": false
  }

  其中：
  - true：满意、可接受
  - false：不满意、不可接受
  KTO 更适合只有点赞/点踩、通过/拒绝之类的非成对反馈数据。
  KTO 学习“这个回答相对于正常水平是好还是坏”。
  ### 怎么选择
   数据情况                               建议
   每个问题都有一好一坏两个答案           DPO
   回答之间没有严格配对，只有好/坏标签    KTO
   只有问题和标准答案                     先只做 SFT

  所以这条命令：

  saddlellm quickstart ".\my-training" `
    --stages sft,preference `
    --preference-method dpo

  表示生成一个 先 SFT、再 DPO 的训练项目。通常初次尝试只运行 SFT 就够了；只有准备了可靠的偏好数据，才有必要增加 DPO 或
  KTO。

正式训练有独立 validation/test 时，一次完成 schema 与跨集合泄漏检查：

```powershell
saddlellm inspect-data data/train.jsonl --task sft `
  --validation-data data/validation.jsonl `
  --test-data data/test.jsonl `
  --group-field case_id --strict-leakage `
  --output outputs/split_audit.json
```

同一 group、规范化后的同题会阻断；高相似改写默认警告，`--strict-leakage` 下也会阻断。
报告默认对 group id 做哈希且不复制题目原文；只有在确认数据可安全展示时才使用
`--include-content-previews`。当配置含独立 `validation_data_path` 时，`validate-config` 和正式
训练前置检查也会自动执行训练/验证泄漏审计。

训练启动后，可在另一个终端查看进度、loss 曲线、阶段状态和日志：

```powershell
saddlellm dashboard outputs/my-sft --open
```

`dashboard` 只读取训练目录中的标准产物，不占用第二份模型显存，也不要求安装单独的前端工程。历史训练目录同样可以打开。

推理问答 Web 的侧栏包含“模型问答”和“SaddleLLM 介绍”。在同一服务的 `/saddle` 页面，可以了解数据检查、训练微调、评估、蒸馏、轻量化与部署，并复制命令示例；切换回来会保留当前聊天和输入草稿。

```powershell
saddlellm serve "D:\models\Qwen3-0.6B" --open
```

介绍页随 `serve` 的问答页面一起提供，无需另外启动服务。使用方式见 [WebUI 与功能介绍](docs/WEBUI_AND_SERVING.md)。

研究项目的历史实验另有可选的只读工作台：

```powershell
saddlellm workbench --project "E:\DEV\saddle-ml-llm\public_policy_rlvr" --study qwen3_06b_direct_seed42 --open
```

首版接入公共政策实验：原始数据与实际模型输入、历史自检与命令、训练记录、同题改对/改错和报告下载。它读取已有证据，不启动训练；新实验仍通过 `saddlellm experiment` 执行。使用与数据口径见 [WebUI 与实验工作台](docs/WEBUI_AND_SERVING.md)。

创建“SFT 后接 DPO”的项目：

```powershell
saddlellm quickstart ".\my-training" --stages sft,preference --preference-method dpo
```

使用自己的数据和模型：

```powershell
saddlellm quickstart ".\my-training" --model "D:\models\my-model" --data "D:\datasets\sft.jsonl"
```

`quickstart` 不会下载模型，也不会开始训练。它只创建配置、示例数据和下一步命令。目标文件已存在时默认拒绝覆盖；`--force` 只覆盖它负责生成的文件。

### 2.3 直接训练，或者使用配置文件

最常见的单阶段 SFT 可以直接写命令，不必先创建 YAML：

```powershell
saddlellm train sft `
  --model Qwen/Qwen3-0.6B `
  --data train.jsonl `
  --epochs 1 `
  --output-dir outputs/my-sft
```

完全等价的显式写法是：

```powershell
saddlellm train --stage sft `
  --model Qwen/Qwen3-0.6B `
  --data train.jsonl `
  --epochs 1 `
  --output-dir outputs/my-sft
```

直接训练默认使用 LoRA、有效 batch 由 `--batch-size 1` 和 `--gradient-accumulation-steps 4` 组成、最大长度 512，并保留 5% 数据作内部验证。常用覆盖参数：

```powershell
saddlellm train sft --model MODEL --data DATA `
  --method lora `
  --learning-rate 5e-5 `
  --batch-size 1 `
  --gradient-accumulation-steps 8 `
  --max-seq-length 384 `
  --max-steps 150 `
  --save-steps 50 `
  --eval-steps 50 `
  --precision bf16 `
  --local-files-only `
  --output-dir outputs/my-sft
```

正式训练建议准备独立验证文件，避免随机划分把同源病例或同一对话的不同轮次分到两边：

```powershell
saddlellm train sft --model MODEL --data train.jsonl `
  --validation-data validation.jsonl `
  --method lora --lora-target-modules all-linear `
  --save-steps 50 --eval-steps 50 `
  --output-dir outputs/my-sft
```

`--validation-data` 与 `--validation-split` 二选一。存在验证集时，SaddleLLM 按 `eval_loss`
记录并在训练结束恢复最佳 checkpoint；按 step 验证时，`--save-steps` 必须是
`--eval-steps` 的整数倍。结果中的 `best_model_checkpoint` 和 `best_metric` 是实际选择证据。

`--method` 支持 `lora`、`qlora` 和 `full`。`--lora-target-modules all-linear` 使用 PEFT 的
架构无关写法覆盖模型中适合 LoRA 的全部线性层，通常比只训练 Q/K/V/O 投影有更大的适配容量，
也会增加可训练参数、显存和耗时；仍可用逗号明确指定层。用 `--dry-run` 只显示解析后的完整配置
而不训练；实际训练时 SaddleLLM 会把等价配置保存为输出目录中的 `command_config.json`，保证
命令行训练仍可复现。

训练中断后，保持原输出目录、模型、数据、更新方法、batch 和并行规模不变，提高总
`--max-steps`，再加不带路径的 `--resume-from-checkpoint`，会自动选择最新 checkpoint：

```powershell
saddlellm train sft --model MODEL --data DATA `
  --method lora --batch-size 1 --max-steps 300 `
  --output-dir outputs/my-sft --resume-from-checkpoint
```

也可以在该参数后显式给出 `checkpoint-*` 路径。直接命令的精确续训一次只允许一个阶段；
多阶段恢复使用 YAML 的 `pipeline.resume`。SaddleLLM 会拒绝有效训练 batch 发生变化的“伪续训”。
结果同时记录 `resumed_from_checkpoint`、原始累计 `train_metrics`、不受续训步数稀释的
`latest_train_metrics`，并在存在内部验证集时保证写出最终 `eval_metrics`。

已有模型做领域继续预训练，也可以直接运行：

```powershell
saddlellm train pretrain `
  --model Qwen/Qwen3-0.6B `
  --data data/nursing_corpus.jsonl `
  --text-column text `
  --max-steps 1000 `
  --global-batch-size 32 `
  --output-dir outputs/nursing-cpt
```

这是 `pretrain_mode: continue`：网络结构和 tokenizer 从 `--model` 继承，数据每行采用
`{"text":"连续的护理教材或规范正文……"}`。从零创建网络时建议改用 YAML 的
`model.config + model.blueprint`，因为网络结构属于模型身份，应该完整保存和审计，而不是堆成一长串命令行参数。

不确定后训练数据该用哪种方法时，使用名称明确的 `auto-posttrain`（不是含义模糊的 `auto`）：

```powershell
saddlellm train --stage auto-posttrain `
  --model Qwen/Qwen3-0.6B `
  --data data/posttrain.jsonl `
  --dry-run
```

它先抽样并检查本地数据，再按数据契约选择：`instruction/output`、messages 或普通
`prompt/completion` → SFT；`prompt/chosen/rejected` → DPO；带显式二元 `label` 的
`prompt/completion` → KTO。结果会输出 `selected_stage`、选择理由、质量警告和完整检查报告。
PPO 永远不会被自动选择；混合格式、未知格式、只有单一正负标签的 KTO 数据会直接拒绝，
避免静默丢样本。确认 dry-run 结果后去掉 `--dry-run` 即可训练。

`auto-posttrain` 只处理后训练，绝不会选择预训练。两者边界如下：

| 入口 | 数据形态 | 目标 |
|---|---|---|
| `pretrain` / `pretrain_mode: continue` | 大规模连续文本语料 | 学习语言、通用或护理领域知识 |
| `auto-posttrain` | 带目标回答或偏好标签的样本 | 自动选择 SFT、DPO 或 KTO，塑造回答方式与偏好 |

直接命令也支持有序的多阶段训练。`--stage` 中的逗号表示严格的先后依赖，不是无序开关；前一阶段成功保存的模型自动成为后一阶段输入：

```powershell
saddlellm train --stage sft,kto `
  --model Qwen/Qwen3-0.6B `
  --data data/sft.jsonl `
  --kto-data data/kto.jsonl `
  --batch-size 2 `
  --output-dir outputs/sft-kto
```

这里 `--data` 属于第一个阶段，后续阶段分别使用 `--dpo-data`、`--orpo-data`、`--kto-data` 或 `--ppo-data`。当前直接命令可真实执行 SFT、DPO、ORPO、KTO，一条命令内最多包含一个 DPO/ORPO/KTO 阶段。PPO 名称已经纳入有序解析和 dry-run，但稳定执行器尚未接入；实际命令包含 PPO 时会在任何训练开始前失败，绝不会跳过 PPO 后继续执行后续阶段：

```powershell
saddlellm train --stage sft,ppo,kto `
  --model MODEL `
  --data data/sft.jsonl `
  --ppo-data data/ppo.jsonl `
  --kto-data data/kto.jsonl `
  --dry-run
```

dry-run 会同时展示用户阶段 `sft → ppo → kto`、内部映射 `sft → rlhf → preference`，以及每个阶段的 `needs` 依赖。真正接通 PPO 前必须明确奖励模型、价值模型、rollout 策略和安全验证过的奖励契约，不能用“计划成功”冒充训练成功。

### 2.4 蒸馏与轻量化命令

轻量化不是一个模糊开关。SaddleLLM 将“得到更小学生模型”“持久化 CPU INT8”“GPU
加载时量化”和“稀疏权重实验”拆成职责清楚的命令：

| 目标 | 命令 | 当前稳定边界 |
|---|---|---|
| 更小的学生模型 | `saddlellm distill --mode logits ...` | 本地 teacher logits + 正确 causal mask，保存可直接加载的 HF 学生模型，并记录参数缩减与前后 PPL |
| 用教师答案训练已有学生 | `saddlellm distill --mode data ...` | 在线教师只生成待审核 JSONL；审核后用 `--teacher-data` 执行 SFT，防止错误医疗答案直接回灌 |
| 持久化 CPU 模型 | `saddlellm quantize ...` | 动态 INT8；`lightweight` 是同义入口；产物可由 `serve --device cpu` 加载 |
| GPU 低显存推理 | `saddlellm serve ... --quantization 8bit/4bit` | bitsandbytes 加载时量化，不另建模型目录 |
| 稀疏权重研究 | `saddlellm prune ...` | 产物可加载，但普通 dense 后端不保证文件更小或推理更快，必须另做效果评估 |

本地 logits 蒸馏会从教师配置创建层数和宽度更小的学生，同时使用真实标签 loss 与
teacher KL loss：

```powershell
saddlellm distill --mode logits `
  --teacher E:\models\Qwen3-4B `
  --data data\nursing_corpus.jsonl `
  --eval-data data\nursing_validation.jsonl `
  --student-layers 12 --student-hidden-size 1024 --student-heads 16 `
  --max-steps 1000 --batch-size 1 --max-seq-length 1024 `
  --output-dir outputs\nursing-student
```

先加 `--dry-run` 可查看最终参数。该模式训练时需同时保留教师和学生，显存不够时应改用
data 模式。学生必须比教师小；不满足时命令会拒绝。输出报告包含
`parameter_reduction_ratio`、teacher/学生初始/学生训练后 PPL 和蒸馏 loss。PPL 降低仍不等于
护理答案正确。

`--eval-data` 是独立于蒸馏语料的能力集。提供后，SaddleLLM 会在蒸馏完成后用完全相同的
设置顺序评估教师和学生，并把 `baseline.json`、`candidate.json`、`comparison.json` 和
`report.md` 写入 `输出目录/evaluation/`。`--eval-task auto` 会识别常见
`question/answer` 选择题或 `instruction/output` completion 数据；也可以显式指定
`multiple-choice` 或 `completion`。选择题报告准确率增减、能力保留率和配对 McNemar p；
completion 报告参考答案 PPL 的前后变化和确定性生成样例。

低显存数据蒸馏可以把教师放在另一个 SaddleLLM 服务中；`--concurrency` 并发请求教师：

```powershell
# 1. 生成候选答案；不会自动训练
saddlellm distill --mode data `
  --student Qwen/Qwen3-0.6B `
  --teacher qwen3-4b-teacher `
  --teacher-url http://127.0.0.1:8002 `
  --data data\nursing_prompts.jsonl `
  --domain nursing --concurrency 4 `
  --output-dir outputs\nursing-distill-data

# 2. 护理教师纠正 JSONL 后，才训练学生
saddlellm distill --mode data `
  --student Qwen/Qwen3-0.6B `
  --teacher-data data\nursing_teacher_reviewed.jsonl `
  --eval-data data\nursing_validation.jsonl `
  --method lora --epochs 1 `
  --output-dir outputs\nursing-distilled-sft
```

`--teacher-url` 接受服务根地址、`/v1` 地址或完整
`/v1/chat/completions` 地址。密钥只通过 `--api-key-env ENV_NAME` 传入。在线结果的
`metadata.review_status` 为 `pending`；在医疗场景，较大的模型也可能给出危险错误，因此
SaddleLLM 不把回答长度启发式当作临床正确性，也不在同一命令中直接启动 SFT。
此时即使传入 `--eval-data` 也只会记录为“等待审核后训练”；没有训练后的学生就不存在可比
评估。审核数据进入第二条命令后，才会自动比较原始 student 与蒸馏后的 adapter/完整模型。

创建持久化 CPU INT8 模型：

```powershell
saddlellm quantize outputs\nursing-release outputs\nursing-int8 --local-files-only
saddlellm serve outputs\nursing-int8 --device cpu --open
```

`saddlellm lightweight ...` 与 `quantize` 完全相同，默认仍为 CPU dynamic INT8。
`--config` 或 `--method awq|gptq|qat` 复用新的配置式函数入口：AWQ/GPTQ 输出 compressed-tensors，
QAT 当前使用可重载的 INT8 权重参考格式。数值、依赖和本地解压加载边界见
[函数式量化入口](docs/QUANTIZATION_FUNCTION_API.md)。每次量化、剪枝或蒸馏后，仍应单独运行 `evaluate`、`compare` 或回归 suite。

复杂、多阶段或需要续训的任务继续使用 YAML/JSON 配置：

一个可直接运行的 SFT 配置：

```yaml
project: saddlellm
experiment: my-sft
stages: [sft]

model:
  name_or_path: Qwen/Qwen2.5-0.5B-Instruct
  tokenizer: Qwen/Qwen2.5-0.5B-Instruct
  backend: hf

sft:
  enabled: true
  data_path: data/sft.jsonl
  use_lora: true
  use_qlora: false
  epochs: 1
  learning_rate: 0.0002
  per_device_batch_size: 1
  gradient_accumulation_steps: 4
  max_seq_length: 1024
  warmup_steps: 0
  save_steps: 50

training:
  max_steps: 100

distributed:
  strategy: single
  num_gpus: 1
  bf16: false
  fp16: false

logging:
  output_dir: outputs/my-sft
  backend: local
```

顶层字段的职责：

| 字段 | 作用 |
|---|---|
| `project` | 项目名，用于记录 |
| `experiment` | 本次实验名 |
| `stages` | 要执行的阶段；必须是非空、无重复的字符串列表 |
| `model` | 模型路径、tokenizer 和后端 |
| `data` | 预训练使用的数据源和清洗参数 |
| `training` | 全局步数、预训练超参数、checkpoint、恢复和 dry-run 设置 |
| `distributed` | 单机或分布式策略、GPU 数和精度 |
| `logging` | 总输出目录和日志后端 |
| `pipeline` | 可选的依赖关系和阶段级恢复设置 |
| 与阶段同名的块 | 该阶段自己的数据、方法和超参数 |

推荐显式写出阶段开关：

```yaml
stages: [sft, preference, eval]

sft:
  enabled: true

preference:
  enabled: true
  method: dpo

eval:
  enabled: true
```

`stages` 决定系统是否调度该阶段；对带 `enabled` 开关的阶段，该开关决定收到调度后是否工作。最不容易出错的写法是：把 `sft`、`preference`、`eval` 等列入 `stages` 时，其配置块同时写 `enabled: true`。

文本后训练方法：

| 方式 | 配置 |
|---|---|
| 全参数训练 | `use_lora: false`，`use_qlora: false` |
| LoRA | `use_lora: true`，`use_qlora: false` |
| QLoRA | `use_lora: true`，`use_qlora: true` |

`training.max_steps` 的含义：

- 大于 0：最多运行指定步数。
- 等于 `-1`：由阶段自己的 `epochs` 决定训练长度。
- 等于 0 或小于 `-1`：配置无效。

对于 `train` 使用的训练配置，相对路径按运行命令时的当前目录解释。为了避免路径错误，建议在仓库根目录运行仓库内的示例，或在 quickstart 目录内运行生成的配置。

### 2.4 推荐的使用顺序

#### 第一步：检查环境

```powershell
saddle-llm doctor
```

检查 Python、PyTorch、Transformers、Datasets、Accelerate、PEFT、TRL、CUDA、GPU、内存和磁盘。

主要返回：

```json
{
  "ready": true,
  "python_version": "...",
  "cuda_available": true,
  "gpu_count": 1,
  "gpu_memory_gb": 20.0,
  "packages": {},
  "issues": [],
  "recommendations": []
}
```

`ready: false` 时看 `issues`。没有 GPU 会被标记为问题，但极小的 CPU 测试仍可能运行。

#### 第二步：检查数据

SFT：

```powershell
saddle-llm inspect-data data/sft.jsonl --task sft
```

DPO/ORPO：

```powershell
saddle-llm inspect-data data/preference.jsonl --task dpo
```

KTO：

```powershell
saddle-llm inspect-data data/kto.jsonl --task kto
```

返回 `ready`、样本数、字段格式、文本长度、重复样本和阻断问题。SFT 数据还会报告
`dominant_prompt_prefix_ratio` 与 `dominant_completion_shape_ratio`：当同一种提示前缀或回答形态占比过高时，
命令会警告模板过拟合风险，并建议加入自然问法、改写问法和无包装问法。同一个原始样本的不同问法必须放在
同一数据切分中，避免变体泄漏到 validation/test。重复的 system message 不计作用户提示模板。数据不合格时退出码为 1；
模板集中属于质量警告，不会替用户武断阻断训练。

#### 第三步：检查配置

```powershell
saddle-llm validate-config config.yaml
```

返回：

```json
{
  "valid": true,
  "stages": ["sft"],
  "issues": [],
  "warnings": [],
  "data_inspections": {},
  "training_estimates": {},
  "normalized_config": {},
  "config_path": "config.yaml"
}
```

它会检查阶段名、必填数据、模型后端、训练步数、精度冲突、分布式能力、动作空间和发布门禁等。只检查结构、不打开本地数据时使用：

```powershell
saddle-llm validate-config config.yaml --skip-data-inspection
```

#### 第四步：查看执行顺序

```powershell
saddle-llm train config.yaml --dry-run
```

返回 `stages`、`pipeline_plan`、`distributed` 和数据源数量。这个参数不会加载模型或训练，只展示调度顺序；完整数据检查仍应使用 `validate-config`。

#### 第五步：正式训练

```powershell
saddle-llm train config.yaml
```

成功时返回：

```json
{
  "ok": true,
  "command": "train",
  "config": "config.yaml",
  "output_dir": "outputs/my-sft",
  "next": {
    "dashboard": "saddlellm dashboard \"outputs/my-sft\" --open"
  },
  "results": {
    "sft": {
      "status": "completed",
      "model_path": "outputs/my-sft/sft_checkpoints"
    }
  }
}
```

失败时返回 `ok: false`、异常类型和错误信息，并以退出码 1 结束。需要完整 Python 堆栈时：

```powershell
saddle-llm train config.yaml --debug
```

### 2.5 基础命令

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `quickstart` | 创建入门训练目录 | `saddle-llm quickstart .\demo` | `config.yaml`、样例数据、下一步命令 |
| `doctor` | 检查环境和依赖 | `saddle-llm doctor` | `ready`、硬件、依赖、问题和建议 |
| `inspect-data` | 检查 SFT/偏好/RL 数据 | `saddle-llm inspect-data data.jsonl --task sft` | 数据质量报告 |
| `inspect-vla` | 检查 VLA 轨迹、图像和动作维度 | `saddle-llm inspect-vla robot.jsonl --image-root images` | VLA 数据报告 |
| `validate-config` | 静态校验训练配置 | `saddle-llm validate-config config.yaml` | `valid`、问题、警告和训练量估算 |
| `evaluate` | 评测基座、完整模型或LoRA | `saddlellm evaluate MODEL eval.jsonl` | 选择题准确率或completion PPL、真实生成和Markdown报告 |
| `compare` | 配对比较validation/test并执行门禁 | `saddlellm compare --base-validation base.json --candidate-validation tuned.json` | 提升、McNemar检验、泄漏检查和晋级决策 |
| `train` | 执行配置或直接SFT | `saddlellm train sft --model MODEL --data DATA` | checkpoint、指标和输出路径 |
| `dashboard`（`webui`） | 查看训练进度、曲线、阶段和日志 | `saddlellm dashboard outputs/my-run --open` | 本地 Web 仪表盘，默认端口 7860 |
| `plan` | 生成启动命令，不训练 | `saddle-llm plan config.yaml` | `launch_plan.json` 和 `launch.ps1` |
| `preflight` | 针对一个后训练任务检查数据并生成计划 | `saddle-llm preflight data.jsonl --stage sft` | `preflight_plan.json`、`config.json`、阻断项和估算 |
| `plugins` | 列出内置和已安装的阶段 | `saddle-llm plugins` | 阶段名称、来源和加载状态 |
| `smoke-test` | 用本地 tiny 模型检查训练主链 | `saddle-llm smoke-test --in-process` | SFT、VLA、DPO、ORPO、KTO 等测试结果 |
| `import-data` | 导入数据 manifest | `saddle-llm import-data dataset_info.json` | 保存位置、数据摘要和导入数量 |

`plan` 根据 `distributed.strategy` 生成启动文件。它不执行训练：

```powershell
saddle-llm plan configs/sft_lora.yaml
.\configs\launch.ps1
```

#### 自动并行训练

最短入口是一条命令。`auto` 会使用当前进程可见的 GPU；可用
`CUDA_VISIBLE_DEVICES` 限制设备：

```powershell
saddlellm train config.yaml --parallel auto
```

也可以只指定每节点 worker 数，系统会自动进入并行决策：

```powershell
saddlellm train config.yaml --num-gpus 8 --launcher torchrun
```

Python 入口使用同一套规划和防递归启动逻辑：

```python
from saddlellm import launch_training

result = launch_training(
    "config.yaml",
    strategy="auto",
    num_gpus=8,       # 每节点 worker 数，不是全局卡数
    num_nodes=1,
    launcher="torchrun",
)
```

控制进程会解析硬件和阶段能力。单机产物写入 `OUTPUT/parallel/`，多机则按
`OUTPUT/parallel/node-<machine_rank>/` 隔离 `resolved_config.json`、
`launch_plan.json` 和 `launch.ps1`，再通过 Accelerate 或 torchrun 启动 worker。
worker 检测到 `WORLD_SIZE/RANK/LOCAL_RANK` 后直接进入训练，不会递归启动第二层进程。拓扑或依赖不完整时在模型加载前失败。

| 策略 | 当前允许范围 | 启动器 | 当前成熟度 |
|---|---|---|---|
| `single` | 各阶段原有单进程能力 | 进程内 | 可用 |
| `ddp` | `saddle` 预训练；HF 全参/LoRA/QLoRA SFT；HF DPO/ORPO/KTO；Saddle 全参 SFT/DPO | Accelerate、torchrun | 实验性；已有 CPU/Gloo 双进程原生 SFT→DPO Trainer 端到端验收，尚无双 GPU 端到端验收 |
| `fsdp` | 仅 `model.backend: saddle` 的预训练 | Accelerate、torchrun | 实验性；配置路径已接通，尚无 GPU 验收 |
| `deepspeed-zero2/3` | 仅 `model.backend: saddle` 的预训练 | Accelerate、torchrun | 条件性实验；需 `pip install -e ".[train,deepspeed]"`，当前环境未做端到端验收 |
| TP/PP/CP/EP、MoE hybrid | `FactoryBackendPlanner` 可给出提示参数 | 尚无 runtime adapter | 仅规划；执行入口明确阻断 |

后训练当前只允许单节点 DDP；FSDP、ZeRO 和多节点后训练会在预检阶段拒绝。其有效全局 batch 为：

```text
per_device_batch_size × gradient_accumulation_steps × world_size
```

增加卡数会改变优化语义，系统不会暗中修改 micro-batch 或梯度累积来保持原 global batch。

| 拓扑 | 当前行为 |
|---|---|
| 单机多卡 | 一个 controller 可直接拉起本机静态 worker |
| 静态多机 | 生成并执行“本节点 agent”命令；必须由外部调度器在每个节点分别设置 machine rank、master address 后启动 |
| Slurm/Kubernetes/Ray | 尚无资源申请和节点 fan-out adapter；可由它们在外层分发逐节点命令 |
| Torch Elastic | 尚无动态成员、rendezvous 生命周期和自动恢复 |
| 百卡/千卡 | 可以对同构静态 worker 数做拓扑预检并生成逐节点命令；未做真实集群验收、性能 benchmark 或容错验证 |

多机示例中的 `--num-gpus 8 --num-nodes 125` 表示 1000 个 worker，但命令必须由外部调度器在 125 个节点各执行一次；它不是从登录节点自动申请并分发 125 台机器。返回字段 `executable: true` 只表示规划器静态检查通过；命令构造和实际运行仍可能失败，更不代表显存、RDMA/NCCL、共享存储、吞吐、容错或千卡规模已经验收。

`auto` 不是显存或吞吐 autotuner：它不做 OOM 自动重试，不自动调整后训练的 batch 或学习率。FSDP/ZeRO 的分片 checkpoint、跨 world-size 重分片和恢复也尚未完成集群验收。`deployment_scope` 与 `worker_scale` 是拓扑描述；64/512 worker 阈值只用于标签和告警，不是能力认证。

`FactoryBackendPlanner` 与这里的 AutoParallel 是两层能力：前者的 `total_gpus` 是全局规模，可在内存中建议 TP/PP/CP/EP；后者的 `--num-gpus` 是每节点 worker 数，目前只执行 single/DDP/FSDP/ZeRO，不会自动消费或执行 hybrid plan。Factory 导出的 orchestrator 配置明确按单节点解释；多节点不能直接复制其中的 `num_gpus`，必须重新提供每节点 `--num-gpus` 和 `--num-nodes`。

并行配置中的 `stages` 必须全部支持同一种策略。若配置还包含单进程 `eval` 或 `export`，请把训练与评测/导出拆成独立命令；系统会在启动 worker 前阻断不兼容的混合阶段，而不是让部分 rank 偷跑。

`preflight` 一次只检查一个后训练任务，所以参数使用单数 `--stage`。真正的训练配置仍使用复数 `stages`：

```powershell
saddle-llm preflight data/posttrain_sft_example.jsonl --stage sft --root-dir .\llm_factory
```

### 2.6 媒体和生成命令

这些命令用于单独执行某个工具；它们的参数文件只是该命令的输入，不是 `train` 使用的训练配置。

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `media-codecs` | 查看已注册的图像、音频或视频编码器 | `saddle-llm media-codecs --modality image` | Codec 名称、模态和能力 |
| `build-media-cache` | 把原始媒体清单编码成可恢复分片 | `saddle-llm build-media-cache configs/media_cache_image.yaml` | cache manifest、分片路径和样本统计 |

同样的媒体处理也可以放入主训练配置：

```yaml
stages: [media_cache, image_generation]
```

完整示例见 [raw_image_to_training.yaml](configs/raw_image_to_training.yaml)。

### 2.7 世界模型和眼动控制命令

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `world-model-backends` | 查看 RSSM 后端 | `saddle-llm world-model-backends` | 可用后端及能力 |
| `train-world-model` | 单独训练世界模型 | `saddle-llm train-world-model configs/world_model.yaml --dry-run` | 推断出的维度、训练计划；去掉 `--dry-run` 后返回 checkpoint 信息 |
| `infer-world-model` | 从 checkpoint 做 rollout 或动作规划 | `saddle-llm infer-world-model CHECKPOINT request.json` | 预测轨迹、奖励或候选动作 |
| `build-eye-control-data` | 生成安全过滤后的双轴眼动轨迹 | `saddle-llm build-eye-control-data --episodes 128` | JSONL 数据路径和生成统计 |
| `simulate-eye-control` | 在软件仿真器评估 PID 或 RSSM+CEM | `saddle-llm simulate-eye-control --config configs/prosthetic_eye_runtime.yaml` | 跟踪误差、安全事件和控制统计 |

世界模型也能作为主训练阶段运行：

```powershell
saddle-llm train configs/world_model_stage.yaml
```

世界模型训练后立即做眼动控制评估：

```powershell
saddle-llm train configs/prosthetic_eye_control.yaml
```

### 2.8 空间智能和 WorldAgent 命令

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `build-spatial-world-data` | 从俯视图生成空间世界模型轨迹 | `saddle-llm build-spatial-world-data map.png --output data/spatial.jsonl` | 轨迹数据和生成统计 |
| `evaluate-spatial-world-model` | 评估多步占用、运动和碰撞预测 | `saddle-llm evaluate-spatial-world-model CHECKPOINT data.jsonl` | 分项评测指标 |
| `plan-spatial-route` | 从地图提取占用网格并规划 Top-K 路线 | `saddle-llm plan-spatial-route map.png --start 10,20 --goal 300,200` | 路线 JSON、HTML 和 PNG |
| `spatial-studio` | 启动可视化空间工作台 | `saddle-llm spatial-studio` | 前台 HTTP 服务，默认端口 7865 |
| `world-agent-run` | 一次执行分析、规划和仿真 | `saddle-llm world-agent-run map.png --start 10,20 --goal 300,200` | 状态、候选路线、仿真和产物路径 |
| `world-agent-api` | 启动无界面的 WorldAgent API | `saddle-llm world-agent-api` | 前台 HTTP 服务，默认端口 7866 |

`dashboard`、`webui`、`spatial-studio`、`world-agent-api` 和 `serve` 是持续运行的服务命令，不会像普通命令一样立即返回。按 `Ctrl+C` 正常停止。

### 2.9 评测、发布和服务命令

以下示例统一使用 `saddlellm`。安装包仍保留 `saddle-llm` 这个可执行文件名；旧子命令 `evaluate-model`、`compare-evaluations`、`export-model`、`serve-model` 也继续作为兼容别名。推荐新脚本使用短命令 `evaluate`、`compare`、`export`、`serve`。

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `report` | 汇总一个训练输出目录 | `saddlellm report outputs/my-run` | 预训练/运行报告 |
| `evaluate` | 对本地JSONL/JSON评测模型、adapter或服务 | `saddlellm evaluate MODEL eval.jsonl --local-files-only` | 选择题准确率、completion PPL或声明式rubric、逐题JSON和Markdown报告 |
| `compare` | 成对比较基座/候选的validation和封存test | `saddlellm compare --base-validation base.json --candidate-validation tuned.json` | McNemar显著性、提升、checkpoint选择和发布决策 |
| `export` | 把 checkpoint 或 adapter 打包成可验证发布目录 | `saddlellm export CHECKPOINT outputs/release` | 模型文件、manifest、可选权重哈希和门禁信息 |
| `dashboard`（`webui`） | 查看训练目录 | `saddlellm dashboard outputs/my-run --open` | 进度、指标、loss 曲线、阶段、产物和日志 |
| `serve` | 启动 OpenAI 兼容 API 和内置对话页 | `saddlellm serve outputs/release --open` | `/`、`/v1/models`、流式 `/v1/chat/completions`、`/v1/completions` |

多选题评测文件默认使用 `instruction`、`question`、`answer`、`id` 字段；可用 `--prompt-field`、`--question-field`、`--answer-field`、`--id-field` 修改。`answer` 必须是 `--choices` 中的一个标签。评测基座：

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B data/validation.jsonl `
  --local-files-only --group-field medical_discipline `
  --batch-size 32 --max-length 384 `
  --output outputs/base_validation.json `
  --markdown-output outputs/base_validation.md
```

若要发现模型是否只记住一种提问模板，可以用领域项目维护的普通 JSON 模板文件，一次展开
同一批题目的多种问法：

```json
{
  "variants": [
    {"name": "standard", "template": "{instruction}"},
    {"name": "natural", "template": "请回答下题，只给选项字母。\n{question}\n{options}"},
    {"name": "bare", "template": "{question}\n{options}"}
  ]
}
```

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B data/validation.jsonl `
  --prompt-variants configs/prompt_variants.json `
  --local-files-only --batch-size 32 `
  --output outputs/base_robustness.json `
  --markdown-output outputs/base_robustness.md
```

模板中的占位符来自每条原始记录；缺字段、重复题号或同题答案不一致会直接报错。报告包含
每种问法准确率、最差问法准确率、预测翻转率，以及“所有问法都答对”的病例比例。也可以先
在数据中自行展开变体，再用 `--case-field` 和 `--variant-field` 指定同题编号与变体名。
这些机制完全与领域无关；护理、法律等具体模板应放在各自项目中，不写进 SaddleLLM 核心。

评测LoRA并运行全部真实生成：

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B data/test.jsonl `
  --adapter outputs/run/sft_checkpoints/checkpoint-100 `
  --local-files-only --generation-all `
  --output outputs/candidate_test.json `
  --markdown-output outputs/candidate_test.md
```

completion-only PPL只计算参考回答token，不把问题token混入loss；数据默认使用 `instruction` 和 `output` 字段：

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B data/sft_validation.jsonl `
  --task completion --local-files-only `
  --generation-samples 3 --max-length 512 `
  --output outputs/completion_eval.json `
  --markdown-output outputs/completion_eval.md
```

PPL只能在相同模型tokenizer、chat template、数据和截断规则下比较；PPL更低不等于事实更正确。自由回答仍要结合生成回归和领域专家审核。指定 `--generation-samples` 或 `--generation-indices` 后，Markdown会同时展开问题、确定性生成和可折叠参考答案，便于直接审查实际回答。

开放回答可使用领域项目维护的声明式 rubric。核心只提供
`contains_any`、`contains_all`、`not_contains_any`、`regex`、`not_regex` 五种通用检查；
护理规则和阈值不能写进 SaddleLLM 核心：

```json
{"id":"case-1","instruction":"...","rubric":{"pass_threshold":0.8,"checks":[{"id":"escalate","type":"contains_any","patterns":["通知医生","启动急救"],"weight":2,"critical":true,"dimension":"escalation"}]}}
```

直接加载本地模型评估：

```powershell
saddlellm evaluate MODEL rubric.jsonl --task rubric --local-files-only `
  --output outputs/rubric.json --markdown-output outputs/rubric.md
```

也可复用已经由 `saddlellm serve` 启动的 OpenAI 兼容服务，不重复占显存：

```powershell
saddlellm evaluate served-model rubric.jsonl --task rubric `
  --endpoint http://127.0.0.1:8000 --max-new-tokens 128 `
  --output outputs/rubric.json --markdown-output outputs/rubric.md
```

rubric 会生成每条回答并报告通过率、加权分、分维度结果和关键失败数；空回答永不通过。
这些确定性规则只是可审计的工程筛查，不是专家评分、事实正确性或临床安全证明。比较 rubric
时可用 `compare --max-critical-failures 0` 阻止平均分上升但关键项仍失败的候选模型。

正式发布时同时比较 validation 和封存 test。若提供数据 manifest，指定的重叠统计必须全部为0：

```powershell
saddlellm compare `
  --base-validation outputs/base_validation.json `
  --candidate-validation outputs/candidate_validation.json `
  --base-test outputs/base_test.json `
  --candidate-test outputs/candidate_test.json `
  --manifest data/dataset_manifest.json `
  --overlap-key exact_question_overlap_after_exclusion `
  --min-gain-pp 5 --max-p 0.05 `
  --output outputs/release_gate.json `
  --markdown-output outputs/release_gate.md
```

`evaluate` 会同时保留每题预测与归一化标签概率；加 `--generation-all` 后还保存贪心生成文本和解析结果。标签概率与开放生成可能不同，因此二者不能互相替代。`compare` 使用同题配对变化和精确McNemar检验，不把训练loss或PPL当作实际正确率。

只传 validation 时，`compare` 可用于选择 checkpoint：达到阈值会写
`decision=validation_passed`、`selection_eligible=true`，但始终保持
`accepted=false`。只有同时提供成对的封存 test，且 validation/test 都通过阈值时，才会写
`decision=promoted`、`accepted=true`。因此 validation-only 报告不能绕过
`export --require-gate`；test 一旦解封也不得继续参与调参。

#### checkpoint、release 与部署

- 推荐让 `train` 配置只包含训练阶段（例如 `stages: [sft]`）。评估、比较、导出和部署随后通过独立命令串联，不需要把它们塞进一次训练调用。
- `checkpoint` 是训练过程存档，适合继续训练和快速调试。
- `release` 是通过 `export` 整理出的交付目录，适合复制到服务器、比赛提交和长期保存。
- `export` 会整理模型权重、配置和 tokenizer。LoRA adapter 默认与基座模型合并，并生成 `release_manifest.json`；它不会执行模型评估。
- `serve` 只加载指定模型并启动服务，不执行评估，也不要求评估通过。

训练完成后可以直接启动 checkpoint，不必先导出：

```powershell
saddlellm serve E:\DEV\medModel\outputs\my-run\sft_checkpoints\checkpoint-100 `
  --local-files-only --device cuda --open
```

显存优先时可做运行时量化（不会另存量化权重）：

```powershell
saddlellm serve E:\DEV\medModel\outputs\my-run\sft_checkpoints\checkpoint-100 `
  --local-files-only --device cuda --dtype bfloat16 --quantization 8bit --open
```

`--quantization` 可选 `none`、本地 bitsandbytes 的 `8bit/4bit`，以及实验性原生 W8A16 `int8`。
bitsandbytes 路线安装 `.[serve,qlora]`，原生 int8 另需兼容的 CUDA/Triton。
量化可能改变任务答案，上线前必须重新评估该精度下的模型效果。

确认版本准备交付时，再导出成独立 release 目录：

```powershell
saddlellm export `
  E:\DEV\medModel\outputs\my-run\sft_checkpoints\checkpoint-100 `
  E:\DEV\medModel\outputs\my-release `
  --local-files-only
```

然后部署 release 模型：

```powershell
saddlellm serve E:\DEV\medModel\outputs\my-release `
  --local-files-only --device cuda `
  --host 127.0.0.1 --port 8000 --open
```

如果团队需要在 CI 或正式发布流水线中自动阻止未达标版本，可以选择在导出阶段检查已有评估报告。普通调试、导出和部署不需要这个参数：

```powershell
saddlellm export CHECKPOINT outputs/release `
  --gate-result outputs/run/release_gate.json --require-gate
```

推荐的完整命令链如下；每一步都可以独立重跑，前一步的文件路径作为后一步输入：

```text
saddlellm train config.yaml
  └─ checkpoint
      └─ saddlellm evaluate ...
          └─ evaluation.json
              └─ saddlellm compare ...
                  └─ comparison.json
                      └─ saddlellm export checkpoint release
                          └─ saddlellm serve release --open
```

浏览器对话和 OpenAI 兼容 API 共用一个模型实例。聊天支持 SSE、真实 usage。
浏览器历史记录保存在启动 `serve` 命令时的目录下 `.saddlellm/chat_history.sqlite3`；刷新可从侧栏恢复，按模型区分。远程访问历史需要配置 API Key。
2.40 默认保留模型模板的思考设置；`--disable-thinking`/`--enable-thinking` 才显式改变。
默认输出预算 256、上限 512，新 token 预算应结合长推理需求和总上下文设置。
`serve` 不执行训练/评估；发布质量由独立 evaluate/compare 和可选 export 门禁负责。
低并发可走 local/native，高并发 auto 可选中兼容的 Linux CUDA vLLM；
`--require-high-throughput` 防止误走普通本地路线，实际容量仍需测量。
安装、SDK、参数和代码链路见 [OpenAI 服务手册](docs/OPENAI_SERVING.md)。

如果需要 Bearer Token，不要把密钥直接写进命令：

```powershell
$env:SADDLE_API_KEY = "your-secret"
saddlellm serve outputs/release --api-key-env SADDLE_API_KEY
```

训练可视化、推理页面、API 示例和生产边界见 [WebUI 与快速部署](docs/WEBUI_AND_SERVING.md)。

#### 已落地的推理优化

普通用户仍用 `saddlellm serve MODEL` 或 Python `model.generate(...)`。
以下模型执行优化主要针对原生 `SaddleForCausalLM`；Hugging Face/custom 模型保留自身生成实现，
不会强行套用原生缓存或采样逻辑。这些实现使用现有 PyTorch 算子，不要求用户额外编写 C++。

| 优化 | 实际效果与边界 |
| --- | --- |
| 请求独立的几何增长 KV / MLA latent 缓存 | 容量足够时直接写入新 token，避免每步拼接完整历史；扩容仍有复制，尚不是分页 KV。 |
| 真正的滑动窗口 KV 回收 | 已有窗口层自动使用镜像环形存储，最多 W 个逻辑 key、2W 个物理槽位；绝对 RoPE/文档位置不重置，全局层保留完整历史。不是给任意全局模型无损加窗口。 |
| 可选的跨请求 prompt 快照 | `--prefix-cache-mb 64` 开启有预算/LRU/失效保护的单请求缓存，默认完整 token 提示词命中；部分复用另加 `--prefix-cache-partial`。命中后私有复制，不是零复制或答案缓存；整段收益须实测。 |
| 最后位置 logits 投影 | 生成时只对最后一个位置计算词表分数，减少长 prompt 的投影计算和 logits 内存；prompt 的 Transformer 计算仍完整执行。 |
| 服务加载时合并 QKV / gate-up 投影 | 原生旧权重在内存中合并计算，减少 Linear 调用，不需重训；`serve` 默认启用，`--no-fuse-projections` 可关闭。训练加载器不自动改参数对象，检查点保持原逻辑键。 |
| 单 token GQA 的紧凑 KV 解码 | 符合条件的 CUDA 无梯度、无 mask 解码将同组查询头作为查询行，避免重复展开 K/V；原生 Flash GQA 保持优先，prefill、训练、带 mask 和自定义算子路径不改。 |
| 默认残差系数的恒等快速路径 | 系数为普通浮点 1.0 时跳过多余乘法；可训练的 tensor 系数继续保留计算及梯度。 |
| token / mask 预留缓冲区 | 新 token 写入预留列，避免逐步复制整段序列；无 padding 时省去冗余全一 mask，提前完成的行仍正确屏蔽 PAD。 |
| 统一的 prefill / step 会话 | 完整 generate 和原生流式共用采样、缓存、停止逻辑；首步复用 prefill 结果，可暂停与取消，不重复处理 prompt。 |
| 原生流式拉取 | 不再为每个普通原生请求额外启动生成线程；HTTP 仍在线程池执行阻塞模型步骤，不是零线程。 |
| 有限的固定微批处理 | `--batch-size` 默认合并同长度、同预算、同参数的原生非流式贪心请求；可选相近长度分桶，保留每行 stop 与真实 usage。 |
| token 预算与防饥饿调度 | 批预算包含左 padding 和输出预留；可选队列 token 上限，取消后正确释放预留。HTTP 短 prompt 优先，等待过久的最老请求优先；不是抢占或延迟保证。 |
| 紧凑重复惩罚索引 | 提示词只去重一次，生成 token 写入预留索引；批量调整分数，不再每步逐行去重完整历史。默认惩罚系数为 1 时不分配索引。 |
| 兼容性优先的 stop 检查 | 每行每次只完整解码一次；符合条件的 ByteLevel 原生请求改用新增 token 和有界后缀，其余 decoder / 自定义生成保留完整解码。流式文本 stop 也只保留待确认后缀。 |
| 停止、计数与资源回收 | stop / EOS / 断连后停止继续生成；生成中的模型步骤结束后再释放许可和会话引用。无 EOS、无自定义停止条件的固定预算生成省去每步 finished 标量同步。 |

想开启固定微批处理，只需要增加一个参数：

```powershell
saddlellm serve outputs/release --batch-size 4
```

客户端须设置 `temperature=0`、`stream=false` 才进入这条合批路径；浏览器默认流式，
不会自动合批。先保留默认 `--max-concurrency 1`，用实际模型测显存和延迟。
`/health` 的 `batching` 与 `streaming_backend` 字段可以确认实际执行路径。

相近长度合批和明确预算可选开启，默认用法不变：

```powershell
saddlellm serve outputs/release --batch-size 4 --batch-length-bucket-size 64 --batch-token-budget 4096 --max-queued-tokens 16384
```

使用、代码流程、回归与独立 GPU 测量见 [窗口 KV 与 token 预算调度](docs/INFERENCE_WINDOW_AND_SCHEDULING.md)。

服务部署也提供 [跨平台指南](docs/CROSS_PLATFORM_SERVING.md)：Windows 原生 CPU/CUDA 已在本机测试，
文本服务新增显式 MPS 路由、`/livez`/`/readyz`、CPU/CUDA 共用的 Dockerfile 和三平台 CPU CI。
后续已在 Ubuntu 22.04 + NVIDIA L20 实机完成 Qwen3-8B BF16 的 CLI/HTTP/流式部署测量，
见 [Qwen3-8B 服务实验](docs/QWEN3_8B_SERVING_EXPERIMENT.md)。macOS 真机和容器构建仍待验收，
不能把这台 Linux 服务器的结果当作全平台认证。
本轮 Qwen3-8B 同权重 BF16 对照中，8 并发、64-token 输出负载下 Saddle HF 约 39.80 tokens/s，
原始 Transformers 约 39.75，vLLM 约 251.77；当前原生引擎优化不会自动加速 HF Qwen3。
Ollama Q4_K_M 的提示词缓存命中结果另列，不能当作同精度加速或无损质量结论。
2.37 已接入可选 `--backend vllm`：直接使用异步引擎做 GPU 调度，Saddle 保留 API、鉴权、
就绪检查、流式真实用量和有界接入。它没有重写 vLLM 的内核，也没有将此前独立 vLLM 的测量
冒充新 adapter 的 8B 性能验收。使用与源码流程见 [vLLM 服务指南](docs/VLLM_SERVING_BACKEND.md)。
2.38 的 `--backend native` 将已支持的普通 Qwen3 safetensors 直接加载到 Saddle 模型中，
让原生 KV、最后位置投影、投影合并和固定微批可以作用于这一适配路径。
不支持的架构/量化/缩放 RoPE 会拒绝，不会暗中退回 HF。数学参考、低精度差异与测速边界见
[Qwen3 原生后端指南](docs/NATIVE_QWEN3_BACKEND.md)；不把小模型测试当成完整 8B 速度结论。
两个 completion 入口均可流式返回实际用量；队列支持凑批提前唤醒、输入加输出预算的工作量优先级，
最近等待/执行 P50/P95/P99 和 padding 统计。`tools/benchmark_serving.py --stream` 测首个非空文本 SSE，
`tools/benchmark_native_serving.py` 提供同权重的真实 loopback HTTP 分桶对照；均不冒充连续批处理。
同一随机未训练混合模型的测试中，持久 KV 减少 57.35%，会话耗时约 1.04× 加速；
这不是总显存下降比例，也不代表真实大模型、生产请求或 vLLM/SGLang 性能。

原生后端已有受限、可选的单请求前缀快照；自身仍没有连续批处理、分页 KV、
窗口感知跨请求前缀复用或 CUDA Graph。前缀快照默认关闭，窗口模型、固定合批和 HF/custom 路径不使用它。
已有 MTP 实验路径也不等于生产级的缓存回滚与推测解码加速。这些边界不能用“接近 vLLM”概括。
现在可通过配置选用 Megatron Bridge 预训练/SFT 和 SGLang 采样（HTTP 或本地 Engine）：

```yaml
backends:
  training: megatron  # 默认 default，沿用原来的训练后端
  rollout: sglang     # 默认 native
```

具体 recipe、模型和数据配置见 [可选运行时使用指南](docs/OPTIONAL_RUNTIME_BACKENDS.md)，
可直接复制 [组合配置示例](configs/megatron_sglang_mopd.yaml)，先用原有
`saddlellm train CONFIG --dry-run` 检查，再实际运行。外部 GPU 运行时按需安装，默认不加载；
Megatron 实际执行要求受支持的 Linux CUDA 环境。当前是分阶段采样/训练与异步采样请求，
不是训练和采样持续重叠的在线 RL；真实 GPU 联调与性能验收尚未完成。

实现原理、Python 调用、代码入口、适用条件和可复现基准见
[原生推理优化说明](docs/NATIVE_INFERENCE_ENGINE.md)；后端分离的路线见
[异步采样后端设计](docs/ROLLOUT_BACKEND_DESIGN.md)。基准区分缓存写入、控制逻辑和整段生成，
不能把局部微基准的加速比直接当作线上 tokens/s 或首 token 延迟的提升。

同一 28M 权重的 RTX 4090 Laptop GPU 对照已完成：FP32、B=4、128 输入/32 输出，
最新原生路径约 274.6 tokens/s，Transformers SDPA 约 221.5 tokens/s，生成 token 一致。
这是本机固定预算模型调用测试，不包含 HTTP、分词或真实负载；不能代表任意大模型，
也不是 vLLM/SGLang 的胜负结论。完整条件、原始 JSON、BF16 差异及复现命令见
[推理性能比较与优化路线](docs/INFERENCE_PERFORMANCE_COMPARISON.md)。
其中已补充 [vLLM 官方公开数据与能力对照](docs/INFERENCE_PERFORMANCE_COMPARISON.md#vllm-public-comparison)，
标明模型、硬件、负载与测试限制；不要求本机安装 vLLM，也不把异机成绩当作同条件胜负。
可选前缀快照的冷/热实测见 [前缀缓存结果](docs/INFERENCE_PERFORMANCE_COMPARISON.md#prefix-cache-results)：
完整命中能减少模型 prefill，整段生成收益不稳定，部分复用本机更慢，默认不启用。
轻量化的已实现能力、KV 量化占位、剪枝形状与持久化缺口见
[轻量化源码审查](docs/MODEL_LIGHTWEIGHT_REVIEW.md)，其中的后续方案尚未全部实现。

### 2.10 Python 调用

直接运行配置：

```python
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

runner = TrainingOrchestrator.from_yaml("config.yaml")
results = runner.run()
print(results)
```

从字典运行：

```python
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

config = {
    "stages": ["sft"],
    "model": {
        "backend": "hf",
        "name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "sft": {
        "enabled": True,
        "data_path": "data/sft.jsonl",
        "use_lora": True,
        "use_qlora": False,
    },
    "training": {"max_steps": 10},
    "distributed": {"strategy": "single", "bf16": False},
    "logging": {"output_dir": "outputs/python-run"},
}

results = TrainingOrchestrator.from_dict(config).run()
```

只做静态校验：

```python
import yaml
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator

with open("config.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

report = TrainingConfigValidator.validate(config)
print(report.to_dict())
```

### 2.11 返回状态和输出目录

阶段结果中的常见 `status`：

| 状态 | 含义 |
|---|---|
| `completed`、`trained` | 已实际执行成功 |
| `planned` | 只生成计划，没有训练 |
| `planned_no_data` | 缺少数据，只保存计划 |
| `planned_unsupported` | 已识别，但真实执行尚未接入 |
| `skipped` | 阶段被关闭 |
| `rejected` | 评测门禁未通过 |
| `blocked` | 因上游条件不满足而阻止执行 |
| `failed` | 执行失败 |

典型输出目录：

```text
outputs/my-run/
├─ plans/
│  ├─ sft_plan.json
│  ├─ preference_plan.json
│  └─ export_plan.json
├─ sft_checkpoints/
├─ preference_checkpoints/
├─ final_model/
├─ release/
├─ pipeline_state.json
├─ release_gate.json
├─ summary.json
└─ training_summary.json
```

不同 `stages` 只生成与自己有关的目录。`summary.json` 和 `training_summary.json` 记录执行顺序、耗时、各阶段结果和产物。

## 3. 原理、代码逻辑和训练过程

### 3.1 一条命令进入系统后发生什么

执行：

```powershell
saddle-llm train config.yaml
```

内部顺序：

```text
cli.py
  1. 读取 YAML/JSON
  2. 检查顶层 stages 是非空、无重复的字符串列表
        ↓
TrainingOrchestrator._parse_config()
  3. 把 model、training、distributed 和各阶段参数解析成配置对象
        ↓
StageRegistry
  4. 确认每个阶段有对应实现，并读取它的能力限制
        ↓
PipelinePlan
  5. 确定执行顺序和依赖关系
        ↓
TrainingOrchestrator._validate_config()
  6. 在模型加载前检查数据、后端、精度、分布式和阶段组合
        ↓
TrainingOrchestrator.run()
  7. 逐个执行阶段，保存状态和产物
        ↓
summary.json + 命令行 JSON
```

CLI 只负责读取参数和展示结果。真正的训练入口位于 `TrainingOrchestrator`，具体数学计算由各 Trainer 完成。

### 3.2 阶段如何排序

默认按 `stages` 的书写顺序执行：

```yaml
stages: [sft, preference, eval, export]
```

顺序就是：

```text
sft → preference → eval → export
```

存在独立分支时可以显式声明依赖：

```yaml
stages: [sft, world_model, preference, eval, export]

pipeline:
  dependencies:
    sft: []
    world_model: []
    preference: [sft]
    eval: [preference, world_model]
    export: [eval]
```

此时 `sft` 与 `world_model` 没有相互依赖；`eval` 必须等待两条分支完成。系统会检查不存在的依赖、重复阶段和依赖环。

简单任务不需要写 `pipeline.dependencies`。

### 3.3 阶段之间怎样交接

文本模型主链：

```text
model.name_or_path
        │
        ├───────────────┐
        ▼               │
     pretrain           │
        │ model_path    │
        ▼               │
       sft ◄────────────┘
        │ model_path
        ▼
 preference / rlhf
        │ model_path
        ▼
       eval
        │ metrics + release_gate
        ▼
      export
```

后一个阶段优先使用前一个阶段返回的 `model_path`。没有上游模型时，才使用 `model.name_or_path`。

数据产物也可以交接：

```text
mopd 产生 on-policy JSONL → sft.data_path 为空时由 SFT 使用
media_cache 产生编码分片 → image/music/video generation 使用
world_model 产生 checkpoint → eye_control 使用
```

每个阶段返回一个普通字典。系统把结果保存在阶段结果表中，同时登记可追踪产物。

### 3.4 预训练逻辑

`pretrain` 有三种启动方式：

| 方式 | 关键配置 | 含义 |
|---|---|---|
| 从零训练 | `pretrain_mode: scratch` | 从内置模型规格或 `model.blueprint` 创建新模型 |
| 继续预训练 | `pretrain_mode: continue` + `model.name_or_path` | 载入已有权重，以新的优化任务继续训练 |
| 精确恢复 | `training.resume_from_checkpoint` | 在支持的 Trainer 中恢复优化器、调度器和训练步数 |

实际过程：

1. 解析模型规格或 blueprint。
2. 创建新模型，或加载已有模型/checkpoint。
3. 加载已有 tokenizer，或使用上游 `tokenizer` 阶段的结果。
4. 从 `data.sources` 收集数据。
5. 清洗、去重和质量过滤后先划分训练/验证集，避免同一数据同时参与训练和验证。
6. tokenize 时保留长文档全部 token，在每个文档末尾补 EOS，再做 sequence packing。
7. 使用只屏蔽真实 padding、不会误屏蔽 EOS 的 causal-LM collator。
8. 创建 Hugging Face `Trainer` 或原生 `SaddleTrainer`，执行 causal language modeling。
9. 开训前检查tokenizer词表、模型embedding、EOS/PAD契约，以及过滤/packing后训练集和验证集是否为空。
10. 按验证 loss 选择最佳 checkpoint，并记录验证 PPL、实际全局 batch、token 吞吐、各清洗阶段样本数和最终序列数。
11. 保存 `final_model`、tokenizer 和 Trainer state。

最小结构：

```yaml
stages: [pretrain]
pretrain_mode: scratch

model:
  backend: saddle
  config: qwen-tiny-160m
  tokenizer: Qwen/Qwen2.5-0.5B

data:
  sources:
    - type: local
      path: data/pretrain.jsonl
      text_column: text
      streaming: false
  append_eos: true
  validation_split: 0.01

training:
  max_steps: 1000
  per_device_batch_size: 2
  global_batch_size: 32
  learning_rate: 0.0003
  min_lr: 0.00003
  eval_every_steps: 100
  save_every_steps: 500
  load_best_model_at_end: true

distributed:
  strategy: single

logging:
  output_dir: outputs/pretrain
```

没有显式设置 `distributed.gradient_accumulation_steps` 时，系统按
`global_batch_size / (per_device_batch_size × num_gpus × num_nodes)` 自动推导；不能整除或显式值
与全局 batch 不一致时会在模型加载前报错。默认 `cosine` 会实际使用 `min_lr`，映射到带最小学习率
的 cosine scheduler。`validation_split: 0` 可以关闭内部验证，但将无法得到验证 PPL和最佳 checkpoint。

完整原生模型结构示例见 [modular_llm.yaml](configs/modular_llm.yaml)。

#### 3.4.1 用简单配置表达网络结构

网络结构采用“预设继承 + 可选覆盖”，不要求每次重复几十个字段：

```yaml
pretrain_mode: scratch
model:
  backend: saddle
  config: qwen-300m              # 预设提供层数、宽度、词表等完整默认值
  tokenizer: Qwen/Qwen2.5-0.5B
  blueprint:
    name: nursing-dense-300m
    max_position_embeddings: 8192
    attention:
      kind: gqa
      backend: sdpa
      num_heads: 16
      num_kv_heads: 4
      rope_theta: 1000000
    ffn:
      kind: swiglu
      intermediate_size: 2816
    residual:
      topology: serial
      initialization: depth_scaled
```

配置分三档：

| 场景 | 写法 | 结构来源 |
|---|---|---|
| 继续预训练 | `pretrain_mode: continue` + `model.name_or_path` | 完全继承 checkpoint，不允许伪造新层数 |
| 常规从零预训练 | `pretrain_mode: scratch` + `model.config` | 使用已登记且测试过的完整结构预设 |
| 架构研究 | 预设再加 `model.blueprint`，或提供自包含 blueprint | 覆盖注意力、FFN、上下文、残差、目标和分层结构 |

`blueprint.layers`还能用`repeat`表达不同层段，例如前16层dense、后4层MoE；没有该字段时所有层采用同一默认结构。预检会验证隐藏维度能否整除头数、GQA头数关系、MoE专家数量、RoPE和上下文参数等。训练保存时同时写出`model_blueprint.json`与`saddle_config.json`，因此 checkpoint 自带真实网络结构。

开训前查看展开后的参数量、激活参数量、KV cache估算、结构轴和风险：

```powershell
saddlellm validate-config configs/nursing_pretrain_scratch.yaml --skip-data-inspection
```

输出中的`architecture_analysis`就是最终生效的结构，不是仅供展示的注释。完整可运行模板见[nursing_pretrain_scratch.yaml](configs/nursing_pretrain_scratch.yaml)。实际护理参评项目若没有数十亿级以上高质量token和相应算力，应优先对成熟Base模型做继续预训练；从零预训练的小模型主要用于验证自主架构和训练能力。

### 3.5 SFT 逻辑

`sft` 用“输入指令 → 目标回答”训练模型遵循任务要求。

支持的数据形式包括：

```json
{"instruction":"概括下面内容","input":"待处理文本","output":"目标回答"}
```

以及 messages：

```json
{"messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"目标回答"}]}
```

多轮 messages 中的**每一个非空 assistant 回答**都会形成一个 completion-only 训练样本；
该回答之前的完整对话作为上下文但不计算监督损失。因此一条 JSONL 对话可能对应多个
`train_samples`，不会再只训练最后一个回答。若回答在 `max_seq_length` 截断后完全消失，
该轮会被丢弃，不会错误地拿 prompt token 充当答案监督。

执行过程：

1. 从上游 `pretrain` 或 `model.name_or_path` 找到模型。
2. 检查数据字段、空文本、重复样本和长度。
3. 生成 `plans/sft_plan.json`。
4. 根据 `use_lora/use_qlora` 选择全参数、LoRA 或 QLoRA。
5. 优先读取独立 `validation_data_path`；未提供时才按 `validation_split` 划分。
6. HF 后端调用 `PeftSFTTrainer`；Saddle 后端调用原生 SFT。
7. 有验证集时按最低 `eval_loss` 恢复最佳 checkpoint，再保存根 `sft_checkpoints` 并返回新的 `model_path`。

示例见 [sft_lora.yaml](configs/sft_lora.yaml)。

### 3.6 偏好训练逻辑

`preference` 用于让模型在多个回答之间学习偏好。

| 方法 | 数据 | 作用 |
|---|---|---|
| DPO | `prompt/chosen/rejected` | 直接提高 chosen 相对 rejected 的概率 |
| ORPO | `prompt/chosen/rejected` | 将监督目标和偏好比率目标结合 |
| KTO | `prompt/completion/label` | 用可取/不可取标签学习偏好；实际 batch 必须大于 1 |

**KTO 是什么：**KTO 全称 **Kahneman-Tversky Optimization**，是一种偏好优化方法。
它不要求像 DPO 那样为同一个问题同时准备 `chosen` 和 `rejected` 成对回答；每条样本只需提供
一个回答，并用 `label: true/false` 标明“可取/不可取”。因此，当护理教师只能逐条审核回答、
而难以为每题编写严格配对答案时，KTO 更容易整理数据。KTO 的主要作用是调整回答偏好、
规范性和安全性，不负责向模型灌入大量新知识；新知识应先通过继续预训练或 SFT 学习。

`--stage sft,kto` 的逗号表示 SaddleLLM 严格按 **SFT → KTO** 执行，并把 SFT 输出交给 KTO；
这只是执行顺序，不代表 KTO 永远必须接在 SFT 后。若基座模型已经具备所需知识和指令能力，
也可以单独运行 `--stage kto`。

DPO/ORPO 数据：

```json
{"prompt":"怎样排错？","chosen":"先看日志并构造最小复现。","rejected":"直接忽略错误。"}
```

KTO 数据：

```json
{"prompt":"怎样排错？","completion":"先看日志。","label":true}
```

执行过程：

1. 取得执行顺序中最近完成的前置语言模型；例如 `sft → ppo → kto` 中 KTO 必须消费 PPO 输出。
2. 按选定方法检查数据。
3. 写入 `plans/preference_plan.json`。
4. 调用对应 DPO、ORPO 或 KTO Trainer。
5. 保存 `preference_checkpoints`，供评测或导出使用。

示例见 [posttrain.yaml](configs/posttrain.yaml) 和 [dpo_qlora.yaml](configs/dpo_qlora.yaml)。

SFT 能力保持、数据回放、多任务退化预算与剪枝后蒸馏恢复，见 [后训练能力保持与压缩恢复](docs/POSTTRAIN_RETENTION_AND_RECOVERY.md)；完整训练配置见 [posttrain_retention.yaml](configs/posttrain_retention.yaml)。新增 `regression` 与 `recover` 命令，回归未通过会阻止配置了该套件的模型导出。AWQ/GPTQ 和专用 QAT 通过默认关闭的函数配置启用，见 [函数式量化入口](docs/QUANTIZATION_FUNCTION_API.md)。

从面试知识理解这些机制，可读 [大模型 40 个核心问题](docs/LLM_INTERVIEW_NOTES_ZH.md)；框架的实际采用程度、旧入口问题、复现证据与整改优先级见 [源码对照审计](docs/LLM_INTERVIEW_FRAMEWORK_AUDIT_ZH.md)。两部分合并的 [Word 阅读版](SaddleLLM_大模型面试知识与框架对照笔记.docx) 含可跳转目录和公式。

2026-10-19 已落实旧评测、投机解码、全文去重和模拟训练入口整改，接入统一切分审计、能力门槛选模及显式 QAT CUDA W8A16 加载，见 [整改与实测记录](docs/CORRECTNESS_HARDENING_20261019.md)。Word 与原审计保留为 2026-10-18 的历史快照；CUDA 路径已验证实际内核执行和微型夹具显存下降，尚不代表普遍提速或生产质量验收。

`rlhf` 阶段当前可实际执行的稳定路径也是 DPO。PPO 需要 reward model、value model、在线 rollout 和经过安全验证的奖励契约，目前实际训练会在所有阶段开始前明确失败；需要稳定离线偏好训练时直接使用 DPO、ORPO 或 KTO。

### 3.7 MOPD 蒸馏逻辑

`mopd` 的目标是先让当前学生模型生成回答，再让一个或多个教师评价或重写，最后形成新的 SFT 数据。

```text
prompts
  → 学生生成候选
  → 教师评分/聚合
  → mopd_on_policy.jsonl
  → 可选后续 sft
```

`mopd.dry_run: true` 只写计划。设置为 `false` 且提供 prompts、教师和可用模型时才收集数据。
远程教师可设置 `teacher_concurrency` 并发数。`requires_review: true` 时，每条记录会标为
`review_status: pending`，且后续 SFT 不能直接消费该 MOPD 产物；必须由领域专家修订后通过
`sft.data_path` 显式传回。直接命令 `saddlellm distill --mode data` 默认采用这个安全语义。

示例见 [mopd_sft.yaml](configs/mopd_sft.yaml)。

### 3.8 多模态和 VLA 逻辑

`vla_sft` 处理“图像/指令/机器人动作”轨迹：

1. 创建并校验动作空间，包括维度、连续/离散类型、范围、夹爪和控制频率。
2. 检查图像路径、动作维度和 episode step 连续性。
3. 规范化为训练 JSONL。
4. 总是保存动作空间和训练计划。
5. 只有 `vla.train: true` 时才调用 `VLATrainer` 做行为克隆训练。

示例见 [vla_sft.yaml](configs/vla_sft.yaml)。该示例默认 `train: false`，因此只做规范化和计划。

`mllm_sft` 和 `vision_alignment` 当前只完成图文数据规范化与计划保存，不应把返回的 `planned` 当作模型已经训练。

### 3.9 图像、音乐和视频生成逻辑

生成训练分为两步：

```text
原始媒体 + 文本
       ↓
media_cache：Codec 编码媒体，文本编码器生成 condition
       ↓
NPZ/分片缓存
       ↓
image_generation / music_generation / video_generation
       ↓
模型 checkpoint
```

- 图像：读取 `latents[N,C,H,W]` 和 `conditions[N,D]`。
- 音乐：读取 `codes[N,Q,T]`、`conditions[N,D]` 和可选 mask。
- 视频：读取 `video_latents[N,C,T,H,W]` 和 `conditions[N,D]`。

训练前会从缓存推断模型维度，并拒绝与手工配置不一致的通道数、码本数或条件维度。

示例：

- [raw_image_to_training.yaml](configs/raw_image_to_training.yaml)
- [music_generation.yaml](configs/music_generation.yaml)
- [video_generation.yaml](configs/video_generation.yaml)

### 3.10 世界模型和控制逻辑

`world_model` 从离线轨迹学习：

```text
当前观测 + 动作
      ↓
隐状态更新
      ↓
下一观测、奖励和 continuation 预测
```

数据通常包含 observation、action、reward 和 done。`rssm` 使用连续随机状态；`categorical_rssm` 使用离散类别状态。训练结果可用于 rollout、候选动作评分和空间控制。

`eye_control` 不直接驱动硬件。它在软件 plant 中应用角度、速度、加速度、电流和温度限制，然后比较 PID 或 RSSM+CEM 控制效果。真实硬件仍需要单独实现安全驱动适配器。

示例：

- [world_model_stage.yaml](configs/world_model_stage.yaml)
- [prosthetic_eye_control.yaml](configs/prosthetic_eye_control.yaml)
- [unified_model_posttrain_world.yaml](configs/unified_model_posttrain_world.yaml)

### 3.11 评测和导出逻辑

`eval` 的模型选择顺序：

```text
rlhf 输出
  → preference 输出
  → sft 输出
  → pretrain 输出
  → model.name_or_path
```

评测成功后可生成 `release_gate.json`。例如要求 perplexity 不超过阈值：

```yaml
eval:
  enabled: true
  tasks: [perplexity]
  gate:
    enabled: true
    rules:
      perplexity:
        max: 30.0
    fail_on_rejection: true
```

`export` 使用最新上游模型。如果配置 `require_gate: true`，没有已接受的门禁结果就拒绝导出。

完整闭环示例见 [posttrain_release.yaml](configs/posttrain_release.yaml)。

### 3.12 Batch、步数和显存逻辑

有效 batch：

```text
effective_batch
= per_device_batch_size
× gradient_accumulation_steps
× num_gpus
```

显存主要受模型参数量、序列长度、每卡 batch、优化器状态、是否全参数训练和精度影响。常用选择：

- 显存充足：全参数或 LoRA。
- 显存较小且有兼容 CUDA/bitsandbytes：QLoRA。
- OOM：先降低 `per_device_batch_size` 和序列长度，再提高梯度累积保持有效 batch。
- CPU：只适合 tiny 模型、数据检查和调度测试。

`bf16` 与 `fp16` 不能同时启用。

### 3.13 恢复、失败和状态逻辑

阶段级恢复：

```yaml
pipeline:
  resume: true
  rerun: []
  state_path: pipeline_state.json
```

系统只会恢复配置指纹和依赖图一致、且上次已完成的阶段。`pipeline.rerun` 指定某个阶段重跑时，它的下游阶段也会失效并重新执行。

Trainer checkpoint 恢复使用：

```yaml
training:
  resume_from_checkpoint: outputs/run/checkpoint-1000
```

这与阶段级恢复不同：前者恢复单个 Trainer 内部状态，后者决定整个阶段是否跳过。
直接单阶段命令也可使用 `--resume-from-checkpoint [PATH]`；省略 PATH 时从该阶段输出目录
选择最大的 `checkpoint-*`。精确恢复允许增大 `max_steps`，但必须保持有效训练 batch 不变，
否则数据游标已经不再等价，系统会在训练前拒绝。

任何阶段抛出异常时：

1. 记录 `failed_stage` 和错误。
2. 更新 `pipeline_state.json`。
3. 尽可能写出 `summary.json`。
4. 停止后续阶段。
5. CLI 返回非零退出码，不把失败包装成成功。

### 3.14 当前文本模型训练验收范围

2026-09-13 在 Windows、单张 RTX 4090 Laptop GPU、项目锁定的 Transformers/TRL/PEFT
环境中，以下路径均由真实 Trainer 执行，不是只做参数解析：

| 路径 | 证据目录 | 关键结果 |
|---|---|---|
| Saddle 原生蓝图从零预训练 + 精确续训 | `E:/DEV/medModel/artifacts/training_acceptance_scratch_native_v1` | 自定义2层GQA、75,520参数，checkpoint-1恢复到step 2，验证PPL 7.9969 |
| HF模型继续预训练 + 精确续训 | `E:/DEV/medModel/artifacts/training_acceptance_pretrain_resume_v1` | checkpoint-2恢复到step 3，验证PPL 12.7266 |
| 全参数SFT + 精确续训 | `E:/DEV/medModel/artifacts/training_acceptance_full_sft_v1` | 恢复到step 3，最终eval loss 3.1267 |
| QLoRA SFT | `E:/DEV/medModel/artifacts/training_acceptance_qlora_eval_v2` | 4bit加载、BF16计算、训练/最终验证/保存均完成 |
| LoRA SFT→KTO串联 | `E:/DEV/medModel/artifacts/training_acceptance_sft_kto_chain_v1` | 两阶段各1 step，`continued_adapter=true` |
| DPO + 验证 + 精确续训 | `E:/DEV/medModel/artifacts/training_acceptance_dpo_eval_v1` | 恢复到step 3，DPO验证指标已写入summary |
| ORPO / KTO | `E:/DEV/medModel/artifacts/training_acceptance_orpo_eval_v2`、`training_acceptance_kto_eval_v2` | 真实训练、最终验证和adapter保存完成 |
| 多轮SFT + 独立验证 + `all-linear` | `E:/DEV/medModel/artifacts/training_acceptance_quality_core_v1` | HF QLoRA/DPO/ORPO/KTO与Saddle原生SFT/DPO均真实完成；2条源记录展开为3个回答样本；各路径选择最佳checkpoint |
| 安装版核心SFT复验 | `E:/DEV/medModel/artifacts/training_acceptance_installed_quality_core_v1` | 从源码目录外调用`saddlellm.exe`；3个训练样本、2个独立验证样本、`all-linear`目标和最佳checkpoint均进入summary |
| logits蒸馏 + 独立评估 | `E:/DEV/medModel/artifacts/direct_distill_with_eval_smoke` | 训练成功，但独立completion PPL退化，报告如实标记`degraded` |
| PPO | `E:/DEV/medModel/artifacts/training_acceptance_ppo_boundary_v1` | 仅支持dry-run；真实执行在任何阶段开始前明确失败 |

这些 tiny smoke 证明数据读取、前向/反向、优化器、验证、checkpoint、续训和报告链路可用，
不证明长时训练、多机多卡或模型业务效果。DDP/FSDP/DeepSpeed仍需在目标集群做容量验收；PPO
尚未实现；护理checkpoint的临床安全与比赛效果必须由独立题集和教师审核另行证明。

该验收对应wheel：
`E:/DEV/medModel/artifacts/wheel_build_quality_core_v1/saddlellm-2.35-py3-none-any.whl`
。交付时用 `Get-FileHash -Algorithm SHA256` 核对项目SOP中记录的最终哈希。
源码全量回归为391 passed、20条第三方warning、0 failed；安装版`saddlellm.exe`又完成了一次
真实LoRA SFT、独立验证和最佳checkpoint恢复。

### 3.15 从哪里继续阅读

- [架构与调用链](docs/ARCHITECTURE.md)：模块关系和详细时序。
- [函数参考](docs/FUNCTION_REFERENCE.md)：按需生成完整函数、方法和行号索引。
- [训练指南](TRAINING_GUIDE.md)：训练检查清单。
- [多模态生成](docs/MULTIMODAL_GENERATION.md)：媒体缓存和生成训练。
- [统一训练与眼动控制](docs/UNIFIED_TRAINING_AND_EYE_CONTROL.md)：世界模型、控制和多分支执行。

开发验证：

```powershell
python -m compileall -q saddlellm tests
python tools/generate_function_reference.py
python tools/generate_function_reference.py --check
python -m pytest -q
```
