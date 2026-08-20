# SaddleLLM

SaddleLLM 是一个面向本地研究与工程实验的大语言模型训练工具包，覆盖数据检查、训练规划、SFT、偏好优化、RL/GRPO、评估、蒸馏、量化和部署等流程。

> A modular toolkit for LLM training, fine-tuning, alignment, multimodal learning, world models, evaluation, and deployment.

当前版本：`2.32`

> 建议先运行环境检查、数据检查和 dry-run，再启动真实训练。仓库包含稳定主线和实验性模块，并非所有功能都适合直接用于大规模生产训练。

## 主要能力

- 训练工厂：领域工作区、Recipe、Orchestrator 和实验产物管理
- 数据处理：SFT、DPO、ORPO、KTO、RL/GRPO 和 VLA 数据检查与规范化
- 后训练：Full Fine-tuning、LoRA、QLoRA、SFT 和 preference optimization
- 强化学习：native GRPO、可验证奖励和 source-frontier profiling
- 模型实验：Dense、GQA、MoE、MLA、MTP 和 Long Context blueprint
- 模型积木：同一份 Blueprint 可由 YAML 配置生成，也可用 Python Builder 逐层组合；支持异构 Attention/FFN 和可配置残差初始化
- 评估与诊断：Benchmark、训练报告、环境检查和 smoke test
- 模型工程：蒸馏、量化、剪枝、部署和安全生成
- 多模态/VLA：提供实验性的数据、模型和训练入口
- 世界模型：提供原生 Gaussian/Categorical RSSM、离线轨迹训练、开放环 rollout、动作评分和轻量规划
- 空间世界模型：俯视图占用建图、Qwen-VL 可选语义、多路线规划、空间 RSSM 重评分和响应式 Web 工作台

## 环境要求

- Python `>=3.10`
- PyTorch
- Transformers
- Datasets
- PEFT
- TRL
- NVIDIA GPU（真实模型训练建议使用）

部分后端能力还需要 DeepSpeed、bitsandbytes、xFormers、flash-attn 或其他可选依赖。

## 安装

建议在独立虚拟环境中使用：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

后训练建议使用仓库已验证的固定依赖栈，避免 Transformers、TRL 与 PEFT 的接口错配：

```powershell
python -m venv .venv-posttrain
.venv-posttrain\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 先按机器的 CUDA/CPU 环境安装合适的 PyTorch
python -m pip install -r requirements-posttrain.txt
python -m pip install -e .
saddle-llm doctor
```

`doctor` 的 `post_training.compatible` 为 `true` 后，再启动 SFT/DPO/KTO。QLoRA 还要求可用的 CUDA 与 bitsandbytes；普通 LoRA 可设置 `qlora: false`、`lora: true`。

验证安装：

```powershell
python -c "import saddle_llm; print(saddle_llm.__version__)"
saddle-llm --help
```

如果不安装 console script，也可以使用：

```powershell
python -m saddle_llm.cli --help
```

## 最短上手流程

### 1. 检查环境

```powershell
saddle-llm doctor
saddle-llm doctor --output build\doctor_report.json
```

### 2. 运行内置 smoke test

```powershell
saddle-llm smoke-test --in-process --skip-doctor `
  --work-dir build\smoke_e2e_fast
```

该命令会创建本地 tiny 模型并真实执行一小步 SFT、VLA、DPO、ORPO 和 KTO；不是只做导入检查。

只检查 fixture、数据和配置，不执行训练：

```powershell
saddle-llm smoke-test --skip-training
```

### 3. 检查训练数据

```powershell
saddle-llm inspect-data data\posttrain_sft_example.jsonl --task sft
saddle-llm inspect-data data\posttrain_preference_example.jsonl --task dpo
saddle-llm inspect-data data\posttrain_kto_example.jsonl --task kto
```

### 4. 检查并编译配置

仓库提供了以下模板：

- `configs/sft_lora.yaml`
- `configs/dpo_qlora.yaml`
- `configs/preflight.yaml`
- `configs/vla_sft.yaml`
- `configs/mopd_sft.yaml`
- `configs/posttrain_simple_flow.yaml`
- `configs/posttrain_release_flow.yaml`
- `configs/world_model.yaml`
- `configs/world_model_categorical.yaml`

### 极简后训练流程配置

SaddleLLM 已支持“极简主配置 + 可选覆盖”。普通后训练不需要手动填写阶段 ID、依赖关系和启用开关，只需选择流程并提供对应数据：

仓库中的三份 `posttrain_*_example.jsonl` 仅用于格式验证和 smoke test，正式训练必须替换为经过清洗、去重和划分的数据集。

```yaml
name: qwen-domain-v1
model: Qwen/Qwen3-4B
flow: sft+dpo

data:
  sft: data/posttrain_sft_example.jsonl
  preference: data/posttrain_preference_example.jsonl

output: outputs/qwen-domain-v1
```

系统根据 `flow` 自动展开为：

```text
数据检查 → SFT → DPO → 标准评测 → 发布门禁 → HF 发布包
```

当前可直接执行的流程包括：

- `sft`：数据检查、SFT 和默认评测；
- `sft+dpo`：SFT 后执行成对偏好优化；
- `sft+kto`：SFT 后执行正负反馈优化；
- `dpo` / `kto`：从已有模型直接执行偏好优化；
- `eval`：只评测已有 checkpoint；
- `export`：把已有 checkpoint 或 LoRA adapter 导出为 HF 发布包；
- `full`：展开为 `sft+dpo+eval+export`，也可通过 `alignment: kto` 改用 KTO。

`serve` 是独立的常驻命令，不放进训练 `flow`；训练完成后使用 `serve-model` 启动。`grpo` 的极简流程简写仍在规划中，当前 CLI 会明确拒绝未接入的阶段。原有 Orchestrator 中的 ORPO 仍可使用完整配置运行。

需要调整常用参数时再增加对应配置块：

```yaml
name: qwen-domain-v1
model: Qwen/Qwen3-4B
flow: sft+kto

data:
  sft: data/posttrain_sft_example.jsonl
  preference: data/posttrain_kto_example.jsonl

train:
  qlora: true
  epochs: 2
  batch_size: 4

kto:
  beta: 0.1
  batch_size: 2

evaluation:
  suite: standard
  fail_on_error: true

output: outputs/qwen-domain-v1
```

低频参数统一放入 `advanced`，使日常配置保持简洁：

```yaml
advanced:
  gradient_accumulation_steps: 8
  max_sequence_length: 4096
  trust_remote_code: false
  resume: true
```

流程编译器会推导 SFT/偏好/评测依赖、应用默认参数，并在 `validate-config` 和真实训练前检查数据格式。`resume` 默认为 `false`；只有显式设为 `true` 或 checkpoint 路径时才续训。`plan` 或 `train --compiled-output` 会保存标准 Orchestrator 配置，便于审计和复现。

```powershell
saddle-llm validate-config configs\posttrain_simple_flow.yaml
saddle-llm plan configs\posttrain_simple_flow.yaml
saddle-llm train configs\posttrain_simple_flow.yaml --dry-run
saddle-llm train configs\posttrain_simple_flow.yaml
```

### 5. Dry-run 和训练

```powershell
saddle-llm train configs\sft_lora.yaml --dry-run
saddle-llm train configs\sft_lora.yaml
```

## 后训练功能与使用方法

### 当前可用功能

| 功能 | 状态 | 推荐入口 | 作用 |
| --- | --- | --- | --- |
| 环境诊断 | 可用 | `doctor` | 检查 CUDA、显存、PyTorch 及后训练依赖兼容性 |
| 数据检查 | 可用 | `inspect-data` | 检查格式、可用样本、重复项和无效偏好对 |
| 数据规范化 | 可用 | Python API | 将 Alpaca、Messages、ShareGPT 和常见偏好格式转为统一 JSONL |
| 静态预检 | 可用 | `preflight` | 不加载模型，生成数据报告、训练估算和 Orchestrator 配置 |
| 极简流程编译 | 可用 | `validate-config` / `plan` | 将 `flow` 配置展开成可审计的标准配置和启动脚本 |
| SFT | 可用 | `flow: sft` | 支持 Full、LoRA 和 QLoRA |
| DPO | 可用 | `flow: dpo` / `sft+dpo` | 使用 chosen/rejected 偏好对训练 |
| KTO | 可用 | `flow: kto` / `sft+kto` | 使用正负反馈标签训练 |
| ORPO | 可用 | 完整 Recipe/Orchestrator 配置 | 尚未加入极简 `flow` 简写 |
| 困惑度评测 | 可用 | `flow: eval` 或 `evaluation` | 可使用本地文件或 Hugging Face 数据集 |
| 评测发布门禁 | 可用 | `evaluation.gate` | 按指标 `min/max` 阈值决定模型能否发布 |
| HF 模型导出 | 可用 | `flow: export` / `export-model` | 合并 LoRA、校验权重并生成发布清单与文件指纹 |
| OpenAI-compatible API | 可用 | `serve-model` | 提供健康检查、模型列表、Chat Completions 与 Completions |
| 断点恢复 | 可用 | `advanced.resume` | 从各阶段已有 checkpoint 继续训练 |
| 真实微型回归 | 可用 | `smoke-test` | 一步跑通 SFT、VLA、DPO、ORPO 和 KTO |

### 选择 Full、LoRA 或 QLoRA

| 方式 | `lora` | `qlora` | 特点 |
| --- | ---: | ---: | --- |
| Full fine-tuning | `false` | `false` | 更新全部参数，显存和磁盘开销最大 |
| LoRA | `true` | `false` | 更新适配器，兼容性好，适合常规单卡训练 |
| QLoRA | `true` | `true` | 4-bit 基座加 LoRA，显存更省，需要 CUDA 和 bitsandbytes |

配置示例：

```yaml
train:
  lora: true
  qlora: false
  epochs: 2
  batch_size: 1
  learning_rate: 0.0002
  max_steps: -1
```

`max_steps: -1` 表示按 `epochs` 训练；设置为正数后以 step 数为准。QLoRA 条件不满足时会直接报出可操作的错误，不会静默退化成普通 LoRA。

### 推荐的完整 CLI 顺序

```powershell
# 1. 检查运行环境
saddle-llm doctor --output build\doctor_report.json

# 2. 检查实际训练数据
saddle-llm inspect-data data\posttrain_sft_example.jsonl --task sft
saddle-llm inspect-data data\posttrain_preference_example.jsonl --task dpo

# 3. 可选：不加载模型的资源与数据预检
saddle-llm preflight data\posttrain_sft_example.jsonl `
  --stage sft --method qlora --base-model Qwen/Qwen3-4B `
  --root-dir build\preflight_sft

# 4. 编译并严格验证极简配置；默认同时检查本地数据
saddle-llm validate-config configs\posttrain_simple_flow.yaml `
  --output build\posttrain_validation.json

# 5. 生成标准配置、启动计划和 PowerShell 启动脚本
saddle-llm plan configs\posttrain_simple_flow.yaml `
  --output build\posttrain_compiled.yaml `
  --launch-plan build\posttrain_launch_plan.json `
  --launch-script build\posttrain_launch.ps1

# 6. 只编译和展示阶段，不下载模型、不训练
saddle-llm train configs\posttrain_simple_flow.yaml --dry-run `
  --compiled-output build\posttrain_compiled.yaml

# 7. 正式训练；失败时 --debug 可显示完整 traceback
saddle-llm train configs\posttrain_simple_flow.yaml `
  --compiled-output build\posttrain_compiled.yaml --debug
```

`validate-config` 失败时不要启动训练。`plan` 只生成产物，不会训练；真正执行入口是 `train` 或生成的 `posttrain_launch.ps1`。

### 极简配置常用字段

| 配置块 | 常用字段 | 说明 |
| --- | --- | --- |
| 顶层 | `name`、`model`、`flow`、`seed`、`output` | 实验名、基座、阶段组合和输出目录 |
| `data` | `sft`、`preference` | SFT 与 DPO/KTO 数据路径 |
| `train` | `lora`、`qlora`、`epochs`、`max_steps`、`batch_size`、`learning_rate` | 通用训练参数 |
| `dpo` / `kto` | `beta`、`learning_rate`、`epochs`、`batch_size` | 偏好阶段覆盖参数 |
| `evaluation` | `enabled`、`suite`、`dataset`、`max_samples`、`fail_on_error`、`gate` | 训练后评测与发布阈值 |
| `export` | `output_dir`、`format`、`merge_lora`、`require_gate`、`hash_weights`、`overwrite` | HF 或原生 Saddle 发布包配置 |
| `advanced` | `gradient_accumulation_steps`、`max_sequence_length`、`bf16`、`fp16`、`local_files_only`、`resume` | 低频和硬件相关参数 |

KTO 的实际 batch size 必须大于 1；单卡时建议至少设置 `kto.batch_size: 2`。`trust_remote_code` 默认关闭，需要自定义模型代码时再显式开启。

ORPO 当前使用完整 Recipe 配置。可以复制 `configs/dpo_qlora.yaml`，将方法改为：

```yaml
stage: preference
method:
  type: qlora
  preference_method: orpo
```

然后照常执行 `validate-config`、`train --dry-run` 和 `train`。内置 smoke test 已覆盖真实的一步 ORPO 训练。

### 使用本地数据评测

默认 `suite: standard` 当前映射到 perplexity。默认数据集是在线的 `wikitext`；离线环境或领域模型建议显式提供本地留出集：

```yaml
evaluation:
  enabled: true
  suite: [perplexity]
  dataset: data/domain_eval.jsonl
  max_samples: 500
  fail_on_error: true
```

`fail_on_error: true` 会让评测失败直接终止流程，防止把没有评测结果的模型误认为成功。设置为 `false` 时会继续流程，但 `summary.json` 中的评测状态仍是 `failed`。

### 评测门禁、导出与推理服务

完整闭环可直接使用 `configs/posttrain_release_flow.yaml`。其中 `perplexity.max` 只是演示阈值，正式项目应先测量基座与历史稳定版本，再按领域留出集设定阈值：

```yaml
flow: full

evaluation:
  enabled: true
  suite: [perplexity]
  dataset: data/domain_eval.jsonl
  max_samples: 500
  fail_on_error: true
  gate:
    enabled: true
    rules:
      perplexity:
        max: 30.0
    require_all: true
    fail_on_rejection: true

export:
  output_dir: outputs/qwen-domain-v1/release
  merge_lora: true
  require_gate: true
  safe_serialization: true
```

规则支持多个指标，每个指标可设置 `min`、`max` 和 `required`。`require_all: true` 表示所有必需规则都通过才可发布。门禁拒绝时，流程返回失败，`release_gate.json` 和 `summary.json` 仍会保存拒绝原因；导出阶段不会执行。

执行完整发布流程：

```powershell
saddle-llm validate-config configs\posttrain_release_flow.yaml
saddle-llm train configs\posttrain_release_flow.yaml --dry-run
saddle-llm train configs\posttrain_release_flow.yaml --debug
```

也可以不训练，单独导出已有完整 checkpoint 或 LoRA adapter。LoRA 默认会读取 `adapter_config.json` 中声明的基座并执行合并：

```powershell
saddle-llm export-model outputs\qwen-domain-v1\preference_checkpoints `
  outputs\qwen-domain-v1\release `
  --gate-result outputs\qwen-domain-v1\release_gate.json --require-gate
```

发布目录必须为空；确实要替换旧发布包时显式增加 `--overwrite`。当前稳定导出格式是 Hugging Face，ONNX、GGUF 和量化发布尚未作为稳定能力开放。默认只给配置、分词器等小文件计算 SHA-256；需要连大权重一起校验时增加 `--hash-weights`。

训练与导出完成后，独立启动 OpenAI-compatible 服务：

```powershell
saddle-llm serve-model outputs\qwen-domain-v1\release `
  --host 127.0.0.1 --port 8000 --device auto
```

需要鉴权时只传环境变量名，不把密钥写进命令历史或配置文件：

```powershell
$env:SADDLELLM_API_KEY = "replace-with-a-private-token"
saddle-llm serve-model outputs\qwen-domain-v1\release `
  --api-key-env SADDLELLM_API_KEY
```

调用 Chat Completions：

```powershell
$headers = @{ Authorization = "Bearer $env:SADDLELLM_API_KEY" }
$body = @{
  model = "release"
  messages = @(@{ role = "user"; content = "请总结这段材料" })
  temperature = 0.2
  max_tokens = 256
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/chat/completions `
  -Headers $headers -ContentType "application/json" -Body $body
```

可用端点为 `GET /health`、`GET /v1/models`、`POST /v1/chat/completions` 和 `POST /v1/completions`。当前稳定版是非流式单机推理；`--max-concurrency` 提供并发上限，`--max-input-tokens` 与 `--max-new-tokens` 提供上下文和输出保护。

### 断点恢复

新任务默认 `resume: false`。已有阶段 checkpoint 时，可以让每个阶段自动寻找自己输出目录下的最近 checkpoint：

```yaml
advanced:
  resume: true
```

单阶段流程也可以指定明确路径：

```yaml
flow: sft
advanced:
  resume: outputs/qwen-domain-v1/sft_checkpoints/checkpoint-200
```

多阶段流程使用明确路径时应拆成单阶段恢复，避免把 SFT checkpoint 误传给偏好阶段。使用 `resume: true` 前必须确保相应阶段目录中确实存在 checkpoint。

### 训练产物

以 `output: outputs/qwen-domain-v1` 为例：

```text
outputs/qwen-domain-v1/
  training.log
  release_gate.json
  plans/
    sft_plan.json
    preference_plan.json
  sft_checkpoints/
  preference_checkpoints/
  release/
    config.json
    model.safetensors
    release_manifest.json
    SADDLELLM_RELEASE.md
  summary.json
  training_summary.json
```

`summary.json` 记录阶段状态、模型路径、数据检查、训练结果、评测指标、门禁结论和耗时。`release_manifest.json` 记录来源 checkpoint、LoRA 是否合并、门禁快照、文件清单和哈希。极简配置编译后的标准配置由 `--compiled-output` 指定，建议和训练产物一起归档。

### 常见问题

- 显存不足：先降低 `max_sequence_length` 和 `batch_size`，再提高 `gradient_accumulation_steps`，或改用 LoRA/QLoRA。
- QLoRA 报错：确认 `doctor` 显示 CUDA 可用，并检查 bitsandbytes 与当前 PyTorch/CUDA 是否匹配。
- KTO 校验失败：保证正负样本都存在，并让实际 batch size 大于 1。
- DPO 无效：检查 `chosen` 与 `rejected` 是否相同、偏好方向是否一致，以及 prompt 是否泄漏答案。
- 评测下载失败：把 `evaluation.dataset` 改成本地 JSONL/JSON/TXT 文件。
- 出现 AutoAWQ 兼容警告：说明隔离环境看到了不可用的可选量化包；普通 LoRA 会继续运行，若要导出 AWQ 模型应另建匹配 AutoAWQ 与 Transformers 的专用环境。
- 只看到简短错误：给 `train` 增加 `--debug`。

机械跑通流程并不能保证模型效果。正式实验至少应保留独立验证集，先记录基座指标，再分别比较 SFT 与偏好训练后的指标，并保存数据版本、编译配置、随机种子和 checkpoint。

## LLM 网络积木与残差初始化

原生 `saddle` 模型支持两条等价构建路径：YAML/JSON 配置和 Python Builder。两条路径都会先归一化为 `ModelBlueprint`，再创建实际网络，因此配置实验与代码实验可以互相保存、比较和复现。

完整训练配置见 [`configs/modular_llm.yaml`](configs/modular_llm.yaml)，其中 `model.blueprint.layers` 按顺序描述层段，`repeat` 展开为彼此独立、不共享参数的 decoder layers。每段可以覆盖：

- Attention：`mha`、`mqa`、`gqa`、`mla`
- FFN：`swiglu`、`moe`
- 残差拓扑：`serial`（Attention 后再 FFN）或 `parallel`（两分支读取同一输入）
- 残差缩放：`attention_scale`、`ffn_scale`，以及 `learnable: true` 的可学习标量门控
- 残差 dropout 与输出投影初始化：`standard`、`depth_scaled`、`zero`

`depth_scaled` 仅按 `1 / sqrt(2 * 层数)` 缩放 Attention `o_proj` 和 FFN/MoE `down_proj`；`zero` 仅清零这些残差分支输出投影。默认 `standard + learnable: false` 保持原行为，也不会增加 state-dict 键。

Python 组合示例：

```python
from saddlellm import SaddleModelBuilder

builder = (
    SaddleModelBuilder("hybrid", hidden_size=512, vocab_size=32000)
    .defaults(
        attention={"preset": "gqa", "num_heads": 8, "num_kv_heads": 2},
        ffn={"preset": "swiglu", "intermediate_size": 1536},
        residual={"preset": "depth_scaled"},
    )
    .add_layers(4, name="dense-stem")
    .add_layer(
        name="moe-tail",
        attention={"preset": "mla", "kv_lora_rank": 32},
        ffn={"preset": "moe", "num_experts": 8, "experts_per_token": 2},
        residual={"topology": "parallel", "learnable": True},
    )
)

blueprint = builder.build_blueprint()
model = builder.build()
```

更完整的可执行脚本见 [`examples/build_modular_llm.py`](examples/build_modular_llm.py)。原生 checkpoint 会保存 `saddle_config.json`、`model_blueprint.json`、`saddle_checkpoint.json` 和权重。推荐使用统一入口加载，它会自动识别原生目录或普通 Hugging Face 模型：

```python
from saddlellm import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer("./outputs/final_model", device="auto")
inputs = tokenizer("你好，世界", return_tensors="pt")
tokens = model.generate(**inputs, max_new_tokens=32, do_sample=False)
```

原生生命周期当前支持 scratch 预训练、weights-only 继续预训练、Trainer 精确恢复、perplexity/文本生成评测、本地 OpenAI 兼容推理服务，以及 `format: saddle` 自包含导出。两种续训语义需要区分：

- `pretrain_mode: continue` + `model.name_or_path`：只继承模型权重，优化器和步数重新开始。
- `training.resume_from_checkpoint: checkpoint-N`（或 `true` 自动选择最新 checkpoint）：恢复模型、优化器、调度器、随机数状态和 global step。

原生模型尚未接入 PEFT/LoRA、QLoRA、ORPO/KTO 和 PPO/GRPO；原生偏好优化当前支持全参数 DPO。未支持组合会在预检阶段明确拒绝。原生发布应设置 `export.format: saddle`，不会将异构逐层架构伪装成有损的通用 HF `config.json`。

当前原生链路支持网络构建、tokenizer、scratch/continue pretrain、全参数 SFT、全参数 DPO、统一评测、Saddle 格式导出和本地服务。`backend: hf` 不会消费 Saddle Blueprint，因此配置会在预检阶段拒绝该组合。原生 LoRA/QLoRA、ORPO/KTO、分布式后训练和无损 HF 格式导出仍未开放；这些组合会在训练前给出明确错误。完整 SFT+DPO 示例见 `configs/native_posttrain.yaml`。

生成式多模态采用独立 codec/latent/objective 路径，不把图像像素或音频码强行映射到文本词表。统一数据 schema 支持 `text/image/audio/video` segments、时间区间以及可选的 `observations/actions/rewards/dones/timestamps`；旧的 `messages + images` 数据仍兼容。`image_generation` 消费 `latents[N,C,H,W] + conditions[N,D]` 并训练图像 latent flow；`music_generation` 消费多码本 `codes[N,Q,T] + conditions[N,D]` 并训练自回归音频码模型；`video_generation` 消费 `video_latents[N,C,T,H,W] + conditions[N,D]` 并训练 3D patch 时空 flow。`media_cache` stage 已能从原始图片、PCM WAV、帧目录或动图生成带 SHA-256 指纹的可续建 NPZ 分片，并直接把输出交给三个生成 stage。默认 Codec 是端到端基线，生产质量仍应替换为真实 VAE、神经音频 Codec 与因果视频 VAE。配置见 `configs/media_cache_*.yaml`、`configs/raw_image_to_training.yaml` 和三个生成配置。

## 世界模型使用方法

RSSM/离散 RSSM 现在也可作为总编排器的 `world_model` stage 运行；其统一 episode 数据可直接使用 `observations[T+1] + actions/rewards/dones[T]`。原有专用 CLI 保持兼容：

统一编排示例见 `configs/world_model_stage.yaml`：

```powershell
saddle-llm train configs\world_model_stage.yaml --dry-run
saddle-llm train configs\world_model_stage.yaml
```

原有独立轨迹训练入口：

```powershell
saddle-llm train-world-model configs\world_model.yaml --dry-run
saddle-llm train-world-model configs\world_model.yaml
saddle-llm train-world-model configs\world_model_categorical.yaml --dry-run
saddle-llm world-model-backends
saddle-llm infer-world-model outputs\world_model_example data\world_model_inference_example.json
```

世界模型训练与推理均使用 `saddlellm/` 内的原生实现；`reference/` 仅用于架构研究，
不会被这些命令导入或启动。

空间世界模型的完整数据、训练和可视化链路：

```powershell
saddle-llm build-spatial-world-data data\spatial_floorplan_example.pbm `
  --output data\spatial_world_model_example.jsonl --episodes 64 --routes-per-pair 2
saddle-llm train-world-model configs\spatial_world_model.yaml --dry-run
saddle-llm train-world-model configs\spatial_world_model.yaml
saddle-llm spatial-studio --world-model outputs\spatial_world_model_example
```

打开 `http://127.0.0.1:7865` 上传俯视图、设置起终点并查看空间结构、候选路线与模型评分。详见[空间世界模型指南](SPATIAL_WORLD_MODEL_GUIDE.md)。

### WorldAgent：让视觉、规划、世界模型和语言解释协同

`WorldAgent` 是位于现有空间模块之上的统一运行时，不是从 `reference/` 复制来的第三方 Agent。它把一次任务拆成可单独调用、可保存和可复现的四个阶段：

```text
图片 / 地图
   -> analyze：几何提取 + 可选 Qwen-VL 语义识别
   -> plan：结构化 WorldState + Top-K 路线 + 可选 RSSM 重排
   -> simulate：硬几何碰撞检查 + 可选世界模型想象
   -> feedback：实际轨迹对比 + 事件记忆 + 可训练 replay JSONL
```

其中 Qwen-VL 是可替换的视觉语义适配器，负责房间、门、障碍物、危险物和连通关系；SaddleLLM 原生 RSSM 才是学习状态转移、奖励、持续概率、碰撞或未来占用的世界模型。没有配置 Qwen-VL 时仍能规划干净的俯视占用图；没有世界模型 checkpoint 时仍能做确定性的栅格验证，但输出会明确标记为 `geometry`，不会伪装成学习预测。

用内置地图一条命令跑通分析、规划和模拟：

```powershell
saddle-llm world-agent-run data\spatial_floorplan_example.pbm `
  --start 2,2 --goal 21,13 `
  --config configs\world_agent.yaml `
  --simulation geometry `
  --output build\world_agent_demo.json
```

结果包含 `saddle.world-state.v1`、`saddle.world-plan.v1` 和 `saddle.world-simulation.v1` 三种稳定协议。占用栅格使用 `rle-v1` 压缩持久化；计划中包含每条路线的长度、转弯数、净空、几何风险、源图坐标、选择原因、置信度和限制说明。

最简配置只需要工作目录；模型均为可选：

```yaml
workspace: ../outputs/world_agent
qwen_model: null
world_model_checkpoint: null
device: auto
```

完整但仍较短的模板见 [`configs/world_agent.yaml`](configs/world_agent.yaml)。需要语义识别和学习预测时设置模型路径：

```yaml
qwen_model: Qwen/Qwen3-VL-4B-Instruct
world_model_checkpoint: ../outputs/spatial_world_model_example
```

然后运行：

```powershell
saddle-llm world-agent-run floorplan.png `
  --start 120,80 --goal 930,620 `
  --config configs\world_agent.yaml `
  --semantic-backend qwen-vl `
  --use-world-model --simulation world_model
```

也可以启动独立 API；原来的 `spatial-studio` 同样会挂载这些端点，并共享已经加载的 Qwen-VL 与 RSSM：

```powershell
saddle-llm world-agent-api --config configs\world_agent.yaml
# 或打开带可视化界面的服务
saddle-llm spatial-studio --qwen-model Qwen/Qwen3-VL-4B-Instruct `
  --world-model outputs\spatial_world_model_example
```

API 文档位于 `http://127.0.0.1:7866/docs`。稳定端点为：

- `POST /v1/world/analyze`：上传图片，返回可持久化的 `WorldState`。
- `POST /v1/world/plan`：传 `state_id`、起点和终点，返回多条路线与推荐解释。
- `POST /v1/world/simulate`：选择 `geometry`、`world_model` 或自动回退的 `auto`。
- `POST /v1/world/feedback`：提交结果和实际路径，生成评估记录及训练 replay。
- `GET /v1/world/memory`：查看状态、计划、模拟、反馈与 replay 数量。

Python 中可以直接复用同一运行时：

```python
from saddlellm import WorldAgentRuntime, WorldAgentSettings

agent = WorldAgentRuntime(
    WorldAgentSettings.from_file("configs/world_agent.yaml")
)
state = agent.analyze_image("data/spatial_floorplan_example.pbm")
plan = agent.plan(state.state_id, start=(2, 2), goal=(21, 13), route_count=3)
simulation = agent.simulate(plan.plan_id, mode="geometry")

# 执行后把真实轨迹反馈回来；只有真实路径才进入训练 replay。
feedback = agent.record_feedback(
    plan.plan_id,
    outcome="success",
    actual_path=plan.routes[0]["points"],
    simulation_id=simulation.simulation_id,
)
```

默认产物目录如下：

```text
outputs/world_agent/
  observations/                  上传图片
  states/                        压缩 WorldState
  plans/                         路线、解释和模型使用信息
  simulations/                   几何验证与学习预测
  feedback/                      实际执行结果
  events.jsonl                   追加式事件索引
  replay/spatial_feedback.jsonl  可直接被原生世界模型数据加载器读取
```

要利用真实反馈继续训练，可把 `configs/spatial_world_model.yaml` 的 `data.train_path` 改为上述 replay 文件，先执行 `train-world-model --dry-run` 验证，再训练新 checkpoint。生产环境应把合成地图数据和真实反馈混合，而不是只用少量在线样本覆盖原训练集。

当前边界：单张透视照片只能提供可见区域语义，不能恢复被遮挡空间；真实行走还需要深度、相机标定、定位/SLAM、动态障碍感知和机器人控制接口。所有 `confidence.calibrated` 当前均为 `false`，几何可行或模型高分都不是现实安全保证。

训练失败时显示完整 traceback：

```powershell
saddle-llm train configs\sft_lora.yaml --debug
```

推荐顺序：

```text
doctor
  -> smoke-test --skip-training
  -> inspect-data
  -> validate-config
  -> preflight
  -> plan
  -> train --dry-run
  -> train
  -> release gate
  -> export
  -> serve-model
  -> report
```

## Python API

### 创建领域训练工厂

```python
from saddle_llm import LLMTrainingFactory

factory = LLMTrainingFactory.for_domain(
    domain="general",
    root_dir="./llm_factory",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    global_batch_size=16,
    num_gpus=1,
    gpu_memory_gb=24.0,
)

factory.create_workspace()
factory.save_plan()
```

### 创建 SFT 计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/posttrain_sft_example.jsonl",
    stage="sft",
    method="qlora",
    max_steps=1000,
    save=True,
)
```

该调用会生成规范化数据、Recipe、Orchestrator config 和训练计划。

### 创建 DPO 计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/posttrain_preference_example.jsonl",
    stage="dpo",
    method="qlora",
    beta=0.1,
    save=True,
)
```

### 执行 Orchestrator 配置

```python
from saddle_llm import TrainingOrchestrator

orchestrator = TrainingOrchestrator.from_yaml("./build/posttrain_compiled.yaml")
result = orchestrator.run()
```

先通过 `saddle-llm plan ... --output build/posttrain_compiled.yaml` 生成标准配置；`run()` 会立即开始真实训练。

### 用 Python 编译并执行极简 flow

```python
from pathlib import Path

import yaml

from saddle_llm import (
    SimpleFlowCompiler,
    TrainingOrchestrator,
    validate_training_config,
)

config_path = Path("configs/posttrain_simple_flow.yaml")
raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
compiled = SimpleFlowCompiler.compile(raw, base_dir=str(config_path.parent))

report = validate_training_config(compiled, inspect_data=True)
if not report["valid"]:
    raise RuntimeError(report["issues"])

result = TrainingOrchestrator.from_dict(compiled).run()
print(result)
```

如果只想编译和检查，到 `report` 为止即可；最后两行会加载模型并开始真实训练。正式运行前请替换仓库中的示例数据。

### 用 Python 规范化后训练数据

```python
from saddle_llm import normalize_post_training_file

report = normalize_post_training_file(
    input_path="data/posttrain_sft_example.jsonl",
    output_path="build/normalized_sft.jsonl",
    task="sft",  # 也可使用 dpo、orpo、kto 或 grpo
)
print(report)
```

规范化会同时生成 `normalized_sft.jsonl.report.json`，记录保留、丢弃和识别到的 schema 数量。

## 数据格式

### SFT：Alpaca JSONL

```json
{"instruction":"总结下面的政策意见","input":"...","output":"..."}
```

### SFT：Messages JSONL

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

### Preference：DPO/ORPO JSONL

```json
{"prompt":"...","chosen":"更好的回答","rejected":"较差的回答"}
```

### KTO JSONL

```json
{"prompt":"...","completion":"可接受或不可接受的回答","label":true}
```

KTO 数据必须同时包含正例与负例；`label` 可使用布尔值，训练前建议通过 `inspect-data --task kto` 检查。

在加载模型前运行 `inspect-data`，可以检查字段、重复样本、长度和 `chosen == rejected` 等问题。


## 测试

## 分布式预训练

当前原生 Saddle 模型支持单机/多机 DDP 预训练，并提供 FSDP 配置入口。先编译
配置和启动脚本，再由启动器创建所有 worker：

```yaml
stages: [pretrain]
model:
  backend: saddle
distributed:
  strategy: ddp       # single | ddp | fsdp
  num_gpus: 4         # 每个节点的进程数
  num_nodes: 1
  ddp_backend: auto   # CUDA 使用 nccl，CPU/Gloo 测试使用 gloo
```

```powershell
saddle-llm plan recipe.yaml --output build/distributed.yaml
& ./build/launch.ps1
```

分布式模式目前只运行 `pretrain`。评估和导出应在训练完成后，用单进程读取
`final_model` 单独执行。TP/PP/CP/EP 仍属于规划能力，运行时会明确拒绝而不会
静默退化。原生 checkpoint 会记录并行策略、world size 和 state-dict 类型；
精确 DDP 恢复要求每个 rank 的 RNG 状态齐全。目前 FSDP/DeepSpeed 的分片恢复
尚未开放。

```powershell
python -m pytest -q
```

当前项目级 `pytest.ini` 会排除 `build/`、`dist/` 和 `external_research/`，避免测试发现过程进入第三方研究仓库。

## 目录结构

```text
configs/                    训练 Recipe 模板
experiments/                可执行实验
outputs/                    实验输出
spatial-studio/             React 空间世界模型工作台源码
saddlellm/                  Python 主包源码
saddle_llm/                 旧包名兼容层
docs/                       架构和函数参考
tools/                      可重复运行的文档生成工具
tests/                      项目测试
SADDLE_LLM_FULL_DOCUMENTATION.md
SADDLE_LLM_CODE_GUIDE.md
TRAINING_GUIDE.md
WORLD_MODEL_TRAINING_GUIDE.md
```

## 文档导航

- [架构、模块逻辑与调用时序](docs/ARCHITECTURE.md)
- [逐函数与方法参考](docs/FUNCTION_REFERENCE.md)
- [完整文档](SADDLE_LLM_FULL_DOCUMENTATION.md)
- [训练指南](TRAINING_GUIDE.md)
- [世界模型训练指南](WORLD_MODEL_TRAINING_GUIDE.md)
- [空间世界模型与可视化工作台](SPATIAL_WORLD_MODEL_GUIDE.md)
- [代码学习指南](SADDLE_LLM_CODE_GUIDE.md)

## 还可以继续建设的后训练能力

评测门禁、HF 导出与 OpenAI-compatible 推理服务已经形成可执行闭环。后续建议按以下顺序继续：

1. GRPO/RLVR 流程接入：把现有 native GRPO、奖励函数和 rollout 数据检查接入 `flow: sft+grpo`。
2. checkpoint 对比评测：自动对比基座、SFT、DPO/KTO 多个 checkpoint，而不只判断单个结果是否过线。
3. 推理性能：增加 SSE 流式输出、连续批处理、KV Cache 管理、超时和取消机制。
4. 更多发布格式：在可重复验证后开放量化、ONNX 和 GGUF 导出。
5. 实验追踪与可视化：统一展示 loss、reward、KL、吞吐、显存、评测变化和 checkpoint 对比。
6. 自动数据闭环：清洗、去重、难例挖掘、拒答样本、训练/验证集切分和数据版本管理。


## 当前边界

- CLI、数据检查、Recipe、Orchestrator 和 smoke-test 是推荐入口。
- 发布闭环当前稳定支持 HF 格式和非流式单机 API；量化导出、SSE 与动态批处理仍待实现。
- native GRPO 适合小规模研究和链路验证；大规模实验应进一步验证吞吐、分布式和 checkpoint 行为。
- VLA、世界模型、多模态、部分高级架构及部分第三方后端仍属于实验性能力。

## License

[MIT](LICENSE)
