# SaddleLLM

当前版本：`2.35`

SaddleLLM 是一个以配置驱动的模型训练与运行工具箱。用户准备数据、模型和一份训练配置，系统负责检查输入、安排训练阶段、调用对应训练器，并保存 checkpoint、评测结果和运行状态。

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
| RLHF | `rlhf` | DPO 可执行；PPO 目前只返回未接入状态 |
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

非单进程的 DDP、FSDP 和 DeepSpeed 路径，目前只对 `model.backend: saddle` 的 `pretrain` 阶段声明为可执行。其他内置阶段使用 `distributed.strategy: single`。

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
| 推理 API | `python -m pip install -e ".[serve]"` |
| 空间智能与眼动控制 | `python -m pip install -e ".[spatial]"` |
| 开发和测试 | `python -m pip install -e ".[dev]"` |

如果要严格使用仓库验证过的后训练版本：

```powershell
python -m pip install -r requirements-posttrain.txt
```

安装完成后确认命令可见：

```powershell
saddle-llm --help
```

也可以不用 console script：

```powershell
python -m saddlellm.cli --help
```

### 2.2 第一次运行

创建一个 10 步 LoRA SFT 小项目：

```powershell
saddle-llm quickstart ".\my-training"
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
saddle-llm doctor
saddle-llm inspect-data data/sft.jsonl --task sft
saddle-llm validate-config config.yaml
saddle-llm train config.yaml --dry-run
saddle-llm train config.yaml
```

创建“SFT 后接 DPO”的项目：

```powershell
saddle-llm quickstart ".\my-training" --stages sft,preference --preference-method dpo
```

使用自己的数据和模型：

```powershell
saddle-llm quickstart ".\my-training" --model "D:\models\my-model" --data "D:\datasets\sft.jsonl"
```

`quickstart` 不会下载模型，也不会开始训练。它只创建配置、示例数据和下一步命令。目标文件已存在时默认拒绝覆盖；`--force` 只覆盖它负责生成的文件。

### 2.3 训练配置怎么写

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

返回 `ready`、样本数、字段格式、文本长度、重复样本和阻断问题。数据不合格时退出码为 1。

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
| `train` | 执行训练配置 | `saddle-llm train config.yaml` | 各阶段结果和输出路径 |
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

`spatial-studio`、`world-agent-api` 和 `serve-model` 是持续运行的服务命令，不会像普通命令一样立即返回。按 `Ctrl+C` 停止。

### 2.9 评测、发布和服务命令

| 命令 | 做什么 | 最小用法 | 主要结果 |
|---|---|---|---|
| `report` | 汇总一个训练输出目录 | `saddle-llm report outputs/my-run` | 预训练/运行报告 |
| `export-model` | 把 checkpoint 或 adapter 打包成可验证发布目录 | `saddle-llm export-model CHECKPOINT outputs/release` | 模型文件、manifest、可选权重哈希和门禁信息 |
| `serve-model` | 启动 OpenAI 兼容推理 API | `saddle-llm serve-model outputs/release` | `/v1/models`、`/v1/chat/completions`、`/v1/completions` |

要求发布门禁通过后才导出：

```powershell
saddle-llm export-model CHECKPOINT outputs/release --gate-result outputs/run/release_gate.json --require-gate
```

启动本地推理服务：

```powershell
saddle-llm serve-model outputs/release --host 127.0.0.1 --port 8000
```

如果需要 Bearer Token，不要把密钥直接写进命令：

```powershell
$env:SADDLE_API_KEY = "your-secret"
saddle-llm serve-model outputs/release --api-key-env SADDLE_API_KEY
```

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
5. 清洗、去重、质量过滤、tokenize 和 sequence packing。
6. 创建 Hugging Face `Trainer` 或原生 `SaddleTrainer`。
7. 执行 causal language modeling。
8. 保存 `final_model`、tokenizer 和 Trainer state。

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

training:
  max_steps: 1000

distributed:
  strategy: single

logging:
  output_dir: outputs/pretrain
```

完整原生模型结构示例见 [modular_llm.yaml](configs/modular_llm.yaml)。

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

执行过程：

1. 从上游 `pretrain` 或 `model.name_or_path` 找到模型。
2. 检查数据字段、空文本、重复样本和长度。
3. 生成 `plans/sft_plan.json`。
4. 根据 `use_lora/use_qlora` 选择全参数、LoRA 或 QLoRA。
5. HF 后端调用 `PeftSFTTrainer`；Saddle 后端调用原生 SFT。
6. 保存 `sft_checkpoints`，并返回新的 `model_path`。

示例见 [sft_lora.yaml](configs/sft_lora.yaml)。

### 3.6 偏好训练逻辑

`preference` 用于让模型在多个回答之间学习偏好。

| 方法 | 数据 | 作用 |
|---|---|---|
| DPO | `prompt/chosen/rejected` | 直接提高 chosen 相对 rejected 的概率 |
| ORPO | `prompt/chosen/rejected` | 将监督目标和偏好比率目标结合 |
| KTO | `prompt/completion/label` | 用可取/不可取标签学习偏好；实际 batch 必须大于 1 |

DPO/ORPO 数据：

```json
{"prompt":"怎样排错？","chosen":"先看日志并构造最小复现。","rejected":"直接忽略错误。"}
```

KTO 数据：

```json
{"prompt":"怎样排错？","completion":"先看日志。","label":true}
```

执行过程：

1. 优先取得 SFT 输出，其次使用预训练输出或基础模型。
2. 按选定方法检查数据。
3. 写入 `plans/preference_plan.json`。
4. 调用对应 DPO、ORPO 或 KTO Trainer。
5. 保存 `preference_checkpoints`，供评测或导出使用。

示例见 [posttrain.yaml](configs/posttrain.yaml) 和 [dpo_qlora.yaml](configs/dpo_qlora.yaml)。

`rlhf` 阶段当前可实际执行的稳定路径也是 DPO。PPO 需要 reward model 和在线 rollout 接线，目前会明确返回 `planned_unsupported`；需要离线偏好训练时直接使用 `preference` 更清楚。

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

任何阶段抛出异常时：

1. 记录 `failed_stage` 和错误。
2. 更新 `pipeline_state.json`。
3. 尽可能写出 `summary.json`。
4. 停止后续阶段。
5. CLI 返回非零退出码，不把失败包装成成功。

### 3.14 从哪里继续阅读

- [架构与调用链](docs/ARCHITECTURE.md)：模块关系和详细时序。
- [函数参考](docs/FUNCTION_REFERENCE.md)：从源码生成的函数、方法和行号。
- [训练指南](TRAINING_GUIDE.md)：训练检查清单。
- [多模态生成](docs/MULTIMODAL_GENERATION.md)：媒体缓存和生成训练。
- [统一训练与眼动控制](docs/UNIFIED_TRAINING_AND_EYE_CONTROL.md)：世界模型、控制和多分支执行。

开发验证：

```powershell
python -m compileall -q saddlellm tests
python tools/generate_function_reference.py --check
python -m pytest -q
```
