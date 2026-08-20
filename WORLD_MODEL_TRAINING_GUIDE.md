# SaddleLLM 世界模型训练指南

本模块提供两套由 SaddleLLM 自己实现、训练和推理的轻量 PyTorch 世界模型。
`reference/` 只用于研究网络设计，不会在训练或推理时被导入、调用或启动。世界模型
与现有 VLA 行为克隆互补：VLA 学习“观测/指令到动作”，世界模型学习“当前状态和
动作会让环境如何变化”。

## 1. 原生网络结构

通过根级 `backend` 选择网络：

- `rssm`：对角高斯随机状态，适合先跑通数据和训练链路；
- `categorical_rssm`：分类随机状态、概率均匀混合、分离的
  dynamics/representation KL、signed-log 观测和 two-hot 奖励头。

### 1.1 Gaussian RSSM

每个时间步包含两类潜状态：

- `deterministic_state`：GRU 保存的历史记忆；
- `stochastic_state`：对当前环境不确定性的对角高斯表示。

训练时，观测编码器和确定性状态形成 posterior；只使用上一潜状态与动作形成
prior。二者通过 KL 损失对齐。潜状态同时预测：

- 下一观测（MSE）；
- transition reward（MSE）；
- episode 是否继续（binary cross entropy）；
- prior/posterior KL（支持 KL balance 与 free nats）。

一维状态向量使用 MLP 编解码器；`[C, H, W]` 图像使用卷积编解码器。图像数据
应采用 CHW 排列，整数像素或数值大于 1 的像素会归一化到 `[0, 1]`。

### 1.2 Categorical RSSM

`CategoricalWorldModel` 使用 `stochastic_size × stochastic_classes` 个分类潜变量，
以 straight-through one-hot 样本参与递归动力学。确定性状态按
`recurrent_blocks` 分组更新；prior 和 posterior 分别承担 dynamics KL 与
representation KL。奖励使用 two-hot symlog 分类回归，向量观测默认在 symlog
空间重建。数据加载、网络模块、损失、训练循环、保存加载和推理规划均为 SaddleLLM
原生实现。

## 2. 轨迹数据格式

推荐一行一个 episode 的 JSONL：

```json
{"episode_id":"ep-1","observations":[[0,0],[0.1,0],[0.2,0.1]],"actions":[[0.1,0],[0.1,0.1]],"rewards":[0,1],"dones":[0,1]}
```

时间对齐必须满足：

- `len(observations) == T + 1`；
- `len(actions) == T`；
- 可选的 `rewards`、`dones` 长度均为 `T`，省略时默认为 0。

也支持一行一个 transition，加载器会按 `episode_id` 聚合：

```json
{"episode_id":"ep-1","step_id":0,"observation":[0,0],"action":[0.1,0],"next_observation":[0.1,0],"reward":0,"done":false}
```

支持 `.json`、`.jsonl`、`.npz`、`.pt` 和 `.pth`。NPZ 至少包含
`observations` 与 `actions` 数组，可再提供 `rewards`、`dones`。

连续动作采用 `[T, action_dim]`；离散动作采用 `[T]` 的整数 id，并在配置中设置
`action_type: discrete`。连续动作最好在写入数据前按每一维归一化。

## 3. 配置和命令行训练

仓库自带可直接运行的向量环境示例：

```powershell
saddle-llm train-world-model configs\world_model.yaml --dry-run
saddle-llm train-world-model configs\world_model.yaml
saddle-llm train-world-model configs\world_model_categorical.yaml --dry-run
saddle-llm train-world-model configs\world_model_categorical.yaml
```

`--dry-run` 会完整加载数据、按 episode 划分训练/验证集、检查形状、实例化网络并
报告参数量，但不会创建输出目录或执行优化。

原生训练配置由后端选择、模型、数据和训练四部分组成：

```yaml
backend: rssm  # 或 categorical_rssm

model:
  observation_shape: [4]   # 可省略并从数据推断
  action_dim: 2            # 可省略并从数据推断
  action_type: continuous
  deterministic_size: 128
  stochastic_size: 16
  kl_weight: 0.1

data:
  train_path: data/world_model_example.jsonl
  sequence_length: 32
  stride: 16
  validation_split: 0.1

training:
  output_dir: outputs/world_model
  num_epochs: 20
  batch_size: 16
  learning_rate: 0.0003
  mixed_precision: auto
  device: auto
```

数据切分发生在 episode 层级，避免同一条轨迹的重叠窗口同时落入训练集和验证集。
不足 `sequence_length` 的尾部会 padding，并通过 mask 从损失中排除。

## 4. Python API

从配置训练：

```python
from saddlellm import train_world_model

result = train_world_model("configs/world_model.yaml")
print(result["output_dir"])
```

直接构造网络：

```python
from saddlellm import WorldModel, WorldModelConfig

model = WorldModel(WorldModelConfig(
    observation_shape=(16,),
    action_dim=4,
    deterministic_size=256,
    stochastic_size=32,
))
```

分类 RSSM 可以直接构造，也可以通过统一注册表创建：

```python
from saddlellm import create_world_model

model = create_world_model("categorical_rssm", {
    "observation_shape": [16],
    "action_dim": 4,
    "deterministic_size": 128,
    "stochastic_size": 8,
    "stochastic_classes": 8,
    "recurrent_blocks": 8,
})
```

训练后的开放环预测：

```python
import torch
from saddlellm import load_world_model

model = load_world_model("outputs/world_model")  # 自动识别两种原生后端
model.eval()

# observations: [B, T+1, *observation_shape]
# actions:      [B, T, action_dim]
filtered = model(observations, actions, sample_state=False)

# future_actions: [B, horizon, action_dim]
future = model.imagine(filtered.final_state, future_actions, deterministic=True)
print(future.observations.shape, future.rewards.shape, future.continuation.shape)
```

`imagine()` 不读取未来真实观测，适合模型预测控制、候选动作序列评分和后续
actor-critic imagination training。

## 5. 原生推理与轻量规划

`WorldModelRuntime` 为两个后端提供同一套推理接口：

```python
import torch
from saddlellm import WorldModelPlannerConfig, WorldModelRuntime

runtime = WorldModelRuntime.from_pretrained("outputs/world_model", device="cpu")

# 只有当前观测时，直接形成 posterior belief。
belief = runtime.encode_observation([0.0, 0.0, 0.0, 0.0])

# 给定动作序列做开放环预测。
future = runtime.rollout(belief, [
    [0.1, 0.0],
    [0.1, 0.05],
    [0.0, 0.05],
])

# 对多条候选动作序列按折扣奖励和存活概率评分。
candidates = torch.randn(64, 6, 2).clamp(-1, 1)
evaluation = runtime.score_action_sequences(belief, candidates)

# 使用轻量 Cross-Entropy Method 规划；离散动作会自动使用分类分布搜索。
plan = runtime.plan(belief, WorldModelPlannerConfig(
    horizon=6,
    num_candidates=64,
    iterations=3,
    action_low=-1.0,
    action_high=1.0,
))
print(plan.action, plan.score)
```

有历史轨迹时使用 `runtime.filter(observations, actions)` 获得最新 belief；也可用
`runtime.predict(observations, history_actions, future_actions)` 一次完成历史滤波和未来
rollout。CLI 接受 JSON/YAML 请求：

```powershell
saddle-llm infer-world-model outputs\world_model_example `
  data\world_model_inference_example.json --device cpu
```

请求可包含单个 `observation`，或成对的 `observations`/`actions` 历史；然后提供
`future_actions`、`planner`，或同时提供两者。

### 5.1 空间导航专用数据与评分

空间工作台复用同一个训练器，但通过 `SpatialObservationEncoder` 固定训练/推理契约。
仓库提供从俯视图生成可训练 episode 的入口和匹配配置：

```powershell
saddle-llm build-spatial-world-data data\spatial_floorplan_example.pbm `
  --output data\spatial_world_model_example.jsonl `
  --episodes 64 --routes-per-pair 2
saddle-llm train-world-model configs\spatial_world_model.yaml
saddle-llm spatial-studio --world-model outputs\spatial_world_model_example
```

`spatial_features_v1` 的前 16 维依次为当前位置、目标位置、相对位移、直线距离、
朝向正余弦、FREE/BLOCKED/UNKNOWN 比例、起终点净空、地图宽高比和分辨率。
自动路线评分要求 `observation_shape: [16]`（或更大）以及连续 `action_dim >= 2`；
额外观测维度会填 0。完整工作流和能力边界见
[空间世界模型指南](SPATIAL_WORLD_MODEL_GUIDE.md)。

## 6. `reference/` 的使用边界

参考目录仅承担“研究样本”角色。目前框架只吸收通用设计思路并重新实现：分类潜变量、
分离 KL、signed-log/two-hot 回归、分组递归，以及面向控制的潜空间 rollout。它不会：

- import 参考项目中的 Python 模块；
- 调用其训练脚本、模型权重或配置；
- 把 GPL、Apache 或受限社区许可代码复制到 `saddlellm/`；
- 要求安装 JAX、MMDetection3D、HY-WorldPlay 等参考工程依赖。

参考源码仍保留各自许可证；具体归属见 `THIRD_PARTY_NOTICES.md`。后续若研究 IRIS 的
离散视觉 tokenizer、Drive-OccWorld 的占用空间表示或视频世界模型，也应先抽象出我们
自己的接口，再在 SaddleLLM 中独立实现和测试。

## 7. 输出文件

`training.output_dir` 会包含：

- `world_model_config.json` 与 `world_model.pt`：最终可加载模型；
- `best_model/`：验证损失最低的模型（配置了验证集时）；
- `training_metrics.json` 与 `training_history.json`；
- `world_model_run_config.json`：实际模型、训练和数据摘要；
- `checkpoints/step-XXXXXXXX/`：模型、优化器、scheduler 与 AMP scaler 状态。

继续训练时，将 checkpoint 路径写到：

```yaml
training:
  resume_from_checkpoint: outputs/world_model/checkpoints/step-00000500
```

## 8. 使用边界

当前实现负责表示学习、动力学、奖励/终止预测和潜空间 rollout，不包含策略网络或
在线环境采样。实际用于机器人控制前，应单独验证多步误差、reward calibration、
分布外动作以及安全约束；长时规划建议从短 horizon 开始逐步增加。
