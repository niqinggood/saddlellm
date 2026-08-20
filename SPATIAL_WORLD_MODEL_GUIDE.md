# 空间世界模型：识图、建图与路线规划

这套实现把“视觉识别”“可通行几何”“路线搜索”和“学习到的动态预测”分开，避免让视觉大模型直接猜路线：

```text
俯视图 / 户型图
       │
       ├─ Qwen-VL（可选）── 房间、门、障碍、风险、语义名称
       │
       └─ 原生建图器 ────── FREE / BLOCKED / UNKNOWN 占用栅格
                              │
起点 + 终点 + 约束 ──────────┤
                              ▼
                  原生 A* + 多样化 Top-K 规划器
                              │
                   RSSM 世界模型（可选重评分）
                              │
                              ▼
                   JSON + 交互 HTML + PNG
```

Qwen-VL 只负责语义理解和坐标指代；是否能走、路径是否穿墙、路线搜索均由 SaddleLLM 自己的建图器和规划器负责。训练后的 RSSM 可以进一步根据历史轨迹预测奖励、终止概率或风险，为几何候选路线重新排序。

## 启动完整可视化工作台

安装项目后直接启动同一个 FastAPI 服务；React 生产包已随 `saddlellm` 一起提供：

```powershell
python -m pip install -e .
saddle-llm spatial-studio --host 127.0.0.1 --port 7865
```

浏览器打开 `http://127.0.0.1:7865`。工作台支持：

- 上传 PNG/JPEG/WebP/BMP/GIF/TIFF 俯视图；
- 输入或在图上点选起点、终点；
- 查看占用层、语义层、多条路线、长度、转弯、净空和风险；
- 查看空间结构、未知区域及实际启用的模型，避免把几何结果误称为学习结果；
- 导出 JSON、PNG 和可脱离服务器查看的交互 HTML；
- 桌面、手机竖屏与手机横屏布局。

开发前端时运行：

```powershell
cd spatial-studio
npm install
npm run dev       # Vite 代理 /api 到 127.0.0.1:7865
npm test
npm run build     # 写入 saddlellm/spatial_studio_web
```

## 直接运行内置示例

```powershell
python -m saddlellm.cli plan-spatial-route `
  data\spatial_floorplan_example.pbm `
  --start 2,2 `
  --goal 21,13 `
  --routes 3 `
  --obstacle-dilation 0 `
  --resolution 0.5 `
  --instruction "避开墙体并给出三条明显不同的路线" `
  --output-json outputs\spatial\demo_routes.json `
  --output-html outputs\spatial\demo_routes.html `
  --output-png outputs\spatial\demo_routes.png
```

输出包括：

- `demo_routes.json`：占用网格摘要、路线点、长度、转弯、净空和风险；
- `demo_routes.html`：可开关占用层、语义层和每条路线的离线交互页面；
- `demo_routes.png`：适合预览和分享的静态结果图。

## 接入 Qwen-VL

安装兼容模型所需的最新版 `transformers` 后，通过 `--qwen-model` 指定本地路径或模型标识：

```powershell
python -m saddlellm.cli plan-spatial-route floorplan.png `
  --start entrance `
  --goal office `
  --qwen-model Qwen/Qwen3-VL-4B-Instruct `
  --routes 3
```

视觉模型需要返回统一的 `SpatialAnalysis` JSON，其中实体坐标归一化为 `0..1000`。也可以通过 `CallableSpatialAnalyzer` 接入任何私有视觉服务，而不绑定某个模型厂商。

工作台按需加载模型，未选择 Qwen-VL 时不会占用显存：

```powershell
saddle-llm spatial-studio `
  --qwen-model Qwen/Qwen3-VL-4B-Instruct `
  --device auto
```

## 训练并接入空间 RSSM

空间 checkpoint 使用稳定的 `spatial_features_v1` 契约：16 维观测描述当前位置、目标、方向、地图占用比例、起终点净空、地图比例和分辨率；动作前两维是归一化的 `dx, dy`。先从俯视图自动生成离线轨迹：

```powershell
saddle-llm build-spatial-world-data `
  data\spatial_floorplan_example.pbm `
  --output data\spatial_world_model_example.jsonl `
  --episodes 64 `
  --routes-per-pair 2 `
  --obstacle-dilation 0 `
  --resolution 0.5
```

生成器会确定性地采样可通行起终点、调用原生规划器、逐步编码观测和动作，并以进度、净空、步长代价和到达奖励构建 episode。随后用通用原生训练器训练：

```powershell
saddle-llm train-world-model configs\spatial_world_model.yaml --dry-run
saddle-llm train-world-model configs\spatial_world_model.yaml
```

把 checkpoint 交给工作台后，在“世界模型”中启用 RSSM 路线重评分：

```powershell
saddle-llm spatial-studio `
  --world-model outputs\spatial_world_model_example `
  --device auto
```

服务只接受扁平观测不少于 16 维、连续动作不少于 2 维的自动路线评分 checkpoint；不兼容模型会明确报错，不会静默伪装为世界模型结果。建议使用多张地图和真实执行日志继续扩充训练集；单图合成数据主要用于跑通链路。

## v2：动作条件占用预测与闭环规划

`spatial_sequence_v2` 是当前推荐的空间世界模型契约。它不再只给整条路线打分，而是在每个动作之后预测局部世界如何变化，并把预测结果交给闭环规划器：

```text
局部部分观测（11 × 16 × 16）+ 历史隐状态 + 候选动作
                           │
                           ▼
              RSSM 动作条件隐空间动力学
                 ├─ 下一帧占用：空闲 / 障碍 / 未知
                 ├─ 自身位移：dx / dy / heading
                 ├─ 碰撞概率
                 ├─ 奖励与继续概率
                 └─ 多步 imagined rollout
                           │
                           ▼
        CEM 候选动作搜索 + 硬几何安全盾 + A* 专家兜底
                           │
                           ▼
                   执行一步、更新观测、重新规划
```

11 个输入通道包含 free、blocked、unknown、clearance、agent、goal、dynamic，以及目标方向和朝向编码。占用头采用动作条件运动学先验加可学习残差；几何盾负责硬安全边界，学习头负责未知区和动态风险。这样既能利用学习到的预测，也不会让神经网络单独决定是否穿墙。

复现当前多地图训练数据：

```powershell
saddle-llm build-spatial-world-data `
  data\spatial_floorplan_example.pbm `
  --additional-image data\spatial_floorplan_variant.pbm `
  --schema v2 `
  --episodes 64 `
  --observation-size 16 `
  --sensor-radius 5 `
  --collision-probability 0.2 `
  --output data\spatial_world_model_v2.jsonl

saddle-llm train-world-model configs\spatial_world_model_v2.yaml
```

配置通过 `map_id` 做整图留出，避免同一张地图的相邻轨迹同时落入训练集和验证集。当前选择的 checkpoint 是 `outputs/spatial_world_model_v2_hybrid`，共 571,172 个参数；它保留碰撞辅助监督，因为消融实验显示该信号也能改善长时隐状态。

在完全留出的 `spatial_floorplan_variant` 上复现评测：

```powershell
saddle-llm evaluate-spatial-world-model `
  outputs\spatial_world_model_v2_hybrid `
  data\spatial_world_model_v2.jsonl `
  --sequence-length 5 `
  --group-key map_id `
  --group-value "1:spatial_floorplan_variant" `
  --output outputs\spatial_world_model_v2_hybrid_eval_h5.json

saddle-llm evaluate-spatial-world-model `
  outputs\spatial_world_model_v2_hybrid `
  data\spatial_world_model_v2.jsonl `
  --sequence-length 20 `
  --group-key map_id `
  --group-value "1:spatial_floorplan_variant" `
  --output outputs\spatial_world_model_v2_hybrid_eval_h20.json
```

| 留出集指标 | 5 步 | 20 步 |
| --- | ---: | ---: |
| 占用准确率 | 0.9300 | 0.8334 |
| 占用 mean IoU | 0.8474 | 0.6280 |
| copy-last mean IoU | 0.5701 | 0.3900 |
| 相对基线增益 | +0.2773 | +0.2380 |
| 自身运动 MAE | 0.0912 | 0.0926 |
| 碰撞 Brier | 0.1099 | 0.1228 |

闭环报告位于 `outputs/spatial_world_model_v2_closed_loop.json`：示例地图与留出变体均到达目标，分别重规划 20 和 21 次，记录到的几何碰撞均为 0。该结果验证的是当前合成俯视图分布，不等价于真实机器人安全认证。

实现选择参考了 [DreamerV3](https://arxiv.org/abs/2301.04104) 的潜变量想象、[TD-MPC2](https://arxiv.org/abs/2310.16828) 的动作条件规划、[OccWorld](https://arxiv.org/abs/2311.16038) 的占用空间预测，以及 [Drive-OccWorld](https://arxiv.org/abs/2408.14197) 的动作可控占用预测与规划结合。项目只借鉴这些公开思想，模型网络、数据契约、训练器、安全盾和工作台代码均在本仓库内独立实现。

## API

- `GET /api/health`：健康检查；
- `GET /api/capabilities`：Qwen-VL、RSSM、限制与观测契约；
- `GET /api/demo`：生成或返回内置示例；
- `POST /api/plan`：`multipart/form-data`，包含 `image` 和 JSON 字符串 `request`；
- `GET /api/jobs/{id}`：读取任务；
- `GET /api/jobs/{id}/{image|mask|json|html|png}`：读取产物；
- `DELETE /api/jobs/{id}`：删除任务目录。

## 能力边界

- 当前阈值建图器适合干净的俯视图、楼层图和简化地图。
- 单张透视照片无法恢复所有被遮挡空间，默认拒绝把它当作完整导航地图。
- 实景导航需要深度估计、多视角 SLAM/VIO、相机标定或已有地图；Qwen-VL 不能替代这些几何模块。
- PNG/HTML 中的“世界模型回报”只有挂接并训练 RSSM 后才有意义；未挂接时展示几何路线指标。
- 示例合成轨迹能验证训练与推理闭环，但不能代表真实机器人分布；生产使用应加入实测轨迹、动态障碍和失败样本。
- 真正控制机器人前必须增加安全距离、动态障碍跟踪、定位误差和紧急停止策略。
