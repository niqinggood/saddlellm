# 训练与采样后端分离：借鉴 MiMo 的边界

这是异步在线训练的架构边界说明。配置可选的 Megatron Bridge 训练与 SGLang 采样适配
现已接入，使用方法见 [可选运行时指南](OPTIONAL_RUNTIME_BACKENDS.md)。大型组件不是默认依赖。
当前实现不等同于下面讨论的持续异步在线 RL，真实 GPU 后端验收仍待完成。

## 为什么值得借鉴

MiMo-V2.6 的公开 Code 复现工程以 verl 组织 RL，训练侧使用 Megatron，
rollout 侧使用 SGLang，默认使用 colocate_async、GRPO 和轨迹队列。
可以借鉴的是分工、数据契约、调度和故障隔离，而不是原样使用集群规模配置。

参考：[官方发布](https://mimo.mi.com/docs/zh-CN/news/latest/v2-6)、
[训练配置](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/config/train.yaml)、
[网关和轨迹说明](https://github.com/XiaomiMiMo/uni-agent/blob/mimo-oss/docs/source/concepts/gateway-and-trajectories.md)。

## 哪些任务需要它

| 任务 | 是否需要独立在线采样端 |
| --- | --- |
| 预训练 | 通常不需要；训练既有文本 token。 |
| 离线 SFT / DPO / KTO | 通常不需要；训练已有监督或偏好数据。 |
| GRPO / 在线 RL | 可以受益；当前策略必须生成样本，获取奖励后更新。 |
| 在线蒸馏 | 可以受益；学生采样、教师反馈与训练可以分开组织。 |

独立采样也可用于离线数据生成，但不意味着每个训练任务都应启动推理服务。

## 职责和最低数据契约

```text
任务 / 提示词 → 采样端 → 轨迹队列 → 奖励与训练端
                     ↑                    |
                     └── 带版本的权重更新 ─┘
```

- 训练端：计算损失、反向传播、优化器更新、保存权重，不管理 HTTP 会话。
- 采样端：分词、批量调度、KV、采样、取消；实现可选的 native/SGLang/vLLM 后端。
- 队列：有限容量、超时、重复提交识别、失败状态，避免无限积压和资源泄漏。
- 调度层：分配 GPU 时间/显存，协调权重版本、收集足够样本和更新节奏。

可用于 RL 的记录至少需要 request_id、初始 prompt token IDs、response token IDs、
模型输出/工具结果的 mask、奖励、完成状态和策略权重版本；算法需要时还应保存
逐 token 的生成 logprob。具体 logprob 是原始策略还是经采样变换后的分布，必须
与训练损失明确对齐，不能只保存文本或直接把 top-p 后的概率当作原始策略概率。

模型、分词器/chat template、特殊 token 和停止规则也需要一致或显式转换。
单有 OpenAI-compatible 文本接口，不代表已经满足 RL token 级训练契约。

## 异步不是给 generate 套 async

训练与推理共享 GPU 时，仍需要显存预算和计算时序，不能让两个大模型同时
无约束占满设备。可以从共享卡的分阶段执行开始；有足够资源时再分卡分进程。
Agent 等待工具时，GPU 调度可继续服务其他任务，但不等同于 GPU 无限并行。

权重更新需要可验证的完成信号，不能让一次生成混用半更新的参数。
发布新版本后必须失效或区分旧 KV/前缀缓存；旧权重生成的轨迹应按算法和配置
限制滞后版本数、丢弃或执行适当的 off-policy 校正。不能用错误版本的 logprob
计算重要性比率。异步流程需要定义采样参数、随机数、恢复点和可复现边界。

## SaddleLLM 当前实际边界

- 已有原生 generate、逐 token 会话、流式服务与有限的固定微批处理。
- backends.training=megatron 通过受控 torchrun worker 调用 Bridge pretrain/finetune。
  recipe 与 overrides 管理实际训练；支持显式 HF 导出及完成产物检查。
- backends.rollout=sglang 支持 native HTTP 异步并发请求或延迟创建本地 Engine。
  MOPD 不再重复加载本地学生；保存采样来源、生成 token 和服务确认的权重版本。
- HTTP 支持阶段开始前的磁盘权重发布与版本核对；本地 Engine 更新时关闭并重建。
  这是批次/阶段屏障，不是训练迭代内的分布式 tensor 权重传输。
- 原有 FactoryBackendPlanner 的 hybrid-style 规划仍不自动等于当前 recipe 适配器，
  也不意味着 Colossal、任意 TP/PP/CP/EP 训练模式已被验证。
- 尚未实现持续的训练/采样重叠、RL 轨迹队列、在线 GRPO 损失和滞后版本校正。
  当前测试使用协议模拟和 Engine/Bridge 替身，未进行真实 GPU 联调。

建议后续先在目标 GPU 环境验证这两个适配器的同步闭环和模型格式转换，再补完整
token 级轨迹契约、在线训练损失、有限异步队列及分布式权重传输。
每一阶段都应验证输出、logprob、mask、奖励和权重版本一致性，以及取消和重启。
用户入口继续由已有配置、命令和 import API 组织，不要求普通用户手工管理进程。
