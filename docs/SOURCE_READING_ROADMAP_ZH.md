# SaddleLLM 源码阅读路线

这份路线用于逐步掌握当前系统的主要功能：能从一个命令找到执行函数，解释数据如何变化，定位错误，并判断一次运行是否真的完成了训练或发布。

核对基线为 2026 年 10 月 1 日工作区中的 2.40 源码，包含本地尚未提交的实现。完整讲解见 [主要功能代码实现流程与源码解读](CORE_FEATURE_CODE_WALKTHROUGH_ZH.md)。源码仍在迭代，后续优先按函数名定位；本文行号是本次核对时的位置。

## 如何开始

建议第一次先用约 60 至 90 分钟完成第 1 至 3 讲，可按自己的基础调整节奏。每次阅读都围绕一个具体输入，记录它经过的函数、关键变量、输出和失败条件。

先读导读的第 1 至 3 节，然后打开 TrainingOrchestrator.run。第一次看到模型或训练器调用时，先记下输入输出，下一讲再进入算法内部。

| 阅读顺序 | 要回答的问题 | 主要源码 | 掌握标准 |
|---|---|---|---|
| 第 1 讲 入口与配置 | 一个 train 命令如何变成 TrainingConfig | cli.py、TrainingOrchestrator._parse_config | 能追踪 model.backend、stages、sft.data_path 的去向 |
| 第 2 讲 阶段编排 | 阶段为何按这个顺序执行 | framework/pipeline.py、stages.py、TrainingOrchestrator.run | 能解释默认顺序、显式依赖图、插件分派 |
| 第 3 讲 状态与产物 | 失败后从哪里恢复 | PipelineStateStore、ArtifactRegistry | 分清执行状态、业务结果、模型 checkpoint |
| 第 4 讲 数据与分词 | 一行文本如何变成训练 batch | data/pipeline.py、TokenizerTrainer.py | 能解释 input_ids、labels、mask、document_ids |
| 第 5 讲 原生模型 | 模型如何计算预测分布与 loss | ModelBlueprint.py、SaddleModeling.py | 画出张量形状变化，解释标签移位 |
| 第 6 讲 预训练 | 谁更新权重，谁保存 checkpoint | _run_pretrain_stage、NativeTrainer.py | 分清从零训练、继续预训练、精确恢复 |
| 第 7 讲 SFT 与 DPO | 指令和偏好如何影响梯度 | NativePostTraining.py、PeftSFTTrainer.py、Preference.py | 找到监督区域、冻结参数和 DPO 参考模型 |
| 第 8 讲 蒸馏与压缩 | 教师信号和轻量化产物如何产生 | OnPolicyDistillation.py、compression/ | 分清采集数据、更新学生、量化、剪枝 |
| 第 9 讲 评测与导出 | 何时允许发布 | LLModelEvalute.py、ReleaseGate.py、ModelExporter.py | 找到 gate 拒绝点并解释 release_manifest |
| 第 10 讲 推理请求 | 一条聊天请求如何返回 token | ServingPlan.py、InferenceServer.py、NativeGeneration.py | 分清聊天模板、prefill、decode、SSE |
| 第 11 讲 并发与性能 | 合批、缓存、取消分别解决什么 | InferenceBatcher.py、NativeStreamBatcher.py、InferenceLifecycle.py | 能追踪单请求资源的获取与释放 |
| 第 12 讲 扩展系统 | 世界模型和空间工作台如何协作 | multimodal/、world_models/、spatial/、spatial-studio/src/ | 分清语言生成、环境预测、路径规划与反馈 |

## 每一讲的阅读方式

1. 在导读中找本讲的功能流程，先确定输入和预期输出。
2. 打开入口函数，沿本次分支向下读；不同后端分开追踪。
3. 在纸上或笔记中记下 3 至 5 个关键变量及其含义。
4. 读关联测试的输入和断言，检查自己预测的行为是否一致。
5. 回答章节自查题，合上文档后尝试复述调用链。

可以使用下面的笔记模板。填不出的项目就是下一步应核对的源码位置。

~~~text
功能：
实际入口：
本次选择的配置和后端：
调用顺序：
输入对象及字段：
内部状态在哪里改变：
输出对象及文件：
失败条件和错误传播：
对应测试与关键断言：
我仍不确定的地方：
~~~

## 第一轮可操作的练习

以下命令在仓库根目录运行。第一条只生成计划；它不会证明语料、权重和 GPU 能完成训练。

~~~powershell
python -m saddlellm.cli train configs/tinystories_tiny_scratch.yaml --dry-run
~~~

观察输出中的 stages、pipeline_plan.execution_order、nodes[].needs 和 consumes/produces。然后在源码中找出这些字段分别由哪个对象生成。

若已有开发测试环境，可按需运行现有的编排测试。测试使用临时目录和轻量阶段帮助理解契约，不等于完整模型训练验收。

~~~powershell
python -m pytest tests/test_unified_pipeline.py tests/test_framework_plugins.py -q
~~~

接着读 [原生后训练示例](../configs/native_posttrain.yaml)，只做纸面追踪：假设 pretrain 产物不存在，SFT 应从哪里找模型；SFT 完成后 preference 从哪里读取新权重；如果 eval 的门禁拒绝，export 会在哪里停止。该示例引用的真实模型和数据需要另行准备，阅读时无需直接启动训练。

## 后续逐段精读的固定结构

每次可以沿下面的顺序讲解一个函数或一条功能链：

| 内容 | 要讲清的事情 |
|---|---|
| 功能场景 | 用户执行什么命令或调用什么 API |
| 源码定位 | 文件、类、函数及其上游调用者 |
| 代码片段 | 选取会改变状态、选择分支或计算结果的代码 |
| 逐段解释 | 每段为何存在，删去后会破坏什么契约 |
| 数据示例 | 真实字段、示意张量或小型输入的变化 |
| 边界情况 | 缺数据、后端不支持、超长输入、取消、恢复 |
| 验证 | 对应现有测试或一次明确限定范围的实验 |
| 自查 | 能否独立解释或预测该分支的结果 |

建议下一次从“train 命令到第一个 Stage”开始，覆盖 cli._cmd_train、TrainingOrchestrator.__init__、PipelinePlan.compile、run 和 run_stage。这条链掌握后，其余功能都能挂到同一张调用图上。

## 判断是否掌握系统

达到下面几项时，就具备独立维护主干的基础。性能调优和复杂分布式实验仍需要各自的实测经验。

- 能画出训练、发布、推理三条调用链，并指出交接的文件或对象。
- 能区分模型后端、训练后端、rollout 后端和服务后端。
- 能说明某个输入 token 是否参与 loss，以及为什么。
- 能解释一次生成中的提示词处理、缓存更新、采样和停止。
- 能从 traceback 回到配置字段或数据契约，而不是只改报错位置。
- 能区分配置可解析、阶段已返回、参数已更新、评测通过和性能提升。
- 能为一个新 Stage 指定输入输出、能力声明、恢复语义和验证方法。

## 本次核对范围

本次通过源码、配置及现有测试用例核对流程，执行记录如下。

| 检查 | 结果 | 证明范围 |
|---|---|---|
| TinyStories CLI dry-run | 正常输出 tokenizer → pretrain 计划 | 命令到计划的当前配置路径 |
| 导读中的纯 Python 计划示例 | 全部断言通过 | 线性顺序、DAG 层级、下游失效集合 |
| test_unified_pipeline.py 与 test_framework_plugins.py | 18 passed | 该测试集覆盖的阶段、插件、产物和恢复契约 |
| 文档源码导航 | 73 个函数/类定位，139 个本地链接检查通过 | 文件存在、源码定义与引用行号有效 |

测试运行另有一条环境中的 pynvml 弃用提示，未造成用例失败。其余测试命令是后续学习入口，不表示本次全部执行。这些检查不验证大模型训练质量、GPU 吞吐或外部服务连通性。

详细函数表可辅助查阅 [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md)。旧架构文档中的版本与能力描述需要结合当前源码判断，本路线以本次源码核对为准。
