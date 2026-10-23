# SaddleLLM 大模型知识采用情况审计

> 历史快照：本文记录 2026-10-18 整改前源码和复现结果。2026-10-19 已修复的 P0、训练契约、选模及 QAT CUDA 执行见 [整改记录](CORRECTNESS_HARDENING_20261019.md)；下文待办状态不代表当前实现。

**结论：框架已经采用许多核心思路，但不能认定面试知识里的好方法都已成熟落地。** 原生模型、SFT 掩码、数据回放、参考 KL、压缩恢复和产物回归有实质实现；PPO 稳定流水线、完整 RAG、本地约束工具生成等仍有缺口。更需要优先处理的是旧公开入口中的错误、模拟逻辑，以及不同后端之间没有统一兑现的语义。

审计日期为 2026 年 10 月 18 日，对象是当前 `2.42` 工作区，包含既有未提交修改。知识背景见 [40 个面试问题](LLM_INTERVIEW_NOTES_ZH.md)。本次形成源码对照、只读复现和定向测试记录；下文整改项仍是待办，不能据此认定问题已经修复。

## 判断标准

| 标记 | 含义 |
|---|---|
| 原生实现 | 仓库内存在实际计算或执行逻辑；不自动代表大规模性能已验收 |
| 第三方集成 | 真实调用外部库/引擎；上游算法与本仓库适配各有责任 |
| 受限采用 | 只适用于特定模型、设备、配置，或仍定位研究实验 |
| 错误或占位 | 公开路径的语义不正确，或仅模拟计算，不能计为可用能力 |
| 缺少闭环 | 有相关组件，但没有形成相应任务的完整执行与验收链 |

对照范围是训练、压缩、推理和应用中的典型知识点，不是对全网题目逐条穷举。是否需要采用一种方法，还取决于框架定位、模型和资源预算；集成可靠上游也是合理采用方式。

## 模型与基础算子

| 知识点 | 采用情况 | 源码证据与边界 |
|---|---|---|
| Causal LM 与 shifted CE | 原生实现 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py#L2050) 的 `_causal_lm_loss`、chunked CE；处理 next-token shift、`-100` 和全掩码情况 |
| RMSNorm 与 SwiGLU | 原生实现 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py#L344) 的 `SaddleRMSNorm`、`SaddleSwiGLU`；存在实际前向和梯度路径 |
| RoPE 与位置缓存 | 原生实现 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py#L376)；[ModelBlueprint.py](../saddlellm/models/ModelBlueprint.py#L519) 的扩展配置明确支持 linear factor，不能泛称支持全部长上下文方案 |
| MHA GQA MQA | 原生加算子集成 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py#L560) 区分 query/KV 头；[OperatorBackends.py](../saddlellm/models/OperatorBackends.py#L75) 处理 compact GQA 与回退 |
| FlashAttention | 第三方集成 | [OperatorBackends.py](../saddlellm/models/OperatorBackends.py#L125) 通过 PyTorch SDPA 调度；设备、dtype、mask 等决定实际 kernel，注册名不是命中证明 |
| Transformer Engine FP8 | 原生路径未实装 | [OperatorBackends.py](../saddlellm/models/OperatorBackends.py#L194) 的相关注册 handler 仍是 SDPA；不能据此宣称原生 TE fused/FP8 已实现 |
| Tokenizer | 第三方集成加原生检查 | [TokenizerTrainer.py](../saddlellm/models/TokenizerTrainer.py#L96) 集成 BPE/Unigram/WordPiece，包含 UNK、压缩率、往返与领域分词评估 |
| MoE | 原生受限实现 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 有专家路由与前向；checkpointing/MoE 梯度测试已有覆盖，本地实现不等于多机专家并行系统 |

对应测试包括 [test_operator_backends.py](../tests/test_operator_backends.py)、[test_model_distributed_runtime.py](../tests/test_model_distributed_runtime.py)、[test_tokenizer_roundtrip.py](../tests/test_tokenizer_roundtrip.py)。

## 数据与预训练

| 知识点 | 采用情况 | 源码证据与边界 |
|---|---|---|
| 文档感知 packing | 原生实现 | [pipeline.py](../saddlellm/data/pipeline.py#L624) 保留文档 ID、重置位置并掩码跨边界目标；原生 [attention mask](../saddlellm/models/SaddleModeling.py#L938) 实施文档隔离 |
| HF packing 语义 | 存在后端差异 | [TrainingOrchestrator.py](../saddlellm/training/TrainingOrchestrator.py#L2333) 选择普通 HF Trainer；实际 tiny GPT2 签名过滤会移除 `document_ids`，不能沿用原生隔离保证 |
| 精确与近重复去重 | 有实现但需整改 | [pipeline.py](../saddlellm/data/pipeline.py#L382) 有 SimHash/MinHash；`_dedup_exact` 只哈希前 2,000 字符，已复现误删不同全文 |
| 固定数据混合 | 已采用 | [pipeline.py](../saddlellm/data/pipeline.py#L277) 调用 `interleave_datasets`；权重是采样概率，`first_exhausted` 不保证完整 token 配额 |
| loss 自适应混合训练 | 模拟占位 | [DataBalance.py](../saddlellm/data/DataBalance.py#L171) 的 `_train_step` 返回随机数，`_save_checkpoint` 也未保存模型/优化器；公开 `balance_data` 别名不能代表真实训练 |
| 训练验证独立性 | 部分采用 | 预训练在分词/packing 前切原始记录；SFT/native DPO 有来源分组；HF preference 和显式数据文件尚需统一检查 |
| benchmark 污染检查 | 有工具但未统一准入 | [DataStrategy.py](../saddlellm/data/DataStrategy.py#L198) 与 [TrainingDataInspector.py](../saddlellm/data/TrainingDataInspector.py#L614) 提供检测/审计；主训练链没有自动强制覆盖全部输入 |
| Scaling law 与预算 | 规划工具 | [ScalingLawAnalyzer.py](../saddlellm/experiments/ScalingLawAnalyzer.py#L132) 有拟合与实验网格；`predict_chinchilla_optimal` 是启发式，不能作为已拟合目标的严格最优解 |
| 混合精度 累积 检查点 | 已采用 | [TrainingOrchestrator.py](../saddlellm/training/TrainingOrchestrator.py#L2161) 校验实际 global batch 并配置 Trainer；原生模型具备 activation recompute 和 checkpoint 恢复 |
| DDP FSDP ZeRO | 第三方集成 | [DistributedConfig.py](../saddlellm/training/DistributedConfig.py#L76)、[DistributedRuntime.py](../saddlellm/training/DistributedRuntime.py#L61) 提供配置/拓扑约束；本次真实运行的是双进程 CPU DDP，非全部 GPU 分片策略 |
| TP PP CP | Megatron 集成 | [MegatronBackend.py](../saddlellm/training/MegatronBackend.py#L87)、[MegatronWorker.py](../saddlellm/training/MegatronWorker.py#L51) 真实接 Bridge worker；本次验证其模拟控制契约，未进行多 GPU TP/PP 训练 |

文档拼接允许跨文档 attention 也可以是一种明确的训练配方。这里需要修正的是后端语义声明：若目标是文档独立，就必须在 HF 路径真正实现隔离，或在不支持时明确拒绝，不能用 EOS 或 position reset 替代。

对应测试包括 [test_document_aware_packing.py](../tests/test_document_aware_packing.py)、[test_distributed_pretrain.py](../tests/test_distributed_pretrain.py)、[test_scratch_pretrain_e2e.py](../tests/test_scratch_pretrain_e2e.py)。

## 后训练与模型压缩

| 知识点 | 采用情况 | 源码证据与边界 |
|---|---|---|
| SFT completion mask | 已采用 | HF [PeftSFTTrainer.py](../saddlellm/training/PeftSFTTrainer.py#L573) 使用 TRL 及监督校验；[NativePostTraining.py](../saddlellm/training/NativePostTraining.py#L83) 原生构造 assistant labels |
| Replay 与参考 KL | 原生实现且可选 | [ReplayData.py](../saddlellm/data/ReplayData.py#L158)、[Retention.py](../saddlellm/training/Retention.py#L44)；含词表对齐、shifted mask、冻结教师和 T²，参考教师目前不适配全部分片方式 |
| LoRA QLoRA | 第三方集成 | [PeftSFTTrainer.py](../saddlellm/training/PeftSFTTrainer.py#L270) 使用 PEFT/bitsandbytes NF4、double quant；支持不等于本次已验证任意 GPU/模型配置 |
| 独立验证与选模 | 已采用但可选 | [PeftSFTTrainer.py](../saddlellm/training/PeftSFTTrainer.py#L297) 可按 eval loss 保存最佳模型；默认验证比例为 0，不自动按旧/新能力联合最优选 checkpoint |
| DPO | 原生加第三方集成 | [NativePostTraining.py](../saddlellm/training/NativePostTraining.py#L283) 实现 reference-relative loss；HF [Preference.py](../saddlellm/training/Preference.py#L302) 委托 TRL |
| ORPO KTO | 第三方集成 | [Preference.py](../saddlellm/training/Preference.py#L183) 使用相应版本 TRL experimental 类；不是本仓库自研目标函数 |
| PPO | 稳定流水线未实现 | [TrainingOrchestrator.py](../saddlellm/training/TrainingOrchestrator.py#L876) 明确拒绝；旧 [RLHFTrainer.py](../saddlellm/alignment/RLHFTrainer.py) 简化路径不能提升为稳定支持 |
| GRPO | 原生研究实现 | [GRPOTrainer.py](../saddlellm/alignment/GRPOTrainer.py#L535) 含组内 advantage、clipped ratio 和 KL；文件明确 small research pilots，需加强数值/梯度验证 |
| Logits 蒸馏 | 新旧路径不一致 | CLI [RealWorldOperators.py](../saddlellm/agents/RealWorldOperators.py#L734) 与恢复 loss 有 shift/mask/T²；旧公开 [DistillationTrainer.py](../saddlellm/distillation/DistillationTrainer.py#L79) 的目标范围和接口行为需统一 |
| AWQ GPTQ | 实际第三方校准 | [CalibratedPTQ.py](../saddlellm/compression/CalibratedPTQ.py#L352) 委托 llmcompressor，校验整数 packed 产物；[本地 loader](../saddlellm/compression/CalibratedPTQ.py#L489) 解压为浮点推理 |
| QAT | 原生参考后端 | [QuantizationAwareTraining.py](../saddlellm/compression/QuantizationAwareTraining.py#L110) 为 W8A32 逐通道 STE，真实 INT8 保存和重载，但执行浮点矩阵乘 |
| 剪枝 | mask 剪枝可用 | [pruner.py](../saddlellm/compression/pruner.py#L241) 支持置零；`ln_structured` 未删除矩阵维度，通用 head/layer 缩模路径明确拒绝 |
| 压缩后恢复 | 原生实现 | [RecoveryTrainer.py](../saddlellm/compression/RecoveryTrainer.py#L315) 有 CE/教师 KL；梯度和权重更新后重施 mask，保留剪枝状态 |
| 量化流水线与导出 | 已接入 | [QuantizationStage.py](../saddlellm/compression/QuantizationStage.py) 控制阶段开关/来源；[ModelExporter.py](../saddlellm/runtime/ModelExporter.py) 保留压缩权重和 manifest，并检查门禁身份 |
| 配对回归与发布阻断 | 原生实现 | [RegressionSuite.py](../saddlellm/evaluation/RegressionSuite.py#L645) 核验数据/配置/身份并拒绝非法指标；训练调度器可阻断未通过的导出 |

需要保留的边界是：LoRA 不自动防遗忘；QAT 数据也要覆盖旧能力；同来源/设置的回归比较不自动证明 benchmark 无训练污染。框架有 [配对统计工具](../saddlellm/evaluation/EvaluationComparison.py)，但没有覆盖全部训练选型和发布任务的统一多 seed 置信区间门槛。

## 推理与应用

| 知识点 | 采用情况 | 源码证据与边界 |
|---|---|---|
| KV 与滑窗缓存 | 原生实现 | [InferenceCache.py](../saddlellm/models/InferenceCache.py#L79) 有几何扩容和窗口 ring；每个请求独立缓存，非 paged allocator |
| Prefix cache | 原生可选实现 | [NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py#L54) 有字节预算、LRU、参数失效和私有请求 KV |
| 原生 batch decode | 受限采用 | [NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py#L91) 批量投影与独立 KV；不能与完整 paged engine 等同 |
| Chunked prefill | 受限采用 | [NativeTokenScheduler.py](../saddlellm/runtime/NativeTokenScheduler.py#L13) 有 decode 优先和 token 预算；只支持指定 dense 浮点配置，部分 cache/架构组合不支持 |
| vLLM continuous batching | 外部引擎集成 | [VLLMServing.py](../saddlellm/runtime/VLLMServing.py#L70) 创建真实 AsyncLLMEngine；Linux CUDA、相应版本及 HF 产物约束仍需满足 |
| SSE 断连与调度 | 原生实现 | [InferenceLifecycle.py](../saddlellm/runtime/InferenceLifecycle.py)、[InferenceServer.py](../saddlellm/runtime/InferenceServer.py) 协调取消、执行许可和背压 |
| 测速与回归基建 | 已采用 | [benchmark_serving.py](../tools/benchmark_serving.py#L419) 记录实际 token、可见文本 TTFT 和分位数；有基建不等于每种模型已验收 |
| 投机解码 | 旧公开分支错误 | [lightweight.py](../saddlellm/compression/lightweight.py#L332) 的拒绝/接受逻辑和预算处理有可复现问题；返回的 speedup 为估算值 |
| 低比特执行内核 | 独立受限路径 | [CUDAInt8.py](../saddlellm/models/CUDAInt8.py)、[Int8WeightKernels.py](../saddlellm/models/Int8WeightKernels.py) 有实验 W8A16 Triton 内核；它们不是新 PTQ/QAT 产物的默认执行器 |
| Embedding Reranker | 服务组件已实现 | [EmbeddingServer.py](../saddlellm/runtime/EmbeddingServer.py)、[RerankerServer.py](../saddlellm/runtime/RerankerServer.py) 有请求接口、预算和模型适配；服务单测采用替身 encoder/scorer |
| 完整 RAG | 缺少仓库内闭环 | 尚未发现完整的文档切块/版本化索引、混合检索、证据引用与端到端检索/忠实性评测链 |
| Agent 工具运行时 | 外部包集成 | [AgentOperators.py](../saddlellm/agents/AgentOperators.py#L8) 重导出 `saddle_ml.agent.operator`；外包中的工具运行时不能视为本仓库原生安全沙箱 |
| Provider 工具适配 | 已采用 | [AgentModelProvider.py](../saddlellm/agents/AgentModelProvider.py#L74) 有 tools 转发及 URL/响应等保护；不能把它等同于完整注入防御 |
| 本地 function calling 与约束 JSON | 尚未支持 | [OpenAIProtocol.py](../saddlellm/runtime/OpenAIProtocol.py#L7) 声明 tools 不支持；本地请求 schema 明确拒绝 tools/response_format |

完整 RAG、通用 Agent 沙箱和本地约束生成是否纳入主包，应按框架定位决定。它们适合独立插件或应用层，不必为了覆盖题库而强行加入训练核心。

## 已复现的问题

### 旧 benchmark 会把困惑度标成数学准确率

`BenchmarkRunner._eval_gsm8k()` 直接返回 `_eval_perplexity()`，但 `run()` 把该结果标为 accuracy。只读探针将 PPL 返回值设为 7.5，公开调度结果就是 `gsm8k: 7.5 (accuracy)`；没有加载模型，也没有运行数学题评测。

同类边界还有：该类 MMLU 默认只选 7 个科目；HumanEval 出现在任务描述中，但没有相应执行分支。这个旧入口仍被 `easy.evaluate` 和公开导出引用，不能用它声称完成标准 benchmark 验收。

证据：[源码](../saddlellm/evaluation/BenchmarkRunner.py#L203)、[只读探针](../outputs/interview-audit/probes.json)。

### 投机解码拒绝草稿后仍会输出错误 token

使用 target 几乎必然输出 token 2、draft 始终输出 token 3 的 toy 模型，设置 `max_new_tokens=1`、`num_draft_tokens=1`：

| 模式 | 应有行为 | 实际返回 |
|---|---|---|
| Greedy | 只输出 token 2 | 输出 `2 3`，生成 2 个 token，接受率 1.0 |
| Sampling | 几乎必然输出 token 2 | 输出 draft token 3，接受率 1.0 |

Greedy 分支在拒绝位置追加 target 后又计为接受；sampling 分支直接接受所有 draft。还存在后续拼接形状、固定截断 2048 和以接受率估算 speedup 的问题。

证据：[源码](../saddlellm/compression/lightweight.py#L445)、[toy 模型与实际输出](../outputs/interview-audit/speculative-probe.json)。

### 精确去重误删不同长文

构造两个前 2,000 字符相同、尾部不同的完整文档，`_dedup_exact` 只保留一个。它使用完整规范化文本的前缀作为哈希输入，因此不是完整文档精确去重。

证据：[源码](../saddlellm/data/pipeline.py#L446)、[两个输入的全文哈希与结果](../outputs/interview-audit/dedup-probe.json)。此外 SimHash 缺依赖时会跳过去重，MinHash 缺依赖时会退到当前 exact；执行报告应明确记录实际算法。

### 动态数据配比类没有执行真实训练

`BalancedPretrainer._train_step` 不调用模型、反向传播或优化器，只返回 `np.random.uniform`。在相同随机种子下，对不同输入得到同一个 loss 0.3651985024；这印证其模拟实现，不能作为真实训练反馈。它与正常数据管线的固定权重混合是两条路径。

证据：[源码](../saddlellm/data/DataBalance.py#L171)、[探针](../outputs/interview-audit/probes.json)。

### 最优规模预测没有使用拟合出的收益系数

`predict_chinchilla_optimal` 使用固定 0.1 系数并忽略拟合 A/B。探针把 A 从 1 改成 100，建议的 N/D 完全不变，均为 278124749/599251477。它可以作为粗略预算启发式，但不应称为拟合目标的最优解。

对于 `L = E + A/N^alpha + B/D^beta`、`C = 6ND` 且 A/B/alpha/beta 为正的连续近似，最优参数量应依赖 A/B：

```text
N* = [alpha A / (beta B)]^(1/(alpha+beta))
     × (C/6)^(beta/(alpha+beta))
D* = C/(6 N*)
```

实际还需处理离散结构、数据可用性和部署约束。证据：[源码](../saddlellm/experiments/ScalingLawAnalyzer.py#L197)、[探针](../outputs/interview-audit/probes.json)。

## 需要统一的训练语义

**HF preference 的自动切分仍按行。** 同一 prompt 的多个 chosen/rejected 组合可能分到两侧。若验证目标是新 prompt 泛化，应按规范化 prompt/来源/case 分组；若有意测同 prompt 新偏好，则应明确这一评测定义。SFT 自动切分和 native DPO 已有分组实现，宜复用到 [Preference.py](../saddlellm/training/Preference.py#L385)。

**显式 train/eval 文件应做内容审计。** 框架已有 `audit_training_splits`，但不同入口未统一强制串联。回归套件的数据指纹证明比较对象一致，不证明数据没有混入训练集。

**旧公开蒸馏类需要收敛到一致契约。** 顶层 `distill_model` 仍指向 [DistillationTrainer](../saddlellm/distillation/DistillationTrainer.py#L79)：KL 覆盖全 `[B,S,V]`，缺少可控 prompt/pad mask 与按有效 token 归约；直接 hidden MSE 没有处理不同维度；`fit` 对 eval_dataset 只分词，batch_size 也未形成对应 DataLoader。全 token 蒸馏可以是合理目标，但不能冒充 completion-only 蒸馏，也不能认为这些接口参数已经兑现。CLI/恢复路径已有更完整的 masked loss，可以作为统一基础。

**GRPO 需要更多直接数值测试。** 已有专项测试主要覆盖配置、累积修正、prompt 轮转和动态采样；应增加独立手算目标、rollout rescoring、KL、全相同奖励、EOS/mask 和梯度方向检查。

## 整改顺序与验收条件

P0 表示会产生误导指标、错误输出或静默数据损失，优先修复或明确拒绝；P1 表示质量契约和实际执行需要统一；P2 表示按产品需要扩展。

| 优先级 | 工作 | 可核验的完成条件 |
|---|---|---|
| P0 | 旧 benchmark 指标纠正 | GSM8K 使用生成/答案解析评分或明确拒绝；科目子集与样本数真实标注，未知任务不伪造成功 |
| P0 | 投机解码修复或关闭 | Greedy 等价、接受拒绝和采样分布测试、EOS/输出预算正确；speedup 来自计时 |
| P0 | 完整文本去重 | 两篇仅尾部不同的长文均保留；完整重复被删除；报告缺依赖后的实际行为 |
| P0 | 模拟训练入口处理 | 真实 forward/backward/optimizer/checkpoint，或明确标记示例并拒绝生产训练调用 |
| P1 | 数据切分与污染准入 | 各训练入口声明分组策略，接入 group/exact/near-duplicate 审计，保存审计配置与结果 |
| P1 | HF packing 策略 | 明确 concat/isolated 语义；隔离模式通过扰动其他文档后输出不变的测试，不支持时拒绝 |
| P1 | 公开蒸馏入口统一 | mask、shift、温度和归约有直接测试；batch/eval 参数实际执行，异构词表/hidden 明确处理 |
| P1 | RL 目标正确性 | GRPO 数值/梯度和 rollout 概率一致性测试；PPO 未完成前继续明确拒绝 |
| P1 | 真实压缩部署 | 最终量化文件直接接目标内核；重载一致性、能力回归、显存、TTFT/TPOT/吞吐同时验收 |
| P1 | Scaling 工具命名与求解 | 启发式明确标注；若称最优，结果随拟合 A/B 改变，并通过约束目标对照 |
| P2 | 多 seed 与业务验收 | 明确数据封存、置信区间、真实业务和泛化门槛，避免一次实验下结论 |
| P2 | RAG 与 Agent 插件 | 按需补完整检索/引用评测或受控工具执行，不仅扩充服务端点 |

对当前训练框架，最值得保留并继续强化的是已有的 mask/分组、能力回放、压缩恢复、产物身份和失败阻断。下一轮开发优先落实上述 P0/P1，不需要先增加更多算法名称。

## 本次核验记录

| 测试组 | 实际结果 | 覆盖与限制 |
|---|---|---|
| 架构与数据 | 71 passed | packing、SDPA/GQA、checkpointing/MoE 梯度、tokenizer 与数据管线 |
| 预训练与分布式契约 | 24 passed | 原生 scratch/保存/恢复/生成、真实双进程 CPU Gloo；Megatron 使用模拟后端 |
| 后训练 | 90 passed | 6 个文件，包含固定兼容环境下真实 SFT/DPO smoke，以及 replay/KL 与蒸馏兼容 |
| 推理与应用 | 345 passed | 14 个文件，覆盖缓存、调度、协议、断连、embedding/reranker/provider 与测速契约 |

各组可能含重叠用例，不把结果相加当全仓唯一用例总数。本次没有执行全仓测试、真实多 GPU FSDP/ZeRO/TP/PP 验收、全套 CUDA INT8 内核验收或真实 vLLM 吞吐测试；服务测试中的 fake engine/encoder 只证明对应适配契约。测试通过与上述未覆盖缺陷同时成立。

之前量化实现的实际校准/整数导出，以及 LoRA 合并→QAT→导出重载记录另见 [量化阶段验收记录](../outputs/quantization-stage-validation/validation.json)，它们不代替低比特执行器的性能验收。

本次原始证据位于 `outputs/interview-audit/`：

- [architecture.json](../outputs/interview-audit/architecture.json) 与两份 JUnit XML：源码状态、实际命令与通过记录。
- [posttraining-evidence.json](../outputs/interview-audit/posttraining-evidence.json)：后训练/压缩/门禁的执行路径和差距。
- [runtime-evidence.json](../outputs/interview-audit/runtime-evidence.json)：运行时 20 项证据、测试命令和边界。
- [probes.json](../outputs/interview-audit/probes.json)、[dedup-probe.json](../outputs/interview-audit/dedup-probe.json)、[speculative-probe.json](../outputs/interview-audit/speculative-probe.json)：只读复现结果及相关源码 SHA256。

源码/行号对应本次工作区快照。后续修复后，应更新问题状态并重新执行相应核验，再修改框架的能力声明。
