# SaddleLLM 主要功能代码实现流程与源码解读

本文帮助项目维护者从主要功能走到实际源码，理解入口、调用顺序、数据变化、状态保存和失败边界。阅读目标是能解释一次任务如何运行，并能独立定位需要修改的位置。

基线为 2026 年 10 月 1 日当前工作区的 SaddleLLM 2.40，包含本地未提交实现。下面的调用链按源码中的分支整理；外部库内部循环以其调用边界为止。示意代码会明确标注，源码链接定位到本次核对时的函数。学习顺序见 [源码阅读路线](SOURCE_READING_ROADMAP_ZH.md)。

## 1 系统对象与主流程

系统围绕配置、阶段、模型和产物协作。配置描述意图；编排器选择执行顺序并调用阶段；执行器处理数据和参数；产物用于后续评测、恢复和服务。

~~~mermaid
flowchart TD
    A[CLI 或 Python API] --> B[配置解析与校验]
    B --> C[PipelinePlan 和 StageRegistry]
    C --> D[TrainingOrchestrator]
    D --> E[数据和 Tokenizer]
    D --> F[模型构建与训练]
    E --> F
    F --> G[评测与发布门禁]
    G --> H[模型导出]
    H --> I[推理服务]
    J[已有模型] --> I
    D --> K[多模态与世界模型训练]
    K --> L[空间规划与反馈]
    D --> M[阶段状态与产物登记]
~~~

| 对象 | 保存什么 | 先看哪里 |
|---|---|---|
| TrainingConfig | 解析后的模型、数据、阶段、训练、分布式配置 | [TrainingConfig](../saddlellm/training/TrainingOrchestrator.py#L501) |
| PipelinePlan | 阶段依赖、执行顺序、能力元数据 | [PipelinePlan](../saddlellm/framework/pipeline.py#L59) |
| StageContext | 单个阶段看到的配置、已有结果和共享服务 | [StageContext](../saddlellm/framework/stages.py#L59) |
| _stage_results | 各阶段返回或写入的结果字典 | [TrainingOrchestrator.run_stage](../saddlellm/training/TrainingOrchestrator.py#L1540) |
| ArtifactRegistry | 产物名称、类型、路径和生产阶段 | [ArtifactRegistry](../saddlellm/framework/artifacts.py#L81) |
| PipelineStateStore | 执行尝试、状态、结果和恢复信息 | [PipelineStateStore](../saddlellm/framework/pipeline.py#L255) |
| SaddleForCausalLM | 网络参数、前向、损失和原生生成入口 | [SaddleForCausalLM](../saddlellm/models/SaddleModeling.py#L1407) |

四类后端配置需要分别理解，名称相似不表示它们可以互换。

| 位置 | 可选值 | 实际问题 |
|---|---|---|
| model.backend | hf、saddle | 训练阶段使用什么模型实现和能力契约 |
| backends.training | default、megatron | 哪套训练运行时执行任务 |
| backends.rollout | native、sglang | 如何产生当前学生策略的样本 |
| serve --backend | auto、local、native、vllm | 在线服务由哪个引擎生成 |

包入口在 [__init__.py](../saddlellm/__init__.py)，通过 _exports.py 延迟加载公共对象。saddle_llm 是兼容命名空间，真正实现集中在 saddlellm。factory 生成配置或计划，通常仍把执行交给编排器；spatial-studio 是独立空间工作台前端。

## 2 从 train 命令到执行计划

用 [TinyStories 示例](../configs/tinystories_tiny_scratch.yaml) 追踪入口最直观：它声明 tokenizer 和 pretrain 两个阶段，并给出原生小模型蓝图。

~~~text
python -m saddlellm.cli train CONFIG --dry-run
  → main 解析 CLI 参数
  → _cmd_train
  → _resolve_train_request 归一化请求
  → 检查蓝图、可选运行时与并行选项
  → PipelinePlan.compile
  → 输出计划并返回

实际执行分支
  → TrainingOrchestrator.from_dict
  → _parse_config
  → TrainingOrchestrator.__init__
  → run
~~~

入口链接：[_cmd_train](../saddlellm/cli.py#L1775)、[TrainingOrchestrator._parse_config](../saddlellm/training/TrainingOrchestrator.py#L3902)。

_parse_config 把字典转换为 dataclass。例如 YAML 中的 model.backend 进入 config.model_backend，model.name_or_path 进入 config.model_name_or_path，sft.data_path 进入 config.sft.data_path。后续阶段读取的是这些解析后的对象。

解析器要求顶层 stages 是非空、无重复的字符串列表，并拒绝旧的 flow/spec 等配置包装。它还处理布尔值、分布式策略、蓝图来源和 pipeline 配置。独立的 [TrainingConfigValidator.py](../saddlellm/training/TrainingConfigValidator.py) 提供配置检查能力；运行时编排器也有自己的 _validate_config。理解校验时要追踪实际入口，不能假设每条入口经过完全相同的检查函数。

__init__ 会发现 Stage、读取能力声明、编译计划，再把 config.stages 替换为计划顺序，随后进行运行配置校验。创建编排器会准备输出目录和日志；CLI 的 dry-run 在实际创建编排器之前有独立返回分支。

断点建议：_cmd_train 解析请求后、_parse_config 返回前、TrainingOrchestrator.__init__ 编译计划后。观察 raw、config.model_backend、config.stages 和 _pipeline_plan.stages。

自查：配置能够输出 dry-run 计划，能否证明数据文件都能读取、模型能训练、GPU 内存足够？不能；这些属于后续执行路径。

## 3 阶段编排和恢复的源码细节

核心函数：[PipelinePlan.compile](../saddlellm/framework/pipeline.py#L74)、[TrainingOrchestrator.run](../saddlellm/training/TrainingOrchestrator.py#L1392)、[OrchestratorMethodStage.run](../saddlellm/framework/stages.py#L153)。

没有 dependencies 时，列表中的每个阶段依赖前一个阶段。指定 pipeline.dependencies 后使用显式依赖图，映射中省略的阶段成为根节点；不存在的依赖、自依赖和环都会报错。同等可执行节点按原声明顺序决定先后，便于重复实验。

下面是依赖配置的结构示意，a、b、c 需要在使用前注册为实际 Stage。

~~~yaml
stages: [c, a, b]
pipeline:
  dependencies:
    a: []
    b: []
    c: [a, b]
~~~

得到的顺序是 a、b、c。计划含层级信息，但 TrainingOrchestrator.run 使用普通 for 循环逐阶段执行，当前没有据此并行启动同层 Stage。

run 中的关键语句如下，省略日志与异常处理，保留实际控制语句：

~~~python
for active_stage in stages:
    if self._pipeline_state_store.should_skip(active_stage):
        continue
    self._pipeline_state_store.begin(
        active_stage,
        write=self._distributed_runtime.is_main_process,
    )
    self.run_stage(active_stage)
    stage_result = self._stage_results.get(active_stage, {})
    self._artifact_registry.capture(active_stage, stage_result)
    self._pipeline_state_store.finish(
        active_stage,
        stage_result,
        artifacts=self._artifact_registry.to_dict(),
        write=self._distributed_runtime.is_main_process,
    )
~~~

这里先检查是否可跳过，再记录开始，然后执行、登记产物并落盘结果。异常路径调用 fail，输出摘要后继续向上抛出异常，CLI 再产生失败响应。

run_stage 从 StageRegistry 创建或复用插件，检查分布式能力，构造 StageContext，再调用 validate 和 run。内置 Stage 使用 OrchestratorMethodStage，通过 method_name 调回 _run_sft_stage 等方法；自定义插件可以直接返回 Mapping。内置方法通常自己写 _stage_results，因此迁移适配器返回 None。

consumes/produces 当前主要是计划和能力描述。它们没有把任意生产者自动连接到任意消费者：例如 SFT 仍显式读取 pretrain.model_path 或配置路径。ArtifactRegistry.capture 会识别 model_path、data_path、plan_path 等惯用字段并登记；登记本身不等于验证每个文件已经存在且完整。

恢复与继续训练的区别：

| 机制 | 恢复或复用的内容 | 源码要点 |
|---|---|---|
| pipeline.resume | 已完成阶段的结果和产物记录 | 校验运行指纹与计划签名，再决定跳过哪些 Stage |
| pipeline.rerun | 指定 Stage 及其依赖下游重新执行 | 通过 descendants 失效阶段与相应产物记录 |
| training.resume_from_checkpoint | 训练器的模型与优化器等训练状态 | 读取 Trainer checkpoint，入口和后端有各自限制 |
| pretrain_mode: continue | 已有模型权重作为新一轮预训练起点 | 不等价于恢复原优化器、步数和调度器 |

还有一个必须读懂的细节：PipelineStateStore.finish 在函数正常返回时，通常把执行状态记成 completed，同时保留业务 result_status。业务结果可能是 planned、rejected，甚至是某条允许不抛错的 failed 分支。因此判断成功要同时看 results、result_status、指标与产物；summary 中 stages_completed 也主要来自结果字典的键。

主要落盘文件是 pipeline_state.json、summary.json、training_summary.json。精确格式看 [PipelineStateStore.finish](../saddlellm/framework/pipeline.py#L350) 和 [TrainingOrchestrator._print_summary](../saddlellm/training/TrainingOrchestrator.py#L3864)。

关联测试：[test_unified_pipeline.py](../tests/test_unified_pipeline.py)、[test_framework_plugins.py](../tests/test_framework_plugins.py)。先读 DAG 排序、恢复和插件调用的断言。

自查：Stage 返回 planned 后，为什么流水线仍可能显示执行完成？为什么重新运行一个上游阶段必须考虑下游结果失效？

## 4 数据和 Tokenizer 如何进入模型

主要入口：[TrainingOrchestrator._run_tokenizer_stage](../saddlellm/training/TrainingOrchestrator.py#L1699)、[TokenizerTrainer.fit](../saddlellm/models/TokenizerTrainer.py#L96)、[DataPipeline.tokenize_and_pack](../saddlellm/data/pipeline.py#L550)。

~~~text
语料配置
  → collect 读取数据
  → clean 清洗
  → deduplicate 去重
  → filter_quality 筛选
  → 按配置划分训练和验证数据
  → tokenize_and_pack
  → CausalLMDataCollator
  → 模型 forward
~~~

Tokenizer 阶段可以训练新词表，也可以加载已有 tokenizer。预训练开始前还会核对词表和特殊 token 与模型的契约，避免 token ID 超出嵌入表或 EOS 配置不一致。

设 batch 大小为 B、序列长度为 T。经过 collator 后，核心张量通常为：

| 字段 | 常见形状 | 含义 |
|---|---|---|
| input_ids | B × T | 输入 token 的词表编号 |
| labels | B × T | 训练目标；-100 表示忽略该位置的 loss |
| attention_mask | B × T | 标识有效输入和 padding，某些无 padding 快路径会省略 |
| document_ids | B × T | 打包后 token 属于哪篇文档；需要时用于边界隔离 |
| position_ids | B × T | 位置编码索引；文档或片段边界可以重新起算 |

打包路径有两个关键点。第一，开启 pack_sequences 后，tokenize 不先按 max_len 截断每篇文档，而是让 packer 分块，避免长文尾部提前丢失。第二，多篇文档拼在一块时，后续文档的首 token 标签会被设为 -100，避免上一文档的末 token 被要求预测下一文档的开头。原生注意力还可使用 document_ids 隔离跨文档注意力；这不能直接推广为所有外部模型均支持同一附加字段。

_pack_sequences 当前只在尾部剩余 token 数至少为 10 时补齐并产出尾块，小于该阈值的尾部会丢弃。理解样本数时要把清洗、去重、分块和这个尾块策略都算进去。

符号示例：input_ids 为 [A, B, C, D]，普通 labels 也为 [A, B, C, D]。模型内部移位后，用 A 位置的预测对齐 B，用 B 对齐 C。若 B 是新文档首 token，把该标签置为 -100，就会忽略 A→B 这项监督。labels 的屏蔽和 attention 的屏蔽解决不同问题。

断点建议：tokenize_and_pack 后看单条样本；CausalLMDataCollator.__call__ 返回前看形状、-100 数量和文档边界。实现见 [CausalLMDataCollator.__call__](../saddlellm/data/pipeline.py#L806)。

关联测试：[test_document_aware_packing.py](../tests/test_document_aware_packing.py)、[test_pretrain_quality_controls.py](../tests/test_pretrain_quality_controls.py)、[test_tokenizer_roundtrip.py](../tests/test_tokenizer_roundtrip.py)。

自查：为什么不能仅用 token ID 是否等于 pad_token_id 判断该位置要不要训练？提示：PAD 和 EOS 可能共用 ID，真实 EOS 仍可能是有效目标。

## 5 模型蓝图和一次前向计算

入口：[ModelBlueprint.from_dict](../saddlellm/models/ModelBlueprint.py#L228)、[ModelBlueprint.build_model](../saddlellm/models/ModelBlueprint.py#L735)、[SaddleForCausalLM.forward](../saddlellm/models/SaddleModeling.py#L1569)。

ModelBlueprint 是可序列化的结构描述，负责归一化、合法性校验、展开层配置及估算。build_model 转入 SaddleModelConfig 和 SaddleForCausalLM，才真正分配网络参数。参数量估算不是训练质量或性能证明。

以普通稠密解码器为例，H 为隐藏维度、V 为词表大小：

~~~text
input_ids          B × T
  → Embedding      B × T × H
  → Decoder × N    B × T × H
  → RMSNorm        B × T × H
  → lm_head        B × T × V
  → shift + cross entropy
  → 标量 loss
~~~

[SaddleDecoderLayer.forward](../saddlellm/models/SaddleModeling.py#L1334) 处理注意力、FFN 和残差连接。普通串行残差可概念化为“先归一化并计算 attention，再加回输入；随后归一化并计算 FFN，再加回状态”。parallel residual 和 Attention Residuals 有不同分支，不能用这个简化式解释全部配置。

[SaddleAttention.forward](../saddlellm/models/SaddleModeling.py#L741) 负责 Q/K/V、位置编码、注意力掩码和缓存。GQA 允许多个 query 头共享较少的 KV 头；MLA 使用不同的表示路径；KDA 切换到独立的递归注意力实现。普通 KV 张量和 KDA 状态不是同一种缓存契约。

FFN 可以使用 SwiGLU 或 MoE。MoE 除了专家计算，还涉及路由选择与辅助统计。模型汇总 loss 的关键语句为：

~~~python
loss = lm_loss + aux_loss
# 配置启用 multi_token_prediction 且存在额外头时：
loss = loss + mtp_loss
~~~

普通 causal LM loss 把 logits 的前 T-1 个位置与 labels 的后 T-1 个位置对齐。训练时若启用分块线性层交叉熵，可以避免同时保留完整 B×T×V logits；这改变中间内存使用方式。返回对象中的 logits 在这条路径可能为 None，调用者必须遵守约定。

模型同时提供 save_pretrained 和 from_pretrained_saddle。原生 checkpoint 使用 saddle_config.json 与 pytorch_model.bin，完整训练/部署目录通常还需要匹配的 tokenizer 文件。HF 与原生加载分流在 [load_causal_lm](../saddlellm/models/ModelLoader.py#L159)。

断点建议：forward 的 embedding 后、第一层 Decoder 后、loss 计算后。记录 B、T、H、V，是否返回 logits，是否启用了 use_cache、checkpointing、MoE 和 MTP。

关联测试：[test_native_model_lifecycle.py](../tests/test_native_model_lifecycle.py)、[test_model_composition.py](../tests/test_model_composition.py)、[test_chunked_lm_loss.py](../tests/test_chunked_lm_loss.py)、[test_hybrid_attention_attnres.py](../tests/test_hybrid_attention_attnres.py)。

自查：一次 forward 和一次 generate 有什么关系？forward 计算指定输入的输出；generate 在生成会话中反复调用前向并采样，逐步扩展序列。

## 6 预训练和分布式训练

主入口是 [TrainingOrchestrator._run_pretrain_stage](../saddlellm/training/TrainingOrchestrator.py#L1747)。先判断 backends.training 是否交给 Megatron；默认路径再处理原生或 HF 模型。

~~~text
解析恢复方式
  → 从 checkpoint 加载 / 继续预训练 / 从蓝图创建
  → 加载 Tokenizer 并检查契约
  → 数据管线与 train/eval 划分
  → TrainingArguments 和分布式配置
  → Trainer 或 SaddleTrainer
  → trainer.train
  → 保存 final_model 与 Tokenizer
  → 写入 _stage_results[pretrain]
~~~

真正的 batch 循环、优化器更新、学习率调度和多数 checkpoint 管理由 Transformers Trainer 及其运行时负责。[SaddleTrainer](../saddlellm/training/NativeTrainer.py#L12) 继承 Trainer，补充原生保存/加载和分块 loss 接口，不是在编排器里另写一个通用反向传播循环。

一次典型优化步骤的概念顺序是：取 batch，forward 得到 loss，backward 累积梯度，到达累积边界后执行裁剪和 optimizer.step，再更新调度器。具体时序与混合精度、分布式封装由训练运行时实现。

用于理解数据并行吞吐的常见关系是：有效 batch ≈ 每设备 microbatch × 梯度累积步数 × 数据并行进程数。张量并行或流水线并行不能直接当作额外的数据并行倍数；配置中的 global_batch_size 还应与实际解析出的梯度累积设置一起核对。

[plan_parallel_training](../saddlellm/training/AutoParallel.py#L168) 做并行决策，启动计划随后通过 CLI 或 launcher 执行。普通预训练 Stage 声明 single、DDP、FSDP、DeepSpeed ZeRO 等策略；SFT/preference Stage 声明 single 和 DDP。是否能执行还取决于模型后端、运行环境和额外校验。

Megatron 是可选训练执行路线，使用独立配置和 launcher；不要同时用普通 distributed 策略再启动一次进程拓扑。默认配置规则见 [BackendConfig.py](../saddlellm/runtime/BackendConfig.py)，执行适配见 [MegatronBackend.py](../saddlellm/training/MegatronBackend.py)。

输出 model_path 会供后续 SFT/eval/export 查找。精确恢复则还需要训练器状态及兼容配置；只有 final_model 权重目录不能自动代表优化器级恢复点。

关联测试：[test_scratch_pretrain_e2e.py](../tests/test_scratch_pretrain_e2e.py)、[test_distributed_pretrain.py](../tests/test_distributed_pretrain.py)、[test_auto_parallel.py](../tests/test_auto_parallel.py)。

自查：为什么“模型已经加载”不等于“训练已恢复到原步数”？应该从哪里查 global_step、优化器和学习率状态？

## 7 监督微调和偏好训练

SFT 入口是 [TrainingOrchestrator._run_sft_stage](../saddlellm/training/TrainingOrchestrator.py#L2261)。它先检查是否启用，处理可选 Megatron 分支，然后选择模型和数据、检查样本、写训练计划，最后调用执行器。

~~~text
模型：pretrain.model_path → 或 model.name_or_path
数据：sft.data_path → 或 mopd.data_path
  → TrainingDataInspector
  → 阶段计划
  → dry_run / preflight_only 时返回计划
  → saddle: train_native_sft
  → hf: train_model → train_sft
  → 保存模型、Tokenizer、训练结果
  → _stage_results[sft]
~~~

选择优先级由源码中的 or 链决定。MOPD 数据若标记 requires_review，且 SFT 准备自动接收它，编排器会要求先提供明确审核后的 sft.data_path。

原生路径见 [train_native_sft](../saddlellm/training/NativePostTraining.py#L429) 和 [_sft_features](../saddlellm/training/NativePostTraining.py#L81)。后者先归一化样本，再构造训练特征。纯 text 样本使用全序列目标；messages 样本按 assistant 回合展开，计算提示前缀边界并屏蔽提示标签。

下面是 _sft_features 中实际的关键语句：

~~~python
prompt_length = _common_prefix_length(prompt_ids, input_ids)
if prompt_length >= len(input_ids):
    continue
labels = [-100] * prompt_length + input_ids[prompt_length:]
~~~

它比较分别编码的提示和完整文本的共同前缀。若截断使回答完全消失，就丢弃该样本；有效样本仅在回答区域产生监督。多轮对话可能展开为多个训练特征，所以 JSONL 行数不必等于最终样本数。

HF 路径见 [train_sft](../saddlellm/training/PeftSFTTrainer.py#L456)：准备 tokenizer 和数据，处理 completion-only 兼容性，加载模型，按需建立量化配置和 LoRA 配置，构造 TRL SFTTrainer，训练并保存。HF 路径是否只监督回答取决于 response_template 及当前 TRL 的能力；不能直接套用原生 messages 路径的默认行为。

LoRA 通过低秩可训练参数修改模型行为，QLoRA 还涉及量化加载基础模型及相应训练准备。当前 SaddleModelAdapter 明确不开放原生 LoRA/QLoRA。是否冻结参数应查看执行器创建后的 requires_grad 和 PEFT 包装，不能只看配置名。

偏好阶段见 [TrainingOrchestrator._run_preference_stage](../saddlellm/training/TrainingOrchestrator.py#L2429)。原生走 train_native_dpo，HF 走 [train_preference](../saddlellm/training/Preference.py#L302)，后者按 method 选择 TRL 的 DPO/ORPO/KTO 训练器。

原生 DPO 核心在 [NativeDPOTrainer.compute_loss](../saddlellm/training/NativePostTraining.py#L253)：

~~~python
logits = self.beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
loss = -F.logsigmoid(logits).mean()
~~~

这里的 policy_chosen 等是回答 token 的序列对数概率之和。策略模型对 chosen/rejected 做前向；参考模型被冻结并在 no_grad 中计算同一对回答。优化目标是提高策略相对参考模型对 chosen 的偏好。这里名为 logits 的变量是偏好差值，不是 B×T×V 的词表 logits。

读取 DPO 时特别关注 _preference_features 的 prompt/completion 边界、截断后是否还保留回答、labels 中 -100 的位置。纯文本 SFT loss、completion-only SFT loss 和 DPO 的序列概率目标有不同含义。

rlhf Stage 当前可执行 DPO 路线；PPO 在稳定流水线中尚未接通，非计划执行会明确阻断。仓库中存在独立 RL 类并不代表所有算法都进入这个 Stage。

关联测试：[test_post_training.py](../tests/test_post_training.py)、[test_native_dpo_boundaries.py](../tests/test_native_dpo_boundaries.py)、[test_native_posttraining_split.py](../tests/test_native_posttraining_split.py)、[test_distributed_posttraining.py](../tests/test_distributed_posttraining.py)。

断点建议：SFT 特征完成后、训练器创建后、DPO compute_loss 返回前。自查：哪些参数参与更新？参考模型是否产生梯度？一个全被截断的回答是否还应成为训练样本？

## 8 蒸馏和模型压缩

MOPD 主入口是 [TrainingOrchestrator._run_mopd_stage](../saddlellm/training/TrainingOrchestrator.py#L3413)，核心采集函数是 [MultiTeacherOnPolicyDistiller.collect](../saddlellm/distillation/OnPolicyDistillation.py#L91)。

~~~text
prompts
  → 当前学生生成回答
  → 各教师针对提示与学生回答提供信号
  → 聚合并选择教师结果
  → instruction/output + student_response + teacher_signals
  → mopd_on_policy.jsonl
  → 后续 SFT 消费数据
~~~

collect 会按 num_rollouts_per_prompt 扩展提示，检查学生是否对每个提示返回一条字符串，再按 teacher_concurrency 调用教师。输出同时保存选中的教师、原学生回答、教师信号和审核状态。该 Stage 的主要产物是数据；学生权重的更新发生在后续训练器中。

backends.rollout=sglang 时通过 RolloutBackend 生成学生样本，并检查策略版本及相关配置。这里的 rollout 后端与 serve --backend 不是同一选择器。缺 prompts 或处于 dry-run 时可能只写 mopd_plan.json，因此应查看 status 和 data_path。

独立 distill 命令还有自己的路径，入口 [_cmd_distill](../saddlellm/cli.py#L2282)，算法模块在 [distillation/](../saddlellm/distillation/)。学习时把“收集教师示例”和“利用教师分布训练学生”两种流程分开追踪；一个 MOPD 采集结果不代表已完成参数蒸馏。

轻量化有两条清晰入口：

| 功能 | 源码链 | 产物与边界 |
|---|---|---|
| quantize | _cmd_quantize → ModelQuantizer.quantize → save | 当前 CLI 路线为 CPU dynamic INT8，保存可重载的量化目录 |
| prune | _cmd_prune → ModelPruner.prune → save | 保存剪枝结果并记录实际稀疏率；稠密权重变成零不自动缩小矩阵或提升速度 |

源码：[_cmd_quantize](../saddlellm/cli.py#L2446)、[ModelQuantizer.quantize](../saddlellm/compression/quantizer.py#L143)、[_cmd_prune](../saddlellm/cli.py#L2496)、[ModelPruner.prune](../saddlellm/compression/pruner.py#L241)。

serve 的 4bit/8bit 加载、原生 CUDA int8 路径和 CPU dynamic INT8 是不同实现。当前原生 CUDA int8 加载器对架构与权重格式有专门限制，不能把一种量化产物交给任意后端。

关联测试：[test_distillation_cli_runtime.py](../tests/test_distillation_cli_runtime.py)、[test_distillation_logits_compat.py](../tests/test_distillation_logits_compat.py)、[compression/test_quantizer.py](../tests/compression/test_quantizer.py)、[compression/test_pruner.py](../tests/compression/test_pruner.py)。

自查：MOPD 已完成但学生回答完全没变化，首先应该确认后续哪一步？剪枝后文件大小未减少，是否必然说明剪枝失败？

## 9 评测和导出如何衔接

主链：[TrainingOrchestrator._run_eval_stage](../saddlellm/training/TrainingOrchestrator.py#L3638) → [Evaluator.evaluate](../saddlellm/evaluation/LLModelEvalute.py#L32) → EvaluationReleaseGate.evaluate → [TrainingOrchestrator._run_export_stage](../saddlellm/training/TrainingOrchestrator.py#L3710) → [ModelExporter.export](../saddlellm/runtime/ModelExporter.py#L77)。

eval 按固定优先级寻找 rlhf、preference、sft、pretrain 的模型产物，再回退到配置中的模型路径。Evaluator 加载记录、模型和 tokenizer，计算配置的指标，返回 metrics 和 error。它会把异常写进 error；编排器会再检查 error，按 fail_on_error 决定是否抛出。

门禁依据 metrics 和 rules 产生 accepted 及失败原因，保存 release_gate.json。启用 eval.gate，或 export.require_gate 为真时，export 必须看到被接受的门禁结果。是否因门禁拒绝立即停止 eval，由 fail_on_rejection 控制；即使 eval 没立即抛错，export 仍有自己的门禁检查。

独立 evaluate CLI 还分流到 completion、multiple-choice、rubric 等评测器；这些能力不能假定都已被统一 eval Stage 自动选择。入口见 [_cmd_evaluate_model](../saddlellm/cli.py#L2594)。

ModelExporter 先验证源和目标关系，再区分原生 checkpoint、PEFT adapter、本地完整模型或远端模型。原生模型保持 Saddle 格式；adapter 按选项合并或保留；导出检查必需文件，并生成 release_manifest.json。保留的 adapter 仍依赖基础模型。失败时会写不完整标记。

当前 ModelExporter 支持 HF 和 Saddle 格式，ONNX/GGUF 在该导出入口尚未启用。导出文件完整性和评测质量是两层判断：manifest 存在不代表质量提升，指标通过也不代表任意 serving 后端都支持该架构。

关联测试：[test_release_pipeline.py](../tests/test_release_pipeline.py)、[test_model_evaluation.py](../tests/test_model_evaluation.py)、[test_completion_evaluation.py](../tests/test_completion_evaluation.py)、[test_rubric_evaluation.py](../tests/test_rubric_evaluation.py)。

自查：为什么一个带 error 的评测返回字典不能当成成功？为什么未合并 LoRA 的目录需要额外检查基础模型来源？

## 10 一条聊天请求的完整调用链

启动入口：[_cmd_serve_model](../saddlellm/cli.py#L3283)、[resolve_serving_settings](../saddlellm/runtime/ServingPlan.py#L57)、[create_inference_app](../saddlellm/runtime/InferenceServer.py#L1081)。

~~~text
serve MODEL
  → InferenceServerSettings
  → resolve_serving_settings
  → 地址检查
  → create_inference_app
      vllm → create_vllm_inference_app
      local/native → load_inference_model → LocalGenerationEngine
  → uvicorn.run
~~~

auto 会综合平台、CUDA、vLLM 版本、checkpoint 类型及显式控制参数选后端。兼容 Linux CUDA 环境的普通 HF 完整模型可走 vLLM；原生/adapter 文件或本地专用参数会影响选择。选中引擎启动失败不会自动换另一套引擎重新试。只有显式 native 路径才承诺使用原生执行；local 也可能加载 Saddle checkpoint，因此 local 不等于只能调用 HF。

原生权重加载在 [load_native_causal_lm](../saddlellm/models/ModelLoader.py#L312)：先识别 Saddle checkpoint；外部架构交给 NativeArchitectures 的适配检查。CUDA int8 使用独立限制路径。支持某个蒸馏版模型取决于其实际架构，不代表支持同名系列所有架构。

~~~mermaid
sequenceDiagram
    participant C as 客户端
    participant R as API 路由
    participant E as GenerationEngine
    participant M as 模型或原生会话
    C->>R: chat/completions 请求
    R->>R: 参数校验与准入
    R->>E: 准备请求或执行生成
    E->>E: chat template 和 tokenize
    E->>E: 输入长度与输出预算检查
    E->>M: prefill 和 decode
    M-->>E: token
    E-->>R: 文本与计数
    R-->>C: JSON 或 SSE
~~~

local/native 的非流式路由进入 nonstream_response，经 run_until_disconnect 管理取消，再进入 nonstream_work。后者按配置走合批，或在 semaphore 保护下在线程中调用 engine.chat/complete。

[LocalGenerationEngine._chat_prompt](../saddlellm/runtime/InferenceServer.py#L695) 应用模型聊天模板；只有显式设置 enable_thinking 才向模板传递覆盖值。[LocalGenerationEngine._prepare_generation](../saddlellm/runtime/InferenceServer.py#L721) 编码提示并核对上下文预算，输出 _PreparedGeneration。默认超长输入报错，显式 truncate_input 才截断；输出预算还会受服务上限和模型上下文限制调整。

[LocalGenerationEngine._generate](../saddlellm/runtime/InferenceServer.py#L751) 把输入移到模型设备，设置生成参数、停止条件和可选前缀缓存。普通路径调用模型 generate；支持逐步生成且需要协作取消时，直接创建原生 session 并执行 step 循环。

流式请求先准备请求，再由 stream_response、engine.stream_chat/stream_complete 等逻辑输出 SSE。原生逐步生成和外部模型 streamer 的桥接路径不同，应在 _stream_generate 中看实际分支。

聊天历史另由 [ChatHistory.py](../saddlellm/runtime/ChatHistory.py) 管理，默认数据库位于服务启动目录下 .saddlellm/chat_history.sqlite3。它保存会话记录；单次生成的 KV 状态由生成会话管理。历史记录存在不代表跨请求自动复用同一份 KV。

协议范围见 [OpenAIProtocol.py](../saddlellm/runtime/OpenAIProtocol.py)：当前文本 API 包含 models、chat/completions、completions 和 SSE；不覆盖工具调用、多模态请求或 Responses API。

断点建议：resolve_serving_settings 返回前、_chat_prompt 返回后、_prepare_generation 返回后、模型调用前。记录实际 selected backend、prompt_tokens、requested_tokens、device 和 do_sample。

关联测试：[test_serving_plan.py](../tests/test_serving_plan.py)、[test_openai_compatibility.py](../tests/test_openai_compatibility.py)、[test_vllm_serving.py](../tests/test_vllm_serving.py)、[test_chat_history_policy.py](../tests/test_chat_history_policy.py)。

自查：用户看到了 SSE 分片，能否据此判断 GPU 正在合批？不能；传输方式、调度策略与模型执行是不同层。

## 11 原生生成和推理性能模块

原生入口是 [SaddleForCausalLM.generate](../saddlellm/models/SaddleModeling.py#L2209)，它创建 [NativeGenerationSession](../saddlellm/models/NativeGeneration.py#L32) 并循环 step。普通原生 generate 与原生流式调用共享这套会话机制。

~~~text
创建 session
  → 准备输出缓冲区、缓存、采样参数
  → prefill 处理提示词
  → 获得下一 token logits
  → 采样并追加 token
  → 检查 EOS、停止条件与 token 上限
  → 若未结束，复用缓存计算下一步
  → close 释放会话资源
~~~

[NativeGenerationSession._step_locked](../saddlellm/models/NativeGeneration.py#L440) 会优先使用 _pending_logits；没有待用 logits 时，第一步进入 prefill 路径，后续调用 _forward_logits。采样后更新 buffers、generated_tokens 和 finished 状态，再返回本步 token。

[NativeGenerationSession._sample](../saddlellm/models/NativeGeneration.py#L403) 先处理重复惩罚和最短生成长度约束。不采样时取 argmax；采样时依次处理 temperature、top_k、top_p，最后从 softmax 分布中抽样。它生成 token ID，文本拼接和 stop 字符串处理还有服务层逻辑。

普通 KV 缓存保存各层历史 key/value，使后续解码可以只计算新增 token；请求需要自己的缓存写入状态。前缀缓存可以复用已计算的提示前缀，但有模型、输入、形状和预算等适用条件。代码会复制/隔离复用数据，避免后续 decode 或取消破坏缓存项。

| 模块 | 工作位置 | 阅读时要区分的事情 |
|---|---|---|
| InferenceCache | 张量存储 | 缓存容量、有效长度、更新与回收 |
| NativePrefixCache | 提示词复用 | 复用多少前缀，与请求私有 KV 的关系 |
| InferenceBatcher | 请求调度 | 排队、兼容分组、请求数和 token 预算 |
| NativeStreamBatcher | 原生会话协作 | register、step、unregister，哪些会话可以共同执行 |
| NativeBatchDecode | 模型执行 | 如何把多个会话的 decode 变成批量前向 |
| InferenceLifecycle | HTTP 取消 | 客户端离开后如何通知并等待实际计算退出 |
| InferenceOptimizer | 模型优化准备 | 投影融合、CPU 预排布等配置与适用条件 |

调度入口：[InferenceBatcher.submit](../saddlellm/runtime/InferenceBatcher.py#L140)、[NativeStreamBatcher.step](../saddlellm/runtime/NativeStreamBatcher.py#L90)。缓存与优化源码在 [InferenceCache.py](../saddlellm/models/InferenceCache.py)、[NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py)、[InferenceOptimizer.py](../saddlellm/models/InferenceOptimizer.py)。

取消的一个关键原因是 asyncio 任务取消不会自动停止 asyncio.to_thread 中已经运行的模型调用。服务需要设置取消事件，并在实际调用结束前保留相应许可，避免表面释放并发槽位后 GPU 上仍残留计算。实现见 [InferenceLifecycle.py](../saddlellm/runtime/InferenceLifecycle.py) 和 InferenceServer 的 nonstream_work。

native_stream_batching 属于显式开启路线，并对 dtype、设备、量化等组合有校验。当前代码将 precision-oriented 和 throughput-oriented 的一些执行选择分开；矩阵形状或融合方式变化可能带来浮点差异。研究优化时要分别比较正确性、单请求延迟和多请求总吞吐，不能从一个内核耗时推断整机服务提升。

关联测试：[test_native_generation_session.py](../tests/test_native_generation_session.py)、[test_native_prefix_cache.py](../tests/test_native_prefix_cache.py)、[test_native_stream_batching.py](../tests/test_native_stream_batching.py)、[test_inference_disconnect.py](../tests/test_inference_disconnect.py)、[test_native_batch_decode.py](../tests/test_native_batch_decode.py)。

自查：semaphore 增加并发许可、请求合批、逐步合批、KV 前缀复用分别改变了哪一层？请求取消时谁负责关闭 session？

## 12 多模态和世界模型扩展

多模态各分支有独立训练目标，需要分别追踪输入表示。

| Stage | 输入与执行链 | 当前结果 |
|---|---|---|
| media_cache | 原始媒体清单 → codec 编码 → 分片缓存与 manifest | 供后续专用训练器读取 |
| image_generation | 图像 latent 与文本条件 → train_latent_flow | 潜变量 flow 模型训练产物 |
| music_generation | 音频码本与文本条件 → train_music_code | 多码本自回归模型训练产物 |
| video_generation | 视频 latent 与文本条件 → train_video_latent_flow | 时空 latent flow 模型训练产物 |
| mllm_sft / vision_alignment | 图文记录 → MultimodalDataAdapter → plan | 当前编排入口主要规范化数据和写计划 |
| vla_sft | 机器人记录 → 动作空间与数据规范化 → 可选 train_vla_sft | vla.train 开启且有规范化数据时进入行为克隆训练 |

代码入口：[build_media_cache](../saddlellm/multimodal/MediaCache.py#L161)、[train_latent_flow](../saddlellm/multimodal/LatentFlowTrainer.py#L202)、[train_music_code](../saddlellm/multimodal/MusicCodeTrainer.py#L230)、[train_video_latent_flow](../saddlellm/multimodal/VideoLatentFlowTrainer.py#L211)、[TrainingOrchestrator._run_multimodal_stage](../saddlellm/training/TrainingOrchestrator.py#L2699)、[TrainingOrchestrator._run_vla_stage](../saddlellm/training/TrainingOrchestrator.py#L3264)。

media_cache 把媒体编码与参数训练分开，便于复用缓存。缓存 shape、codec 和模型配置需要匹配，不能仅凭缓存目录存在就继续训练。内置基线 codec 可用于跑通路径，实际生成质量还取决于媒体表示和模型训练。

世界模型可从独立 train-world-model 命令进入，也可由 [TrainingOrchestrator._run_world_model_stage](../saddlellm/training/TrainingOrchestrator.py#L3116) 纳入统一流水线。

~~~text
轨迹数据中的 observation、action、reward、done
  → prepare_world_model_training
  → WorldModelTrainer.train
  → 观测编码
  → RSSM 的确定性状态和随机潜状态更新
  → 观测、奖励、继续概率预测
  → 多项 loss 加权与参数更新
  → checkpoint 和评测结果
~~~

入口：[prepare_world_model_training](../saddlellm/world_models/_WorldModelTrainer.py#L489)、[WorldModelTrainer.train](../saddlellm/world_models/_WorldModelTrainer.py#L174)、[WorldModel](../saddlellm/world_models/_WorldModel.py#L482)。

RSSM 即循环状态空间模型：用历史状态和动作产生先验预测，用实际观测更新后验。训练让模型重建观测、预测奖励与 episode 是否继续，同时用 KL 项约束先验和后验。Gaussian 实现中的基础损失包括 reconstruction、reward、continuation 和平衡 KL；空间相关可选头还可能增加损失。categorical 变体使用离散潜状态，不能逐字段套用 Gaussian 的均值/标准差表示。

imagine 从已有状态出发，依据动作在潜空间展开未来预测，见 [WorldModel.imagine](../saddlellm/world_models/_WorldModel.py#L823)。模型预测的轨迹需要与实际反馈或受控仿真比较；预测本身不证明真实环境中的结果。

关联测试：[test_media_cache.py](../tests/test_media_cache.py)、[test_latent_flow_trainer.py](../tests/test_latent_flow_trainer.py)、[test_music_code_trainer.py](../tests/test_music_code_trainer.py)、[test_video_latent_flow.py](../tests/test_video_latent_flow.py)、[test_world_model.py](../tests/test_world_model.py)、[test_native_world_model_framework.py](../tests/test_native_world_model_framework.py)。

自查：为什么通用图文 Stage 写出了 plan，不能推出视觉语言模型参数已更新？为什么世界模型的潜状态和语言模型的 KV cache 用途不同？

## 13 空间规划和 WorldAgent 的状态流

一次空间规划从 [SpatialWorldModelCoordinator.plan_image](../saddlellm/spatial/SpatialWorldModel.py#L433) 进入：

~~~text
俯视图或平面图
  → TopDownMapExtractor 提取 OccupancyGrid
  → 可选语义分析
  → 解析起点和终点并校正至可通行位置
  → GridPathPlanner 生成候选路线
  → 可选 WorldModelRouteScorer 评分
  → SpatialPlanResult
~~~

几何地图与语义分析各有职责。未配置视觉语言分析器时仍可从图像提取几何信息，但不会凭空获得语义标签。单张透视图的地图能力受显式限制。配置世界模型评分器时，还需要相应 world_state。

[WorldAgentRuntime.run_image](../saddlellm/spatial/WorldAgent.py#L872) 则组合 analyze_image、plan 和 simulate，把单次规划扩展成带 ID 和持久化的状态流程：

~~~text
WorldObservation
  → WorldState
  → WorldPlan
  → WorldSimulation
  → 外部实际执行反馈
  → WorldFeedback 与可用 replay
~~~

run_image 完成分析、计划和仿真；真实反馈需另外进入 [WorldAgentRuntime.record_feedback](../saddlellm/spatial/WorldAgent.py#L793)。反馈函数核对 plan、route、simulation 的关联，转换实际轨迹坐标并比较计划。实际路径至少包含两个点时，才会追加可用的反馈 replay 记录。

WorldAgentStore 负责对象持久化，规划算法在空间模块，动态预测在世界模型模块。记录 feedback 并不会在这个方法里直接启动世界模型再训练；后续数据转换和训练需要走相应入口。

空间工作台的前后端链是：

~~~text
spatial-studio/src/App.tsx 与组件
  → api/spatialApi.ts 的 planSpatialImage
  → POST /api/plan
  → SpatialAPI
  → 空间规划协调器
  → 地图、路线与分析结果
  → MapCanvas 和 InspectorPanel 展示
~~~

源码：[spatialApi.ts](../spatial-studio/src/api/spatialApi.ts)、[SpatialAPI.py](../saddlellm/spatial/SpatialAPI.py)、[MapCanvas.tsx](../spatial-studio/src/components/MapCanvas.tsx)。WorldAgentAPI 另外暴露有状态的 agent 操作；不要把两个 API 面的能力混为一谈。

关联测试：[test_spatial_world_model.py](../tests/test_spatial_world_model.py)、[test_spatial_api.py](../tests/test_spatial_api.py)、[test_world_agent.py](../tests/test_world_agent.py)。

自查：地图上的路线评分、仿真成功、实际执行成功分别由谁产生？缺少实际轨迹时，能否直接视为已有可训练 replay？

## 14 周边功能的阅读入口

主干掌握后再按使用频率进入这些模块。它们的存在不意味着每次 train 或 serve 都会自动执行。

| 功能 | 入口与主要实现 | 阅读重点 |
|---|---|---|
| 工作区和训练计划 | factory/Quickstart.py、TrainingFactory.py、FactoryBackendPlanner.py | 生成哪些配置，何时交还编排器 |
| 环境与数据诊断 | cli._cmd_doctor、TrainingDataInspector.py、EnvironmentDiagnostics.py | 检查的条件、抽样范围和错误如何返回 |
| 训练看板 | runtime/WebUI.py、TrainingProgress.py、monitordashbord.py | 如何从状态、日志和指标文件生成展示 |
| Embedding 服务 | runtime/EmbeddingServer.py | 文本批处理、向量维度、归一化和响应结构 |
| Reranker 服务 | runtime/RerankerServer.py | query/document 如何编码、打分与排序 |
| 实验规划与报告 | experiments/ | 区分估算、实际测量、日志汇总 |
| 外部算子 | agents/RealWorldOperators.py、operator Stage | 外部输入、调用边界、结果与产物登记 |
| 对齐实验模块 | alignment/、tuning/ | 逐入口确认是否接入稳定流水线及所需依赖 |

公共 API 由 [_exports.py](../saddlellm/_exports.py) 统一登记；CLI 能力以 [cli.py](../saddlellm/cli.py) 中实际参数与 _cmd_* 实现为准。发布依赖分组在 [pyproject.toml](../pyproject.toml)，没有必要为了阅读控制流程安装全部 GPU 运行时。

## 15 故障定位与最小学习实验

先判断问题发生在哪个边界，再进入算法细节。

| 现象 | 先检查的变量或文件 | 源码位置 |
|---|---|---|
| train 一开始就拒绝配置 | stages、model.backend、blueprint、distributed | _parse_config、_validate_config、ModelAdapter |
| 计划成功但训练找不到数据 | 原始路径、规范化后的 data_path、实际工作目录 | DataPipeline.collect、TrainingDataInspector |
| loss 异常或回答不学提示要求 | labels、-100、截断、有效回答长度 | collator、_sft_features、模型 loss |
| 恢复提示配置不一致 | run_fingerprint、plan_signature、Trainer checkpoint | PipelineStateStore、恢复校验函数 |
| 显示完成但找不到模型 | result_status、model_path、planned/skipped 分支 | Stage 实现和 summary.results |
| export 被阻止 | gate.accepted、require_gate、评测 error | _run_eval_stage、_run_export_stage |
| 部署用了意外的后端 | backend_selection.requested/selected/reason | resolve_serving_settings |
| 多轮回答不对 | 模板、messages、tokenizer 与模型匹配 | _chat_prompt、ChatWebRoutes |
| 长输入被拒绝 | prompt_tokens、requested_tokens、上下文上限 | _prepare_generation |
| 请求取消后仍有计算 | cancel_event、线程实际退出、session.close | InferenceLifecycle、nonstream_work |
| 并发增加但吞吐未提高 | 是否实际合批、组内兼容性、每步模型调用 | InferenceBatcher、NativeStreamBatcher |
| 空间评分没有使用世界模型 | route_scorer、world_state、use_world_model | SpatialWorldModelCoordinator、WorldAgentRuntime |

下面的纯计划示例可以在已能导入 saddlellm 的 Python 环境运行，无需下载模型。a、b、c 只是计划节点，不会执行训练。

~~~python
from saddlellm.framework import PipelinePlan

linear = PipelinePlan.compile(["a", "b", "c"])
assert linear.stages == ("a", "b", "c")
assert linear.levels == (("a",), ("b",), ("c",))

dag = PipelinePlan.compile(
    ["c", "a", "b"],
    dependencies={"a": [], "b": [], "c": ["a", "b"]},
)
assert dag.stages == ("a", "b", "c")
assert dag.levels == (("a", "b"), ("c",))
assert dag.descendants(["a"]) == ("a", "c")
print(dag.to_dict())
~~~

随后可以尝试把 c 依赖 a、a 又依赖 c，预测异常再执行。理解这个例子后，再读 test_unified_pipeline.py 中使用 FunctionStage 的测试，跟踪 StageContext.results 如何逐步积累。

若已有项目测试环境，下面是现有测试的学习分组，不代表本次全部运行：

~~~powershell
# 编排、插件与恢复
python -m pytest tests/test_unified_pipeline.py tests/test_framework_plugins.py -q

# 数据与模型契约
python -m pytest tests/test_document_aware_packing.py tests/test_native_model_lifecycle.py -q

# 生成会话与协议
python -m pytest tests/test_native_generation_session.py tests/test_openai_compatibility.py -q
~~~

## 16 当前能力边界与阅读核对

学习时优先记住这些会影响判断的事实：

- Stage 支持依赖图排序，当前编排循环仍串行执行各阶段。
- 原生模型实现与训练器基础设施分开：SaddleTrainer 复用 Transformers Trainer 生命周期。
- pipeline 状态、业务 result_status、模型 checkpoint 和质量指标是不同证据。
- consumes/produces 声明和产物登记没有消除内置阶段的固定字段衔接逻辑。
- 原生后训练当前支持全参数 SFT 与 DPO；HF 路线覆盖更多 PEFT/偏好训练组合。
- 通用图文 Stage 的计划产物、VLA 的可选训练、专用媒体训练需要分别判断。
- native/local/vllm、CPU dynamic INT8、CUDA int8 和 4bit/8bit 加载各有适用范围。
- 空间规划、仿真和真实反馈分别产生证据，replay 收集之后仍需要后续训练流程。

本次文档通过当前源码和现有测试内容核对，源码链接按 AST 中实际定义定位。文中的训练质量、吞吐和外部后端能力不作本次实测结论。CLI dry-run 与纯计划示例的检查只验证各自路径；测试执行记录见阅读路线中的本次核对范围。

需要查完整函数签名时使用 [FUNCTION_REFERENCE.md](FUNCTION_REFERENCE.md)，需要按课推进时回到 [源码阅读路线](SOURCE_READING_ROADMAP_ZH.md)。
