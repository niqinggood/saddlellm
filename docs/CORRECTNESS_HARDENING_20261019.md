# 2026-10-19：正确性与训练、压缩链路整改

本轮落实 [2026-10-18 源码审计](LLM_INTERVIEW_FRAMEWORK_AUDIT_ZH.md) 中的 P0 问题，并接通数据切分审计、公开蒸馏、能力门槛选模和一条 QAT 产物的实际 CUDA 执行路径。版本仍为 2.42，在既有未提交修改上增量实现。

## 已完成的整改

| 范围 | 当前行为 | 验收边界 |
|---|---|---|
| 旧 benchmark | GSM8K 生成答案后做数值匹配；MMLU 如实记录科目子集、各科有效样本与失败；未知任务/HumanEval 明确拒绝；运行失败保留空分数及错误状态 | 使用确定性本地夹具验证评分；没有下载标准测试集测生产模型分数 |
| 投机解码 | greedy 拒绝位置替换正确；采样按 min(1,p/q) 接受，拒绝后从正部 p-q 归一化分布采样；处理 bonus/EOS/预算，保留完整上下文 | 单序列、同 token-ID 词表、完整 causal logits；参考实现不继承 generate 的隐藏 processors；没有 baseline 实测时不返回加速比 |
| 精确去重 | 对完整 NFC/空白/casefold 规范化文本计算 SHA256；尾部不同的长文均保留；缺近似依赖时明确记录 exact fallback | 规范化全文去重不代表完成全部近重复或 benchmark 污染治理 |
| 模拟训练入口 | BalancedPretrainer / balance_data 在下载、读数据和写输出前明确拒绝 | 正常数据混合与真实 pretrain 流水线保留；没有把随机 loss 原型改名后继续使用 |
| Scaling | 拟合与 Chinchilla 约束最优解使用 A/B/alpha/beta，校验参数及预算 | 连续近似不替代离散架构、数据可用性与部署约束 |
| 数据切分 | HF preference 按 prompt/显式来源组切分；HF/native SFT 与 DPO、HF ORPO/KTO 和公开蒸馏串联运行时审计 | 相同来源、prompt、内容或实际 token 输入会阻断；近重复为提示，不宣称无污染 |
| 蒸馏与 GRPO | 公开蒸馏使用 shift/mask/tokenmean/T²，真实执行 batch 与独立验证；冻结教师，校验词表及 hidden；GRPO 保留生成 EOS，忽略其后 padding | 异构词表/hidden 不静默对齐；PPO 稳定流水线仍不在本轮实现范围 |
| 选模与导出 | 先通过全套能力门槛，再按指定新任务指标排序；导出获选模型；拒绝无合格候选、报告路径冲突和内容变更 | 显式候选列表；完整/已合并模型；不自动扫描目录，也不以 eval loss 代替能力回归 |
| 公开导出 | 消除两个 RolloutSample 的重复字典键；保留原生效的运行时 RolloutSample，将对齐模块类型显式命名为 RLScalingRolloutSample | 保持已有顶层 RolloutSample 的实际指向，不再静默遮蔽另一类型 |
| QAT CUDA | portable QAT 的既有 INT8 codes/FP32 scales 可直接用于已有 Triton W8A16 内核，无重新量化或常驻稠密权重副本 | 显式选择 CUDA 与 FP16/BF16；只支持推理；激活精度变化需要独立质量验证 |

## 数据和 packing 的配置

训练切分、回放过滤和运行时审计复用相同来源身份：`group_id`、`conversation_id`、`case_id`、`patient_id`、`document_id`、`source_record_id`，也识别 metadata 中的这些字段。泛化验证会检查同 prompt；源记录分组发生在对话展开前。分组 ID 0 有效，布尔值不会被当成来源 ID。

运行时在训练前保存 `split_audit.json`；HF 还检查 TRL 实际准备后的输入并保存 `prepared_split_audit.json`。检查覆盖 padding、截断、completion mask / assistant masks，以及 DPO/ORPO/KTO 不同字段布局。报告不包含原始 prompt 或来源 ID；失败也有记录。数据指纹用于绑定来源，不能证明测试题从未进入训练集。

```yaml
data:
  pack_sequences: true
  packing_attention: auto  # auto / concat / isolated
```

- `auto`：原生 Saddle 使用文档隔离；通用 HF 使用 concat，并在结果中记录实际策略。
- `concat`：移除 document_ids 与重置的 position_ids，使用普通块内 causal attention；已有边界 label mask 继续保留。
- `isolated`：当前需要原生 Saddle；通用 HF/Megatron 在训练启动前明确拒绝。

这里把 attention 可见性、位置编号与 loss mask 分开声明。默认 `auto` 保持原有行为，并保持旧运行配置指纹兼容。

## 用能力门槛选择 checkpoint

```yaml
stages: [eval, export]
eval:
  regression_baseline: ./models/teacher
  regression_suite: ./configs/capability_suite.yaml
  checkpoint_candidates:
    - ./outputs/run/checkpoint-100
    - ./outputs/run/checkpoint-200
  selection_task: new-task
  selection_metric: accuracy
  selection_higher_is_better: true
export:
  enabled: true
  require_gate: true
```

`selection_task` / `selection_metric` 必须已在 regression suite 的规则中声明，指标方向必须一致。suite 应同时包含旧任务退化预算、新任务绝对门槛以及压缩后需要保留的能力。示例模型、数据与套件路径须替换为项目实际产物。

也可直接调用：

```python
from saddlellm import select_checkpoint

selection = select_checkpoint(
    "./models/teacher",
    ["./outputs/checkpoint-100", "./outputs/checkpoint-200"],
    "./configs/capability_suite.yaml",
    selection_task="new-task",
    selection_metric="accuracy",
    output="./outputs/selection.json",
)
print(selection["selected_model"])
```

每个候选保留完整回归报告；只对所有门槛通过的候选排序，完全相同的分数按输入顺序选择。无候选通过时保存拒绝报告并抛错。候选、baseline、数据/设置内容在选择过程中改变会失去资格或使整个选择失效。导出再次核对获选模型身份，评估之后修改权重不能沿用旧 gate。

## QAT 产物的显式 CUDA 加载

```python
import torch
from saddlellm.models.ModelLoader import load_causal_lm
from saddlellm.models.QATW8A16 import qat_cuda_status

model = load_causal_lm(
    "./outputs/qat",
    device="cuda:0",
    dtype="bfloat16",
    quantized_backend="cuda-w8a16",
    local_files_only=True,
)
with torch.inference_mode():
    outputs = model(input_ids=input_ids.to("cuda:0"))
print(qat_cuda_status(model))
```

需要实际可用的 CUDA/Triton；不满足设备、dtype 或内核条件时拒绝，不静默改用浮点路径。该模式拒绝训练、labels、梯度和普通浮点 checkpoint 导出。读取 `.weight` 的兼容路径会临时物化权重并计数；应通过 `qat_cuda_status` 判断实际内核调用和物化次数。

默认加载仍是 portable W8A32 参考语义。CUDA 模式保留整数 codes 与 FP32 scales，但将激活改为 FP16/BF16，不能宣称与 FP32 逐位一致；也没有将 PTQ 的 compressed-tensors 自动接入这一格式专用路径。

## 实测与证据

最终主环境合并回归 **563 passed、3 skipped**（28 个测试文件，约 231 秒）。跳过的三项均为可选真实 PTQ 后端用例，已在既有独立环境补测为 **3 passed、0 skipped**；汇总核对了补测的 testcase ID 与主环境跳过项完全一致。安装包、轻量导入、版本元数据、函数索引和仓库必检 Ruff 规则均通过。没有执行全仓测试或真实多 GPU 验收。

本机 RTX 4090 Laptop，Torch 2.6.0+cu124，BF16；以下仅是合成微型 native/HF GPT2 夹具，prompt 128 token、生成 16 token，预热 2 次、重复 5 次，串行并同步 CUDA。显存是本进程 PyTorch 分配峰值，单位 MiB。

| 夹具 | 执行路径 | 中位 token/s | 峰值 MiB |
|---|---|---:|---:|
| Native | 默认 portable W8A32 | 81.87 | 10.05 |
| Native | 相同 codes 的 dense BF16 | 97.73 | 1.47 |
| Native | 直接 W8A16 内核 | 95.16 | 1.08 |
| HF GPT2 | 默认 portable W8A32 | 117.59 | 11.12 |
| HF GPT2 | 相同 codes 的 dense BF16 | 141.22 | 2.54 |
| HF GPT2 | 直接 W8A16 内核 | 102.39 | 1.50 |

短 prompt 下 native W8A16 也比默认参考路径慢。上述样本 greedy 序列相同，但 logits 有混合精度误差；实际内核调用为正，临时权重物化为 0。结果说明整数产物已能直接执行并降低这些夹具的分配峰值，不能推广成普遍提速、生产模型显存收益或业务质量结论。生产发布仍需要封存业务数据、能力退化预算和目标负载下的延迟/吞吐验收。

原始证据：

- [最终合并测试](../outputs/correctness-hardening/final.xml) 与 [汇总记录](../outputs/correctness-hardening/validation.json)。
- [评测、去重与 Scaling](../outputs/correctness-hardening/evaluation-data-evidence.json)、[checkpoint 选择](../outputs/correctness-hardening/checkpoint-selection-evidence.json)。
- [投机解码](../outputs/correctness-hardening/speculative-validation.json)、[修复后的探针](../outputs/correctness-hardening/speculative-probe-after.json)。
- [蒸馏与 GRPO](../outputs/correctness-hardening/posttraining-evidence.json)、[QAT CUDA 验证](../outputs/correctness-hardening/qat-w8a16-validation.json)。
- [Native GPU 原始计时](../outputs/correctness-hardening/qat-w8a16-native-benchmark.json)、[HF GPU 原始计时](../outputs/correctness-hardening/qat-w8a16-hf-benchmark.json)：命令、环境、逐次样本及文件 SHA256。

2026-10-18 的面试 Word 与原审计保留为整改前快照；本页和上述测试记录描述本轮完成状态。完整 RAG/Agent、PPO、通用结构缩模和生产质量/性能验收仍按业务需要推进。

## 同日可读性整理

随后按阅读体验要求整理主流程：展开密集条件和字典，明确原始/训练/验证数据命名，将报告路径校验、原子写入、产物加载与推理保护放入各自职责。入口按执行顺序展开，未增加功能、配置或策略层。公开签名、数据类字段和关键报告/异常行为经过前后核对；相关 18 文件回归为 **343 passed、0 skipped**，不与前面的整改测试数相加。详细记录见 [可读性验证汇总](../outputs/readability-refinement/validation.json)。

## 2026-10-20：恢复、剪枝保存与导出门槛补齐

恢复训练与 QAT 现在复用统一切分审计，按实际分词、截断和 padding 处理后的完整输入与有效提示检查重叠。同一提示搭配不同答案也会在优化前拒绝，QAT 尚未替换 fake-quant 模块；成功报告的 `data.split_audit` 保存检查结果。

剪枝保存从当前 `orig × mask` 物化权重，避免优化后立即保存时读取过期缓存。精确掩码及参数清单随模型保存、加载和再次剪枝保留，同目录保存会刷新 sidecar。恢复与 QAT 对已声明但缺失、不完整或与实体权重不一致的掩码明确拒绝；没有精确掩码声明的旧产物仍保留零值推断兼容路径。

普通指标门槛与 regression suite 均在导出前强制核对评估产物的内容身份。导出先在目标同级临时目录构建，核对原样复制内容，并在发布前复查来源与 adapter；通过后才替换目标。中途失败保留旧产物，发布失败尝试回滚，回滚失败保留旧产物备份并报告路径。合法 LoRA 合并、序列化和 tokenizer 兼容规范化仍作为转换处理，记录来源与最终产物各自的身份。

本轮修改 5 个生产文件；541 项公开签名和带注解字段的结构与上一轮 wheel 一致。两组互不重叠的 18 文件回归分别为 **225 passed** 和 **195 passed、3 skipped**；跳过的三项真实 PTQ 已在既有独立环境全部通过，核对 testcase ID 完全对应，共验证 423 个不同用例。新增用例在上一轮 wheel 上复现 8 项失败，在当前源码上全部通过。

版本仍为 2.42。wheel 构建、内容与 5 个源码文件字节核对、无 torch/transformers 的隔离导入、必检 Ruff、函数索引、版本元数据及 diff 空白检查通过。没有执行全仓测试、生产能力评测或新的 GPU 性能验收。详见 [本轮机器可读记录](../outputs/compression-contracts-20261020/validation.json)。
