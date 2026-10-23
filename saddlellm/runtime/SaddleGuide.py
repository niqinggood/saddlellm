"""Dependency-free, local introduction to supported SaddleLLM workflows.

The catalog is instructional content, not an execution plan or live telemetry.
Importing it does not load models, inspect projects, or start training.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
from copy import deepcopy
from typing import Any

_CATALOG = {
    "catalog_version": 1,
    "title": "SaddleLLM 介绍",
    "description": "从数据准备、训练微调到评测、轻量化和部署，了解 SaddleLLM 的其他用法。这里提供用途、教学数据样式和可复制的命令；实际执行在你自己的终端中完成。",
    "workflow": [
        {
            "id": "prepare",
            "title": "准备数据",
            "description": "明确要学的任务，检查环境、字段与数据划分。",
        },
        {
            "id": "train",
            "title": "训练微调",
            "description": "按数据条件选择 SFT、偏好优化或任务奖励训练。",
        },
        {
            "id": "evaluate",
            "title": "比较效果",
            "description": "同题对比训练前后，保留退化样本和失败记录。",
        },
        {
            "id": "deploy",
            "title": "整理部署",
            "description": "导出真实模型产物，再选择聊天、向量或排序服务。",
        },
    ],
    "categories": [
        {"id": "prepare", "title": "数据与环境"},
        {"id": "train", "title": "训练与优化"},
        {"id": "evaluate", "title": "评测与比较"},
        {"id": "deploy", "title": "导出与服务"},
    ],
    "features": [
        {
            "id": "prepare",
            "title": "先检查环境和数据",
            "category": "prepare",
            "summary": "在真正训练前，确认依赖、设备、数据格式和训练计划。",
            "use_when": "第一次使用工具，或换了模型、数据、机器、训练配置时。",
            "steps": [
                "准备训练集、验证集和最终测试集，避免同一案例跨集合重复。",
                "运行 doctor 查看依赖与设备；运行 inspect-data 查看字段、样本数量和重复风险。",
                "用 preflight 生成配置与预检计划，阅读其中的阻断项和估算。输出通常包括 preflight_plan.json 和 config.json。",
            ],
            "commands": [
                {
                    "label": "检查训练环境",
                    "command": "saddlellm doctor --profile train --output outputs/doctor.json",
                    "note": "不加载模型权重；通过仅表示依赖与设备预检通过。",
                },
                {
                    "label": "检查 SFT 数据",
                    "command": "saddlellm inspect-data data/sft.train.jsonl --task sft --validation-data data/sft.validation.jsonl --test-data data/sft.test.jsonl --output outputs/data-check.json",
                    "note": "默认最多检查 256 条，报告中应区分抽查范围和全量数据。",
                },
                {
                    "label": "生成训练预检计划",
                    "command": "saddlellm preflight data/sft.train.jsonl --stage sft --method lora --base-model ./models/Qwen3-0.6B --root-dir outputs/first-preflight",
                    "note": "生成计划，不开始训练。路径均需替换为你已准备的文件。",
                },
            ],
            "data_example": {
                "label": "SFT JSONL：每行一个 JSON 对象",
                "content": '{"instruction":"将 red 翻译成中文。","output":"红色"}',
            },
            "notes": [
                "模型和输入的实际长度、batch size、优化器仍会影响训练显存；doctor 不等于容量保证。",
                "inspect-data 用于 SFT、偏好和 RL 文本数据；结构化 decision 数据使用自己的目录契约与训练预检。",
            ],
        },
        {
            "id": "sft",
            "title": "SFT：用好答案教模型",
            "category": "train",
            "summary": "给模型问题和期望回答，让它学习回答内容与格式。",
            "use_when": "已经有可靠的问题—答案样本，希望模型更适合自己的任务。",
            "steps": [
                "准备 instruction/output 或工具支持的对话数据，并另外准备验证集。",
                "先查看 dry-run 训练计划，确认基础模型、数据、参数与输出目录。",
                "删除命令末尾的 --dry-run 才会训练；查看 training_summary.json 中 results.sft.model_path，找到实际保存的模型或 adapter。",
            ],
            "commands": [
                {
                    "label": "第一次 LoRA 微调的计划",
                    "command": "saddlellm train sft --model ./models/Qwen3-0.6B --data data/sft.train.jsonl --validation-data data/sft.validation.jsonl --method lora --lora-target-modules all-linear --max-steps 5 --batch-size 1 --gradient-accumulation-steps 1 --max-seq-length 256 --save-steps 5 --eval-steps 5 --local-files-only --output-dir outputs/first-lora --dry-run",
                    "note": "5 步用于跑通流程。正式实验应根据数据规模、验证结果另设训练预算。",
                },
                {
                    "label": "检查已有配置的训练计划",
                    "command": "saddlellm train config.yaml --dry-run",
                    "note": "参数较多时保存为 YAML/JSON 配置，更容易复现。",
                },
            ],
            "data_example": {
                "label": "输入与监督目标",
                "content": '{"instruction":"只输出 3 加 4 的结果。","output":"7"}\n{"instruction":"用礼貌的方式请求同事发文件。","output":"请把文件发给我，谢谢。"}',
            },
            "notes": [
                "SFT 是学习目标；LoRA、QLoRA、full 是参数更新方式，两者不是同一层面的选项。",
                "LoRA 训练少量附加参数；QLoRA 量化加载基础模型后训练 adapter，需要对应依赖与设备；full 更新全部参数。",
                "LoRA adapter 依赖其基础模型。不要把日志目录误认为最终模型路径。",
            ],
        },
        {
            "id": "preference",
            "title": "DPO / ORPO / KTO：学习回答偏好",
            "category": "train",
            "summary": "告诉模型哪些回答更好，或哪些回答可取、不可取。",
            "use_when": "已有回答质量标注，希望改善风格、规范性、任务表现。",
            "steps": [
                "有成对好坏答案时准备 prompt/chosen/rejected；只有逐条审核结论时准备 prompt/completion/label。",
                "选择 DPO、ORPO 或 KTO，先检查数据并查看训练计划。",
                "执行后保存训练摘要与偏好 checkpoint，再用独立问题比较原模型和候选模型。",
            ],
            "commands": [
                {
                    "label": "DPO 计划",
                    "command": "saddlellm train dpo --model ./models/Qwen3-0.6B --data data/pairs.jsonl --method lora --output-dir outputs/first-dpo --dry-run",
                    "note": "DPO 提高 chosen 相对 rejected 的概率。",
                },
                {
                    "label": "ORPO 计划",
                    "command": "saddlellm train orpo --model ./models/Qwen3-0.6B --data data/pairs.jsonl --method lora --output-dir outputs/first-orpo --dry-run",
                    "note": "ORPO 结合监督目标与偏好比率目标。",
                },
                {
                    "label": "KTO 计划",
                    "command": "saddlellm train kto --model ./models/Qwen3-0.6B --data data/reviewed.jsonl --method lora --batch-size 2 --output-dir outputs/first-kto --dry-run",
                    "note": "KTO 使用 true/false 审核标签，实际 batch 必须大于 1。",
                },
            ],
            "data_example": {
                "label": "两种偏好数据（教学示例）",
                "content": 'DPO / ORPO：\n{"prompt":"怎样排错？","chosen":"先看日志并构造最小复现。","rejected":"直接忽略错误。"}\n\nKTO：\n{"prompt":"怎样排错？","completion":"先看日志。","label":true}',
            },
            "notes": [
                "偏好标注的质量会直接影响学习方向，合法格式不代表答案可靠。",
                "可以先 SFT 再做偏好优化，也可以对已有能力的模型单独做偏好训练。命令中的阶段顺序必须明确。",
            ],
        },
        {
            "id": "grpo",
            "title": "GRPO / RLVR：通过可验证奖励学习",
            "category": "train",
            "summary": "模型针对同一问题生成多个回答，用任务验证器计算奖励，再更新模型。",
            "use_when": "答案能通过标签、规则、程序或其他可靠验证器打分。",
            "steps": [
                "在任务项目中定义原始数据、提示词、验证器和训练配置。",
                "先运行该项目的 doctor 与 dry-run，确认奖励定义和样本选择。",
                "运行实验，查看实际优化器更新、采样组、有效回答、奖励与最终任务指标；保留零信号组和退化结果。",
            ],
            "commands": [
                {
                    "label": "公共政策项目预检",
                    "command": "saddlellm experiment --project ../public_policy_rlvr dry-run --config ../public_policy_rlvr/configs/studies/qwen3_06b_direct.yaml --model ./models/Qwen3-0.6B --output-root ../public_policy_rlvr/outputs/studies/my-preview",
                    "note": "此处是公共政策项目提供的 dry-run 子命令。其他项目应先查看自己的帮助。",
                },
                {
                    "label": "公共政策方法比较",
                    "command": "saddlellm experiment --project ../public_policy_rlvr all --config ../public_policy_rlvr/configs/studies/qwen3_06b_direct.yaml --model ./models/Qwen3-0.6B --output-root ../public_policy_rlvr/outputs/studies/my-new-study",
                    "note": "会实际启动项目配置中的整组实验，包括 SFT/DPO/GRPO 等；使用新的输出目录。",
                },
            ],
            "data_example": {
                "label": "公共政策项目的奖励配置片段",
                "content": "reward:\n  correctness_weight: 0.95\n  format_weight: 0.05\nrl:\n  num_generations: 4\n\n回答标签与金标比较，格式分检查 answer 标签；完整数据和验证器由该项目定义。",
            },
            "notes": [
                "GRPO 是优化方法，RLVR 描述奖励可验证的任务设置；不是一种通用 JSONL 格式。",
                "当前这里通过 experiment 项目入口使用 GRPO；不要把它误写成通用 train grpo 命令。",
                "奖励升高需要结合独立测试结果解释；计划步数不一定等于实际有效更新次数。",
            ],
        },
        {
            "id": "decision",
            "title": "结构化决策：选择、评分与是非判断",
            "category": "train",
            "summary": "训练输出候选概率的模型，支持 Choice、Score、Noul 等结构化任务。",
            "use_when": "希望直接得到类别、评分或真假概率，而非自由生成长回答。",
            "steps": [
                "按目录契约准备 dataset.json、train.jsonl、validation.jsonl、calibration.jsonl、test.jsonl。",
                "先验证 typed 训练配置并查看计划，再训练。独立校准集用于拟合概率温度。",
                "开发时评估 validation；冻结方案后才使用最终 test。evaluate-decision 输出 report.json、predictions.jsonl、logits.jsonl 和 token_preflight.json。",
            ],
            "commands": [
                {
                    "label": "结构化训练计划",
                    "command": "saddlellm train configs/decision_typed.yaml --dry-run",
                    "note": "仓库配置使用原生小模型和教学数据。需在包含 configs/data 的源码目录中使用。",
                },
                {
                    "label": "独立评估 typed 模型",
                    "command": "saddlellm evaluate-decision --model ./saved-typed-model --data ./typed-data --output-dir outputs/typed-validation --splits validation --device cpu",
                    "note": "使用实际已保存的 typed 模型目录；仅评估，不更新权重。输出目录必须是新的。",
                },
                {
                    "label": "提供结构化概率接口",
                    "command": "saddlellm serve-decision ./saved-typed-model --device cpu --port 8010",
                    "note": "模型加载后提供 /v1/decisions 接口。",
                },
            ],
            "data_example": {
                "label": "Choice 训练记录（目录内 JSONL 的一行）",
                "content": '{"id":"train-1","state_id":"case-1","task_id":"color","type":"choice","state":"Color: red.","question":"What color is the signal?","choices":[{"id":"red","text":"red"},{"id":"blue","text":"blue"}],"label":"red"}',
            },
            "notes": [
                "Choice 是选项分类，Score 是有序等级评分，Noul 是是非判断；数据字段按各自契约定义。",
                "旧 Choice 冻结骨干评分头与 typed 模型有不同训练契约；evaluate-decision 用于 typed 模型。",
                "本实现不是商业 Jev 的效果复现，模型是否好用需在自己的真实任务上评估。",
            ],
        },
        {
            "id": "distill",
            "title": "蒸馏：训练较小的学生模型",
            "category": "train",
            "summary": "让学生学习教师的输出分布，或用审核后的教师答案做 SFT。",
            "use_when": "希望减少参数规模，或把教师的任务能力迁移到已有小模型。",
            "steps": [
                "logits 模式准备本地教师和文本语料；data 模式准备问题与教师服务，或已审核的教师答案。",
                "先查看参数计划；在线教师采集只生成待审核数据，不自动训练学生。",
                "训练后在独立评测集比较学生前后和教师表现，检查参数量、loss、PPL 或任务准确率。",
            ],
            "commands": [
                {
                    "label": "较小网络的蒸馏计划",
                    "command": "saddlellm distill --mode logits --teacher ./models/Qwen3-0.6B --data data/corpus.jsonl --student-layers 6 --student-hidden-size 256 --student-heads 4 --max-steps 24 --output-dir outputs/small-student --dry-run",
                    "note": "logits 模式会同时加载教师与学生。删除 --dry-run 才会执行蒸馏。",
                },
                {
                    "label": "用已审核教师数据微调学生",
                    "command": "saddlellm distill --mode data --student ./models/Qwen3-0.6B --teacher-data data/teacher-reviewed.jsonl --eval-data data/sft.validation.jsonl --method lora --epochs 1 --output-dir outputs/distilled-sft --dry-run",
                    "note": "teacher-data 是真实审核后的 instruction/output 数据，而非自动生成即可信。",
                },
            ],
            "data_example": {
                "label": "文本语料与已审核答案",
                "content": 'logits 模式语料：\n{"text":"学习时应记录数据来源、训练配置与评测结果。"}\n\ndata 模式教师答案：\n{"instruction":"怎样让实验可复现？","output":"固定数据和配置，并保存命令、日志与结果。"}',
            },
            "notes": [
                "参数少、训练 loss 下降都不能单独证明能力保留。需要独立的真实任务评测。",
                "蒸馏中的数据模式与缩小网络的 logits 模式是两条不同路径。",
            ],
        },
        {
            "id": "compress",
            "title": "轻量化：量化与剪枝",
            "category": "train",
            "summary": "减少数值精度或使部分权重稀疏，研究内存、性能与质量的取舍。",
            "use_when": "已有可用模型，想适配 CPU 或降低推理资源消耗。",
            "steps": [
                "先评测原模型，记录质量和资源基线。",
                "选择 CPU 持久化 INT8、GPU 加载时量化，或稀疏权重剪枝。",
                "再评测产物，在目标设备上检查实际内存与速度，保留质量变化。",
            ],
            "commands": [
                {
                    "label": "持久化 CPU INT8 计划",
                    "command": "saddlellm quantize outputs/release outputs/release-int8 --method dynamic-int8 --local-files-only --dry-run",
                    "note": "删除 --dry-run 后产生新的可加载 INT8 目录；当前 quantize 公开的是 CPU dynamic INT8。",
                },
                {
                    "label": "稀疏剪枝计划",
                    "command": "saddlellm prune outputs/release outputs/release-pruned --ratio 0.15 --method l1_unstructured --local-files-only --dry-run",
                    "note": "保存稀疏权重；普通 dense 后端不保证模型更小或推理更快。",
                },
                {
                    "label": "GPU 加载时 8bit 量化",
                    "command": "saddlellm serve outputs/release --device cuda --quantization 8bit --local-files-only --open",
                    "note": "需要相应依赖；此路径在加载时量化，不生成持久化量化目录。",
                },
            ],
            "data_example": {
                "label": "产物与对比关系",
                "content": "outputs/release          原模型\noutputs/release-int8     持久化 CPU INT8 模型\noutputs/release-pruned   稀疏权重模型\n\n各产物都用相同的独立评测问题比较。",
            },
            "notes": [
                "量化与剪枝可能改变回答。部署成功和效果通过需要分别验证。",
                "QLoRA 属于训练时的参数更新方案；quantize 属于模型产物的轻量化，两者用途不同。",
            ],
        },
        {
            "id": "evaluate",
            "title": "评测与比较：训练到底有没有用",
            "category": "evaluate",
            "summary": "在同一批问题上评估基座和训练后模型，查看整体指标与逐例回答。",
            "use_when": "训练后准备选 checkpoint、判断改进或整理实验结论。",
            "steps": [
                "使用独立验证数据和一致的设备、模板、精度与生成参数。",
                "分别 evaluate 基座和候选 adapter，生成 JSON 与可读报告。",
                "compare 配对比较 validation；选定方案后再评估封存 test。验证集比较用于选择候选，不等于正式发布验收。",
            ],
            "commands": [
                {
                    "label": "评测原模型",
                    "command": "saddlellm evaluate ./models/Qwen3-0.6B data/sft.validation.jsonl --task completion --local-files-only --generation-all --max-new-tokens 64 --output outputs/base-validation.json",
                    "note": "completion 评测关注监督答案 PPL 和实际生成；选择题使用对应的 multiple-choice 契约。",
                },
                {
                    "label": "评测训练后的 adapter",
                    "command": "saddlellm evaluate ./models/Qwen3-0.6B data/sft.validation.jsonl --adapter ./my-adapter --task completion --local-files-only --generation-all --max-new-tokens 64 --output outputs/candidate-validation.json",
                    "note": "./my-adapter 替换为 training_summary.json 中的真实模型路径。",
                },
                {
                    "label": "配对比较验证结果",
                    "command": "saddlellm compare --base-validation outputs/base-validation.json --candidate-validation outputs/candidate-validation.json --output outputs/validation-comparison.json --markdown-output outputs/validation-comparison.md",
                    "note": "只有 validation 时 accepted=false 是预期行为；最终门禁还需 base-test 与 candidate-test。",
                },
            ],
            "data_example": {
                "label": "Completion 评测数据的一行",
                "content": '{"instruction":"只输出 5 加 6 的结果。","output":"11"}',
            },
            "notes": [
                "训练 loss 不能跨算法直接排名，也不能替代真实任务指标。",
                "业务验收应记录改对、改错、格式错误、类别表现与训练成本；小样例只证明工程链路。",
            ],
        },
        {
            "id": "export",
            "title": "导出模型，再接入原推理问答页",
            "category": "deploy",
            "summary": "把真实 checkpoint 或 adapter 整理成发布目录，并通过现有 Web 与接口试用。",
            "use_when": "候选模型已完成评测，希望换机器、交付或重新部署。",
            "steps": [
                "从训练摘要找到实际模型目录，不凭 checkpoint 名称猜测。",
                "export 保存模型、tokenizer 与 manifest；可按需带入已通过的发布门禁。",
                "serve 加载导出目录，重新测试聊天回答和接口。服务启动成功还需独立的质量、容量与稳定性验证。",
            ],
            "commands": [
                {
                    "label": "本地功能试用导出",
                    "command": "saddlellm export ./my-adapter outputs/first-release --local-files-only",
                    "note": "默认合并 LoRA，可能需要额外内存；未带门禁时仅是功能试用导出。",
                },
                {
                    "label": "正式门禁导出",
                    "command": "saddlellm export ./my-adapter outputs/approved-release --gate-result outputs/release-gate.json --require-gate --local-files-only",
                    "note": "需要真实 compare 结果 accepted=true；只有 validation_passed 不满足发布门禁。",
                },
                {
                    "label": "打开推理问答 Web",
                    "command": "saddlellm serve outputs/first-release --backend local --local-files-only --open",
                    "note": "提供现有 Web 对话和兼容接口；本介绍页面不会另行加载第二个模型。",
                },
            ],
            "data_example": {
                "label": "最终交付内容",
                "content": "模型与 tokenizer 文件\nmanifest 与可选权重哈希\n实际训练配置、评测 JSON、逐例回答\n发布门禁结果（正式交付时）",
            },
            "notes": [
                "export 的任务是整理模型产物，serve 的任务是提供推理；两者不会自动训练或自动评测。",
                "如果终端没有 saddlellm 命令，可使用同一环境中的 python -m saddlellm.cli 替换。",
            ],
        },
        {
            "id": "search-services",
            "title": "向量与重排服务：为知识检索配套",
            "category": "deploy",
            "summary": "向量模型把文本变成可检索的表示；重排模型把候选资料按相关性重新排序。",
            "use_when": "为问答系统接入资料库，希望先找到相关片段再生成回答。",
            "steps": [
                "准备独立的本地 embedding 与 reranker 模型，聊天模型不能自动替代它们。",
                "分别启动 serve-embedding 与 serve-reranker，使用各自的服务端口。",
                "调用 /v1/embeddings 获得向量；向 /rerank 提交 query 与 documents 获得排序分数。索引与完整检索工作流由应用侧组织。",
            ],
            "commands": [
                {
                    "label": "启动向量服务",
                    "command": "saddlellm serve-embedding ./models/Qwen3-Embedding-0.6B --device cpu --port 8098 --local-files-only",
                    "note": "需要可由 sentence-transformers 加载的本地向量模型。",
                },
                {
                    "label": "启动 Qwen3 重排服务",
                    "command": "saddlellm serve-reranker ./models/Qwen3-Reranker-0.6B --device cpu --port 8099 --local-files-only",
                    "note": "使用专用 Qwen3 reranker 模型；CPU 支持不意味着足够快。",
                },
            ],
            "data_example": {
                "label": "两种接口请求的关键字段",
                "content": '向量：\n{"input":["怎样让实验可复现？"]}\n\n重排：\n{"query":"怎样让实验可复现？","documents":["固定数据版本和配置。","今天的天气很好。"]}',
            },
            "notes": [
                "这两条命令提供模型服务，不自动创建资料索引，也不代表完整 RAG 系统已经部署。",
                "聊天、结构化决策、向量和重排是不同模型任务，部署时按实际用途选择。",
            ],
        },
    ],
}


_PRETRAIN = {
    "id": "pretrain", "title": "预训练：从零学习或继续学习语料", "category": "train",
    "summary": "通过连续文本的下一个 token 预测学习语言与领域知识，不要求逐条问题—答案。",
    "use_when": "要研究自己的小模型架构，或让已有模型继续学习可靠的领域语料。",
    "steps": ["从零训练使用随机初始化的模型与明确蓝图；继续预训练加载已有权重与对应 tokenizer。两者都与学习问题—答案的 SFT 不同。", "准备 text 字段语料，先去重并划分训练/验证。检查词表、特殊 token、文档结束标记与 packing；示例配置里的 tokenizer 阶段按配置执行，不另造一个 CLI。", "先查看配置 dry-run，准备好语料及继续训练的前置模型后再执行。查看独立验证 loss/PPL、保存 checkpoint 与生成样本，同时检查已有能力是否退化。"],
    "commands": [
        {"label": "从零训练的小模型配置计划", "command": "saddlellm train configs/tinystories_tiny_scratch.yaml --dry-run", "note": "真实配置使用 scratch 模式、tokenizer → pretrain 阶段与 4096 词表的小模型；需要先准备配置所指的故事语料。"},
        {"label": "已有模型继续预训练的计划", "command": "saddlellm train configs/tinystories_zh_16k_continue.yaml --dry-run", "note": "真实配置使用 continue，加载此前 bilingual16k/final_model 的权重与 tokenizer；需要对应前置产物与语料。"},
    ],
    "data_example": {"label": "预训练语料 JSONL：每行一个完整文档", "content": '{"text":"从前有一只小猫，它每天在窗边等朋友。"}\n{"text":"实验记录应保存数据版本、训练配置与独立评测结果。"}'},
    "notes": ["这两个配置路径在项目源码中；配置指定的语料和前置模型不是随安装包自动生成的通用数据。dry-run 不启动训练。", "scratch 例的 4096 词表与 continue 例的 16384 词表不是直接互换的前后阶段。继续训练必须使用配置对应的已有模型与 tokenizer。", "packing 将短文档组织进训练序列；明确 EOS 和文档边界，避免把不相干文本误当同一文档。", "领域语料上的 PPL 下降不等于问答表现提升，需评测目标任务以及希望保留的已有能力。"],
}
_CATALOG["features"].insert(1, _PRETRAIN)
_CATALOG["workflow"][1]["description"] = "按目标选择从零或继续预训练、SFT、偏好优化或任务奖励训练。"
_CATALOG["description"] += " 疑难问题部分提供常见可能原因、排查与可验证修复，不是某次运行的实测日志。"
_PRETRAIN["notes"].append("原生 continue 是从已有权重开始的新训练，使用新配置的学习率和预算，不恢复旧优化器或步数；完整断点续训使用 training.resume_from_checkpoint 或 --resume-from-checkpoint，不能同时把原生 continue 当作完整恢复。")

_EN_TEXT = {
    "title": "About SaddleLLM",
    "description": "Explore SaddleLLM beyond chat: prepare data, train and fine-tune models, evaluate changes, compress models, and deploy services. This guide provides explanations and commands to copy into your own terminal. Commands and teaching examples keep their original literal text. Troubleshooting covers possible causes, checks, and verifiable repairs rather than measured failures from a current run.",
    "workflow": [
        {"title": "Prepare data", "description": "Define the task and check the environment, data fields, and dataset splits."},
        {"title": "Train a model", "description": "Choose pretraining from scratch or continued pretraining, SFT, preference optimization, or training with task rewards."},
        {"title": "Compare results", "description": "Compare models on the same questions and retain regressions and failure records."},
        {"title": "Package and deploy", "description": "Export the actual model artifact, then choose chat, embedding, or reranking services."},
    ],
    "categories": {"prepare": "Data and environment", "train": "Training and optimization", "evaluate": "Evaluation and comparison", "deploy": "Export and services"},
    "features": {
        "prepare": {
            "title": "Check the environment and data first", "summary": "Check dependencies, devices, data formats, and the training plan before training.",
            "use_when": "When starting out or changing the model, data, machine, or training configuration.",
            "steps": ["Prepare separate training, validation, and final test sets. Keep the same case out of different splits.", "Run doctor to inspect dependencies and devices. Run inspect-data to inspect fields, sample counts, and duplication risks.", "Generate a configuration and preflight plan. Read the blockers and estimates. Outputs typically include preflight_plan.json and config.json."],
            "commands": [
                {"label": "Check the training environment", "note": "Does not load model weights. A pass covers dependency and device readiness only."},
                {"label": "Inspect SFT data", "note": "Checks at most 256 records by default. Distinguish the inspected sample from the full dataset."},
                {"label": "Create a preflight plan", "note": "Creates a plan without starting training. Replace the example paths with prepared files."},
            ],
            "data_example_label": "SFT JSONL: one JSON object per line",
            "notes": ["Model size, input length, batch size, and optimizer states still affect training memory. doctor does not guarantee capacity.", "inspect-data handles SFT, preference, and RL text data. Structured decision data uses its own directory contract and training preflight."],
        },
        "pretrain": {
            "title": "Pretraining: start from scratch or continue learning", "summary": "Learn language and domain patterns by predicting the next token in continuous text, without question-answer pairs.",
            "use_when": "When researching a small model architecture or adapting an existing model to reliable domain text.",
            "steps": ["Training from scratch starts with random weights and an explicit architecture. Continued pretraining loads existing weights and their tokenizer. Both differ from SFT on question-answer pairs.", "Prepare a text-field corpus, remove duplicates, and separate training and validation documents. Check the vocabulary, special tokens, document endings, and packing. The example's tokenizer stage runs through its configuration.", "Inspect the dry-run plan before executing. Prepare the corpus and any prerequisite model. Examine held-out loss and PPL, checkpoints, and generated samples, and check whether existing abilities are retained."],
            "commands": [
                {"label": "Plan a small model trained from scratch", "note": "The real configuration uses scratch mode, tokenizer then pretrain stages, and a 4096-token vocabulary. Prepare its referenced story corpus first."},
                {"label": "Plan continued pretraining", "note": "The real configuration uses continue mode and loads the earlier bilingual16k/final_model weights and tokenizer. It requires those prerequisite artifacts and its corpus."},
            ],
            "data_example_label": "Pretraining JSONL: one complete document per line",
            "notes": ["These configuration paths are in the source repository. Their corpus and prerequisite models are not generated automatically by installation. dry-run does not start training.", "The scratch example's 4096-token vocabulary and the continuation example's 16384-token vocabulary are not interchangeable stages. Use the continuation configuration's matching model and tokenizer.", "Packing combines short documents into training sequences. Make EOS markers and document boundaries explicit so unrelated text is not treated as one document.", "Lower domain PPL does not prove better answers. Evaluate the target task and the existing abilities you need to retain.", "Native continue starts a new run from existing weights, with the new learning rate and budget. It does not restore optimizer or step state. Full checkpoint resume uses training.resume_from_checkpoint or --resume-from-checkpoint; do not combine native continuation with a full resume."],
        },
        "sft": {
            "title": "SFT: teach the model with good answers", "summary": "Provide questions and expected answers so the model can learn their content and format.",
            "use_when": "When reliable question-answer examples are available for your task.",
            "steps": ["Prepare instruction/output examples or a supported conversation format, plus a separate validation set.", "Inspect the dry-run plan and confirm the base model, data, parameters, and output directory.", "Remove --dry-run to train. Read results.sft.model_path in training_summary.json to locate the actual model or adapter."],
            "commands": [
                {"label": "Plan a first LoRA fine-tuning run", "note": "Five steps verify the workflow. Set a separate training budget for a meaningful experiment."},
                {"label": "Inspect an existing training configuration", "note": "Use a YAML or JSON configuration when there are many parameters to reproduce."},
            ],
            "data_example_label": "Input and supervised target",
            "notes": ["SFT is a learning objective. LoRA, QLoRA, and full are ways to update parameters, so they are different kinds of choices.", "LoRA trains added parameters. QLoRA loads a quantized base model and trains an adapter, with suitable dependencies and hardware. full updates all parameters.", "A LoRA adapter still depends on its base model. The log directory may not be the final model directory."],
        },
        "preference": {
            "title": "DPO / ORPO / KTO: learn answer preferences", "summary": "Show which answer is better, or whether an individual answer is desirable.",
            "use_when": "When answer-quality labels are available and you want to improve style, consistency, or task performance.",
            "steps": ["Prepare prompt/chosen/rejected for answer pairs, or prompt/completion/label for individual reviews.", "Choose DPO, ORPO, or KTO. Inspect the data and training plan first.", "Retain the training summary and preference checkpoint, then compare the base and candidate models on independent questions."],
            "commands": [
                {"label": "Plan DPO training", "note": "DPO increases the probability of chosen relative to rejected."},
                {"label": "Plan ORPO training", "note": "ORPO combines a supervised objective with a preference ratio objective."},
                {"label": "Plan KTO training", "note": "KTO uses true/false review labels. Its actual batch must contain more than one example."},
            ],
            "data_example_label": "Two preference formats: teaching examples",
            "notes": ["The quality of preference labels affects what the model learns. A valid format does not make an answer reliable.", "Preference optimization may follow SFT, or run on a model that already has the needed abilities. Specify the stage order explicitly."],
        },
        "grpo": {
            "title": "GRPO / RLVR: learn from verifiable rewards", "summary": "Generate several answers to a question, score them with a task verifier, and update the model.",
            "use_when": "When labels, rules, programs, or other reliable verifiers can score answers.",
            "steps": ["Define the data, prompts, verifier, and training configuration in a task project.", "Run that project's doctor and dry-run first to inspect its rewards and sample selection.", "Inspect actual optimizer updates, sampled groups, effective answers, rewards, and final task metrics. Retain groups with no learning signal and regressions."],
            "commands": [
                {"label": "Preflight the public-policy project", "note": "This dry-run subcommand belongs to the public-policy project. Other projects define their own commands."},
                {"label": "Compare public-policy training methods", "note": "Starts the project's configured experiment matrix, including methods such as SFT, DPO, and GRPO. Use a new output directory."},
            ],
            "data_example_label": "Reward configuration excerpt from the public-policy project",
            "notes": ["GRPO is an optimization method. RLVR describes a task with verifiable rewards; it is not one universal JSONL format.", "This workflow uses GRPO through the experiment project entry point, rather than a generic train grpo command.", "Interpret reward improvements alongside independent task results. Planned microsteps and effective optimizer updates may differ."],
        },
        "decision": {
            "title": "Structured decisions: choices, scores, and yes/no", "summary": "Train candidate-probability models for Choice, Score, and Noul tasks.",
            "use_when": "When you need categories, ratings, or true/false probabilities instead of free-form answers.",
            "steps": ["Prepare dataset.json, train.jsonl, validation.jsonl, calibration.jsonl, and test.jsonl according to the directory contract.", "Validate the typed training configuration and inspect its plan before training. Fit probability temperature on a separate calibration set.", "Use validation during development and final test only after freezing the plan. evaluate-decision writes report.json, predictions.jsonl, logits.jsonl, and token_preflight.json."],
            "commands": [
                {"label": "Inspect a structured-training plan", "note": "The repository example uses a small native model and teaching data. Run it where the source configs and data are available."},
                {"label": "Evaluate a saved typed model", "note": "Use the actual saved typed model directory. This evaluates without updating weights, into a new directory."},
                {"label": "Serve structured probabilities", "note": "Loads the model and provides the /v1/decisions endpoint."},
            ],
            "data_example_label": "One Choice training record inside the dataset directory",
            "notes": ["Choice selects an option, Score predicts an ordered rating, and Noul is a yes/no decision. Each uses its own record contract.", "The older frozen-backbone Choice head and typed models have different training contracts. evaluate-decision is for typed models.", "This implementation does not reproduce commercial Jev quality. Evaluate it on your own real task."],
        },
        "distill": {
            "title": "Distillation: train a smaller student", "summary": "Teach a student from teacher output distributions or run SFT on reviewed teacher answers.",
            "use_when": "When reducing parameter count or transferring a teacher's task abilities to an existing small model.",
            "steps": ["For logits mode, prepare a local teacher and text corpus. For data mode, prepare prompts and a teacher service, or reviewed teacher answers.", "Inspect the plan first. Online answer collection creates data for review and does not automatically train the student.", "Evaluate independently before and after training, and against the teacher. Examine parameter count, loss, PPL, or task accuracy."],
            "commands": [
                {"label": "Plan distillation into a smaller network", "note": "Logits mode loads both teacher and student. Remove --dry-run to execute."},
                {"label": "Fine-tune on reviewed teacher answers", "note": "teacher-data must be reviewed instruction/output examples, rather than automatically trusted generated answers."},
            ],
            "data_example_label": "Text corpus and reviewed teacher answers",
            "notes": ["Fewer parameters or lower training loss alone do not demonstrate retained abilities. Use independent task evaluation.", "Data distillation and logits distillation into a smaller network are different workflows."],
        },
        "compress": {
            "title": "Compression: quantization and pruning", "summary": "Reduce numerical precision or make weights sparse to study memory, speed, and quality tradeoffs.",
            "use_when": "When adapting a working model to CPU or reducing inference resource use.",
            "steps": ["Record the original model's quality and resource baseline.", "Choose persistent CPU INT8, quantization during GPU loading, or sparse-weight pruning.", "Evaluate the result and measure actual memory and speed on the target device. Keep the quality changes."],
            "commands": [
                {"label": "Plan persistent CPU INT8", "note": "Remove --dry-run to create a new loadable directory. quantize currently exposes CPU dynamic INT8."},
                {"label": "Plan sparse pruning", "note": "Saves sparse weights. Ordinary dense backends do not guarantee a smaller file or faster inference."},
                {"label": "Load a model with GPU 8bit quantization", "note": "Requires the relevant dependencies. Quantizes during loading rather than creating a persistent quantized directory."},
            ],
            "data_example_label": "Artifacts to compare",
            "notes": ["Quantization and pruning can change answers. Verify service startup and model quality separately.", "QLoRA is a training update scheme. quantize creates a compressed model artifact; the purposes differ."],
        },
        "evaluate": {
            "title": "Evaluation: did training help?", "summary": "Evaluate the base and trained model on the same questions, with aggregate metrics and individual answers.",
            "use_when": "When selecting a checkpoint, assessing improvements, or writing experiment conclusions.",
            "steps": ["Use independent validation data with matching devices, prompts, precision, and decoding settings.", "Run evaluate for the base model and candidate adapter to create JSON and readable reports.", "Use compare to pair validation results. Evaluate the sealed test set after selecting the plan. Validation selection is not final release approval."],
            "commands": [
                {"label": "Evaluate the original model", "note": "Completion evaluation measures answer PPL and generation. Multiple-choice tasks use their corresponding data contract."},
                {"label": "Evaluate the trained adapter", "note": "Replace ./my-adapter with the actual model path from training_summary.json."},
                {"label": "Compare paired validation results", "note": "accepted=false is expected with validation only. Final approval also requires base-test and candidate-test."},
            ],
            "data_example_label": "One completion evaluation example",
            "notes": ["Training loss cannot rank different algorithms directly or replace task metrics.", "Track cases that improved or regressed, format failures, class performance, and training cost. Tiny examples verify the engineering workflow only."],
        },
        "export": {
            "title": "Export a model and use the existing chat page", "summary": "Package the actual checkpoint or adapter into a release directory, then try it through the existing Web UI and API.",
            "use_when": "When a candidate has been evaluated and needs to move machines, be delivered, or redeployed.",
            "steps": ["Locate the actual model directory from the training summary rather than guessing from a checkpoint name.", "Export weights, tokenizer, and a manifest. Include a passed release gate when required.", "Serve the exported model and retest chat and API responses. Startup still needs separate quality, capacity, and stability checks."],
            "commands": [
                {"label": "Export for a local functional trial", "note": "Merges LoRA by default and may require extra memory. Without a gate, this is a functional export only."},
                {"label": "Export with a release gate", "note": "Requires a real compare result with accepted=true. validation_passed alone does not meet the release gate."},
                {"label": "Open the inference chat Web UI", "note": "Provides the existing chat Web UI and compatible API. This guide does not load another model."},
            ],
            "data_example_label": "What to deliver",
            "notes": ["export packages model artifacts and serve provides inference. Neither automatically trains or evaluates.", "If saddlellm is not on PATH, use python -m saddlellm.cli from the same environment."],
        },
        "search-services": {
            "title": "Embedding and reranking services for retrieval", "summary": "Embedding models create searchable text vectors. Rerankers sort candidate passages by relevance.",
            "use_when": "When connecting a document collection to question answering and retrieving useful passages first.",
            "steps": ["Prepare separate local embedding and reranker models. A chat model does not automatically replace them.", "Start serve-embedding and serve-reranker on separate ports.", "Call /v1/embeddings for vectors and /rerank with query and documents for scores. The application organizes the index and full retrieval workflow."],
            "commands": [
                {"label": "Start an embedding service", "note": "Requires a local embedding model that sentence-transformers can load."},
                {"label": "Start a Qwen3 reranker service", "note": "Use a dedicated Qwen3 reranker model. CPU support does not guarantee sufficient speed."},
            ],
            "data_example_label": "Key fields in the two API requests",
            "notes": ["These commands provide model services. They do not automatically build a document index or deploy a complete RAG system.", "Chat, structured decisions, embeddings, and reranking are different model tasks. Choose the service that matches the need."],
        },
    },
}


def _issue(symptom: str, causes: str, checks: str, fixes: str, verification: str) -> dict[str, Any]:
    """Build a static troubleshooting entry with the public list contract."""
    return {"symptom": symptom, "causes": [causes], "checks": [checks], "fixes": [fixes], "verification": [verification]}


_TROUBLESHOOTING = {
    "zh": {
        "prepare": [
            _issue("数据能读出来，但训练检查不通过", "可能是缺少任务必需字段、字段类型不对，或把其他任务的数据格式用于当前训练。", "核对任务类型与示例字段；查看具体失败行、空值、answer 或 label 的类型。", "按所选任务的契约修复数据，先对少量记录重新检查；不要只改字段名称而忽略含义。", "确认检查报告无阻断项，并抽看实际模型输入和监督目标。"),
            _issue("划分后的指标很好，但可能存在数据泄漏", "可能是相同案例、同组织记录或相似正文跨训练和评测集合。", "检查案例 ID、分组字段、完全重复与近似重复；区分去重前和实际抽样后的重叠。", "按实际泛化目标划分案例或组织，先分组再抽样；受污染测试集应作为历史数据另行记录。", "重新审计划分，并在新的未用于调参的留出数据上验收。"),
        ],
        "pretrain": [
            _issue("继续预训练启动失败，或出现词表大小不匹配", "可能加载了另一词表的 tokenizer、模型蓝图或特殊 token 配置。", "比较 tokenizer 词表、模型 embedding 尺寸和 token ID；核对 continue 配置指向的权重与 tokenizer。", "使用对应的 tokenizer 与已有模型；确需扩词表时显式实现和核验 resize，不能靠替换路径完成。", "确认所有 token ID 在模型范围内，检查短文本分词、前向和保存重载的一致性。"),
            _issue("语料很多，但有效训练 token 很少或 loss 异常", "可能是过滤后语料过少、长度策略丢弃正文，或 packing 与 EOS 处理不符合预期。", "查看过滤前后文档数、实际 token 数、打包序列和文档边界；抽看 labels 与 padding。", "修正 text 字段、过滤阈值与分段/packing 配置，明确文档结束 token。", "统计有效监督 token，并检查多文档样本的边界与损失是否合理。"),
            _issue("训练 loss 或领域 PPL 下降，但原有任务能力退化", "可能是领域语料比例过高、更新幅度过大，或在训练语料上做了评测。", "同时查看独立领域验证集和原有任务题集，核对评测语料重叠、学习率与训练预算。", "调整语料混合、更新预算或学习率，在验证集选择候选，不用最终测试集反复调参。", "用相同设置比较领域指标和保留任务，报告提升与退化两部分。"),
            _issue("以为在续训，实际重新初始化了模型或优化器", "可能混淆 scratch、continue、加载已有权重和恢复完整训练状态。", "核对 pretrain_mode、实际加载的权重，以及 checkpoint 中是否有 optimizer、scheduler 和步数。", "从零用 scratch；已有权重继续学习用 continue；恢复中断训练使用支持的 checkpoint 恢复设置。", "记录加载路径、训练起始步数和优化器状态是否恢复，不能把重新开始写成完整续训。"),
        ],
        "sft": [
            _issue("训练样本有答案，但没有有效监督 token", "可能是 completion mask、response_template 或截断位置不匹配，导致 labels 全为 -100。", "抽看 tokenized input、labels 与非 -100 数量；检查被截断后的答案和 EOS 是否仍参与因果损失。", "修正数据模板和监督区间，缩短过长提示或合理提高序列长度；不要把全部 prompt 当答案训练。", "确认每个训练样本都有有效的答案监督 token；短跑时检查损失和梯度为有限值。"),
            _issue("训练 loss 下降，验证或实际回答却变差", "可能是过拟合、数据标签质量问题、模板记忆或训练/评测设置不同。", "同题比较训练前后回答，查看验证曲线、类别表现、重复数据与截断记录。", "修正错误标签，调整数据覆盖和训练预算，按验证指标选择 checkpoint。", "在相同独立评测集上报告改对、改错和关键任务表现，而非只展示 loss。"),
            _issue("训练开始或第一次更新时显存 OOM", "可能是序列太长、batch 太大，或梯度、优化器状态及其他进程占用超出预算。", "区分加载、前向、反向和优化器更新发生 OOM 的阶段；检查 GPU 实际可用显存和训练配置。", "优先减少序列长度或单卡 batch，用梯度累积保持所需批量；按能力使用 checkpointing 或 QLoRA。", "重新检查实际峰值与有效训练 token；持续运行到优化器更新成功，不能只验证模型加载。"),
            _issue("LoRA 看起来没学习，或部署仍像原模型", "可能是 target modules 未命中、没有可训练参数，或评测/服务加载了错误 adapter 或基础模型。", "检查可训练参数、adapter 权重变化和 summary 中真实 model_path；核对评测请求加载路径。", "修正目标模块，使用正确配对的基座与 adapter；合并导出后重新加载验证。", "确认 adapter 参数实际更新，且保存前后重载结果一致；再比较独立题集的输出。"),
            _issue("loss 或梯度出现 NaN/Inf", "可能是数值精度、学习率、异常长样本或数据监督异常导致不稳定。", "定位首个非有限值的步骤，检查对应样本、梯度范数、学习率与精度；不要跳过错误记录。", "先修复异常数据，按验证结果降低更新幅度或换适合设备的精度，在小规模运行中排查。", "确认 loss、梯度和更新后的参数均有限，并保留失败与修复后的配置对照。"),
            _issue("从 checkpoint 恢复后，步数或学习率从头开始", "可能只有 adapter 权重，缺少 optimizer、scheduler 或 trainer 状态；仅加载模型不等于完整恢复。", "查看 checkpoint 内容与日志中 resumed_from_checkpoint、global_step 和学习率。", "恢复中断任务时使用完整支持的训练 checkpoint；只有模型权重时作为新训练并明确记录。", "确认恢复步数和调度连续，保存重载后继续运行；别把权重初始化称作完整断点续训。"),
        ],
        "preference": [
            _issue("DPO 数据通过 JSON 解析，却没有偏好学习效果", "可能是 chosen 与 rejected 相同、好坏颠倒，或缺少正确配对的 prompt。", "逐对检查问题、两个回答和审核依据；统计相同答案、空答案与矛盾偏好。", "删除无区别样本并纠正偏好方向，保持同一 prompt 下的有效好坏回答。", "确认有效偏好对数量，并抽查训练后同题回答是否朝审核目标变化。"),
            _issue("DPO/ORPO 长样本被截断后 loss 异常", "可能提示占满序列，chosen 或 rejected 的答案部分被截掉，没有足够有效 token。", "分别检查 prompt、chosen、rejected 的 token 长度和截断后的 completion mask。", "合理缩短提示或分段，调整 prompt 和总序列预算，保持两边答案监督有效。", "检查两支答案均保留有效 completion token，记录真实截断比例。"),
            _issue("DPO loss 变好、reward margin 变大，实际评测反而退化", "可能只学会区分训练偏好对，或偏好标注与实际业务目标不一致。", "比较独立题集、验证指标和逐例退化；查看奖励统计是否来自训练对而非实际任务。", "纠正偏好数据覆盖和标注依据，调整训练预算与 beta，并按独立验证结果选候选。", "同时报告偏好训练指标与真实任务指标；margin 不能替代质量验收。"),
            _issue("DPO 的显存占用比 SFT 高很多或 OOM", "可能是长 prompt 的成对答案计算、参考模型或相关缓存占用了额外内存。", "检查实际后端是否加载参考模型、成对 token 数、batch 与 OOM 发生阶段。", "减少成对长度和单卡 batch，按后端能力选择参数高效训练与 checkpointing；先核验实现再改参考模型设置。", "验证完整优化器更新和峰值显存，同时保证偏好对没有被截成空监督。"),
            _issue("DPO adapter 保存成功，但重载或对话格式不匹配", "可能是基座版本、TRL/Transformers/PEFT 兼容性、chat_template 或特殊 token 设置不同。", "核对 adapter_config、基座 revision、依赖版本、tokenizer 与同题最终 prompt。", "使用匹配的依赖和基座，保持训练与评测模板一致；用实际保存产物做重新加载测试。", "记录重载输出和格式有效率，并在相同解码配置下核对评测结果。"),
            _issue("KTO 报 batch 或 label 问题", "可能有效 batch 只有一条，或 label 是字符串而非明确的 true/false，数据只含单一审核方向。", "检查实际 batch、label 类型和可取/不可取数量，而非只看命令的配置名称。", "按 KTO 契约提供二元标签并使用大于一条的实际 batch；补充有依据的审核样本。", "确认训练检查通过、实际批次满足要求，再比较独立任务表现。"),
        ],
        "grpo": [
            _issue("采样很久却很少更新，奖励标准差总是零", "可能同组回答奖励完全相同，题目太易/太难，或验证器把不同答案都判成同一分数。", "查看原始回答、奖励分解、group reward std、接受与丢弃组数和零信号步数。", "先修复验证器与奖励分辨率，再调整题目覆盖和采样设置；零差异组不要伪装成有效学习。", "确认奖励能区分真实质量不同的回答，并报告有效组数与实际优化器更新。"),
            _issue("奖励上涨，但内容质量下降或只学会格式", "可能存在奖励投机：格式、长度等容易获得的分数掩盖了答题错误。", "分开统计正确性、格式和其他奖励分量；抽查高奖励但答错的真实回答。", "修正验证器与奖励权重，补充对抗样本，保持独立任务指标作为验收依据。", "用新留出题检查内容正确性与格式有效率，不能只看总奖励。"),
            _issue("KL、loss 或梯度突然变大并出现 NaN", "可能更新幅度、采样设置、参考策略计算或长回答造成数值不稳定。", "定位异常步骤，查看 KL、梯度范数、概率/log-prob、奖励和完成长度的原始记录。", "修复非有限值来源，按实现调整学习率、KL 控制和长度预算；保留失败样本再做小规模排查。", "确认采样和更新数值有限，并在同设置下观察独立评测是否稳定。"),
            _issue("计划 128 步，但实际学习次数少于 128", "可能动态采样丢弃无信号组，或某些微步骤没有执行优化器更新。", "分别查看 microsteps、optimizer_updates、accepted/discarded groups、sampled/effective rollouts。", "在报告中拆开预算与有效学习数量；若有效率低，先审计题目和奖励，而非改摘要凑齐步数。", "以原始训练日志核对实际更新、耗时与有效回答，并据此比较训练成本。"),
        ],
        "decision": [
            _issue("模型几乎总预测同一个类别", "可能类别失衡、输入信息不足、初始化或学习率导致类别塌缩；多任务训练也可能相互影响。", "查看逐类预测数、召回率、混淆矩阵，核对正文与实例问题是否进入模型。", "先修复输入与标签，再比较单任务和联合任务；按验证结果调整训练参数。", "确认类别覆盖和少数类召回，在相同独立样本上比较，而不只看平均准确率。"),
            _issue("温度校准后，分类准确率没有变", "可能预期了温度缩放并不具备的行为：正温度改变概率尺度，通常不改变 logits 的 argmax。", "分别比较未校准/校准的 NLL、Brier、ECE 和预测类别；核对校准范围。", "用校准改善概率解释；要改变错误类别需优化数据和模型，不能靠温度保证改对。", "同时报告分类质量和概率质量，保留校准退化的情况。"),
            _issue("Score 准确率看着接近，实际评分误差却大", "可能等级有顺序，错误跨越很多等级；只看完全匹配率掩盖了误差幅度。", "检查等级定义、预测分布、MAE、within-one-level 和不同真实等级的误差。", "纠正等级映射和数据覆盖，按任务损失和独立验证结果优化，不随意重映射标签。", "比较准确率及 MAE/等级误差，说明使用 argmax 分数还是期望分数。"),
        ],
        "distill": [
            _issue("教师和学生的 logits 无法对应或出现维度错误", "可能词表、tokenizer、特殊 token 或 token 对齐不同，不能逐位置直接计算 KL。", "核对教师/学生词表与 token ID，检查 causal shift、mask 和 padding 对齐。", "使用实现支持的对应词表与 tokenizer；跨词表蒸馏需显式对齐，不能只改 hidden size。", "在短样本上核验 KL 与监督损失，检查有效 token 和保存重载一致性。"),
            _issue("学生学到了教师错误，或变小后质量明显下降", "可能教师数据未审核、教师能力不足、学生容量/语料覆盖不足，或只在训练数据上验收。", "抽查教师答案，比较教师、原学生与新学生在独立任务上的错误类型。", "修正教师数据，补充可靠样本，按验证结果调整学生规模和训练预算。", "报告参数缩减与能力保留两者；保留教师也答错的样本。"),
        ],
        "compress": [
            _issue("剪枝后模型稀疏了，但文件或推理并未变小变快", "可能仍使用 dense 存储和 dense kernel，零权重并未减少实际计算。", "检查存储格式、推理后端，以及相同设备和负载下的峰值内存与耗时。", "采用真正支持稀疏的后端或结构化压缩方案；否则把它作为稀疏权重研究而非加速结论。", "在目标设备重复测量性能，并同时评估质量，不能以非零权重数代替速度。"),
            _issue("INT8/4bit/8bit 模型能启动，但未明显加速", "可能转换开销、CPU/GPU 实现、batch 或模型大小使低精度优势没有体现。", "确认使用的是 CPU 持久化 INT8 还是 GPU 加载时量化，测量预热后延迟与实际内存。", "选择匹配设备的实现，按真实负载比较精度与 batch；避免把不同后端的结果直接归因于位宽。", "分别记录加载、预热和稳定推理成本，在相同输入下比较。"),
            _issue("量化或剪枝后出现新的错答", "可能精度误差或被移除的权重影响了关键任务；压缩比不等于质量保证。", "逐例对比原模型和产物，查看关键任务、少数类别与格式失效。", "按验证结果减小压缩力度或选择其他配置，保留原模型作为对照。", "重新评测独立题集，再决定是否采用该压缩产物。"),
        ],
        "evaluate": [
            _issue("评测提升很大，但训练/测试数据可能重复", "可能使用训练集评测，或反复依据测试集挑参数而失去最终留出意义。", "核对数据 ID、内容哈希、分组和选择历史；检查实际评测文件与训练文件。", "用验证集选择方案，冻结后在未用于调参的留出集评测；受污染结果明确标记为历史记录。", "保存数据清单、选择时间与配对题目，重新审计无直接泄漏。"),
            _issue("准确率很高，但少数类别或多标签表现差", "可能多数类占比高，或把多标签部分命中当成完全正确。", "查看标签支持数、逐类召回、Macro-F1、混淆和标签集合完全匹配率。", "使用符合任务的数据统计与指标，补充代表性样本；不要只选最有利的一项指标。", "同时报告整体和逐类表现，标明 @1、avg@k、pass@k 的区别。"),
            _issue("compare 无法配对，或同题对比结果不可信", "可能 ID、题目、金标、模板、精度或生成参数不同；新模型还可能加载错了 adapter。", "检查两个评测文件的样本身份、输入和运行设置，以及真实模型路径。", "对相同题集用一致设置重新评测，缺失或不匹配的样本单独列为未比较。", "核对配对数量与逐例改对/改错，说明比较结论的范围。"),
        ],
        "export": [
            _issue("adapter 导出或移动机器后无法加载", "可能缺少匹配基座、tokenizer、adapter_config，或把日志目录当模型目录。", "读取训练摘要中的真实 model_path，核对 adapter_config 引用的基座与 revision。", "准备匹配基础模型，或使用受支持的合并导出；带上 tokenizer 与完整 manifest。", "在新路径重新加载并发送真实请求，核对保存前后的同题输出。"),
            _issue("export --require-gate 拒绝了验证集通过的模型", "可能只有 validation_passed、accepted=false，尚未完成封存测试门禁。", "读取 compare 决策和 accepted 字段，核对 base-test/candidate-test 是否提供且通过。", "先冻结候选再完成真实最终测试；功能试用导出与正式门禁导出分别标记。", "确认正式 gate accepted=true 且对应同一候选产物，不能修改报告字段伪造通过。"),
            _issue("服务启动了，但长问题失败或回答被截断", "可能输入加输出超过 token 预算，或服务的长度/并发限制与实际请求不匹配。", "核对实际 token 数、输入和生成上限、finish reason 与服务错误信息。", "合理分段资料，调整输入/输出预算并重新测算资源；不能只提高限制而不验证容量。", "对短输入、边界长输入与并发请求分别验收，记录输出截断与稳定性。"),
        ],
        "search-services": [
            _issue("向量或重排接口无法加载模型", "可能用了聊天模型目录、模型不符合支持格式，或缺少对应依赖与 tokenizer。", "核对服务命令、模型类型、目录配置与加载错误；向量和 reranker 需要各自模型。", "使用可由 sentence-transformers 加载的向量模型或支持的 Qwen3 reranker，并匹配依赖。", "分别提交一条真实向量/重排请求，检查输出维度、分数与模型身份。"),
            _issue("检索排序差，或长文档请求超出限制", "可能索引与查询的向量模型不一致、重排输入太长，或批量/长度限制未考虑。", "核对索引模型版本、查询指令、候选文档、实际 token 与服务限制。", "统一 embedding 版本，合理分段并限制候选数量，按任务评测召回与重排。", "对固定标注查询比较召回和排序，同时测试请求边界与资源消耗。"),
        ],
    },
    "en": {
        "prepare": [
            _issue("The file parses, but training validation fails", "Required task fields may be missing or have the wrong types, or the data may use another task's format.", "Check the task contract, failing rows, nulls, and the types of answer and label fields.", "Repair the records for the selected task and reinspect a small sample. Renaming fields alone may not fix their meaning.", "Confirm there are no blockers, and inspect actual model inputs and targets."),
            _issue("Strong evaluation results may include data leakage", "Cases, organizations, or similar text may occur in both training and evaluation splits.", "Inspect case IDs, groups, exact duplicates, and near duplicates. Separate full-split overlap from overlap in the selected sample.", "Split by the unit of generalization before sampling. Retain contaminated test results as historical evidence.", "Audit the new split and assess a new holdout that was not used for tuning."),
        ],
        "pretrain": [
            _issue("Continued pretraining fails with a vocabulary mismatch", "The tokenizer, architecture, or special tokens may belong to a different vocabulary.", "Compare vocabulary size, embedding dimensions, and token IDs. Check the weights and tokenizer paths in the continuation configuration.", "Use the matching model and tokenizer. Vocabulary expansion needs explicit, verified resizing rather than a path substitution.", "Check token IDs are in range and verify tokenization, forward computation, and save/reload consistency on short text."),
            _issue("A large corpus produces few useful tokens or unusual loss", "Filtering may remove most documents, length handling may discard content, or packing and EOS behavior may differ from expectations.", "Inspect document counts before and after filtering, actual tokens, packed sequences, boundaries, labels, and padding.", "Correct the text field, filters, splitting, and packing configuration. Define document endings explicitly.", "Count supervised tokens and inspect boundaries and loss in packed examples."),
            _issue("Domain PPL improves while existing abilities regress", "The domain mix or updates may be too aggressive, or evaluation may use training text.", "Inspect independent domain validation and retained-task results, corpus overlap, learning rate, and budget.", "Adjust corpus mix, update budget, or learning rate using validation results, rather than repeatedly tuning against final test.", "Compare domain gains and retained abilities under the same settings, reporting both gains and regressions."),
            _issue("A supposed continuation resets the model or optimizer", "Training from scratch, loading existing weights, and restoring the full training state may have been confused.", "Check pretrain_mode, loaded weights, and the checkpoint's optimizer, scheduler, and step state.", "Use scratch for random initialization and continue for existing weights. Use supported checkpoint restoration to resume an interrupted run.", "Record the loaded path, starting step, and whether optimizer state was restored. A new run from weights is not a full resume."),
        ],
        "sft": [
            _issue("An answer is present, but no answer tokens receive supervision", "Completion masks, response templates, or truncation may leave every label at -100.", "Inspect tokenized inputs, labels, and supervised-token counts. Confirm that answer tokens and EOS remain after truncation and causal shifting.", "Correct the template and supervision span. Shorten the prompt or choose an appropriate sequence budget without supervising the entire prompt as the answer.", "Confirm each training example retains answer tokens. In a small trial, check that loss and gradients are finite."),
            _issue("Training loss falls, but validation or answers get worse", "Overfitting, incorrect labels, template memorization, or mismatched evaluation settings may be responsible.", "Compare answers before and after training. Inspect validation curves, classes, duplicate data, and truncation records.", "Correct labels, improve coverage, and adjust the budget. Choose checkpoints from validation results.", "Report improved and regressed cases and key task results on the same independent set, rather than loss alone."),
            _issue("GPU memory runs out at startup or the first update", "Sequences, batches, gradients, optimizer states, or other processes may exceed the available memory.", "Identify whether loading, forward, backward, or optimizer update fails. Inspect actual free memory and the training configuration.", "Reduce sequence length or per-device batch first. Use gradient accumulation, checkpointing, or QLoRA where supported and appropriate.", "Measure peak memory and supervised tokens through a successful optimizer update, rather than checking model loading alone."),
            _issue("LoRA seems unchanged, or deployment behaves like the base model", "Target modules may not match, no parameters may be trainable, or evaluation may load the wrong adapter or base model.", "Check trainable parameters, adapter weight changes, the summary's actual model_path, and the loaded evaluation path.", "Correct the target modules and base/adapter pairing. Reload and verify after any merged export.", "Confirm adapter parameters changed and save/reload results agree, then compare independent task outputs."),
            _issue("Loss or gradients contain NaN or Inf", "Precision, learning rate, unusual long samples, or broken supervision may cause numerical instability.", "Locate the first nonfinite step and inspect its data, gradient norm, learning rate, and precision. Retain the failure record.", "Fix abnormal data first. Adjust update size or device-appropriate precision and investigate in a small trial.", "Check that loss, gradients, and updated parameters are finite. Preserve the failed and repaired configurations."),
            _issue("Checkpoint resume resets the step count or learning rate", "The artifact may contain adapter weights without optimizer, scheduler, or trainer state. Loading weights is not a full resume.", "Inspect checkpoint contents, resumed_from_checkpoint, global_step, and the logged learning rate.", "Use a complete supported training checkpoint to resume an interrupted run. Treat weights-only initialization as a new run.", "Confirm step and scheduler continuity, then save, reload, and continue. Describe weights-only initialization accurately."),
        ],
        "preference": [
            _issue("Valid DPO JSON produces no useful preference learning", "Chosen and rejected answers may be identical, reversed, or attached to a missing or mismatched prompt.", "Review the prompt, both answers, and their review rationale. Count identical, empty, and contradictory pairs.", "Remove pairs without a preference distinction and correct the direction while keeping both answers tied to the same prompt.", "Count usable pairs and inspect whether independent answers move toward the reviewed objective."),
            _issue("Long DPO or ORPO examples produce unusual loss after truncation", "The prompt may fill the sequence and leave too few chosen or rejected answer tokens.", "Inspect prompt and answer lengths separately, including the completion masks after truncation.", "Shorten or segment prompts and adjust prompt and total sequence budgets so both answers remain supervised.", "Confirm both branches retain completion tokens and record the actual truncation rate."),
            _issue("DPO loss or reward margin improves while task quality regresses", "The model may only separate training pairs, or labels may not match the real task objective.", "Inspect independent questions, validation metrics, and regressions. Determine whether reward statistics describe training pairs or task performance.", "Improve preference coverage and labels, then adjust the budget and beta using independent validation.", "Report preference-training metrics and task quality together. Margin does not replace quality approval."),
            _issue("DPO uses much more memory than SFT or runs out of memory", "Long paired completions, a reference model, or associated caches may require extra memory.", "Check the actual backend's reference-model behavior, paired token counts, batch size, and failing stage.", "Reduce pair lengths and per-device batches. Use supported parameter-efficient training and checkpointing, and inspect the implementation before changing reference settings.", "Verify complete optimizer updates and peak memory while ensuring neither completion becomes empty supervision."),
            _issue("A saved DPO adapter reloads with a different conversation format", "Base revisions, dependency versions, chat templates, or special tokens may differ.", "Check adapter_config, the base revision, TRL/Transformers/PEFT versions, tokenizer, and the final prompt for the same question.", "Use compatible dependencies and the matching base. Keep training and evaluation templates consistent and reload the actual saved artifact.", "Record reloaded answers and format validity and compare results under the same decoding settings."),
            _issue("KTO reports batch or label errors", "The effective batch may have one example, labels may be strings instead of booleans, or reviews may cover only one direction.", "Inspect actual batches, label types, and desirable/undesirable counts rather than configuration names alone.", "Provide boolean labels under the KTO contract and an actual batch larger than one. Add reviews with a clear rationale.", "Confirm preflight and actual batches meet the requirements, then evaluate independent task performance."),
        ],
        "grpo": [
            _issue("Sampling takes time, but rewards have zero variance and few updates occur", "Answers within a group may receive identical rewards, tasks may be too easy or difficult, or the verifier may collapse distinct answers to one score.", "Inspect answers, reward components, group reward standard deviation, accepted/discarded groups, and steps with no signal.", "Fix the verifier and reward discrimination first, then adjust task coverage and sampling. Do not count indistinguishable groups as useful learning.", "Check that genuinely different answer quality receives different rewards and report effective groups and optimizer updates."),
            _issue("Rewards rise while content gets worse or only formatting improves", "Reward hacking may allow easy format or length rewards to hide incorrect content.", "Separate correctness, format, and other reward components. Inspect high-reward wrong answers.", "Repair the verifier and reward weights, add challenging cases, and retain independent task metrics for approval.", "Check content accuracy and format validity on new holdout questions rather than total reward alone."),
            _issue("KL, loss, or gradients spike and become NaN", "Update size, sampling settings, reference-policy computation, or long completions may destabilize training.", "Locate the failing step and inspect KL, gradient norms, probabilities/log-probabilities, rewards, and completion lengths.", "Fix the source of nonfinite values and adjust learning rate, KL controls, and length budgets according to the implementation. Retain failing samples.", "Check finite sampling and updates and assess independent evaluation stability under matching settings."),
            _issue("A 128-step plan has fewer than 128 effective updates", "Dynamic sampling may discard groups without learning signal, and some microsteps may perform no optimizer update.", "Inspect microsteps, optimizer_updates, accepted/discarded groups, and sampled/effective rollouts separately.", "Report planned budget and effective learning separately. Audit tasks and rewards when effectiveness is low rather than editing summaries to match the plan.", "Reconcile actual updates, time, and useful answers against raw logs before comparing training cost."),
        ],
        "decision": [
            _issue("The model predicts almost one class", "Class imbalance, missing input information, initialization, or learning rate may cause collapse. Joint tasks may also interfere.", "Inspect class prediction counts, recalls, and confusion matrices, and verify the text and instance question reach the model.", "Repair inputs and labels first, then compare single-task and joint training and tune from validation results.", "Check class coverage and minority recall on the same independent examples rather than aggregate accuracy alone."),
            _issue("Temperature calibration leaves classification accuracy unchanged", "Positive temperature scaling changes probability sharpness but ordinarily preserves the logits' argmax.", "Compare raw and calibrated NLL, Brier, ECE, and predicted classes, and check the calibration scope.", "Use calibration to improve probability interpretation. Correcting wrong classes requires improvements to data or the model.", "Report both classification and probability quality, including cases where calibration makes results worse."),
            _issue("Similar Score accuracy hides much larger rating errors", "Ordered levels allow errors across several grades, which exact-match accuracy may hide.", "Check level definitions, prediction distributions, MAE, within-one-level accuracy, and errors by true grade.", "Repair grade mappings and coverage and optimize using the task loss and independent validation without casually relabeling levels.", "Report accuracy and rating error, specifying whether scores use the argmax level or the expected value."),
        ],
        "distill": [
            _issue("Teacher and student logits do not align", "Vocabularies, tokenizers, special tokens, or token positions may differ, so position-wise KL may be invalid.", "Check vocabularies, token IDs, causal shifts, masks, and padding alignment.", "Use the implementation's supported matching vocabulary and tokenizer. Distillation across vocabularies needs explicit alignment, not just a hidden-size change.", "Verify KL and supervised loss on short samples, effective tokens, and save/reload consistency."),
            _issue("The student copies teacher errors or loses too much quality", "Teacher data may be unreviewed, teacher quality or student capacity may be insufficient, or evaluation may only cover training examples.", "Review teacher answers and compare the teacher, original student, and new student on independent error cases.", "Correct teacher data, add reliable examples, and adjust capacity and training budget using validation results.", "Report parameter reduction and retained abilities together, including examples the teacher also gets wrong."),
        ],
        "compress": [
            _issue("Pruning creates zeros but does not reduce file size or latency", "Dense storage and dense kernels may still store and compute those weights.", "Inspect storage and the inference backend, then compare peak memory and timing with the same device and workload.", "Use a backend that exploits sparsity or a supported structural compression method. Otherwise describe the result as sparse-weight research.", "Measure target-device performance and quality. A lower nonzero-weight count is not a speed measurement."),
            _issue("A lower-precision model starts but is not faster", "Conversion overhead, the CPU/GPU implementation, batch size, or model size may offset its potential benefit.", "Distinguish persistent CPU INT8 from quantization during GPU loading. Measure warmed-up latency and actual memory.", "Choose an implementation suitable for the device and compare precision and batching on the real workload without attributing backend changes solely to bit width.", "Record loading, warmup, and steady inference costs separately under matching inputs."),
            _issue("Quantization or pruning introduces new wrong answers", "Numerical error or removed weights may affect key tasks. Compression ratio does not guarantee quality.", "Compare original and compressed outputs case by case, including key tasks, minority classes, and format failures.", "Reduce compression or choose another configuration from validation results and retain the original baseline.", "Evaluate independent questions again before adopting the compressed artifact."),
        ],
        "evaluate": [
            _issue("Large gains may include training/test overlap", "Evaluation may use training data, or repeated test-based tuning may have invalidated the final holdout.", "Check IDs, text hashes, groups, selection history, and the actual evaluation and training files.", "Select using validation, freeze the candidate, and evaluate an unused holdout. Mark contaminated results as historical.", "Retain data manifests, selection times, and paired questions and re-audit direct leakage."),
            _issue("High accuracy hides poor minority or multilabel performance", "A majority class may dominate, or partial multilabel matches may be mistaken for exact correctness.", "Inspect label support, per-class recall, Macro-F1, confusion, and exact label-set matches.", "Use task-appropriate statistics and representative examples rather than selecting only favorable metrics.", "Report aggregate and class results and distinguish @1, avg@k, and pass@k."),
            _issue("compare cannot pair results, or the comparison is unreliable", "IDs, questions, labels, templates, precision, or decoding may differ, or the candidate may load the wrong adapter.", "Inspect both evaluations' sample identities, inputs, settings, and actual model paths.", "Reevaluate the same questions with matching settings and mark missing or mismatched samples as unpaired.", "Reconcile pair counts and improved/regressed cases and state the comparison's scope."),
        ],
        "export": [
            _issue("An adapter fails after export or a machine move", "The matching base, tokenizer, or adapter_config may be missing, or a log directory may have been mistaken for the model.", "Read the actual model_path from the training summary and check the base and revision in adapter_config.", "Prepare the matching base or use supported merged export. Include tokenizer files and the full manifest.", "Reload from the new path, send a real request, and compare the same answers before and after saving."),
            _issue("The release gate rejects a validation-passed model", "The result may only be validation_passed with accepted=false and lack a passed sealed-test gate.", "Read compare's decision and accepted fields and confirm both test evaluations were supplied and passed.", "Freeze the candidate before final testing. Distinguish functional exports from exports with formal release approval.", "Confirm accepted=true refers to the same candidate artifact. Editing report fields does not establish a pass."),
            _issue("The service starts, but long inputs fail or answers are cut off", "Input plus output may exceed the token budget, or request limits and concurrency may not match the workload.", "Inspect actual token counts, input/generation limits, finish reason, and service errors.", "Segment source text and adjust budgets with a fresh capacity estimate. Higher limits alone do not verify capacity.", "Test short inputs, boundary-length inputs, and concurrent requests separately and record truncation and stability."),
        ],
        "search-services": [
            _issue("The embedding or reranking service cannot load its model", "The directory may contain a chat model, use an unsupported format, or lack dependencies or a tokenizer.", "Check the service command, model type, configuration, and loading error. Embeddings and reranking need their respective models.", "Use a sentence-transformers-compatible embedding model or supported Qwen3 reranker with matching dependencies.", "Send a real request to each service and check vector dimensions, scores, and model identity."),
            _issue("Retrieval ranking is poor, or long documents exceed limits", "Index and query embeddings may use different models, reranking inputs may be too long, or batch and length limits may be overlooked.", "Check model versions, query instructions, candidate passages, actual token counts, and service limits.", "Use one embedding version, segment documents, limit candidate counts, and evaluate recall and ranking on the task.", "Compare fixed labeled queries and test request boundaries and resource use."),
        ],
    },
}


def saddle_catalog(lang: str = "zh") -> dict[str, Any]:
    """Return fresh static guide content in Chinese or English, without I/O."""
    if lang not in {"zh", "en"}:
        raise ValueError("Unsupported guide language; use zh or en")
    catalog = deepcopy(_CATALOG)
    catalog["lang"] = lang
    if lang == "en":
        catalog["title"] = _EN_TEXT["title"]
        catalog["description"] = _EN_TEXT["description"]
        for step, text in zip(catalog["workflow"], _EN_TEXT["workflow"]):
            step.update(text)
        for category in catalog["categories"]:
            category["title"] = _EN_TEXT["categories"][category["id"]]
        for feature in catalog["features"]:
            text = _EN_TEXT["features"][feature["id"]]
            for field in ("title", "summary", "use_when", "steps", "notes"):
                feature[field] = deepcopy(text[field])
            for command, translated in zip(feature["commands"], text["commands"]):
                command.update(translated)
            feature["data_example"]["label"] = text["data_example_label"]
    for feature in catalog["features"]:
        feature["troubleshooting"] = deepcopy(_TROUBLESHOOTING[lang][feature["id"]])
    return catalog
