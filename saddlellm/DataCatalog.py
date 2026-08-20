"""
经典数据集目录 — 知道用什么数据, 怎么拉, 但不会自动下载 (省磁盘)

设计原则:
  1. 只注册元信息 (名称/大小/质量/适合什么模型)
  2. fetch() 时才真正拉取
  3. 优先 streaming (不落盘)
  4. 每个数据集标注推荐场景和配比

用法:
  from saddlellm import DataCatalog

  # 查看所有可用数据集
  DataCatalog.list_all()

  # 查看推荐给小模型的数据
  DataCatalog.recommend("small")

  # 拉取数据 (streaming, 不占磁盘)
  dataset = DataCatalog.fetch("wikitext", streaming=True)

  # 一键获取小模型训练数据包
  pack = DataCatalog.small_model_pack(lang="zh")
"""
import logging
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    """单个数据集的元信息"""
    name: str                        # 简称
    full_name: str                   # 全称
    category: str                    # "pretrain" | "instruct" | "code" | "math" | "chat" | "wiki"
    hf_path: str                     # HuggingFace 路径 (或 None)
    hf_config: Optional[str] = None  # HF config name
    hf_split: str = "train"
    text_column: str = "text"
    size_gb: float = 0               # 大概大小 (压缩后, GB)
    quality: str = "high"            # "high" | "medium" | "variable"
    language: str = "multi"          # "zh" | "en" | "multi"
    description: str = ""
    recommended_for: List[str] = field(default_factory=lambda: ["pretrain"])
    min_model_size: str = "100m"     # 至少多大模型推荐用
    streaming: bool = True           # 是否支持流式加载
    notes: str = ""


# ═══════════════════════════════════════════════════════════════
# 经典数据集注册表
# ═══════════════════════════════════════════════════════════════

CLASSIC_DATASETS: Dict[str, DatasetEntry] = {

    # ──── 英文预训练 ────
    "wikitext": DatasetEntry(
        name="wikitext",
        full_name="WikiText-103",
        category="pretrain",
        hf_path="wikitext",
        hf_config="wikitext-103-raw-v1",
        size_gb=0.5,
        quality="high",
        language="en",
        description="经典英文维基文本, 高质量基准数据集。适合所有模型大小。",
        recommended_for=["pretrain", "eval"],
        min_model_size="100m",
    ),
    "wikitext-2": DatasetEntry(
        name="wikitext-2",
        full_name="WikiText-2",
        category="pretrain",
        hf_path="wikitext",
        hf_config="wikitext-2-raw-v1",
        size_gb=0.01,
        quality="high",
        language="en",
        description="WikiText-2 更小, 适合快速验证。",
        recommended_for=["eval", "quick_test"],
        min_model_size="100m",
    ),
    "c4-en": DatasetEntry(
        name="c4-en",
        full_name="C4 (Colossal Clean Crawled Corpus)",
        category="pretrain",
        hf_path="c4",
        hf_config="en",
        size_gb=300,
        quality="medium",
        language="en",
        description="Common Crawl 清洗版, 大规模英文网页文本。1B+ 模型必用。",
        recommended_for=["pretrain"],
        min_model_size="300m",
        notes="文件很大, 强烈建议 streaming=True, 不要全量下载",
    ),
    "openwebtext": DatasetEntry(
        name="openwebtext",
        full_name="OpenWebText",
        category="pretrain",
        hf_path="openwebtext",
        size_gb=38,
        quality="high",
        language="en",
        description="GPT-2 训练数据的开源复现。Reddit 高质量链接。",
        recommended_for=["pretrain"],
        min_model_size="300m",
    ),
    "the-pile": DatasetEntry(
        name="the-pile",
        full_name="The Pile (EleutherAI)",
        category="pretrain",
        hf_path="EleutherAI/the_pile",
        size_gb=800,
        quality="variable",
        language="en",
        description="22 个子集的混合, 包括书籍/论文/代码/论坛。GPT-NeoX 训练集。",
        recommended_for=["pretrain"],
        min_model_size="1b",
        notes="子集质量不一, 建议筛选后使用。推荐子集: books3, github, pubmed, stackexchange",
    ),

    # ──── 中文预训练 ────
    "wikipedia-zh": DatasetEntry(
        name="wikipedia-zh",
        full_name="中文维基百科",
        category="wiki",
        hf_path="wikipedia",
        hf_config="20240301.zh",
        size_gb=1.5,
        quality="high",
        language="zh",
        description="中文维基百科, 高质量中文基准。中英混合模型必加。",
        recommended_for=["pretrain", "eval"],
        min_model_size="100m",
    ),
    "wikipedia-en": DatasetEntry(
        name="wikipedia-en",
        full_name="英文维基百科",
        category="wiki",
        hf_path="wikipedia",
        hf_config="20240301.en",
        size_gb=18,
        quality="high",
        language="en",
        description="英文维基百科。",
        recommended_for=["pretrain"],
        min_model_size="100m",
    ),
    "wuDaoCorpora": DatasetEntry(
        name="wuDaoCorpora",
        full_name="WuDaoCorpora 中文语料",
        category="pretrain",
        hf_path="BAAI/WuDaoCorporaText",
        size_gb=200,
        quality="variable",
        language="zh",
        description="悟道中文语料, 大规模中文预训练数据。",
        recommended_for=["pretrain"],
        min_model_size="1b",
        notes="数据量极大, 建议流式 + 质量过滤",
    ),
    "skywork-chinese": DatasetEntry(
        name="skywork-chinese",
        full_name="SkyPile-150B 中文预训练语料",
        category="pretrain",
        hf_path="Skywork/SkyPile-150B",
        size_gb=600,
        quality="medium",
        language="zh",
        description="昆仑万维开源的中文预训练数据, 150B tokens。",
        recommended_for=["pretrain"],
        min_model_size="1b",
    ),

    # ──── 代码 ────
    "the-stack": DatasetEntry(
        name="the-stack",
        full_name="The Stack (BigCode)",
        category="code",
        hf_path="bigcode/the-stack",
        size_gb=3000,
        quality="variable",
        language="multi",
        description="大规模代码数据集, 涵盖 30+ 编程语言。推荐子集: Python, Java, JS。",
        recommended_for=["pretrain", "code"],
        min_model_size="1b",
        notes="使用 data_dir 参数选语言",
    ),
    "starcoder": DatasetEntry(
        name="starcoder",
        full_name="StarCoder 训练数据",
        category="code",
        hf_path="bigcode/starcoderdata",
        size_gb=250,
        quality="high",
        language="multi",
        description="StarCoder 使用的清洗版代码数据。",
        recommended_for=["pretrain", "code"],
        min_model_size="300m",
    ),
    "codealpaca": DatasetEntry(
        name="codealpaca",
        full_name="Code Alpaca",
        category="instruct",
        hf_path="sahil2801/CodeAlpaca-20k",
        size_gb=0.01,
        quality="high",
        language="en",
        description="20K 代码指令数据。微调代码能力专用。",
        recommended_for=["sft", "code"],
        min_model_size="100m",
        text_column="instruction",  # 需要特殊处理
        notes="字段: instruction, input, output",
    ),

    # ──── 数学/推理 ────
    "gsm8k": DatasetEntry(
        name="gsm8k",
        full_name="Grade School Math 8K",
        category="math",
        hf_path="gsm8k",
        hf_config="main",
        size_gb=0.002,
        quality="high",
        language="en",
        description="小学数学应用题。评测 + SFT 微调数学能力。",
        recommended_for=["eval", "sft"],
        min_model_size="300m",
        text_column="question",
    ),
    "math": DatasetEntry(
        name="math",
        full_name="MATH (Hendrycks)",
        category="math",
        hf_path="hendrycks/math",
        size_gb=0.01,
        quality="high",
        language="en",
        description="竞赛级数学题。GRPO 对齐专用。",
        recommended_for=["align", "grpo"],
        min_model_size="1b",
        notes="有 5 个难度等级, 建议先用 level 1-2",
    ),

    # ──── 指令/微调 ────
    "alpaca": DatasetEntry(
        name="alpaca",
        full_name="Alpaca 52K",
        category="instruct",
        hf_path="tatsu-lab/alpaca",
        size_gb=0.02,
        quality="high",
        language="en",
        description="52K GPT-3.5 生成的指令数据。SFT 基准数据集。",
        recommended_for=["sft"],
        min_model_size="100m",
        text_column="instruction",  # + input + output
    ),
    "dolly": DatasetEntry(
        name="dolly",
        full_name="Databricks Dolly 15K",
        category="instruct",
        hf_path="databricks/databricks-dolly-15k",
        size_gb=0.005,
        quality="high",
        language="en",
        description="人工标注的高质量指令数据。",
        recommended_for=["sft"],
        min_model_size="100m",
    ),
    "sharegpt": DatasetEntry(
        name="sharegpt",
        full_name="ShareGPT 对话",
        category="chat",
        hf_path="anon8231489123/ShareGPT_Vicuna_unfiltered",
        size_gb=0.1,
        quality="high",
        language="multi",
        description="用户与 ChatGPT 的真实对话。SFT 对话能力。",
        recommended_for=["sft", "chat"],
        min_model_size="300m",
    ),
    "ultrachat": DatasetEntry(
        name="ultrachat",
        full_name="UltraChat 200K",
        category="chat",
        hf_path="HuggingFaceH4/ultrachat_200k",
        size_gb=0.3,
        quality="high",
        language="en",
        description="大规模合成对话数据。Zephyr 模型训练集。",
        recommended_for=["sft", "chat"],
        min_model_size="1b",
    ),

    # ──── 偏好对齐 (DPO/RLHF) ────
    "hh-rlhf": DatasetEntry(
        name="hh-rlhf",
        full_name="Anthropic HH-RLHF",
        category="chat",
        hf_path="Anthropic/hh-rlhf",
        size_gb=0.1,
        quality="high",
        language="en",
        description="Anthropic 的有用+无害偏好数据。DPO 训练专用。",
        recommended_for=["align", "dpo"],
        min_model_size="1b",
    ),
    "ultrafeedback": DatasetEntry(
        name="ultrafeedback",
        full_name="UltraFeedback",
        category="chat",
        hf_path="HuggingFaceH4/ultrafeedback_binarized",
        size_gb=0.5,
        quality="high",
        language="en",
        description="GPT-4 评分的偏好数据。Zephyr DPO 训练集。",
        recommended_for=["align", "dpo"],
        min_model_size="1b",
    ),

    # ──── 评测 ────
    "hellaswag": DatasetEntry(
        name="hellaswag",
        full_name="HellaSwag",
        category="eval",
        hf_path="Rowan/hellaswag",
        size_gb=0.05,
        quality="high",
        language="en",
        description="常识推理评测。",
        recommended_for=["eval"],
        min_model_size="100m",
    ),
    "mmlu": DatasetEntry(
        name="mmlu",
        full_name="MMLU (Massive Multitask Language Understanding)",
        category="eval",
        hf_path="cais/mmlu",
        hf_config="all",
        size_gb=0.01,
        quality="high",
        language="en",
        description="57 个学科的多选题评测。业界标准。",
        recommended_for=["eval"],
        min_model_size="1b",
    ),

    # ──── 小模型专用 (高质量, 体积小) ────
    "tiny-stories": DatasetEntry(
        name="tiny-stories",
        full_name="TinyStories",
        category="pretrain",
        hf_path="roneneldan/TinyStories",
        size_gb=0.3,
        quality="high",
        language="en",
        description="GPT-3.5/4 生成的短故事, 词汇简单、语法正确。小模型预训练神器!",
        recommended_for=["pretrain"],
        min_model_size="10m",
        notes="100M 以下模型的最佳训练数据",
    ),
    "tiny-textbooks": DatasetEntry(
        name="tiny-textbooks",
        full_name="TinyTextbooks",
        category="pretrain",
        hf_path="nampdn-ai/tiny-textbooks",
        size_gb=0.5,
        quality="high",
        language="en",
        description="合成的教科书文本。知识密度高, 适合小模型。",
        recommended_for=["pretrain"],
        min_model_size="100m",
    ),
    "cosmopedia": DatasetEntry(
        name="cosmopedia",
        full_name="Cosmopedia (HuggingFace)",
        category="pretrain",
        hf_path="HuggingFaceTB/cosmopedia",
        size_gb=25,
        quality="high",
        language="en",
        description="Mistral 生成的教科书级合成数据。30M-1B 模型最佳选择之一。",
        recommended_for=["pretrain"],
        min_model_size="100m",
        notes="合成数据, 质量极高, 知识密度大",
    ),
    "fineweb-edu": DatasetEntry(
        name="fineweb-edu",
        full_name="FineWeb-Edu",
        category="pretrain",
        hf_path="HuggingFaceFW/fineweb-edu",
        size_gb=2000,
        quality="high",
        language="en",
        description="高质量教育网页。经过质量分类器筛选。300M+ 模型推荐。",
        recommended_for=["pretrain"],
        min_model_size="300m",
        notes="size_gb 是总量, 可只取 sample-10BT 子集",
    ),
}


class DataCatalog:
    """
    经典数据集目录。

    用法:
        # 查看所有
        DataCatalog.list_all()

        # 按分类筛选
        DataCatalog.list_by_category("pretrain")

        # 推荐给小模型
        DataCatalog.recommend("small")  # "small" | "medium" | "large"

        # 获取数据 (streaming, 不落盘)
        ds = DataCatalog.fetch("wikitext", streaming=True)

        # 一键获取小模型训练包
        pack = DataCatalog.small_model_pack(lang="zh")
        # → 返回配好比的 DataMixer, 直接用于训练
    """

    @staticmethod
    def list_all(category: str = None) -> List[DatasetEntry]:
        """列出所有(或按分类筛选)数据集。"""
        datasets = list(CLASSIC_DATASETS.values())
        if category:
            datasets = [d for d in datasets if d.category == category]
        return sorted(datasets, key=lambda d: d.size_gb)

    @staticmethod
    def get(name: str) -> DatasetEntry:
        if name not in CLASSIC_DATASETS:
            similar = [k for k in CLASSIC_DATASETS if name.lower() in k.lower()]
            hint = f" 相似的: {similar}" if similar else ""
            raise KeyError(f"未知数据集: {name}.{hint}\n  用 DataCatalog.list_all() 查看全部")
        return CLASSIC_DATASETS[name]

    @staticmethod
    def recommend(model_size: str = "small") -> List[DatasetEntry]:
        """
        根据模型大小推荐数据集。

        model_size:
          "tiny"  → < 100M 模型
          "small" → 100M - 1B
          "medium"→ 1B - 7B
          "large" → 7B+
        """
        thresholds = {"tiny": "10m", "small": "100m", "medium": "1b", "large": "7b"}
        threshold = thresholds.get(model_size, "100m")

        order = ["10m", "100m", "300m", "1b", "7b"]
        max_idx = order.index(threshold)

        recommended = []
        for entry in CLASSIC_DATASETS.values():
            entry_idx = order.index(entry.min_model_size) if entry.min_model_size in order else 99
            if entry_idx <= max_idx:
                recommended.append(entry)

        return sorted(recommended, key=lambda d: (d.quality != "high", d.size_gb))

    @staticmethod
    def fetch(
        name: str,
        streaming: bool = True,
        split: str = None,
        max_samples: int = None,
    ):
        """
        拉取数据集 (默认 streaming, 不占磁盘)。

        streaming=True:  逐条读取, 不下载 → 省磁盘
        streaming=False: 下载到本地 (需要空间)
        """
        from datasets import load_dataset

        entry = DataCatalog.get(name)

        kwargs = {"path": entry.hf_path, "split": split or entry.hf_split}
        if entry.hf_config:
            kwargs["name"] = entry.hf_config

        if streaming:
            kwargs["streaming"] = True

        try:
            dataset = load_dataset(**kwargs)
        except Exception as e:
            # 回退: 尝试常见变体
            try:
                kwargs["streaming"] = True
                kwargs["split"] = "train"
                dataset = load_dataset(**kwargs)
            except Exception:
                raise RuntimeError(f"无法加载 {name}: {e}")

        if max_samples and streaming:
            return dataset.take(max_samples)
        elif max_samples:
            return dataset.select(range(min(max_samples, len(dataset))))

        return dataset

    @staticmethod
    def small_model_pack(
        lang: str = "zh",
        total_tokens_target: int = 2_000_000_000,  # 2B tokens, 适合 100M 模型
        streaming: bool = True,
    ):
        """
        一键获取小模型训练数据包。

        自动选择最适合小模型的高质量数据, 按推荐配比混合。

        lang="zh":  中文为主
        lang="en":  英文为主
        lang="mix": 中英混合

        返回: 可直接用于训练的 dataset
        """
        logger.info(f"=== 小模型数据包 (target: {total_tokens_target/1e9:.1f}B tokens) ===")

        if lang == "zh":
            recipe = [
                ("wikipedia-zh", 0.35),
                ("wikipedia-en", 0.20),
                ("tiny-stories", 0.20),
                ("cosmopedia", 0.15),
                ("codealpaca", 0.10),
            ]
        elif lang == "en":
            recipe = [
                ("tiny-stories", 0.25),
                ("cosmopedia", 0.25),
                ("wikitext", 0.20),
                ("wikipedia-en", 0.15),
                ("starcoder", 0.15),
            ]
        else:
            recipe = [
                ("wikipedia-zh", 0.20),
                ("wikipedia-en", 0.20),
                ("tiny-stories", 0.15),
                ("cosmopedia", 0.15),
                ("wikitext", 0.15),
                ("starcoder", 0.15),
            ]

        from datasets import interleave_datasets

        all_ds = []
        all_weights = []
        total_loaded = 0

        for name, weight in recipe:
            try:
                entry = CLASSIC_DATASETS[name]
                ds = DataCatalog.fetch(name, streaming=streaming)
                all_ds.append(ds)
                all_weights.append(weight)
                total_loaded += 1
                logger.info(f"  ✓ {name}: {entry.description[:60]}")
            except Exception as e:
                logger.warning(f"  ✗ {name}: 加载失败 ({e}), 跳过")
                continue

        if not all_ds:
            raise RuntimeError("所有数据源加载失败。请检查网络连接。")

        # 归一化
        total_w = sum(all_weights)
        all_weights = [w / total_w for w in all_weights]

        mixed = interleave_datasets(all_ds, probabilities=all_weights, seed=42)
        logger.info(f"  混合完成: {total_loaded} 个数据源, 配比={dict(zip([r[0] for r in recipe[:total_loaded]], [f'{w:.0%}' for w in all_weights]))}")

        return mixed

    @staticmethod
    def print_catalog():
        """打印完整数据目录。"""
        for cat in ["pretrain", "instruct", "code", "math", "chat", "eval"]:
            entries = DataCatalog.list_all(category=cat)
            if not entries:
                continue
            print(f"\n{'='*60}")
            print(f"  {cat.upper()} ({len(entries)} datasets)")
            print(f"{'='*60}")
            for e in entries:
                size_str = f"{e.size_gb:.1f}GB" if e.size_gb >= 1 else f"{e.size_gb*1000:.0f}MB"
                print(f"  {e.name:<22} {size_str:>8}  {e.quality:<8} [{e.language}]")
                print(f"    {e.description}")
                if e.notes:
                    print(f"    ⚠ {e.notes}")


# ============================================================
# Data Quality Classifier: 自动评估文本质量
# ============================================================

class DataQualityClassifier:
    """
    自动文本质量分类 — 对原始文本打分, 过滤低质量。

    评分维度 (0-1, 越高越好):
      - 长度合理性: 太短(<50字)或太长(>10000字)扣分
      - 结构完整性: 有标点、分段、句子完整
      - 语言一致性: 不中英混杂
      - 信息密度:  不是重复灌水
      - 代码/噪声:  过滤纯代码或乱码

    用法:
        qc = DataQualityClassifier()

        score = qc.score("这是一段高质量的中文文本...")
        # → 0.85

        # 过滤数据
        good = [t for t in texts if qc.score(t) > 0.5]
    """

    def __init__(self, lang: str = "auto", min_score: float = 0.5):
        self.lang = lang
        self.min_score = min_score

    def score(self, text: str) -> float:
        """对单条文本打分 (0-1)。"""
        if not isinstance(text, str) or not text.strip():
            return 0.0

        scores = {}

        # 1. 长度分 (0-0.3)
        n = len(text)
        if n < 20:
            scores["length"] = 0.0
        elif n < 50:
            scores["length"] = 0.1
        elif n < 100:
            scores["length"] = 0.2
        elif n < 10000:
            scores["length"] = 0.3
        else:
            scores["length"] = 0.25  # 太长稍扣

        # 2. 结构分 (0-0.25)
        import re
        struct = 0.0
        if any(p in text for p in [".", "。", "!", "！", "?", "？"]):
            struct += 0.1  # 有句末标点
        if "\n" in text:
            struct += 0.05  # 有分段
        if len(re.findall(r"[a-zA-Z]", text)) / max(n, 1) < 0.6:
            struct += 0.05  # 不是全英文乱码
        if len(text.split()) >= 10 or len(text) >= 50:
            struct += 0.05  # 有一定长度
        scores["structure"] = min(0.25, struct)

        # 3. 语言一致性 (0-0.15)
        letters = len(re.findall(r"[a-zA-Z]", text))
        cjk = len(re.findall(r"[一-鿿]", text))
        total_chars = max(letters + cjk, 1)
        mix_ratio = min(letters, cjk) / total_chars
        if mix_ratio < 0.1:
            scores["lang_consistency"] = 0.15  # 纯中文或纯英文
        elif mix_ratio < 0.3:
            scores["lang_consistency"] = 0.1
        else:
            scores["lang_consistency"] = 0.05

        # 4. 信息密度 (0-0.15)
        lines = text.split("\n")
        unique_lines = len(set(lines))
        if len(lines) > 1:
            repeat_ratio = 1 - (unique_lines / len(lines))
            if repeat_ratio < 0.1:
                scores["density"] = 0.15
            elif repeat_ratio < 0.3:
                scores["density"] = 0.1
            else:
                scores["density"] = 0.0  # 重复太多
        else:
            scores["density"] = 0.1

        # 5. 噪声分 (0-0.15) — 纯 URL/代码/乱码扣分
        url_ratio = len(re.findall(r"https?://", text)) / max(n, 1) * 100
        special_ratio = len(re.findall(r"[^\w\s一-鿿.,!?，。！？\"\"''：:;；()（）、。]", text)) / max(n, 1)
        if url_ratio > 5 or special_ratio > 0.3:
            scores["noise"] = 0.0
        elif url_ratio > 2 or special_ratio > 0.15:
            scores["noise"] = 0.08
        else:
            scores["noise"] = 0.15

        return round(sum(scores.values()), 3)

    def is_quality(self, text: str) -> bool:
        return self.score(text) >= self.min_score

    def filter(self, texts: List[str]) -> List[str]:
        """过滤低质量文本。"""
        return [t for t in texts if self.score(t) >= self.min_score]

    def filter_dataset(self, dataset, text_column: str = "text", batch_size: int = 1000):
        """过滤 HuggingFace dataset。"""
        def _filter(example):
            text = example.get(text_column, "")
            return self.is_quality(text)

        return dataset.filter(_filter)

