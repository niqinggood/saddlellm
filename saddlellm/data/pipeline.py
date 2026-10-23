"""
数据管线 - 预训练数据的端到端处理
收集 → 清洗 → 去重 → 质量过滤 → 分片 → 分词 → 打包
"""

import os
import logging
import hashlib
import math
import re
import unicodedata
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field, replace
from collections import Counter

logger = logging.getLogger(__name__)

__all__ = [
    "CausalLMDataCollator",
    "DataMixConfig",
    "DataMixer",
    "DataPipeline",
    "PipelineConfig",
]


@dataclass
class PipelineConfig:
    sources: List[Dict] = field(default_factory=list)
    source_weights: Optional[List[float]] = None  # 各数据源配比权重, 如 [0.4,0.3,0.3]
    mix_strategy: str = "interleave"  # interleave | oversample | stratified
    oversample_factor: float = 3.0  # oversample 策略的放大因子
    output_dir: str = "./processed_data"
    max_seq_length: int = 2048
    min_text_length: int = 50
    dedup_method: str = "simhash"
    dedup_threshold: float = 0.8
    quality_min_score: float = 0.3
    lang_filter: Optional[str] = None
    shuffle_seed: int = 42
    split_ratios: List[float] = field(default_factory=lambda: [0.98, 0.02])
    num_proc: int = 4
    streaming: bool = True
    cache_dir: Optional[str] = None
    tokenizer_path: Optional[str] = None
    pack_sequences: bool = True
    append_eos: bool = True
    remove_personal_info: bool = True
    filter_urls: bool = True
    filter_code_comments: bool = False


class DataPipeline:
    """
    预训练数据管线。

    用法:
        pipeline = DataPipeline(PipelineConfig(sources=[...]))
        pipeline.collect().clean().deduplicate().filter_quality()
        pipeline.tokenize_and_pack(tokenizer)
        dataset = pipeline.to_iterable_dataset()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._dataset = None
        self._stats = Counter()
        self._validate_config()

    def _validate_config(self):
        if self.config.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if self.config.min_text_length < 0:
            raise ValueError("min_text_length must be non-negative")
        if self.config.num_proc < 1:
            raise ValueError("num_proc must be at least 1")
        if self.config.oversample_factor <= 0:
            raise ValueError("oversample_factor must be positive")
        if self.config.dedup_method not in {"simhash", "minhash", "exact", "none"}:
            raise ValueError(
                "dedup_method must be one of: simhash, minhash, exact, none"
            )
        if not 0 <= self.config.dedup_threshold <= 1:
            raise ValueError("dedup_threshold must be in the range [0, 1]")
        if not 0 <= self.config.quality_min_score <= 1:
            raise ValueError("quality_min_score must be in the range [0, 1]")
        if self.config.mix_strategy not in {"interleave", "oversample", "stratified"}:
            raise ValueError(
                "mix_strategy must be one of: interleave, oversample, stratified"
            )

        ratios = self.config.split_ratios
        if (
            len(ratios) != 2
            or any(not math.isfinite(value) or value <= 0 for value in ratios)
            or not math.isclose(sum(ratios), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError(
                "split_ratios must contain two positive values that sum to 1"
            )

        weights = self.config.source_weights
        if weights is not None:
            if len(weights) != len(self.config.sources):
                raise ValueError("source_weights length must match sources length")
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError(
                    "source_weights must contain finite non-negative values"
                )
            if sum(weights) <= 0:
                raise ValueError("source_weights must have a positive total")

    def _ensure_dataset(self):
        if self._dataset is None:
            raise RuntimeError("Dataset is empty. Call collect() before this step.")

    @staticmethod
    def dataset_size(dataset) -> Optional[int]:
        """Return materialized cardinality without consuming a stream."""
        num_rows = getattr(dataset, "num_rows", None)
        if isinstance(num_rows, int):
            return num_rows
        try:
            return len(dataset)
        except (TypeError, AttributeError, NotImplementedError):
            return None

    def cardinality(self) -> Optional[int]:
        self._ensure_dataset()
        return self.dataset_size(self._dataset)

    def _record_cardinality(self, name: str) -> None:
        size = self.cardinality()
        self._stats[name] = -1 if size is None else size

    def _stateless_transform_kwargs(self, dataset=None) -> Dict[str, int]:
        """Enable HF map-style multiprocessing for stateless transforms only.

        ``datasets.IterableDataset`` does not accept ``num_proc`` and arbitrary
        map/filter test doubles need not implement the full Hugging Face API.
        Requiring a real ``datasets.Dataset`` keeps both cases on their serial,
        backwards-compatible path.  Stateful deduplication deliberately does
        not use this helper because worker-local ``seen`` sets would make its
        result incorrect.
        """
        if self.config.num_proc <= 1:
            return {}
        dataset = self._dataset if dataset is None else dataset
        try:
            from datasets import Dataset
        except (ImportError, AttributeError):
            return {}
        try:
            from datasets import IterableDataset
        except (ImportError, AttributeError):
            IterableDataset = ()

        try:
            is_map_style = isinstance(dataset, Dataset)
            is_streaming = isinstance(dataset, IterableDataset)
        except TypeError:
            # Some lightweight tests expose factories rather than dataset
            # classes.  They should retain the legacy serial call signature.
            return {}
        if not is_map_style or is_streaming:
            return {}
        return {"num_proc": self.config.num_proc}

    def collect(self) -> "DataPipeline":
        """从多个数据源收集文本"""
        from datasets import load_dataset

        all_splits = []
        for source in self.config.sources:
            src_type = source.get("type", "local")
            src_path = source.get("path", "")
            src_split = source.get("split", "train")
            src_name = source.get("name", None)
            streaming = source.get("streaming", self.config.streaming)

            logger.info(f"收集数据源: type={src_type}, path={src_path}")

            if src_type == "huggingface":
                ds = load_dataset(
                    src_path,
                    src_name,
                    split=src_split,
                    streaming=streaming,
                    cache_dir=self.config.cache_dir,
                )
                col = source.get("text_column", "text")
                if col != "text":
                    ds = ds.rename_column(col, "text")
                all_splits.append(ds)

            elif src_type == "local":
                path = src_path
                fmt = source.get("format", "auto")
                text_col = source.get("text_column", "text")

                import glob

                if any(ch in path for ch in "*?[]"):
                    files = sorted(glob.glob(path, recursive=True))
                elif os.path.isfile(path):
                    files = [path]
                elif os.path.isdir(path):
                    pattern = source.get("pattern", "**/*.jsonl")
                    files = sorted(
                        glob.glob(os.path.join(path, pattern), recursive=True)
                    )
                else:
                    files = []

                if not files:
                    raise FileNotFoundError(
                        f"Local data source not found or empty: {path}"
                    )

                fmt_f = self._detect_format(files[0]) if fmt == "auto" else fmt
                ds = load_dataset(
                    fmt_f,
                    data_files=files,
                    split="train",
                    streaming=streaming,
                    cache_dir=self.config.cache_dir,
                )
                if text_col != "text":
                    ds = ds.rename_column(text_col, "text")
                all_splits.append(ds)

            elif src_type == "wikitext":
                ds = load_dataset(
                    "wikitext",
                    "wikitext-103-raw-v1",
                    split=src_split,
                    streaming=streaming,
                    cache_dir=self.config.cache_dir,
                )
                all_splits.append(ds)

            elif src_type == "wikipedia":
                lang = source.get("lang", "zh")
                date = source.get("date", "20240301")
                ds = load_dataset(
                    "wikipedia",
                    f"{date}.{lang}",
                    split=src_split,
                    streaming=streaming,
                    cache_dir=self.config.cache_dir,
                )
                all_splits.append(ds)

            elif src_type == "c4":
                lang = source.get("lang", "en")
                ds = load_dataset(
                    "c4",
                    lang,
                    split=src_split,
                    streaming=streaming,
                    cache_dir=self.config.cache_dir,
                )
                all_splits.append(ds)

            else:
                raise ValueError(f"unsupported data source type: {src_type}")

            self._stats["source_files"] += 1

        if not all_splits:
            raise ValueError("没有收集到任何数据,请检查 sources 配置")

        from datasets import interleave_datasets

        if len(all_splits) == 1:
            self._dataset = all_splits[0]
        else:
            weights = self.config.source_weights
            if weights:
                total = sum(weights)
                weights = [w / total for w in weights] if total > 0 else None
            self._dataset = interleave_datasets(
                all_splits,
                probabilities=weights,
                seed=self.config.shuffle_seed,
                stopping_strategy="first_exhausted",
            )
        self._record_cardinality("raw_samples")

        logger.info(f"数据收集完成, 数据源数: {len(all_splits)}")
        return self

    def clean(self) -> "DataPipeline":
        """清洗文本: 去除 HTML/URL/特殊字符,利用 TextProcess.py"""
        self._ensure_dataset()

        cleaner = None
        try:
            from .text_processing import TextCleaner

            cleaner = TextCleaner(
                {
                    "fix_encoding_errors": True,
                    "remove_html_tags": True,
                    "remove_js_code": True,
                    "remove_css_code": True,
                    "remove_urls": self.config.filter_urls,
                    "remove_emails": self.config.remove_personal_info,
                    "remove_phone_numbers": self.config.remove_personal_info,
                    "remove_special_chars": "keep",
                    "normalize_unicode": "NFC",
                    "min_length": self.config.min_text_length,
                }
            )
        except Exception as exc:
            logger.warning(
                "TextCleaner unavailable, using lightweight cleaner: %s", exc
            )

        def _clean(example):
            text = example.get("text", "")
            if not text or not isinstance(text, str):
                return {"text": "", "valid": False}
            text = cleaner.clean_text(text) if cleaner else self._basic_clean_text(text)
            return {
                "text": text,
                "valid": bool(text) and len(text) >= self.config.min_text_length,
            }

        transform_kwargs = self._stateless_transform_kwargs()
        self._dataset = self._dataset.map(_clean, **transform_kwargs)
        self._dataset = self._dataset.filter(
            lambda x: x.get("valid", False), **transform_kwargs
        )
        if "valid" in (getattr(self._dataset, "column_names", None) or []):
            self._dataset = self._dataset.remove_columns(["valid"])
        self._record_cardinality("clean_samples")
        return self

    def _basic_clean_text(self, text: str) -> str:
        text = re.sub(
            r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", text, flags=re.I
        )
        text = re.sub(
            r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.I
        )
        text = re.sub(r"<[^>]+>", " ", text)
        if self.config.filter_urls:
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        if self.config.remove_personal_info:
            text = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text
            )
            text = re.sub(
                r"\b(?:\+?\d{1,3})?[-. (]*\d{3}[-. )]*\d{3}[-. ]*\d{4}\b", " ", text
            )
        return re.sub(r"\s+", " ", text).strip()

    def deduplicate(
        self, method: Optional[str] = None, threshold: Optional[float] = None
    ) -> "DataPipeline":
        """文本去重: simhash / minhash / exact"""
        self._ensure_dataset()
        method = method or self.config.dedup_method
        threshold = self.config.dedup_threshold if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("deduplication threshold must be in the range [0, 1]")

        self._stats["dedup_requested_method"] = method
        self._stats["dedup_effective_method"] = method
        self._stats.pop("dedup_fallback_reason", None)
        self._stats.pop("dedup_normalization", None)

        if method == "none":
            return self
        elif method == "simhash":
            self._dedup_simhash(threshold)
        elif method == "minhash":
            self._dedup_minhash(threshold)
        elif method == "exact":
            self._dedup_exact()
        else:
            raise ValueError(f"不支持的查重方法: {method}")

        self._record_cardinality("deduplicated_samples")
        return self

    def _dedup_simhash(self, threshold):
        try:
            from simhash import Simhash
        except ImportError:
            logger.warning("simhash 未安装, SimHash 去重降级为 exact 去重")
            self._stats["dedup_effective_method"] = "exact"
            self._stats["dedup_fallback_reason"] = "simhash is not installed"
            self._dedup_exact()
            return

        seen = set()

        def _is_dup(text):
            if not text or len(text) < 50:
                return False
            sig = Simhash(text).value
            # 检查是否有相近的 hash
            for s in seen:
                if bin(sig ^ s).count("1") <= (1 - threshold) * 64:
                    return True
            seen.add(sig)
            if len(seen) > 1000000:
                seen.clear()
            return False

        self._dataset = self._dataset.filter(lambda x: not _is_dup(x.get("text", "")))
        logger.info(f"Simhash 去重完成, 保留指纹数: {min(len(seen), 1000000)}")

    def _dedup_minhash(self, threshold):
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch 未安装, MinHash 去重降级为 exact 去重")
            self._stats["dedup_effective_method"] = "exact"
            self._stats["dedup_fallback_reason"] = "datasketch is not installed"
            self._dedup_exact()
            return

        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        seen_count = 0

        def _tokens(text: str):
            text = text.lower()
            if re.search(r"[\u4e00-\u9fff]", text):
                return {text[i : i + 3] for i in range(max(1, len(text) - 2))}
            return set(re.findall(r"\w+", text))

        def _signature(text: str):
            mh = MinHash(num_perm=128)
            for token in _tokens(text):
                mh.update(token.encode("utf-8", errors="ignore"))
            return mh

        def _is_dup(example):
            nonlocal seen_count
            text = example.get("text", "")
            if not text or len(text) < self.config.min_text_length:
                return False
            mh = _signature(text)
            if lsh.query(mh):
                return True
            key = f"m{seen_count}"
            lsh.insert(key, mh)
            seen_count += 1
            return False

        self._dataset = self._dataset.filter(lambda x: not _is_dup(x))
        logger.info("MinHash 去重完成, 保留签名数: %s", seen_count)

    def _dedup_exact(self):
        """Hash the complete text after NFC, whitespace and case normalization."""
        seen = set()
        self._stats["dedup_normalization"] = "NFC + collapse whitespace + casefold"

        def _is_dup(text):
            normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().casefold()
            key = hashlib.sha256(normalized.encode("utf-8")).digest()
            if key in seen:
                return True
            seen.add(key)
            return False

        self._dataset = self._dataset.filter(lambda x: not _is_dup(x.get("text", "")))

    def filter_quality(
        self, min_length: Optional[int] = None, lang: Optional[str] = None
    ) -> "DataPipeline":
        """
        质量过滤:
        - 最小文本长度
        - 语言检测
        - 困惑度/质量评分
        - URL/邮箱/手机号过滤
        """
        self._ensure_dataset()
        min_len = self.config.min_text_length if min_length is None else min_length
        if min_len < 0:
            raise ValueError("min_length must be non-negative")
        lang_filter = self.config.lang_filter if lang is None else lang
        quality_classifier = None
        if self.config.quality_min_score and self.config.quality_min_score > 0:
            try:
                from .DataCatalog import DataQualityClassifier

                quality_classifier = DataQualityClassifier(
                    lang=lang_filter or "auto", min_score=self.config.quality_min_score
                )
            except Exception as exc:
                logger.warning(
                    "DataQualityClassifier unavailable, using heuristic filter only: %s",
                    exc,
                )

        def _filter(example):
            text = example.get("text", "")
            if not text or len(text) < min_len:
                return False

            # 统计信息
            chars = len(text)
            has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
            if not has_cjk:
                words = len(text.split())
                if words == 0:
                    return False
                avg_word_len = chars / words

            # 过滤低质量: 太短的词平均长度, 过多的特殊字符
            if not has_cjk and (avg_word_len > 30 or avg_word_len < 1.5):
                return False

            # 标点符号比例
            punct_ratio = sum(
                1 for c in text if c in "，。！？；：、''（）《》【】…—·,.!?;:()[]{}"
            )
            if punct_ratio / max(chars, 1) > 0.3:
                return False

            # 大写字母比例
            upper_ratio = sum(1 for c in text if c.isupper()) / max(chars, 1)
            if upper_ratio > 0.5 and chars > 100:
                return False

            # 语言过滤
            if lang_filter:
                try:
                    from langdetect import detect

                    detected = detect(text[:200])
                    if detected != lang_filter:
                        return False
                except Exception:
                    pass

            # URL/邮箱/手机号过滤
            if self.config.filter_urls:
                if (
                    re.search(r"https?://\S+", text)
                    and len(re.findall(r"https?://\S+", text)) > 5
                ):
                    return False

            if quality_classifier and not quality_classifier.is_quality(text):
                return False

            return True

        self._dataset = self._dataset.filter(
            _filter, **self._stateless_transform_kwargs()
        )
        self._record_cardinality("quality_samples")
        logger.info("质量过滤完成")
        return self

    def tokenize_and_pack(
        self,
        tokenizer,
        text_column: str = "text",
        max_seq_length: Optional[int] = None,
    ) -> "DataPipeline":
        """分词并打包为固定长度序列"""
        self._ensure_dataset()
        max_len = (
            self.config.max_seq_length if max_seq_length is None else max_seq_length
        )
        if max_len <= 0:
            raise ValueError("max_seq_length must be positive")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        def _tokenize(examples):
            # Packed pretraining must keep every token.  Truncating each source
            # document here silently discards all content after max_len before
            # the packer ever sees it.
            tokenize_kwargs = {
                "truncation": not self.config.pack_sequences,
                "padding": False,
                "return_attention_mask": False,
            }
            if not self.config.pack_sequences:
                reserve_eos = bool(self.config.append_eos and eos_token_id is not None)
                tokenize_kwargs["max_length"] = max(1, max_len - int(reserve_eos))
            encoded = tokenizer(examples[text_column], **tokenize_kwargs)
            input_ids = []
            for raw_ids in encoded.get("input_ids", []):
                ids = list(raw_ids)
                if (
                    self.config.append_eos
                    and eos_token_id is not None
                    and (not ids or ids[-1] != eos_token_id)
                ):
                    ids.append(eos_token_id)
                if not self.config.pack_sequences:
                    ids = ids[:max_len]
                input_ids.append(ids)
            return {"input_ids": input_ids}

        columns = getattr(self._dataset, "column_names", None) or []
        self._dataset = self._dataset.map(
            _tokenize,
            batched=True,
            remove_columns=[c for c in columns if c != "input_ids"],
            **self._stateless_transform_kwargs(),
        )

        if self.config.pack_sequences:
            self._dataset = self._pack_sequences(self._dataset, max_len, pad_token_id)
        else:
            def _add_causal_fields(example):
                ids = list(example.get("input_ids", []))
                return {
                    "input_ids": ids,
                    "labels": list(ids),
                    "attention_mask": [1] * len(ids),
                }

            self._dataset = self._dataset.map(
                _add_causal_fields, **self._stateless_transform_kwargs()
            )

        self._record_cardinality("training_sequences")
        return self

    def _pack_sequences(self, dataset, max_len: int, pad_token_id: int = 0):
        """将多个短序列拼接打包到 max_len"""

        def _pack_generator():
            # Keep generator state local so retries/fingerprinting do not reuse
            # leftover tokens from an earlier iteration.
            token_buffer = []
            label_buffer = []
            document_buffer = []
            position_buffer = []
            for document_id, example in enumerate(dataset):
                ids = list(example.get("input_ids", []))
                offset = 0
                while offset < len(ids):
                    available = max_len - len(token_buffer)
                    take = min(available, len(ids) - offset)
                    piece = ids[offset : offset + take]
                    piece_labels = list(piece)

                    # Labels are shifted by the model. Mask the first token of
                    # every later document so the preceding document's final
                    # token does not learn an artificial cross-document target.
                    # A document-ending EOS itself remains a training target.
                    if offset == 0 and token_buffer:
                        piece_labels[0] = -100

                    token_buffer.extend(piece)
                    label_buffer.extend(piece_labels)
                    document_buffer.extend([document_id] * take)
                    # A document fragment crossing a training block starts a
                    # fresh context in the next block, including RoPE position.
                    position_buffer.extend(range(take))
                    offset += take

                    if len(token_buffer) != max_len:
                        continue
                    yield {
                        "input_ids": token_buffer,
                        "labels": label_buffer,
                        "attention_mask": [1] * max_len,
                        "document_ids": document_buffer,
                        "position_ids": position_buffer,
                    }
                    token_buffer = []
                    label_buffer = []
                    document_buffer = []
                    position_buffer = []
            if len(token_buffer) >= 10:
                original_len = len(token_buffer)
                padding = max_len - original_len
                yield {
                    "input_ids": token_buffer + [pad_token_id] * padding,
                    "labels": label_buffer + [-100] * padding,
                    "attention_mask": [1] * original_len + [0] * padding,
                    "document_ids": document_buffer + [-1] * padding,
                    "position_ids": position_buffer + [0] * padding,
                }

        # Keep an IterableDataset iterable.  Materializing a streaming corpus
        # here defeats --streaming and can exhaust memory before training.
        from datasets import Dataset

        try:
            from datasets import IterableDataset
        except ImportError:  # Small test doubles and old datasets versions.
            IterableDataset = ()
        if isinstance(dataset, IterableDataset):
            return IterableDataset.from_generator(_pack_generator)
        generator_kwargs = {}
        if self.config.cache_dir:
            generator_kwargs["cache_dir"] = self.config.cache_dir
        return Dataset.from_generator(_pack_generator, **generator_kwargs)

    def to_iterable_dataset(self):
        """转换为 HuggingFace IterableDataset (用于流式训练)"""
        self._ensure_dataset()
        return self._dataset

    def save_to_disk(self, path: Optional[str] = None):
        """保存处理后的数据到磁盘"""
        self._ensure_dataset()
        output = path or self.config.output_dir
        os.makedirs(output, exist_ok=True)

        if hasattr(self._dataset, "save_to_disk"):
            self._dataset.save_to_disk(output)
        else:
            # fallback: 生成器写入 parquet
            import pyarrow.parquet as pq
            import pyarrow as pa

            batch_size = 10000
            batch = []
            for i, example in enumerate(self._dataset):
                batch.append(example)
                if len(batch) >= batch_size:
                    pq.write_table(
                        pa.Table.from_pylist(batch),
                        os.path.join(output, f"part-{i // batch_size:05d}.parquet"),
                    )
                    batch = []
            if batch:
                pq.write_table(
                    pa.Table.from_pylist(batch),
                    os.path.join(output, "part-final.parquet"),
                )

        logger.info(f"数据已保存到 {output}")
        return output

    def split(self, ratios: Optional[List[float]] = None, seed: Optional[int] = None):
        """划分训练/验证集"""
        self._ensure_dataset()
        ratios = self.config.split_ratios if ratios is None else ratios
        seed = self.config.shuffle_seed if seed is None else seed
        if (
            len(ratios) != 2
            or any(not math.isfinite(value) or value <= 0 for value in ratios)
            or sum(ratios) <= 0
        ):
            raise ValueError("ratios must contain two positive values")

        validation_fraction = ratios[1] / sum(ratios)
        if hasattr(self._dataset, "train_test_split"):
            split_dataset = self._dataset.train_test_split(
                test_size=validation_fraction, seed=seed, shuffle=True
            )
            return {"train": split_dataset["train"], "val": split_dataset["test"]}

        if not hasattr(self._dataset, "filter"):
            raise TypeError("streaming dataset must provide filter() for a leakage-free split")

        # IterableDataset has no random-access train_test_split.  Partition by
        # a stable content hash instead of returning the same stream as both
        # train and validation, which would make validation loss meaningless.
        boundary = int(validation_fraction * (1 << 64))

        def _is_validation(example: Dict[str, Any]) -> bool:
            text = str(example.get("text", ""))
            digest = hashlib.sha1(
                f"{seed}\0{text}".encode("utf-8", errors="ignore")
            ).digest()
            return int.from_bytes(digest[:8], "big") < boundary

        return {
            "train": self._dataset.filter(lambda example: not _is_validation(example)),
            "val": self._dataset.filter(_is_validation),
        }

    @property
    def stats(self) -> Dict:
        return dict(self._stats)

    def _detect_format(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jsonl",):
            return "json"
        if ext in (".json",):
            return "json"
        if ext in (".parquet",):
            return "parquet"
        if ext in (".txt", ".md"):
            return "text"
        if os.path.isdir(path):
            return "text"
        return "json"


@dataclass
class CausalLMDataCollator:
    """Pad causal-LM batches without masking real EOS tokens.

    Hugging Face's generic language-modeling collator masks every token whose
    id equals ``pad_token_id``.  Many decoder-only tokenizers intentionally use
    EOS as PAD, so that behaviour also masks genuine document-ending EOS tokens.
    This collator masks only padding positions introduced for this batch.
    """

    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    omit_all_one_attention_mask: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("features must not be empty")
        import torch

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

        max_length = max(len(feature.get("input_ids", [])) for feature in features)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if multiple <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            max_length = int(math.ceil(max_length / multiple) * multiple)
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        has_document_ids = any("document_ids" in feature for feature in features)
        # Document boundaries imply reset positions even when a custom dataset
        # supplies only document_ids.
        has_position_ids = has_document_ids or any(
            "position_ids" in feature for feature in features
        )
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        if has_document_ids:
            batch["document_ids"] = []
        if has_position_ids:
            batch["position_ids"] = []
        mask_has_padding = False
        has_document_boundaries = False
        has_nonstandard_positions = False
        for feature in features:
            input_ids = list(feature.get("input_ids", []))
            labels = list(feature.get("labels", input_ids))
            attention_mask = list(
                feature.get("attention_mask", [1] * len(input_ids))
            )
            if len(labels) != len(input_ids) or len(attention_mask) != len(input_ids):
                raise ValueError("input_ids, labels, and attention_mask lengths must match")
            document_ids = None
            if has_document_ids:
                document_ids = list(
                    feature.get("document_ids", [0] * len(input_ids))
                )
                if len(document_ids) != len(input_ids):
                    raise ValueError("document_ids and input_ids lengths must match")
            position_ids = None
            if has_position_ids:
                if "position_ids" in feature:
                    position_ids = list(feature["position_ids"])
                elif document_ids is not None:
                    position_ids = []
                    previous_document = object()
                    offset = 0
                    for document_id in document_ids:
                        if document_id != previous_document:
                            offset = 0
                        position_ids.append(offset)
                        offset += 1
                        previous_document = document_id
                else:
                    position_ids = list(range(len(input_ids)))
                if len(position_ids) != len(input_ids):
                    raise ValueError("position_ids and input_ids lengths must match")

            valid_indices = [
                index for index, value in enumerate(attention_mask) if value != 0
            ]
            if document_ids is not None:
                valid_documents = [document_ids[index] for index in valid_indices]
                has_document_boundaries = has_document_boundaries or any(
                    current != previous
                    for previous, current in zip(
                        valid_documents, valid_documents[1:]
                    )
                )
            if position_ids is not None:
                valid_positions = [position_ids[index] for index in valid_indices]
                has_nonstandard_positions = (
                    has_nonstandard_positions
                    or valid_positions != list(range(len(valid_positions)))
                )
            padding = max_length - len(input_ids)
            mask_has_padding = (
                mask_has_padding
                or padding > 0
                or any(value != 1 for value in attention_mask)
            )
            if padding_side == "left":
                batch["input_ids"].append([pad_token_id] * padding + input_ids)
                batch["labels"].append([-100] * padding + labels)
                batch["attention_mask"].append([0] * padding + attention_mask)
                if document_ids is not None:
                    batch["document_ids"].append([-1] * padding + document_ids)
                if position_ids is not None:
                    batch["position_ids"].append([0] * padding + position_ids)
            else:
                batch["input_ids"].append(input_ids + [pad_token_id] * padding)
                batch["labels"].append(labels + [-100] * padding)
                batch["attention_mask"].append(attention_mask + [0] * padding)
                if document_ids is not None:
                    batch["document_ids"].append(document_ids + [-1] * padding)
                if position_ids is not None:
                    batch["position_ids"].append(position_ids + [0] * padding)

        if self.omit_all_one_attention_mask and not mask_has_padding:
            # The native model can use SDPA's causal fast path directly.  Do
            # this on CPU while collating instead of asking every rank to
            # inspect an all-one CUDA tensor during each forward pass.
            batch.pop("attention_mask")
        if has_document_ids and not has_document_boundaries:
            # A single document per row needs no block-diagonal mask. Keeping
            # this metadata would disable SDPA's causal fast path for no gain.
            batch.pop("document_ids")
        if (
            has_position_ids
            and not has_document_boundaries
            and not has_nonstandard_positions
        ):
            batch.pop("position_ids")
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


# ============================================================
# DataMixer: 多数据源加权混合
# ============================================================


@dataclass
class DataMixConfig:
    """单个数据源的配比配置"""

    name: str  # 数据源名称
    weight: float  # 配比权重 (所有源权重之和归一化)
    sources: List[Dict] = field(default_factory=list)
    max_seq_length_override: Optional[int] = None
    num_repetitions: int = 1  # 小数据集重复次数
    quality_min_score: float = 0.3
    lang_filter: Optional[str] = None
    description: str = ""


class DataMixer:
    """
    多源数据加权混合器。

    用法:
        mixer = DataMixer([
            DataMixConfig("中文百科", weight=0.4, sources=[
                {"type": "wikipedia", "lang": "zh", "split": "train"}
            ]),
            DataMixConfig("英文百科", weight=0.2, sources=[
                {"type": "wikipedia", "lang": "en", "split": "train"}
            ]),
            DataMixConfig("代码", weight=0.3, sources=[
                {"type": "local", "path": "data/code/*.jsonl"}
            ]),
            DataMixConfig("对话", weight=0.1, sources=[
                {"type": "local", "path": "data/chat/*.jsonl"}
            ]),
        ])
        # 混合后总权重归一化 -> 中文百科40% + 英文百科20% + 代码30% + 对话10%
        mixed_dataset = mixer.mix_and_process(tokenizer, max_seq_length=2048)

    参考配比 (DeepSeek 风格):
        代码 40% + 中英文百科 25% + 论文 10% + 书籍 10% + 网页 10% + 对话 5%

    参考配比 (Llama 3 风格):
        网页 50% + 代码 25% + 百科 10% + 书籍 10% + 推理 5%
    """

    # 预设配比方案
    PRESETS = {
        "balanced_zh": [
            ("中文网页", 0.35),
            ("中文百科", 0.20),
            ("代码", 0.20),
            ("英文百科", 0.10),
            ("书籍", 0.10),
            ("对话", 0.05),
        ],
        "deepseek_style": [
            ("代码", 0.40),
            ("中英百科", 0.25),
            ("学术论文", 0.10),
            ("书籍", 0.10),
            ("通用网页", 0.10),
            ("对话", 0.05),
        ],
        "code_heavy": [
            ("代码", 0.50),
            ("技术文档", 0.20),
            ("百科", 0.15),
            ("通用文本", 0.15),
        ],
        "chinese_focused": [
            ("中文百科", 0.25),
            ("中文网页", 0.25),
            ("中文书籍", 0.15),
            ("中文对话", 0.10),
            ("代码", 0.15),
            ("英文百科", 0.10),
        ],
    }

    def __init__(
        self,
        configs: List[DataMixConfig],
        mix_strategy: str = "interleave",
        shuffle_seed: int = 42,
    ):
        if not configs:
            raise ValueError("configs must contain at least one data source")
        if mix_strategy not in {"interleave", "oversample", "stratified"}:
            raise ValueError(
                "mix_strategy must be one of: interleave, oversample, stratified"
            )
        if any(
            not math.isfinite(config.weight) or config.weight < 0 for config in configs
        ):
            raise ValueError("data source weights must be finite and non-negative")
        if any(config.num_repetitions < 1 for config in configs):
            raise ValueError("num_repetitions must be at least 1")
        if any(
            config.max_seq_length_override is not None
            and config.max_seq_length_override <= 0
            for config in configs
        ):
            raise ValueError("max_seq_length_override must be positive")

        # Normalize copies so caller-owned configuration objects are not
        # unexpectedly modified in place.
        self.configs = [replace(config) for config in configs]
        self.mix_strategy = mix_strategy
        self.shuffle_seed = shuffle_seed

        # 归一化权重
        total_w = sum(config.weight for config in self.configs)
        if total_w <= 0:
            raise ValueError("data source weights must have a positive total")
        for config in self.configs:
            config.weight /= total_w

    @classmethod
    def from_preset(
        cls, preset_name: str, source_paths: Dict[str, List[Dict]]
    ) -> "DataMixer":
        """从预设配比创建 DataMixer"""
        if preset_name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise KeyError(f"未知预设: {preset_name}. 可用: {available}")

        configs = []
        for name, weight in cls.PRESETS[preset_name]:
            sources = source_paths.get(name, [{"type": "wikitext", "split": "train"}])
            configs.append(
                DataMixConfig(
                    name=name,
                    weight=weight,
                    sources=sources,
                    lang_filter="zh" if "中文" in name else None,
                )
            )
        return cls(configs)

    def mix_and_process(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        num_proc: int = 4,
    ):
        """混合多源数据并进行处理,返回可用于训练的 dataset"""
        from datasets import interleave_datasets, concatenate_datasets

        all_datasets = []
        all_weights = []

        for cfg in self.configs:
            logger.info(f"处理数据源: {cfg.name} (权重: {cfg.weight:.2f})")

            pipe_config = PipelineConfig(
                sources=cfg.sources,
                max_seq_length=cfg.max_seq_length_override or max_seq_length,
                quality_min_score=cfg.quality_min_score,
                lang_filter=cfg.lang_filter,
                num_proc=num_proc,
                shuffle_seed=self.shuffle_seed,
            )

            pipeline = DataPipeline(pipe_config)
            pipeline.collect().clean().deduplicate().filter_quality()
            pipeline.tokenize_and_pack(tokenizer)
            ds = pipeline.to_iterable_dataset()

            # 小数据集重复
            if cfg.num_repetitions > 1:
                ds = concatenate_datasets([ds] * cfg.num_repetitions)

            all_datasets.append(ds)
            all_weights.append(cfg.weight)

        if len(all_datasets) == 1:
            logger.info("只有一个数据源,无需混合")
            return all_datasets[0]

        # 加权交错混合
        logger.info(
            f"混合 {len(all_datasets)} 个数据源, 权重: "
            f"{[f'{c.name}={c.weight:.2f}' for c in self.configs]}"
        )

        # interleave_datasets 的 probabilities 控制每个源被采样的概率
        if self.mix_strategy == "interleave":
            mixed = interleave_datasets(
                all_datasets,
                probabilities=all_weights,
                seed=self.shuffle_seed,
                stopping_strategy="first_exhausted",
            )
        elif self.mix_strategy == "oversample":
            # 按权重放大每个数据集
            min_weight = min(weight for weight in all_weights if weight > 0)
            repeated = []
            for ds, w in zip(all_datasets, all_weights):
                if w == 0:
                    continue
                factor = max(1, round(w / min_weight))
                repeated.append(concatenate_datasets([ds] * factor))
            mixed = interleave_datasets(repeated, seed=self.shuffle_seed)
        else:  # stratified
            mixed = interleave_datasets(
                all_datasets,
                probabilities=all_weights,
                seed=self.shuffle_seed,
                stopping_strategy="all_exhausted",
            )

        return mixed

    def print_mix_summary(self) -> str:
        """打印混合配比摘要"""
        lines = ["数据配比方案", "=" * 50]
        total_w = sum(c.weight for c in self.configs)
        for cfg in self.configs:
            pct = cfg.weight / max(total_w, 1e-6) * 100
            bar = "█" * int(pct / 2)
            lines.append(f"  {cfg.name:<15} {pct:>5.1f}% {bar}")
        lines.append("=" * 50)
        return "\n".join(lines)
