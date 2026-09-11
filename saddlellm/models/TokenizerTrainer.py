"""
分词器训练器 - 从零训练 BPE/SentencePiece tokenizer
支持中文优化、大规模语料流式处理
"""
import os
import json
import logging
from typing import Dict, Iterator, List, Optional, Sequence, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    algorithm: str = "bpe"  # bpe, unigram, wordpiece
    min_frequency: int = 2
    special_tokens: List[str] = field(default_factory=lambda: [
        "<pad>", "<unk>", "<bos>", "<eos>", "<mask>"
    ])
    max_token_length: int = 128
    byte_level: bool = True
    unicode_normalizer: Optional[str] = None  # nfkc, nfc, nfkd, nfd
    chinese_char_coverage: float = 0.995  # 中文字符覆盖率目标


@dataclass
class TokenizerEvalResult:
    samples: int
    chars: int
    tokens: int
    compression_chars_per_token: float
    tokens_per_char: float
    unk_tokens: int
    unk_rate: float
    roundtrip_exact_rate: float
    avg_tokens_per_domain_term: float = 0.0
    single_token_domain_term_rate: float = 0.0
    domain_terms: int = 0

    def to_dict(self) -> Dict:
        return {
            "samples": self.samples,
            "chars": self.chars,
            "tokens": self.tokens,
            "compression_chars_per_token": self.compression_chars_per_token,
            "tokens_per_char": self.tokens_per_char,
            "unk_tokens": self.unk_tokens,
            "unk_rate": self.unk_rate,
            "roundtrip_exact_rate": self.roundtrip_exact_rate,
            "avg_tokens_per_domain_term": self.avg_tokens_per_domain_term,
            "single_token_domain_term_rate": self.single_token_domain_term_rate,
            "domain_terms": self.domain_terms,
        }


class TokenizerTrainer:
    """
    从零训练 BPE / Unigram (SentencePiece) tokenizer。

    用法:
        trainer = TokenizerTrainer(vocab_size=32000, algorithm="bpe")
        trainer.fit(["data/corpus/*.txt", "data/wiki.jsonl"])
        trainer.save("./my_tokenizer")

        # 加载使用
        tokenizer = TokenizerTrainer.load("./my_tokenizer")
        ids = tokenizer.encode("你好世界")
        text = tokenizer.decode(ids)
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        algorithm: str = "bpe",
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        max_token_length: int = 128,
        byte_level: bool = True,
        unicode_normalizer: Optional[str] = None,
        chinese_char_coverage: float = 0.995,
    ):
        self.config = TokenizerConfig(
            vocab_size=vocab_size,
            algorithm=algorithm,
            min_frequency=min_frequency,
            special_tokens=special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"],
            max_token_length=max_token_length,
            byte_level=byte_level,
            unicode_normalizer=unicode_normalizer,
            chinese_char_coverage=chinese_char_coverage,
        )
        self._tokenizer = None

    def fit(
        self,
        corpus_files: Union[str, List[str]],
        file_format: str = "auto",
        text_column: str = "text",
        limit_gb: Optional[float] = None,
    ) -> "TokenizerTrainer":
        """
        在语料上训练 tokenizer。

        参数:
            corpus_files: 文件路径或路径列表,支持 glob pattern
            file_format: "auto" | "txt" | "jsonl" | "parquet"
            text_column: jsonl/parquet 中的文本列名
            limit_gb: 限制读取的语料大小(GB),用于快速测试
        """
        files = self._resolve_files(corpus_files)
        if not files:
            raise FileNotFoundError(f"未找到语料文件: {corpus_files}")

        logger.info(f"开始训练 tokenizer,语料文件数: {len(files)}, 目标词表: {self.config.vocab_size}")

        if self.config.algorithm == "bpe":
            self._train_bpe(files, file_format, text_column, limit_gb)
        elif self.config.algorithm == "unigram":
            self._train_unigram(files, file_format, text_column, limit_gb)
        elif self.config.algorithm == "wordpiece":
            self._train_wordpiece(files, file_format, text_column, limit_gb)
        else:
            raise ValueError(f"不支持的算法: {self.config.algorithm}")

        logger.info(f"训练完成,词表大小: {self._tokenizer.get_vocab_size()}")
        return self

    def _train_bpe(self, files, file_format, text_column, limit_gb):
        """使用 HuggingFace tokenizers 库训练 BPE"""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=self.config.byte_level))

        if self.config.byte_level:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.decoder = decoders.ByteLevel()
        else:
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        if self.config.unicode_normalizer:
            tokenizer.normalizer = normalizers.NFKC() if self.config.unicode_normalizer == "nfkc" else normalizers.NFD()

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if self.config.byte_level else [],
        )

        def corpus_iterator():
            total_bytes = 0
            limit_bytes = (limit_gb or float("inf")) * 1024**3
            for fpath in files:
                fmt = self._detect_format(fpath, file_format)
                for text in self._read_texts(fpath, fmt, text_column):
                    if text and len(text.strip()) > 10:
                        total_bytes += len(text.encode("utf-8"))
                        if total_bytes > limit_bytes:
                            return
                        yield text

        tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
        bos_id = tokenizer.token_to_id("<bos>")
        eos_id = tokenizer.token_to_id("<eos>")
        if bos_id is not None and eos_id is not None:
            tokenizer.post_processor = processors.TemplateProcessing(
                single="<bos> $A <eos>",
                pair="<bos> $A <eos> $B:1 <eos>:1",
                special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
            )
        self._tokenizer = tokenizer
        self._tokenizer.enable_padding(pad_id=self._tokenizer.token_to_id("<pad>"), pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=self.config.max_token_length)

    def _train_unigram(self, files, file_format, text_column, limit_gb):
        """使用 SentencePiece 训练 Unigram 模型"""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("训练 Unigram 需要 sentencepiece: pip install sentencepiece")

        merged_path = self._merge_corpus_to_temp(files, file_format, text_column, limit_gb)
        model_prefix = merged_path + ".spm"

        spm.SentencePieceTrainer.train(
            input=merged_path,
            model_prefix=model_prefix,
            vocab_size=self.config.vocab_size,
            model_type="unigram",
            character_coverage=self.config.chinese_char_coverage,
            input_sentence_size=10000000,
            shuffle_input_sentence=True,
            num_threads=os.cpu_count() or 4,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>",
            bos_piece="<bos>", eos_piece="<eos>",
            user_defined_symbols=self.config.special_tokens,
        )

        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(model_prefix + ".model" if os.path.exists(model_prefix + ".model") else model_prefix)

        for tmp in [merged_path, model_prefix + ".model", model_prefix + ".vocab"]:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _train_wordpiece(self, files, file_format, text_column, limit_gb):
        """使用 HuggingFace tokenizers 训练 WordPiece"""
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.config.vocab_size,
            special_tokens=self.config.special_tokens,
            min_frequency=self.config.min_frequency,
        )

        def corpus_iterator():
            total_bytes = 0
            limit_bytes = (limit_gb or float("inf")) * 1024**3
            for fpath in files:
                fmt = self._detect_format(fpath, file_format)
                for text in self._read_texts(fpath, fmt, text_column):
                    if text and len(text.strip()) > 10:
                        total_bytes += len(text.encode("utf-8"))
                        if total_bytes > limit_bytes:
                            return
                        yield text

        tokenizer.train_from_iterator(corpus_iterator(), trainer=trainer)
        self._tokenizer = tokenizer
        self._tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=self.config.max_token_length)

    def save(self, path: str) -> str:
        """保存 tokenizer 到目录"""
        if self._tokenizer is None:
            raise RuntimeError("请先调用 fit() 训练 tokenizer")
        os.makedirs(path, exist_ok=True)

        if hasattr(self._tokenizer, "save"):
            self._tokenizer.save(os.path.join(path, "tokenizer.json"))

        config = {
            "vocab_size": self.config.vocab_size,
            "algorithm": self.config.algorithm,
            "special_tokens": self.config.special_tokens,
            "max_token_length": self.config.max_token_length,
            "byte_level": self.config.byte_level,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 同时生成 HuggingFace transformers 兼容的配置
        hf_config = {
            "add_prefix_space": False,
            "added_tokens_decoder": {},
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "model_max_length": self.config.max_token_length * 16,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
        }
        with open(os.path.join(path, "special_tokens_map.json"), "w", encoding="utf-8") as f:
            json.dump(hf_config, f, ensure_ascii=False, indent=2)

        # 保存特殊 token 映射
        token_map = {}
        for tok in self.config.special_tokens:
            tid = self._tokenizer.token_to_id(tok)
            if tid is not None:
                token_map[tok] = tid
        with open(os.path.join(path, "token_map.json"), "w", encoding="utf-8") as f:
            json.dump(token_map, f, ensure_ascii=False, indent=2)

        logger.info(f"Tokenizer 已保存到 {path}")
        return path

    @classmethod
    def load(cls, path: str) -> "TokenizerTrainer":
        """加载已保存的 tokenizer"""
        config_path = os.path.join(path, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        instance = cls(
            vocab_size=config["vocab_size"],
            algorithm=config.get("algorithm", "bpe"),
            special_tokens=config.get("special_tokens"),
            max_token_length=config.get("max_token_length", 128),
            byte_level=config.get("byte_level", True),
        )

        from tokenizers import Tokenizer
        tok_path = os.path.join(path, "tokenizer.json")
        if os.path.exists(tok_path):
            instance._tokenizer = Tokenizer.from_file(tok_path)
        else:
            raise FileNotFoundError(f"找不到 tokenizer.json: {tok_path}")

        return instance

    def encode(self, text: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """编码文本为 token IDs"""
        if self._tokenizer is None:
            raise RuntimeError("请先训练或加载 tokenizer")
        if isinstance(text, str):
            return self._tokenizer.encode(text).ids
        return [self._tokenizer.encode(t).ids for t in text]

    def decode(self, ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """解码 token IDs 为文本"""
        if self._tokenizer is None:
            raise RuntimeError("请先训练或加载 tokenizer")
        if ids and isinstance(ids[0], int):
            return self._tokenizer.decode(ids)
        return [self._tokenizer.decode(i) for i in ids]

    def evaluate(
        self,
        corpus_files: Optional[Union[str, List[str]]] = None,
        texts: Optional[Sequence[str]] = None,
        file_format: str = "auto",
        text_column: str = "text",
        max_samples: int = 1000,
        domain_terms: Optional[Sequence[str]] = None,
        output_path: Optional[str] = None,
    ) -> TokenizerEvalResult:
        """Evaluate tokenizer quality before scratch pretraining."""
        if self._tokenizer is None:
            raise RuntimeError("璇峰厛璁粌鎴栧姞杞?tokenizer")

        samples = 0
        char_count = 0
        token_count = 0
        unk_count = 0
        roundtrip_ok = 0
        unk_id = self._tokenizer.token_to_id("<unk>") if hasattr(self._tokenizer, "token_to_id") else None

        for text in self._iter_eval_texts(corpus_files, texts, file_format, text_column):
            text = text or ""
            if not text.strip():
                continue
            encoded = self._tokenizer.encode(text)
            ids = list(getattr(encoded, "ids", encoded))
            samples += 1
            char_count += len(text)
            token_count += len(ids)
            if unk_id is not None:
                unk_count += sum(1 for idx in ids if idx == unk_id)

            try:
                decoded = self._tokenizer.decode(ids, skip_special_tokens=True)
            except TypeError:
                decoded = self._tokenizer.decode(ids)
            if self._normalize_for_roundtrip(decoded) == self._normalize_for_roundtrip(text):
                roundtrip_ok += 1
            if samples >= max_samples:
                break

        term_stats = self._evaluate_domain_terms(domain_terms or [])
        result = TokenizerEvalResult(
            samples=samples,
            chars=char_count,
            tokens=token_count,
            compression_chars_per_token=(char_count / token_count) if token_count else 0.0,
            tokens_per_char=(token_count / char_count) if char_count else 0.0,
            unk_tokens=unk_count,
            unk_rate=(unk_count / token_count) if token_count else 0.0,
            roundtrip_exact_rate=(roundtrip_ok / samples) if samples else 0.0,
            avg_tokens_per_domain_term=term_stats["avg_tokens_per_term"],
            single_token_domain_term_rate=term_stats["single_token_rate"],
            domain_terms=term_stats["domain_terms"],
        )
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        return result

    def get_hf_tokenizer(self):
        """转换为 HuggingFace PreTrainedTokenizerFast (用于 transformers 训练)"""
        from transformers import PreTrainedTokenizerFast

        if self._tokenizer is None:
            raise RuntimeError("请先训练或加载 tokenizer")

        tok_path = os.path.join(os.path.dirname(__file__), "_tmp_tok.json")
        if hasattr(self._tokenizer, "save"):
            self._tokenizer.save(tok_path)
            hf_tok = PreTrainedTokenizerFast(
                tokenizer_file=tok_path,
                bos_token="<bos>",
                eos_token="<eos>",
                unk_token="<unk>",
                pad_token="<pad>",
                model_max_length=self.config.max_token_length * 16,
            )
            os.remove(tok_path)
            return hf_tok
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        if self._tokenizer:
            return self._tokenizer.get_vocab_size()
        return 0

    # ---------- 内部工具方法 ----------

    def _iter_eval_texts(
        self,
        corpus_files: Optional[Union[str, List[str]]],
        texts: Optional[Sequence[str]],
        file_format: str,
        text_column: str,
    ) -> Iterator[str]:
        if texts is not None:
            for text in texts:
                yield text
            return
        if corpus_files is None:
            return
        for fpath in self._resolve_files(corpus_files):
            fmt = self._detect_format(fpath, file_format)
            yield from self._read_texts(fpath, fmt, text_column)

    def _evaluate_domain_terms(self, domain_terms: Sequence[str]) -> Dict[str, float]:
        if not domain_terms:
            return {"domain_terms": 0, "avg_tokens_per_term": 0.0, "single_token_rate": 0.0}
        lengths = []
        for term in domain_terms:
            if not term:
                continue
            encoded = self._tokenizer.encode(term)
            ids = list(getattr(encoded, "ids", encoded))
            lengths.append(len(ids))
        if not lengths:
            return {"domain_terms": 0, "avg_tokens_per_term": 0.0, "single_token_rate": 0.0}
        return {
            "domain_terms": len(lengths),
            "avg_tokens_per_term": sum(lengths) / len(lengths),
            "single_token_rate": sum(1 for n in lengths if n == 1) / len(lengths),
        }

    def _normalize_for_roundtrip(self, text: str) -> str:
        return " ".join((text or "").strip().split())

    def _resolve_files(self, corpus_files) -> List[str]:
        import glob
        if isinstance(corpus_files, str):
            corpus_files = [corpus_files]
        resolved = []
        for pattern in corpus_files:
            matched = glob.glob(pattern, recursive=True)
            if matched:
                resolved.extend(matched)
            elif os.path.isfile(pattern):
                resolved.append(pattern)
            elif os.path.isdir(pattern):
                for root, _, files in os.walk(pattern):
                    for f in files:
                        if f.endswith((".txt", ".jsonl", ".json", ".parquet")):
                            resolved.append(os.path.join(root, f))
        return sorted(resolved)

    def _detect_format(self, fpath: str, hint: str) -> str:
        if hint != "auto":
            return hint
        ext = os.path.splitext(fpath)[1].lower()
        return {".jsonl": "jsonl", ".json": "json", ".parquet": "parquet"}.get(ext, "txt")

    def _read_texts(self, fpath: str, fmt: str, text_column: str) -> Iterator[str]:
        if fmt == "jsonl":
            import json as _json
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        obj = _json.loads(line)
                        yield obj.get(text_column, "")
                    except _json.JSONDecodeError:
                        continue
        elif fmt == "json":
            import json as _json
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                data = _json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        yield obj.get(text_column, "") if isinstance(obj, dict) else str(obj)
        elif fmt == "parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(fpath, columns=[text_column])
            for batch in table.to_batches():
                for text in batch.column(0).to_pylist():
                    yield str(text) if text else ""
        else:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                yield f.read()

    def _merge_corpus_to_temp(self, files, fmt, text_column, limit_gb) -> str:
        """将语料合并到临时文件 (SentencePiece 需要单个文件输入)"""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        total_bytes = 0
        limit_bytes = (limit_gb or float("inf")) * 1024**3
        for fpath in files:
            file_fmt = self._detect_format(fpath, fmt)
            for text in self._read_texts(fpath, file_fmt, text_column):
                if text and len(text.strip()) > 10:
                    total_bytes += len(text.encode("utf-8"))
                    if total_bytes > limit_bytes:
                        break
                    tmp.write(text.strip() + "\n")
            if total_bytes > limit_bytes:
                break
        tmp.close()
        return tmp.name
