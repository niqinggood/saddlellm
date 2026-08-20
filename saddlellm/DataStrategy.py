"""Data strategy tools for scratch and domain pretraining.

The goal is to make data decisions explicit before a long run starts:
domain/general mix ratios, source wiring, and benchmark contamination checks.
"""
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union


@dataclass
class DataBucketPlan:
    name: str
    weight: float
    role: str
    quality_min_score: float = 0.3
    max_repetition: int = 1
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DataMixPlan:
    domain: str
    buckets: List[DataBucketPlan]
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def normalized(self) -> "DataMixPlan":
        total = sum(max(0.0, b.weight) for b in self.buckets)
        if total <= 0:
            return self
        return DataMixPlan(
            domain=self.domain,
            buckets=[
                DataBucketPlan(
                    name=b.name,
                    weight=b.weight / total,
                    role=b.role,
                    quality_min_score=b.quality_min_score,
                    max_repetition=b.max_repetition,
                    notes=b.notes,
                )
                for b in self.buckets
            ],
            warnings=list(self.warnings),
            notes=list(self.notes),
        )

    def to_dict(self) -> Dict:
        plan = self.normalized()
        return {
            "domain": plan.domain,
            "buckets": [b.to_dict() for b in plan.buckets],
            "warnings": plan.warnings,
            "notes": plan.notes,
        }

    def source_weights(self) -> List[float]:
        return [b.weight for b in self.normalized().buckets]

    def to_pipeline_config(
        self,
        sources_by_bucket: Dict[str, Sequence[Dict]],
        output_dir: str = "./processed_data",
        max_seq_length: int = 2048,
        dedup_method: str = "minhash",
        quality_min_score: Optional[float] = None,
    ) -> Dict:
        plan = self.normalized()
        sources = []
        weights = []
        missing = []
        for bucket in plan.buckets:
            bucket_sources = list(sources_by_bucket.get(bucket.name, []))
            if not bucket_sources:
                missing.append(bucket.name)
                continue
            for source in bucket_sources:
                src = dict(source)
                src.setdefault("bucket", bucket.name)
                sources.append(src)
                weights.append(bucket.weight / len(bucket_sources))
        if not sources:
            raise ValueError("No sources provided for any data bucket")
        total = sum(weights)
        weights = [w / total for w in weights] if total else None
        return {
            "sources": sources,
            "source_weights": weights,
            "output_dir": output_dir,
            "max_seq_length": max_seq_length,
            "dedup_method": dedup_method,
            "quality_min_score": quality_min_score if quality_min_score is not None else 0.35,
            "pack_sequences": True,
            "missing_buckets": missing,
            "mix_plan": plan.to_dict(),
        }


class DataMixPlanner:
    """Create conservative data mixes for domain pretraining."""

    BASE_BUCKETS = {
        "common_web": ("general_language", 0.20, "General language and breadth."),
        "encyclopedia": ("factual_general", 0.12, "High-density factual grounding."),
        "books": ("long_form", 0.10, "Long-context prose and coherence."),
        "papers": ("technical", 0.14, "Research and formal terminology."),
        "code": ("reasoning_code", 0.08, "Symbolic/code reasoning and tool syntax."),
        "domain_docs": ("domain_knowledge", 0.28, "Core domain corpus."),
        "qa_dialogue": ("instruction_seed", 0.08, "Small amount of QA/dialogue style data."),
    }

    DOMAIN_OVERRIDES = {
        "medical": {"domain_docs": 0.34, "papers": 0.16, "qa_dialogue": 0.06, "code": 0.04},
        "biology": {"domain_docs": 0.32, "papers": 0.20, "qa_dialogue": 0.04, "code": 0.04},
        "research": {"domain_docs": 0.24, "papers": 0.24, "books": 0.12, "code": 0.10},
        "risk": {"domain_docs": 0.34, "encyclopedia": 0.10, "qa_dialogue": 0.10, "code": 0.04},
        "semiconductor": {"domain_docs": 0.36, "papers": 0.16, "code": 0.08, "qa_dialogue": 0.04},
    }

    @classmethod
    def for_domain(
        cls,
        domain: str,
        domain_boost: float = 0.0,
        include_code: bool = True,
        include_dialogue: bool = True,
    ) -> DataMixPlan:
        key = (domain or "general").lower()
        weights = {name: default for name, (_, default, _) in cls.BASE_BUCKETS.items()}
        weights.update(cls.DOMAIN_OVERRIDES.get(key, {}))
        if domain_boost:
            weights["domain_docs"] = max(0.0, weights.get("domain_docs", 0.0) + domain_boost)
        if not include_code:
            weights["code"] = 0.0
        if not include_dialogue:
            weights["qa_dialogue"] = 0.0

        buckets = [
            DataBucketPlan(
                name=name,
                weight=weight,
                role=cls.BASE_BUCKETS[name][0],
                quality_min_score=0.4 if name in {"papers", "domain_docs"} else 0.3,
                max_repetition=2 if name == "domain_docs" else 1,
                notes=cls.BASE_BUCKETS[name][2],
            )
            for name, weight in weights.items()
            if weight > 0
        ]
        warnings = []
        domain_weight = weights.get("domain_docs", 0.0)
        if domain_weight > 0.45:
            warnings.append("domain_docs weight is high; keep general data to avoid capability collapse")
        if weights.get("qa_dialogue", 0.0) > 0.15:
            warnings.append("qa_dialogue is high for pretraining; prefer SFT for instruction style")
        notes = [
            "Use this as a pretraining mix, not as an SFT data recipe.",
            "Keep eval/benchmark data outside all buckets and scan for contamination before training.",
        ]
        return DataMixPlan(domain=key, buckets=buckets, warnings=warnings, notes=notes).normalized()


@dataclass
class ContaminationMatch:
    reference_id: str
    sample_id: str
    score: float
    overlap: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ContaminationReport:
    checked_samples: int
    contaminated_samples: int
    contamination_rate: float
    threshold: float
    matches: List[ContaminationMatch]

    def to_dict(self) -> Dict:
        return {
            "checked_samples": self.checked_samples,
            "contaminated_samples": self.contaminated_samples,
            "contamination_rate": self.contamination_rate,
            "threshold": self.threshold,
            "matches": [m.to_dict() for m in self.matches],
        }


class ContaminationDetector:
    """Approximate benchmark contamination detector based on word shingles."""

    def __init__(self, ngram: int = 8, threshold: float = 0.35):
        if ngram <= 0:
            raise ValueError("ngram must be positive")
        self.ngram = ngram
        self.threshold = threshold
        self._references: Dict[str, set] = {}

    def add_references(self, references: Union[Dict[str, str], Sequence[str]]):
        if isinstance(references, dict):
            items = references.items()
        else:
            items = [(f"ref-{i}", text) for i, text in enumerate(references)]
        for ref_id, text in items:
            shingles = self._shingles(text)
            if shingles:
                self._references[str(ref_id)] = shingles
        return self

    def add_reference_files(
        self,
        paths: Union[str, Sequence[str]],
        text_column: str = "text",
        max_samples: Optional[int] = None,
    ):
        for i, text in enumerate(self._iter_files(paths, text_column=text_column)):
            self.add_references({f"file-ref-{i}": text})
            if max_samples and i + 1 >= max_samples:
                break
        return self

    def scan_texts(self, texts: Sequence[str], threshold: Optional[float] = None) -> ContaminationReport:
        return self._scan(((f"sample-{i}", text) for i, text in enumerate(texts)), threshold)

    def scan_files(
        self,
        paths: Union[str, Sequence[str]],
        text_column: str = "text",
        max_samples: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> ContaminationReport:
        def iterator():
            for i, text in enumerate(self._iter_files(paths, text_column=text_column)):
                yield f"file-sample-{i}", text
                if max_samples and i + 1 >= max_samples:
                    break
        return self._scan(iterator(), threshold)

    def save_report(self, path: str, report: ContaminationReport) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def _scan(self, samples: Iterator[Tuple[str, str]], threshold: Optional[float]) -> ContaminationReport:
        if not self._references:
            raise ValueError("No references loaded. Call add_references() first.")
        threshold = self.threshold if threshold is None else threshold
        checked = 0
        contaminated = 0
        matches: List[ContaminationMatch] = []
        for sample_id, text in samples:
            checked += 1
            shingles = self._shingles(text)
            best = None
            for ref_id, ref_shingles in self._references.items():
                if not shingles or not ref_shingles:
                    continue
                overlap = len(shingles & ref_shingles)
                denom = min(len(shingles), len(ref_shingles))
                score = overlap / max(1, denom)
                if best is None or score > best.score:
                    best = ContaminationMatch(ref_id, sample_id, score, overlap)
            if best and best.score >= threshold:
                contaminated += 1
                matches.append(best)
        return ContaminationReport(
            checked_samples=checked,
            contaminated_samples=contaminated,
            contamination_rate=(contaminated / checked) if checked else 0.0,
            threshold=threshold,
            matches=matches,
        )

    def _shingles(self, text: str) -> set:
        tokens = self._tokens(text)
        if len(tokens) < self.ngram:
            return set(tokens) if tokens else set()
        return {" ".join(tokens[i:i + self.ngram]) for i in range(len(tokens) - self.ngram + 1)}

    def _tokens(self, text: str) -> List[str]:
        text = (text or "").lower()
        return re.findall(r"[\w\u4e00-\u9fff]+", text)

    def _iter_files(self, paths: Union[str, Sequence[str]], text_column: str = "text") -> Iterator[str]:
        import glob

        path_list = [paths] if isinstance(paths, str) else list(paths)
        files = []
        for path in path_list:
            if any(ch in path for ch in "*?[]"):
                files.extend(glob.glob(path, recursive=True))
            elif os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "**/*.*"), recursive=True))
            elif os.path.isfile(path):
                files.append(path)
        for file_path in sorted(files):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".jsonl":
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield str(obj.get(text_column, ""))
            elif ext == ".json":
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        yield str(obj.get(text_column, "")) if isinstance(obj, dict) else str(obj)
                elif isinstance(data, dict):
                    yield str(data.get(text_column, ""))
            else:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    yield f.read()

