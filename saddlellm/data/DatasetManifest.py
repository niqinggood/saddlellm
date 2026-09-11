"""Dataset manifest utilities for LLM training factories.

The manifest is a small control-plane format inspired by dataset registry
files used by modern LLM training frameworks.  It normalizes local files,
Hugging Face datasets, ShareGPT/Alpaca-style columns, roles, and source
weights into the source dictionaries expected by SaddleLLM's data pipeline.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass
class DatasetManifestEntry:
    name: str
    source_type: str
    source: str
    formatting: str = "alpaca"
    columns: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    split: str = "train"
    role: str = "pretrain"
    ranking: bool = False
    weight: float = 1.0
    license: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def text_column(self) -> str:
        for key in ("text", "content", "prompt", "messages"):
            if key in self.columns:
                return self.columns[key]
        return "text"

    def to_saddle_source(self) -> Dict:
        """Convert to a DataPipeline-compatible source dictionary."""
        source = {
            "name": self.name,
            "split": self.split,
            "formatting": self.formatting,
            "columns": dict(self.columns),
            "tags": dict(self.tags),
            "role": self.role,
            "ranking": self.ranking,
            "text_column": self.text_column(),
        }
        if self.source_type in {"hf_hub", "huggingface"}:
            source.update({"type": "huggingface", "path": self.source})
        elif self.source_type in {"file", "local"}:
            source.update({"type": "local", "path": self.source, "format": "auto"})
        else:
            source.update({"type": self.source_type, "path": self.source})
        return source


@dataclass
class DatasetManifest:
    entries: List[DatasetManifestEntry] = field(default_factory=list)
    name: str = "saddlellm-dataset-manifest"
    version: str = "1.0"
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetManifest":
        entries = [
            DatasetManifestEntry(**entry)
            for entry in data.get("entries", [])
        ]
        return cls(
            entries=entries,
            name=data.get("name", "saddlellm-dataset-manifest"),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[DatasetManifestEntry],
        name: str = "saddlellm-dataset-manifest",
        metadata: Optional[Dict] = None,
    ) -> "DatasetManifest":
        return cls(entries=list(entries), name=name, metadata=metadata or {})

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def from_llamafactory_json(
        cls,
        path: str,
        dataset_dir: Optional[str] = None,
        selected_names: Optional[Sequence[str]] = None,
        default_role: str = "sft",
        default_weight: float = 1.0,
    ) -> "DatasetManifest":
        """Load a LLaMA-Factory-style dataset_info.json into a neutral manifest."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        selected = set(selected_names or raw.keys())
        base_dir = dataset_dir or os.path.dirname(path)
        entries: List[DatasetManifestEntry] = []

        for name, info in raw.items():
            if name not in selected:
                continue
            source_type, source = cls._resolve_llamafactory_source(info, base_dir)
            if not source:
                continue
            role = cls._infer_role(info, default_role)
            entry = DatasetManifestEntry(
                name=name,
                source_type=source_type,
                source=source,
                formatting=info.get("formatting", "alpaca"),
                columns=dict(info.get("columns", {})),
                tags=dict(info.get("tags", {})),
                split=info.get("split", "train"),
                role=role,
                ranking=bool(info.get("ranking", False)),
                weight=float(info.get("weight", default_weight)),
                license=info.get("license"),
                metadata={
                    key: value
                    for key, value in info.items()
                    if key
                    not in {
                        "file_name",
                        "hf_hub_url",
                        "ms_hub_url",
                        "om_hub_url",
                        "script_url",
                        "cloud_file_name",
                        "formatting",
                        "columns",
                        "tags",
                        "split",
                        "ranking",
                        "weight",
                        "license",
                    }
                },
            )
            entries.append(entry)

        return cls(
            entries=entries,
            name=os.path.splitext(os.path.basename(path))[0],
            metadata={
                "source_format": "llamafactory_dataset_info",
                "source_path": os.path.abspath(path),
            },
        )

    def filter(
        self,
        role: Optional[str] = None,
        formatting: Optional[str] = None,
        ranking: Optional[bool] = None,
    ) -> "DatasetManifest":
        entries = self.entries
        if role is not None:
            entries = [entry for entry in entries if entry.role == role]
        if formatting is not None:
            entries = [entry for entry in entries if entry.formatting == formatting]
        if ranking is not None:
            entries = [entry for entry in entries if entry.ranking == ranking]
        return DatasetManifest(entries=entries, name=self.name, version=self.version, metadata=dict(self.metadata))

    def to_saddle_sources(self, role: Optional[str] = None) -> List[Dict]:
        entries = self.filter(role=role).entries if role else self.entries
        return [entry.to_saddle_source() for entry in entries]

    def source_weights(self, role: Optional[str] = None, normalize: bool = True) -> List[float]:
        entries = self.filter(role=role).entries if role else self.entries
        weights = [max(0.0, float(entry.weight)) for entry in entries]
        total = sum(weights)
        if normalize and total > 0:
            return [weight / total for weight in weights]
        return weights

    def summary(self) -> Dict:
        by_role: Dict[str, int] = {}
        by_formatting: Dict[str, int] = {}
        by_source_type: Dict[str, int] = {}
        for entry in self.entries:
            by_role[entry.role] = by_role.get(entry.role, 0) + 1
            by_formatting[entry.formatting] = by_formatting.get(entry.formatting, 0) + 1
            by_source_type[entry.source_type] = by_source_type.get(entry.source_type, 0) + 1
        return {
            "name": self.name,
            "version": self.version,
            "num_entries": len(self.entries),
            "by_role": by_role,
            "by_formatting": by_formatting,
            "by_source_type": by_source_type,
            "total_weight": sum(float(entry.weight) for entry in self.entries),
        }

    @staticmethod
    def _resolve_llamafactory_source(info: Dict, base_dir: str) -> (str, str):
        if "file_name" in info:
            return "local", os.path.abspath(os.path.join(base_dir, info["file_name"]))
        if "hf_hub_url" in info:
            return "hf_hub", info["hf_hub_url"]
        if "ms_hub_url" in info:
            return "modelscope_hub", info["ms_hub_url"]
        if "om_hub_url" in info:
            return "openmind_hub", info["om_hub_url"]
        if "script_url" in info:
            return "script", info["script_url"]
        if "cloud_file_name" in info:
            return "cloud_file", info["cloud_file_name"]
        return "unknown", ""

    @staticmethod
    def _infer_role(info: Dict, default_role: str) -> str:
        if info.get("ranking", False):
            return "preference"
        formatting = info.get("formatting", "alpaca")
        columns = info.get("columns", {})
        if "messages" in columns or formatting in {"sharegpt", "alpaca"}:
            return default_role
        return "pretrain"
