"""Train a task-specific Choice head and held-out temperature calibration.

Usage: python -m saddlellm.decision.training --help
The local backbone is frozen. This supervised baseline does not implement RLCD.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .Calibration import fit_temperature, probability_metrics
from .dataset import read_decision_dataset, sha256
from .Model import ChoiceDecisionModel
from .Schema import DecisionModelConfig


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)







def load_dataset(directory: str | Path) -> tuple[dict, dict, dict]:
    descriptor, records, hashes = read_decision_dataset(directory)
    labels = {choice["id"]: index for index, choice in enumerate(descriptor["choices"])}
    splits = {name: {"rows": rows, "labels": torch.tensor([labels[row["label"]] for row in rows], dtype=torch.long)} for name, rows in records.items()}
    return descriptor, splits, hashes


@dataclass(frozen=True)
class ChoiceTrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    projection_rank: int = 64
    seed: int = 42
    cpu_threads: int = 4
    encoder_batch_size: int = 16
    max_length: int = 512

    def __post_init__(self):
        for name in ("epochs", "batch_size", "projection_rank", "cpu_threads", "encoder_batch_size", "max_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0 or not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")


def fit_head(model, candidates, features, config, output):
    """Only train and validation participate in optimization/model selection."""
    optimizer = torch.optim.AdamW(model.head_parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    with torch.no_grad():
        initial = model.score(features["validation"]["features"], candidates)
        best_nll = float(F.cross_entropy(initial, features["validation"]["labels"]))
    best_epoch = 0
    best_state = {key: value.detach().clone() for key, value in model.head.state_dict().items()}
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(features["train"]["labels"]), generator=generator)
        total_loss = 0.0
        for indices in permutation.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model.score(features["train"]["features"][indices], candidates)
            loss = F.cross_entropy(logits, features["train"]["labels"][indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.head_parameters()), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits = model.score(features["validation"]["features"], candidates)
            nll = float(F.cross_entropy(logits, features["validation"]["labels"]))
            accuracy = float((logits.argmax(-1) == features["validation"]["labels"]).float().mean())
        if nll < best_nll:
            best_nll, best_epoch = nll, epoch
            best_state = {key: value.detach().clone() for key, value in model.head.state_dict().items()}
        row = {"epoch": epoch, "train_nll": total_loss / len(permutation), "validation_nll": nll, "validation_accuracy": accuracy}
        history.append(row)
        with (output / "training_history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    model.head.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_validation_nll": best_nll, "selection_metric": "validation_nll", "history": history}


def train_choice(dataset, backbone, output, *, config=None, encoder_device="cuda", encoder_dtype="bfloat16"):
    config = config or ChoiceTrainingConfig()
    descriptor, splits, hashes = load_dataset(dataset)
    backbone = Path(backbone).expanduser().resolve()
    backbone_config = json.loads((backbone / "config.json").read_text(encoding="utf-8"))
    identity = {p.name: sha256(p) for p in sorted(backbone.iterdir()) if p.is_file() and p.suffix in {".json", ".safetensors", ".model"}}
    model_config = DecisionModelConfig(backbone_name_or_path=str(backbone), embedding_dim=int(backbone_config["hidden_size"]), projection_rank=config.projection_rank, max_length=config.max_length, encoder_batch_size=config.encoder_batch_size, backbone_identity=identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    metadata = {"question": descriptor["question"], "choices": descriptor["choices"], "dataset_hashes": hashes, "scope": "Task-specific frozen-backbone Choice baseline; no Score/Noul or RLCD"}
    write_json(output / "run_config.json", {"training": asdict(config), "model": model_config.to_dict(), "metadata": metadata, "encoder_device": encoder_device, "encoder_dtype": encoder_dtype, "head_device": "cpu"})
    model = ChoiceDecisionModel(model_config, load_encoder=True, encoder_device=encoder_device, encoder_dtype=encoder_dtype, metadata=metadata)
    feature_dir = output / "features"
    feature_dir.mkdir()

    def encode_split(name):
        print(json.dumps({"phase": "encoding", "split": name, "samples": len(splits[name]["rows"])}), flush=True)
        values = model.encode_states([row["state"] for row in splits[name]["rows"]], descriptor["question"])
        save_file({"features": values, "labels": splits[name]["labels"]}, str(feature_dir / f"{name}.safetensors"))
        return {"features": values, "labels": splits[name]["labels"]}

    candidates = model.encode_choices([row["text"] for row in descriptor["choices"]], descriptor["question"])
    save_file({"candidates": candidates}, str(feature_dir / "candidates.safetensors"))
    features = {name: encode_split(name) for name in ("train", "validation")}
    baseline = ChoiceDecisionModel(model_config)
    baseline.head.load_state_dict(model.head.state_dict())
    selection = fit_head(model, candidates, features, config, output)
    del features
    calibration = encode_split("calibration")
    with torch.no_grad():
        logits = model.score(calibration["features"], candidates).numpy()
        baseline_logits = baseline.score(calibration["features"], candidates).numpy()
    fitted = fit_temperature(logits, calibration["labels"].numpy())
    baseline_fitted = fit_temperature(baseline_logits, calibration["labels"].numpy())
    model.set_calibration(fitted["temperature"], {"fit": fitted, "split_sha256": hashes["calibration"], "scope": {"question": descriptor["question"], "choices": descriptor["choices"]}})
    model.save_pretrained(output / "model")
    test = encode_split("test")
    with torch.no_grad():
        logits = model.score(test["features"], candidates).numpy()
        baseline_logits = baseline.score(test["features"], candidates).numpy()
    labels = test["labels"].numpy()
    report = {"schema_version": 1, "selection": selection, "calibration": fitted, "baseline_calibration": baseline_fitted, "test": {
        "cosine_baseline": probability_metrics(baseline_logits, labels),
        "cosine_baseline_calibrated": probability_metrics(baseline_logits, labels, temperature=baseline_fitted["temperature"]),
        "trained_uncalibrated": probability_metrics(logits, labels),
        "trained_calibrated": probability_metrics(logits, labels, temperature=fitted["temperature"]),
    }, "dataset_hashes": hashes, "counts": {name: len(value["rows"]) for name, value in splits.items()}, "training": asdict(config), "elapsed_seconds": time.perf_counter() - start, "scope": metadata["scope"], "test_used_for_selection": False}
    indices_path = Path(dataset) / "test_indices.json"
    if indices_path.exists():
        subsets = json.loads(indices_path.read_text(encoding="utf-8"))
        report["test_subsets"] = {}
        for name in ("deduplicated_test_indices", "clean_test_indices"):
            indices = np.asarray(subsets[name], dtype=np.int64)
            if len(indices) and len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(labels))):
                report["test_subsets"][name] = {"baseline": probability_metrics(baseline_logits[indices], labels[indices], temperature=baseline_fitted["temperature"]), "trained": probability_metrics(logits[indices], labels[indices], temperature=fitted["temperature"])}
            elif len(indices):
                raise ValueError("invalid test subset indices")
    reloaded = ChoiceDecisionModel.from_pretrained(output / "model", expected_backbone_identity=identity)
    with torch.no_grad():
        torch.testing.assert_close(reloaded.predict_proba(test["features"][:16], candidates), model.predict_proba(test["features"][:16], candidates), rtol=0, atol=0)
    report["artifact_reload_verified"] = True
    write_json(output / "report.json", report)
    write_json(output / "artifact_manifest.json", {"files": {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}, "backbone_bundled": False, "backbone": str(backbone), "backbone_identity": identity})
    print(json.dumps({"phase": "completed", "output": str(output), "best_epoch": selection["best_epoch"], "test_accuracy": report["test"]["trained_calibrated"]["accuracy"], "test_ece": report["test"]["trained_calibrated"]["top_label_ece"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--encoder-dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    for name, default in asdict(ChoiceTrainingConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    values = vars(parser.parse_args())
    config = ChoiceTrainingConfig(**{name: values.pop(name) for name in asdict(ChoiceTrainingConfig())})
    train_choice(**values, config=config)


if __name__ == "__main__":
    main()
