"""Joint supervised training and an experimental REINFORCE probability policy.

REINFORCE samples Gaussian logit actions, receives negative Brier rewards from
known outcome distributions, and uses a leave-one-out baseline. It is our
offline experiment, not a reproduction of TypeSafe's unpublished RLCD.
"""

# ruff: noqa: N999
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .TypedModel import TypedDecisionModel
from .TypedSchema import (
    SPLITS,
    TYPES,
    TypedDecisionConfig,
    candidate_query_text,
    candidates,
    read_typed_dataset,
    scope_key,
)


@dataclass(frozen=True)
class TypedTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    seed: int = 42
    cpu_threads: int = 4
    method: str = "supervised"
    brier_weight: float = 0.1
    ordinal_weight: float = 0.1
    rl_weight: float = 0.1
    rl_samples: int = 8
    rl_std: float = 0.5
    supervised_weight: float = 1.0
    optimizer_device: str = "model"
    backbone_learning_rate: float | None = None
    head_learning_rate: float | None = None
    head_warmup_epochs: int = 0
    early_stopping_patience: int = 0
    evaluate_test: bool = True

    def __post_init__(self):
        for name in ("epochs", "batch_size", "cpu_threads", "rl_samples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        for name in ("head_warmup_epochs", "early_stopping_patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.evaluate_test) is not bool:
            raise ValueError("evaluate_test must be boolean")
        for name in ("backbone_learning_rate", "head_learning_rate"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be None or a finite positive number")
        for name in (
            "learning_rate",
            "weight_decay",
            "brier_weight",
            "ordinal_weight",
            "rl_weight",
            "rl_std",
            "supervised_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.rl_std == 0 or self.rl_samples < 2:
            raise ValueError(
                "learning_rate/rl_std must be positive; rl_samples must be >=2"
            )
        if self.method not in {"supervised", "reinforce_brier"}:
            raise ValueError("typed method must be supervised or reinforce_brier")
        if self.optimizer_device not in {"model", "cpu"}:
            raise ValueError("optimizer_device must be model or cpu")
        if (
            self.method == "supervised"
            and self.supervised_weight == 0
            or self.method == "reinforce_brier"
            and self.rl_weight == 0
        ):
            raise ValueError("selected training objective must have a positive weight")


def supervised_loss(logits, rows, *, brier_weight=0.1, ordinal_weight=0.1):
    losses = []
    for values, row in zip(logits, rows):
        target = values.new_tensor(row["target"])
        probs = values.softmax(-1)
        loss = (
            -(target * values.log_softmax(-1)).sum()
            + brier_weight * (probs - target).square().sum()
        )
        if row["type"] == "score":
            # Ranked probability score respects the ordinal relation of levels.
            loss = (
                loss
                + ordinal_weight
                * (probs.cumsum(-1)[:-1] - target.cumsum(-1)[:-1]).square().mean()
            )
        losses.append(loss)
    return torch.stack(losses).mean()


def reinforce_brier_loss(logits, rows, *, samples=8, std=0.5):
    if samples < 2 or not math.isfinite(std) or std <= 0:
        raise ValueError("REINFORCE requires >=2 samples and positive finite std")
    losses, rewards = [], []
    for values, row in zip(logits, rows):
        distribution = torch.distributions.Normal(values, std)
        # A sampled action must be detached. rsample() here would cancel the
        # score-function gradient or incorrectly differentiate the reward.
        actions = distribution.sample((samples,))
        target = values.new_tensor(row["target"])
        reward = -(actions.softmax(-1) - target).square().sum(-1)
        baseline = (reward.sum() - reward) / (samples - 1)
        advantage = (reward - baseline).detach()
        losses.append(-(advantage * distribution.log_prob(actions).sum(-1)).mean())
        rewards.append(reward.mean())
    return torch.stack(losses).mean(), torch.stack(rewards).mean().detach()


def _nll(logits, rows, temperature=1.0):
    terms = []
    for values, row in zip(logits, rows):
        values = np.asarray(values, dtype=np.float64) / temperature
        shifted = values - values.max()
        logp = shifted - np.log(np.exp(shifted).sum())
        terms.append(-float(np.dot(row["target"], logp)))
    return float(np.mean(terms))


def fit_typed_temperature(logits, rows):
    left, right, ratio = math.log(0.01), math.log(100.0), (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = _nll(logits, rows, math.exp(x1)), _nll(logits, rows, math.exp(x2))
    for _ in range(60):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _nll(logits, rows, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _nll(logits, rows, math.exp(x2))
    t = min(
        [1.0, 0.01, 100.0, math.exp((left + right) / 2)],
        key=lambda v: _nll(logits, rows, v),
    )
    return {
        "temperature": t,
        "samples": len(rows),
        "nll_before": _nll(logits, rows),
        "nll_after": _nll(logits, rows, t),
        "method": "held_out_scalar_temperature_soft_nll",
    }


def typed_metrics(logits, rows, calibration=None):
    calibration = calibration or {}
    if not rows:
        return {"samples": 0}
    nll, brier, confidences, correctness, hard_correct, ordinal = [], [], [], [], [], []
    reliability = []
    diagnostics = {}
    score_mae, score_expected_mae, score_within_one = [], [], []
    for values, row in zip(logits, rows):
        t = calibration.get(scope_key(row), {}).get("temperature", 1.0)
        values = np.asarray(values, dtype=np.float64) / t
        values -= values.max()
        logp = values - np.log(np.exp(values).sum())
        probs, target = np.exp(logp), np.asarray(row["target"])
        options = candidates(row)
        # Match the prediction API independently of candidate presentation order.
        selected = min(
            range(len(options)), key=lambda index: (-probs[index], options[index]["id"])
        )
        nll.append(-float(target @ logp))
        brier.append(float(np.square(probs - target).sum()))
        confidences.append(float(probs[selected]))
        correctness.append(float(target[selected]))
        scope = scope_key(row)
        identifiers = sorted(item["id"] for item in options)
        entry = diagnostics.setdefault(
            scope,
            {
                "candidate_ids": identifiers,
                "prediction_counts": dict.fromkeys(identifiers, 0),
                "hard_target_counts": dict.fromkeys(identifiers, 0),
                "confusion": np.zeros((len(identifiers), len(identifiers)), dtype=int),
            },
        )
        selected_id = options[selected]["id"]
        entry["prediction_counts"][selected_id] += 1
        if np.count_nonzero(target) == 1:
            hard_correct.append(float(target[selected] == 1))
            truth = int(target.argmax())
            truth_id = options[truth]["id"]
            entry["hard_target_counts"][truth_id] += 1
            entry["confusion"][
                identifiers.index(truth_id), identifiers.index(selected_id)
            ] += 1
            if row["type"] == "score":
                score_mae.append(
                    abs(options[selected]["value"] - options[truth]["value"])
                )
                mean_score = sum(
                    option["value"] * p for option, p in zip(options, probs)
                )
                score_expected_mae.append(abs(mean_score - options[truth]["value"]))
                score_within_one.append(abs(selected - truth) <= 1)
        if row["type"] == "score":
            ordinal.append(
                float(np.square(np.cumsum(probs)[:-1] - np.cumsum(target)[:-1]).mean())
            )
    confidence, accuracy = np.asarray(confidences), np.asarray(correctness)
    buckets = np.minimum((confidence * 15).astype(int), 14)
    ece = 0.0
    for index in range(15):
        chosen = buckets == index
        if chosen.any():
            avg_p, avg_truth = (
                float(confidence[chosen].mean()),
                float(accuracy[chosen].mean()),
            )
            ece += float(chosen.mean()) * abs(avg_p - avg_truth)
            reliability.append(
                {
                    "bin": index,
                    "samples": int(chosen.sum()),
                    "mean_probability": avg_p,
                    "mean_target_mass": avg_truth,
                }
            )
    result = {
        "samples": len(rows),
        "nll": float(np.mean(nll)),
        "multiclass_brier": float(np.mean(brier)),
        "accuracy": float(np.mean(hard_correct)) if hard_correct else None,
        "hard_label_samples": len(hard_correct),
        "expected_accuracy": float(accuracy.mean()),
        "top_label_ece": ece,
        "reliability": reliability,
        "calibration_in_scope_samples": sum(scope_key(r) in calibration for r in rows),
        "selective_risk": [
            {
                "threshold": t,
                "coverage": float((confidence >= t).mean()),
                "expected_error": float(1 - accuracy[confidence >= t].mean())
                if (confidence >= t).any()
                else None,
            }
            for t in (0.5, 0.7, 0.9, 0.95)
        ],
    }
    if ordinal:
        result["ranked_probability_score"] = float(np.mean(ordinal))
    if score_mae:
        result["score"] = {
            "hard_label_samples": len(score_mae),
            "mae_argmax_value": float(np.mean(score_mae)),
            "mae_expected_score": float(np.mean(score_expected_mae)),
            "within_one_level_accuracy": float(np.mean(score_within_one)),
        }
    result["class_diagnostics"] = {}
    for scope, entry in diagnostics.items():
        matrix = entry.pop("confusion")
        recalls = {}
        for index, identifier in enumerate(entry["candidate_ids"]):
            count = entry["hard_target_counts"][identifier]
            recalls[identifier] = float(matrix[index, index] / count) if count else None
        result["class_diagnostics"][scope] = {
            **entry,
            "recall_by_label": recalls,
            "confusion": {
                "rows": "gold",
                "columns": "prediction",
                "matrix": matrix.tolist(),
            },
        }
    return result


@torch.inference_mode()
def collect_logits(model, rows, batch_size):
    model.eval()
    return [
        v.cpu().numpy().astype(np.float64)
        for start in range(0, len(rows), batch_size)
        for v in model(rows[start : start + batch_size])
    ]


def _json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_typed(
    dataset,
    output,
    *,
    model_config=None,
    config=None,
    device="auto",
    initial_model=None,
):
    config, model_config = (
        config or TypedTrainingConfig(),
        model_config or TypedDecisionConfig(),
    )
    _, splits, hashes = read_typed_dataset(dataset)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    model = (
        TypedDecisionModel.from_pretrained(initial_model, device=device)
        if initial_model
        else TypedDecisionModel(model_config).to(device)
    )
    if initial_model and model.config != model_config:
        raise ValueError("initial model config must match the requested typed config")
    model.calibration = {}  # Updating any weights invalidates fitted temperatures.
    if (
        model.config.backbone == "hf"
        and not model.config.freeze_backbone
        and getattr(model.backbone, "supports_gradient_checkpointing", False)
    ):
        model.backbone.gradient_checkpointing_enable()
    parameters = [p for p in model.parameters() if p.requires_grad]
    # CPU master parameters retain fp32 Adam states while the complete backbone
    # forward/backward remains on its model device. No layers are frozen here.
    masters = (
        [torch.nn.Parameter(p.detach().float().cpu().clone()) for p in parameters]
        if config.optimizer_device == "cpu"
        else parameters
    )
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    # Preserve the legacy flat parameter list when no group rates were requested.
    optimizer_input = masters
    optimizer_groups = [
        {
            "name": "all_trainable",
            "learning_rate": config.learning_rate,
            "tensor_count": len(parameters),
            "parameter_count": sum(p.numel() for p in parameters),
        }
    ]
    if (
        config.backbone_learning_rate is not None
        or config.head_learning_rate is not None
    ):
        optimizer_input, optimizer_groups = [], []
        for name, backbone, rate in (
            ("backbone", True, config.backbone_learning_rate),
            ("head", False, config.head_learning_rate),
        ):
            indices = [
                i
                for i, p in enumerate(parameters)
                if (id(p) in backbone_ids) == backbone
            ]
            if not indices:
                continue
            rate = config.learning_rate if rate is None else rate
            optimizer_input.append(
                {"params": [masters[i] for i in indices], "lr": rate, "name": name}
            )
            optimizer_groups.append(
                {
                    "name": name,
                    "learning_rate": rate,
                    "tensor_count": len(indices),
                    "parameter_count": sum(parameters[i].numel() for i in indices),
                }
            )
    # Avoid foreach's additional parameter-sized CUDA temporaries on shared GPUs.
    optimizer = torch.optim.AdamW(
        optimizer_input,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    # Preflight every formatted input before creating an output or updating weights.
    initial = collect_logits(model, splits["validation"], config.batch_size)
    preflight_splits = ("train", "calibration") + (
        ("test",) if config.evaluate_test else ()
    )
    for name in preflight_splits:
        for row in splits[name]:
            model._tokens([row["state"]])
            for item in candidates(row):
                model._tokens(
                    [
                        candidate_query_text(
                            row,
                            item,
                            instance_question_in_query=model.config.instance_question_in_query,
                        )
                    ]
                )
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _json(
        destination / "run_config.json",
        {
            "training": asdict(config),
            "model": model_config.to_dict(),
            "device": device,
            "dataset_hashes": hashes,
            "initial_model": str(initial_model) if initial_model else None,
            "optimizer_groups": optimizer_groups,
            "test_inputs_preflighted": config.evaluate_test,
        },
    )
    best_nll, best_epoch = _nll(initial, splits["validation"]), 0
    selected_validation = initial
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    probe = next(
        (
            (name, p)
            for name, p in model.backbone.named_parameters()
            if p.requires_grad and p.numel() <= 4096
        ),
        None,
    )
    probe_before = probe[1].detach().cpu().clone() if probe else None
    history = []
    stopped_early, epochs_without_improvement = False, 0
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        warmup = epoch <= config.head_warmup_epochs and not model.config.freeze_backbone
        phase = (
            "head_warmup"
            if warmup
            else "head_only"
            if model.config.freeze_backbone
            else "joint"
        )
        model.backbone.requires_grad_(not model.config.freeze_backbone and not warmup)
        model.train()
        if warmup:
            model.backbone.eval()
        indices = torch.randperm(len(splits["train"]), generator=generator).tolist()
        total, reward_total = 0.0, 0.0
        for start in range(0, len(indices), config.batch_size):
            batch = [
                splits["train"][i] for i in indices[start : start + config.batch_size]
            ]
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = config.supervised_weight * supervised_loss(
                logits,
                batch,
                brier_weight=config.brier_weight,
                ordinal_weight=config.ordinal_weight,
            )
            if config.method == "reinforce_brier":
                policy_loss, reward = reinforce_brier_loss(
                    logits, batch, samples=config.rl_samples, std=config.rl_std
                )
                loss = loss + config.rl_weight * policy_loss
                reward_total += float(reward) * len(batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite typed training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if config.optimizer_device == "cpu":
                for parameter, master in zip(parameters, masters):
                    master.grad = (
                        parameter.grad.detach().float().cpu()
                        if parameter.grad is not None
                        else None
                    )
            optimizer.step()
            if config.optimizer_device == "cpu":
                with torch.no_grad():
                    for parameter, master in zip(parameters, masters):
                        parameter.copy_(master)
            total += float(loss.detach()) * len(batch)
            step = start // config.batch_size + 1
            steps = math.ceil(len(indices) / config.batch_size)
            if step == 1 or step % 25 == 0 or step == steps:
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "training_phase": phase,
                            "epoch": epoch,
                            "step": step,
                            "steps": steps,
                            "samples_processed": start + len(batch),
                            "mean_train_objective": total / (start + len(batch)),
                            "epoch_elapsed_seconds": time.perf_counter()
                            - epoch_started,
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
        train_logits = collect_logits(model, splits["train"], config.batch_size)
        validation = collect_logits(model, splits["validation"], config.batch_size)
        nll = _nll(validation, splits["validation"])
        improved = nll < best_nll
        if improved:
            best_nll, best_epoch = nll, epoch
            selected_validation = validation
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        record = {
            "epoch": epoch,
            "phase": phase,
            "train_objective": total / len(indices),
            "validation_nll": nll,
            "train_by_type": {},
            "validation_by_type": {},
            "backbone_probe_delta_l2": float(
                (probe[1].detach().cpu() - probe_before).norm()
            )
            if probe
            else 0.0,
        }
        for kind in TYPES:
            chosen = [i for i, row in enumerate(splits["train"]) if row["type"] == kind]
            record["train_by_type"][kind] = typed_metrics(
                [train_logits[i] for i in chosen],
                [splits["train"][i] for i in chosen],
            )
            chosen = [
                i for i, row in enumerate(splits["validation"]) if row["type"] == kind
            ]
            record["validation_by_type"][kind] = typed_metrics(
                [validation[i] for i in chosen],
                [splits["validation"][i] for i in chosen],
            )
        if config.method == "reinforce_brier":
            record["sampled_negative_brier_reward"] = reward_total / len(indices)
        history.append(record)
        with (destination / "training_history.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        # Do not end a requested head warmup before the backbone phase can start.
        if not warmup:
            epochs_without_improvement = (
                0 if improved else epochs_without_improvement + 1
            )
            if (
                config.early_stopping_patience
                and epochs_without_improvement >= config.early_stopping_patience
            ):
                stopped_early = epoch < config.epochs
                break
    update_delta = (
        float((probe[1].detach().cpu() - probe_before).norm()) if probe else 0.0
    )
    model.backbone.requires_grad_(not model.config.freeze_backbone)
    model.load_state_dict(best_state)
    checkpoint_delta = (
        float((probe[1].detach().cpu() - probe_before).norm()) if probe else 0.0
    )
    optimizer.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    del optimizer, best_state, masters, optimizer_input
    model.eval()
    calibration_logits = collect_logits(model, splits["calibration"], config.batch_size)
    groups = {}
    for i, row in enumerate(splits["calibration"]):
        groups.setdefault(scope_key(row), []).append(i)
    for scope, indices in groups.items():
        model.calibration[scope] = {
            **fit_typed_temperature(
                [calibration_logits[i] for i in indices],
                [splits["calibration"][i] for i in indices],
            ),
            "split_sha256": hashes["calibration"],
            "scope": "Exact question/type/candidate schema on this calibration distribution",
        }
    model.metadata = {
        "supported_types": list(TYPES),
        "dataset_hashes": hashes,
        "training_method": config.method,
        "scope": "Independent SaddleLLM typed decision research; not Jev or RLCD reproduction",
    }
    model.save_pretrained(destination / "model")
    test = (
        collect_logits(model, splits["test"], config.batch_size)
        if config.evaluate_test
        else None
    )
    training_tasks = {scope_key(row) for row in splits["train"]}
    report = {
        "schema_version": 2,
        "counts": {name: len(splits[name]) for name in SPLITS},
        "dataset_hashes": hashes,
        "selection": {
            "best_epoch": best_epoch,
            "best_validation_nll": best_nll,
            "initial_validation_nll": _nll(initial, splits["validation"]),
            "selection_metric": "validation_nll",
            "history": history,
        },
        "test_used_for_selection": False,
        "calibration_used_for_selection": False,
        "validation": {
            "trained_uncalibrated": typed_metrics(
                selected_validation, splits["validation"]
            ),
            "trained_calibrated": typed_metrics(
                selected_validation, splits["validation"], model.calibration
            ),
        },
        "test_evaluated": config.evaluate_test,
        "test": {
            "trained_uncalibrated": typed_metrics(test, splits["test"]),
            "trained_calibrated": typed_metrics(
                test, splits["test"], model.calibration
            ),
        }
        if config.evaluate_test
        else {},
        "test_by_type": {},
        "test_by_task_scope": {},
        "training": asdict(config),
        "optimizer_groups": optimizer_groups,
        "epochs_completed": len(history),
        "early_stopping": {
            "patience": config.early_stopping_patience,
            "stopped_early": stopped_early,
            "epochs_without_improvement": epochs_without_improvement,
            "head_warmup_excluded_from_patience": True,
            "reason": "validation_nll_no_improvement" if stopped_early else None,
        },
        "backbone_bundled": True,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "backbone_trainable_parameters": sum(
            p.numel() for p in model.backbone.parameters() if p.requires_grad
        ),
        "backbone_update_probe": {
            "parameter": probe[0] if probe else None,
            "optimizer_delta_l2": update_delta,
            "selected_checkpoint_delta_l2": checkpoint_delta,
        },
        "rlcd_reproduction": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if config.evaluate_test:
        for kind in TYPES:
            indices = [i for i, row in enumerate(splits["test"]) if row["type"] == kind]
            report["test_by_type"][kind] = typed_metrics(
                [test[i] for i in indices],
                [splits["test"][i] for i in indices],
                model.calibration,
            )
        for name, seen in (("seen", True), ("unseen", False)):
            indices = [
                i
                for i, row in enumerate(splits["test"])
                if (scope_key(row) in training_tasks) == seen
            ]
            report["test_by_task_scope"][name] = typed_metrics(
                [test[i] for i in indices],
                [splits["test"][i] for i in indices],
                model.calibration,
            )
    reloaded = TypedDecisionModel.from_pretrained(destination / "model", device=device)
    reload_split = "test" if config.evaluate_test else "validation"
    sample = splits[reload_split][: min(8, len(splits[reload_split]))]
    with torch.inference_mode():
        for a, b in zip(model(sample), reloaded(sample)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    if model.predict(sample) != reloaded.predict(sample):
        raise RuntimeError("reloaded typed predictions differ")
    report["artifact_reload_verified"] = True
    report["artifact_reload_split"] = reload_split
    _json(destination / "report.json", report)
    print(
        json.dumps(
            {
                "phase": "completed",
                "output": str(destination),
                "best_epoch": best_epoch,
                "test_evaluated": config.evaluate_test,
                "test": report["test"].get("trained_calibrated"),
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backbone", default="native", choices=("native", "hf"))
    parser.add_argument("--backbone-name-or-path")
    parser.add_argument("--initial-model")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--query-pooling",
        choices=("mean", "last_token"),
        default=TypedDecisionConfig().query_pooling,
    )
    parser.add_argument(
        "--instance-question-in-query",
        action=argparse.BooleanOptionalAction,
        default=TypedDecisionConfig().instance_question_in_query,
    )
    for name in (
        "hidden_size",
        "num_layers",
        "num_heads",
        "max_length",
        "max_candidates",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(TypedDecisionConfig(), name),
        )
    for name, default in asdict(TypedTrainingConfig()).items():
        option = "--" + name.replace("_", "-")
        if type(default) is bool:
            parser.add_argument(
                option, action=argparse.BooleanOptionalAction, default=default
            )
        else:
            parser.add_argument(
                option,
                type=float if default is None else type(default),
                default=default,
            )
    values = vars(parser.parse_args())
    model_config = TypedDecisionConfig(
        **{k: values.pop(k) for k in TypedDecisionConfig.__dataclass_fields__}
    )
    config = TypedTrainingConfig(
        **{k: values.pop(k) for k in TypedTrainingConfig.__dataclass_fields__}
    )
    train_typed(**values, model_config=model_config, config=config)


if __name__ == "__main__":
    main()
