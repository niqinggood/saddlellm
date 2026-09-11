#!/usr/bin/env python3
"""Evaluation utilities for SaddleLLM.

The module is usable both as a Python API and as a CLI:

    Evaluator().evaluate(model_path="./model", dataset="wikitext")
    python -m saddlellm.evaluation.LLModelEvalute --model-path ./model --dataset-path data.jsonl
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


class Evaluator:
    """LLM evaluator with API and CLI compatibility."""

    def __init__(self, args: Optional[Any] = None, **defaults):
        self.args = args
        self.defaults = defaults

    def evaluate(
        self,
        model_path: str,
        dataset: str,
        dataset_config: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        max_samples: int = 1000,
        task_type: str = "generation",
        split: str = "validation",
        text_column: Optional[str] = None,
        target_column: Optional[str] = None,
        max_length: int = 512,
        max_new_tokens: int = 128,
        batch_size: int = 4,
        cpu: bool = False,
        output_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> Dict[str, Any]:
        metrics = metrics or ["perplexity"]
        device = "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        result = {
            "model": model_path,
            "task_type": task_type,
            "dataset": dataset,
            "dataset_config": dataset_config,
            "metrics": {},
            "error": None,
        }

        try:
            records = self._load_records(dataset, dataset_config, split, max_samples)
            if not records:
                raise ValueError("evaluation dataset is empty")

            model, tokenizer = self._load_model(model_path, task_type, device, trust_remote_code)
            text_column = text_column or self._infer_text_column(records[0])

            if task_type == "classification":
                result["metrics"].update(
                    self._evaluate_classification(
                        model, tokenizer, records, text_column, target_column or "target",
                        metrics, device, max_length, batch_size
                    )
                )
            else:
                result["metrics"].update(
                    self._evaluate_generation(
                        model, tokenizer, records, text_column, target_column,
                        metrics, device, max_length, max_new_tokens, batch_size
                    )
                )

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "evaluation_results.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info("Evaluation results saved to %s", output_file)
            return result
        except Exception as exc:
            logger.exception("Evaluation failed")
            result["error"] = str(exc)
            return result

    def run(self) -> Dict[str, Any]:
        if self.args is None:
            raise ValueError("Evaluator.run() requires argparse-style args")
        args = self.args
        return self.evaluate(
            model_path=args.model_path,
            dataset=args.dataset_path,
            dataset_config=getattr(args, "dataset_config", None),
            metrics=args.metrics,
            max_samples=args.max_samples,
            task_type=args.task_type,
            split=getattr(args, "split", "validation"),
            text_column=getattr(args, "text_column", None),
            target_column=getattr(args, "target_column", None),
            max_length=args.max_length,
            max_new_tokens=getattr(args, "max_new_tokens", 128),
            batch_size=args.batch_size,
            cpu=args.cpu,
            output_dir=args.output_dir,
        )

    def _load_model(self, model_path: str, task_type: str, device: str, trust_remote_code: bool):
        logger.info("Loading model: %s", model_path)
        dtype = torch.float16 if device == "cuda" else torch.float32

        if task_type == "classification":
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            ).to(device)
        else:
            from ..models.ModelLoader import load_model_and_tokenizer

            model, tokenizer = load_model_and_tokenizer(
                model_path,
                device=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

    def _load_records(
        self,
        dataset: str,
        dataset_config: Optional[str],
        split: str,
        max_samples: int,
    ) -> List[Dict[str, Any]]:
        path = Path(dataset)
        if path.exists():
            records = self._load_local_records(path)
        else:
            try:
                ds = load_dataset(dataset, dataset_config, split=split)
            except Exception:
                ds = load_dataset(dataset, dataset_config, split="test")
            records = self._dataset_to_records(ds, max_samples)
        return records[:max_samples]

    def _load_local_records(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        with open(path, "r", encoding="utf-8") as f:
            if suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            if suffix == ".json":
                data = json.load(f)
                if isinstance(data, dict):
                    for key in ("data", "train", "validation", "test"):
                        if isinstance(data.get(key), list):
                            return data[key]
                    return [data]
                return data
            if suffix in {".txt", ".md"}:
                return [{"text": line.strip()} for line in f if line.strip()]
        raise ValueError(f"Unsupported dataset file format: {path.suffix}")

    def _dataset_to_records(self, ds, max_samples: int) -> List[Dict[str, Any]]:
        if hasattr(ds, "select") and hasattr(ds, "__len__"):
            ds = ds.select(range(min(len(ds), max_samples)))
        records = []
        for item in ds:
            records.append(dict(item))
            if len(records) >= max_samples:
                break
        return records

    def _infer_text_column(self, sample: Dict[str, Any]) -> str:
        for key in ("text", "input", "prompt", "question", "instruction"):
            if key in sample:
                return key
        for key, value in sample.items():
            if isinstance(value, str):
                return key
        raise ValueError("Cannot infer text column from evaluation sample")

    def _evaluate_generation(
        self,
        model,
        tokenizer,
        records: List[Dict[str, Any]],
        text_column: str,
        target_column: Optional[str],
        metrics: List[str],
        device: str,
        max_length: int,
        max_new_tokens: int,
        batch_size: int,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        texts = [str(r.get(text_column, "")) for r in records if r.get(text_column)]

        if "perplexity" in metrics:
            ppl = self._compute_perplexity(model, tokenizer, texts, device, max_length, batch_size)
            results["perplexity"] = {
                "score": ppl,
                "interpretation": "Perplexity, lower is better",
                "higher_is_better": False,
            }

        if target_column and any(m in metrics for m in ("rouge", "bleu")):
            predictions = self._generate_predictions(
                model, tokenizer, texts[: min(100, len(texts))], device, max_length, max_new_tokens
            )
            references = [str(r.get(target_column, "")) for r in records[: len(predictions)]]
            results.update(self._compute_text_metrics(predictions, references, metrics))

        return results

    def _compute_perplexity(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: str,
        max_length: int,
        batch_size: int,
    ) -> float:
        losses = []
        token_counts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = encoded["input_ids"].clone()
            labels[encoded.get("attention_mask", torch.ones_like(labels)) == 0] = -100
            with torch.no_grad():
                output = model(**encoded, labels=labels)
            valid_tokens = (labels != -100).sum().item()
            if valid_tokens > 1:
                losses.append(float(output.loss) * valid_tokens)
                token_counts.append(valid_tokens)
        if not losses:
            return float("inf")
        mean_loss = sum(losses) / max(1, sum(token_counts))
        return round(math.exp(min(mean_loss, 20)), 4)

    def _generate_predictions(
        self,
        model,
        tokenizer,
        texts: List[str],
        device: str,
        max_length: int,
        max_new_tokens: int,
    ) -> List[str]:
        predictions = []
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(output[0], skip_special_tokens=True)
            predictions.append(generated[len(text):].strip() or generated.strip())
        return predictions

    def _compute_text_metrics(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        try:
            import evaluate
        except Exception as exc:
            logger.warning("evaluate package unavailable, skipping text metrics: %s", exc)
            return results

        if "rouge" in metrics:
            rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
            results["rouge"] = {
                "score": rouge.get("rougeL", 0.0),
                "interpretation": "ROUGE-L, higher is better",
                "higher_is_better": True,
            }
        if "bleu" in metrics:
            bleu = evaluate.load("bleu").compute(predictions=predictions, references=[[r] for r in references])
            results["bleu"] = {
                "score": bleu.get("bleu", 0.0),
                "interpretation": "BLEU, higher is better",
                "higher_is_better": True,
            }
        return results

    def _evaluate_classification(
        self,
        model,
        tokenizer,
        records: List[Dict[str, Any]],
        text_column: str,
        target_column: str,
        metrics: List[str],
        device: str,
        max_length: int,
        batch_size: int,
    ) -> Dict[str, Dict[str, Any]]:
        if "accuracy" not in metrics:
            return {}

        clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        total = 0
        correct = 0
        for record in records:
            if text_column not in record or target_column not in record:
                continue
            pred = clf(str(record[text_column]), truncation=True, max_length=max_length)[0]["label"]
            correct += int(str(pred) == str(record[target_column]))
            total += 1
        score = correct / total if total else 0.0
        return {
            "accuracy": {
                "score": round(score, 4),
                "interpretation": "Accuracy, higher is better",
                "higher_is_better": True,
            }
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a language model")
    parser.add_argument("--model-path", required=True, help="Path or name of the model")
    parser.add_argument("--dataset-path", required=True, help="Local dataset path or HuggingFace dataset name")
    parser.add_argument("--dataset-config", default=None, help="HuggingFace dataset config")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--task-type", default="generation", choices=["classification", "generation"])
    parser.add_argument("--metrics", nargs="+", default=["perplexity"],
                        choices=["accuracy", "perplexity", "bleu", "rouge"])
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    result = Evaluator(args).run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
