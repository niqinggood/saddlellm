"""Command line interface for the SaddleLLM training factory."""
import argparse
import json
import os
import sys
from typing import Optional

from . import __version__


def _configure_windows_stdio() -> None:
    """Keep CLI JSON and non-ASCII labels readable through Windows pipes."""
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Embedded hosts may expose a stream that cannot be reconfigured.
            pass


def main(argv: Optional[list] = None) -> int:
    _configure_windows_stdio()
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv and effective_argv[0] == "experiment":
        from .experiments.ProjectRunner import main as run_project

        return run_project(effective_argv[1:])
    parser = argparse.ArgumentParser(
        prog="saddlellm",
        description="SaddleLLM training factory CLI",
        epilog="New here? Run: saddlellm quickstart",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("experiment", help="Run a task-owned research project; see saddlellm experiment --help")

    quickstart_p = sub.add_parser(
        "quickstart",
        help="Create a beginner-friendly post-training project with sample data",
    )
    quickstart_p.add_argument("directory", nargs="?", default="./saddlellm-quickstart")
    quickstart_p.add_argument(
        "--stages",
        default="sft",
        help="Comma-separated stages to prepare (sft or sft,preference)",
    )
    quickstart_p.add_argument(
        "--preference-method",
        default="dpo",
        choices=["dpo", "kto"],
        help="Method used when preference is listed in --stages",
    )
    quickstart_p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local model path or Hugging Face model id",
    )
    quickstart_p.add_argument(
        "--data",
        "--sft-data",
        dest="sft_data",
        default=None,
        help="Existing SFT JSONL/JSON/CSV file; otherwise sample data is created",
    )
    quickstart_p.add_argument(
        "--preference-data",
        default=None,
        help="Existing DPO/KTO data file; otherwise sample data is created",
    )
    quickstart_p.add_argument(
        "--qlora",
        action="store_true",
        help="Enable QLoRA (requires CUDA and bitsandbytes); defaults to portable LoRA",
    )
    quickstart_p.add_argument(
        "--force",
        action="store_true",
        help="Replace only quickstart-owned files that already exist",
    )
    quickstart_p.add_argument("--output", default=None, help="Optional JSON result path")

    import_p = sub.add_parser("import-data", help="Import a dataset manifest or LLaMA-Factory dataset_info.json")
    import_p.add_argument("manifest")
    import_p.add_argument("--root-dir", default="./llm_factory")
    import_p.add_argument("--dataset-dir", default=None)
    import_p.add_argument("--selected", default=None, help="Comma-separated dataset names")
    import_p.add_argument("--role", default=None)

    plan_p = sub.add_parser("plan", help="Generate launch artifacts from a training config")
    plan_p.add_argument("config")
    plan_p.add_argument("--launch-script", default=None)
    plan_p.add_argument("--launch-plan", default=None)
    plan_p.add_argument(
        "--parallel",
        choices=["auto", "single", "ddp", "fsdp", "deepspeed-zero2", "deepspeed-zero3"],
        default=None,
        help="Override distributed.strategy while generating the launch plan",
    )
    plan_p.add_argument("--num-gpus", type=int, default=None, help="GPU workers per node")
    plan_p.add_argument("--num-nodes", type=int, default=None)
    plan_p.add_argument(
        "--launcher",
        choices=["auto", "accelerate", "torchrun"],
        default=None,
    )

    train_p = sub.add_parser(
        "train",
        help="Train from a config file or common single-stage command options",
    )
    train_p.add_argument(
        "config",
        nargs="?",
        help="YAML/JSON config path, or ordered stages such as 'sft,kto'",
    )
    train_p.add_argument(
        "--stage",
        default=None,
        help=(
            "Ordered comma-separated direct stages: pretrain,sft,dpo,orpo,kto,ppo; "
            "decision trains Choice or typed decisions with held-out calibration; "
            "or auto-posttrain to inspect one dataset and choose SFT/DPO/KTO. "
            "Stages run strictly from left to right"
        ),
    )
    train_p.add_argument("--model", default=None, help="Base model path or Hugging Face model id")
    train_p.add_argument(
        "--data",
        default=None,
        help="Data for the first/only direct stage; use stage-specific data options for later stages",
    )
    train_p.add_argument("--sft-data", default=None, help="SFT training data for a multi-stage command")
    train_p.add_argument("--pretrain-data", default=None, help="Continuous-text corpus for a pretrain stage")
    train_p.add_argument("--dpo-data", default=None, help="DPO chosen/rejected training data")
    train_p.add_argument("--orpo-data", default=None, help="ORPO chosen/rejected training data")
    train_p.add_argument("--kto-data", default=None, help="KTO completion/label training data")
    train_p.add_argument("--ppo-data", default=None, help="PPO prompt/rollout data (PPO execution is not yet wired)")
    train_p.add_argument("--output-dir", default=None, help="Training output directory")
    train_p.add_argument(
        "--pretrain-mode",
        choices=["scratch", "continue"],
        default=None,
        help="Pretrain from scratch or continue an existing model; defaults to continue",
    )
    train_p.add_argument(
        "--model-config",
        default=None,
        help="Registered model specification for scratch pretraining",
    )
    train_p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or model id; required for scratch pretraining",
    )
    train_p.add_argument(
        "--backend",
        choices=["hf", "saddle"],
        default=None,
        help="Model backend; defaults to hf for continue and saddle for scratch",
    )
    train_p.add_argument(
        "--data-type",
        choices=["local", "huggingface"],
        default=None,
        help="Pretraining corpus source type; defaults to local",
    )
    train_p.add_argument("--text-column", default=None, help="Pretraining text field; defaults to text")
    train_p.add_argument("--min-text-length", type=int, default=None)
    train_p.add_argument("--quality-min-score", type=float, default=None)
    train_p.add_argument(
        "--dedup-method",
        choices=["simhash", "minhash", "exact", "none"],
        default=None,
    )
    train_p.add_argument("--global-batch-size", type=int, default=None)
    train_p.add_argument("--min-learning-rate", type=float, default=None)
    train_p.add_argument("--streaming", action="store_true", help="Stream the pretraining corpus")
    train_p.add_argument("--no-pack-sequences", action="store_true")
    train_p.add_argument("--no-append-eos", action="store_true")
    train_p.add_argument("--epochs", type=int, default=None)
    train_p.add_argument("--learning-rate", type=float, default=None)
    train_p.add_argument("--batch-size", type=int, default=None, help="Per-device batch size")
    train_p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    train_p.add_argument("--max-seq-length", type=int, default=None)
    train_p.add_argument("--max-steps", type=int, default=None)
    train_p.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Resume one direct training stage from PATH; omit PATH to use the "
            "latest checkpoint in its output directory"
        ),
    )
    train_p.add_argument("--warmup-steps", type=int, default=None)
    train_p.add_argument("--save-steps", type=int, default=None)
    train_p.add_argument("--eval-steps", type=int, default=None)
    train_p.add_argument("--logging-steps", type=int, default=None)
    train_p.add_argument("--validation-split", type=float, default=None)
    train_p.add_argument(
        "--validation-data",
        default=None,
        help="Explicit validation JSON/JSONL/CSV for one SFT/DPO/ORPO/KTO stage",
    )
    train_p.add_argument(
        "--method",
        choices=["lora", "qlora", "full"],
        default=None,
        help="Parameter-update method for direct stages; defaults to lora",
    )
    train_p.add_argument("--lora-r", type=int, default=None)
    train_p.add_argument("--lora-alpha", type=int, default=None)
    train_p.add_argument("--lora-dropout", type=float, default=None)
    train_p.add_argument(
        "--lora-target-modules",
        default=None,
        help=(
            "Comma-separated module names, for example q_proj,v_proj, or "
            "all-linear"
        ),
    )
    train_p.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default=None,
    )
    train_p.add_argument("--response-template", default=None)
    train_p.add_argument("--replay-data", default=None, help="General/history SFT data mixed into training after validation splitting")
    train_p.add_argument("--replay-ratio", type=float, default=None, help="Replay fraction of final training samples, in [0, 1)")
    train_p.add_argument("--replay-count", type=int, default=None, help="Explicit replay sample count instead of --replay-ratio")
    train_p.add_argument("--reference-model", default=None, help="SFT capability-retention teacher; defaults to starting model when KL is enabled")
    train_p.add_argument("--retention-kl-weight", type=float, default=None, help="Nonnegative reference KL weight; defaults to disabled")
    train_p.add_argument("--retention-temperature", type=float, default=None, help="Reference KL temperature; defaults to 1")
    train_p.add_argument("--seed", type=int, default=None)
    train_p.add_argument("--local-files-only", action="store_true")
    train_p.add_argument("--encoder-device", choices=["auto", "cpu", "cuda"], default=None, help="Decision stage text encoder device")
    train_p.add_argument("--projection-rank", type=int, default=None, help="Decision head residual projection rank")
    train_p.add_argument("--encoder-batch-size", type=int, default=None, help="Decision stage text encoding batch size")
    train_p.add_argument("--cpu-threads", type=int, default=None, help="Decision head CPU threads")
    train_p.add_argument("--weight-decay", type=float, default=None, help="Decision head AdamW weight decay")
    train_p.add_argument("--decision-type", choices=["choice", "typed"], default=None, help="Decision contract; inferred from dataset.json when available")
    train_p.add_argument("--decision-backbone", choices=["native", "hf"], default=None, help="Typed decision backbone; native without --model, otherwise hf")
    train_p.add_argument("--decision-method", choices=["supervised", "reinforce_brier"], default=None, help="Typed decision training objective")
    train_p.add_argument("--optimizer-device", choices=["model", "cpu"], default=None, help="Typed Adam state placement; cpu reduces GPU memory")
    train_p.add_argument("--query-pooling", choices=["mean", "last_token"], default=None, help="Typed candidate query token pooling")
    train_p.add_argument("--instance-question-in-query", action=argparse.BooleanOptionalAction, default=None, help="Include the typed record's optional instance_question in its candidate queries")
    train_p.add_argument("--backbone-learning-rate", type=float, default=None, help="Typed pretrained backbone learning rate")
    train_p.add_argument("--head-learning-rate", type=float, default=None, help="Typed new cross-attention and decision head learning rate")
    train_p.add_argument("--head-warmup-epochs", type=int, default=None, help="Typed epochs that train new layers before updating the backbone")
    train_p.add_argument("--early-stopping-patience", type=int, default=None, help="Typed validation NLL patience; 0 disables early stopping")
    train_p.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None, help="Typed final test evaluation; disable while developing configurations")
    train_p.add_argument("--decision-hidden-size", type=int, default=None, help="Typed hidden width; inferred from local HF config")
    train_p.add_argument("--decision-heads", type=int, default=None, help="Typed cross attention heads")
    train_p.add_argument("--trust-remote-code", action="store_true")
    train_p.add_argument("--no-gradient-checkpointing", action="store_true")
    train_p.add_argument(
        "--parallel",
        choices=["auto", "single", "ddp", "fsdp", "deepspeed-zero2", "deepspeed-zero3"],
        default=None,
        help=(
            "Resolve and launch the requested parallel strategy; auto uses all "
            "visible GPUs without changing full/LoRA/QLoRA semantics"
        ),
    )
    train_p.add_argument("--num-gpus", type=int, default=None, help="GPU workers per node")
    train_p.add_argument("--num-nodes", type=int, default=None)
    train_p.add_argument(
        "--launcher",
        choices=["auto", "accelerate", "torchrun"],
        default=None,
        help="Process launcher; backend selection remains controlled by --parallel",
    )
    train_p.add_argument("--dry-run", action="store_true")
    train_p.add_argument("--debug", action="store_true", help="Show full traceback instead of a structured error")

    distill_p = sub.add_parser(
        "distill",
        help="Build a smaller student or train one from reviewed teacher answers",
    )
    distill_p.add_argument(
        "--mode",
        choices=["logits", "data"],
        default="logits",
        help=(
            "logits builds a smaller local student; data either collects answers for "
            "review or trains from --teacher-data"
        ),
    )
    distill_p.add_argument(
        "--teacher",
        default=None,
        help="Local teacher model/id for logits mode, or endpoint model name for data mode",
    )
    distill_p.add_argument(
        "--student",
        default=None,
        help="Student base model for data mode; logits mode creates the student architecture",
    )
    distill_p.add_argument("--data", default=None, help="Training text (logits) or prompt data (data mode)")
    distill_p.add_argument(
        "--teacher-data",
        default=None,
        help="Clinician-reviewed instruction/output data; skips remote teacher collection",
    )
    distill_p.add_argument(
        "--teacher-url",
        default=None,
        help="OpenAI-compatible teacher base URL or /v1/chat/completions URL",
    )
    distill_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the teacher bearer token",
    )
    distill_p.add_argument("--output-dir", default="outputs/distill")
    distill_p.add_argument("--student-layers", type=int, default=6)
    distill_p.add_argument("--student-hidden-size", type=int, default=256)
    distill_p.add_argument("--student-heads", type=int, default=4)
    distill_p.add_argument("--temperature", type=float, default=2.0)
    distill_p.add_argument("--teacher-temperature", type=float, default=0.2)
    distill_p.add_argument("--alpha", type=float, default=0.5)
    distill_p.add_argument("--max-steps", type=int, default=24)
    distill_p.add_argument("--batch-size", type=int, default=2)
    distill_p.add_argument("--learning-rate", type=float, default=5e-4)
    distill_p.add_argument("--max-seq-length", type=int, default=128)
    distill_p.add_argument("--max-train-samples", type=int, default=256)
    distill_p.add_argument("--max-eval-samples", type=int, default=64)
    distill_p.add_argument(
        "--eval-data",
        default=None,
        help=(
            "Independent JSON/JSONL benchmark evaluated on the pre-distillation "
            "baseline and trained student"
        ),
    )
    distill_p.add_argument(
        "--eval-task",
        choices=["auto", "multiple-choice", "completion"],
        default="auto",
        help="Evaluation task; auto detects common question/answer or instruction/output data",
    )
    distill_p.add_argument("--eval-limit", type=int, default=0)
    distill_p.add_argument("--eval-batch-size", type=int, default=16)
    distill_p.add_argument("--eval-max-length", type=int, default=512)
    distill_p.add_argument("--eval-max-new-tokens", type=int, default=96)
    distill_p.add_argument("--eval-generation-samples", type=int, default=3)
    distill_p.add_argument("--eval-choices", default="ABCDE")
    distill_p.add_argument(
        "--eval-case-field",
        default=None,
        help="Optional shared case id for prompt-variant robustness statistics",
    )
    distill_p.add_argument(
        "--eval-variant-field",
        default=None,
        help="Optional prompt-variant label; requires --eval-case-field",
    )
    distill_p.add_argument(
        "--eval-prompt-variants",
        default=None,
        help="JSON file of declarative prompt templates; manages case/variant fields automatically",
    )
    distill_p.add_argument("--epochs", type=int, default=1)
    distill_p.add_argument("--method", choices=["lora", "qlora", "full"], default="lora")
    distill_p.add_argument("--lora-r", type=int, default=8)
    distill_p.add_argument("--lora-alpha", type=int, default=16)
    distill_p.add_argument("--concurrency", type=int, default=4)
    distill_p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for student rollouts and endpoint-teacher answers",
    )
    distill_p.add_argument("--domain", default="general")
    distill_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    distill_p.add_argument("--local-files-only", action="store_true")
    distill_p.add_argument("--trust-remote-code", action="store_true")
    distill_p.add_argument("--dry-run", action="store_true")
    distill_p.add_argument("--debug", action="store_true")

    quantize_p = sub.add_parser(
        "quantize",
        aliases=["lightweight"],
        help="Quantize using dynamic INT8, calibrated AWQ/GPTQ, or a configured QAT profile",
    )
    quantize_p.add_argument("model", help="Hugging Face model path or id")
    quantize_p.add_argument("output_dir", help="New quantized-model directory")
    quantize_p.add_argument(
        "--method",
        choices=["dynamic-int8", "awq", "gptq", "qat"],
        default=None,
        help="Defaults to CPU dynamic INT8; AWQ/GPTQ and QAT use the function configuration entry point",
    )
    quantize_p.add_argument("--config", default=None, help="JSON/YAML with optional ptq/qat configuration groups")
    quantize_p.add_argument("--calibration-data", default=None, help="Local calibration JSON/JSONL/TXT for AWQ/GPTQ")
    quantize_p.add_argument("--bits", type=int, choices=[4, 8], default=None)
    quantize_p.add_argument("--group-size", type=int, default=None)
    quantize_p.add_argument("--calibration-samples", type=int, default=None)
    quantize_p.add_argument("--train-data", default=None, help="Local QAT training data")
    quantize_p.add_argument("--validation-data", default=None)
    quantize_p.add_argument("--validation-split", type=float, default=None)
    quantize_p.add_argument("--max-steps", type=int, default=None)
    quantize_p.add_argument("--batch-size", type=int, default=None)
    quantize_p.add_argument("--learning-rate", type=float, default=None)
    quantize_p.add_argument("--max-seq-length", type=int, default=None)
    quantize_p.add_argument("--seed", type=int, default=None)
    quantize_p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    quantize_p.add_argument("--suite", default=None, help="Run capability regression after conversion")
    quantize_p.add_argument("--local-files-only", action="store_true")
    quantize_p.add_argument("--trust-remote-code", action="store_true")
    quantize_p.add_argument("--dry-run", action="store_true")
    quantize_p.add_argument("--output", default=None, help="Optional JSON result path")
    quantize_p.add_argument("--debug", action="store_true")

    prune_p = sub.add_parser(
        "prune",
        help="Create a loadable sparse-weight checkpoint (quality-sensitive; no speedup guarantee)",
    )
    prune_p.add_argument("model", help="Hugging Face model path or id")
    prune_p.add_argument("output_dir", help="New pruned-model directory")
    prune_p.add_argument("--ratio", type=float, default=0.15)
    prune_p.add_argument(
        "--method",
        choices=["l1_unstructured", "random_unstructured", "ln_structured"],
        default="l1_unstructured",
    )
    prune_p.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    prune_p.add_argument("--local-files-only", action="store_true")
    prune_p.add_argument("--trust-remote-code", action="store_true")
    prune_p.add_argument("--dry-run", action="store_true")
    prune_p.add_argument("--output", default=None, help="Optional JSON result path")
    prune_p.add_argument("--debug", action="store_true")

    regression_p = sub.add_parser("regression", help="Compare baseline/candidate on a fixed multi-task capability suite")
    regression_p.add_argument("suite", help="JSON/YAML suite with task data and metric regression budgets")
    regression_p.add_argument("--baseline", required=True)
    regression_p.add_argument("--candidate", required=True)
    regression_p.add_argument("--baseline-adapter", default=None)
    regression_p.add_argument("--candidate-adapter", default=None)
    regression_p.add_argument("--output", default="outputs/regression/report.json")
    regression_p.add_argument("--dry-run", action="store_true")
    regression_p.add_argument("--debug", action="store_true")

    recover_p = sub.add_parser("recover", help="Recover an existing floating-point/pruned checkpoint with CE and optional teacher distillation")
    recover_p.add_argument("model", help="Existing trainable checkpoint to recover")
    recover_p.add_argument("data", help="Local training JSON/JSONL")
    recover_p.add_argument("output_dir", help="New recovered checkpoint directory")
    recover_p.add_argument("--teacher", default=None, help="Pre-compression teacher checkpoint for logits distillation")
    recover_p.add_argument("--validation-data", default=None)
    recover_p.add_argument("--validation-split", type=float, default=None, help="Defaults to 0.1, or 0 with explicit --validation-data")
    recover_p.add_argument("--max-steps", type=int, default=100)
    recover_p.add_argument("--batch-size", type=int, default=2)
    recover_p.add_argument("--learning-rate", type=float, default=2e-5)
    recover_p.add_argument("--max-seq-length", type=int, default=512)
    recover_p.add_argument("--temperature", type=float, default=2.0)
    recover_p.add_argument("--distill-weight", type=float, default=0.5)
    recover_p.add_argument("--seed", type=int, default=42)
    recover_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    recover_p.add_argument("--local-files-only", action="store_true")
    recover_p.add_argument("--trust-remote-code", action="store_true")
    recover_p.add_argument("--suite", default=None, help="Run capability regression against --baseline or --teacher after recovery")
    recover_p.add_argument("--baseline", default=None, help="Capability baseline; defaults to --teacher or the input checkpoint")
    recover_p.add_argument("--output", default=None, help="Optional command result JSON path")
    recover_p.add_argument("--dry-run", action="store_true")
    recover_p.add_argument("--debug", action="store_true")

    dashboard_p = sub.add_parser(
        "dashboard",
        aliases=["webui"],
        help="View training progress and results in a local Web dashboard",
    )
    dashboard_p.add_argument(
        "output_dir",
        nargs="?",
        default="./outputs",
        help="A training run directory, or a parent containing runs",
    )
    dashboard_p.add_argument("--host", default="127.0.0.1")
    dashboard_p.add_argument("--port", type=int, default=7860)
    dashboard_p.add_argument("--open", action="store_true", help="Open the page in the default browser")

    workbench_p = sub.add_parser(
        "workbench", help="Explore a research project's data, runs, results and reproduction evidence",
    )
    workbench_p.add_argument("--project", required=True, help="Local experiment project directory (read-only)")
    workbench_p.add_argument("--study", default=None, help="Initial study ID shown in the workbench")
    workbench_p.add_argument("--host", default="127.0.0.1")
    workbench_p.add_argument("--port", type=int, default=7861)
    workbench_p.add_argument("--open", action="store_true", help="Open the page in the default browser")

    validate_p = sub.add_parser("validate-config", help="Validate a training config")
    validate_p.add_argument("config")
    validate_p.add_argument("--output", default=None)
    validate_p.add_argument("--skip-data-inspection", action="store_true")

    evaluate_p = sub.add_parser(
        "evaluate",
        aliases=["evaluate-model"],
        help="Evaluate a base, full, or PEFT model on a local benchmark",
    )
    evaluate_p.add_argument("model", help="Base/full model path or Hugging Face model id")
    evaluate_p.add_argument("data", help="Local JSONL or JSON evaluation data")
    evaluate_p.add_argument("--split", default=None, help="Explicit split key for multi-split JSON evaluation data")
    evaluate_p.add_argument(
        "--task",
        default="multiple-choice",
        choices=["multiple-choice", "completion", "rubric"],
    )
    evaluate_p.add_argument("--adapter", default=None, help="Optional PEFT adapter path")
    evaluate_p.add_argument("--choices", default="ABCDE")
    evaluate_p.add_argument("--prompt-field", default="instruction")
    evaluate_p.add_argument("--question-field", default="question")
    evaluate_p.add_argument("--answer-field", default="answer")
    evaluate_p.add_argument("--completion-field", default="output")
    evaluate_p.add_argument("--rubric-field", default="rubric")
    evaluate_p.add_argument("--reference-field", default="reference")
    evaluate_p.add_argument(
        "--pass-threshold",
        type=float,
        default=0.8,
        help="Default weighted pass threshold for rubric records",
    )
    evaluate_p.add_argument("--id-field", default="id")
    evaluate_p.add_argument("--group-field", default=None)
    evaluate_p.add_argument(
        "--case-field",
        default=None,
        help="Shared case id used to measure consistency across prompt variants",
    )
    evaluate_p.add_argument(
        "--variant-field",
        default=None,
        help="Prompt-variant label used for per-variant accuracy; requires --case-field",
    )
    evaluate_p.add_argument(
        "--prompt-variants",
        default=None,
        help="JSON file of prompt templates expanded over each source case",
    )
    evaluate_p.add_argument("--limit", type=int, default=0)
    evaluate_p.add_argument("--batch-size", type=int, default=16)
    evaluate_p.add_argument("--max-length", type=int, default=512)
    generation_group = evaluate_p.add_mutually_exclusive_group()
    generation_group.add_argument("--generation-samples", type=int, default=0)
    generation_group.add_argument("--generation-all", action="store_true")
    generation_group.add_argument(
        "--generation-indices",
        default="",
        help="Comma-separated zero-based record indices",
    )
    evaluate_p.add_argument("--max-new-tokens", type=int, default=96)
    evaluate_p.add_argument("--repetition-penalty", type=float, default=1.0)
    evaluate_p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    evaluate_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    evaluate_p.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    evaluate_p.add_argument("--enable-thinking", action="store_true")
    evaluate_p.add_argument("--trust-remote-code", action="store_true")
    evaluate_p.add_argument("--local-files-only", action="store_true")
    evaluate_p.add_argument(
        "--endpoint",
        default=None,
        help="OpenAI-compatible base URL for rubric evaluation instead of local loading",
    )
    evaluate_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the endpoint bearer token",
    )
    evaluate_p.add_argument("--request-timeout", type=float, default=120.0)
    evaluate_p.add_argument("--output", default="evaluation_results.json")
    evaluate_p.add_argument("--markdown-output", default=None)
    evaluate_p.add_argument("--debug", action="store_true")

    compare_p = sub.add_parser(
        "compare",
        aliases=["compare-evaluations"],
        help="Compare paired validation/test results and apply a promotion gate",
    )
    compare_p.add_argument("--base-validation", required=True)
    compare_p.add_argument("--candidate-validation", required=True)
    compare_p.add_argument("--base-test", default=None)
    compare_p.add_argument("--candidate-test", default=None)
    compare_p.add_argument("--manifest", default=None)
    compare_p.add_argument(
        "--overlap-key", default="exact_question_overlap_after_exclusion"
    )
    compare_p.add_argument(
        "--min-gain-pp",
        type=float,
        default=0.0,
        help=(
            "Minimum accuracy gain in percentage points; for completion results, "
            "minimum relative PPL improvement percent"
        ),
    )
    compare_p.add_argument(
        "--max-p",
        type=float,
        default=0.05,
        help="Exclusive paired-test p-value threshold",
    )
    compare_p.add_argument(
        "--max-critical-failures",
        type=int,
        default=0,
        help="Maximum candidate critical failures allowed for rubric evaluations",
    )
    compare_p.add_argument("--experiment", default="model-comparison")
    compare_p.add_argument("--scope", default="multiple-choice evaluation")
    compare_p.add_argument("--output", default="evaluation_comparison.json")
    compare_p.add_argument("--markdown-output", default=None)
    compare_p.add_argument("--debug", action="store_true")

    doctor_p = sub.add_parser("doctor", help="Check dependencies and devices for training or serving without loading weights")
    doctor_p.add_argument("--profile", choices=["train", "serve", "native", "vllm"], default="train")
    doctor_p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    doctor_p.add_argument("--output", default=None)

    plugins_p = sub.add_parser(
        "plugins",
        help="List built-in and discovered training stages without loading them",
    )
    plugins_p.add_argument("--output", default=None)

    smoke_p = sub.add_parser("smoke-test", help="Run local tiny-model smoke tests for SFT/VLA/preference training")
    smoke_p.add_argument("--work-dir", default="build/smoke_e2e_auto")
    smoke_p.add_argument("--skip-training", action="store_true", help="Only run fixture generation, inspection, and validation")
    smoke_p.add_argument("--keep", action="store_true", help="Reuse an existing work directory instead of deleting it first")
    smoke_p.add_argument("--skip-doctor", action="store_true")
    smoke_p.add_argument("--in-process", action="store_true", help="Run checks in the current Python process for faster local iteration")
    smoke_p.add_argument("--timeout-seconds", type=int, default=180)
    smoke_p.add_argument("--output", default=None)

    media_cache_p = sub.add_parser(
        "build-media-cache",
        help="Build resumable image/audio/video training-cache shards from a manifest",
    )
    media_cache_p.add_argument("config", help="Media-cache YAML or JSON config")
    media_cache_p.add_argument("--no-resume", action="store_true")
    media_cache_p.add_argument("--overwrite", action="store_true")
    media_cache_p.add_argument("--output", default=None, help="Optional JSON result path")

    media_codecs_p = sub.add_parser(
        "media-codecs", help="List registered built-in media codecs"
    )
    media_codecs_p.add_argument("--modality", choices=["image", "audio", "video"])
    media_codecs_p.add_argument("--output", default=None)

    inspect_p = sub.add_parser("inspect-data", help="Inspect local SFT/preference/RL data before training")
    inspect_p.add_argument("data_path")
    inspect_p.add_argument("--task", default="sft", choices=["sft", "preference", "dpo", "orpo", "kto", "rl", "rlhf", "grpo"])
    inspect_p.add_argument("--max-records", type=int, default=256)
    inspect_p.add_argument(
        "--validation-data",
        default=None,
        help="Optional validation split; enables cross-split leakage auditing",
    )
    inspect_p.add_argument(
        "--test-data",
        default=None,
        help="Optional test split; enables cross-split leakage auditing",
    )
    inspect_p.add_argument(
        "--group-field",
        action="append",
        default=None,
        help="Case/group id field used to prevent grouped leakage; repeatable",
    )
    inspect_p.add_argument(
        "--content-field",
        action="append",
        default=None,
        help="Prompt text field used for duplicate checks; repeatable",
    )
    inspect_p.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.96,
        help="Similarity threshold for approximate cross-split duplicate warnings",
    )
    inspect_p.add_argument(
        "--strict-leakage",
        action="store_true",
        help="Also block likely near duplicates and records without group ids",
    )
    inspect_p.add_argument(
        "--include-content-previews",
        action="store_true",
        help="Include prompt snippets in leakage examples; may expose sensitive data",
    )
    inspect_p.add_argument("--output", default=None)

    inspect_vla_p = sub.add_parser("inspect-vla", help="Inspect local VLA robot trajectory data before training")
    inspect_vla_p.add_argument("data_path")
    inspect_vla_p.add_argument("--image-root", default=None)
    inspect_vla_p.add_argument("--max-records", type=int, default=256)
    inspect_vla_p.add_argument("--action-dim", type=int, default=7)
    inspect_vla_p.add_argument("--action-type", default="continuous", choices=["continuous", "discrete", "text"])
    inspect_vla_p.add_argument("--action-bins", type=int, default=256)
    inspect_vla_p.add_argument("--min-value", type=float, default=-1.0)
    inspect_vla_p.add_argument("--max-value", type=float, default=1.0)
    inspect_vla_p.add_argument("--no-gripper", action="store_true")
    inspect_vla_p.add_argument("--output", default=None)

    world_model_p = sub.add_parser(
        "train-world-model",
        help="Train a SaddleLLM-native world model from offline trajectories",
    )
    world_model_p.add_argument("config", help="World-model YAML or JSON config")
    world_model_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/config and infer network dimensions without training",
    )
    world_model_p.add_argument("--output", default=None, help="Optional JSON result path")

    world_model_backends_p = sub.add_parser(
        "world-model-backends",
        help="List SaddleLLM-native world-model backends",
    )
    world_model_backends_p.add_argument("--output", default=None)

    world_model_infer_p = sub.add_parser(
        "infer-world-model",
        help="Run native rollout or lightweight action planning from a checkpoint",
    )
    world_model_infer_p.add_argument("checkpoint")
    world_model_infer_p.add_argument("request", help="Inference request JSON or YAML")
    world_model_infer_p.add_argument("--device", default="auto")
    world_model_infer_p.add_argument("--output", default=None)

    eye_data_p = sub.add_parser(
        "build-eye-control-data",
        help="Generate safety-filtered two-axis eye-control trajectories",
    )
    eye_data_p.add_argument(
        "--output", default="data/prosthetic_eye_trajectories.jsonl"
    )
    eye_data_p.add_argument("--config", default=None, help="Optional eye runtime YAML/JSON")
    eye_data_p.add_argument("--episodes", type=int, default=128)
    eye_data_p.add_argument("--steps", type=int, default=100)
    eye_data_p.add_argument("--seed", type=int, default=42)
    eye_data_p.add_argument("--moving-target-ratio", type=float, default=0.5)
    eye_data_p.add_argument("--target-dropout-probability", type=float, default=0.03)
    eye_data_p.add_argument("--result-output", default=None)

    eye_sim_p = sub.add_parser(
        "simulate-eye-control",
        help="Evaluate PID or RSSM+CEM eye control in the safe simulator",
    )
    eye_sim_p.add_argument("--checkpoint", default=None)
    eye_sim_p.add_argument("--config", default=None, help="Optional eye runtime YAML/JSON")
    eye_sim_p.add_argument("--episodes", type=int, default=8)
    eye_sim_p.add_argument("--steps", type=int, default=150)
    eye_sim_p.add_argument("--seed", type=int, default=42)
    eye_sim_p.add_argument("--device", default="auto")
    eye_sim_p.add_argument("--include-traces", action="store_true")
    eye_sim_p.add_argument("--output", default=None)

    spatial_data_p = sub.add_parser(
        "build-spatial-world-data",
        help="Generate native spatial RSSM trajectories from a top-down map",
    )
    spatial_data_p.add_argument("image")
    spatial_data_p.add_argument(
        "--additional-image",
        action="append",
        default=[],
        help="Additional map image; repeat for held-out multi-map training",
    )
    spatial_data_p.add_argument(
        "--schema", choices=["v1", "v2"], default="v1"
    )
    spatial_data_p.add_argument("--output", default="data/spatial_world_model.jsonl")
    spatial_data_p.add_argument("--episodes", type=int, default=64)
    spatial_data_p.add_argument("--routes-per-pair", type=int, default=2)
    spatial_data_p.add_argument("--seed", type=int, default=42)
    spatial_data_p.add_argument("--minimum-distance", type=float, default=0.25)
    spatial_data_p.add_argument("--max-steps", type=int, default=64)
    spatial_data_p.add_argument("--observation-size", type=int, default=16)
    spatial_data_p.add_argument("--crop-size", type=int, default=16)
    spatial_data_p.add_argument("--sensor-radius", type=int, default=6)
    spatial_data_p.add_argument("--collision-probability", type=float, default=0.15)
    spatial_data_p.add_argument("--free-threshold", type=float, default=0.72)
    spatial_data_p.add_argument("--free-is-dark", action="store_true")
    spatial_data_p.add_argument("--uncertainty-band", type=float, default=0.0)
    spatial_data_p.add_argument("--obstacle-dilation", type=int, default=0)
    spatial_data_p.add_argument("--max-dimension", type=int, default=512)
    spatial_data_p.add_argument("--resolution", type=float, default=0.5)
    spatial_data_p.add_argument("--no-diagonal", action="store_true")
    spatial_data_p.add_argument("--allow-unknown", action="store_true")

    spatial_eval_p = sub.add_parser(
        "evaluate-spatial-world-model",
        help="Evaluate multi-step occupancy, motion, and collision predictions",
    )
    spatial_eval_p.add_argument("checkpoint")
    spatial_eval_p.add_argument("data")
    spatial_eval_p.add_argument("--device", default="auto")
    spatial_eval_p.add_argument("--sequence-length", type=int, default=20)
    spatial_eval_p.add_argument("--max-windows", type=int, default=None)
    spatial_eval_p.add_argument("--group-key", default=None)
    spatial_eval_p.add_argument("--group-value", default=None)
    spatial_eval_p.add_argument("--output", default=None)

    spatial_route_p = sub.add_parser(
        "plan-spatial-route",
        help="Extract a top-down occupancy map and visualize Top-K walking routes",
    )
    spatial_route_p.add_argument("image")
    spatial_route_p.add_argument("--start", required=True, help="Source pixel x,y or VLM entity name")
    spatial_route_p.add_argument("--goal", required=True, help="Source pixel x,y or VLM entity name")
    spatial_route_p.add_argument("--routes", type=int, default=3)
    spatial_route_p.add_argument("--instruction", default="")
    spatial_route_p.add_argument("--free-threshold", type=float, default=0.72)
    spatial_route_p.add_argument("--free-is-dark", action="store_true")
    spatial_route_p.add_argument("--uncertainty-band", type=float, default=0.0)
    spatial_route_p.add_argument("--obstacle-dilation", type=int, default=1)
    spatial_route_p.add_argument("--max-dimension", type=int, default=512)
    spatial_route_p.add_argument("--resolution", type=float, default=1.0)
    spatial_route_p.add_argument("--no-diagonal", action="store_true")
    spatial_route_p.add_argument("--allow-unknown", action="store_true")
    spatial_route_p.add_argument("--clearance-weight", type=float, default=0.0)
    spatial_route_p.add_argument("--diversity-weight", type=float, default=2.0)
    spatial_route_p.add_argument("--qwen-model", default=None)
    spatial_route_p.add_argument("--qwen-device-map", default="auto")
    spatial_route_p.add_argument("--allow-perspective", action="store_true")
    spatial_route_p.add_argument("--output-json", default=None)
    spatial_route_p.add_argument("--output-html", default=None)
    spatial_route_p.add_argument("--output-png", default=None)

    spatial_studio_p = sub.add_parser(
        "spatial-studio",
        help="Run the end-to-end spatial world-model web studio",
    )
    spatial_studio_p.add_argument("--host", default="127.0.0.1")
    spatial_studio_p.add_argument("--port", type=int, default=7865)
    spatial_studio_p.add_argument("--workspace", default="outputs/spatial_studio")
    spatial_studio_p.add_argument("--frontend-dist", default=None)
    spatial_studio_p.add_argument("--demo-image", default="data/spatial_floorplan_example.pbm")
    spatial_studio_p.add_argument("--qwen-model", default=None)
    spatial_studio_p.add_argument("--world-model", default=None)
    spatial_studio_p.add_argument("--device", default="auto")

    world_agent_run_p = sub.add_parser(
        "world-agent-run",
        help="Run analyze, plan, and simulation through the unified WorldAgent runtime",
    )
    world_agent_run_p.add_argument("image")
    world_agent_run_p.add_argument("--start", required=True, help="Source pixel x,y or VLM entity name")
    world_agent_run_p.add_argument("--goal", required=True, help="Source pixel x,y or VLM entity name")
    world_agent_run_p.add_argument("--config", default=None)
    world_agent_run_p.add_argument("--workspace", default=None)
    world_agent_run_p.add_argument("--instruction", default="")
    world_agent_run_p.add_argument("--routes", type=int, default=3)
    world_agent_run_p.add_argument(
        "--semantic-backend", choices=["disabled", "qwen-vl"], default="disabled"
    )
    world_agent_run_p.add_argument("--qwen-model", default=None)
    world_agent_run_p.add_argument("--world-model", default=None)
    world_agent_run_p.add_argument("--use-world-model", action="store_true")
    world_agent_run_p.add_argument(
        "--simulation", choices=["auto", "geometry", "world_model"], default="auto"
    )
    world_agent_run_p.add_argument("--device", default=None)
    world_agent_run_p.add_argument("--output", default=None)

    world_agent_api_p = sub.add_parser(
        "world-agent-api",
        help="Run the headless analyze/plan/simulate/feedback WorldAgent API",
    )
    world_agent_api_p.add_argument("--config", default=None)
    world_agent_api_p.add_argument("--host", default="127.0.0.1")
    world_agent_api_p.add_argument("--port", type=int, default=7866)
    world_agent_api_p.add_argument("--workspace", default=None)
    world_agent_api_p.add_argument("--qwen-model", default=None)
    world_agent_api_p.add_argument("--world-model", default=None)
    world_agent_api_p.add_argument("--device", default=None)

    preflight_p = sub.add_parser("preflight", help="Create a no-training post-training preflight plan")
    preflight_p.add_argument("data_path")
    preflight_p.add_argument("--stage", default="sft", choices=["sft", "preference", "rlhf", "dpo", "orpo", "kto"])
    preflight_p.add_argument("--method", default="lora")
    preflight_p.add_argument("--preference-method", default="dpo", choices=["dpo", "orpo", "kto"])
    preflight_p.add_argument("--root-dir", default="./llm_factory")
    preflight_p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    preflight_p.add_argument("--output-dir", default=None)
    preflight_p.add_argument("--no-save", action="store_true")

    report_p = sub.add_parser("report", help="Generate a pretrain report for an output directory")
    report_p.add_argument("output_dir")
    report_p.add_argument("--root-dir", default=None)

    export_p = sub.add_parser(
        "export",
        aliases=["export-model"],
        help="Package a checkpoint as a verified HF release",
    )
    export_p.add_argument("model_path", help="Local checkpoint/adapter path or Hugging Face model id")
    export_p.add_argument("output_dir", help="New release directory")
    export_p.add_argument("--format", default="hf", choices=["hf", "saddle", "native"])
    export_p.add_argument("--no-merge-lora", action="store_true")
    export_p.add_argument("--no-safe-serialization", action="store_true")
    export_p.add_argument("--trust-remote-code", action="store_true")
    export_p.add_argument("--local-files-only", action="store_true")
    export_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    export_p.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    export_p.add_argument("--hash-weights", action="store_true")
    export_p.add_argument("--overwrite", action="store_true")
    export_p.add_argument("--gate-result", default=None, help="Optional release_gate.json to embed")
    export_p.add_argument("--require-gate", action="store_true")
    export_p.add_argument("--result-output", default=None, help="Optional JSON command result path")

    serve_p = sub.add_parser(
        "serve",
        aliases=["serve-model"],
        help="Run an OpenAI-compatible local inference API",
    )
    serve_p.add_argument("model_path", help="Exported release, full checkpoint, adapter, or HF model id")
    serve_p.add_argument("--model-name", default=None)
    serve_p.add_argument("--backend", choices=["auto", "local", "native", "vllm"], default="auto",
                         help="auto: compatible Linux CUDA vLLM or transparent local route; native: Saddle engine; local: native/HF; vllm: continuous batching")
    serve_p.add_argument("--performance", choices=["auto", "latency", "throughput"], default="auto",
                         help="Existing settings (auto), bounded repeated-prefix reuse (latency), or compatible batching (throughput); explicit options win, dtype/thinking unchanged")
    serve_p.add_argument("--require-high-throughput", action="store_true",
                         help="Require continuous-batching vLLM; fail instead of selecting native/local")
    serve_p.add_argument("--dry-run", action="store_true",
                         help="Show deployment selection without downloading/loading model weights or starting HTTP")
    serve_p.add_argument("--native-attention-backend", choices=["sdpa", "sdpa_reference", "eager"], default=None,
                         help="Optional native attention override: sdpa (speed-oriented), sdpa_reference (compact-GQA compatibility), eager (math reference)")
    serve_p.add_argument("--native-stream-batching", action=argparse.BooleanOptionalAction, default=None,
                         help="Experimental dense greedy/sampling step batching, including independent KV for unequal lengths; requires --backend native, float weights, --batch-size >= 2 and --max-concurrency >= 2")
    serve_p.add_argument("--native-decode-mode", choices=["compatible", "throughput"], default="compatible",
                         help="Native BF16 decode: compatible preserves B1 Linear shapes; throughput uses batched Linear and may change output tokens; requires --native-stream-batching")
    serve_p.add_argument("--batch-wait-ms", type=float, default=5.0,
                         help="Maximum native batch collection wait in milliseconds (default: 5)")
    serve_p.add_argument("--gpu-memory-utilization", type=float, default=None,
                         help="vLLM GPU memory fraction; budget shared GPUs explicitly (default: 0.9)")
    serve_p.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM tensor parallel GPU count")
    serve_p.add_argument("--max-model-len", type=int, default=None, help="vLLM input + output context limit")
    serve_p.add_argument("--max-num-seqs", type=int, default=None, help="vLLM scheduled sequence cap (default: 16)")
    serve_p.add_argument("--max-num-batched-tokens", type=int, default=None, help="vLLM per-iteration token budget")
    serve_p.add_argument("--vllm-prefix-caching", action=argparse.BooleanOptionalAction, default=None,
                         help="Enable/disable vLLM automatic prefix caching; overrides performance preset")
    serve_p.add_argument("--enforce-eager", action="store_true", help="vLLM debugging mode without CUDA Graph capture")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    serve_p.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    serve_p.add_argument(
        "--quantization",
        default="none",
        choices=["none", "4bit", "8bit", "int8"],
        help="local: bitsandbytes 4bit/8bit; native: experimental W8A16 int8 weights (CUDA/Triton)",
    )
    serve_p.add_argument("--trust-remote-code", action="store_true")
    serve_p.add_argument("--local-files-only", action="store_true")
    serve_p.add_argument("--no-merge-adapter", action="store_true")
    serve_p.add_argument("--no-fuse-projections", action="store_true",
                         help="Disable native inference QKV/gate-up packing for comparison")
    serve_p.add_argument("--cpu-prepack", action="store_true",
                         help="Opt into native CPU oneDNN decode weight caches; needs extra RAM, no quantization")
    serve_p.add_argument("--native-fused-kernels", action="store_true",
                         help="Opt into native CUDA RMSNorm/RoPE decode fusion; requires Triton, warms kernels at startup")
    serve_p.add_argument("--native-prefill-chunk-size", type=int, default=0,
                         help="Bound prompt tokens per native stream prefill chunk; 0 disables; requires native stream batching")
    serve_p.add_argument("--native-max-step-tokens", type=int, default=None,
                         help="Compute tokens per scheduler tick, distinct from KV batch-token-budget; requires chunked prefill")
    serve_p.add_argument("--max-concurrency", type=int, default=None, help="Local generate concurrency (default: 1, throughput preset: 4); vLLM uses --max-num-seqs")
    serve_p.add_argument("--max-pending-requests", type=int, default=64,
                         help="Bound local pending queue or vLLM HTTP admission beyond scheduled slots")
    serve_p.add_argument("--batch-size", type=int, default=None,
                         help="Merge compatible native, non-streaming greedy requests (default: disabled)")
    serve_p.add_argument("--batch-token-budget", type=int, default=None,
                         help="Maximum padded prompt + output token slots per native microbatch (default: input limit + output limit)")
    serve_p.add_argument("--batch-length-bucket-size", type=int, default=0,
                         help="Opt into nearby-length native greedy batches; 0 keeps exact-length grouping")
    serve_p.add_argument("--max-queued-tokens", type=int, default=None,
                         help="Optional total prompt + output reservation limit for the local batch queue or vLLM HTTP admission")
    serve_p.add_argument("--prefix-cache-mb", type=int, default=None,
                         help="Native single-request prompt KV cache budget in MiB (0 disables)")
    serve_p.add_argument("--prefix-cache-partial", action=argparse.BooleanOptionalAction, default=None,
                         help="Opt into partial token-prefix reuse; benchmark latency/quality first")
    serve_p.add_argument("--max-input-tokens", type=int, default=4096)
    serve_p.add_argument("--default-max-new-tokens", type=int, default=256)
    serve_p.add_argument("--max-new-tokens", type=int, default=512)
    thinking_p = serve_p.add_mutually_exclusive_group()
    thinking_p.add_argument("--enable-thinking", dest="enable_thinking", action="store_const", const=True,
                            default=None, help="Explicitly enable model-template reasoning (default: preserve template)")
    thinking_p.add_argument("--disable-thinking", dest="enable_thinking", action="store_const", const=False,
                            help="Explicit latency/quality tradeoff: disable model-template reasoning")
    serve_p.add_argument("--truncate-input", action="store_true",
                         help="Opt into legacy local/native prompt truncation; default rejects overlong input")
    serve_p.add_argument("--no-web-ui", action="store_true", help="Serve API endpoints only")
    serve_p.add_argument("--no-chat-history", action="store_true", help="Disable built-in chat history reads/writes; keep existing history files untouched")
    serve_p.add_argument("--title", default="SaddleLLM 对话", help="Title shown in the built-in chat page")
    serve_p.add_argument("--open", action="store_true", help="Open the chat page in the default browser")
    serve_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the bearer token; never pass the secret itself",
    )

    embedding_p = sub.add_parser(
        "serve-embedding",
        help="Run an OpenAI-compatible local embedding API",
    )
    embedding_p.add_argument("model_path", help="Local sentence-transformer model directory")
    embedding_p.add_argument("--model-name", default=None)
    embedding_p.add_argument("--host", default="127.0.0.1")
    embedding_p.add_argument("--port", type=int, default=8098)
    embedding_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    embedding_p.add_argument("--batch-size", type=int, default=32)
    embedding_p.add_argument("--max-concurrency", type=int, default=1)
    embedding_p.add_argument("--max-inputs", type=int, default=256)
    embedding_p.add_argument("--max-input-chars", type=int, default=32768)
    embedding_p.add_argument("--max-total-chars", type=int, default=262144)
    embedding_p.add_argument("--max-input-tokens", type=int, default=8192)
    embedding_p.add_argument("--dimensions", type=int, default=None)
    embedding_p.add_argument("--trust-remote-code", action="store_true")
    embedding_p.add_argument("--local-files-only", action="store_true")
    embedding_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the bearer token; never pass the secret itself",
    )

    reranker_p = sub.add_parser(
        "serve-reranker",
        help="Run a local Qwen3 passage reranking API",
    )
    reranker_p.add_argument("model_path", help="Local Qwen3 reranker model directory")
    reranker_p.add_argument("--model-name", default=None)
    reranker_p.add_argument("--host", default="127.0.0.1")
    reranker_p.add_argument("--port", type=int, default=8099)
    reranker_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    reranker_p.add_argument("--batch-size", type=int, default=16)
    reranker_p.add_argument("--max-concurrency", type=int, default=1)
    reranker_p.add_argument("--max-documents", type=int, default=100)
    reranker_p.add_argument("--max-document-chars", type=int, default=32768)
    reranker_p.add_argument("--max-total-chars", type=int, default=524288)
    reranker_p.add_argument("--max-length", type=int, default=1024)
    reranker_p.add_argument(
        "--instruction",
        default=(
            "Given a query, retrieve the most relevant passage that answers it"
        ),
    )
    reranker_p.add_argument("--trust-remote-code", action="store_true")
    reranker_p.add_argument("--local-files-only", action="store_true")
    reranker_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the bearer token; never pass the secret itself",
    )

    decision_serve_p = sub.add_parser("serve-decision", help="Serve a trained Saddle Choice model as structured probabilities")
    decision_serve_p.add_argument("model", help="Directory containing decision_config.json")
    decision_serve_p.add_argument("--backbone", default=None, help="Override the external local backbone path")
    decision_serve_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    decision_serve_p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    decision_serve_p.add_argument("--host", default="127.0.0.1")
    decision_serve_p.add_argument("--port", type=int, default=8010)
    decision_serve_p.add_argument("--api-key-env", default=None)
    decision_serve_p.add_argument("--no-web-ui", action="store_true")
    decision_serve_p.add_argument("--dry-run", action="store_true")

    decision_eval_p = sub.add_parser("evaluate-decision", help="Evaluate a trained typed Choice/Score/Noul model without updating it")
    decision_eval_p.add_argument("--model", required=True, help="Saved typed decision model directory")
    decision_eval_p.add_argument("--data", required=True, help="Typed dataset directory")
    decision_eval_p.add_argument("--output-dir", "--output", dest="output_dir", required=True, help="New evaluation directory")
    decision_eval_p.add_argument("--splits", default="test", help="Comma-separated requested splits; default test")
    decision_eval_p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    decision_eval_p.add_argument("--batch-size", type=int, default=4)
    decision_eval_p.add_argument("--cpu-threads", type=int, default=4)
    decision_eval_p.add_argument("--run-label", default="model")
    decision_eval_p.add_argument("--logits-reference", default=None, help="Prior logits JSONL for exact reload verification")

    args = parser.parse_args(argv)
    if args.command is None:
        print(_getting_started_text())
        return 0

    handlers = {
        "quickstart": _cmd_quickstart,
        "import-data": _cmd_import_data,
        "plan": _cmd_plan,
        "train": _cmd_train,
        "distill": _cmd_distill,
        "quantize": _cmd_quantize,
        "lightweight": _cmd_quantize,
        "prune": _cmd_prune,
        "regression": _cmd_regression,
        "recover": _cmd_recover,
        "dashboard": _cmd_dashboard,
        "webui": _cmd_dashboard,
        "workbench": _cmd_workbench,
        "validate-config": _cmd_validate_config,
        "evaluate": _cmd_evaluate_model,
        "evaluate-model": _cmd_evaluate_model,
        "compare": _cmd_compare_evaluations,
        "compare-evaluations": _cmd_compare_evaluations,
        "doctor": _cmd_doctor,
        "plugins": _cmd_plugins,
        "smoke-test": _cmd_smoke_test,
        "build-media-cache": _cmd_build_media_cache,
        "media-codecs": _cmd_media_codecs,
        "inspect-data": _cmd_inspect_data,
        "inspect-vla": _cmd_inspect_vla,
        "train-world-model": _cmd_train_world_model,
        "world-model-backends": _cmd_world_model_backends,
        "infer-world-model": _cmd_infer_world_model,
        "build-eye-control-data": _cmd_build_eye_control_data,
        "simulate-eye-control": _cmd_simulate_eye_control,
        "build-spatial-world-data": _cmd_build_spatial_world_data,
        "evaluate-spatial-world-model": _cmd_evaluate_spatial_world_model,
        "plan-spatial-route": _cmd_plan_spatial_route,
        "spatial-studio": _cmd_spatial_studio,
        "world-agent-run": _cmd_world_agent_run,
        "world-agent-api": _cmd_world_agent_api,
        "preflight": _cmd_preflight,
        "report": _cmd_report,
        "export": _cmd_export_model,
        "export-model": _cmd_export_model,
        "serve": _cmd_serve_model,
        "serve-model": _cmd_serve_model,
        "serve-embedding": _cmd_serve_embedding,
        "serve-reranker": _cmd_serve_reranker,
        "serve-decision": _cmd_serve_decision,
        "evaluate-decision": _cmd_evaluate_decision,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        if args.command in {
            "dashboard",
            "webui",
            "workbench",
            "serve",
            "serve-model",
            "serve-embedding",
            "serve-reranker",
            "serve-decision",
            "spatial-studio",
            "world-agent-api",
        }:
            print("\nSaddleLLM service stopped.")
            return 0
        _emit_json({"ok": False, "command": args.command, "error": "Interrupted by user."})
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        _emit_json(
            {
                "ok": False,
                "command": args.command,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hint": "Use --help for command options; use --debug on train for a traceback.",
            }
        )
        return 1


def _getting_started_text() -> str:
    return """SaddleLLM — 从一个可检查的小项目开始

  saddlellm quickstart
  cd saddlellm-quickstart
  saddlellm validate-config config.yaml
  saddlellm train --stage sft --model MODEL --data train.jsonl
  saddlellm dashboard outputs/run --open
  saddlellm serve outputs/release --open

查看全部命令：saddlellm --help
已有配置：    saddlellm validate-config <配置文件>"""


def _cmd_quickstart(args) -> int:
    from .factory.Quickstart import create_quickstart

    result = create_quickstart(
        args.directory,
        stages=args.stages,
        preference_method=args.preference_method,
        model=args.model,
        sft_data=args.sft_data,
        preference_data=args.preference_data,
        qlora=args.qlora,
        overwrite=args.force,
    )
    _emit_json(result.to_dict(), args.output)
    return 0


def _cmd_plugins(args) -> int:
    from .framework import STAGE_ENTRY_POINT_GROUP, get_stage_registry

    def _items(registry):
        return [
            {
                "name": item.name,
                "source": item.source,
                "loaded": item.loaded,
                "entry_point": item.entry_point,
            }
            for item in registry.info()
        ]

    stage_registry = get_stage_registry()
    payload = {
        "stages": {
            "entry_point_group": STAGE_ENTRY_POINT_GROUP,
            "plugins": _items(stage_registry),
        },
        "discovery_errors": list(stage_registry.discovery_errors),
    }
    _emit_json(payload, args.output)
    return 0


def _cmd_import_data(args) -> int:
    from .factory.TrainingFactory import FactoryConfig, LLMTrainingFactory

    selected = [item.strip() for item in args.selected.split(",") if item.strip()] if args.selected else None
    factory = LLMTrainingFactory(FactoryConfig(root_dir=args.root_dir))
    result = factory.import_dataset_manifest(
        args.manifest,
        dataset_dir=args.dataset_dir,
        selected_names=selected,
        role=args.role,
    )
    print(json.dumps({
        "saved_to": result["saved_to"],
        "summary": result["summary"],
        "num_saddle_sources": len(result["saddle_sources"]),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_plan(args) -> int:
    from .factory.BackendAdapters import BackendAdapterRegistry
    from .training.AutoParallel import (
        create_parallel_launch_plan,
        materialize_parallel_config,
        plan_parallel_training,
    )

    config = _load_training_config(args.config)
    decision = plan_parallel_training(
        config,
        strategy=args.parallel,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        launcher=args.launcher,
    )
    decision.require_executable()
    output_dir = os.path.dirname(args.config) or "."
    launch_plan_path = args.launch_plan or os.path.join(output_dir, "launch_plan.json")
    launch_script_path = args.launch_script or os.path.join(output_dir, "launch.ps1")
    resolved_config_path = materialize_parallel_config(
        config,
        decision,
        output_path=os.path.join(output_dir, "resolved_parallel_config.json"),
    )
    plan = create_parallel_launch_plan(resolved_config_path, decision)
    BackendAdapterRegistry.save_launch_script(plan, launch_script_path)
    BackendAdapterRegistry.save_launch_plan(plan, launch_plan_path)
    print(json.dumps({
        "config": args.config,
        "resolved_config": resolved_config_path,
        "launch_plan": launch_plan_path,
        "launch_script": launch_script_path,
        "backend": plan.backend,
        "parallel": decision.to_dict(),
        "command": plan.command,
    }, ensure_ascii=False, indent=2))
    return 0


def _direct_train_option_was_used(args) -> bool:
    values = (
        args.stage,
        args.model,
        args.data,
        args.pretrain_data,
        args.sft_data,
        args.dpo_data,
        args.orpo_data,
        args.kto_data,
        args.ppo_data,
        args.output_dir,
        args.pretrain_mode,
        args.model_config,
        args.tokenizer,
        args.backend,
        args.data_type,
        args.text_column,
        args.min_text_length,
        args.quality_min_score,
        args.dedup_method,
        args.global_batch_size,
        args.min_learning_rate,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.max_seq_length,
        args.max_steps,
        args.resume_from_checkpoint,
        args.warmup_steps,
        args.save_steps,
        args.eval_steps,
        args.logging_steps,
        args.validation_split,
        args.validation_data,
        args.method,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_target_modules,
        args.precision,
        args.response_template,
        args.replay_data,
        args.replay_ratio,
        args.replay_count,
        args.reference_model,
        args.retention_kl_weight,
        args.retention_temperature,
        args.seed,
        args.encoder_device,
        args.projection_rank,
        args.encoder_batch_size,
        args.cpu_threads,
        args.weight_decay,
        args.decision_type,
        args.decision_backbone,
        args.decision_method,
        args.optimizer_device,
        args.decision_hidden_size,
        args.decision_heads,
        args.query_pooling,
        args.instance_question_in_query,
        args.backbone_learning_rate,
        args.head_learning_rate,
        args.head_warmup_epochs,
        args.early_stopping_patience,
        args.evaluate_test,
    )
    return any(value is not None for value in values) or any(
        (
            args.local_files_only,
            args.trust_remote_code,
            args.no_gradient_checkpointing,
            args.streaming,
            args.no_pack_sequences,
            args.no_append_eos,
        )
    )


_DIRECT_TRAIN_STAGES = ("pretrain", "sft", "dpo", "orpo", "kto", "ppo")
_DIRECT_TRAIN_EXPRESSIONS = _DIRECT_TRAIN_STAGES + ("decision", "auto-posttrain")
_DIRECT_STAGE_MAPPING = {
    "pretrain": "pretrain",
    "sft": "sft",
    "dpo": "preference",
    "orpo": "preference",
    "kto": "preference",
    "ppo": "rlhf",
}


def _parse_direct_stages(value: str) -> list[str]:
    stages = [item.strip().lower() for item in str(value or "").split(",")]
    if not stages or any(not item for item in stages):
        raise ValueError("--stage must contain one or more comma-separated stage names.")
    unknown = [item for item in stages if item not in _DIRECT_TRAIN_EXPRESSIONS]
    if unknown:
        raise ValueError(
            "Unsupported direct training stage(s): "
            + ", ".join(unknown)
            + ". Expected: "
            + ", ".join(_DIRECT_TRAIN_EXPRESSIONS)
            + "."
        )
    if "auto-posttrain" in stages and stages != ["auto-posttrain"]:
        raise ValueError(
            "auto-posttrain must be used alone because it selects the concrete "
            "post-training stage after inspecting --data."
        )
    if "decision" in stages and stages != ["decision"]:
        raise ValueError("decision must be used alone; it includes evaluation, calibration and model save")
    if "pretrain" in stages and stages[0] != "pretrain":
        raise ValueError(
            "pretrain must be the first stage because later stages consume its model output."
        )
    duplicates = sorted(item for item in set(stages) if stages.count(item) > 1)
    if duplicates:
        raise ValueError(
            "Direct training stages must be unique: " + ", ".join(duplicates) + "."
        )
    preference_stages = [item for item in stages if item in {"dpo", "orpo", "kto"}]
    if len(preference_stages) > 1:
        raise ValueError(
            "One direct command currently supports at most one of dpo/orpo/kto; "
            "run separate commands when chaining multiple preference algorithms."
        )
    return stages


def _looks_like_direct_stage_expression(value: str) -> bool:
    parts = [item.strip().lower() for item in str(value or "").split(",")]
    return bool(parts) and all(parts) and all(
        item in _DIRECT_TRAIN_EXPRESSIONS for item in parts
    )


def _resolve_direct_stage_data(args, stages: list[str]) -> dict[str, str]:
    unused = [
        stage
        for stage in _DIRECT_TRAIN_STAGES
        if stage not in stages and getattr(args, f"{stage}_data")
    ]
    if unused:
        raise ValueError(
            "Data was provided for stage(s) not listed in --stage: "
            + ", ".join(unused)
            + "."
        )
    resolved = {
        stage: getattr(args, f"{stage}_data")
        for stage in stages
    }
    if args.data:
        first = stages[0]
        if (
            resolved[first]
            and os.path.abspath(resolved[first]) != os.path.abspath(args.data)
        ):
            raise ValueError(
                f"Data for first stage {first!r} was provided twice; use either "
                f"--data or --{first}-data."
            )
        resolved[first] = args.data
    missing = [stage for stage in stages if not resolved[stage]]
    if missing:
        flags = ", ".join(f"--{stage}-data" for stage in missing)
        raise ValueError(
            "Missing data for direct stage(s): "
            + ", ".join(missing)
            + f". Provide {flags}."
        )
    return resolved


def _build_direct_train_config(args, requested_stages: list[str]) -> dict:
    if "sft" not in requested_stages and any(value is not None for value in (args.replay_data, args.replay_ratio, args.replay_count, args.reference_model, args.retention_kl_weight, args.retention_temperature)):
        raise ValueError("Replay options require sft in --stage.")
    if args.resume_from_checkpoint is not None and len(requested_stages) != 1:
        raise ValueError(
            "--resume-from-checkpoint is unambiguous only for one direct stage; "
            "resume multi-stage workflows with an explicit YAML/JSON config."
        )
    if args.validation_data and (
        len(requested_stages) != 1
        or requested_stages[0] not in {"sft", "dpo", "orpo", "kto"}
    ):
        raise ValueError(
            "--validation-data applies to one direct SFT/DPO/ORPO/KTO stage; "
            "use stage-specific validation_data_path values in YAML for a "
            "multi-stage workflow."
        )
    if args.validation_data and args.validation_split not in {None, 0, 0.0}:
        raise ValueError(
            "Use either --validation-data or --validation-split, not both."
        )
    has_pretrain = "pretrain" in requested_stages
    pretrain_mode = args.pretrain_mode or "continue"
    if not has_pretrain and any(
        (
            args.pretrain_data,
            args.pretrain_mode,
            args.model_config,
            args.tokenizer,
            args.data_type,
            args.text_column,
            args.min_text_length,
            args.quality_min_score,
            args.dedup_method,
            args.global_batch_size,
            args.min_learning_rate,
            args.streaming,
            args.no_pack_sequences,
            args.no_append_eos,
        )
    ):
        raise ValueError(
            "Pretraining-only options require pretrain in --stage."
        )
    if has_pretrain and pretrain_mode == "scratch":
        if not args.model_config:
            raise ValueError(
                "Scratch pretraining requires --model-config for the architecture."
            )
        if not args.tokenizer:
            raise ValueError(
                "Scratch pretraining requires --tokenizer; tokenizer training remains "
                "an explicit YAML pipeline stage."
            )
    elif not args.model:
        raise ValueError("Direct training requires --model.")
    stage_data = _resolve_direct_stage_data(args, requested_stages)

    def positive(name: str, value, default):
        resolved = default if value is None else value
        if resolved <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return resolved

    method = args.method or "lora"
    output_name = "-".join(requested_stages)
    output_dir = args.output_dir or os.path.join("outputs", output_name)
    epochs = positive("--epochs", args.epochs, 1)
    learning_rate = positive(
        "--learning-rate", args.learning_rate, 3e-4 if has_pretrain else 2e-4
    )
    sft_batch_size = positive("--batch-size", args.batch_size, 1)
    preference_stage = next(
        (stage for stage in requested_stages if stage in {"dpo", "orpo", "kto"}),
        None,
    )
    preference_batch_size = positive(
        "--batch-size",
        args.batch_size,
        2 if preference_stage == "kto" else 1,
    )
    grad_accumulation = positive(
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        4,
    )
    max_seq_length = positive(
        "--max-seq-length", args.max_seq_length, 1024 if has_pretrain else 512
    )
    warmup_steps = (
        100 if has_pretrain and args.warmup_steps is None
        else 10 if args.warmup_steps is None
        else args.warmup_steps
    )
    save_steps = positive(
        "--save-steps", args.save_steps, 500 if has_pretrain else 100
    )
    logging_steps = positive("--logging-steps", args.logging_steps, 10)
    validation_split = 0.0 if args.validation_data else (
        (0.01 if has_pretrain else 0.05)
        if args.validation_split is None
        else args.validation_split
    )
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be zero or greater.")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("--validation-split must be in the range [0, 1).")
    eval_steps = args.eval_steps
    if eval_steps is None:
        has_validation = bool(args.validation_data or validation_split > 0)
        eval_steps = (100 if has_pretrain else 50) if has_validation else 0
    if eval_steps < 0:
        raise ValueError("--eval-steps must be zero or greater.")
    max_steps = (
        1000 if has_pretrain and args.max_steps is None
        else -1 if args.max_steps is None
        else args.max_steps
    )
    if max_steps == 0 or max_steps < -1:
        raise ValueError("--max-steps must be -1 or greater than zero.")
    lora_r = positive("--lora-r", args.lora_r, 8)
    lora_alpha = positive("--lora-alpha", args.lora_alpha, 16)
    lora_dropout = 0.05 if args.lora_dropout is None else args.lora_dropout
    if not 0.0 <= lora_dropout < 1.0:
        raise ValueError("--lora-dropout must be in the range [0, 1).")
    raw_target_modules = (
        args.lora_target_modules or "q_proj,k_proj,v_proj,o_proj"
    ).strip()
    target_modules = (
        "all-linear"
        if raw_target_modules.lower() == "all-linear"
        else [
            item.strip()
            for item in raw_target_modules.split(",")
            if item.strip()
        ]
    )
    if method != "full" and not target_modules:
        raise ValueError("--lora-target-modules must contain at least one module.")
    precision = args.precision or "auto"
    bf16 = precision in {"auto", "bf16"}
    fp16 = precision == "fp16"

    backend = args.backend or (
        "saddle" if has_pretrain and pretrain_mode == "scratch" else "hf"
    )
    if (
        has_pretrain
        and pretrain_mode == "scratch"
        and backend == "saddle"
        and len(requested_stages) > 1
        and method != "full"
    ):
        raise ValueError(
            "The saddle scratch backend requires --method full for chained "
            "post-training; use --backend hf if you need LoRA stages."
        )
    if (
        has_pretrain
        and pretrain_mode == "scratch"
        and backend == "saddle"
        and preference_stage not in {None, "dpo"}
    ):
        raise ValueError(
            "The saddle backend currently supports DPO only for preference training."
        )

    tokenizer_reference = args.tokenizer or args.model
    model_reference = None if pretrain_mode == "scratch" and has_pretrain else args.model
    global_batch_size = positive(
        "--global-batch-size",
        args.global_batch_size,
        sft_batch_size * grad_accumulation,
    )
    min_learning_rate = (
        learning_rate * 0.1
        if args.min_learning_rate is None
        else args.min_learning_rate
    )
    if has_pretrain and not 0 <= min_learning_rate <= learning_rate:
        raise ValueError(
            "--min-learning-rate must be between zero and --learning-rate."
        )
    quality_min_score = (
        0.3 if args.quality_min_score is None else args.quality_min_score
    )
    if not 0 <= quality_min_score <= 1:
        raise ValueError("--quality-min-score must be in the range [0, 1].")
    min_text_length = (
        50 if args.min_text_length is None else args.min_text_length
    )
    if min_text_length < 0:
        raise ValueError("--min-text-length must be zero or greater.")

    canonical_stages = [_DIRECT_STAGE_MAPPING[stage] for stage in requested_stages]
    config = {
        "project": "saddlellm-direct",
        "experiment": os.path.basename(os.path.normpath(output_dir)) or output_name,
        "seed": 42 if args.seed is None else args.seed,
        "stages": canonical_stages,
        "direct_pipeline": {
            "requested_stages": requested_stages,
            "stage_mapping": [
                {"requested": stage, "orchestrator": _DIRECT_STAGE_MAPPING[stage]}
                for stage in requested_stages
            ],
            "execution_semantics": "strict_left_to_right",
        },
        "model": {
            "name_or_path": model_reference,
            "tokenizer": tokenizer_reference,
            "backend": backend,
            "config": args.model_config or "gpt2-small-124m",
            "local_files_only": args.local_files_only,
        },
        "pretrain_mode": pretrain_mode if has_pretrain else "continue",
        "training": {
            "max_steps": max_steps,
            "resume_from_checkpoint": args.resume_from_checkpoint or False,
            "preflight_only": False,
            "dry_run": False,
        },
        "distributed": {
            "strategy": "single",
            "num_gpus": 1,
            "bf16": bf16,
            "fp16": fp16,
        },
        "logging": {
            "output_dir": output_dir,
            "backend": "local",
            "save_total_limit": 3,
        },
    }
    if has_pretrain:
        config["data"] = {
            "sources": [
                {
                    "type": args.data_type or "local",
                    "path": stage_data["pretrain"],
                    "text_column": args.text_column or "text",
                    "streaming": bool(args.streaming),
                }
            ],
            "max_seq_length": max_seq_length,
            "min_text_length": min_text_length,
            "dedup_method": args.dedup_method or "minhash",
            "quality_min_score": quality_min_score,
            "pack_sequences": not args.no_pack_sequences,
            "append_eos": not args.no_append_eos,
            "validation_split": validation_split,
        }
        config["training"].update(
            {
                "per_device_batch_size": sft_batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": learning_rate,
                "min_lr": min_learning_rate,
                "warmup_steps": warmup_steps,
                "save_every_steps": save_steps,
                "eval_every_steps": eval_steps,
                "log_every_steps": logging_steps,
                "keep_last_n": 3,
                "load_best_model_at_end": validation_split > 0,
                "dataloader_num_workers": 0 if os.name == "nt" else 4,
                "gradient_checkpointing": not args.no_gradient_checkpointing,
                "stability_monitor": True,
            }
        )
        if args.gradient_accumulation_steps is not None:
            config["distributed"]["gradient_accumulation_steps"] = (
                grad_accumulation
            )
    if "sft" in requested_stages:
        config["sft"] = {
            "enabled": True,
            "data_path": stage_data["sft"],
            "use_lora": method != "full",
            "use_qlora": method == "qlora",
            "lora": {
                "r": lora_r,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "target_modules": target_modules,
            },
            "epochs": epochs,
            "learning_rate": learning_rate,
            "per_device_batch_size": sft_batch_size,
            "max_seq_length": max_seq_length,
            "gradient_accumulation_steps": grad_accumulation,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "logging_steps": logging_steps,
            "validation_split": validation_split,
            "validation_data_path": args.validation_data,
            "load_best_model_at_end": True,
            "response_template": args.response_template or "### Answer:",
            "replay_data_path": args.replay_data,
            "replay_ratio": args.replay_ratio if args.replay_ratio is not None else 0.0,
            "replay_count": args.replay_count,
            "reference_model_path": args.reference_model,
            "retention_kl_weight": args.retention_kl_weight if args.retention_kl_weight is not None else 0.0,
            "retention_temperature": args.retention_temperature if args.retention_temperature is not None else 1.0,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "precision": precision,
        }
    if preference_stage:
        if preference_stage == "kto" and preference_batch_size <= 1:
            raise ValueError("KTO requires --batch-size greater than 1.")
        config["preference"] = {
            "enabled": True,
            "method": preference_stage,
            "data_path": stage_data[preference_stage],
            "use_lora": method != "full",
            "use_qlora": method == "qlora",
            "lora": {
                "r": lora_r,
                "alpha": lora_alpha,
                "dropout": lora_dropout,
                "target_modules": target_modules,
            },
            "epochs": epochs,
            "learning_rate": args.learning_rate or 5e-6,
            "per_device_batch_size": preference_batch_size,
            "max_length": max_seq_length,
            "max_prompt_length": max(1, max_seq_length // 2),
            "gradient_accumulation_steps": grad_accumulation,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "logging_steps": logging_steps,
            "validation_split": validation_split,
            "validation_data_path": args.validation_data,
            "load_best_model_at_end": True,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "gradient_checkpointing": not args.no_gradient_checkpointing,
            "precision": precision,
        }
    if "ppo" in requested_stages:
        config["rlhf"] = {
            "enabled": True,
            "method": "ppo",
            "data_path": stage_data["ppo"],
            "use_lora": method != "full",
            "use_qlora": method == "qlora",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "epochs": epochs,
            "learning_rate": args.learning_rate or 5e-5,
            "per_device_batch_size": positive("--batch-size", args.batch_size, 1),
            "max_length": max_seq_length,
            "max_prompt_length": max(1, max_seq_length // 2),
            "gradient_accumulation_steps": grad_accumulation,
            "warmup_steps": warmup_steps,
            "save_steps": save_steps,
        }
    return config


def _build_direct_sft_config(args) -> dict:
    """Backward-compatible internal helper for the original SFT shortcut."""
    return _build_direct_train_config(args, ["sft"])


def _build_direct_decision_config(args) -> dict:
    """Infer a decision dataset contract, keeping the original Choice defaults."""
    from pathlib import Path
    kind = args.decision_type
    descriptor = Path(args.data).expanduser() / "dataset.json" if args.data else None
    if kind is None and descriptor and descriptor.is_file():
        contract = json.loads(descriptor.read_text(encoding="utf-8"))
        kind = "typed" if contract.get("schema_version") == 2 and contract.get("task_type") == "typed" else "choice"
    if kind == "typed":
        return _build_direct_typed_decision_config(args)
    if any(getattr(args, name) is not None for name in ("decision_backbone", "decision_method", "optimizer_device", "decision_hidden_size", "decision_heads", "query_pooling", "instance_question_in_query", "backbone_learning_rate", "head_learning_rate", "head_warmup_epochs", "early_stopping_patience", "evaluate_test")):
        raise ValueError("Typed decision options require a typed dataset or --decision-type typed")
    if not args.model or not args.data:
        raise ValueError("decision requires --model LOCAL_BACKBONE and --data DATASET_DIRECTORY")
    allowed = {"command", "config", "stage", "model", "data", "output_dir", "epochs", "learning_rate", "batch_size", "max_seq_length", "precision", "seed", "local_files_only", "encoder_device", "projection_rank", "encoder_batch_size", "cpu_threads", "weight_decay", "dry_run", "debug", "parallel", "num_gpus", "num_nodes", "launcher", "backend", "decision_type"}
    unsupported = ["--" + name.replace("_", "-") for name, value in vars(args).items()
                   if name not in allowed and value is not None and value is not False]
    if unsupported:
        raise ValueError("Unsupported options for decision head training: " + ", ".join(unsupported))
    if args.backend not in {None, "hf"}:
        raise ValueError("decision requires a local HF text backbone")
    settings = {"dataset_dir": os.path.abspath(os.path.expanduser(args.data)),
                "backbone_name_or_path": os.path.abspath(os.path.expanduser(args.model)),
                "encoder_dtype": {None: "auto", "auto": "auto", "bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[args.precision]}
    for argument, field_name in {"epochs": "epochs", "learning_rate": "learning_rate", "batch_size": "batch_size", "max_seq_length": "max_length", "encoder_device": "encoder_device", "projection_rank": "projection_rank", "encoder_batch_size": "encoder_batch_size", "cpu_threads": "cpu_threads", "weight_decay": "weight_decay"}.items():
        value = getattr(args, argument)
        if value is not None:
            settings[field_name] = value
    return {"project": "saddlellm", "experiment": "decision", "stages": ["decision"],
            "seed": args.seed if args.seed is not None else 42,
            "model": {"backend": "hf", "name_or_path": settings["backbone_name_or_path"], "local_files_only": True},
            "decision": settings,
            "logging": {"output_dir": args.output_dir or os.path.join("outputs", "decision"), "logging_backend": "local"},
            "eval": {"enabled": False}, "distributed": {"strategy": "single"},
            "direct_pipeline": {"requested_stages": ["decision"], "stage_mapping": [{"requested": "decision", "orchestrator": "decision"}]}}


def _build_direct_typed_decision_config(args):
    if not args.data:
        raise ValueError("typed decision requires --data DATASET_DIRECTORY")
    backbone = args.decision_backbone or ("hf" if args.model else "native")
    if backbone == "hf" and not args.model:
        raise ValueError("typed HF training requires --model LOCAL_BACKBONE")
    if backbone == "native" and args.model:
        raise ValueError("native typed training does not use --model; choose --decision-backbone hf to train that model")
    if args.backend not in {None, "hf"}:
        raise ValueError("typed HF/native selection uses --decision-backbone, not the causal LM --backend")
    if args.precision not in {None, "auto", "fp32"}:
        raise ValueError("typed training currently uses FP32 weights; use --optimizer-device cpu for lower GPU memory")
    allowed = {"command", "config", "stage", "data", "model", "output_dir", "epochs", "learning_rate", "batch_size", "max_seq_length", "precision", "seed",
               "local_files_only", "encoder_device", "cpu_threads", "weight_decay", "dry_run", "debug", "parallel", "num_gpus", "num_nodes", "launcher", "backend",
               "decision_type", "decision_backbone", "decision_method", "optimizer_device", "decision_hidden_size", "decision_heads",
               "query_pooling", "instance_question_in_query", "backbone_learning_rate", "head_learning_rate", "head_warmup_epochs", "early_stopping_patience", "evaluate_test"}
    unsupported = ["--" + name.replace("_", "-") for name, value in vars(args).items() if name not in allowed and value is not None and value is not False]
    if unsupported:
        raise ValueError("Unsupported options for typed decision training: " + ", ".join(unsupported))
    model_path = os.path.abspath(os.path.expanduser(args.model)) if args.model else "native-byte-transformer"
    settings = {"task_type": "typed", "typed_backbone": backbone, "dataset_dir": os.path.abspath(os.path.expanduser(args.data))}
    mapping = {"epochs": "epochs", "learning_rate": "learning_rate", "batch_size": "batch_size", "max_seq_length": "max_length", "encoder_device": "encoder_device",
               "cpu_threads": "cpu_threads", "weight_decay": "weight_decay", "decision_method": "method", "optimizer_device": "optimizer_device",
               "decision_hidden_size": "hidden_size", "decision_heads": "num_heads", "query_pooling": "query_pooling", "instance_question_in_query": "instance_question_in_query",
               "backbone_learning_rate": "backbone_learning_rate", "head_learning_rate": "head_learning_rate",
               "head_warmup_epochs": "head_warmup_epochs", "early_stopping_patience": "early_stopping_patience", "evaluate_test": "evaluate_test"}
    for name, field in mapping.items():
        if getattr(args, name) is not None:
            settings[field] = getattr(args, name)
    return {"project": "saddlellm", "experiment": "typed_decision", "seed": args.seed if args.seed is not None else 42,
            "stages": ["decision"], "model": {"backend": "hf", "name_or_path": model_path, "local_files_only": True},
            "decision": settings, "distributed": {"strategy": "single"}, "eval": {"enabled": False},
            "logging": {"output_dir": args.output_dir or "outputs/decision_typed", "logging_backend": "local"},
            "direct_pipeline": {"requested_stages": ["decision"], "stage_mapping": [{"requested": "decision", "orchestrator": "decision"}]}}


def _resolve_train_request(args):
    config_argument = args.config
    stage = args.stage
    if (
        config_argument
        and not os.path.isfile(config_argument)
        and _looks_like_direct_stage_expression(config_argument)
    ):
        if stage:
            raise ValueError("Do not specify the training stage twice.")
        stage = config_argument
        config_argument = None

    if config_argument:
        if _direct_train_option_was_used(args):
            raise ValueError(
                "Use either a YAML/JSON config or direct --stage/--model/--data "
                "options, not both."
            )
        return _load_training_config(config_argument), config_argument, "file"
    if not stage:
        raise ValueError(
            "Provide a YAML/JSON config, or use: saddlellm train --stage sft "
            "--model MODEL --data DATA."
        )
    requested_stages = _parse_direct_stages(stage)
    args.stage = ",".join(requested_stages)
    if requested_stages == ["decision"]:
        return _build_direct_decision_config(args), "<command-line:decision>", "command-line"
    if any(getattr(args, name) is not None for name in ("encoder_device", "projection_rank", "encoder_batch_size", "cpu_threads", "weight_decay", "decision_type", "decision_backbone", "decision_method", "optimizer_device", "decision_hidden_size", "decision_heads", "query_pooling", "instance_question_in_query", "backbone_learning_rate", "head_learning_rate", "head_warmup_epochs", "early_stopping_patience", "evaluate_test")):
        raise ValueError("Decision-specific options require --stage decision")
    if requested_stages == ["auto-posttrain"]:
        if not args.data:
            raise ValueError("auto-posttrain requires --data for schema inspection.")
        stage_specific_data = [
            name
            for name in _DIRECT_TRAIN_STAGES
            if getattr(args, f"{name}_data")
        ]
        if stage_specific_data:
            raise ValueError(
                "auto-posttrain accepts one --data file only; stage-specific data "
                "belongs to an explicit ordered --stage command."
            )
        from .data.TrainingDataInspector import TrainingDataInspector

        decision = TrainingDataInspector.select_post_training(args.data).to_dict()
        selected_stage = decision["selected_stage"]
        config = _build_direct_train_config(args, [selected_stage])
        if args.output_dir is None:
            config["experiment"] = "auto-posttrain"
            config["logging"]["output_dir"] = os.path.join(
                "outputs", "auto-posttrain"
            )
        config["direct_pipeline"].update(
            {
                "requested_stages": ["auto-posttrain"],
                "selected_stages": [selected_stage],
                "stage_mapping": [
                    {
                        "requested": "auto-posttrain",
                        "selected": selected_stage,
                        "orchestrator": _DIRECT_STAGE_MAPPING[selected_stage],
                    }
                ],
                "auto_decision": decision,
            }
        )
        config["auto_posttrain"] = decision
        return config, "<command-line:auto-posttrain>", "command-line"
    return (
        _build_direct_train_config(args, requested_stages),
        f"<command-line:{args.stage}>",
        "command-line",
    )


def _save_direct_train_config(config: dict) -> str:
    output_dir = config.get("logging", {}).get("output_dir", "./outputs/sft")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(output_dir, "command_config.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _cmd_train(args) -> int:
    config_path = args.config or "<command-line>"
    try:
        config, config_path, config_source = _resolve_train_request(args)
        decision_training_plan = None
        if "decision" in config.get("stages", []):
            from .decision.config import decision_plan
            from .training.TrainingOrchestrator import TrainingOrchestrator

            decision_training_plan = decision_plan(TrainingOrchestrator._parse_config(config))
        # A dry run must not report success for a malformed native blueprint.
        # Resolve structure only: no model allocation, tokenizer/data loading,
        # optional runtime import or output-directory creation is required.
        model_options = config.get("model") or {}
        if model_options.get("blueprint") or config.get("model_blueprint"):
            from .training.TrainingOrchestrator import TrainingOrchestrator

            parsed = TrainingOrchestrator._parse_config(config)
            if parsed.model_backend != "saddle":
                raise ValueError("model.blueprint requires model.backend='saddle'; HF cannot consume native layer architecture")
            TrainingOrchestrator.resolve_architecture_blueprint(
                parsed.model_config, parsed.model_blueprint
            )
        if config.get("backends"):
            from .training.TrainingOrchestrator import TrainingOrchestrator
            from .runtime.BackendConfig import validate_runtime_backends
            validate_runtime_backends(TrainingOrchestrator._parse_config(config))
            if config["backends"].get("training", "default") == "megatron" and any(
                value is not None for value in (args.parallel, args.num_gpus, args.num_nodes, args.launcher)
            ):
                raise ValueError("Megatron owns torchrun; configure topology in backends.megatron instead of --parallel")
        distributed = config.get("distributed", {})
        configured_strategy = (
            str(distributed.get("strategy", "single")).strip().lower()
            if isinstance(distributed, dict)
            else "single"
        )
        parallel_requested = bool(
            args.parallel is not None
            or args.num_gpus is not None
            or args.num_nodes is not None
            or args.launcher is not None
            or configured_strategy not in {"", "single"}
        )
        parallel_decision = None
        if parallel_requested:
            from .training.AutoParallel import (
                apply_parallel_decision,
                create_parallel_launch_plan,
                materialize_parallel_config,
                parallel_artifact_directory,
                plan_parallel_training,
                run_launch_plan,
            )

            parallel_decision = plan_parallel_training(
                config,
                strategy=args.parallel,
                num_gpus=args.num_gpus,
                num_nodes=args.num_nodes,
                launcher=args.launcher,
            )
            parallel_decision.require_executable()
            config = apply_parallel_decision(config, parallel_decision)
            is_parallel_child = os.environ.get("SADDLELLM_PARALLEL_CHILD") == "1"
            if (
                is_parallel_child
                and parallel_decision.strategy != "single"
                and not parallel_decision.already_launched
            ):
                raise RuntimeError(
                    "parallel worker marker is present but WORLD_SIZE is not initialized; "
                    "the launcher failed before worker startup"
                )

            # A controller starts workers exactly once. Inside torchrun or
            # Accelerate, WORLD_SIZE is already present and each worker moves
            # straight on to TrainingOrchestrator.
            if (
                not args.dry_run
                and parallel_decision.strategy != "single"
                and not parallel_decision.already_launched
                and not is_parallel_child
            ):
                parallel_dir = parallel_artifact_directory(
                    config, parallel_decision
                )
                resolved_path = materialize_parallel_config(
                    config,
                    parallel_decision,
                    output_path=os.path.join(parallel_dir, "resolved_config.json"),
                )
                launch_plan = create_parallel_launch_plan(
                    resolved_path, parallel_decision
                )
                if args.debug:
                    launch_plan.command.append("--debug")
                from .factory.BackendAdapters import BackendAdapterRegistry

                launch_plan_path = os.path.join(parallel_dir, "launch_plan.json")
                launch_script_path = os.path.join(parallel_dir, "launch.ps1")
                BackendAdapterRegistry.save_launch_script(
                    launch_plan, launch_script_path
                )
                BackendAdapterRegistry.save_launch_plan(
                    launch_plan, launch_plan_path
                )
                launch_status = {
                    "ok": True,
                    "command": "train",
                    "status": "launching",
                    "config": resolved_path,
                    "parallel": parallel_decision.to_dict(),
                    "launch_plan": launch_plan_path,
                    "launch_script": launch_script_path,
                    "launch_command": launch_plan.command,
                }
                print(
                    json.dumps(launch_status, ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
                returncode = run_launch_plan(launch_plan)
                if returncode:
                    _emit_json(
                        {
                            "ok": False,
                            "command": "train",
                            "status": "launcher_failed",
                            "config": resolved_path,
                            "parallel": parallel_decision.to_dict(),
                            "launch_plan": launch_plan_path,
                            "returncode": returncode,
                        }
                    )
                return returncode

        if args.dry_run:
            from .framework import PipelinePlan, get_stage_registry

            pipeline_config = config.get("pipeline", {})
            dependencies = None
            if isinstance(pipeline_config, dict):
                dependencies = pipeline_config.get(
                    "dependencies", pipeline_config.get("depends_on")
                )
            stage_names = config.get("stages", [])
            stage_registry = get_stage_registry()
            capabilities = {
                stage: stage_registry.create(stage).capabilities
                for stage in stage_names
                if stage_registry.contains(stage)
            }
            pipeline_plan = PipelinePlan.compile(
                stage_names,
                dependencies=dependencies,
                capabilities=capabilities,
            )
            payload = {
                "config": config_path,
                "config_source": config_source,
                "stages": config.get("direct_pipeline", {}).get(
                    "requested_stages", list(pipeline_plan.stages)
                ),
                "orchestrator_stages": list(pipeline_plan.stages),
                "pipeline_plan": pipeline_plan.to_dict(),
                "distributed": config.get("distributed", {}),
                "data_sources": len(config.get("data", {}).get("sources", [])),
            }
            if config.get("backends"):
                payload["backends"] = config["backends"]
                payload["backend_notes"] = [
                    "Optional runtimes are not loaded or started in dry-run.",
                    "Megatron recipes/overrides own its training settings; there is no fallback.",
                ]
            if parallel_decision is not None:
                payload["parallel"] = parallel_decision.to_dict()
            if config.get("direct_pipeline"):
                payload["stage_mapping"] = config["direct_pipeline"]["stage_mapping"]
                if config["direct_pipeline"].get("selected_stages"):
                    payload["selected_stages"] = config["direct_pipeline"][
                        "selected_stages"
                    ]
                if config["direct_pipeline"].get("auto_decision"):
                    payload["auto_decision"] = config["direct_pipeline"][
                        "auto_decision"
                    ]
            if config_source == "command-line":
                payload["resolved_config"] = config
            if decision_training_plan is not None:
                payload["decision"] = decision_training_plan
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if config_source == "command-line":
            config_path = _save_direct_train_config(config)
        from .training.TrainingOrchestrator import TrainingOrchestrator

        orchestrator = TrainingOrchestrator.from_dict(config)
        results = orchestrator.run()
        output_dir = config.get("logging", {}).get("output_dir", "./outputs")
        payload = {
            "ok": True,
            "command": "train",
            "config": config_path,
            "output_dir": output_dir,
            "next": {
                "dashboard": f'saddlellm dashboard "{output_dir}" --open',
            },
            "results": results,
        }
        if "decision" in results and results["decision"].get("model_path"):
            payload["next"]["serve"] = f'saddlellm serve-decision "{results["decision"]["model_path"]}"'
        if config.get("auto_posttrain"):
            payload["auto_decision"] = config["auto_posttrain"]
        if parallel_decision is not None:
            payload["parallel"] = parallel_decision.to_dict()
        if os.environ.get("RANK", "0") in {"", "0"}:
            _emit_json(payload)
        return 0
    except Exception as exc:
        if args.debug:
            raise
        error_payload = {
            "ok": False,
            "command": "train",
            "config": config_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": "Run the same command with --debug to show the full traceback.",
        }
        is_managed_child = os.environ.get("SADDLELLM_PARALLEL_CHILD") == "1"
        is_distributed_worker = (
            os.environ.get("WORLD_SIZE", "1") not in {"", "1"}
            and os.environ.get("RANK") not in {None, ""}
        )
        if is_managed_child:
            if os.environ.get("RANK", "0") in {"", "0"}:
                print(
                    json.dumps(error_payload, ensure_ascii=False, indent=2),
                    file=sys.stderr,
                )
        elif is_distributed_worker:
            if os.environ.get("RANK", "0") in {"", "0"}:
                _emit_json(error_payload)
        else:
            _emit_json(error_payload)
    return 1


def _require_new_output_directory(path: str) -> str:
    """Return an absolute output path without overwriting an existing run."""
    from pathlib import Path

    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Choose a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    return str(output)


def _artifact_bytes(path: str) -> int:
    from pathlib import Path

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def _validate_distill_numbers(args) -> None:
    for name in (
        "student_layers",
        "student_hidden_size",
        "student_heads",
        "max_steps",
        "batch_size",
        "max_seq_length",
        "max_train_samples",
        "max_eval_samples",
        "eval_batch_size",
        "eval_max_length",
        "eval_max_new_tokens",
        "epochs",
        "lora_r",
        "lora_alpha",
        "concurrency",
        "max_new_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be greater than zero.")
    if args.eval_limit < 0:
        raise ValueError("--eval-limit cannot be negative.")
    if args.eval_generation_samples < 0:
        raise ValueError("--eval-generation-samples cannot be negative.")
    if len(set(args.eval_choices)) < 2:
        raise ValueError("--eval-choices must contain at least two unique labels.")
    if args.eval_variant_field and not args.eval_case_field:
        raise ValueError("--eval-variant-field requires --eval-case-field.")
    if args.eval_prompt_variants and (
        args.eval_case_field or args.eval_variant_field
    ):
        raise ValueError(
            "--eval-prompt-variants cannot be combined with explicit eval case/variant fields."
        )
    if args.student_hidden_size % args.student_heads:
        raise ValueError("--student-hidden-size must be divisible by --student-heads.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero.")
    if not 0.0 <= args.teacher_temperature <= 2.0:
        raise ValueError("--teacher-temperature must be in the range [0, 2].")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in the range [0, 1].")


def _build_data_distill_config(args, output_dir: str) -> dict:
    if not args.student:
        raise ValueError("data distillation requires --student.")
    if args.teacher_data and (args.teacher_url or args.data):
        raise ValueError(
            "Use either --teacher-data, or --teacher-url with --data prompts, not both."
        )
    if not args.teacher_data:
        if not args.teacher_url:
            raise ValueError("data distillation requires --teacher-url.")
        if not args.data:
            raise ValueError("data distillation requires prompt records in --data.")

    use_lora = args.method != "full"
    use_qlora = args.method == "qlora"
    teacher_data = args.teacher_data
    # Endpoint output is deliberately collected first and never trained in the
    # same command. This makes clinician review a visible step instead of
    # silently teaching unsafe or hallucinated answers to the student.
    stages = ["sft"] if teacher_data else ["mopd"]

    config = {
        "project": "saddlellm-distill",
        "experiment": os.path.basename(os.path.normpath(output_dir)) or "distill",
        "seed": 42,
        "stages": stages,
        "model": {
            "name_or_path": args.student,
            "tokenizer": args.student,
            "backend": "hf",
            "local_files_only": args.local_files_only,
        },
        "pretrain_mode": "continue",
        "training": {
            "max_steps": args.max_steps,
            "dry_run": False,
            "preflight_only": False,
        },
        "distributed": {
            "strategy": "single",
            "num_gpus": 1,
            "bf16": args.device != "cpu",
            "fp16": False,
        },
        "logging": {
            "output_dir": output_dir,
            "backend": "local",
            "save_total_limit": 3,
        },
        "distillation": {
            "mode": "reviewed_teacher_data" if teacher_data else "on_policy_data",
            "teacher_data": teacher_data,
            "requires_review": not bool(teacher_data),
        },
    }
    if not teacher_data:
        config["mopd"] = {
            "enabled": True,
            "prompts_path": args.data,
            "output_dir": os.path.join(output_dir, "teacher_data"),
            "teachers": [
                {
                    "name": args.teacher or "teacher",
                    "model": args.teacher or "teacher",
                    "type": "endpoint",
                    "endpoint_url": args.teacher_url,
                    "api_key_env": args.api_key_env,
                    "domain": args.domain,
                }
            ],
            "num_rollouts_per_prompt": 1,
            "max_new_tokens": args.max_new_tokens,
            "student_temperature": 0.0,
            "teacher_temperature": args.teacher_temperature,
            "teacher_concurrency": args.concurrency,
            "aggregation": "best_score",
            "requires_review": True,
            "dry_run": False,
        }
    if "sft" in stages:
        config["sft"] = {
            "enabled": True,
            "data_path": teacher_data or "",
            "use_lora": use_lora,
            "use_qlora": use_qlora,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 10,
            "save_steps": 100,
            "eval_steps": 50,
            "logging_steps": 10,
            "validation_split": 0.05,
            "response_template": "### Answer:",
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
            "gradient_checkpointing": True,
        }
    return config


def _distillation_evaluation_plan(args, *, deferred: bool = False) -> dict:
    if not args.eval_data:
        return {
            "status": "not_requested",
            "hint": "Pass --eval-data with an independent benchmark to compare baseline and student.",
        }
    return {
        "status": "deferred_until_reviewed_training" if deferred else "planned",
        "data": os.path.abspath(args.eval_data),
        "task": args.eval_task,
        "limit": args.eval_limit,
        "generation_samples": args.eval_generation_samples,
        "case_field": args.eval_case_field,
        "variant_field": args.eval_variant_field,
        "prompt_variants": (
            os.path.abspath(args.eval_prompt_variants)
            if args.eval_prompt_variants
            else None
        ),
        "output_dir": os.path.abspath(os.path.join(args.output_dir, "evaluation")),
        "note": (
            "Online teacher data must be reviewed and trained before a student can be evaluated."
            if deferred
            else "The pre-distillation baseline and trained student use the same data and settings."
        ),
    }


def _run_distillation_evaluation(
    args,
    *,
    baseline_model: str,
    candidate_model: str,
    output_dir: str,
    candidate_adapter: Optional[str] = None,
) -> dict:
    from .evaluation.DistillationEvaluator import (
        DistillationEvaluationRequest,
        evaluate_distillation_pair,
    )

    request = DistillationEvaluationRequest(
        baseline_model=baseline_model,
        candidate_model=candidate_model,
        candidate_adapter=candidate_adapter,
        data=args.eval_data,
        output_dir=os.path.join(output_dir, "evaluation"),
        task=args.eval_task,
        choices=args.eval_choices,
        case_field=args.eval_case_field,
        variant_field=args.eval_variant_field,
        prompt_variants=args.eval_prompt_variants,
        limit=args.eval_limit,
        batch_size=args.eval_batch_size,
        max_length=args.eval_max_length,
        max_new_tokens=args.eval_max_new_tokens,
        generation_samples=args.eval_generation_samples,
        device=args.device,
        dtype="auto",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    return evaluate_distillation_pair(request)


def _logits_internal_evaluation(result: dict) -> dict:
    metrics = result.get("metrics", {})
    initial = metrics.get("initial_student_perplexity")
    final = metrics.get("perplexity")
    outcome = "unavailable"
    if initial is not None and final is not None:
        outcome = "improved" if final < initial else (
            "degraded" if final > initial else "unchanged"
        )
    return {
        "status": "completed" if final is not None else "unavailable",
        "scope": "internal held-out language-model evaluation",
        "outcome": outcome,
        "teacher_perplexity": metrics.get("teacher_perplexity"),
        "initial_student_perplexity": initial,
        "distilled_student_perplexity": final,
        "eval_samples": result.get("lineage", {}).get("evalSamples"),
        "limitations": [
            "This holdout comes from the supplied distillation corpus.",
            "Perplexity does not establish domain correctness, safety, or release readiness.",
        ],
    }


def _cmd_distill(args) -> int:
    _validate_distill_numbers(args)
    if args.mode == "logits":
        if not args.teacher:
            raise ValueError("logits distillation requires --teacher.")
        if not args.data:
            raise ValueError("logits distillation requires --data.")
        if args.teacher_url or args.teacher_data or args.student:
            raise ValueError(
                "--teacher-url, --teacher-data and --student belong to --mode data."
            )
        output_dir = os.path.abspath(args.output_dir)
        config = {
            "model": args.teacher,
            "datasetPath": args.data,
            "studentLayers": args.student_layers,
            "studentHiddenSize": args.student_hidden_size,
            "studentHeads": args.student_heads,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "maxSteps": args.max_steps,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "maxLength": args.max_seq_length,
            "maxTrainSamples": args.max_train_samples,
            "maxEvalSamples": args.max_eval_samples,
            "device": args.device,
            "localFilesOnly": args.local_files_only,
            "trustRemoteCode": args.trust_remote_code,
        }
        if args.dry_run:
            _emit_json(
                {
                    "ok": True,
                    "command": "distill",
                    "mode": "logits",
                    "status": "planned",
                    "resolved_config": config,
                    "output_dir": output_dir,
                    "evaluation": _distillation_evaluation_plan(args),
                }
            )
            return 0
        output_dir = _require_new_output_directory(output_dir)
        from .agents.RealWorldOperators import run_large_model_operator

        result = run_large_model_operator("Distillation", config, None, output_dir)
        payload = {
            "ok": result.get("status") == "success",
            "command": "distill",
            "mode": "logits",
            "output_dir": output_dir,
            "result": result,
            "evaluation": {
                "internal": _logits_internal_evaluation(result),
                "external": _distillation_evaluation_plan(args),
            },
        }
        if payload["ok"] and result.get("model_path"):
            payload["next"] = {
                "evaluate": f'saddlellm evaluate "{result["model_path"]}" DATA.jsonl',
                "serve": f'saddlellm serve "{result["model_path"]}" --open',
            }
            if args.eval_data:
                try:
                    payload["evaluation"]["external"] = _run_distillation_evaluation(
                        args,
                        baseline_model=args.teacher,
                        candidate_model=result["model_path"],
                        output_dir=output_dir,
                    )
                    payload["next"].pop("evaluate", None)
                except Exception as exc:
                    payload["ok"] = False
                    payload["evaluation"]["external"] = {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "artifact_available": result["model_path"],
                    }
        _emit_json(payload)
        return 0 if payload["ok"] else 1

    output_dir = os.path.abspath(args.output_dir)
    config = _build_data_distill_config(args, output_dir)
    if args.dry_run:
        _emit_json(
            {
                "ok": True,
                "command": "distill",
                "mode": "data",
                "status": "planned",
                "resolved_config": config,
                "output_dir": output_dir,
                "evaluation": _distillation_evaluation_plan(
                    args, deferred=not bool(args.teacher_data)
                ),
            }
        )
        return 0
    output_dir = _require_new_output_directory(output_dir)
    config["logging"]["output_dir"] = output_dir
    config_path = os.path.join(output_dir, "distill_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    from .training.TrainingOrchestrator import TrainingOrchestrator

    results = TrainingOrchestrator.from_dict(config).run()
    payload = {
        "ok": True,
        "command": "distill",
        "mode": "data",
        "config": config_path,
        "output_dir": output_dir,
        "results": results,
        "next": {"dashboard": f'saddlellm dashboard "{output_dir}" --open'},
        "evaluation": _distillation_evaluation_plan(
            args, deferred=not bool(args.teacher_data)
        ),
    }
    teacher_path = results.get("mopd", {}).get("data_path")
    if teacher_path:
        payload["review_required"] = True
        payload["next"].update(
            {
                "review": f'Review and correct "{teacher_path}" with a domain expert.',
                "train_reviewed": (
                    f'saddlellm distill --mode data --student "{args.student}" '
                    f'--teacher-data "{teacher_path}" --output-dir "{output_dir}-trained"'
                ),
            }
        )
    elif args.teacher_data:
        candidate_path = results.get("sft", {}).get("model_path")
        if args.eval_data and candidate_path:
            candidate_is_adapter = args.method in {"lora", "qlora"}
            try:
                payload["evaluation"] = _run_distillation_evaluation(
                    args,
                    baseline_model=args.student,
                    candidate_model=(args.student if candidate_is_adapter else candidate_path),
                    candidate_adapter=(candidate_path if candidate_is_adapter else None),
                    output_dir=output_dir,
                )
            except Exception as exc:
                payload["ok"] = False
                payload["evaluation"] = {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "artifact_available": candidate_path,
                }
        elif candidate_path:
            payload["next"]["evaluate"] = (
                f'saddlellm evaluate "{args.student}" DATA.jsonl '
                f'--adapter "{candidate_path}"'
                if args.method in {"lora", "qlora"}
                else f'saddlellm evaluate "{candidate_path}" DATA.jsonl'
            )
    _emit_json(payload)
    return 0 if payload["ok"] else 1


def _check_lifecycle_report_output(output, *, sources=(), model_directories=()):
    """Reports may never replace source data, configurations or checkpoints."""
    if not output:
        return
    from pathlib import Path

    destination = Path(output).expanduser().resolve()
    for source in sources:
        if source and destination == Path(source).expanduser().resolve():
            raise ValueError("Report output must differ from input data/configuration paths.")
    for source in model_directories:
        if source:
            directory = Path(source).expanduser().resolve()
            if directory == destination or directory in destination.parents:
                raise ValueError("Command report output must be outside checkpoint directories.")


def _cmd_regression(args) -> int:
    from .evaluation.RegressionSuite import load_regression_suite, run_regression_suite

    _check_lifecycle_report_output(args.output, sources=(args.suite,))
    suite = load_regression_suite(args.suite)
    if args.dry_run:
        _emit_json({
            "ok": True, "command": "regression", "status": "planned",
            "baseline": args.baseline, "candidate": args.candidate,
            "baseline_adapter": args.baseline_adapter, "candidate_adapter": args.candidate_adapter,
            "suite": suite, "output": os.path.abspath(args.output),
        })
        return 0
    report = run_regression_suite(
        args.baseline, args.candidate, suite,
        baseline_adapter=args.baseline_adapter,
        candidate_adapter=args.candidate_adapter, output=args.output,
    )
    accepted = report.get("accepted") is True
    _emit_json({"ok": accepted, "command": "regression", **report})
    return 0 if accepted else 1


def _cmd_recover(args) -> int:
    from .compression.RecoveryTrainer import (
        RecoveryConfig,
        run_recovery,
        validate_recovery_config,
    )

    _check_lifecycle_report_output(
        args.output, sources=(args.data, args.validation_data, args.suite),
        model_directories=(args.model, args.teacher, args.output_dir),
    )
    config = RecoveryConfig(
        student_checkpoint=args.model, train_data=args.data, output_dir=args.output_dir,
        teacher_checkpoint=args.teacher, validation_data=args.validation_data,
        validation_split=(args.validation_split if args.validation_split is not None else 0.0 if args.validation_data else 0.1),
        max_steps=args.max_steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length, temperature=args.temperature,
        distill_weight=args.distill_weight, seed=args.seed, device=args.device,
        local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code,
    )
    plan = validate_recovery_config(config)
    suite = None
    if args.suite:
        from .evaluation.RegressionSuite import load_regression_suite

        suite = load_regression_suite(args.suite)
    if args.dry_run:
        _emit_json({
            **plan, "ok": True, "command": "recover", "status": "planned",
            "regression_suite": suite, "regression_baseline": args.baseline or args.teacher or args.model,
        }, args.output)
        return 0
    report = run_recovery(config)
    payload = {**report, "ok": True, "command": "recover"}
    if suite is not None:
        from .evaluation.RegressionSuite import run_regression_suite

        gate_path = os.path.join(report["output_dir"], "regression_report.json")
        regression = run_regression_suite(
            args.baseline or args.teacher or args.model, report["output_dir"], suite, output=gate_path,
        )
        payload["regression"] = regression
        payload["ok"] = regression.get("accepted") is True
    _emit_json(payload, args.output)
    return 0 if payload["ok"] else 1


def _cmd_quantize(args) -> int:
    new_options = (
        args.calibration_data, args.bits, args.group_size, args.calibration_samples,
        args.train_data, args.validation_data, args.validation_split, args.max_steps,
        args.batch_size, args.learning_rate, args.max_seq_length, args.seed, args.device,
    )
    if args.config or args.method in {"awq", "gptq", "qat"}:
        return _cmd_configured_quantize(args)
    if any(value is not None for value in new_options) or args.suite:
        raise ValueError("These quantization options require --method awq|gptq|qat or --config.")
    output_dir = os.path.abspath(args.output_dir)
    source_bytes = _artifact_bytes(args.model) if os.path.exists(args.model) else None
    plan = {
        "source_model": args.model,
        "method": "dynamic",
        "precision": "int8",
        "device": "cpu",
        "output_dir": output_dir,
        "serve_compatible": True,
        "limitations": [
            "CPU dynamic INT8 is intended for CPU deployment.",
            "For CUDA 4/8-bit loading use saddlellm serve MODEL --quantization 4bit|8bit.",
        ],
    }
    if args.dry_run:
        _emit_json({"ok": True, "command": "quantize", "status": "planned", **plan}, args.output)
        return 0
    output_dir = _require_new_output_directory(output_dir)
    from .compression.quantizer import ModelQuantizer

    source_model = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    quantizer = ModelQuantizer(
        model_name=source_model,
        method="dynamic",
        precision="int8",
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    quantizer.quantize()
    quantizer.save(output_dir)
    artifact_bytes = _artifact_bytes(output_dir)
    result = {
        "ok": True,
        "command": "quantize",
        "status": "completed",
        **plan,
        "output_dir": output_dir,
        "source_bytes": source_bytes,
        "artifact_bytes": artifact_bytes,
        "size_change_ratio": (
            (artifact_bytes - source_bytes) / source_bytes if source_bytes else None
        ),
        "next": {"serve": f'saddlellm serve "{output_dir}" --device cpu --open'},
    }
    _emit_json(result, args.output)
    return 0


def _cmd_configured_quantize(args) -> int:
    from .compression.QuantizationPipeline import apply_quantization

    route_options = (
        args.calibration_data, args.bits, args.group_size, args.calibration_samples,
        args.train_data, args.validation_data, args.validation_split, args.max_steps,
        args.batch_size, args.learning_rate, args.max_seq_length, args.seed, args.device,
    )
    if args.config:
        if args.method is not None or any(value is not None for value in route_options):
            raise ValueError("Use either --config or explicit quantization route options.")
        config = _load_config(args.config)
    elif args.method in {"awq", "gptq"}:
        if any(value is not None for value in (args.train_data, args.validation_data, args.validation_split, args.max_steps, args.batch_size, args.learning_rate)):
            raise ValueError("Training/validation options require --method qat.")
        config = {"ptq": {"enabled": True, "method": args.method, "calibration_path": args.calibration_data}}
        for argument, key in (("bits", "bits"), ("group_size", "group_size"), ("calibration_samples", "max_samples"), ("max_seq_length", "max_seq_length"), ("seed", "seed"), ("device", "device")):
            if getattr(args, argument) is not None:
                config["ptq"][key] = getattr(args, argument)
    else:
        if any(value is not None for value in (args.calibration_data, args.bits, args.group_size, args.calibration_samples)):
            raise ValueError("Calibration/bit-width options require AWQ/GPTQ; QAT uses its named INT8 profile.")
        config = {"qat": {"enabled": True, "train_data": args.train_data}}
        for argument in ("validation_data", "validation_split", "max_steps", "batch_size", "learning_rate", "max_seq_length", "seed", "device"):
            if getattr(args, argument) is not None:
                config["qat"][argument] = getattr(args, argument)
        if args.validation_data and args.validation_split is None:
            config["qat"]["validation_split"] = 0.0
    if not isinstance(config, dict):
        raise ValueError("Quantization config must be a ptq/qat mapping.")
    # Command flags explicitly set to true may tighten the file configuration.
    for section in ("ptq", "qat"):
        if section in config and isinstance(config[section], dict):
            for flag in ("local_files_only", "trust_remote_code"):
                if getattr(args, flag):
                    config[section][flag] = True
    _check_lifecycle_report_output(
        args.output, sources=tuple(value for value in (
            args.config, args.calibration_data, args.train_data, args.validation_data, args.suite,
            (config.get("ptq") or {}).get("calibration_path") if isinstance(config.get("ptq"), dict) else None,
            (config.get("ptq") or {}).get("calibration_data") if isinstance(config.get("ptq"), dict) else None,
            (config.get("qat") or {}).get("train_data") if isinstance(config.get("qat"), dict) else None,
            (config.get("qat") or {}).get("validation_data") if isinstance(config.get("qat"), dict) else None,
        ) if isinstance(value, (str, os.PathLike))),
        model_directories=(args.model, args.output_dir, (config.get("qat") or {}).get("teacher_checkpoint") if isinstance(config.get("qat"), dict) else None),
    )
    suite = None
    if args.suite:
        from .evaluation.RegressionSuite import load_regression_suite

        suite = load_regression_suite(args.suite)
    result = apply_quantization(args.model, config=config, output_dir=args.output_dir, dry_run=args.dry_run)
    payload = {**result.to_dict(), "ok": True, "command": "quantize"}
    if suite and result.output_dir and not args.dry_run:
        from .evaluation.RegressionSuite import run_regression_suite

        regression = run_regression_suite(args.model, result.output_dir, suite, output=os.path.join(result.output_dir, "regression_report.json"))
        payload.update(regression=regression, ok=regression.get("accepted") is True)
    _emit_json(payload, args.output)
    return 0 if payload["ok"] else 1


def _cmd_prune(args) -> int:
    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be between zero and one.")
    targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not targets:
        raise ValueError("--target-modules must contain at least one module name.")
    output_dir = os.path.abspath(args.output_dir)
    plan = {
        "source_model": args.model,
        "method": args.method,
        "requested_sparsity": args.ratio,
        "target_modules": targets,
        "output_dir": output_dir,
        "serve_compatible": True,
        "warnings": [
            "Pruning changes model outputs and must be evaluated separately.",
            "Dense Hugging Face files usually do not shrink or run faster merely because weights are zero.",
        ],
    }
    if args.dry_run:
        _emit_json({"ok": True, "command": "prune", "status": "planned", **plan}, args.output)
        return 0
    output_dir = _require_new_output_directory(output_dir)
    from .compression.pruner import ModelPruner

    source_model = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    pruner = ModelPruner(
        model_name=source_model,
        pruning_method=args.method,
        pruning_ratio=args.ratio,
        device_map="cpu",
        target_modules=targets,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    pruner.prune()
    achieved = sum(int((mask == 0).sum().item()) for mask in pruner.pruning_masks.values())
    selected = sum(int(mask.numel()) for mask in pruner.pruning_masks.values())
    pruner.save(output_dir)
    result = {
        "ok": True,
        "command": "prune",
        "status": "completed",
        **plan,
        "output_dir": output_dir,
        "achieved_sparsity": achieved / selected if selected else 0.0,
        "selected_parameters": selected,
        "artifact_bytes": _artifact_bytes(output_dir),
        "next": {
            "evaluate": f'saddlellm evaluate "{output_dir}" DATA.jsonl',
            "serve": f'saddlellm serve "{output_dir}" --open',
        },
    }
    _emit_json(result, args.output)
    return 0


def _cmd_dashboard(args) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("The dashboard requires the SaddleLLM serve extra.") from exc
    from .runtime.WebUI import create_dashboard_app

    app = create_dashboard_app(args.output_dir)
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{args.port}"
    if args.open:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"SaddleLLM training dashboard: {url}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_workbench(args) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("The workbench requires the SaddleLLM serve extra.") from exc
    from .runtime.WorkbenchAPI import create_workbench_app

    app = create_workbench_app(args.project, study=args.study)
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{args.port}"
    if args.open:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"SaddleLLM experiment workbench (read-only): {url}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_validate_config(args) -> int:
    from .training.TrainingConfigValidator import TrainingConfigValidator

    config_path = args.config
    if not os.path.exists(config_path):
        _emit_json({"valid": False, "issues": [f"Config file not found: {config_path}"]}, args.output)
        return 1
    try:
        config = _load_training_config(config_path)
    except (TypeError, ValueError) as exc:
        _emit_json(
            {"valid": False, "issues": [str(exc)], "config_path": config_path},
            args.output,
        )
        return 1
    report = TrainingConfigValidator.validate(config, inspect_data=not args.skip_data_inspection).to_dict()
    report["config_path"] = config_path
    _emit_json(report, args.output)
    return 0 if report.get("valid", False) else 1


def _cmd_evaluate_model(args) -> int:
    if args.task != "multiple-choice" and (
        args.case_field or args.variant_field or args.prompt_variants
    ):
        raise ValueError(
            "--case-field, --variant-field, and --prompt-variants are only supported "
            "with --task multiple-choice"
        )
    if args.task != "rubric" and (args.endpoint or args.api_key_env):
        raise ValueError("--endpoint and --api-key-env are only supported with --task rubric")
    if args.task == "rubric" and (
        args.generation_samples or args.generation_all or args.generation_indices
    ):
        raise ValueError(
            "rubric evaluation always generates every record; remove generation selection options"
        )
    from .evaluation.EvaluationRunner import run_evaluation, summarize_evaluation

    common = {
        "adapter": args.adapter,
        "prompt_field": args.prompt_field,
        "id_field": args.id_field,
        "limit": args.limit,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "dtype": args.dtype,
        "enable_thinking": args.enable_thinking,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.task == "rubric":
        common.update(
            rubric_field=args.rubric_field,
            reference_field=args.reference_field,
            pass_threshold=args.pass_threshold,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            endpoint=args.endpoint,
            api_key_env=args.api_key_env,
            request_timeout=args.request_timeout,
        )
    else:
        common.update(
            generation_samples=sys.maxsize if args.generation_all else args.generation_samples,
            generation_indices=tuple(
                int(value.strip())
                for value in args.generation_indices.split(",") if value.strip()
            ),
        )
        if args.task == "completion":
            common.update(
                completion_field=args.completion_field,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        else:
            common.update(
                choices=args.choices,
                question_field=args.question_field,
                answer_field=args.answer_field,
                group_field=args.group_field,
                case_field=args.case_field,
                variant_field=args.variant_field,
                prompt_variants=args.prompt_variants,
                batch_size=args.batch_size,
            )
    result = run_evaluation(
        args.model, args.data,
        task=args.task,
        split=getattr(args, "split", None),
        output=args.output,
        markdown_output=args.markdown_output,
        **common,
    )
    summary = summarize_evaluation(result)
    summary["output"] = os.path.abspath(args.output)
    summary["markdown_output"] = (
        os.path.abspath(args.markdown_output) if args.markdown_output else None
    )
    _emit_json(summary)
    return 0


def _cmd_compare_evaluations(args) -> int:
    from .evaluation.EvaluationComparison import (
        build_evaluation_comparison,
        load_evaluation,
        save_comparison,
    )

    if bool(args.base_test) != bool(args.candidate_test):
        raise ValueError("--base-test and --candidate-test must be provided together")
    manifest = None
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as handle:
            manifest = json.load(handle)
    result = build_evaluation_comparison(
        base_validation=load_evaluation(args.base_validation),
        candidate_validation=load_evaluation(args.candidate_validation),
        base_test=load_evaluation(args.base_test) if args.base_test else None,
        candidate_test=(
            load_evaluation(args.candidate_test) if args.candidate_test else None
        ),
        minimum_gain_percentage_points=args.min_gain_pp,
        maximum_p_exclusive=args.max_p,
        maximum_candidate_critical_failures=args.max_critical_failures,
        manifest=manifest,
        overlap_key=args.overlap_key,
        experiment=args.experiment,
        scope=args.scope,
    )
    save_comparison(result, args.output, args.markdown_output)
    result["output"] = os.path.abspath(args.output)
    result["markdown_output"] = (
        os.path.abspath(args.markdown_output) if args.markdown_output else None
    )
    _emit_json(result)
    return 0


def _cmd_doctor(args) -> int:
    from .utils.EnvironmentDiagnostics import environment_report

    payload = environment_report(
        profile=getattr(args, "profile", "train"),
        device=getattr(args, "device", "auto"),
    )
    _emit_json(payload, args.output)
    return 0 if payload["ready"] else 1


def _cmd_smoke_test(args) -> int:
    from .evaluation.SmokeTestRunner import run_smoke_tests

    result = run_smoke_tests(
        work_dir=args.work_dir,
        run_training=not args.skip_training,
        clean=not args.keep,
        run_doctor=not args.skip_doctor,
        timeout_seconds=args.timeout_seconds,
        use_subprocess=not args.in_process,
    )
    _emit_json(result, args.output)
    return 0 if result.get("ready", False) else 1


def _cmd_inspect_data(args) -> int:
    from .data.TrainingDataInspector import (
        audit_training_splits,
        inspect_training_data,
    )

    result = inspect_training_data(args.data_path, task=args.task, max_records=args.max_records)
    if args.validation_data or args.test_data:
        leakage = audit_training_splits(
            args.data_path,
            validation_path=args.validation_data,
            test_path=args.test_data,
            group_fields=args.group_field,
            content_fields=args.content_field,
            near_duplicate_threshold=args.near_duplicate_threshold,
            strict=args.strict_leakage,
            include_content_previews=args.include_content_previews,
        )
        result["split_leakage"] = leakage
        result.setdefault("errors", []).extend(
            f"split leakage: {issue}" for issue in leakage["issues"]
        )
        result.setdefault("warnings", []).extend(
            f"split leakage: {warning}" for warning in leakage["warnings"]
        )
        if not leakage["passed"]:
            result["ready"] = False
            result["severity"] = "error"
    _emit_json(result, args.output)
    return 0 if result.get("ready", False) else 1


def _cmd_inspect_vla(args) -> int:
    from .multimodal.VLA import VLAActionSpace
    from .multimodal.VLADataInspector import inspect_vla_data

    action_space = VLAActionSpace(
        action_dim=args.action_dim,
        action_type=args.action_type,
        bins=args.action_bins,
        min_value=args.min_value,
        max_value=args.max_value,
        include_gripper=not args.no_gripper,
    )
    result = inspect_vla_data(
        args.data_path,
        image_root=args.image_root,
        action_space=action_space,
        max_records=args.max_records,
    )
    _emit_json(result, args.output)
    return 0 if result.get("ready", False) else 1


def _cmd_train_world_model(args) -> int:
    from .world_models._WorldModelTrainer import train_world_model_from_config

    result = train_world_model_from_config(args.config, dry_run=args.dry_run)
    _emit_json(result, args.output)
    return 0


def _cmd_build_media_cache(args) -> int:
    from .multimodal.MediaCache import MediaCacheBuildConfig, build_media_cache

    payload = _load_config(args.config)
    if not isinstance(payload, dict):
        raise ValueError("Media-cache config root must be a mapping")
    if "cache" in payload:
        payload = payload["cache"]
    base_dir = os.path.dirname(os.path.abspath(args.config)) or "."
    for key in ("input_path", "output_dir"):
        value = payload.get(key)
        if value and not os.path.isabs(value):
            payload[key] = os.path.abspath(os.path.join(base_dir, value))
    if args.overwrite:
        payload["overwrite"] = True
        payload["resume"] = False
    elif args.no_resume:
        payload["resume"] = False
    result = build_media_cache(MediaCacheBuildConfig(**payload))
    _emit_json(result, args.output)
    return 0


def _cmd_media_codecs(args) -> int:
    from dataclasses import asdict

    from .multimodal.BuiltinMediaCodecs import register_builtin_media_codecs
    from .multimodal.ModalityCodec import ModalityCodecRegistry

    register_builtin_media_codecs()
    result = {
        "codecs": [
            asdict(spec) for spec in ModalityCodecRegistry.list_specs(args.modality)
        ]
    }
    _emit_json(result, args.output)
    return 0


def _cmd_world_model_backends(args) -> int:
    from .world_models.WorldModelBackends import list_world_model_backends

    result = {"backends": list_world_model_backends()}
    _emit_json(result, args.output)
    return 0


def _cmd_infer_world_model(args) -> int:
    from .world_models.WorldModelInference import run_world_model_inference

    request = _load_config(args.request)
    result = run_world_model_inference(
        args.checkpoint,
        request,
        device=args.device,
    )
    _emit_json(result, args.output)
    return 0


def _eye_runtime_config(path):
    from .spatial.prosthetic_eye_control import EyePIDConfig, EyePlantConfig, EyeSafetyConfig

    payload = _load_config(path) if path else {}
    if not isinstance(payload, dict):
        raise ValueError("Eye runtime config root must be a mapping")
    if "eye_control" in payload:
        payload = payload["eye_control"]
    if not isinstance(payload, dict):
        raise ValueError("eye_control must be a mapping")
    allowed = {"safety", "pid", "plant", "planner"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError("Unknown eye runtime config fields: " + ", ".join(unknown))
    planner = payload.get("planner")
    if planner is not None and not isinstance(planner, dict):
        raise ValueError("eye_control.planner must be a mapping")
    return {
        "safety_config": EyeSafetyConfig.from_dict(payload.get("safety")),
        "pid_config": EyePIDConfig.from_dict(payload.get("pid")),
        "plant_config": EyePlantConfig.from_dict(payload.get("plant")),
        "planner_config": dict(planner or {}),
    }


def _cmd_build_eye_control_data(args) -> int:
    from .spatial.prosthetic_eye_control import generate_prosthetic_eye_dataset

    runtime = _eye_runtime_config(args.config)
    runtime.pop("planner_config", None)
    result = generate_prosthetic_eye_dataset(
        args.output,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        moving_target_ratio=args.moving_target_ratio,
        target_dropout_probability=args.target_dropout_probability,
        **runtime,
    )
    _emit_json(result, args.result_output)
    return 0


def _cmd_simulate_eye_control(args) -> int:
    from .spatial.prosthetic_eye_control import simulate_prosthetic_eye_control

    result = simulate_prosthetic_eye_control(
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        include_traces=args.include_traces,
        **_eye_runtime_config(args.config),
    )
    _emit_json(result, args.output)
    return 0


def _cmd_build_spatial_world_data(args) -> int:
    from .spatial.SpatialPerception import MapExtractionConfig
    from .spatial.SpatialPlanner import SpatialPlannerConfig
    from .spatial.SpatialWorldModelData import (
        SpatialSequenceConfig,
        SpatialTrajectoryConfig,
        generate_spatial_sequence_dataset,
        generate_spatial_world_model_dataset,
    )

    map_config = MapExtractionConfig(
        free_threshold=args.free_threshold,
        free_is_bright=not args.free_is_dark,
        uncertainty_band=args.uncertainty_band,
        obstacle_dilation=args.obstacle_dilation,
        max_dimension=args.max_dimension,
        resolution=args.resolution,
    )
    planner_config = SpatialPlannerConfig(
        diagonal=not args.no_diagonal,
        allow_unknown=args.allow_unknown,
        route_count=args.routes_per_pair,
        clearance_weight=0.1,
        diversity_weight=2.0,
    )
    if args.schema == "v2":
        result = generate_spatial_sequence_dataset(
            [args.image, *args.additional_image],
            args.output,
            episodes=args.episodes,
            routes_per_pair=args.routes_per_pair,
            seed=args.seed,
            minimum_distance=args.minimum_distance,
            map_config=map_config,
            planner_config=planner_config,
            sequence_config=SpatialSequenceConfig(
                crop_size=args.crop_size,
                sensor_radius=args.sensor_radius,
                action_dim=2,
                max_steps=args.max_steps,
                collision_probability=args.collision_probability,
            ),
        )
    else:
        result = generate_spatial_world_model_dataset(
            args.image,
            args.output,
            episodes=args.episodes,
            routes_per_pair=args.routes_per_pair,
            seed=args.seed,
            minimum_distance=args.minimum_distance,
            map_config=map_config,
            planner_config=planner_config,
            trajectory_config=SpatialTrajectoryConfig(
                observation_size=args.observation_size,
                action_dim=2,
                max_steps=args.max_steps,
            ),
        )
    _emit_json(result)
    return 0


def _cmd_evaluate_spatial_world_model(args) -> int:
    from .spatial.SpatialWorldModelEvaluation import evaluate_spatial_world_model

    result = evaluate_spatial_world_model(
        args.checkpoint,
        args.data,
        device=args.device,
        sequence_length=args.sequence_length,
        max_windows=args.max_windows,
        group_key=args.group_key,
        group_value=args.group_value,
    )
    _emit_json(result, args.output)
    return 0


def _cmd_plan_spatial_route(args) -> int:
    from .spatial.SpatialPerception import (
        MapExtractionConfig,
        QwenVLSpatialAnalyzer,
        TopDownMapExtractor,
    )
    from .spatial.SpatialPlanner import GridPathPlanner, SpatialPlannerConfig
    from .spatial.SpatialVisualization import (
        render_spatial_plan_html,
        render_spatial_plan_png,
        save_spatial_plan_json,
    )
    from .spatial.SpatialWorldModel import SpatialWorldModelCoordinator

    analyzer = (
        QwenVLSpatialAnalyzer.from_pretrained(
            args.qwen_model,
            device_map=args.qwen_device_map,
        )
        if args.qwen_model
        else None
    )
    extractor = TopDownMapExtractor(
        MapExtractionConfig(
            free_threshold=args.free_threshold,
            free_is_bright=not args.free_is_dark,
            uncertainty_band=args.uncertainty_band,
            obstacle_dilation=args.obstacle_dilation,
            max_dimension=args.max_dimension,
            resolution=args.resolution,
        )
    )
    planner = GridPathPlanner(
        SpatialPlannerConfig(
            diagonal=not args.no_diagonal,
            allow_unknown=args.allow_unknown,
            clearance_weight=args.clearance_weight,
            diversity_weight=args.diversity_weight,
            route_count=args.routes,
        )
    )
    coordinator = SpatialWorldModelCoordinator(
        extractor=extractor,
        planner=planner,
        analyzer=analyzer,
        allow_perspective=args.allow_perspective,
    )
    result = coordinator.plan_image(
        args.image,
        start=_parse_spatial_point(args.start),
        goal=_parse_spatial_point(args.goal),
        instruction=args.instruction,
        route_count=args.routes,
    )
    stem = os.path.splitext(os.path.basename(args.image))[0]
    output_json = args.output_json or os.path.join("outputs", "spatial", f"{stem}_routes.json")
    output_html = args.output_html or os.path.join("outputs", "spatial", f"{stem}_routes.html")
    output_png = args.output_png or os.path.join("outputs", "spatial", f"{stem}_routes.png")
    json_path = save_spatial_plan_json(result, output_json)
    html_path = render_spatial_plan_html(result, output_html)
    png_path = render_spatial_plan_png(result, output_png)
    _emit_json(
        {
            "status": "completed",
            "image": os.path.abspath(args.image),
            "routes": len(result.routes),
            "start": list(result.start),
            "goal": list(result.goal),
            "json": json_path,
            "visualization": html_path,
            "preview": png_path,
            "semantic_model": args.qwen_model,
        }
    )
    return 0


def _cmd_spatial_studio(args) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError("uvicorn is required to run the spatial studio") from error
    from .spatial.SpatialAPI import (
        SpatialStudioSettings,
        create_spatial_studio_app,
    )

    settings = SpatialStudioSettings(
        workspace=args.workspace,
        frontend_dist=args.frontend_dist,
        demo_image=args.demo_image,
        qwen_model=args.qwen_model,
        world_model_checkpoint=args.world_model,
        device=args.device,
    )
    app = create_spatial_studio_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _world_agent_settings(args):
    from .spatial.WorldAgent import WorldAgentSettings

    settings = (
        WorldAgentSettings.from_file(args.config)
        if getattr(args, "config", None)
        else WorldAgentSettings()
    )
    for argument, attribute in (
        ("workspace", "workspace"),
        ("qwen_model", "qwen_model"),
        ("world_model", "world_model_checkpoint"),
        ("device", "device"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            setattr(settings, attribute, value)
    return settings


def _cmd_world_agent_run(args) -> int:
    from .spatial.WorldAgent import WorldAgentRuntime

    settings = _world_agent_settings(args)
    runtime = WorldAgentRuntime(settings)
    result = runtime.run_image(
        args.image,
        start=_parse_spatial_point(args.start),
        goal=_parse_spatial_point(args.goal),
        instruction=args.instruction,
        semantic_backend=args.semantic_backend,
        route_count=args.routes,
        use_world_model=args.use_world_model,
        simulation_mode=args.simulation,
    )
    _emit_json(result, args.output)
    return 0


def _cmd_world_agent_api(args) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise ImportError("uvicorn is required to run the WorldAgent API") from error
    from .spatial.WorldAgentAPI import create_world_agent_app

    settings = _world_agent_settings(args)
    app = create_world_agent_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_preflight(args) -> int:
    from .factory.TrainingFactory import FactoryConfig, LLMTrainingFactory

    stage = args.stage
    preference_method = args.preference_method
    if stage in {"dpo", "orpo", "kto"}:
        preference_method = stage
        stage = "preference"
    factory = LLMTrainingFactory(FactoryConfig(root_dir=args.root_dir, base_model=args.base_model))
    overrides = {}
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    result = factory.create_preflight_plan(
        data_path=args.data_path,
        stage=stage,
        method=args.method,
        preference_method=preference_method,
        save=not args.no_save,
        **overrides,
    )
    _emit_json({
        "ready": result.get("ready", False),
        "stage": result.get("stage"),
        "method": result.get("method"),
        "plan_path": result.get("plan_path"),
        "config_path": result.get("config_path"),
        "blocking_errors": result.get("blocking_errors", []),
        "recommendations": result.get("recommendations", []),
        "training_estimate": result.get("training_estimate", {}),
    })
    return 0 if result.get("ready", False) else 1


def _cmd_report(args) -> int:
    from .factory.TrainingFactory import FactoryConfig, LLMTrainingFactory

    root = args.root_dir or args.output_dir
    result = LLMTrainingFactory(FactoryConfig(root_dir=root)).report(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_export_model(args) -> int:
    from .evaluation.ReleaseGate import release_gate_accepted
    from .runtime.ModelExporter import ModelExportRequest, ModelExporter

    gate = None
    if args.gate_result:
        gate = _load_config(args.gate_result)
    if args.require_gate and not release_gate_accepted(gate):
        _emit_json(
            {
                "status": "blocked",
                "error": (
                    "--require-gate needs a --gate-result with an explicit "
                    "accepted/promoted decision."
                ),
            },
            args.result_output,
        )
        return 1
    try:
        result = ModelExporter.export(
            ModelExportRequest(
                model_path=args.model_path,
                output_dir=args.output_dir,
                format=args.format,
                merge_lora=not args.no_merge_lora,
                safe_serialization=not args.no_safe_serialization,
                trust_remote_code=args.trust_remote_code,
                local_files_only=args.local_files_only,
                device=args.device,
                dtype=args.dtype,
                hash_weights=args.hash_weights,
                require_gate=args.require_gate,
                overwrite=args.overwrite,
            ),
            gate=gate,
        ).to_dict()
    except Exception as exc:
        _emit_json(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            args.result_output,
        )
        return 1
    _emit_json(result, args.result_output)
    return 0


def _cmd_serve_model(args) -> int:
    from .runtime.InferenceServer import create_inference_app
    from .runtime.ServingPlan import check_serving_address
    from .runtime.ServingPerformance import build_serving_settings, performance_settings_summary

    vllm_options = {
        name: getattr(args, name) for name in (
            "gpu_memory_utilization", "tensor_parallel_size", "max_model_len",
            "max_num_seqs", "max_num_batched_tokens",
        ) if getattr(args, name) is not None
    }
    if args.backend not in {"auto", "vllm"} and (vllm_options or args.vllm_prefix_caching is not None or args.enforce_eager):
        raise ValueError("vLLM engine options require --backend auto or vllm.")
    if args.vllm_prefix_caching is not None:
        vllm_options["enable_prefix_caching"] = args.vllm_prefix_caching
    if args.enforce_eager:
        vllm_options["enforce_eager"] = True

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {args.api_key_env!r} is empty or missing.")
    if args.max_concurrency is not None and args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1.")
    if args.max_input_tokens < 1 or args.default_max_new_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("Token limits must be positive integers.")
    if args.default_max_new_tokens > args.max_new_tokens:
        raise ValueError("--default-max-new-tokens cannot exceed --max-new-tokens.")
    settings_options = dict(
        model_name=args.model_name,
        require_high_throughput=args.require_high_throughput,
        truncate_input=args.truncate_input,
        native_attention_backend=args.native_attention_backend,
        native_stream_batching=args.native_stream_batching,
        native_decode_mode=args.native_decode_mode,
        batch_wait_ms=args.batch_wait_ms,
        cpu_prepack_linear=args.cpu_prepack,
        native_fused_kernels=args.native_fused_kernels,
        native_prefill_chunk_size=args.native_prefill_chunk_size,
        native_max_step_tokens=args.native_max_step_tokens,
        vllm=vllm_options,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_concurrency=args.max_concurrency,
        max_pending_requests=args.max_pending_requests,
        batch_size=args.batch_size,
        batch_token_budget=args.batch_token_budget,
        batch_length_bucket_size=args.batch_length_bucket_size,
        max_pending_tokens=args.max_queued_tokens,
        prefix_cache_mb=args.prefix_cache_mb,
        prefix_cache_partial=args.prefix_cache_partial,
        max_input_tokens=args.max_input_tokens,
        default_max_new_tokens=args.default_max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        api_key=api_key,
        merge_adapter=not args.no_merge_adapter,
        fuse_projections=not args.no_fuse_projections,
        enable_thinking=args.enable_thinking,
        web_ui=not args.no_web_ui,
        chat_history=not args.no_chat_history,
        ui_title=args.title,
    )
    for name in ("native_stream_batching", "max_concurrency", "batch_size",
                 "batch_token_budget", "prefix_cache_mb", "prefix_cache_partial"):
        if settings_options[name] is None:
            del settings_options[name]
    settings = build_serving_settings(args.model_path, backend=args.backend,
                                      performance=args.performance, **settings_options)
    selection = settings.backend_selection
    if args.dry_run:
        _emit_json({"status": "planned_no_model_loaded", "model": args.model_path,
                    "backend_selection": selection, "dtype": settings.dtype,
                    "quantization": settings.quantization,
                    "thinking_policy": "model_default" if settings.enable_thinking is None
                    else "enabled" if settings.enable_thinking else "disabled",
                    "input_policy": "truncate" if settings.truncate_input else "reject_overlong",
                    "native_stream_batching": settings.native_stream_batching,
                    "native_decode_mode": settings.native_decode_mode,
                    "native_fused_kernels": settings.native_fused_kernels,
                    "native_prefill_chunk_size": settings.native_prefill_chunk_size,
                    "native_max_step_tokens": settings.native_max_step_tokens,
                    "batch_wait_ms": settings.batch_wait_ms,
                    "performance": performance_settings_summary(settings),
                    "host": args.host, "port": args.port,
                    "openai_base_path": "/v1"})
        return 0
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required; install SaddleLLM serving dependencies.") from exc
    check_serving_address(args.host, args.port)
    print(f"SaddleLLM backend: {selection['requested']} -> {selection['selected']}; {selection['reason']}", flush=True)
    for warning in selection["warnings"]:
        print(f"Serving warning: {warning}", file=sys.stderr, flush=True)
    app = create_inference_app(settings)
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    url = f"http://{browser_host}:{args.port}"
    if args.open and not args.no_web_ui:
        import threading
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Starting SaddleLLM chat/API: {url}; OpenAI base_url: {url}/v1; readiness: {url}/readyz", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_evaluate_decision(args) -> int:
    from pathlib import Path

    from .decision.Evaluation import evaluate_typed

    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    report = evaluate_typed(
        args.data, args.model, output_dir=args.output_dir, device=args.device,
        batch_size=args.batch_size, cpu_threads=args.cpu_threads, run_label=args.run_label,
        splits=splits, logits_reference=args.logits_reference, progress=False,
    )
    reload_passed = not args.logits_reference or report["reload_consistency"]["passed"]
    _emit_json({"ok": reload_passed, "command": "evaluate-decision",
                "report_path": str((Path(args.output_dir) / "report.json").resolve()),
                "splits": report["splits"], "reload_consistency": report.get("reload_consistency")})
    return 0 if reload_passed else 1


def _cmd_serve_decision(args) -> int:
    from pathlib import Path
    from .decision.Server import DecisionServerSettings

    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError("API key environment variable is unset or empty")
    settings = DecisionServerSettings(model_path=args.model, device=args.device,
                                      host=args.host, api_key=api_key,
                                      encoder_dtype=args.dtype, backbone_name_or_path=args.backbone,
                                      web_ui=not args.no_web_ui)
    if args.dry_run:
        supported_types = ["Choice"]
        config_path = Path(args.model) / "decision_config.json"
        if config_path.is_file():
            import json
            if json.loads(config_path.read_text(encoding="utf-8")).get("model_type") == "saddle_typed_decision":
                supported_types = ["Choice", "Score", "Noul"]
        _emit_json({"ok": True, "command": "serve-decision", "status": "planned",
                    "model_path": args.model, "backbone": args.backbone,
                    "device": args.device, "dtype": args.dtype,
                    "address": f"http://{args.host}:{args.port}",
                    "supported_types": supported_types, "requires_auth": api_key is not None,
                    "endpoint": "/v1/decisions"})
        return 0
    import uvicorn

    from .decision.Server import create_decision_app

    uvicorn.run(create_decision_app(settings), host=args.host, port=args.port)
    return 0


def _cmd_serve_embedding(args) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required to serve embeddings.") from exc
    from .runtime.EmbeddingServer import (
        EmbeddingServerSettings,
        api_key_from_env,
        create_embedding_app,
    )

    settings = EmbeddingServerSettings(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        max_inputs=args.max_inputs,
        max_input_chars=args.max_input_chars,
        max_total_chars=args.max_total_chars,
        max_input_tokens=args.max_input_tokens,
        expected_dimensions=args.dimensions,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        api_key=api_key_from_env(args.api_key_env),
    )
    app = create_embedding_app(settings)
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(
        f"SaddleLLM embedding API: http://{browser_host}:{args.port}/v1/embeddings",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_serve_reranker(args) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required to serve a reranker.") from exc
    from .runtime.RerankerServer import (
        RerankerServerSettings,
        api_key_from_env,
        create_reranker_app,
    )

    settings = RerankerServerSettings(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
        max_documents=args.max_documents,
        max_document_chars=args.max_document_chars,
        max_total_chars=args.max_total_chars,
        max_length=args.max_length,
        instruction=args.instruction,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        api_key=api_key_from_env(args.api_key_env),
    )
    app = create_reranker_app(settings)
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(
        f"SaddleLLM reranker API: http://{browser_host}:{args.port}/rerank",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _emit_json(payload: dict, output: Optional[str] = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    print(text)


def _parse_spatial_point(value: str):
    text = str(value).strip()
    if "," not in text:
        return text
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError("Spatial points must use x,y")
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as error:
        raise ValueError("Spatial point coordinates must be numeric") from error


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def _load_training_config(path: str) -> dict:
    config = _load_config(path)
    if not isinstance(config, dict):
        raise TypeError("Training config must be a YAML/JSON object.")

    old_fields = [name for name in ("flow", "stage", "apiVersion", "kind", "spec") if name in config]
    if old_fields:
        raise ValueError(
            "Unsupported old config fields: "
            + ", ".join(old_fields)
            + ". Use one plain config with a top-level `stages` list."
        )

    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Training config requires a non-empty top-level `stages` list.")
    if any(not isinstance(stage, str) or not stage.strip() for stage in stages):
        raise ValueError("Every item in `stages` must be a non-empty string.")
    if len(stages) != len(set(stages)):
        raise ValueError("`stages` must not contain duplicates.")
    return config


if __name__ == "__main__":
    sys.exit(main())
