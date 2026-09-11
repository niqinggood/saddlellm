"""Command line interface for the SaddleLLM training factory."""
import argparse
import json
import os
import sys
from typing import Optional


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="saddle-llm",
        description="SaddleLLM training factory CLI",
        epilog="New here? Run: saddle-llm quickstart",
    )
    sub = parser.add_subparsers(dest="command")

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

    train_p = sub.add_parser("train", help="Run a training config")
    train_p.add_argument("config")
    train_p.add_argument("--dry-run", action="store_true")
    train_p.add_argument("--debug", action="store_true", help="Show full traceback instead of a structured error")

    validate_p = sub.add_parser("validate-config", help="Validate a training config")
    validate_p.add_argument("config")
    validate_p.add_argument("--output", default=None)
    validate_p.add_argument("--skip-data-inspection", action="store_true")

    doctor_p = sub.add_parser("doctor", help="Check local training environment and dependencies")
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

    export_p = sub.add_parser("export-model", help="Package a checkpoint as a verified HF release")
    export_p.add_argument("model_path", help="Local checkpoint/adapter path or Hugging Face model id")
    export_p.add_argument("output_dir", help="New release directory")
    export_p.add_argument("--format", default="hf", choices=["hf"])
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

    serve_p = sub.add_parser("serve-model", help="Run an OpenAI-compatible local inference API")
    serve_p.add_argument("model_path", help="Exported release, full checkpoint, adapter, or HF model id")
    serve_p.add_argument("--model-name", default=None)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    serve_p.add_argument(
        "--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    serve_p.add_argument("--trust-remote-code", action="store_true")
    serve_p.add_argument("--local-files-only", action="store_true")
    serve_p.add_argument("--no-merge-adapter", action="store_true")
    serve_p.add_argument("--max-concurrency", type=int, default=1)
    serve_p.add_argument("--max-input-tokens", type=int, default=4096)
    serve_p.add_argument("--default-max-new-tokens", type=int, default=256)
    serve_p.add_argument("--max-new-tokens", type=int, default=1024)
    serve_p.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the bearer token; never pass the secret itself",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        print(_getting_started_text())
        return 0

    handlers = {
        "quickstart": _cmd_quickstart,
        "import-data": _cmd_import_data,
        "plan": _cmd_plan,
        "train": _cmd_train,
        "validate-config": _cmd_validate_config,
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
        "export-model": _cmd_export_model,
        "serve-model": _cmd_serve_model,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
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

  saddle-llm quickstart
  cd saddlellm-quickstart
  saddle-llm validate-config config.yaml
  saddle-llm train config.yaml --dry-run

查看全部命令：saddle-llm --help
已有配置：    saddle-llm validate-config <配置文件>"""


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

    config = _load_training_config(args.config)
    backend_plan_path = config.get("distributed", {}).get("backend_plan_path")
    plan = BackendAdapterRegistry.create_launch_plan(
        args.config,
        backend=config.get("distributed", {}).get("strategy", "single"),
        num_gpus=config.get("distributed", {}).get("num_gpus", 1),
        num_nodes=config.get("distributed", {}).get("num_nodes", 1),
        backend_plan_path=backend_plan_path,
    )
    output_dir = os.path.dirname(args.config) or "."
    launch_plan_path = args.launch_plan or os.path.join(output_dir, "launch_plan.json")
    launch_script_path = args.launch_script or os.path.join(output_dir, "launch.ps1")
    BackendAdapterRegistry.save_launch_plan(plan, launch_plan_path)
    BackendAdapterRegistry.save_launch_script(plan, launch_script_path)
    print(json.dumps({
        "config": args.config,
        "launch_plan": launch_plan_path,
        "launch_script": launch_script_path,
        "backend": plan.backend,
        "command": plan.command,
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_train(args) -> int:
    from .training.TrainingOrchestrator import TrainingOrchestrator

    config_path = args.config
    try:
        config = _load_training_config(config_path)
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
            print(json.dumps({
                "config": config_path,
                "stages": list(pipeline_plan.stages),
                "pipeline_plan": pipeline_plan.to_dict(),
                "distributed": config.get("distributed", {}),
                "data_sources": len(config.get("data", {}).get("sources", [])),
            }, ensure_ascii=False, indent=2))
            return 0
        orchestrator = TrainingOrchestrator.from_dict(config)
        results = orchestrator.run()
        _emit_json({
            "ok": True,
            "command": "train",
            "config": config_path,
            "results": results,
        })
        return 0
    except Exception as exc:
        if args.debug:
            raise
        _emit_json({
            "ok": False,
            "command": "train",
            "config": config_path,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": "Run the same command with --debug to show the full traceback.",
        })
        return 1


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


def _cmd_doctor(args) -> int:
    import shutil

    issues = []
    recommendations = []
    versions = {}
    for pkg in ["torch", "transformers", "datasets", "accelerate", "peft", "trl"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "missing"
            issues.append(f"Missing dependency: {pkg}")

    torch = sys.modules.get("torch")
    cuda_available = bool(torch and torch.cuda.is_available())
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None) or "N/A" if torch else "N/A"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_count else ""
    gpu_total_memory_gb = 0.0
    gpu_free_memory_gb = 0.0
    if gpu_count:
        props = torch.cuda.get_device_properties(0)
        gpu_total_memory_gb = props.total_memory / (1024**3)
        gpu_free_memory_gb = gpu_total_memory_gb - torch.cuda.memory_allocated(0) / (1024**3)
    else:
        issues.append("No GPU detected; only very small local training runs are practical.")

    try:
        import psutil

        ram = psutil.virtual_memory()
        ram_total_gb = ram.total / (1024**3)
        ram_available_gb = ram.available / (1024**3)
    except Exception:
        ram_total_gb = 0.0
        ram_available_gb = 0.0
        recommendations.append("Install psutil for RAM diagnostics.")

    disk = shutil.disk_usage(os.getcwd())
    disk_free_gb = disk.free / (1024**3)

    if gpu_count and gpu_free_memory_gb < 4:
        issues.append(f"GPU free memory is low: {gpu_free_memory_gb:.1f}GB.")
    if ram_available_gb and ram_available_gb < 8:
        issues.append(f"Available RAM is low: {ram_available_gb:.1f}GB.")
    if disk_free_gb < 10:
        issues.append(f"Disk free space is low: {disk_free_gb:.1f}GB.")
    elif disk_free_gb < 50:
        recommendations.append(f"Disk free space is {disk_free_gb:.0f}GB; enough for small runs, tight for long runs.")

    if gpu_count >= 8:
        recommendations.append("8+ GPUs: DeepSpeed ZeRO-3 is suitable for 7B+ training.")
    elif gpu_count >= 4:
        recommendations.append("4+ GPUs: DeepSpeed ZeRO-2 is suitable for 3B-class training.")
    elif gpu_count >= 1:
        approx_params_b = gpu_total_memory_gb * 0.7 / 2
        recommendations.append(f"Single GPU: roughly {approx_params_b:.1f}B bf16 parameters are practical before optimizer/activation overhead.")

    from .training.PostTrainingCompatibility import (
        post_training_runtime_report,
        stabilize_peft_optional_backends,
    )

    post_training = post_training_runtime_report()
    optional_backend_warnings = stabilize_peft_optional_backends()
    post_training["optional_backend_warnings"] = optional_backend_warnings
    recommendations.extend(optional_backend_warnings)
    if not post_training["compatible"]:
        issues.extend(f"Post-training dependency: {issue}" for issue in post_training["issues"])
        recommendations.append(
            "Use an isolated environment and run `python -m pip install -r requirements-posttrain.txt`."
        )

    payload = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_free_memory_gb,
        "gpu_total_memory_gb": gpu_total_memory_gb,
        "gpu_total_memoryory_gb": gpu_total_memory_gb,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "disk_free_gb": disk_free_gb,
        "pytorch_version": versions.get("torch", "N/A"),
        "transformers_version": versions.get("transformers", "N/A"),
        "packages": versions,
        "packages_ok": all(value != "missing" for value in versions.values()),
        "post_training": post_training,
        "issues": issues,
        "recommendations": recommendations,
        "ready": not issues,
    }
    _emit_json(payload, args.output)
    return 0 if payload.get("ready", False) else 1


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
    from .data.TrainingDataInspector import inspect_training_data

    result = inspect_training_data(args.data_path, task=args.task, max_records=args.max_records)
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
    from .runtime.ModelExporter import ModelExportRequest, ModelExporter

    gate = None
    if args.gate_result:
        gate = _load_config(args.gate_result)
    if args.require_gate and not (isinstance(gate, dict) and gate.get("accepted")):
        _emit_json(
            {
                "status": "blocked",
                "error": "--require-gate needs a --gate-result whose accepted field is true.",
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
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("uvicorn is required to serve a model.") from exc
    from .runtime.InferenceServer import InferenceServerSettings, create_inference_app

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {args.api_key_env!r} is empty or missing.")
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1.")
    if args.max_input_tokens < 1 or args.default_max_new_tokens < 1 or args.max_new_tokens < 1:
        raise ValueError("Token limits must be positive integers.")
    if args.default_max_new_tokens > args.max_new_tokens:
        raise ValueError("--default-max-new-tokens cannot exceed --max-new-tokens.")
    settings = InferenceServerSettings(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        max_concurrency=args.max_concurrency,
        max_input_tokens=args.max_input_tokens,
        default_max_new_tokens=args.default_max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        api_key=api_key,
        merge_adapter=not args.no_merge_adapter,
    )
    app = create_inference_app(settings)
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
