"""Experiment runner for low-cost scratch pretraining grids.

It ties together scaling-law run plans, data mix plans, eval suites,
distributed templates, training configs, and pretrain reports.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class PretrainExperimentConfig:
    domain: str = "research"
    output_dir: str = "./pretrain_experiments"
    model_names: List[str] = field(default_factory=lambda: ["qwen-tiny-160m", "qwen-300m"])
    token_multipliers: List[float] = field(default_factory=lambda: [0.05, 0.1])
    global_batch_size: int = 64
    seq_length: int = 1024
    max_tokens: Optional[int] = None
    max_params: Optional[int] = None
    tokenizer_path: Optional[str] = None
    corpus_sources: List[Dict] = field(default_factory=list)
    sources_by_bucket: Optional[Dict[str, Sequence[Dict]]] = None
    learning_rate: float = 3e-4
    per_device_batch_size: int = 1
    eval_max_samples: int = 200
    stability_monitor: bool = True
    gpu_memory_gb: Optional[float] = None
    num_gpus: int = 1

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.sources_by_bucket is not None:
            data["sources_by_bucket"] = {
                key: list(value) for key, value in self.sources_by_bucket.items()
            }
        return data


@dataclass
class ExperimentRunManifest:
    run_id: str
    model_name: str
    output_dir: str
    config_path: str
    report_path: str
    target_tokens: int
    max_steps: int
    approx_flops: float
    status: str = "planned"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExperimentBundle:
    output_dir: str
    config: Dict
    data_mix_path: Optional[str]
    eval_suite_path: str
    runs: List[ExperimentRunManifest]
    manifest_path: str

    def to_dict(self) -> Dict:
        return {
            "output_dir": self.output_dir,
            "config": self.config,
            "data_mix_path": self.data_mix_path,
            "eval_suite_path": self.eval_suite_path,
            "runs": [run.to_dict() for run in self.runs],
            "manifest_path": self.manifest_path,
        }


class PretrainExperimentRunner:
    """Prepare and optionally run a scratch-pretraining experiment grid."""

    def __init__(self, config: Optional[PretrainExperimentConfig] = None):
        self.config = config or PretrainExperimentConfig()

    def build_bundle(self) -> ExperimentBundle:
        from ..data.DataStrategy import DataMixPlanner
        from ..training.DistributedConfig import DistributedConfig
        from ..evaluation.PretrainEvalSuite import PretrainEvalSuite
        from .ScalingLawAnalyzer import ScalingLawAnalyzer

        os.makedirs(self.config.output_dir, exist_ok=True)
        self._save_json(os.path.join(self.config.output_dir, "experiment_config.json"), self.config.to_dict())

        data_mix_path = None
        data_sources = list(self.config.corpus_sources)
        source_weights = None
        mix_plan = DataMixPlanner.for_domain(self.config.domain)
        if self.config.sources_by_bucket:
            data_cfg = mix_plan.to_pipeline_config(
                sources_by_bucket=self.config.sources_by_bucket,
                output_dir=os.path.join(self.config.output_dir, "processed_data"),
                max_seq_length=self.config.seq_length,
            )
            data_sources = data_cfg["sources"]
            source_weights = data_cfg["source_weights"]
            data_mix_path = self._save_json(
                os.path.join(self.config.output_dir, "data_mix_plan.json"),
                data_cfg["mix_plan"],
            )
        else:
            data_mix_path = self._save_json(
                os.path.join(self.config.output_dir, "data_mix_plan.json"),
                mix_plan.to_dict(),
            )

        suite = PretrainEvalSuite.for_domain(self.config.domain, max_samples=self.config.eval_max_samples)
        eval_suite_path = suite.save(os.path.join(self.config.output_dir, "eval_suite.json"))

        plans = ScalingLawAnalyzer().plan_pretrain_grid(
            model_names=self.config.model_names,
            token_multipliers=self.config.token_multipliers,
            global_batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            max_tokens=self.config.max_tokens,
            max_params=self.config.max_params,
        )

        run_manifests: List[ExperimentRunManifest] = []
        for idx, plan in enumerate(plans):
            run_id = f"{idx:03d}-{plan.model_name}-{self._format_multiplier(plan.target_tokens, plan.active_params)}"
            run_dir = os.path.join(self.config.output_dir, "runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            cfg = plan.to_config(
                corpus_sources=data_sources,
                output_dir=run_dir,
                tokenizer_path=self.config.tokenizer_path,
                learning_rate=self.config.learning_rate,
                per_device_batch_size=self.config.per_device_batch_size,
                stability_monitor=self.config.stability_monitor,
            )
            cfg["data"]["source_weights"] = source_weights
            cfg["eval_suite"] = suite.to_dict()
            if not data_sources:
                cfg["stages"] = ["eval"]
                cfg.setdefault("notes", []).append("No corpus sources supplied; config is dry-run/planning only.")
            if self.config.gpu_memory_gb:
                templates = DistributedConfig.save_template_bundle(
                    os.path.join(run_dir, "distributed"),
                    params=plan.params,
                    gpu_memory_gb=self.config.gpu_memory_gb,
                    num_gpus=self.config.num_gpus,
                )
                cfg["distributed"]["strategy"] = templates.get("strategy", cfg["distributed"].get("strategy", "single"))
                if "deepspeed" in templates:
                    cfg["distributed"]["deepspeed_config_path"] = templates["deepspeed"]
                cfg["distributed_templates"] = templates
            config_path = self._save_json(os.path.join(run_dir, "config.json"), cfg)
            self._save_json(os.path.join(run_dir, "run_plan.json"), plan.to_dict())
            run_manifests.append(ExperimentRunManifest(
                run_id=run_id,
                model_name=plan.model_name,
                output_dir=run_dir,
                config_path=config_path,
                report_path=os.path.join(run_dir, "pretrain_report.json"),
                target_tokens=plan.target_tokens,
                max_steps=plan.max_steps,
                approx_flops=plan.approx_flops,
            ))

        manifest_path = self._save_json(
            os.path.join(self.config.output_dir, "manifest.json"),
            {
                "config": self.config.to_dict(),
                "data_mix_path": data_mix_path,
                "eval_suite_path": eval_suite_path,
                "runs": [run.to_dict() for run in run_manifests],
            },
        )
        return ExperimentBundle(
            output_dir=self.config.output_dir,
            config=self.config.to_dict(),
            data_mix_path=data_mix_path,
            eval_suite_path=eval_suite_path,
            runs=run_manifests,
            manifest_path=manifest_path,
        )

    def run(self, dry_run: bool = True, limit: Optional[int] = None) -> ExperimentBundle:
        bundle = self.build_bundle()
        if dry_run:
            return bundle

        from .PretrainReport import PretrainReport
        from ..training.TrainingOrchestrator import TrainingOrchestrator

        for run in bundle.runs[: limit or len(bundle.runs)]:
            with open(run.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not cfg.get("data", {}).get("sources") and "pretrain" in cfg.get("stages", []):
                run.status = "skipped_no_data"
                continue
            orchestrator = TrainingOrchestrator.from_dict(cfg)
            orchestrator.run()
            PretrainReport.generate(run.output_dir)
            run.status = "completed"
        self._save_json(bundle.manifest_path, bundle.to_dict())
        return bundle

    def compare_reports(self, bundle: Optional[ExperimentBundle] = None) -> Dict:
        bundle = bundle or self.build_bundle()
        reports = []
        for run in bundle.runs:
            if os.path.exists(run.report_path):
                reports.append(run.report_path)
        rows = []
        for path in reports:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
            rows.append({
                "path": path,
                "grade": report.get("grade"),
                "score": report.get("score"),
                "findings": len(report.get("findings", [])),
            })
        comparison = {"reports": rows}
        self._save_json(os.path.join(self.config.output_dir, "report_comparison.json"), comparison)
        return comparison

    def _save_json(self, path: str, data: Dict) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _format_multiplier(self, target_tokens: int, active_params: int) -> str:
        mult = target_tokens / max(1, active_params)
        return f"{mult:.2f}x".replace(".", "p")
