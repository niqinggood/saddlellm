"""High-level builder for domain LLM training workflows.

The builder turns a domain goal into a concrete SaddleLLM training recipe:
continued pretraining, SFT, preference training, evaluation, and export-ready
paths.  It does not hide the underlying config; callers can inspect or edit the
generated dict before running it.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Union


@dataclass
class DomainRecipe:
    name: str
    recommended_stages: List[str]
    eval_tasks: List[str] = field(default_factory=lambda: ["perplexity"])
    risk_controls: List[str] = field(default_factory=list)
    data_buckets: List[str] = field(default_factory=list)
    notes: str = ""


DOMAIN_RECIPES: Dict[str, DomainRecipe] = {
    "medical": DomainRecipe(
        name="medical",
        recommended_stages=["pretrain", "sft", "preference", "eval"],
        risk_controls=["pii_filter", "clinical_safety_refusal", "evidence_required"],
        data_buckets=["guidelines", "textbooks", "papers", "deidentified_cases", "qa"],
        notes="Use RAG and refusal policies for diagnosis, prescription, and emergency advice.",
    ),
    "biology": DomainRecipe(
        name="biology",
        recommended_stages=["pretrain", "sft", "preference", "eval"],
        risk_controls=["biosecurity_filter", "protocol_safety_review"],
        data_buckets=["papers", "protocols", "gene_protein_notes", "qa"],
        notes="Separate benign education from operational wet-lab instructions when needed.",
    ),
    "research": DomainRecipe(
        name="research",
        recommended_stages=["pretrain", "sft", "preference", "eval"],
        risk_controls=["citation_check", "hallucination_eval"],
        data_buckets=["papers", "reports", "code", "datasets", "reviews"],
        notes="Evaluation should include citation faithfulness and method-summary quality.",
    ),
    "risk": DomainRecipe(
        name="risk",
        recommended_stages=["pretrain", "sft", "preference", "eval"],
        risk_controls=["pii_filter", "policy_consistency", "audit_trail"],
        data_buckets=["policies", "regulations", "cases", "transactions", "analyst_notes"],
        notes="Optimize for traceable reasoning and calibrated refusal on unsupported claims.",
    ),
    "semiconductor": DomainRecipe(
        name="semiconductor",
        recommended_stages=["pretrain", "sft", "preference", "eval"],
        risk_controls=["ip_filter", "source_access_control", "evidence_required"],
        data_buckets=["process_docs", "equipment_logs", "specs", "patents", "papers", "qa"],
        notes="Keep fab/vendor confidential documents isolated by permission boundary.",
    ),
}


@dataclass
class DomainBuildConfig:
    domain: str
    base_model: str
    output_dir: str = "./domain_model"
    model_config: str = "llama-300m"
    tokenizer_name_or_path: Optional[str] = None
    pretrain_mode: str = "continue"
    seed: int = 42
    max_seq_length: int = 2048


class DomainModelBuilder:
    """Build and optionally run a domain-model training recipe."""

    def __init__(
        self,
        domain: str,
        base_model: str,
        output_dir: str = "./domain_model",
        model_config: str = "llama-300m",
        tokenizer_name_or_path: Optional[str] = None,
    ):
        domain_key = domain.lower()
        if domain_key not in DOMAIN_RECIPES:
            available = ", ".join(sorted(DOMAIN_RECIPES))
            raise KeyError(f"Unknown domain: {domain}. Available domains: {available}")
        self.config = DomainBuildConfig(
            domain=domain_key,
            base_model=base_model,
            output_dir=output_dir,
            model_config=model_config,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        self.recipe = DOMAIN_RECIPES[domain_key]

    def architecture_report(self) -> Dict:
        from ..models.Architecture import ArchitectureRegistry

        support = ArchitectureRegistry.detect_from_model_name(self.config.base_model)
        return {
            "base_model": self.config.base_model,
            "architecture": support.to_dict(),
            "domain": asdict(self.recipe),
        }

    def training_strategy(
        self,
        target_family: Optional[str] = None,
        budget: str = "low",
        prefer_scratch: bool = False,
        teacher_models: Optional[Sequence[str]] = None,
    ) -> Dict:
        from .TrainingStrategyAdvisor import TrainingStrategyAdvisor

        family = target_family or self.config.base_model
        return TrainingStrategyAdvisor.analyze(
            target_family=family,
            base_model=self.config.base_model,
            domain=self.config.domain,
            budget=budget,
            prefer_scratch=prefer_scratch,
            teacher_models=teacher_models,
        ).to_dict()

    def low_cost_recipe(
        self,
        target_family: Optional[str] = None,
        teacher_models: Optional[Sequence[str]] = None,
    ) -> List[str]:
        from .TrainingStrategyAdvisor import TrainingStrategyAdvisor

        return TrainingStrategyAdvisor.low_cost_recipe(
            target_family=target_family or self.config.base_model,
            base_model=self.config.base_model,
            domain=self.config.domain,
            teacher_models=teacher_models,
        )

    def data_mix_plan(
        self,
        sources_by_bucket: Optional[Dict[str, Sequence[Dict]]] = None,
        output_dir: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        domain_boost: float = 0.0,
    ) -> Dict:
        from ..data.DataStrategy import DataMixPlanner

        plan = DataMixPlanner.for_domain(self.config.domain, domain_boost=domain_boost)
        if sources_by_bucket:
            return plan.to_pipeline_config(
                sources_by_bucket=sources_by_bucket,
                output_dir=output_dir or os.path.join(self.config.output_dir, "processed_data"),
                max_seq_length=max_seq_length or self.config.max_seq_length,
            )
        return plan.to_dict()

    def contamination_report(
        self,
        reference_texts: Sequence[str],
        corpus_files: Union[str, Sequence[str]],
        text_column: str = "text",
        threshold: float = 0.35,
        max_samples: Optional[int] = None,
    ) -> Dict:
        from ..data.DataStrategy import ContaminationDetector

        detector = ContaminationDetector(threshold=threshold).add_references(reference_texts)
        return detector.scan_files(
            corpus_files,
            text_column=text_column,
            max_samples=max_samples,
        ).to_dict()

    def eval_suite(self, max_samples: int = 500, save_path: Optional[str] = None) -> Dict:
        from ..evaluation.PretrainEvalSuite import PretrainEvalSuite

        suite = PretrainEvalSuite.for_domain(self.config.domain, max_samples=max_samples)
        if save_path:
            suite.save(save_path)
        return suite.to_dict()

    def pretrain_experiments(
        self,
        output_dir: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        token_multipliers: Optional[List[float]] = None,
        corpus_sources: Optional[Sequence[Dict]] = None,
        sources_by_bucket: Optional[Dict[str, Sequence[Dict]]] = None,
        dry_run: bool = True,
    ) -> Dict:
        from ..experiments.PretrainExperimentRunner import PretrainExperimentConfig, PretrainExperimentRunner

        runner = PretrainExperimentRunner(PretrainExperimentConfig(
            domain=self.config.domain,
            output_dir=output_dir or os.path.join(self.config.output_dir, "pretrain_experiments"),
            model_names=model_names or ["qwen-tiny-160m", "qwen-300m"],
            token_multipliers=token_multipliers or [0.05, 0.1],
            tokenizer_path=self.config.tokenizer_name_or_path,
            corpus_sources=list(corpus_sources or []),
            sources_by_bucket=sources_by_bucket,
            seq_length=self.config.max_seq_length,
        ))
        return runner.run(dry_run=dry_run).to_dict()

    def build_config(
        self,
        corpus_sources: Optional[Sequence[Union[str, Dict]]] = None,
        sft_data: Optional[str] = None,
        preference_data: Optional[str] = None,
        eval_data: Optional[str] = None,
        stages: Optional[List[str]] = None,
        max_steps: int = 1000,
        sft_epochs: int = 1,
        preference_epochs: int = 1,
        learning_rate: float = 2e-5,
        sft_learning_rate: float = 2e-4,
        preference_learning_rate: float = 5e-6,
    ) -> Dict:
        stage_list = stages or self._default_stages(corpus_sources, sft_data, preference_data, eval_data)
        sources = self._normalize_sources(corpus_sources or [])

        cfg = {
            "project": "saddlellm-domain",
            "experiment": f"{self.config.domain}-{self._safe_model_name(self.config.base_model)}",
            "seed": self.config.seed,
            "pretrain_mode": self.config.pretrain_mode,
            "stages": stage_list,
            "model": {
                "config": self.config.model_config,
                "name_or_path": self.config.base_model,
                "tokenizer": self.config.tokenizer_name_or_path or self.config.base_model,
            },
            "data": {
                "sources": sources,
                "output_dir": os.path.join(self.config.output_dir, "processed_data"),
                "max_seq_length": self.config.max_seq_length,
                "min_text_length": 50,
                "dedup_method": "minhash",
                "quality_min_score": 0.35,
                "pack_sequences": True,
            },
            "training": {
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "per_device_batch_size": 1,
                "global_batch_size": 64,
                "warmup_steps": min(100, max(1, max_steps // 20)),
                "save_every_steps": max(100, max_steps // 5),
                "eval_every_steps": max(100, max_steps // 5),
                "log_every_steps": 10,
            },
            "logging": {
                "output_dir": self.config.output_dir,
                "backend": "tensorboard",
            },
            "sft": {
                "enabled": bool(sft_data),
                "data_path": sft_data or "",
                "epochs": sft_epochs,
                "learning_rate": sft_learning_rate,
                "per_device_batch_size": 1,
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
            },
            "preference": {
                "enabled": bool(preference_data),
                "method": "dpo",
                "data_path": preference_data or "",
                "epochs": preference_epochs,
                "learning_rate": preference_learning_rate,
                "per_device_batch_size": 1,
                "beta": 0.1,
                "use_lora": True,
                "use_qlora": True,
            },
            "eval": {
                "enabled": bool(eval_data),
                "dataset": eval_data or "wikitext",
                "dataset_config": None if eval_data else "wikitext-2-raw-v1",
                "tasks": ["perplexity"],
                "max_samples": 500,
            },
            "domain": asdict(self.recipe),
        }
        return cfg

    def save_config(self, path: str, config: Dict) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path

    def create_orchestrator(self, config: Dict = None):
        from ..training.TrainingOrchestrator import TrainingOrchestrator

        return TrainingOrchestrator.from_dict(config or self.build_config())

    def run(self, config: Dict = None):
        orchestrator = self.create_orchestrator(config)
        orchestrator.run()
        return orchestrator

    def sft(self, dataset_path: str, output_path: Optional[str] = None, **kwargs):
        from ..training.PeftSFTTrainer import train_model

        return train_model(
            model_path=kwargs.pop("model_path", self.config.base_model),
            dataset_path=dataset_path,
            output_path=output_path or os.path.join(self.config.output_dir, "sft"),
            **kwargs,
        )

    def preference_train(self, dataset_path: str, output_path: Optional[str] = None, method: str = "dpo", **kwargs):
        from ..training.Preference import PreferenceTrainConfig, train_preference

        cfg = PreferenceTrainConfig(
            model_path=kwargs.pop("model_path", self.config.base_model),
            dataset_path=dataset_path,
            output_path=output_path or os.path.join(self.config.output_dir, "preference"),
            method=method,
            **kwargs,
        )
        return train_preference(cfg)

    def evaluate(self, dataset_path: str, model_path: Optional[str] = None, **kwargs):
        from ..evaluation.LLModelEvalute import Evaluator

        return Evaluator().evaluate(
            model_path=model_path or self.config.base_model,
            dataset=dataset_path,
            metrics=kwargs.pop("metrics", ["perplexity"]),
            **kwargs,
        )

    def _default_stages(self, corpus_sources, sft_data, preference_data, eval_data) -> List[str]:
        stages = []
        if corpus_sources:
            stages.append("pretrain")
        if sft_data:
            stages.append("sft")
        if preference_data:
            stages.append("preference")
        if eval_data:
            stages.append("eval")
        return stages or ["eval"]

    def _normalize_sources(self, sources: Sequence[Union[str, Dict]]) -> List[Dict]:
        normalized = []
        for source in sources:
            if isinstance(source, str):
                normalized.append({
                    "type": "local",
                    "path": source,
                    "format": "auto",
                    "text_column": "text",
                    "streaming": True,
                })
            else:
                normalized.append(dict(source))
        return normalized

    def _safe_model_name(self, model_name: str) -> str:
        return model_name.replace("/", "-").replace("\\", "-").replace(":", "-")


def list_domain_recipes() -> Dict[str, Dict]:
    return {name: asdict(recipe) for name, recipe in DOMAIN_RECIPES.items()}
