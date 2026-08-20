"""Feasibility advisor for pretraining and post-training plans.

This module is intentionally conservative.  It distinguishes what SaddleLLM can
instantiate natively from what can be fine-tuned through an existing
HuggingFace-compatible checkpoint, and from what is only realistic through
distillation or API-generated data on a low budget.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .Architecture import CUSTOM_REQUIRED, HF_AUTO, NATIVE, ArchitectureRegistry, ArchitectureSupport


@dataclass(frozen=True)
class BudgetProfile:
    name: str
    description: str
    scratch_scope: str
    can_frontier_scratch: bool = False
    can_native_small_scratch: bool = False
    preferred_model_scale: str = "1B-7B"


BUDGETS: Dict[str, BudgetProfile] = {
    "low": BudgetProfile(
        name="low",
        description="single workstation or a few rented GPUs",
        scratch_scope="toy/small models only; use checkpoints for real capability",
        can_native_small_scratch=True,
        preferred_model_scale="1B-7B QLoRA/LoRA",
    ),
    "team": BudgetProfile(
        name="team",
        description="small team with multi-GPU nodes",
        scratch_scope="small dense models may be possible; frontier scratch remains unrealistic",
        can_native_small_scratch=True,
        preferred_model_scale="7B-14B continued pretrain + LoRA/full SFT",
    ),
    "lab": BudgetProfile(
        name="lab",
        description="cluster budget with dedicated data and infra",
        scratch_scope="native dense/Mixtral-style experiments are possible",
        can_native_small_scratch=True,
        preferred_model_scale="7B-70B continued pretrain/post-train",
    ),
    "frontier": BudgetProfile(
        name="frontier",
        description="large-scale model lab",
        scratch_scope="frontier scratch possible only with custom infra and very large corpora",
        can_frontier_scratch=True,
        can_native_small_scratch=True,
        preferred_model_scale="custom large dense/MoE",
    ),
}


@dataclass
class TrainingRoute:
    name: str
    feasible: bool
    confidence: str
    stages: List[str]
    reason: str
    requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeasibilityReport:
    target_family: str
    budget: str
    base_model: Optional[str]
    architecture: Dict
    can_train_from_scratch: bool
    recommended_path: str
    routes: List[TrainingRoute]
    blockers: List[str]
    low_cost_recipe: List[str]
    notes: List[str]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["routes"] = [route.to_dict() for route in self.routes]
        return data

    def to_markdown(self) -> str:
        lines = [
            f"# Training feasibility: {self.target_family}",
            "",
            f"- Budget: {self.budget}",
            f"- Base model: {self.base_model or 'not specified'}",
            f"- Native scratch: {'yes' if self.can_train_from_scratch else 'no'}",
            f"- Recommended path: {self.recommended_path}",
            "",
            "## Routes",
        ]
        for route in self.routes:
            mark = "yes" if route.feasible else "no"
            lines.append(f"- {route.name}: {mark} ({route.confidence}) - {route.reason}")
        if self.blockers:
            lines.extend(["", "## Blockers"])
            lines.extend(f"- {item}" for item in self.blockers)
        if self.low_cost_recipe:
            lines.extend(["", "## Low-cost recipe"])
            lines.extend(f"{i + 1}. {step}" for i, step in enumerate(self.low_cost_recipe))
        if self.notes:
            lines.extend(["", "## Notes"])
            lines.extend(f"- {item}" for item in self.notes)
        return "\n".join(lines)


class TrainingStrategyAdvisor:
    """Analyze realistic training paths for a target model family."""

    DEFAULT_STUDENTS = {
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "glm": "zai-org/glm-4-9b-hf",
        "minimax": "Qwen/Qwen2.5-7B-Instruct",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "llama": "meta-llama/Llama-3.1-8B",
        "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    }

    DOMAIN_EVALS = {
        "medical": ["clinical QA", "evidence citation", "refusal/safety"],
        "biology": ["paper QA", "biosecurity refusal", "protocol safety"],
        "research": ["citation faithfulness", "method summary", "long-context QA"],
        "risk": ["policy consistency", "auditability", "calibrated refusal"],
        "semiconductor": ["process QA", "patent/spec retrieval", "IP leakage checks"],
    }

    @staticmethod
    def analyze(
        target_family: str,
        base_model: Optional[str] = None,
        domain: Optional[str] = None,
        budget: str = "low",
        prefer_scratch: bool = False,
        teacher_models: Optional[Sequence[str]] = None,
    ) -> FeasibilityReport:
        budget_profile = BUDGETS.get((budget or "low").lower(), BUDGETS["low"])
        support = TrainingStrategyAdvisor._resolve_architecture(target_family, base_model)
        family = support.name
        teacher_list = list(teacher_models or [])
        selected_base = base_model or TrainingStrategyAdvisor.DEFAULT_STUDENTS.get(family)

        can_scratch = bool(
            support.native_pretrain
            and budget_profile.can_native_small_scratch
            and (budget_profile.can_frontier_scratch or family in {"llama", "gpt2", "gpt-neox", "mixtral"})
        )
        if family in {"deepseek", "minimax", "glm"} and not budget_profile.can_frontier_scratch:
            can_scratch = False

        routes = [
            TrainingStrategyAdvisor._scratch_route(support, budget_profile, can_scratch),
            TrainingStrategyAdvisor._continue_route(support, selected_base),
            TrainingStrategyAdvisor._sft_route(support, selected_base),
            TrainingStrategyAdvisor._preference_route(support, selected_base),
            TrainingStrategyAdvisor._rl_route(support, selected_base),
            TrainingStrategyAdvisor._distill_route(support, selected_base, teacher_list),
        ]
        blockers = TrainingStrategyAdvisor._blockers(support, budget_profile, prefer_scratch)
        recipe = TrainingStrategyAdvisor.low_cost_recipe(
            target_family=family,
            base_model=selected_base,
            domain=domain,
            teacher_models=teacher_list,
        )
        recommended = TrainingStrategyAdvisor._recommended_path(support, budget_profile, prefer_scratch)
        notes = TrainingStrategyAdvisor._notes(support, budget_profile, domain)

        return FeasibilityReport(
            target_family=family,
            budget=budget_profile.name,
            base_model=selected_base,
            architecture=support.to_dict(),
            can_train_from_scratch=can_scratch,
            recommended_path=recommended,
            routes=routes,
            blockers=blockers,
            low_cost_recipe=recipe,
            notes=notes,
        )

    @staticmethod
    def low_cost_recipe(
        target_family: str,
        base_model: Optional[str] = None,
        domain: Optional[str] = None,
        teacher_models: Optional[Sequence[str]] = None,
    ) -> List[str]:
        family = ArchitectureRegistry.normalize_name(target_family)
        student = base_model or TrainingStrategyAdvisor.DEFAULT_STUDENTS.get(family, "Qwen/Qwen2.5-7B-Instruct")
        teachers = ", ".join(teacher_models or TrainingStrategyAdvisor._default_teachers(family))
        domain_text = f" for {domain}" if domain else ""
        evals = TrainingStrategyAdvisor.DOMAIN_EVALS.get((domain or "").lower(), ["perplexity", "task QA", "safety"])
        return [
            f"Start from student checkpoint {student}; do not train a frontier architecture from random init on a low budget.",
            f"Run domain continued pretraining{domain_text} on cleaned corpus with dedup, quality scoring, and held-out perplexity.",
            "Run SFT on curated instruction/answer data; keep LoRA/QLoRA as the default cost mode.",
            "Run DPO/ORPO/KTO on preference pairs for answer style, refusal, and factual discipline.",
            "Use RL scaling/GRPO only on verifiable tasks such as math, code, retrieval-grounded QA, or rule-based risk checks.",
            f"Distill from teacher model(s): {teachers}; filter low-quality generations before SFT.",
            f"Evaluate with: {', '.join(evals)}; keep domain-specific regression sets fixed across runs.",
        ]

    @staticmethod
    def _resolve_architecture(target_family: str, base_model: Optional[str]) -> ArchitectureSupport:
        try:
            return ArchitectureRegistry.get(target_family)
        except Exception:
            if base_model:
                return ArchitectureRegistry.detect_from_model_name(base_model)
            return ArchitectureRegistry.detect_from_model_name(target_family)

    @staticmethod
    def _scratch_route(support: ArchitectureSupport, budget: BudgetProfile, can_scratch: bool) -> TrainingRoute:
        if can_scratch:
            return TrainingRoute(
                name="scratch_pretrain",
                feasible=True,
                confidence="medium",
                stages=["tokenizer", "pretrain", "eval"],
                reason=f"Architecture is native in SaddleLLM; budget scope is {budget.scratch_scope}.",
                requirements=["large clean corpus", "tokenizer training", "scaling-law run grid"],
            )
        reason = "native config is not implemented" if not support.native_pretrain else budget.scratch_scope
        return TrainingRoute(
            name="scratch_pretrain",
            feasible=False,
            confidence="high",
            stages=["tokenizer", "pretrain", "eval"],
            reason=reason,
            requirements=["custom modeling code" if support.support_level == CUSTOM_REQUIRED else "more compute/data"],
        )

    @staticmethod
    def _continue_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute:
        feasible = bool(support.hf_auto_load and base_model)
        return TrainingRoute(
            name="continued_pretrain",
            feasible=feasible,
            confidence="high" if feasible else "medium",
            stages=["pretrain", "eval"],
            reason="uses existing checkpoint through AutoModelForCausalLM" if feasible else "requires a base checkpoint",
            requirements=["domain corpus", "tokenizer compatible with base model"],
        )

    @staticmethod
    def _sft_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute:
        feasible = bool(support.sft and base_model)
        return TrainingRoute(
            name="sft_lora_qlora",
            feasible=feasible,
            confidence="high" if feasible else "medium",
            stages=["sft", "eval"],
            reason="standard low-cost adaptation path" if feasible else "requires CausalLM checkpoint support",
            requirements=["instruction dataset", "chat template validation"],
        )

    @staticmethod
    def _preference_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute:
        feasible = bool(support.preference and base_model)
        return TrainingRoute(
            name="preference_dpo_orpo_kto",
            feasible=feasible,
            confidence="medium" if feasible else "low",
            stages=["preference", "eval"],
            reason="preference trainer can operate on existing CausalLM checkpoints" if feasible else "unsupported architecture or missing checkpoint",
            requirements=["chosen/rejected pairs", "reward/safety eval"],
        )

    @staticmethod
    def _rl_route(support: ArchitectureSupport, base_model: Optional[str]) -> TrainingRoute:
        feasible = bool((support.grpo or support.rl_scaling) and base_model)
        return TrainingRoute(
            name="rl_scaling_grpo",
            feasible=feasible,
            confidence="medium" if feasible else "low",
            stages=["rollout", "reward", "filter", "grpo"],
            reason="best for verifiable long-reasoning tasks, not generic knowledge injection" if feasible else "requires generation-capable checkpoint",
            requirements=["reward functions", "prompt set", "rollout budget"],
        )

    @staticmethod
    def _distill_route(
        support: ArchitectureSupport,
        base_model: Optional[str],
        teacher_models: Sequence[str],
    ) -> TrainingRoute:
        feasible = bool(base_model or support.hf_auto_load)
        teacher_text = ", ".join(teacher_models) if teacher_models else "API/local teacher"
        return TrainingRoute(
            name="distillation",
            feasible=feasible,
            confidence="high" if feasible else "medium",
            stages=["generate", "filter", "sft", "preference", "eval"],
            reason=f"most realistic low-cost path to imitate target behavior using {teacher_text}",
            requirements=["teacher access", "prompt coverage", "quality filtering"],
        )

    @staticmethod
    def _blockers(support: ArchitectureSupport, budget: BudgetProfile, prefer_scratch: bool) -> List[str]:
        blockers = []
        if support.support_level == CUSTOM_REQUIRED:
            blockers.append("native modeling/training code is required before faithful scratch pretraining")
        if support.name == "deepseek":
            blockers.append("DeepSeek-style MLA, DeepSeekMoE routing, aux-loss-free balancing, and MTP are not a full native stack here")
        if support.name == "minimax":
            blockers.append("MiniMax-style Lightning Attention plus large MoE is not implemented natively")
        if support.name == "glm":
            blockers.append("GLM scratch config/training is not wired; use existing GLM checkpoints or another student architecture")
        if not budget.can_frontier_scratch:
            blockers.append("frontier-model scratch training needs far more data, compute, tokenizer work, and distributed infra than a low-cost setup")
        if prefer_scratch and not support.native_pretrain:
            blockers.append("requested scratch path conflicts with current architecture support")
        return blockers

    @staticmethod
    def _recommended_path(support: ArchitectureSupport, budget: BudgetProfile, prefer_scratch: bool) -> str:
        if prefer_scratch and support.native_pretrain and budget.can_frontier_scratch:
            return "scratch pretrain plus SFT/preference/RL"
        if support.name in {"deepseek", "minimax"}:
            return "distill into a Qwen/Llama/DeepSeek-distill student, then domain continued pretrain + SFT + preference/RL"
        if support.name == "glm":
            return "fine-tune an existing GLM checkpoint, or use Qwen/Llama as the student and distill GLM behavior"
        if budget.name in {"low", "team"}:
            return "continued pretrain on a strong checkpoint, then SFT, preference training, and selective distillation"
        return "native scratch only for supported architectures; otherwise checkpoint adaptation"

    @staticmethod
    def _notes(support: ArchitectureSupport, budget: BudgetProfile, domain: Optional[str]) -> List[str]:
        notes = [
            f"Budget profile: {budget.description}; preferred scale: {budget.preferred_model_scale}.",
            f"Architecture support level: {support.support_level}.",
        ]
        if support.support_level == HF_AUTO:
            notes.append("HF AutoModel support means adaptation is practical; it does not imply native random-init pretraining.")
        if support.support_level == NATIVE:
            notes.append("Native support means SaddleLLM can create a config from scratch, but capability still depends on data/compute scale.")
        if domain:
            notes.append(f"Domain '{domain}' should keep a fixed eval and safety set before training starts.")
        return notes

    @staticmethod
    def _default_teachers(family: str) -> List[str]:
        if family == "deepseek":
            return ["DeepSeek-R1/DeepSeek-V3 API or local distill checkpoint"]
        if family == "glm":
            return ["GLM-4 API/checkpoint"]
        if family == "minimax":
            return ["MiniMax API/checkpoint"]
        return ["strong API teacher", "local larger checkpoint"]

