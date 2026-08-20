"""Architecture support matrix for SaddleLLM.

This registry answers a different question than ``ModelRegistry``:

* ModelRegistry: which model sizes/configs can SaddleLLM create?
* ArchitectureRegistry: which model families can each training stage support?

The levels are intentionally conservative.  ``hf_auto`` means SaddleLLM can
usually fine-tune/load an existing checkpoint through HuggingFace
``AutoModelForCausalLM``.  ``native`` means SaddleLLM can also instantiate a
model config for pretraining from scratch.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


NATIVE = "native"
HF_AUTO = "hf_auto"
CUSTOM_REQUIRED = "custom_required"
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ArchitectureSupport:
    name: str
    family: str
    support_level: str
    model_type_aliases: List[str] = field(default_factory=list)
    example_models: List[str] = field(default_factory=list)
    native_pretrain: bool = False
    hf_auto_load: bool = True
    causal_lm: bool = True
    sft: bool = True
    lora: bool = True
    qlora: bool = True
    preference: bool = True
    grpo: bool = True
    rl_scaling: bool = True
    quantization: bool = True
    export: bool = True
    notes: str = ""

    def supports(self, capability: str) -> bool:
        aliases = {
            "pretrain": "native_pretrain",
            "continue_pretrain": "hf_auto_load",
            "load": "hf_auto_load",
            "inference": "hf_auto_load",
            "rl": "rl_scaling",
        }
        attr = aliases.get(capability, capability)
        return bool(getattr(self, attr, False))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "family": self.family,
            "support_level": self.support_level,
            "model_type_aliases": self.model_type_aliases,
            "example_models": self.example_models,
            "native_pretrain": self.native_pretrain,
            "hf_auto_load": self.hf_auto_load,
            "causal_lm": self.causal_lm,
            "sft": self.sft,
            "lora": self.lora,
            "qlora": self.qlora,
            "preference": self.preference,
            "grpo": self.grpo,
            "rl_scaling": self.rl_scaling,
            "quantization": self.quantization,
            "export": self.export,
            "notes": self.notes,
        }


ARCHITECTURES: Dict[str, ArchitectureSupport] = {
    "llama": ArchitectureSupport(
        name="llama",
        family="dense_transformer",
        support_level=NATIVE,
        model_type_aliases=["llama", "mllama_text_model"],
        example_models=["meta-llama/Llama-3.1-8B", "TinyLlama/TinyLlama-1.1B"],
        native_pretrain=True,
        notes="Native from-scratch config via LlamaConfig; strongest support path.",
    ),
    "gpt2": ArchitectureSupport(
        name="gpt2",
        family="dense_transformer",
        support_level=NATIVE,
        model_type_aliases=["gpt2"],
        example_models=["openai-community/gpt2"],
        native_pretrain=True,
        notes="Native GPT-2 style pretraining config.",
    ),
    "gpt-neox": ArchitectureSupport(
        name="gpt-neox",
        family="dense_transformer",
        support_level=NATIVE,
        model_type_aliases=["gpt_neox", "gpt-neox"],
        example_models=["EleutherAI/pythia-410m"],
        native_pretrain=True,
        notes="Native GPT-NeoX/Pythia style pretraining config.",
    ),
    "mixtral": ArchitectureSupport(
        name="mixtral",
        family="moe_transformer",
        support_level=NATIVE,
        model_type_aliases=["mixtral"],
        example_models=["mistralai/Mixtral-8x7B-v0.1"],
        native_pretrain=True,
        notes="Native MixtralConfig MoE support; requires much more memory than dense models.",
    ),
    "mistral": ArchitectureSupport(
        name="mistral",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["mistral"],
        example_models=["mistralai/Mistral-7B-v0.1"],
        native_pretrain=False,
        notes="Existing checkpoints fine-tune via HF AutoModel; native scratch config not wired yet.",
    ),
    "qwen": ArchitectureSupport(
        name="qwen",
        family="dense_transformer",
        support_level=NATIVE,
        model_type_aliases=["qwen", "qwen2", "qwen3", "qwen2_moe"],
        example_models=["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"],
        native_pretrain=True,
        notes="Native Qwen2Config scratch support plus HF AutoModel checkpoint adaptation.",
    ),
    "deepseek": ArchitectureSupport(
        name="deepseek",
        family="moe_transformer",
        support_level=CUSTOM_REQUIRED,
        model_type_aliases=["deepseek", "deepseek_v2", "deepseek_v3"],
        example_models=["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"],
        native_pretrain=False,
        hf_auto_load=True,
        notes=(
            "Distill checkpoints can usually be fine-tuned through their base architecture. "
            "Native DeepSeek-V3 MLA/aux-loss-free/MTP is not fully implemented."
        ),
    ),
    "minimax": ArchitectureSupport(
        name="minimax",
        family="hybrid_attention_moe",
        support_level=CUSTOM_REQUIRED,
        model_type_aliases=["minimax", "minimax_text", "minimax-text", "minimax01", "minimax-01"],
        example_models=["MiniMaxAI/MiniMax-Text-01"],
        native_pretrain=False,
        hf_auto_load=True,
        notes=(
            "Existing checkpoints may be usable when their HuggingFace modeling code is installed. "
            "Native Lightning Attention/MoE pretraining is not implemented."
        ),
    ),
    "glm": ArchitectureSupport(
        name="glm",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["chatglm", "glm", "glm4"],
        example_models=["THUDM/glm-4-9b-chat"],
        native_pretrain=False,
        notes="Requires trust_remote_code for many checkpoints.",
    ),
    "internlm": ArchitectureSupport(
        name="internlm",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["internlm", "internlm2", "internlm3"],
        example_models=["internlm/internlm2_5-7b-chat"],
        native_pretrain=False,
    ),
    "baichuan": ArchitectureSupport(
        name="baichuan",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["baichuan"],
        example_models=["baichuan-inc/Baichuan2-7B-Base"],
        native_pretrain=False,
    ),
    "yi": ArchitectureSupport(
        name="yi",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["yi"],
        example_models=["01-ai/Yi-1.5-9B"],
        native_pretrain=False,
    ),
    "phi": ArchitectureSupport(
        name="phi",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["phi", "phi3", "phimoe"],
        example_models=["microsoft/Phi-3-mini-4k-instruct"],
        native_pretrain=False,
    ),
    "gemma": ArchitectureSupport(
        name="gemma",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["gemma", "gemma2", "gemma3_text"],
        example_models=["google/gemma-2-9b"],
        native_pretrain=False,
    ),
    "falcon": ArchitectureSupport(
        name="falcon",
        family="dense_transformer",
        support_level=HF_AUTO,
        model_type_aliases=["falcon"],
        example_models=["tiiuae/falcon-7b"],
        native_pretrain=False,
    ),
    "mamba": ArchitectureSupport(
        name="mamba",
        family="state_space_model",
        support_level=CUSTOM_REQUIRED,
        model_type_aliases=["mamba", "mamba2"],
        example_models=["state-spaces/mamba-130m-hf"],
        native_pretrain=False,
        sft=False,
        lora=False,
        qlora=False,
        preference=False,
        grpo=False,
        rl_scaling=False,
        notes="SSM architecture needs dedicated modeling/training adapters before regular SFT/RL.",
    ),
    "rwkv": ArchitectureSupport(
        name="rwkv",
        family="rnn_transformer_hybrid",
        support_level=CUSTOM_REQUIRED,
        model_type_aliases=["rwkv"],
        native_pretrain=False,
        sft=False,
        lora=False,
        qlora=False,
        preference=False,
        grpo=False,
        rl_scaling=False,
        notes="Needs custom training loop and tokenizer/model wrappers.",
    ),
    "encoder_decoder": ArchitectureSupport(
        name="encoder_decoder",
        family="seq2seq",
        support_level=UNSUPPORTED,
        model_type_aliases=["t5", "bart", "mt5"],
        example_models=["google/flan-t5-large"],
        native_pretrain=False,
        causal_lm=False,
        sft=False,
        lora=True,
        qlora=False,
        preference=False,
        grpo=False,
        rl_scaling=False,
        notes="Current framework is CausalLM-first; seq2seq needs separate trainers.",
    ),
    "encoder_only": ArchitectureSupport(
        name="encoder_only",
        family="encoder",
        support_level=UNSUPPORTED,
        model_type_aliases=["bert", "roberta", "deberta"],
        native_pretrain=False,
        causal_lm=False,
        sft=False,
        lora=True,
        qlora=False,
        preference=False,
        grpo=False,
        rl_scaling=False,
        notes="Use for classifiers/embeddings, not current CausalLM post-training path.",
    ),
    "multimodal": ArchitectureSupport(
        name="multimodal",
        family="vision_language",
        support_level=CUSTOM_REQUIRED,
        model_type_aliases=["llava", "qwen2_vl", "internvl", "mllama"],
        native_pretrain=False,
        sft=False,
        lora=True,
        qlora=False,
        preference=False,
        grpo=False,
        rl_scaling=False,
        notes="Needs image/video processors and multimodal collators before full support.",
    ),
}


class ArchitectureRegistry:
    """Query and validate architecture support."""

    @staticmethod
    def get(name: str) -> ArchitectureSupport:
        key = ArchitectureRegistry.normalize_name(name)
        if key not in ARCHITECTURES:
            raise KeyError(f"Unknown architecture: {name}. Available: {', '.join(sorted(ARCHITECTURES))}")
        return ARCHITECTURES[key]

    @staticmethod
    def list_all(level: Optional[str] = None) -> List[ArchitectureSupport]:
        values = list(ARCHITECTURES.values())
        if level:
            values = [v for v in values if v.support_level == level]
        return sorted(values, key=lambda x: (x.support_level, x.name))

    @staticmethod
    def normalize_name(name: str) -> str:
        raw = (name or "").lower().replace("_", "-")
        alias_map = {}
        for arch, support in ARCHITECTURES.items():
            alias_map[arch.lower().replace("_", "-")] = arch
            for alias in support.model_type_aliases:
                alias_map[alias.lower().replace("_", "-")] = arch
        if raw in alias_map:
            return alias_map[raw]
        for alias, arch in alias_map.items():
            if alias and alias in raw:
                return arch
        return raw

    @staticmethod
    def detect_from_model_name(model_name_or_path: str) -> ArchitectureSupport:
        text = (model_name_or_path or "").lower()
        heuristics = [
            ("qwen", "qwen"),
            ("deepseek", "deepseek"),
            ("minimax", "minimax"),
            ("chatglm", "glm"),
            ("glm", "glm"),
            ("internlm", "internlm"),
            ("baichuan", "baichuan"),
            ("mixtral", "mixtral"),
            ("mistral", "mistral"),
            ("llama", "llama"),
            ("tinyllama", "llama"),
            ("gpt2", "gpt2"),
            ("pythia", "gpt-neox"),
            ("gpt-neox", "gpt-neox"),
            ("gemma", "gemma"),
            ("phi", "phi"),
            ("falcon", "falcon"),
            ("mamba", "mamba"),
            ("rwkv", "rwkv"),
            ("llava", "multimodal"),
            ("qwen-vl", "multimodal"),
            ("internvl", "multimodal"),
            ("flan-t5", "encoder_decoder"),
            ("t5", "encoder_decoder"),
            ("bert", "encoder_only"),
        ]
        for needle, arch in heuristics:
            if needle in text:
                return ARCHITECTURES[arch]
        return ArchitectureSupport(
            name="unknown",
            family="unknown",
            support_level=HF_AUTO,
            model_type_aliases=[],
            example_models=[],
            native_pretrain=False,
            notes="Unknown model family; will try HuggingFace AutoModelForCausalLM.",
        )

    @staticmethod
    def detect_from_hf_config(model_name_or_path: str, trust_remote_code: bool = True) -> ArchitectureSupport:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            model_type = getattr(cfg, "model_type", "") or cfg.__class__.__name__
            return ArchitectureRegistry.get(model_type)
        except Exception:
            return ArchitectureRegistry.detect_from_model_name(model_name_or_path)

    @staticmethod
    def from_model_spec(spec) -> ArchitectureSupport:
        return ArchitectureRegistry.get(getattr(spec, "architecture", ""))

    @staticmethod
    def supports(architecture: str, capability: str) -> bool:
        return ArchitectureRegistry.get(architecture).supports(capability)

    @staticmethod
    def require(architecture: str, capability: str) -> ArchitectureSupport:
        support = ArchitectureRegistry.get(architecture)
        if not support.supports(capability):
            raise ValueError(
                f"Architecture {support.name!r} does not support capability {capability!r}. "
                f"Support level={support.support_level}. Notes: {support.notes}"
            )
        return support

    @staticmethod
    def recommend(stage: str) -> List[ArchitectureSupport]:
        stage = stage.lower()
        return [arch for arch in ArchitectureRegistry.list_all() if arch.supports(stage)]

    @staticmethod
    def support_matrix(markdown: bool = True) -> str:
        rows = []
        headers = ["Architecture", "Level", "Native PT", "HF Load", "SFT", "Pref", "GRPO", "Notes"]
        for arch in ArchitectureRegistry.list_all():
            rows.append([
                arch.name,
                arch.support_level,
                "Y" if arch.native_pretrain else "-",
                "Y" if arch.hf_auto_load else "-",
                "Y" if arch.sft else "-",
                "Y" if arch.preference else "-",
                "Y" if arch.grpo else "-",
                arch.notes,
            ])
        if not markdown:
            return "\n".join("\t".join(row) for row in [headers] + rows)

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
        return "\n".join(lines)
