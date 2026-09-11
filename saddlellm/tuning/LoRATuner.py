import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from typing import Dict, Optional, List

from ..utils.OptionalDependencies import require_distribution


class LoRATuner:
    """
    A sklearn-style wrapper for LoRA fine-tuning.
    Example:
    >>> lora = LoRATuner(model_name="Llama-3-8B")
    >>> lora.fit(train_dataset)
    >>> lora.predict("What is AI?")
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            max_length: int = 1024,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            target_modules: Optional[List[str]] = None,
            use_4bit: bool = True,  # QLoRA if True
            device_map: str = "auto",
            trust_remote_code: bool = False,
            local_files_only: bool = False,
            adapter_path: Optional[str] = None,
            adapter_trainable: bool = True,
    ):
        """
        Initialize LoRA fine-tuning.
        :param model_name: Hugging Face model ID
        :param use_4bit: Enable 4-bit quantization (QLoRA)
        :param target_modules: Modules to apply LoRA (e.g., ["q_proj", "v_proj"])
        """
        if not model_name:
            raise ValueError("model_name is required; LoRATuner never downloads a default model")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if lora_rank < 1:
            raise ValueError("lora_rank must be at least 1")
        if lora_alpha < 1:
            raise ValueError("lora_alpha must be at least 1")
        if not 0 <= lora_dropout < 1:
            raise ValueError("lora_dropout must be in the range [0, 1)")
        resolved_targets = ["q_proj", "v_proj"] if target_modules is None else list(target_modules)
        if not resolved_targets:
            raise ValueError("target_modules must contain at least one module name")

        self.model_name = model_name
        self.max_length = max_length
        self.use_4bit = bool(use_4bit)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.adapter_trainable = bool(adapter_trainable)

        if self.use_4bit:
            require_distribution(
                "bitsandbytes", extra="posttrain,qlora", capability="QLoRA"
            )

        cuda_available = torch.cuda.is_available()
        bf16_available = cuda_available and torch.cuda.is_bf16_supported()
        compute_dtype = (
            torch.bfloat16
            if bf16_available
            else torch.float16 if cuda_available else torch.float32
        )

        # Load model with optional 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=compute_dtype if not self.use_4bit else None,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=self.use_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ) if self.use_4bit else None,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )

        # Prepare model for k-bit training
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Initialize LoRA
        if adapter_path:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=self.adapter_trainable,
            )
        else:
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=resolved_targets,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # 打印可训练参数占比

        # Load tokenizer
        tokenizer_source = adapter_path if adapter_path and os.path.isdir(adapter_path) else model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        except (OSError, ValueError):
            if tokenizer_source == model_name:
                raise
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def fit(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            epochs: int = 3,
            batch_size: int = 2,
            learning_rate: float = 2e-4,  # LoRA通常需要更高学习率
            output_dir: str = "./lora_output",
            logging_steps: int = 10,
            save_strategy: str = "steps",
            save_steps: int = 500,
            gradient_accumulation_steps: int = 4,
            deepspeed: Optional[str] = None,
    ) -> None:
        """Run LoRA fine-tuning."""
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")

        # Tokenization
        def tokenize_fn(examples: Dict) -> Dict:
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

        # Training arguments
        cuda_available = torch.cuda.is_available()
        use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            output_dir=output_dir,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            bf16=use_bf16,
            fp16=cuda_available and not use_bf16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_8bit" if self.use_4bit else "adamw_torch",
            report_to="none",
            deepspeed=deepspeed,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Start training
        self.trainer.train()

    def predict(self, text: str, max_new_tokens: int = 100) -> str:
        """Generate text from input prompt."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: str) -> None:
        """Save only LoRA weights (轻量保存)."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, base_model_name: Optional[str] = None, **kwargs) -> "LoRATuner":
        """
        Load LoRA weights.
        :param base_model_name: 基础模型名称（如果未保存tokenizer）
        """
        resolved_base = base_model_name
        if not resolved_base:
            config_path = os.path.join(path, "adapter_config.json")
            if not os.path.isfile(config_path):
                raise ValueError(
                    "base_model_name is required when adapter_config.json is missing"
                )
            with open(config_path, "r", encoding="utf-8") as handle:
                resolved_base = json.load(handle).get("base_model_name_or_path")
            if not resolved_base:
                raise ValueError(
                    "adapter_config.json does not declare base_model_name_or_path"
                )
        return cls(
            model_name=resolved_base,
            adapter_path=path,
            **kwargs,
        )

if __name__ == "__main__":
    from datasets import load_dataset

    # 1. 初始化（QLoRA模式）
    lora = LoRATuner(
        model_name="meta-llama/Meta-Llama-3-8B",
        use_4bit=True,  # 启用QLoRA
        lora_rank=64,  # 更大的rank可能需要更多显存
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 覆盖更多模块
    )

    # 2. 加载数据（示例使用Alpaca格式）
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    dataset = dataset.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}"})

    # 3. 训练（单卡RTX 3090 24GB可运行）
    lora.fit(
        train_dataset=dataset,
        epochs=1,
        batch_size=2,  # 根据显存调整
        learning_rate=1e-4,
        output_dir="./llama3_8b_lora",
    )

    # 4. 推理
    print(lora.predict("How to cook pasta?"))

    # 5. 保存LoRA权重（仅几MB）
    lora.save("./my_llama3_lora")

    # 6. 加载保存的LoRA
    new_lora = LoRATuner.load(
        path="./my_llama3_lora",
        base_model_name="meta-llama/Meta-Llama-3-8B",
    )
