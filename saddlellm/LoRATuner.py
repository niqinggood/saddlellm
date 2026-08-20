import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from typing import Dict, Optional, List
from bitsandbytes import optim


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
            model_name: str = "meta-llama/Meta-Llama-3-8B",
            max_length: int = 1024,
            lora_rank: int = 8,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05,
            target_modules: List[str] = ["q_proj", "v_proj"],
            use_4bit: bool = True,  # QLoRA if True
            device_map: str = "auto",
    ):
        """
        Initialize LoRA fine-tuning.
        :param model_name: Hugging Face model ID
        :param use_4bit: Enable 4-bit quantization (QLoRA)
        :param target_modules: Modules to apply LoRA (e.g., ["q_proj", "v_proj"])
        """
        self.model_name = model_name
        self.max_length = max_length

        # Load model with optional 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=use_4bit,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if not use_4bit else None,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ) if use_4bit else None,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Initialize LoRA
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # 打印可训练参数占比

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            output_dir=output_dir,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_8bit",  # 优化器适配QLoRA
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
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: str) -> None:
        """Save only LoRA weights (轻量保存)."""
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path: str, base_model_name: Optional[str] = None, **kwargs) -> "LoRATuner":
        """
        Load LoRA weights.
        :param base_model_name: 基础模型名称（如果未保存tokenizer）
        """
        instance = cls(model_name=base_model_name or path, **kwargs)
        instance.model = PeftModel.from_pretrained(instance.model, path)
        return instance

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