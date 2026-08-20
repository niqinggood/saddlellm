import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    PromptTuningConfig,
    PrefixTuningConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
from typing import Dict, Optional, Literal


class PromptTuner:
    """
    Prompt Tuning Wrapper with sklearn-style API.
    Supports both Soft Prompt and Prefix Tuning methods.

    Example:
    >>> tuner = PromptTuner(
            model_name="meta-llama/Meta-Llama-3-8B",
            method="soft",
            num_virtual_tokens=20
        )
    >>> tuner.fit(train_dataset)
    >>> tuner.generate("What is AI?")
    """

    def __init__(
            self,
            model_name: str = "meta-llama/Meta-Llama-3-8B",
            method: Literal["soft", "prefix"] = "soft",
            num_virtual_tokens: int = 20,
            max_length: int = 1024,
            device_map: str = "auto",
            chat_template: str = "default",  # "default", "alpaca", "chatml"
    ):
        """
        Initialize Prompt Tuner.

        :param method: "soft" (PromptTuning) or "prefix" (PrefixTuning)
        :param num_virtual_tokens: Number of virtual tokens to add
        """
        self.model_name = model_name
        self.method = method
        self.num_virtual_tokens = num_virtual_tokens
        self.max_length = max_length

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_chat_template(chat_template)

        # Initialize prompt tuning method
        if method == "soft":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens,
                tokenizer_name_or_path=model_name,
            )
        else:  # prefix
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=num_virtual_tokens,
            )

        # Convert model to PEFT model
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def set_chat_template(self, template_type: str):
        """Set chat template (same as RLHF example)"""
        if template_type == "alpaca":
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}\n\n{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n\n{% endif %}{% endfor %}### Response:\n"
        elif template_type == "chatml":
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}<|im_start|>assistant\n"
        else:  # default
            self.tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"

    def fit(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            epochs: int = 3,
            batch_size: int = 2,
            learning_rate: float = 3e-2,  # Prompt tuning typically needs higher LR
            output_dir: str = "./prompt_tuning_output",
            logging_steps: int = 10,
    ):
        """Run prompt tuning training."""

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
            remove_unused_columns=False,
            report_to="none",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Start training
        trainer.train()

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text with learned prompts."""
        inputs = self.tokenizer(
            prompt,
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

    def save(self, path: str):
        """Save only prompt embeddings (tiny files)."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load trained prompt tuner."""
        instance = cls(model_name=path, **kwargs)
        return instance

if __name__ == '__main__':
    from datasets import load_dataset

    # 初始化Soft Prompt Tuning
    tuner = PromptTuner(
        model_name="meta-llama/Meta-Llama-3-8B",
        method="soft",
        num_virtual_tokens=20,  # 添加20个虚拟token
    )

    # 加载数据（示例使用Alpaca格式）
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:100]")
    dataset = dataset.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}"})

    # 训练（仅更新prompt参数，基础模型冻结）
    tuner.fit(
        train_dataset=dataset,
        epochs=3,
        batch_size=2,
        learning_rate=0.03,  # Soft prompt需要较大学习率
    )

    # 生成测试
    print(tuner.generate("How to make coffee?"))

    # 保存（仅保存prompt参数，通常<1MB）
    tuner.save("./llama3_soft_prompt")