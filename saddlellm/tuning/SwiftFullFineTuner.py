import torch
from swift import Swift, TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Dict, Optional, Union

class SwiftFullFineTuner:
    """
    A sklearn-style wrapper for full fine-tuning with Swift (Aliyun).
    Example:
    >>> ft = SwiftFullFineTuner(model_name="Qwen-7B")
    >>> ft.fit(train_dataset)
    >>> ft.predict("What is AI?")
    """
    def __init__(
        self,
        model_name: str = "Qwen-7B",
        max_length: int = 2048,
        use_flash_attn: bool = True,
        bf16: bool = True,
        device_map: str = "auto",
    ):
        """
        Initialize Swift for full fine-tuning.
        :param model_name: Model ID (e.g., "Qwen-7B", "Llama-3-8B")
        :param max_length: Max sequence length
        :param use_flash_attn: Enable FlashAttention-2 for speed
        :param bf16: Use bfloat16 precision (recommended for Ampere+ GPUs)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Load model (禁用量化以进行全量微调)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map=device_map,
            use_flash_attention_2=use_flash_attn,
            trust_remote_code=True  # Required for Qwen
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False if "qwen" in model_name.lower() else True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充token
        
        # Swift初始化（全量微调无需添加适配器）
        self.swift_model = Swift.prepare_model(self.model)

    def fit(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 2e-5,
        output_dir: str = "./output",
        logging_steps: int = 10,
        save_strategy: str = "steps",
        save_steps: int = 500,
        gradient_accumulation_steps: int = 4,
        deepspeed: Optional[str] = None,  # e.g., "configs/deepspeed_zero3.json"
    ) -> None:
        """
        Full fine-tuning with Swift.
        :param train_dataset: HF Dataset with "text" column
        :param deepspeed: Path to DeepSpeed config (for multi-GPU)
        """
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
            report_to="none",  # Disable WandB by default
            deepspeed=deepspeed,  # Enable DeepSpeed if provided
        )

        # Trainer
        self.trainer = Trainer(
            model=self.swift_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Start training
        self.trainer.train()

    def predict(self, text: str, max_new_tokens: int = 100) -> str:
        """Generate text from input prompt."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs) -> "SwiftFullFineTuner":
        """Load a saved model."""
        return cls(model_name=path, **kwargs)
import os
def load_imdb_dataset(dataset_path: str, sample_size: int = None):
    """加载IMDB数据集"""
    texts = []
    labels = []

    # 加载正面评价
    pos_path = os.path.join(dataset_path, "train", "pos")
    for filename in os.listdir(pos_path)[:sample_size]:
        with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(1)

    # 加载负面评价
    neg_path = os.path.join(dataset_path, "train", "neg")
    for filename in os.listdir(neg_path)[:sample_size]:
        with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
            labels.append(0)

    return Dataset.from_dict({"text": texts, "label": labels})
if __name__ == "__main__":
    from datasets import load_dataset

    # 1. 初始化（全量微调需禁用量化）
    ft = SwiftFullFineTuner(
        model_name="E:/ml_data/Llama-3___2-1B",
        max_length=1024,
        bf16=True,
    )

    # 2. 加载数据（示例使用IMDB，实际需替换为指令数据）
    # dataset = load_dataset("imdb", split="train[:100]")
    # dataset = dataset.rename_column("text", "text")  # 确保列名为"text"
    imdb_path = "E:/ml_data/aclImdb"  # 替换为你的实际路径
    dataset = load_imdb_dataset(imdb_path, sample_size=50)
    print(f"成功加载数据集，样本数: {len(dataset)}")

    # 3. 全量微调（单卡）
    ft.fit(
        train_dataset=dataset,
        epochs=1,
        batch_size=1,  # 根据显存调整（Qwen-7B全量微调需要约80GB显存）
        learning_rate=2e-5,
        output_dir="./qwen_7b_ft",
    )

    # 4. 多卡训练（需Deepspeed配置）
    # ft.fit(train_dataset, deepspeed="configs/deepspeed_zero3.json")

    # 5. 推理
    print(ft.predict("The future of AI is"))

    # 6. 保存模型
    ft.save("./my_qwen_7b_ft")
