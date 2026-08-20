import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
from trl import (
    PPOTrainer,
    DPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)
from datasets import Dataset
from typing import Dict, List, Optional, Literal
from peft import LoraConfig, get_peft_model


class RLHFTrainer:
    """
    RLHF Trainer with sklearn-style API.
    Supports both PPO and DPO methods.

    Example:
    >>> rlhf = RLHFTrainer(
            method="dpo",
            model_name="meta-llama/Meta-Llama-3-8B"
        )
    >>> rlhf.fit(preference_dataset)
    >>> rlhf.generate("What is RLHF?")
    """

    def __init__(
            self,
            method: Literal["ppo", "dpo"] = "dpo",
            model_name: str = "meta-llama/Meta-Llama-3-8B",
            reward_model: Optional[str] = None,  # PPO专用
            max_length: int = 1024,
            use_lora: bool = True,
            lora_rank: int = 64,
            device_map: str = "auto",
            chat_template: str = "default",  # "default", "alpaca", "chatml"
    ):
        """
        Initialize RLHF trainer.

        :param method: "ppo" 或 "dpo"
        :param reward_model: PPO使用的奖励模型路径（None时使用默认）
        :param use_lora: 是否使用LoRA适配器
        """
        self.method = method
        self.model_name = model_name
        self.max_length = max_length

        # 加载基础模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.set_chat_template(chat_template)

        # 初始化不同RLHF方法
        if method == "ppo":
            # PPO需要带Value Head的模型
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            self.ref_model = create_reference_model(self.model)

            # 加载奖励模型
            self.reward_model = pipeline(
                "text-classification",
                model=reward_model or "OpenAssistant/reward-model-deberta-v3-large",
                device=device_map,
            )
        else:  # DPO
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )

        # 可选：添加LoRA适配器
        if use_lora:
            peft_config = LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["lm_head"],
            )
            self.model = get_peft_model(self.model, peft_config)
            if method == "dpo":
                self.ref_model = get_peft_model(self.ref_model, peft_config)

    def set_chat_template(self, template_type: str):
        """设置对话模板（同SFT示例）"""
        if template_type == "alpaca":
            self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}\n\n{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}### Instruction:\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}### Response:\n{{ message['content'] }}\n\n{% endif %}{% endfor %}### Response:\n"
        elif template_type == "chatml":
            self.tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}<|im_start|>assistant\n"
        else:  # default
            self.tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"

    def fit(
            self,
            dataset: Dataset,
            epochs: int = 3,
            batch_size: int = 2,
            learning_rate: float = 1e-5,
            output_dir: str = "./rlhf_output",
            beta: float = 0.1,  # DPO的温度参数
            kl_penalty: float = 0.2,  # PPO的KL惩罚系数
    ):
        """
        Run RLHF training.

        :param dataset: 需包含:
            - PPO: "query"字段 + 人工/奖励模型生成的response
            - DPO: "prompt", "chosen", "rejected"字段
        """

        # 数据预处理
        def tokenize_fn(examples: Dict) -> Dict:
            if self.method == "ppo":
                return self.tokenizer(examples["query"], truncation=True, max_length=self.max_length)
            else:  # DPO
                return {
                    "prompt": self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": examples["prompt"]}],
                        tokenize=False
                    ),
                    "chosen": examples["chosen"],
                    "rejected": examples["rejected"],
                }

        dataset = dataset.map(tokenize_fn, batched=True)

        # 训练参数
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            output_dir=output_dir,
            remove_unused_columns=False,
            report_to="none",
            bf16=torch.cuda.is_bf16_supported(),
        )

        # 初始化Trainer
        if self.method == "ppo":
            self.trainer = PPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=self.tokenizer,
                args=training_args,
            )

            # PPO训练循环（简化版）
            for epoch in range(epochs):
                for batch in dataset:
                    query_tensors = self.tokenizer(
                        batch["query"], return_tensors="pt", padding=True
                    ).input_ids.to(self.model.device)

                    # 生成响应
                    response_tensors = self.trainer.generate(
                        query_tensors,
                        max_length=self.max_length,
                    )

                    # 计算奖励（实际应用应替换为人工标注或更复杂的奖励模型）
                    rewards = [
                        torch.tensor(self.reward_model(
                            self.tokenizer.decode(r, skip_special_tokens=True)
                        )[0]["score"])
                                     for r in response_tensors
                    ]

                    # PPO更新
                    stats = self.trainer.step(
                        query_tensors,
                        response_tensors,
                        rewards,
                    )
                    print(f"Epoch {epoch}, Reward: {torch.mean(torch.stack(rewards)):.4f}")
        else:  # DPO
            self.trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                beta=beta,
                train_dataset=dataset,
                tokenizer=self.tokenizer,
            )
            self.trainer.train()

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """生成文本"""
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
        """保存模型（含适配器）"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """加载训练后的模型"""
        instance = cls(model_name=path, **kwargs)
        return instance