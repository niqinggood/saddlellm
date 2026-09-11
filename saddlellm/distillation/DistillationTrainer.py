import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import Dict, Optional, Union
from torch.nn import KLDivLoss
from torch.optim import AdamW

class DistillationTrainer:
    """
    Knowledge Distillation for LLMs with sklearn-style API.
    Supports both logits-based and hidden-states distillation.

    Example:
    >>> distiller = DistillationTrainer(
            teacher="meta-llama/Llama-3-70B",
            student="TinyLlama/TinyLlama-1.1B"
        )
    >>> distiller.fit(train_dataset)
    >>> distiller.predict("What is AI?")
    """
    def __init__(
        self,
        teacher: str,
        student: str,
        temperature: float = 2.0,  # 蒸馏温度
        alpha_ce: float = 0.5,     # 蒸馏损失权重
        alpha_hidden: float = 0.3, # 隐藏层匹配损失权重
        max_length: int = 1024,
        device_map: str = "auto",
    ):
        """
        Initialize distillation trainer.
        
        :param teacher: 教师模型HF ID或路径
        :param student: 学生模型HF ID或路径
        :param temperature: 软化logits的温度参数
        :param alpha_ce: 交叉熵损失权重（学生vs真实标签）
        :param alpha_hidden: 隐藏层匹配损失权重
        """
        # 加载教师模型（冻结参数）
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 加载学生模型
        self.student = AutoModelForCausalLM.from_pretrained(
            student,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

        # 加载tokenizer（假设师生模型使用相同tokenizer）
        self.tokenizer = AutoTokenizer.from_pretrained(student)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 蒸馏参数
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_hidden = alpha_hidden
        self.max_length = max_length

        # 损失函数
        self.kl_loss = KLDivLoss(reduction="batchmean")
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def compute_distillation_loss(self, student_outputs, teacher_outputs, labels):
        """
        计算蒸馏损失（logits + 可选隐藏层匹配）
        """
        # Logits蒸馏（软化后KL散度）
        student_logits = student_outputs.logits / self.temperature
        teacher_logits = teacher_outputs.logits / self.temperature
        loss_kl = self.kl_loss(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1),
        ) * (self.temperature ** 2)  # 温度缩放补偿

        # 学生vs真实标签的交叉熵
        shift_logits = student_outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # 隐藏层匹配损失（可选）
        loss_hidden = 0
        if self.alpha_hidden > 0:
            # 取最后一层隐藏状态（可扩展为多层）
            s_hidden = student_outputs.hidden_states[-1]
            t_hidden = teacher_outputs.hidden_states[-1].detach()
            loss_hidden = torch.nn.functional.mse_loss(s_hidden, t_hidden)

        # 加权总损失
        total_loss = (
            (1 - self.alpha_ce - self.alpha_hidden) * loss_ce +
            self.alpha_ce * loss_kl +
            self.alpha_hidden * loss_hidden
        )
        return total_loss

    def fit(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        output_dir: str = "./distill_output",
        logging_steps: int = 10,
    ):
        """Run distillation training."""
        # 数据预处理
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

        # 自定义训练循环（因需同时计算师生模型输出）
        optimizer = AdamW(self.student.parameters(), lr=learning_rate)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataset):
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.student.device)
                labels = inputs["input_ids"].clone()

                # 教师模型预测
                with torch.no_grad():
                    teacher_outputs = self.teacher(**inputs, output_hidden_states=True)

                # 学生模型预测
                student_outputs = self.student(**inputs, output_hidden_states=True)

                # 计算蒸馏损失
                loss = self.compute_distillation_loss(
                    student_outputs, teacher_outputs, labels
                )
                
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                if step % logging_steps == 0:
                    print(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}")

            print(f"Epoch {epoch} Average Loss: {total_loss / len(train_dataset):.4f}")

        # 保存学生模型
        self.student.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, text: str, max_new_tokens: int = 100) -> str:
        """使用学生模型生成文本"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.student.device)
        
        outputs = self.student.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: str):
        """保存学生模型"""
        self.student.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **kwargs):
        """加载蒸馏后的学生模型"""
        instance = cls(teacher="dummy", student=path, **kwargs)  # 教师模型不再需要
        return instance
