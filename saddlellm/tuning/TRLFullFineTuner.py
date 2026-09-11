import logging
import torch
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
import argparse
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)




def generate_response(model, tokenizer, prompt, device="cuda", max_length=512):
    """生成模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class ValidationCallback(TrainerCallback):
    """自定义回调函数用于验证"""

    def __init__(self, model, tokenizer, test_prompts, eval_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n=== 第 {state.global_step} 步模型验证 ===")
            self.model.eval()
            with torch.no_grad():
                for prompt in self.test_prompts:
                    full_prompt = f"### Question: {prompt}\n### Answer:"
                    device = next(self.model.parameters()).device
                    response = generate_response(self.model, self.tokenizer, full_prompt, device)
                    print(f"Prompt: {prompt}")
                    print(f"Response: {response[len(full_prompt):]}\n")
            self.model.train()
        return control


def train_model(
        model_path: str ,
        dataset_path: str ,
        output_path: str ,
        learning_rate: float = 2e-5,
        batch_size: int = 2,
        num_epochs: int = 3,
        save_steps: int = 100,
        max_seq_length: int = 512,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        eval_steps: int = 50,
        test_prompts: list = None,
        use_bf16: bool = True,
        use_gradient_checkpointing: bool = True,
        use_quantization: bool = False
):
    """全参数微调主函数"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if test_prompts is None:
        test_prompts = [
            "什么是糖尿病？",
            "如何预防感冒？",
            "心脏病的常见症状有哪些？",
            "解释一下高血压的病因和治疗方法",
            "什么是冠心病？有哪些预防措施？"
        ]

    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_path)
    logging.info(f'Dataset info:{dataset}')

    if 'train' in dataset:
        dataset = dataset['train']
    else:
        raise ValueError("Dataset format incorrect, missing 'train' split")

    # 量化配置
    bnb_config = None
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        quantization_config=bnb_config if use_quantization else None,
        trust_remote_code=True
    )

    logging.info(f"模型数据类型: {next(model.parameters()).dtype}")

    # 数据格式化函数（包含长度控制）
    def formatting_prompts_func(example):
        texts = []
        for i in range(len(example['instruction'])):
            instruction = example['instruction'][i]
            output = example['output'][i] if 'output' in example else ""
            text = f"### Question: {instruction}\n### Answer: {output}"
            # 编码并截断
            input_ids = tokenizer.encode(text)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                text = tokenizer.decode(input_ids)
            texts.append(text)
        return {'text': texts}

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 数据整理器
    response_template = "### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=False,
        bf16=use_bf16,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=save_steps,
        save_strategy="steps",
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        dataloader_num_workers=4,
        gradient_checkpointing=use_gradient_checkpointing,
        report_to="none",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0
    )

    # 格式化数据集
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )

    # 初始化Trainer（移除了所有不支持的参数）
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer
    )

    # 添加验证回调
    trainer.add_callback(
        ValidationCallback(model, tokenizer, test_prompts, eval_steps)
    )

    # 训练前验证
    logging.info("\n=== 训练前模型验证 ===")
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            full_prompt = f"### Question: {prompt}\n### Answer:"
            response = generate_response(model, tokenizer, full_prompt, device)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Response: {response[len(full_prompt):]}\n")
    model.train()

    # 开始训练
    trainer.train()

    # 最终验证
    print("\n=== 训练完成，最终验证 ===")
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            full_prompt = f"### Question: {prompt}\n### Answer:"
            response = generate_response(model, tokenizer, full_prompt, device)
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Response: {response[len(full_prompt):]}\n")

    # 保存模型
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    logging.info(f"Model saved to: {output_path}")


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="Full Fine-Tuning Script")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="预训练模型路径")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="训练数据集路径")
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="模型输出路径")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=2, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数间隔")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--eval_steps", type=int, default=50, help="验证步数间隔")
    parser.add_argument("--use_bf16", action="store_true", help="使用bfloat16精度")
    parser.add_argument("--use_quantization", action="store_true", help="使用4位量化")
    return parser.parse_args()


if __name__ == "__main__":
    # 直接设置的路径变量
    MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
    DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
    OUTPUT_PATH = r'E:\reactflow_test\backend\model\full_finetuned_model'
    test_prompts = [
        "什么是糖尿病？",
        "如何预防感冒？",
        "心脏病的常见症状有哪些？",
        "解释一下高血压的病因和治疗方法",
        "什么是冠心病？有哪些预防措施？"
    ]
    args = parse_args()
    train_model(
        model_path=MODEL_PATH,
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        use_bf16=args.use_bf16,
        use_quantization=args.use_quantization,
        test_prompts = test_prompts

    )

# import logging
# import torch
# from datasets import load_dataset
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     BitsAndBytesConfig,
#     TrainerCallback
# )
# import argparse
# from tqdm import tqdm
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
# # 直接设置的路径变量
# MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
# DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
# OUTPUT_PATH = r'E:\reactflow_test\backend\model\full_finetuned_model'
#
#
# def generate_response(model, tokenizer, prompt, device="cuda", max_length=512):
#     """生成模型响应"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     # 确保输入数据与模型数据类型一致
#     inputs = {k: v.to(dtype=torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
#
#     with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 使用自动混合精度
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#
# class ValidationCallback(TrainerCallback):
#     """自定义回调函数用于验证"""
#
#     def __init__(self, model, tokenizer, test_prompts, eval_steps):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.test_prompts = test_prompts
#         self.eval_steps = eval_steps
#
#     def on_step_end(self, args, state, control, **kwargs):
#         if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
#             print(f"\n=== 第 {state.global_step} 步模型验证 ===")
#             self.model.eval()
#             with torch.no_grad():
#                 for prompt in self.test_prompts:
#                     full_prompt = f"### Question: {prompt}\n### Answer:"
#                     device = next(self.model.parameters()).device
#                     response = generate_response(self.model, self.tokenizer, full_prompt, device)
#                     print(f"Prompt: {prompt}")
#                     print(f"Response: {response[len(full_prompt):]}\n")
#             self.model.train()
#         return control
#
#
# def train_model(
#         model_path: str = MODEL_PATH,
#         dataset_path: str = DATASET_PATH,
#         output_path: str = OUTPUT_PATH,
#         learning_rate: float = 2e-5,
#         batch_size: int = 2,  # 全参数微调可能需要较小的batch size
#         num_epochs: int = 3,
#         save_steps: int = 100,
#         max_seq_length: int = 512,
#         gradient_accumulation_steps: int = 4,
#         warmup_steps: int = 100,
#         eval_steps: int = 50,
#         test_prompts: list = None,
#         use_bf16: bool = True,
#         use_gradient_checkpointing: bool = True,
#         use_quantization: bool = False
# ):
#     """全参数微调主函数"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logging.info(f"Using device: {device}")
#
#     # 默认测试提示词
#     if test_prompts is None:
#         test_prompts = [
#             "什么是糖尿病？",
#             "如何预防感冒？",
#             "心脏病的常见症状有哪些？",
#             "解释一下高血压的病因和治疗方法",
#             "什么是冠心病？有哪些预防措施？"
#         ]
#
#     # 加载数据集
#     dataset = load_dataset("json", data_files=dataset_path)
#     logging.info(f'Dataset info:{dataset}')
#
#     if 'train' in dataset:
#         dataset = dataset['train']
#     else:
#         raise ValueError("Dataset format incorrect, missing 'train' split")
#
#     # 量化配置
#     bnb_config = None
#     if use_quantization:
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4',
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#
#     # 加载模型
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
#         quantization_config=bnb_config if use_quantization else None,
#         trust_remote_code=True
#     )
#
#     # 打印模型数据类型
#     logging.info(f"模型数据类型: {next(model.parameters()).dtype}")
#
#     # 数据格式化函数
#     def formatting_prompts_func(example):
#         texts = []
#         for i in range(len(example['instruction'])):
#             instruction = example['instruction'][i]
#             output = example['output'][i] if 'output' in example else ""
#             text = f"### Question: {instruction}\n### Answer: {output}"
#             texts.append(text)
#         return {'text': texts}
#
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#
#     # 数据整理器（确保只计算Answer部分的损失）
#     response_template = "### Answer:"
#     collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
#
#     # 训练参数配置
#     training_args = TrainingArguments(
#         output_dir=output_path,
#         per_device_train_batch_size=batch_size,
#         num_train_epochs=num_epochs,
#         learning_rate=learning_rate,
#         fp16=False,
#         bf16=use_bf16,
#         optim="adamw_torch_fused",
#         logging_steps=10,
#         save_steps=save_steps,
#         save_strategy="steps",
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         warmup_steps=warmup_steps,
#         dataloader_num_workers=4,
#         gradient_checkpointing=use_gradient_checkpointing,
#         report_to="none",
#         lr_scheduler_type="cosine",
#         weight_decay=0.01,
#         max_grad_norm=1.0,
#         max_length=max_seq_length  # 最大长度在这里设置
#     )
#
#     # 格式化数据集
#     formatted_dataset = dataset.map(
#         formatting_prompts_func,
#         batched=True,
#         remove_columns=dataset.column_names
#     )
#
#     # 初始化Trainer
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=formatted_dataset,
#         args=training_args,
#         data_collator=collator,
#         tokenizer=tokenizer,
#         packing=True  # 添加这个参数可以优化长序列处理
#     )
#
#     # 添加验证回调
#     trainer.add_callback(
#         ValidationCallback(model, tokenizer, test_prompts, eval_steps)
#     )
#
#     # 训练前验证
#     logging.info("\n=== 训练前模型验证 ===")
#     model.eval()
#     with torch.no_grad():
#         for prompt in test_prompts:
#             full_prompt = f"### Question: {prompt}\n### Answer:"
#             response = generate_response(model, tokenizer, full_prompt, device)
#             logging.info(f"Prompt: {prompt}")
#             logging.info(f"Response: {response[len(full_prompt):]}\n")
#     model.train()
#
#     # 开始训练
#     trainer.train()
#
#     # 最终验证
#     print("\n=== 训练完成，最终验证 ===")
#     model.eval()
#     with torch.no_grad():
#         for prompt in test_prompts:
#             full_prompt = f"### Question: {prompt}\n### Answer:"
#             response = generate_response(model, tokenizer, full_prompt, device)
#             logging.info(f"Prompt: {prompt}")
#             logging.info(f"Response: {response[len(full_prompt):]}\n")
#
#     # 保存模型
#     trainer.save_model(output_path)
#     tokenizer.save_pretrained(output_path)
#     logging.info(f"Model saved to: {output_path}")
#
#
# def parse_args():
#     """命令行参数解析"""
#     parser = argparse.ArgumentParser(description="Full Fine-Tuning Script")
#     parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="预训练模型路径")
#     parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="训练数据集路径")
#     parser.add_argument("--output_path", type=str, default=OUTPUT_PATH, help="模型输出路径")
#     parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
#     parser.add_argument("--batch_size", type=int, default=2, help="批量大小")
#     parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
#     parser.add_argument("--save_steps", type=int, default=100, help="保存步数间隔")
#     parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
#     parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
#     parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
#     parser.add_argument("--eval_steps", type=int, default=50, help="验证步数间隔")
#     parser.add_argument("--use_bf16", action="store_true", help="使用bfloat16精度")
#     parser.add_argument("--use_quantization", action="store_true", help="使用4位量化")
#     return parser.parse_args()
#
#
# if __name__ == "__main__":
#     args = parse_args()
#     train_model(
#         model_path=args.model_path,
#         dataset_path=args.dataset_path,
#         output_path=args.output_path,
#         learning_rate=args.learning_rate,
#         batch_size=args.batch_size,
#         num_epochs=args.num_epochs,
#         save_steps=args.save_steps,
#         max_seq_length=args.max_seq_length,
#         gradient_accumulation_steps=args.gradient_accumulation,
#         warmup_steps=args.warmup_steps,
#         eval_steps=args.eval_steps,
#         use_bf16=args.use_bf16,
#         use_quantization=args.use_quantization
#     )


# import logging
# import torch
# from datasets import load_dataset
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     BitsAndBytesConfig,
#     TrainerCallback
# )
# import argparse
# from tqdm import tqdm
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
#
# def generate_response(model, tokenizer, prompt, device="cuda", max_length=512):
#     """生成模型响应"""
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_length=max_length,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=True
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
#
# class ValidationCallback(TrainerCallback):
#     """自定义回调函数用于验证"""
#     def __init__(self, model, tokenizer, test_prompts, eval_steps):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.test_prompts = test_prompts
#         self.eval_steps = eval_steps
#
#     def on_step_end(self, args, state, control, **kwargs):
#         if self.eval_steps > 0 and state.global_step % self.eval_steps == 0:
#             logging.info(f"\n=== Step {state.global_step} Validation ===")
#             self.model.eval()
#             for prompt in self.test_prompts:
#                 full_prompt = f"### Question: {prompt}\n### Answer:"
#                 response = generate_response(
#                     self.model, self.tokenizer, full_prompt, next(self.model.parameters()).device
#                 )
#                 logging.info(f"Prompt: {prompt}")
#                 logging.info(f"Response: {response[len(full_prompt):]}\n")
#             self.model.train()
#         return control
#
# def train_model(
#     model_path: str,
#     dataset_path: str,
#     output_path: str,
#     learning_rate: float = 2e-5,
#     batch_size: int = 8,
#     num_epochs: int = 3,
#     max_seq_length: int = 512,
#     gradient_accumulation_steps: int = 4,
#     warmup_steps: int = 100,
#     eval_steps: int = 50,
#     test_prompts: list = None,
#     use_bf16: bool = True,
#     use_gradient_checkpointing: bool = True
# ):
#     """全参数微调主函数"""
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     logging.info(f"Using device: {device}")
#
#     # 默认测试提示词
#     if test_prompts is None:
#         test_prompts = [
#             "什么是糖尿病？",
#             "如何预防感冒？",
#             "心脏病的常见症状有哪些？"
#         ]
#
#     # 加载数据集
#     dataset = load_dataset("json", data_files=dataset_path)
#     if 'train' not in dataset:
#         raise ValueError("Dataset must contain 'train' split")
#     dataset = dataset['train']
#
#     # 加载模型和分词器（全精度，非量化）
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         device_map="auto",
#         torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
#         trust_remote_code=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#
#     # 数据格式化函数
#     def format_example(example):
#         text = f"### Question: {example['instruction']}\n### Answer: {example.get('output', '')}"
#         return {"text": text}
#
#     # 训练参数配置
#     training_args = TrainingArguments(
#         output_dir=output_path,
#         per_device_train_batch_size=batch_size,
#         num_train_epochs=num_epochs,
#         learning_rate=learning_rate,
#         bf16=use_bf16,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         warmup_steps=warmup_steps,
#         logging_steps=10,
#         save_steps=eval_steps,
#         save_strategy="steps",
#         gradient_checkpointing=use_gradient_checkpointing,
#         report_to="none",
#         lr_scheduler_type="cosine",
#         weight_decay=0.01,
#         max_grad_norm=1.0,
#         optim="adamw_torch_fused",
#         dataloader_num_workers=4
#     )
#
#     # 数据整理器（确保只计算Answer部分的损失）
#     collator = DataCollatorForCompletionOnlyLM(
#         "### Answer:",
#         tokenizer=tokenizer
#     )
#
#     # 初始化Trainer
#     trainer = SFTTrainer(
#         model=model,
#         train_dataset=dataset.map(format_example),
#         args=training_args,
#         data_collator=collator,
#         tokenizer=tokenizer,
#         max_seq_length=max_seq_length
#     )
#
#     # 添加验证回调
#     trainer.add_callback(
#         ValidationCallback(model, tokenizer, test_prompts, eval_steps)
#     )
#
#     # 训练前验证
#     logging.info("\n=== Pre-Training Validation ===")
#     model.eval()
#     for prompt in test_prompts:
#         full_prompt = f"### Question: {prompt}\n### Answer:"
#         response = generate_response(model, tokenizer, full_prompt, device)
#         logging.info(f"Prompt: {prompt}")
#         logging.info(f"Response: {response[len(full_prompt):]}\n")
#     model.train()
#
#     # 开始训练
#     trainer.train()
#
#     # 保存模型
#     trainer.save_model(output_path)
#     tokenizer.save_pretrained(output_path)
#     logging.info(f"Model saved to: {output_path}")
#
# def parse_args():
#     """命令行参数解析"""
#     parser = argparse.ArgumentParser(description="Full Fine-Tuning Script")
#     parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
#     parser.add_argument("--dataset_path", type=str, required=True, help="训练数据集路径")
#     parser.add_argument("--output_path", type=str, required=True, help="模型输出路径")
#     parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
#     parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
#     parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
#     parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
#     parser.add_argument("--gradient_accumulation", type=int, default=4, help="梯度累积步数")
#     parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
#     parser.add_argument("--eval_steps", type=int, default=50, help="验证步数间隔")
#     return parser.parse_args()
#
# if __name__ == "__main__":
#     args = parse_args()
#     train_model(
#         model_path=args.model_path,
#         dataset_path=args.dataset_path,
#         output_path=args.output_path,
#         learning_rate=args.learning_rate,
#         batch_size=args.batch_size,
#         num_epochs=args.num_epochs,
#         max_seq_length=args.max_seq_length,
#         gradient_accumulation_steps=args.gradient_accumulation,
#         warmup_steps=args.warmup_steps,
#         eval_steps=args.eval_steps
#     )
