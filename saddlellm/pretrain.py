import os
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GPTNeoXForCausalLM,
    LlamaForCausalLM
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from datasets import load_dataset
from argparse import ArgumentParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pretrained_model(model_path):
    """加载预训练模型"""
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )


def create_model_from_scratch(model_type):
    """从头创建模型"""
    if model_type == "llama-7b":
        config = LlamaConfig(
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            vocab_size=32000
        )
        return LlamaForCausalLM(config)
    elif model_type == "gpt-neox-20b":
        config = GPTNeoXConfig(
            hidden_size=6144,
            num_hidden_layers=44,
            num_attention_heads=64,
            vocab_size=50432
        )
        return GPTNeoXForCausalLM(config)
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def load_data(data_path, data_format):
    """加载训练数据"""
    if data_format == "text":
        return load_dataset("text", data_files=data_path)
    elif data_format == "jsonl":
        return load_dataset("json", data_files=data_path)
    else:
        raise ValueError(f"不支持的数据格式: {data_format}")


def train(args):
    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载或创建模型
    if args.pretrain_mode == "continue":
        logger.info(f"从 {args.model_path} 加载预训练模型")
        model = load_pretrained_model(args.model_path)
    else:
        logger.info(f"创建新模型: {args.model_config}")
        model = create_model_from_scratch(args.model_config)

    model.to(device)

    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path if args.tokenizer_path else args.model_path,
        trust_remote_code=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 加载数据
    logger.info(f"加载数据从 {args.train_data}")
    dataset = load_data(args.train_data, args.data_format)

    # 4. 数据预处理
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.data_process_workers,
        remove_columns=["text"]
    )

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size // args.gradient_accum,
        gradient_accumulation_steps=args.gradient_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        logging_steps=args.log_interval,
        save_steps=args.save_interval,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        gradient_checkpointing=True,
        report_to="none"
    )

    # 6. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 使用CLM（因果语言建模）
    )

    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    logger.info("***** 开始训练 *****")
    trainer.train()

    # 8. 保存最终模型
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # 训练模式
    parser.add_argument("--pretrain-mode", choices=["continue", "scratch"], required=True)

    # 模型配置
    parser.add_argument("--model-path", help="继续训练时的模型路径")
    parser.add_argument("--model-config", choices=["llama-7b", "gpt-neox-20b"], help="从头训练时的模型架构")
    parser.add_argument("--tokenizer-path", help="自定义分词器路径")

    # 数据配置
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--data-format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--data-process-workers", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    # 训练参数
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gradient-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")

    # 输出配置
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()

    # 验证参数
    if args.pretrain_mode == "continue" and not args.model_path:
        raise ValueError("继续训练必须指定 --model-path")
    if args.pretrain_mode == "scratch" and not args.model_config:
        raise ValueError("从头训练必须指定 --model-config")

    train(args)