import torch
import random
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def UnslotSFTTrainer(
    MODEL_PATH, DATASET_PATH, OUTPUT_PATH, max_seq_length=2048, random_state=3407
):
    import os

    os.environ["TORCHDYNAMO_DISABLE"] = "1"  # 禁用torch.compile
    os.environ["PYTORCH_TRITON_DISABLE"] = "1"  # 禁用Triton编译器

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    torch.backends.cudnn.enabled = False  # 禁用 cuDNN 自动优化
    torch.backends.cuda.enable_flash_sdp(False)  # 禁用 Flash Attention
    torch.backends.cuda.enable_mem_efficient_sdp(False)  # 禁用内存优化版 Attention

    dtype = None
    load_in_4bit = True
    from transformers import TrainerCallback
    import gc

    class SaveEveryNStepsCallback(TrainerCallback):
        def __init__(self, save_steps=40, output_dir=OUTPUT_PATH):
            self.save_steps = save_steps
            self.output_dir = output_dir

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step % self.save_steps == 0:
                print(f"\n*** Saving model at step {state.global_step} ***")
                output_dir = f"{self.output_dir}_step_{state.global_step}"
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f"Model saved to {output_dir}")

                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()

            # 每10步检查一次GPU内存
            if state.global_step % 10 == 0:
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**3
                    mem_cached = torch.cuda.memory_reserved() / 1024**3
                    print(
                        f"GPU Memory: Allocated {mem_allocated:.2f} GB, Cached {mem_cached:.2f} GB"
                    )

    # 加载模型和tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=random_state,
        use_rslora=False,
        loftq_config=None,
    )

    alpaca_prompt = """### Question:
        {}

        ### Response:
        {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]
        texts = []
        for instruction, output in zip(instructions, outputs):
            text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    # 从本地加载数据集
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"基本信息#####################：\n{dataset}")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 修改TrainingArguments，禁用复杂编译
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=200,
        learning_rate=2e-4,
        # fp16=not is_bfloat16_supported(),
        # bf16=is_bfloat16_supported(),
        fp16=False,  # 强制禁用
        bf16=False,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=random_state,
        output_dir="outputs",
        report_to="none",
        torch_compile=False,  # 禁用torch.compile
        dataloader_num_workers=0,
    )
    # 创建自定义回调实例
    save_callback = SaveEveryNStepsCallback(save_steps=40, output_dir=OUTPUT_PATH)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
        callbacks=[save_callback],  # 添加自定义回调
    )

    print("***" * 10, "begin train")
    try:
        trainer.train()
        print("***" * 10, "after train")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保在异常发生时也保存模型
        print("Saving final model...")
        model.save_pretrained(OUTPUT_PATH)
        tokenizer.save_pretrained(OUTPUT_PATH)
        # 清理显存
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
    DATASET_PATH = r"E:\ml_data\medical_zh\train_zh_0.jsonl"
    OUTPUT_PATH = r"E:\reactflow_test\backend\model\lora_model_2424"
    # transfomer_load_lora(    MODEL_PATH=r'E:\reactflow_test\backend\model\lora_model_2423_step_120')
    # unsloth_load_lora_model( MODEL_PATH = r'E:\reactflow_test\backend\model\lora_model_2423_step_120' )
    # src_mode_transformer_load()
    UnslotSFTTrainer(
        MODEL_PATH=r"E:\reactflow_test\backend\model\lora_model_2423_step_120",
        DATASET_PATH=DATASET_PATH,
        OUTPUT_PATH=OUTPUT_PATH,
        random_state=random.randint(2000, 3000),
    )

# import torch
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling,
#     BitsAndBytesConfig
# )
# from datasets import Dataset
# from typing import Dict, Optional, List
# from peft import LoraConfig, prepare_model_for_kbit_training  # 可选：用于LoRA
#
#
# class SFTTrainer:
#     """
#     Supervised Fine-Tuning (SFT) for LLMs with sklearn-style API.
#     Supports both full fine-tuning and LoRA.
#
#     Example:
#     >>> sft = SFTTrainer(model_name="Llama-3-8B")
#     >>> sft.fit(train_dataset)
#     >>> print(sft.predict("What is AI?"))
#     """
#
#     def __init__(
#             self,
#             model_name: str = "meta-llama/Meta-Llama-3-8B",
#             max_length: int = 2048,
#             use_lora: bool = True,  # 是否使用LoRA
#             lora_rank: int = 64,  # LoRA秩
#             use_4bit: bool = True,  # 是否使用QLoRA
#             device_map: str = "auto",
#             chat_template: str = "default",  # 支持"default", "alpaca", "chatml"
#     ):
#         """
#         Initialize SFT trainer.
#
#         :param model_name: Hugging Face模型ID或本地路径
#         :param use_lora: 使用LoRA适配器（否则全量微调）
#         :param use_4bit: 使用4-bit量化（QLoRA）
#         :param chat_template: 对话模板格式
#         """
#         self.model_name = model_name
#         self.max_length = max_length
#         self.use_lora = use_lora
#
#         # 加载模型（可选4-bit量化）
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=use_4bit,
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#         ) if use_4bit else None
#
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             quantization_config=quantization_config,
#             device_map=device_map,
#             torch_dtype=torch.bfloat16 if not use_4bit else None,
#             trust_remote_code=True,
#         )
#
#         # 准备4-bit训练
#         if use_4bit:
#             self.model = prepare_model_for_kbit_training(self.model)
#
#         # 初始化LoRA（如果启用）
#         if use_lora:
#             peft_config = LoraConfig(
#                 r=lora_rank,
#                 lora_alpha=2 * lora_rank,
#                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#                 lora_dropout=0.05,
#                 bias="none",
#                 task_type="CAUSAL_LM",
#             )
#             self.model = get_peft_model(self.model, peft_config)
#             self.model.print_trainable_parameters()  # 打印可训练参数
#
#         # 加载tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         # 设置对话模板
#         self.set_chat_template(chat_template)
#
#     def set_chat_template(self, template_type: str):
#         """设置对话模板"""
#         if template_type == "alpaca":
#             self.tokenizer.chat_template = (
#                 "{% if messages[0]['role'] == 'system' %}"
#                 "{{ messages[0]['content'] }}\n\n"
#                 "{% endif %}"
#                 "{% for message in messages %}"
#                 "{% if message['role'] == 'user' %}"
#                 "### Instruction:\n{{ message['content'] }}\n\n"
#                 "{% elif message['role'] == 'assistant' %}"
#                 "### Response:\n{{ message['content'] }}\n\n"
#                 "{% endif %}"
#                 "{% endfor %}"
#                 "### Response:\n"
#             )
#         elif template_type == "chatml":
#             self.tokenizer.chat_template = (
#                 "{% for message in messages %}"
#                 "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
#                 "{% endfor %}"
#                 "<|im_start|>assistant\n"
#             )
#         else:  # default
#             self.tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
#
#     def fit(
#             self,
#             train_dataset: Dataset,
#             eval_dataset: Optional[Dataset] = None,
#             epochs: int = 3,
#             batch_size: int = 2,
#             learning_rate: float = 2e-5,
#             output_dir: str = "./sft_output",
#             logging_steps: int = 50,
#             save_strategy: str = "steps",
#             save_steps: int = 500,
#             gradient_accumulation_steps: int = 4,
#             deepspeed: Optional[str] = None,
#     ):
#         """
#         Run SFT training.
#
#         :param train_dataset: 数据集需包含"text"或"messages"列
#         :param deepspeed: DeepSpeed配置文件路径（多卡训练）
#         """
#
#         # 数据预处理
#         def tokenize_fn(examples: Dict) -> Dict:
#             if "messages" in examples:  # 对话格式
#                 text = self.tokenizer.apply_chat_template(
#                     examples["messages"],
#                     tokenize=False,
#                     add_generation_prompt=False,
#                 )
#             else:  # 普通文本
#                 text = examples["text"]
#
#             return self.tokenizer(
#                 text,
#                 truncation=True,
#                 max_length=self.max_length,
#                 padding="max_length",
#             )
#
#         train_dataset = train_dataset.map(tokenize_fn, batched=True)
#         if eval_dataset is not None:
#             eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
#
#         # 训练参数
#         training_args = TrainingArguments(
#             per_device_train_batch_size=batch_size,
#             per_device_eval_batch_size=batch_size,
#             num_train_epochs=epochs,
#             learning_rate=learning_rate,
#             output_dir=output_dir,
#             logging_steps=logging_steps,
#             save_strategy=save_strategy,
#             save_steps=save_steps,
#             bf16=torch.cuda.is_bf16_supported(),
#             fp16=not torch.cuda.is_bf16_supported(),
#             gradient_accumulation_steps=gradient_accumulation_steps,
#             optim="adamw_torch",
#             report_to="none",
#             deepspeed=deepspeed,
#             remove_unused_columns=False,  # 保留原始数据列
#         )
#
#         # 数据收集器
#         data_collator = DataCollatorForLanguageModeling(
#             tokenizer=self.tokenizer,
#             mlm=False,
#         )
#
#         # 初始化Trainer
#         self.trainer = Trainer(
#             model=self.model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             data_collator=data_collator,
#         )
#
#         # 开始训练
#         self.trainer.train()
#
#     def predict(self, text: str, max_new_tokens: int = 100) -> str:
#         """生成文本"""
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=self.max_length,
#         ).to(self.model.device)
#
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             pad_token_id=self.tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=0.7,
#         )
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     def save(self, path: str):
#         """保存模型"""
#         if self.use_lora:
#             self.model.save_pretrained(path)  # 仅保存LoRA权重
#         else:
#             self.model.save_pretrained(path)
#         self.tokenizer.save_pretrained(path)
#
#     @classmethod
#     def load(cls, path: str, **kwargs):
#         """加载模型"""
#         instance = cls(model_name=path, **kwargs)
#         return instance
#
# def base_usage( ):
#     from datasets import load_dataset
#
#     # 初始化（使用QLoRA+LoRA）
#     sft = SFTTrainer(
#         model_name="meta-llama/Meta-Llama-3-8B",
#         use_lora=True,
#         use_4bit=True,
#         max_length=1024,
#     )
#
#     # 加载数据（Alpaca格式示例）
#     dataset  = load_dataset("tatsu-lab/alpaca", split="train[:100]")
#     dataset  = dataset.map(lambda x: {
#         "messages": [
#             {"role": "user", "content": x["instruction"] + ("\n" + x["input"] if x["input"] else "")},
#             {"role": "assistant", "content": x["output"]}
#         ]
#     })
#
#     # 训练
#     sft.fit(
#         train_dataset=dataset,
#         epochs=1,
#         batch_size=2,
#         learning_rate=2e-5,
#         output_dir="./llama3_sft",
#     )
#
#     # 推理
#     print(sft.predict("How to make a cake?"))
#
#     # 保存
#     sft.save("./my_llama3_sft")
#
# def mult_dialog():
#     # 多轮对话数据示例
#     chat_dataset = Dataset.from_dict({
#         "messages": [
#             [
#                 {"role": "user", "content": "What's the capital of France?"},
#                 {"role": "assistant", "content": "The capital is Paris."},
#                 {"role": "user", "content": "What's its population?"},
#                 {"role": "assistant", "content": "Around 2.2 million people."}
#             ]
#         ]
#     })
#
#     # 使用ChatML模板
#     sft = SFTTrainer(
#         model_name="Qwen-7B",
#         chat_template="chatml",
#     )
#     sft.fit(train_dataset=chat_dataset)
#
# if __name__ == "__main__":
#     mult_dialog()
