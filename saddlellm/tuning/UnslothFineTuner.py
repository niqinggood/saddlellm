# -*- coding: utf-8 -*-
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer,DataCollatorForLanguageModeling
from datasets import Dataset
from typing import Optional, Union, Dict, Callable, List
import os


class UnslothFineTuner:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int = 1024,
            load_in_4bit: bool = False,
            use_gradient_checkpointing: bool = True,
            device_map: str = "auto",
    ):
        self.model_path = model_path
        self.max_seq_length = max_seq_length

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )

        for param in self.model.parameters():
            param.requires_grad = True

        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.trainer = None

    def fit(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            text_column: str = "text",
            label_column: Optional[str] = None,
            training_args: Optional[TrainingArguments] = None,
    ) -> None:
        def tokenize_function(examples: Dict) -> Dict:
            tokenized = self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            if label_column and label_column in examples:
                # 有标签的情况（如分类或生成任务）
                tokenized_labels = self.tokenizer(
                    examples[label_column],
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                tokenized["labels"] = tokenized_labels["input_ids"]
            else:
                # 无标签的情况（语言模型训练）
                # 语言模型训练需要将input_ids复制到labels
                tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )

        if training_args is None:
            training_args = TrainingArguments(
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_train_epochs=3,
                learning_rate=5e-5,
                fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                logging_steps=10,
                save_steps=500,
                output_dir="./output",
                optim="adamw_torch",
                eval_strategy="steps" if eval_dataset else "no",
                save_total_limit=2,
                gradient_accumulation_steps=4,
                report_to="none",
                weight_decay=0.01,
                warmup_ratio=0.1,
            )
            # 创建语言模型训练的数据收集器
        data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # 对于自回归模型，设置为False
            )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,  # 添加数据收集器
        )

        self.trainer.train()

    def predict(self, text: str, max_new_tokens: int = 100, **generate_kwargs) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        default_generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        default_generate_kwargs.update(generate_kwargs)

        outputs = self.model.generate(
            **inputs,
            **default_generate_kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

import random
def generate_base_model_training_data(size: int = 100) -> Dataset:
    """
    生成用于训练基座模型的示例语料

    包含多种类型的文本，模拟通用基座模型训练
    """
    # 定义不同类型的文本模板
    templates = [
        "自然语言处理是人工智能的一个重要领域，它研究如何让计算机理解和生成人类语言。",
        "大型语言模型通过在海量文本上训练，学习语言的统计规律，从而能够生成连贯的文本。",
        "数学是科学的语言，它为我们提供了描述自然规律的精确工具。",
        "历史是一面镜子，它让我们能够从过去的经验中学习，避免重复错误。",
        "哲学思考帮助我们理解世界的本质和人类的存在意义。",
        "计算机科学的发展彻底改变了我们的生活方式和工作方式。",
        "物理学研究宇宙的基本规律，从微观粒子到宏观宇宙。",
        "生物学探索生命的奥秘，包括生物的结构、功能和演化。",
        "经济学研究人类如何分配有限的资源，以满足无限的需求。",
        "文学作品通过语言表达人类的情感、思想和想象力。"
    ]

    # 生成多样化的示例文本
    texts = []

    # 1. 直接使用模板
    texts.extend(templates)

    # 2. 组合不同模板的片段
    for i in range(size - len(templates)):
        parts = [templates[j] for j in torch.randint(0, len(templates), (torch.randint(2, 5, (1,))[0],))]
        # 使用Python的random.shuffle替代torch.random.shuffle
        random.shuffle(parts)
        texts.append(" ".join(parts))

    # 创建数据集
    return Dataset.from_dict({"text": texts})

def gen_textsDataset_from_list(text_list):
    """
        生成用于训练基座模型的示例语料

        包含多种类型的文本，模拟通用基座模型训练
        """
    # 定义不同类型的文本模板
    # 创建数据集
    return Dataset.from_dict({"text": text_list})

def test_train(  train_csvdata_path=r'E:\ml_data\chinese-poetry-collection\train.csv',
                 eval_csvdata_path =r'E:\ml_data\chinese-poetry-collection\test.csv',
                 model_path         = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct",
                 num_train_epochs   = 2,
                 learning_rate      = 1e-5,
                 output_dir         = "./output_base_model",
                 log_dir            = "./logs",
                 save_steps         = 400,
                 test_prompts       =  ['你是谁','床前明月光','美国在哪里',"历史研究可以帮助我们"],
                 test_prompts_generate= 250,
                 per_device_train_batch_size = 2,
                 per_device_eval_batch_size = 2
                 ) :
    # 3. 生成训练数据
    # train_dataset = generate_base_model_training_data(size=100)
    # # 4. 划分训练集和评估集
    import pandas as pd
    train_data = pd.read_csv( train_csvdata_path )
    eval_data  = pd.read_csv( eval_csvdata_path )

    train_dataset  = Dataset.from_dict({"text": train_data["text1"].tolist()})
    eval_dataset   = Dataset.from_dict({"text": eval_data["text1"].tolist()} )
    print( " len %s, len %s "%(len(train_dataset) , len( eval_dataset ) )  )
    print( 'eval_dataset',eval_dataset )

    # 1. 定义模型路径
    # 2. 初始化模型微调器
    # save_steps = int( len(train_data)/ per_device_train_batch_size /20 )
    print(  'change save_steps:%s'%save_steps )
    ft = UnslothFineTuner(  model_path=model_path,  max_seq_length=512,  load_in_4bit=False,)
    # 5. 定义训练参数 - 兼容新旧版本
    training_args_kwargs = {
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size":  per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "learning_rate":    learning_rate,
        "output_dir":       output_dir,
        "logging_dir":      log_dir,
        "logging_steps": 5,
        "save_steps":    save_steps,
        "weight_decay":  0.01,
        "warmup_ratio":  0.1  }

    # 检测Transformers版本，使用兼容的参数名
    try:
        from transformers import __version__ as transformers_version
        from packaging import version

        if version.parse(transformers_version) >= version.parse("4.22.0"):
            training_args_kwargs["evaluation_strategy"] = "steps"
            training_args_kwargs["eval_steps"] = 20
        else:
            training_args_kwargs["eval_strategy"] = "steps"
            training_args_kwargs["eval_steps"] = 20
    except (ImportError, AttributeError):
        # 旧版本Transformers，使用eval_strategy
        training_args_kwargs["eval_strategy"] = "steps"
        training_args_kwargs["eval_steps"] = 20

    # 创建TrainingArguments对象
    try:
        training_args = TrainingArguments(**training_args_kwargs)
    except TypeError as e:
        # 处理可能的参数错误
        print(f"警告: 训练参数设置失败 - {str(e)}")
        print("尝试使用默认评估策略...")

        # 移除可能不支持的参数
        if "evaluation_strategy" in training_args_kwargs:
            del training_args_kwargs["evaluation_strategy"]
        if "eval_strategy" in training_args_kwargs:
            del training_args_kwargs["eval_strategy"]
        if "eval_steps" in training_args_kwargs:
            del training_args_kwargs["eval_steps"]

        # 重新创建对象
        training_args = TrainingArguments(**training_args_kwargs)
    # 6. 微调模型
    ft.fit(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        text_column="text",
        label_column=None,
        training_args=training_args,
    )

    # 7. 测试模型
    print("测试基座模型:")
    for prompt in test_prompts:
        response = ft.predict(prompt, max_new_tokens=test_prompts_generate)
        print(f"输入: {prompt}")
        print(f"输出: {response}")
        print("-" * 50)

    # 8. 保存模型
    ft.save(output_dir)


from transformers import AutoModelForCausalLM, AutoTokenizer
def evaluate_checkpoint(checkpoint_dir="./output_base_model", eval_dataset=None):
    from transformers import pipeline
    # 1. 找到最新检查点
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)

    # 2. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # 3. 评估
    if eval_dataset:
        trainer = Trainer(
            model=model,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        results = trainer.evaluate()
        print(results)

    # 4. 测试生成
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(generator("自然语言处理是", max_new_tokens=50))


if __name__ == "__main__":
    test_train(  )
    evaluate_checkpoint(  )
