import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import argparse
import logging
from datasets import load_dataset, Dataset
import os
import json
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义数据集路径
DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'


@dataclass
class DistillationArguments:
    """
    知识蒸馏的参数配置类
    """
    teacher_model_path: str = field(
        default=None,
        metadata={"help": "教师模型路径"}
    )
    student_model_path: str = field(
        default=None,
        metadata={"help": "学生模型路径"}
    )
    output_dir: str = field(
        default="./distillation_output",
        metadata={"help": "蒸馏后模型输出目录"}
    )
    temperature: float = field(
        default=2.0,
        metadata={"help": "蒸馏温度"}
    )
    alpha: float = field(
        default=0.5,
        metadata={"help": "蒸馏损失权重"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "最大序列长度"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "批次大小"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "训练轮数"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "学习率"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "权重衰减"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "日志记录步数"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "模型保存步数"}
    )
    eval_steps: int = field(
        default=500,
        metadata={"help": "评估步数"}
    )
    dataset_path: str = field(
        default=DATASET_PATH,
        metadata={"help": "自定义数据集路径"}
    )
    use_custom_data: bool = field(
        default=True,
        metadata={"help": "是否使用自定义数据"}
    )
    use_cuda: bool = field(
        default=True,
        metadata={"help": "是否使用CUDA"}
    )
    num_samples: int = field(
        default=1000,
        metadata={"help": "当没有数据集时生成的样本数"}
    )
    prompt_template: str = field(
        default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
        metadata={"help": "提示模板"}
    )
import inspect  # 新增导入用于检查函数签名

class DistillationTrainer(Trainer):
    """
    自定义蒸馏Trainer
    """

    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算蒸馏损失 (改进版)
        """
        # 检查模型需要哪些输入参数
        signature = inspect.signature(model.forward)
        expected_inputs = list(signature.parameters.keys())

        # 过滤掉模型不需要的参数
        filtered_inputs = {k: v for k, v in inputs.items() if k in expected_inputs}

        # 确保只提供input_ids或inputs_embeds中的一个
        if 'input_ids' in filtered_inputs and 'inputs_embeds' in filtered_inputs:
            # 优先使用input_ids
            del filtered_inputs['inputs_embeds']

        # 如果两者都没有，尝试使用默认的input_ids
        if 'input_ids' not in filtered_inputs and 'inputs_embeds' not in filtered_inputs:
            logger.warning("既没有提供input_ids也没有提供inputs_embeds，尝试使用默认的input_ids")
            if 'input_ids' in inputs:
                filtered_inputs['input_ids'] = inputs['input_ids']

        # 获取学生模型输出
        student_outputs = model(**filtered_inputs)
        student_logits = student_outputs.logits

        # 计算学生模型的交叉熵损失
        ce_loss = student_outputs.loss

        # 禁用教师模型的梯度计算
        with torch.no_grad():
            teacher_outputs = self.teacher(**filtered_inputs)
            teacher_logits = teacher_outputs.logits

        # 计算蒸馏损失 (KL散度)
        distillation_loss = F.kl_div(
            input=F.log_softmax(student_logits / self.temperature, dim=-1),
            target=F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # 组合损失
        loss = (1. - self.alpha) * ce_loss + self.alpha * distillation_loss

        return (loss, student_outputs) if return_outputs else loss


class ModelDistiller:
    """
    大模型知识蒸馏封装类
    """

    def __init__(self, args: DistillationArguments):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

        # 加载教师模型和学生模型
        logger.info("加载教师模型...")
        self.teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_path)
        self.teacher.to(self.device)

        logger.info("加载学生模型...")
        self.student = AutoModelForCausalLM.from_pretrained(args.student_model_path)
        self.student.to(self.device)

        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, example):
        """
        格式化提示
        """
        if isinstance(example, str):
            # 如果是字符串，直接返回
            return example

        input_text = example.get("input", "")
        instruction = example.get("instruction", "")

        if instruction:
            if input_text:
                return self.args.prompt_template.format(
                    instruction=instruction,
                    input=input_text
                )
            return instruction
        return input_text if input_text else ""

    def generate_sample_data(self, num_samples=1000):
        """
        生成模拟数据（当没有真实数据时使用）
        """
        logger.info(f"生成 {num_samples} 条模拟数据...")

        # 生成一些简单的医学问答数据
        medical_questions = [
            {
                "instruction": "如何预防高血压？",
                "input": "",
                "output": "预防高血压的方法包括：1. 保持健康饮食，减少盐分摄入；2. 定期锻炼；3. 控制体重；4. 限制酒精摄入；5. 不吸烟；6. 管理压力。"
            },
            {
                "instruction": "糖尿病的早期症状有哪些？",
                "input": "",
                "output": "糖尿病的早期症状包括：1. 频繁口渴；2. 尿频；3. 容易饥饿；4. 疲劳；5. 视力模糊；6. 伤口愈合缓慢。"
            },
            {
                "instruction": "根据患者症状判断可能疾病",
                "input": "患者主诉：持续咳嗽两周，伴有低烧和夜间盗汗",
                "output": "根据症状描述，可能的诊断包括：1. 肺结核；2. 支气管炎；3. 肺炎。建议进行胸部X光检查和痰培养以确诊。"
            }
        ]

        # 扩展样本数据
        samples = []
        for i in range(num_samples):
            base_sample = random.choice(medical_questions)
            sample = base_sample.copy()
            if i > len(medical_questions):
                # 对基础样本进行一些变化
                sample["instruction"] = sample["instruction"].replace("？", "？请详细说明。")
                sample["output"] = sample["output"] + " 具体情况需要医生进一步诊断。"
            samples.append(sample)

        return Dataset.from_list(samples)

    def load_custom_dataset(self):
        """
        加载自定义数据集
        """
        logger.info(f"从 {self.args.dataset_path} 加载自定义数据集...")

        if not os.path.exists(self.args.dataset_path):
            logger.warning(f"数据集文件 {self.args.dataset_path} 不存在，将生成模拟数据")
            return self.generate_sample_data(self.args.num_samples)

        # 读取JSONL文件
        data = []
        with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 确保数据是字典格式
                    if not isinstance(item, dict):
                        item = {"text": str(item)}

                    # 确保有必要的字段
                    if "instruction" not in item and "output" not in item:
                        # 如果不是instruction/output格式，作为普通文本处理
                        item = {"instruction": str(item), "input": "", "output": str(item)}
                    elif "instruction" in item and "output" not in item:
                        item["output"] = str(item.get("input", ""))
                    elif "output" in item and "instruction" not in item:
                        item["instruction"] = str(item.get("input", ""))

                    if "input" not in item:
                        item["input"] = ""

                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON行失败: {e}, 内容: {line}")
                    # 将无法解析的行作为普通文本处理
                    data.append({"instruction": line.strip(), "input": "", "output": line.strip()})

        if not data:
            logger.warning("自定义数据集中没有有效数据，将生成模拟数据")
            return self.generate_sample_data(self.args.num_samples)

        return Dataset.from_list(data)

    def prepare_dataset(self):
        """
        准备数据集
        """
        if self.args.use_custom_data:
            dataset = self.load_custom_dataset()
        else:
            logger.info("使用默认的wikitext数据集...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
            # 将wikitext数据转换为类似instruction的格式
            dataset = dataset.map(lambda x: {"instruction": x["text"], "input": "", "output": ""})

        # 预处理函数 - 将instruction和input组合成模型输入，output作为标签
        def tokenize_function(examples):
            # 确保examples是字典格式
            if not isinstance(examples, dict):
                examples = {"instruction": [str(ex) for ex in examples]}

            # 获取instruction、input和output
            instructions = examples.get("instruction", [""] * len(examples.get("input", [""])))
            inputs = examples.get("input", [""] * len(instructions))
            outputs = examples.get("output", [""] * len(instructions))

            # 格式化提示
            prompts = []
            labels = []
            for i in range(len(instructions)):
                # 格式化输入
                prompt = self.format_prompt({
                    "instruction": instructions[i],
                    "input": inputs[i]
                })
                prompts.append(prompt)

                # 准备标签
                labels.append(outputs[i] if i < len(outputs) else "")

            # Tokenize输入
            model_inputs = self.tokenizer(
                prompts,
                truncation=True,
                max_length=self.args.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            # Tokenize标签
            labels = self.tokenizer(
                labels,
                truncation=True,
                max_length=self.args.max_length,
                padding="max_length",
                return_tensors="pt"
            )["input_ids"]

            # 将输出作为标签
            model_inputs["labels"] = labels
            return model_inputs

        # 对数据集进行tokenize
        try:
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=4
            )
        except Exception as e:
            logger.error(f"数据集处理失败: {e}")
            raise

        # 划分训练集和验证集
        if isinstance(tokenized_datasets, Dataset):
            # 如果是单个数据集，则分割
            split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
            tokenized_datasets = {
                "train": split_datasets["train"],
                "validation": split_datasets["test"]
            }
        elif "validation" not in tokenized_datasets:
            tokenized_datasets["validation"] = tokenized_datasets["train"].train_test_split(test_size=0.1)["test"]

        return tokenized_datasets

    def distill(self):
        """
        执行蒸馏过程
        """
        # 准备数据集
        try:
            tokenized_datasets = self.prepare_dataset()
        except Exception as e:
            logger.error(f"准备数据集失败: {e}")
            raise

        # 训练参数配置
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.num_epochs,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            eval_strategy="steps",
            eval_steps=self.args.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to=["tensorboard"],
            fp16=torch.cuda.is_available(),
        )

        # 创建蒸馏Trainer
        trainer = DistillationTrainer(
            teacher_model=self.teacher,
            temperature=self.args.temperature,
            alpha=self.args.alpha,
            model=self.student,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation", None),
            tokenizer=self.tokenizer,
        )

        # 开始训练
        logger.info("开始知识蒸馏训练...")
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise

        # 保存最终模型
        logger.info("训练完成，保存模型...")
        trainer.save_model(os.path.join(self.args.output_dir, "final_model"))

        return trainer


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="大模型知识蒸馏")

    parser.add_argument("--teacher_model_path", type=str, required=True,
                        help="教师模型路径")
    parser.add_argument("--student_model_path", type=str, required=True,
                        help="学生模型路径")
    parser.add_argument("--output_dir", type=str, default="./distillation_output",
                        help="蒸馏后模型输出目录")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="蒸馏温度")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="蒸馏损失权重")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="模型保存步数")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help="自定义数据集路径")
    # 解析命令行参数（续）
    parser.add_argument("--use_custom_data", action="store_true",
                        help="是否使用自定义数据")
    parser.add_argument("--no_use_custom_data", dest="use_custom_data", action="store_false",
                        help="不使用自定义数据")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="当没有数据集时生成的样本数")
    parser.add_argument("--prompt_template", type=str,
                        default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
                        help="提示模板")
    parser.add_argument("--use_cuda", action="store_true",
                        help="是否使用CUDA")

    parser.set_defaults(use_custom_data=True)
    return parser.parse_args()


def test_distillation():
    """
    测试蒸馏过程
    """
    # 填充测试参数
    args = DistillationArguments(
        teacher_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen2___5-0___5B-Instruct',
        student_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen1___5-0___5B',
        output_dir="./distillation_test_output",
        temperature=2.0,
        alpha=0.5,
        max_length=256,
        batch_size=2,
        num_epochs=1,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        dataset_path=DATASET_PATH,
        use_custom_data=True,
        num_samples=100,
        prompt_template="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
        use_cuda=True
    )

    # 创建蒸馏器
    distiller = ModelDistiller(args)

    # 执行蒸馏
    try:
        trainer = distiller.distill()
        logger.info("知识蒸馏测试成功完成!")
        return trainer
    except Exception as e:
        logger.error(f"知识蒸馏测试失败: {str(e)}")
        raise e


if __name__ == "__main__":
    # 测试蒸馏
    test_distillation()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from dataclasses import dataclass, field
# from typing import Optional, Dict, List, Tuple
# import argparse
# import logging
# from datasets import load_dataset, Dataset
# import os
# import json
# import random
#
# # 设置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # 自定义数据集路径
# DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
#
#
# @dataclass
# class DistillationArguments:
#     """
#     知识蒸馏的参数配置类
#     """
#     teacher_model_path: str = field(
#         default=None,
#         metadata={"help": "教师模型路径"}
#     )
#     student_model_path: str = field(
#         default=None,
#         metadata={"help": "学生模型路径"}
#     )
#     output_dir: str = field(
#         default="./distillation_output",
#         metadata={"help": "蒸馏后模型输出目录"}
#     )
#     temperature: float = field(
#         default=2.0,
#         metadata={"help": "蒸馏温度"}
#     )
#     alpha: float = field(
#         default=0.5,
#         metadata={"help": "蒸馏损失权重"}
#     )
#     max_length: int = field(
#         default=512,
#         metadata={"help": "最大序列长度"}
#     )
#     batch_size: int = field(
#         default=4,
#         metadata={"help": "批次大小"}
#     )
#     num_epochs: int = field(
#         default=3,
#         metadata={"help": "训练轮数"}
#     )
#     learning_rate: float = field(
#         default=5e-5,
#         metadata={"help": "学习率"}
#     )
#     weight_decay: float = field(
#         default=0.01,
#         metadata={"help": "权重衰减"}
#     )
#     logging_steps: int = field(
#         default=100,
#         metadata={"help": "日志记录步数"}
#     )
#     save_steps: int = field(
#         default=500,
#         metadata={"help": "模型保存步数"}
#     )
#     eval_steps: int = field(
#         default=500,
#         metadata={"help": "评估步数"}
#     )
#     dataset_path: str = field(
#         default=DATASET_PATH,
#         metadata={"help": "自定义数据集路径"}
#     )
#     use_custom_data: bool = field(
#         default=True,
#         metadata={"help": "是否使用自定义数据"}
#     )
#     use_cuda: bool = field(
#         default=True,
#         metadata={"help": "是否使用CUDA"}
#     )
#     num_samples: int = field(
#         default=1000,
#         metadata={"help": "当没有数据集时生成的样本数"}
#     )
#     prompt_template: str = field(
#         default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
#         metadata={"help": "提示模板"}
#     )
#
#
# class DistillationTrainer(Trainer):
#     """
#     自定义蒸馏Trainer
#     """
#
#     def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.teacher = teacher_model
#         self.temperature = temperature
#         self.alpha = alpha
#         self.teacher.eval()
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         计算蒸馏损失
#         """
#         # 获取学生模型输出
#         model_inputs = {
#             k: v for k, v in inputs.items()
#             if k not in ['num_items_in_batch', 'other_unexpected_args']
#         }
#         student_outputs = model(**inputs)
#         student_logits = student_outputs.logits
#
#         # 计算学生模型的交叉熵损失
#         ce_loss = student_outputs.loss
#         # 禁用教师模型的梯度计算
#         with torch.no_grad():
#             teacher_outputs = self.teacher(**model_inputs)
#             teacher_logits = teacher_outputs.logits
#
#         # 计算蒸馏损失 (KL散度)
#         distillation_loss = F.kl_div(
#             input=F.log_softmax(student_logits / self.temperature, dim=-1),
#             target=F.softmax(teacher_logits / self.temperature, dim=-1),
#             reduction="batchmean"
#         ) * (self.temperature ** 2)
#
#         # 组合损失
#         loss = (1. - self.alpha) * ce_loss + self.alpha * distillation_loss
#
#         return (loss, student_outputs) if return_outputs else loss
#
#
# class ModelDistiller:
#     """
#     大模型知识蒸馏封装类
#     """
#
#     def __init__(self, args: DistillationArguments):
#         self.args = args
#         self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
#
#         # 加载教师模型和学生模型
#         logger.info("加载教师模型...")
#         self.teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_path)
#         self.teacher.to(self.device)
#
#         logger.info("加载学生模型...")
#         self.student = AutoModelForCausalLM.from_pretrained(args.student_model_path)
#         self.student.to(self.device)
#
#         # 加载tokenizer
#         logger.info("加载tokenizer...")
#         self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#     def format_prompt(self, example):
#         """
#         格式化提示
#         """
#         if isinstance(example, str):
#             # 如果是字符串，直接返回
#             return example
#
#         input_text = example.get("input", "")
#         instruction = example.get("instruction", "")
#
#         if instruction:
#             if input_text:
#                 return self.args.prompt_template.format(
#                     instruction=instruction,
#                     input=input_text
#                 )
#             return instruction
#         return input_text if input_text else ""
#
#     def generate_sample_data(self, num_samples=1000):
#         """
#         生成模拟数据（当没有真实数据时使用）
#         """
#         logger.info(f"生成 {num_samples} 条模拟数据...")
#
#         # 生成一些简单的医学问答数据
#         medical_questions = [
#             {
#                 "instruction": "如何预防高血压？",
#                 "input": "",
#                 "output": "预防高血压的方法包括：1. 保持健康饮食，减少盐分摄入；2. 定期锻炼；3. 控制体重；4. 限制酒精摄入；5. 不吸烟；6. 管理压力。"
#             },
#             {
#                 "instruction": "糖尿病的早期症状有哪些？",
#                 "input": "",
#                 "output": "糖尿病的早期症状包括：1. 频繁口渴；2. 尿频；3. 容易饥饿；4. 疲劳；5. 视力模糊；6. 伤口愈合缓慢。"
#             },
#             {
#                 "instruction": "根据患者症状判断可能疾病",
#                 "input": "患者主诉：持续咳嗽两周，伴有低烧和夜间盗汗",
#                 "output": "根据症状描述，可能的诊断包括：1. 肺结核；2. 支气管炎；3. 肺炎。建议进行胸部X光检查和痰培养以确诊。"
#             }
#         ]
#
#         # 扩展样本数据
#         samples = []
#         for i in range(num_samples):
#             base_sample = random.choice(medical_questions)
#             sample = base_sample.copy()
#             if i > len(medical_questions):
#                 # 对基础样本进行一些变化
#                 sample["instruction"] = sample["instruction"].replace("？", "？请详细说明。")
#                 sample["output"] = sample["output"] + " 具体情况需要医生进一步诊断。"
#             samples.append(sample)
#
#         return Dataset.from_list(samples)
#
#     def load_custom_dataset(self):
#         """
#         加载自定义数据集
#         """
#         logger.info(f"从 {self.args.dataset_path} 加载自定义数据集...")
#
#         if not os.path.exists(self.args.dataset_path):
#             logger.warning(f"数据集文件 {self.args.dataset_path} 不存在，将生成模拟数据")
#             return self.generate_sample_data(self.args.num_samples)
#
#         # 读取JSONL文件
#         data = []
#         with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     item = json.loads(line.strip())
#                     # 确保数据是字典格式
#                     if not isinstance(item, dict):
#                         item = {"text": str(item)}
#
#                     # 确保有必要的字段
#                     if "instruction" not in item and "output" not in item:
#                         # 如果不是instruction/output格式，作为普通文本处理
#                         item = {"instruction": str(item), "input": "", "output": str(item)}
#                     elif "instruction" in item and "output" not in item:
#                         item["output"] = str(item.get("input", ""))
#                     elif "output" in item and "instruction" not in item:
#                         item["instruction"] = str(item.get("input", ""))
#
#                     if "input" not in item:
#                         item["input"] = ""
#
#                     data.append(item)
#                 except json.JSONDecodeError as e:
#                     logger.warning(f"解析JSON行失败: {e}, 内容: {line}")
#                     # 将无法解析的行作为普通文本处理
#                     data.append({"instruction": line.strip(), "input": "", "output": line.strip()})
#
#         if not data:
#             logger.warning("自定义数据集中没有有效数据，将生成模拟数据")
#             return self.generate_sample_data(self.args.num_samples)
#
#         return Dataset.from_list(data)
#
#     def prepare_dataset(self):
#         """
#         准备数据集
#         """
#         if self.args.use_custom_data:
#             dataset = self.load_custom_dataset()
#         else:
#             logger.info("使用默认的wikitext数据集...")
#             dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
#             # 将wikitext数据转换为类似instruction的格式
#             dataset = dataset.map(lambda x: {"instruction": x["text"], "input": "", "output": ""})
#
#         # 预处理函数 - 将instruction和input组合成模型输入，output作为标签
#         def tokenize_function(examples):
#             # 确保examples是字典格式
#             if not isinstance(examples, dict):
#                 examples = {"instruction": [str(ex) for ex in examples]}
#
#             # 获取instruction、input和output
#             instructions = examples.get("instruction", [""] * len(examples.get("input", [""])))
#             inputs = examples.get("input", [""] * len(instructions))
#             outputs = examples.get("output", [""] * len(instructions))
#
#             # 格式化提示
#             prompts = []
#             labels = []
#             for i in range(len(instructions)):
#                 # 格式化输入
#                 prompt = self.format_prompt({
#                     "instruction": instructions[i],
#                     "input": inputs[i]
#                 })
#                 prompts.append(prompt)
#
#                 # 准备标签
#                 labels.append(outputs[i] if i < len(outputs) else "")
#
#             # Tokenize输入
#             model_inputs = self.tokenizer(
#                 prompts,
#                 truncation=True,
#                 max_length=self.args.max_length,
#                 padding="max_length",
#                 return_tensors="pt"
#             )
#
#             # Tokenize标签
#             labels = self.tokenizer(
#                 labels,
#                 truncation=True,
#                 max_length=self.args.max_length,
#                 padding="max_length",
#                 return_tensors="pt"
#             )["input_ids"]
#
#             # 将输出作为标签
#             model_inputs["labels"] = labels
#             return model_inputs
#
#         # 对数据集进行tokenize
#         try:
#             tokenized_datasets = dataset.map(
#                 tokenize_function,
#                 batched=True,
#                 remove_columns=dataset.column_names,
#                 num_proc=4
#             )
#         except Exception as e:
#             logger.error(f"数据集处理失败: {e}")
#             raise
#
#         # 划分训练集和验证集
#         if isinstance(tokenized_datasets, Dataset):
#             # 如果是单个数据集，则分割
#             split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
#             tokenized_datasets = {
#                 "train": split_datasets["train"],
#                 "validation": split_datasets["test"]
#             }
#         elif "validation" not in tokenized_datasets:
#             tokenized_datasets["validation"] = tokenized_datasets["train"].train_test_split(test_size=0.1)["test"]
#
#         return tokenized_datasets
#
#     def distill(self):
#         """
#         执行蒸馏过程
#         """
#         # 准备数据集
#         try:
#             tokenized_datasets = self.prepare_dataset()
#         except Exception as e:
#             logger.error(f"准备数据集失败: {e}")
#             raise
#
#         # 训练参数配置
#         training_args = TrainingArguments(
#             output_dir=self.args.output_dir,
#             per_device_train_batch_size=self.args.batch_size,
#             per_device_eval_batch_size=self.args.batch_size,
#             num_train_epochs=self.args.num_epochs,
#             learning_rate=self.args.learning_rate,
#             weight_decay=self.args.weight_decay,
#             logging_steps=self.args.logging_steps,
#             save_steps=self.args.save_steps,
#             eval_strategy="steps",
#             eval_steps=self.args.eval_steps,
#             save_total_limit=2,
#             load_best_model_at_end=True,
#             report_to=["tensorboard"],
#             fp16=torch.cuda.is_available(),
#         )
#
#         # 创建蒸馏Trainer
#         trainer = DistillationTrainer(
#             teacher_model=self.teacher,
#             temperature=self.args.temperature,
#             alpha=self.args.alpha,
#             model=self.student,
#             args=training_args,
#             train_dataset=tokenized_datasets["train"],
#             eval_dataset=tokenized_datasets.get("validation", None),
#             tokenizer=self.tokenizer,
#         )
#
#         # 开始训练
#         logger.info("开始知识蒸馏训练...")
#         try:
#             trainer.train()
#         except Exception as e:
#             logger.error(f"训练过程中出错: {e}")
#             raise
#
#         # 保存最终模型
#         logger.info("训练完成，保存模型...")
#         trainer.save_model(os.path.join(self.args.output_dir, "final_model"))
#
#         return trainer
#
#
# def parse_args():
#     """
#     解析命令行参数
#     """
#     parser = argparse.ArgumentParser(description="大模型知识蒸馏")
#
#     parser.add_argument("--teacher_model_path", type=str, required=True,
#                         help="教师模型路径")
#     parser.add_argument("--student_model_path", type=str, required=True,
#                         help="学生模型路径")
#     parser.add_argument("--output_dir", type=str, default="./distillation_output",
#                         help="蒸馏后模型输出目录")
#     parser.add_argument("--temperature", type=float, default=2.0,
#                         help="蒸馏温度")
#     parser.add_argument("--alpha", type=float, default=0.5,
#                         help="蒸馏损失权重")
#     parser.add_argument("--max_length", type=int, default=512,
#                         help="最大序列长度")
#     parser.add_argument("--batch_size", type=int, default=4,
#                         help="批次大小")
#     parser.add_argument("--num_epochs", type=int, default=3,
#                         help="训练轮数")
#     parser.add_argument("--learning_rate", type=float, default=5e-5,
#                         help="学习率")
#     parser.add_argument("--weight_decay", type=float, default=0.01,
#                         help="权重衰减")
#     parser.add_argument("--logging_steps", type=int, default=100,
#                         help="日志记录步数")
#     parser.add_argument("--save_steps", type=int, default=500,
#                         help="模型保存步数")
#     parser.add_argument("--eval_steps", type=int, default=500,
#                         help="评估步数")
#     parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
#                         help="自定义数据集路径")
#     parser.add_argument("--use_custom_data", action="store_true",
#                         help="是否使用自定义数据")
#     parser.add_argument("--no_use_custom_data", dest="use_custom_data", action="store_false",
#                         help="不使用自定义数据")
#     parser.add_argument("--num_samples", type=int, default=1000,
#                         help="当没有数据集时生成的样本数")
#     parser.add_argument("--prompt_template", type=str,
#                         default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
#                         help="提示模板")
#     parser.add_argument("--use_cuda", action="store_true",
#                         help="是否使用CUDA")
#
#     parser.set_defaults(use_custom_data=True)
#     return parser.parse_args()
#
#
# def test_distillation():
#     """
#     测试蒸馏过程
#     """
#     # 填充测试参数
#     args = DistillationArguments(
#         teacher_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen2___5-0___5B-Instruct',
#         student_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen1___5-0___5B',
#         output_dir="./distillation_test_output",
#         temperature=2.0,
#         alpha=0.5,
#         max_length=256,
#         batch_size=2,
#         num_epochs=1,
#         learning_rate=5e-5,
#         weight_decay=0.01,
#         logging_steps=10,
#         save_steps=50,
#         eval_steps=50,
#         dataset_path=DATASET_PATH,
#         use_custom_data=True,
#         num_samples=100,
#         prompt_template="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
#         use_cuda=True
#     )
#
#     # 创建蒸馏器
#     distiller = ModelDistiller(args)
#
#     # 执行蒸馏
#     try:
#         trainer = distiller.distill()
#         logger.info("知识蒸馏测试成功完成!")
#         return trainer
#     except Exception as e:
#         logger.error(f"知识蒸馏测试失败: {str(e)}")
#         raise e
#
#
# if __name__ == "__main__":
#     # 测试蒸馏
#     test_distillation()
#
#     # 或者使用命令行参数运行
#     # args = parse_args()
#     # distiller = ModelDistiller(args)
#     # distiller.distill()
#
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# # from dataclasses import dataclass, field
# # from typing import Optional, Dict, List, Tuple
# # import argparse
# # import logging
# # from datasets import load_dataset, Dataset
# # import os
# # import json
# # import random
# #
# # # 设置日志
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)
# #
# # # 自定义数据集路径
# # DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
# #
# #
# # @dataclass
# # class DistillationArguments:
# #     """
# #     知识蒸馏的参数配置类
# #     """
# #     teacher_model_path: str = field(
# #         default=None,
# #         metadata={"help": "教师模型路径"}
# #     )
# #     student_model_path: str = field(
# #         default=None,
# #         metadata={"help": "学生模型路径"}
# #     )
# #     output_dir: str = field(
# #         default="./distillation_output",
# #         metadata={"help": "蒸馏后模型输出目录"}
# #     )
# #     temperature: float = field(
# #         default=2.0,
# #         metadata={"help": "蒸馏温度"}
# #     )
# #     alpha: float = field(
# #         default=0.5,
# #         metadata={"help": "蒸馏损失权重"}
# #     )
# #     max_length: int = field(
# #         default=512,
# #         metadata={"help": "最大序列长度"}
# #     )
# #     batch_size: int = field(
# #         default=4,
# #         metadata={"help": "批次大小"}
# #     )
# #     num_epochs: int = field(
# #         default=3,
# #         metadata={"help": "训练轮数"}
# #     )
# #     learning_rate: float = field(
# #         default=5e-5,
# #         metadata={"help": "学习率"}
# #     )
# #     weight_decay: float = field(
# #         default=0.01,
# #         metadata={"help": "权重衰减"}
# #     )
# #     logging_steps: int = field(
# #         default=100,
# #         metadata={"help": "日志记录步数"}
# #     )
# #     save_steps: int = field(
# #         default=500,
# #         metadata={"help": "模型保存步数"}
# #     )
# #     eval_steps: int = field(
# #         default=500,
# #         metadata={"help": "评估步数"}
# #     )
# #     dataset_path: str = field(
# #         default=DATASET_PATH,
# #         metadata={"help": "自定义数据集路径"}
# #     )
# #     use_custom_data: bool = field(
# #         default=True,
# #         metadata={"help": "是否使用自定义数据"}
# #     )
# #     use_cuda: bool = field(
# #         default=True,
# #         metadata={"help": "是否使用CUDA"}
# #     )
# #     num_samples: int = field(
# #         default=1000,
# #         metadata={"help": "当没有数据集时生成的样本数"}
# #     )
# #     prompt_template: str = field(
# #         default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
# #         metadata={"help": "提示模板"}
# #     )
# #
# #
# # class DistillationTrainer(Trainer):
# #     """
# #     自定义蒸馏Trainer
# #     """
# #
# #     def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.teacher = teacher_model
# #         self.temperature = temperature
# #         self.alpha = alpha
# #         self.teacher.eval()
# #
# #     def compute_loss(self, model, inputs, return_outputs=False):
# #         """
# #         计算蒸馏损失
# #         """
# #         # 获取学生模型输出
# #         student_outputs = model(**inputs)
# #         student_logits = student_outputs.logits
# #
# #         # 计算学生模型的交叉熵损失
# #         ce_loss = student_outputs.loss
# #
# #         # 禁用教师模型的梯度计算
# #         with torch.no_grad():
# #             teacher_outputs = self.teacher(**inputs)
# #             teacher_logits = teacher_outputs.logits
# #
# #         # 计算蒸馏损失 (KL散度)
# #         distillation_loss = F.kl_div(
# #             input=F.log_softmax(student_logits / self.temperature, dim=-1),
# #             target=F.softmax(teacher_logits / self.temperature, dim=-1),
# #             reduction="batchmean"
# #         ) * (self.temperature ** 2)
# #
# #         # 组合损失
# #         loss = (1. - self.alpha) * ce_loss + self.alpha * distillation_loss
# #
# #         return (loss, student_outputs) if return_outputs else loss
# #
# #
# # class ModelDistiller:
# #     """
# #     大模型知识蒸馏封装类
# #     """
# #
# #     def __init__(self, args: DistillationArguments):
# #         self.args = args
# #         self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
# #
# #         # 加载教师模型和学生模型
# #         logger.info("加载教师模型...")
# #         self.teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_path)
# #         self.teacher.to(self.device)
# #
# #         logger.info("加载学生模型...")
# #         self.student = AutoModelForCausalLM.from_pretrained(args.student_model_path)
# #         self.student.to(self.device)
# #
# #         # 加载tokenizer
# #         logger.info("加载tokenizer...")
# #         self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
# #         if self.tokenizer.pad_token is None:
# #             self.tokenizer.pad_token = self.tokenizer.eos_token
# #
# #     def format_prompt(self, example):
# #         """
# #         格式化提示
# #         """
# #         input_text = example.get("input", "")
# #         if input_text:
# #             return self.args.prompt_template.format(
# #                 instruction=example["instruction"],
# #                 input=input_text
# #             )
# #         return example["instruction"]
# #
# #     def generate_sample_data(self, num_samples=1000):
# #         """
# #         生成模拟数据（当没有真实数据时使用）
# #         """
# #         logger.info(f"生成 {num_samples} 条模拟数据...")
# #
# #         # 生成一些简单的医学问答数据
# #         medical_questions = [
# #             {
# #                 "instruction": "如何预防高血压？",
# #                 "input": "",
# #                 "output": "预防高血压的方法包括：1. 保持健康饮食，减少盐分摄入；2. 定期锻炼；3. 控制体重；4. 限制酒精摄入；5. 不吸烟；6. 管理压力。"
# #             },
# #             {
# #                 "instruction": "糖尿病的早期症状有哪些？",
# #                 "input": "",
# #                 "output": "糖尿病的早期症状包括：1. 频繁口渴；2. 尿频；3. 容易饥饿；4. 疲劳；5. 视力模糊；6. 伤口愈合缓慢。"
# #             },
# #             {
# #                 "instruction": "根据患者症状判断可能疾病",
# #                 "input": "患者主诉：持续咳嗽两周，伴有低烧和夜间盗汗",
# #                 "output": "根据症状描述，可能的诊断包括：1. 肺结核；2. 支气管炎；3. 肺炎。建议进行胸部X光检查和痰培养以确诊。"
# #             }
# #         ]
# #
# #         # 扩展样本数据
# #         samples = []
# #         for i in range(num_samples):
# #             base_sample = random.choice(medical_questions)
# #             sample = base_sample.copy()
# #             if i > len(medical_questions):
# #                 # 对基础样本进行一些变化
# #                 sample["instruction"] = sample["instruction"].replace("？", "？请详细说明。")
# #                 sample["output"] = sample["output"] + " 具体情况需要医生进一步诊断。"
# #             samples.append(sample)
# #
# #         return Dataset.from_list(samples)
# #
# #     def load_custom_dataset(self):
# #         """
# #         加载自定义数据集
# #         """
# #         logger.info(f"从 {self.args.dataset_path} 加载自定义数据集...")
# #
# #         if not os.path.exists(self.args.dataset_path):
# #             logger.warning(f"数据集文件 {self.args.dataset_path} 不存在，将生成模拟数据")
# #             return self.generate_sample_data(self.args.num_samples)
# #
# #         # 读取JSONL文件
# #         data = []
# #         with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
# #             for line in f:
# #                 try:
# #                     item = json.loads(line)
# #                     # 确保有必要的字段
# #                     if "instruction" in item and "output" in item:
# #                         if "input" not in item:
# #                             item["input"] = ""
# #                         data.append(item)
# #                 except json.JSONDecodeError as e:
# #                     logger.warning(f"解析JSON行失败: {e}")
# #
# #         if not data:
# #             logger.warning("自定义数据集中没有有效数据，将生成模拟数据")
# #             return self.generate_sample_data(self.args.num_samples)
# #
# #         return Dataset.from_list(data)
# #
# #     def prepare_dataset(self):
# #         """
# #         准备数据集
# #         """
# #         if self.args.use_custom_data:
# #             dataset = self.load_custom_dataset()
# #         else:
# #             logger.info("使用默认的wikitext数据集...")
# #             dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# #             # 将wikitext数据转换为类似instruction的格式
# #             dataset = dataset.map(lambda x: {"instruction": x["text"], "input": "", "output": ""})
# #
# #         # 预处理函数 - 将instruction和input组合成模型输入，output作为标签
# #         def tokenize_function(examples):
# #             # 格式化提示
# #             prompts = [self.format_prompt(ex) for ex in examples]
# #             # 组合输入和输出
# #             inputs = self.tokenizer(
# #                 prompts,
# #                 truncation=True,
# #                 max_length=self.args.max_length,
# #                 padding="max_length",
# #                 return_tensors="pt"
# #             )
# #
# #             # 对输出进行tokenize
# #             outputs = self.tokenizer(
# #                 examples["output"],
# #                 truncation=True,
# #                 max_length=self.args.max_length,
# #                 padding="max_length",
# #                 return_tensors="pt"
# #             )
# #
# #             # 将输出作为标签
# #             inputs["labels"] = outputs["input_ids"]
# #             return inputs
# #
# #         # 对数据集进行tokenize
# #         tokenized_datasets = dataset.map(
# #             tokenize_function,
# #             batched=True,
# #             remove_columns=["instruction", "input", "output"] if all(k in dataset.column_names for k in ["instruction", "input", "output"]) else dataset.column_names,
# #             num_proc=4
# #         )
# #
# #         # 划分训练集和验证集
# #         if isinstance(tokenized_datasets, Dataset):
# #             # 如果是单个数据集，则分割
# #             split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
# #             tokenized_datasets = {
# #                 "train": split_datasets["train"],
# #                 "validation": split_datasets["test"]
# #             }
# #         elif "validation" not in tokenized_datasets:
# #             tokenized_datasets["validation"] = tokenized_datasets["train"].train_test_split(test_size=0.1)["test"]
# #
# #         return tokenized_datasets
# #
# #     def distill(self):
# #         """
# #         执行蒸馏过程
# #         """
# #         # 准备数据集
# #         tokenized_datasets = self.prepare_dataset()
# #
# #         # 训练参数配置
# #         training_args = TrainingArguments(
# #             output_dir=self.args.output_dir,
# #             per_device_train_batch_size=self.args.batch_size,
# #             per_device_eval_batch_size=self.args.batch_size,
# #             num_train_epochs=self.args.num_epochs,
# #             learning_rate=self.args.learning_rate,
# #             weight_decay=self.args.weight_decay,
# #             logging_steps=self.args.logging_steps,
# #             save_steps=self.args.save_steps,
# #             evaluation_strategy="steps",
# #             eval_steps=self.args.eval_steps,
# #             save_total_limit=2,
# #             load_best_model_at_end=True,
# #             report_to=["tensorboard"],
# #             fp16=torch.cuda.is_available(),
# #         )
# #
# #         # 创建蒸馏Trainer
# #         trainer = DistillationTrainer(
# #             teacher_model=self.teacher,
# #             temperature=self.args.temperature,
# #             alpha=self.args.alpha,
# #             model=self.student,
# #             args=training_args,
# #             train_dataset=tokenized_datasets["train"],
# #             eval_dataset=tokenized_datasets.get("validation", None),
# #             tokenizer=self.tokenizer,
# #         )
# #
# #         # 开始训练
# #         logger.info("开始知识蒸馏训练...")
# #         trainer.train()
# #
# #         # 保存最终模型
# #         logger.info("训练完成，保存模型...")
# #         trainer.save_model(os.path.join(self.args.output_dir, "final_model"))
# #
# #         return trainer
# #
# #
# # def parse_args():
# #     """
# #     解析命令行参数
# #     """
# #     parser = argparse.ArgumentParser(description="大模型知识蒸馏")
# #
# #     parser.add_argument("--teacher_model_path", type=str, required=True,
# #                         help="教师模型路径")
# #     parser.add_argument("--student_model_path", type=str, required=True,
# #                         help="学生模型路径")
# #     parser.add_argument("--output_dir", type=str, default="./distillation_output",
# #                         help="蒸馏后模型输出目录")
# #     parser.add_argument("--temperature", type=float, default=2.0,
# #                         help="蒸馏温度")
# #     parser.add_argument("--alpha", type=float, default=0.5,
# #                         help="蒸馏损失权重")
# #     parser.add_argument("--max_length", type=int, default=512,
# #                         help="最大序列长度")
# #     parser.add_argument("--batch_size", type=int, default=4,
# #                         help="批次大小")
# #     parser.add_argument("--num_epochs", type=int, default=3,
# #                         help="训练轮数")
# #     parser.add_argument("--learning_rate", type=float, default=5e-5,
# #                         help="学习率")
# #     parser.add_argument("--weight_decay", type=float, default=0.01,
# #                         help="权重衰减")
# #     parser.add_argument("--logging_steps", type=int, default=100,
# #                         help="日志记录步数")
# #     parser.add_argument("--save_steps", type=int, default=500,
# #                         help="模型保存步数")
# #     parser.add_argument("--eval_steps", type=int, default=500,
# #                         help="评估步数")
# #     parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
# #                         help="自定义数据集路径")
# #     parser.add_argument("--use_custom_data", action="store_true",
# #                         help="是否使用自定义数据")
# #     parser.add_argument("--no_use_custom_data", dest="use_custom_data", action="store_false",
# #                         help="不使用自定义数据")
# #     parser.add_argument("--num_samples", type=int, default=1000,
# #                         help="当没有数据集时生成的样本数")
# #     parser.add_argument("--prompt_template", type=str,
# #                         default="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
# #                         help="提示模板")
# #     parser.add_argument("--use_cuda", action="store_true",
# #                         help="是否使用CUDA")
# #
# #     parser.set_defaults(use_custom_data=True)
# #     return parser.parse_args()
# #
# #
# # def test_distillation():
# #     """
# #     测试蒸馏过程
# #     """
# #     # 填充测试参数
# #     args = DistillationArguments(
# #         teacher_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen2___5-0___5B-Instruct',
# #         student_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen1___5-0___5B',
# #         output_dir="./distillation_test_output",
# #         temperature=2.0,
# #         alpha=0.5,
# #         max_length=256,
# #         batch_size=2,
# #         num_epochs=1,
# #         learning_rate=5e-5,
# #         weight_decay=0.01,
# #         logging_steps=10,
# #         save_steps=50,
# #         eval_steps=50,
# #         dataset_path=DATASET_PATH,
# #         use_custom_data=True,
# #         num_samples=100,
# #         prompt_template="以下是一个医学相关问题，请根据要求回答。\n指令: {instruction}\n输入: {input}\n回答:",
# #         use_cuda=True
# #     )
# #
# #     # 创建蒸馏器
# #     distiller = ModelDistiller(args)
# #
# #     # 执行蒸馏
# #     try:
# #         trainer = distiller.distill()
# #         logger.info("知识蒸馏测试成功完成!")
# #         return trainer
# #     except Exception as e:
# #         logger.error(f"知识蒸馏测试失败: {str(e)}")
# #         raise e
# #
# #
# # if __name__ == "__main__":
# #     # 测试蒸馏
# #     test_distillation()
# #
# #     # 或者使用命令行参数运行
# #     # args = parse_args()
# #     # distiller = ModelDistiller(args)
# #     # distiller.distill()
# #
# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F
# # # from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# # # from dataclasses import dataclass, field
# # # from typing import Optional, Dict, List, Tuple
# # # import argparse
# # # import logging
# # # from datasets import load_dataset, Dataset
# # # import os
# # # import json
# # # import random
# # #
# # # # 设置日志
# # # logging.basicConfig(level=logging.INFO)
# # # logger = logging.getLogger(__name__)
# # #
# # # # 自定义数据集路径
# # # DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'
# # #
# # #
# # # @dataclass
# # # class DistillationArguments:
# # #     """
# # #     知识蒸馏的参数配置类
# # #     """
# # #     teacher_model_path: str = field(
# # #         default=None,
# # #         metadata={"help": "教师模型路径"}
# # #     )
# # #     student_model_path: str = field(
# # #         default=None,
# # #         metadata={"help": "学生模型路径"}
# # #     )
# # #     output_dir: str = field(
# # #         default="./distillation_output",
# # #         metadata={"help": "蒸馏后模型输出目录"}
# # #     )
# # #     temperature: float = field(
# # #         default=2.0,
# # #         metadata={"help": "蒸馏温度"}
# # #     )
# # #     alpha: float = field(
# # #         default=0.5,
# # #         metadata={"help": "蒸馏损失权重"}
# # #     )
# # #     max_length: int = field(
# # #         default=512,
# # #         metadata={"help": "最大序列长度"}
# # #     )
# # #     batch_size: int = field(
# # #         default=4,
# # #         metadata={"help": "批次大小"}
# # #     )
# # #     num_epochs: int = field(
# # #         default=3,
# # #         metadata={"help": "训练轮数"}
# # #     )
# # #     learning_rate: float = field(
# # #         default=5e-5,
# # #         metadata={"help": "学习率"}
# # #     )
# # #     weight_decay: float = field(
# # #         default=0.01,
# # #         metadata={"help": "权重衰减"}
# # #     )
# # #     logging_steps: int = field(
# # #         default=100,
# # #         metadata={"help": "日志记录步数"}
# # #     )
# # #     save_steps: int = field(
# # #         default=500,
# # #         metadata={"help": "模型保存步数"}
# # #     )
# # #     eval_steps: int = field(
# # #         default=500,
# # #         metadata={"help": "评估步数"}
# # #     )
# # #     dataset_path: str = field(
# # #         default=DATASET_PATH,
# # #         metadata={"help": "自定义数据集路径"}
# # #     )
# # #     use_custom_data: bool = field(
# # #         default=True,
# # #         metadata={"help": "是否使用自定义数据"}
# # #     )
# # #     use_cuda: bool = field(
# # #         default=True,
# # #         metadata={"help": "是否使用CUDA"}
# # #     )
# # #     num_samples: int = field(
# # #         default=1000,
# # #         metadata={"help": "当没有数据集时生成的样本数"}
# # #     )
# # #
# # #
# # # class DistillationTrainer(Trainer):
# # #     """
# # #     自定义蒸馏Trainer
# # #     """
# # #
# # #     def __init__(self, teacher_model=None, temperature=2.0, alpha=0.5, *args, **kwargs):
# # #         super().__init__(*args, **kwargs)
# # #         self.teacher = teacher_model
# # #         self.temperature = temperature
# # #         self.alpha = alpha
# # #         self.teacher.eval()
# # #
# # #     def compute_loss(self, model, inputs, return_outputs=False):
# # #         """
# # #         计算蒸馏损失
# # #         """
# # #         # 获取学生模型输出
# # #         student_outputs = model(**inputs)
# # #         student_logits = student_outputs.logits
# # #
# # #         # 计算学生模型的交叉熵损失
# # #         ce_loss = student_outputs.loss
# # #
# # #         # 禁用教师模型的梯度计算
# # #         with torch.no_grad():
# # #             teacher_outputs = self.teacher(**inputs)
# # #             teacher_logits = teacher_outputs.logits
# # #
# # #         # 计算蒸馏损失 (KL散度)
# # #         distillation_loss = F.kl_div(
# # #             input=F.log_softmax(student_logits / self.temperature, dim=-1),
# # #             target=F.softmax(teacher_logits / self.temperature, dim=-1),
# # #             reduction="batchmean"
# # #         ) * (self.temperature ** 2)
# # #
# # #         # 组合损失
# # #         loss = (1. - self.alpha) * ce_loss + self.alpha * distillation_loss
# # #
# # #         return (loss, student_outputs) if return_outputs else loss
# # #
# # #
# # # class ModelDistiller:
# # #     """
# # #     大模型知识蒸馏封装类
# # #     """
# # #
# # #     def __init__(self, args: DistillationArguments):
# # #         self.args = args
# # #         self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
# # #
# # #         # 加载教师模型和学生模型
# # #         logger.info("加载教师模型...")
# # #         self.teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model_path)
# # #         self.teacher.to(self.device)
# # #
# # #         logger.info("加载学生模型...")
# # #         self.student = AutoModelForCausalLM.from_pretrained(args.student_model_path)
# # #         self.student.to(self.device)
# # #
# # #         # 加载tokenizer
# # #         logger.info("加载tokenizer...")
# # #         self.tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
# # #         if self.tokenizer.pad_token is None:
# # #             self.tokenizer.pad_token = self.tokenizer.eos_token
# # #
# # #     def generate_sample_data(self, num_samples=1000):
# # #         """
# # #         生成模拟数据（当没有真实数据时使用）
# # #         """
# # #         logger.info(f"生成 {num_samples} 条模拟数据...")
# # #
# # #         # 生成一些简单的医学相关文本
# # #         medical_topics = [
# # #             "心脏病诊断和治疗方案",
# # #             "糖尿病患者的饮食建议",
# # #             "高血压的预防措施",
# # #             "癌症早期筛查指南",
# # #             "心理健康咨询案例",
# # #             "儿科常见疾病处理",
# # #             "老年人健康管理",
# # #             "中医养生方法",
# # #             "手术后的康复指导",
# # #             "药物相互作用分析"
# # #         ]
# # #
# # #         samples = []
# # #         for i in range(num_samples):
# # #             topic = random.choice(medical_topics)
# # #             text = f"这是一篇关于{topic}的医学文章。文章详细介绍了{topic}的相关知识。"
# # #             samples.append({"text": text})
# # #
# # #         return Dataset.from_list(samples)
# # #
# # #     def load_custom_dataset(self):
# # #         """
# # #         加载自定义数据集
# # #         """
# # #         logger.info(f"从 {self.args.dataset_path} 加载自定义数据集...")
# # #
# # #         if not os.path.exists(self.args.dataset_path):
# # #             logger.warning(f"数据集文件 {self.args.dataset_path} 不存在，将生成模拟数据")
# # #             return self.generate_sample_data(self.args.num_samples)
# # #
# # #         # 读取JSONL文件
# # #         data = []
# # #         with open(self.args.dataset_path, 'r', encoding='utf-8') as f:
# # #             for line in f:
# # #                 try:
# # #                     data.append(json.loads(line))
# # #                 except json.JSONDecodeError as e:
# # #                     logger.warning(f"解析JSON行失败: {e}")
# # #
# # #         if not data:
# # #             logger.warning("自定义数据集中没有有效数据，将生成模拟数据")
# # #             return self.generate_sample_data(self.args.num_samples)
# # #
# # #         return Dataset.from_list(data)
# # #
# # #     def prepare_dataset(self):
# # #         """
# # #         准备数据集
# # #         """
# # #         if self.args.use_custom_data:
# # #             dataset = self.load_custom_dataset()
# # #         else:
# # #             logger.info("使用默认的wikitext数据集...")
# # #             dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
# # #
# # #         # 预处理函数
# # #         def tokenize_function(examples):
# # #             return self.tokenizer(
# # #                 examples["text"],
# # #                 truncation=True,
# # #                 max_length=self.args.max_length,
# # #                 padding="max_length",
# # #                 return_tensors="pt"
# # #             )
# # #
# # #         # 对数据集进行tokenize
# # #         tokenized_datasets = dataset.map(
# # #             tokenize_function,
# # #             batched=True,
# # #             remove_columns=["text"] if "text" in dataset.column_names else dataset.column_names,
# # #             num_proc=4
# # #         )
# # #
# # #         # 划分训练集和验证集
# # #         if isinstance(tokenized_datasets, Dataset):
# # #             # 如果是单个数据集，则分割
# # #             split_datasets = tokenized_datasets.train_test_split(test_size=0.1)
# # #             tokenized_datasets = {
# # #                 "train": split_datasets["train"],
# # #                 "validation": split_datasets["test"]
# # #             }
# # #         elif "validation" not in tokenized_datasets:
# # #             tokenized_datasets["validation"] = tokenized_datasets["train"].train_test_split(test_size=0.1)["test"]
# # #
# # #         return tokenized_datasets
# # #
# # #     def distill(self):
# # #         """
# # #         执行蒸馏过程
# # #         """
# # #         # 准备数据集
# # #         tokenized_datasets = self.prepare_dataset()
# # #
# # #         # 训练参数配置
# # #         training_args = TrainingArguments(
# # #             output_dir=self.args.output_dir,
# # #             per_device_train_batch_size=self.args.batch_size,
# # #             per_device_eval_batch_size=self.args.batch_size,
# # #             num_train_epochs=self.args.num_epochs,
# # #             learning_rate=self.args.learning_rate,
# # #             weight_decay=self.args.weight_decay,
# # #             logging_steps=self.args.logging_steps,
# # #             save_steps=self.args.save_steps,
# # #             evaluation_strategy="steps",
# # #             eval_steps=self.args.eval_steps,
# # #             save_total_limit=2,
# # #             load_best_model_at_end=True,
# # #             report_to=["tensorboard"],
# # #             fp16=torch.cuda.is_available(),
# # #         )
# # #
# # #         # 创建蒸馏Trainer
# # #         trainer = DistillationTrainer(
# # #             teacher_model=self.teacher,
# # #             temperature=self.args.temperature,
# # #             alpha=self.args.alpha,
# # #             model=self.student,
# # #             args=training_args,
# # #             train_dataset=tokenized_datasets["train"],
# # #             eval_dataset=tokenized_datasets.get("validation", None),
# # #             tokenizer=self.tokenizer,
# # #         )
# # #
# # #         # 开始训练
# # #         logger.info("开始知识蒸馏训练...")
# # #         trainer.train()
# # #
# # #         # 保存最终模型
# # #         logger.info("训练完成，保存模型...")
# # #         trainer.save_model(os.path.join(self.args.output_dir, "final_model"))
# # #
# # #         return trainer
# # #
# # #
# # # def parse_args():
# # #     """
# # #     解析命令行参数
# # #     """
# # #     parser = argparse.ArgumentParser(description="大模型知识蒸馏")
# # #
# # #     parser.add_argument("--teacher_model_path", type=str, required=True,
# # #                         help="教师模型路径")
# # #     parser.add_argument("--student_model_path", type=str, required=True,
# # #                         help="学生模型路径")
# # #     parser.add_argument("--output_dir", type=str, default="./distillation_output",
# # #                         help="蒸馏后模型输出目录")
# # #     parser.add_argument("--temperature", type=float, default=2.0,
# # #                         help="蒸馏温度")
# # #     parser.add_argument("--alpha", type=float, default=0.5,
# # #                         help="蒸馏损失权重")
# # #     parser.add_argument("--max_length", type=int, default=512,
# # #                         help="最大序列长度")
# # #     parser.add_argument("--batch_size", type=int, default=4,
# # #                         help="批次大小")
# # #     parser.add_argument("--num_epochs", type=int, default=3,
# # #                         help="训练轮数")
# # #     parser.add_argument("--learning_rate", type=float, default=5e-5,
# # #                         help="学习率")
# # #     parser.add_argument("--weight_decay", type=float, default=0.01,
# # #                         help="权重衰减")
# # #     parser.add_argument("--logging_steps", type=int, default=100,
# # #                         help="日志记录步数")
# # #     parser.add_argument("--save_steps", type=int, default=500,
# # #                         help="模型保存步数")
# # #     parser.add_argument("--eval_steps", type=int, default=500,
# # #                         help="评估步数")
# # #     parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
# # #                         help="自定义数据集路径")
# # #     parser.add_argument("--use_custom_data", action="store_true",
# # #                         help="是否使用自定义数据")
# # #     parser.add_argument("--no_use_custom_data", dest="use_custom_data", action="store_false",
# # #                         help="不使用自定义数据")
# # #     parser.add_argument("--num_samples", type=int, default=1000,
# # #                         help="当没有数据集时生成的样本数")
# # #     parser.add_argument("--use_cuda", action="store_true",
# # #                         help="是否使用CUDA")
# # #
# # #     parser.set_defaults(use_custom_data=True)
# # #     return parser.parse_args()
# # #
# # #
# # # def test_distillation():
# # #     """
# # #     测试蒸馏过程
# # #     """
# # #     # 填充测试参数
# # #     args = DistillationArguments(
# # #         teacher_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen2___5-0___5B-Instruct',
# # #         student_model_path=r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen1___5-0___5B',
# # #         output_dir="./distillation_test_output",
# # #         temperature=2.0,
# # #         alpha=0.5,
# # #         max_length=256,
# # #         batch_size=2,
# # #         num_epochs=1,
# # #         learning_rate=5e-5,
# # #         weight_decay=0.01,
# # #         logging_steps=10,
# # #         save_steps=50,
# # #         eval_steps=50,
# # #         dataset_path=DATASET_PATH,
# # #         use_custom_data=True,
# # #         num_samples=100,
# # #         use_cuda=True
# # #     )
# # #
# # #     # 创建蒸馏器
# # #     distiller = ModelDistiller(args)
# # #
# # #     # 执行蒸馏
# # #     try:
# # #         trainer = distiller.distill()
# # #         logger.info("知识蒸馏测试成功完成!")
# # #         return trainer
# # #     except Exception as e:
# # #         logger.error(f"知识蒸馏测试失败: {str(e)}")
# # #         raise e
# # #
# # #
# # # if __name__ == "__main__":
# # #     # 测试蒸馏
# # #     test_distillation()
# # #
# # #     # 或者使用命令行参数运行
# # #     # args = parse_args()
# # #     # distiller = ModelDistiller(args)
# # #     # distiller.distill()
# # #
# # # # import argparse
# # # # import torch
# # # # import os
# # # # from transformers import (
# # # #     AutoModelForCausalLM,
# # # #     AutoTokenizer,
# # # #     TrainingArguments,
# # # #     Trainer,
# # # #     DataCollatorForLanguageModeling
# # # # )
# # # # from datasets import load_dataset
# # # # import logging
# # # # from typing import Dict, List
# # # # import numpy as np
# # # # from tqdm import tqdm
# # # # import json
# # # # from modelscope import snapshot_download
# # # # from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
# # # #
# # # # # 配置日志
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format="%(asctime)s - %(levelname)s - %(message)s"
# # # # )
# # # # logger = logging.getLogger(__name__)
# # # #
# # # # # 设置国内镜像源
# # # #
# # # # from copy import deepcopy
# # # #
# # # # class DistillationTrainer(Trainer):
# # # #     """自定义Trainer实现知识蒸馏"""
# # # #
# # # #     def __init__(self, teacher_model, distill_config: Dict, **kwargs):
# # # #         super().__init__(**kwargs)
# # # #
# # # #         # 确保教师模型与学生模型在同一设备
# # # #         self.teacher = teacher_model.to(self.args.device)
# # # #         self.teacher.eval()
# # # #
# # # #         # 初始化蒸馏参数
# # # #         self.distill_config = distill_config
# # # #         self.temperature = distill_config.get('temperature', 2.0)
# # # #         self.loss_weights = distill_config.get('loss_weights', [0.7, 0.3])
# # # #         self.strategy = distill_config.get('strategy', 'logits')
# # # #
# # # #         # 添加梯度裁剪保护
# # # #         self.max_grad_norm = distill_config.get('max_grad_norm', 1.0)
# # # #         self.gradient_accumulation_steps = max(1, self.args.gradient_accumulation_steps)
# # # #
# # # #         # 初始化层映射关系
# # # #         self.layer_mapping = self._init_layer_mapping()
# # # #
# # # #     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
# # # #         """
# # # #         计算损失并更新剪枝
# # # #
# # # #         参数:
# # # #             model: 模型实例
# # # #             inputs: 输入数据
# # # #             return_outputs: 是否返回输出
# # # #             **kwargs: 接收所有额外参数以避免版本兼容性问题
# # # #         """
# # # #         # 过滤掉不需要的参数
# # # #         model_inputs = {
# # # #             k: v for k, v in inputs.items()
# # # #             if k not in ['num_items_in_batch', 'other_unexpected_args']
# # # #         }
# # # #
# # # #         # 确保输入在正确的设备上
# # # #         for key in model_inputs:
# # # #             if isinstance(model_inputs[key], torch.Tensor):
# # # #                 model_inputs[key] = model_inputs[key].to(model.device)
# # # #
# # # #         outputs = model(**model_inputs)
# # # #         loss = outputs.loss
# # # #
# # # #         # 定期更新剪枝掩码
# # # #         if (self.pruning_method != 'static' and
# # # #                 self.global_step % self.pruning_config['update_freq'] == 0 and
# # # #                 self.global_step > self.pruning_config['warmup_steps']):
# # # #             self._apply_pruning()
# # # #
# # # #         self.global_step += 1
# # # #
# # # #         # 记录最佳损失
# # # #         if loss < self.best_loss:
# # # #             self.best_loss = loss
# # # #             self.best_model_state = deepcopy(model.state_dict())
# # # #             for key in self.best_model_state:
# # # #                 if isinstance(self.best_model_state[key], torch.Tensor):
# # # #                     self.best_model_state[key] = self.best_model_state[key].cpu()
# # # #
# # # #         return (loss, outputs) if return_outputs else loss
# # # #
# # # #     def training_step(self, model: torch.nn.Module, inputs) -> torch.Tensor:
# # # #         """
# # # #         核心训练步骤，处理前向和反向传播
# # # #         修复：正确获取和使用优化器
# # # #         """
# # # #         # 确保输入数据在正确设备上
# # # #         inputs = {k: v.to(self.args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
# # # #
# # # #         # 获取优化器和学习率调度器
# # # #         optimizer = self.optimizer
# # # #         lr_scheduler = self.lr_scheduler
# # # #
# # # #         # 日志记录
# # # #         logger.debug(f"training_step获取到优化器: {optimizer.__class__.__name__}")
# # # #
# # # #         # 获取损失
# # # #         loss = self.compute_loss(model, inputs)
# # # #
# # # #         # 梯度保护
# # # #         if self.args.gradient_accumulation_steps > 1:
# # # #             loss = loss / self.args.gradient_accumulation_steps
# # # #
# # # #         # 反向传播
# # # #         if self.use_grad_scaler:
# # # #             self.scaler.scale(loss).backward()
# # # #         else:
# # # #             loss.backward()
# # # #
# # # #         # 梯度裁剪和优化器更新
# # # #         if self.args.max_grad_norm > 0:
# # # #             if self.use_grad_scaler:
# # # #                 # 解缩放梯度并进行裁剪
# # # #                 self.scaler.unscale_(optimizer)
# # # #                 torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
# # # #             else:
# # # #                 torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
# # # #
# # # #         # 更新参数
# # # #         optimizer_was_run = True
# # # #         if self.use_grad_scaler:
# # # #             self.scaler.step(optimizer)
# # # #             self.scaler.update()
# # # #         else:
# # # #             optimizer.step()
# # # #
# # # #         # 更新学习率
# # # #         if optimizer_was_run and lr_scheduler is not None:
# # # #             lr_scheduler.step()
# # # #
# # # #         # 重置梯度
# # # #         optimizer.zero_grad()
# # # #
# # # #         return loss.detach()
# # # #
# # # #     @property
# # # #     def use_grad_scaler(self):
# # # #         """检查是否应该使用梯度缩放"""
# # # #         return (
# # # #                 self.args.fp16
# # # #                 and hasattr(self, '_scaler')
# # # #                 and self._scaler is not None
# # # #         )
# # # #
# # # #     def _init_layer_mapping(self) -> Dict[int, int]:
# # # #         """初始化教师-学生层映射关系"""
# # # #         if self.strategy in ['hidden', 'attention']:
# # # #             mapping_type = self.distill_config.get('layer_mapping', 'auto')
# # # #
# # # #             if mapping_type == 'auto':
# # # #                 # 自动均匀映射
# # # #                 teacher_layers = len(self.teacher.model.layers) if hasattr(self.teacher.model, 'layers') else 0
# # # #                 student_layers = len(self.model.model.layers) if hasattr(self.model.model, 'layers') else 0
# # # #
# # # #                 if teacher_layers == 0 or student_layers == 0:
# # # #                     logger.warning("无法自动获取模型层数，使用空层映射")
# # # #                     return {}
# # # #
# # # #                 return {i: int(i * student_layers / teacher_layers)
# # # #                         for i in range(teacher_layers)}
# # # #
# # # #             elif mapping_type == 'manual':
# # # #                 # 手动指定映射
# # # #                 pairs = self.distill_config.get('layer_pairs', '')
# # # #                 if not pairs:
# # # #                     logger.warning("手动层映射配置为空，使用空层映射")
# # # #                     return {}
# # # #                 try:
# # # #                     return dict([map(int, pair.split(':'))
# # # #                                  for pair in pairs.split(',')])
# # # #                 except Exception as e:
# # # #                     logger.error(f"解析手动层映射失败: {str(e)}，使用空层映射")
# # # #                     return {}
# # # #
# # # #         return {}
# # # #
# # # #     def _logits_distillation_loss(self, student_logits, teacher_logits):
# # # #         """计算logits蒸馏损失"""
# # # #         soft_teacher = torch.nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
# # # #         soft_student = torch.nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
# # # #         return torch.nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean')
# # # #
# # # #     def _hidden_distillation_loss(self, student_hiddens, teacher_hiddens):
# # # #         """计算隐藏层蒸馏损失"""
# # # #         loss = 0.0
# # # #         # 确保获取到有效的隐藏层
# # # #         if not hasattr(self.model, 'config') or not self.model.config.is_encoder_decoder:
# # # #             student_hiddens = student_hiddens[1:]  # 跳过embedding层
# # # #             teacher_hiddens = teacher_hiddens[1:]
# # # #
# # # #         for t_layer, s_layer in self.layer_mapping.items():
# # # #             if t_layer < len(teacher_hiddens) and s_layer < len(student_hiddens):
# # # #                 loss += torch.nn.functional.mse_loss(
# # # #                     student_hiddens[s_layer],
# # # #                     teacher_hiddens[t_layer]
# # # #                 )
# # # #         return loss / max(1, len(self.layer_mapping))
# # # #
# # # #     def _attention_distillation_loss(self, student_attns, teacher_attns):
# # # #         """计算注意力矩阵蒸馏损失"""
# # # #         loss = 0.0
# # # #         # 确保获取到有效的注意力层
# # # #         if not hasattr(self.model, 'config') or not self.model.config.is_encoder_decoder:
# # # #             student_attns = student_attns[1:]  # 跳过第一层
# # # #             teacher_attns = teacher_attns[1:]
# # # #
# # # #         for t_layer, s_layer in self.layer_mapping.items():
# # # #             if t_layer < len(teacher_attns) and s_layer < len(student_attns):
# # # #                 s_attn = student_attns[s_layer]  # [batch, heads, seq, seq]
# # # #                 t_attn = teacher_attns[t_layer]
# # # #
# # # #                 # 对齐注意力头维度
# # # #                 if s_attn.size(1) != t_attn.size(1):
# # # #                     t_attn = t_attn.repeat_interleave(s_attn.size(1) // t_attn.size(1), dim=1)
# # # #
# # # #                 loss += torch.nn.functional.mse_loss(s_attn, t_attn)
# # # #         return loss / max(1, len(self.layer_mapping))
# # # #
# # # #
# # # # def load_model(model_path: str, device_map: str = "auto"):
# # # #     """加载模型并分配到指定设备"""
# # # #     logger.info(f"加载模型: {model_path}")
# # # #     return AutoModelForCausalLM.from_pretrained(
# # # #         model_path,
# # # #         device_map=device_map,
# # # #         torch_dtype=torch.float16,
# # # #         trust_remote_code=True
# # # #     )
# # # #
# # # #
# # # # def init_student(student_path: str, teacher=None, init_layers: int = 0):
# # # #     """初始化学生模型"""
# # # #     if teacher and init_layers > 0:
# # # #         logger.info(f"从教师模型初始化前{init_layers}层")
# # # #         student = AutoModelForCausalLM.from_pretrained(
# # # #             student_path,
# # # #             torch_dtype=torch.float16,
# # # #             trust_remote_code=True
# # # #         )
# # # #
# # # #         # 复制教师模型前N层
# # # #         teacher_layers = len(teacher.model.layers)
# # # #         for i in range(min(init_layers, teacher_layers)):
# # # #             if i < len(student.model.layers):
# # # #                 student.model.layers[i].load_state_dict(
# # # #                     teacher.model.layers[i].state_dict()
# # # #                 )
# # # #         return student
# # # #     else:
# # # #         return load_model(student_path)
# # # #
# # # #
# # # # def prepare_dataset(tokenizer, data_path: str, max_length: int = 512):
# # # #     """准备数据集，确保正确将文本转换为模型所需的输入ID"""
# # # #
# # # #     def tokenize_function(examples):
# # # #         # 对文本进行分词，返回输入ID和注意力掩码
# # # #         outputs = tokenizer(
# # # #             examples['text'],
# # # #             truncation=True,
# # # #             max_length=max_length,
# # # #             padding='max_length',
# # # #             return_tensors='np'  # 返回numpy数组便于dataset处理
# # # #         )
# # # #         return outputs
# # # #
# # # #     # 加载原始数据集
# # # #     dataset = load_dataset('json', data_files={'train': data_path})
# # # #
# # # #     # 应用分词函数，num_proc参数可加速处理
# # # #     tokenized_dataset = dataset.map(
# # # #         tokenize_function,
# # # #         batched=True,
# # # #         num_proc=4,  # 使用4个进程并行处理
# # # #         remove_columns=['text']  # 移除原始文本列，避免混淆
# # # #     )
# # # #
# # # #     # 确保数据集格式正确
# # # #     if 'input_ids' not in tokenized_dataset['train'].features:
# # # #         raise ValueError("分词后的数据缺少input_ids字段，请检查tokenizer配置")
# # # #
# # # #     return tokenized_dataset
# # # #
# # # #
# # # # def train(args):
# # # #     # 1. 加载教师模型
# # # #     teacher_device_map = "cpu" if args.teacher_device == "cpu" else "auto"
# # # #     teacher = load_model(args.teacher_path, teacher_device_map)
# # # #
# # # #     # 2. 初始化学生模型
# # # #     student = init_student(
# # # #         args.student_path,
# # # #         teacher=teacher if args.init_strategy == "teacher" else None,
# # # #         init_layers=args.init_layers
# # # #     )
# # # #
# # # #     # 3. 加载分词器
# # # #     tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True)
# # # #     tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
# # # #
# # # #     # 4. 加载并处理数据
# # # #     dataset = prepare_dataset(tokenizer, args.train_data)
# # # #
# # # #     # 5. 准备蒸馏配置
# # # #     distill_config = {
# # # #         'strategy': args.strategy,
# # # #         'temperature': args.temperature,
# # # #         'loss_weights': args.loss_weights,
# # # #         'layer_mapping': args.layer_mapping,
# # # #         'layer_pairs': args.layer_pairs,
# # # #         'max_grad_norm': args.max_grad_norm if hasattr(args, 'max_grad_norm') else 1.0
# # # #     }
# # # #
# # # #     # 6. 训练参数
# # # #     training_args = TrainingArguments(
# # # #         output_dir=args.output_dir,
# # # #         per_device_train_batch_size=args.batch_size,
# # # #         num_train_epochs=args.epochs,
# # # #         learning_rate=args.learning_rate,
# # # #         gradient_accumulation_steps=args.gradient_accum,
# # # #         eval_strategy="steps" if args.val_data else "no",
# # # #         save_strategy=args.save_strategy if hasattr(args, 'save_strategy') else "no",
# # # #         save_steps=args.save_steps if args.save_strategy == "steps" else None,
# # # #         logging_steps=100,
# # # #         fp16=True,  # 使用FP16混合精度训练
# # # #         fp16_full_eval=True,  # 在评估时也使用FP16
# # # #         report_to="none",
# # # #         max_grad_norm=distill_config['max_grad_norm'],
# # # #         remove_unused_columns=False,  # 保留所有列，避免数据丢失
# # # #     )
# # # #
# # # #     # 7. 数据整理器
# # # #     data_collator = DataCollatorForLanguageModeling(
# # # #         tokenizer=tokenizer,
# # # #         mlm=False
# # # #     )
# # # #
# # # #     # 8. 创建Trainer
# # # #     trainer = DistillationTrainer(
# # # #         teacher_model=teacher,
# # # #         distill_config=distill_config,
# # # #         model=student,
# # # #         args=training_args,
# # # #         train_dataset=dataset['train'],
# # # #         eval_dataset=None,
# # # #         data_collator=data_collator,
# # # #         tokenizer=tokenizer
# # # #     )
# # # #
# # # #     # 9. 开始训练
# # # #     logger.info("***** 开始知识蒸馏 *****")
# # # #     trainer.train()
# # # #
# # # #     # 10. 保存最终模型
# # # #     final_path = os.path.join(args.output_dir, "final")
# # # #     trainer.save_model(final_path)
# # # #     logger.info(f"蒸馏完成，模型已保存到: {final_path}")
# # # #
# # # #
# # # # def test_distillation():
# # # #     """使用ModelScope下载模型并测试蒸馏的完整流程"""
# # # #     logger.info("=== 开始知识蒸馏测试 ===")
# # # #
# # # #     # 创建测试目录
# # # #     os.makedirs("test_output", exist_ok=True)
# # # #
# # # #     # 1. 准备测试数据
# # # #     test_data_path = "test_data.json"
# # # #     if not os.path.exists(test_data_path):
# # # #         logger.info("生成测试数据...")
# # # #         test_data = [
# # # #             {"text": "知识蒸馏是一种模型压缩技术。"},
# # # #             {"text": "大语言模型需要大量计算资源。"},
# # # #             {"text": "模型量化可以减少模型大小。"},
# # # #             {"text": "注意力机制是Transformer的核心。"},
# # # #             {"text": "深度学习在NLP领域应用广泛。"},
# # # #             {"text": "中国在AI领域发展迅速。"},
# # # #             {"text": "上海是中国的经济中心之一。"},
# # # #             {"text": "北京有许多高科技公司。"},
# # # #             {"text": "深圳被称为中国的硅谷。"},
# # # #             {"text": "杭州有著名的互联网公司。"}
# # # #         ]
# # # #         with open(test_data_path, "w", encoding="utf-8") as f:
# # # #             for item in test_data:
# # # #                 f.write(json.dumps(item, ensure_ascii=False) + "\n")
# # # #
# # # #     # 2. 设置测试参数
# # # #     class TestArgs:
# # # #         teacher_path = r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen2___5-0___5B-Instruct'
# # # #         student_path = r'E:\workspace\saddlellm\saddlellm\test_models\qwen\Qwen1___5-0___5B'
# # # #         train_data = test_data_path
# # # #         val_data = None
# # # #         output_dir = "./test_output"
# # # #         batch_size = 2
# # # #         epochs = 1
# # # #         learning_rate = 5e-5
# # # #         gradient_accum = 1
# # # #         strategy = "logits"
# # # #         temperature = 2.0
# # # #         loss_weights = [0.7, 0.3]
# # # #         layer_mapping = "auto"
# # # #         layer_pairs = ""
# # # #         init_strategy = "none"
# # # #         init_layers = 0
# # # #         teacher_device = "cpu"  # 测试时使用CPU节省资源
# # # #         save_strategy = "no"
# # # #         save_steps = 1000
# # # #         max_grad_norm = 1.0  # 明确设置梯度范数
# # # #
# # # #     args = TestArgs()
# # # #
# # # #     # 3. 运行蒸馏流程
# # # #     try:
# # # #         logger.info("=== 开始蒸馏测试 ===")
# # # #         train(args)
# # # #         logger.info("=== 蒸馏测试成功完成 ===")
# # # #         return True
# # # #     except Exception as e:
# # # #         logger.error(f"蒸馏测试失败: {str(e)}", exc_info=True)
# # # #         return False
# # # #
# # # #
# # # # if __name__ == "__main__":
# # # #     # 运行测试
# # # #     test_result = test_distillation()
# # # #
# # # #     if test_result:
# # # #         print("✅ 知识蒸馏测试成功完成！")
# # # #     else:
# # # #         print("❌ 知识蒸馏测试失败，请检查日志")
