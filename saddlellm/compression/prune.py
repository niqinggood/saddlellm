import argparse
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset, Dataset
import logging
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy
import json
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 本地模型路径
MODEL_PATH = r"E:\reactflow_test\backend\model\Qwen\Qwen2___5-0___5B-Instruct"
DATASET_PATH = r'E:\ml_data\medical_zh\train_zh_1000.jsonl'

class ModelPruner:
    """模型剪枝工具类，优化设备管理和兼容性"""

    @staticmethod
    def get_prunable_layers(model: nn.Module, target_modules: List[str]) -> List[Tuple[str, nn.Module]]:
        """获取可剪枝的层，增加设备兼容性检查"""
        prunable_layers = []
        for name, module in model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prunable_layers.append((name, module))
        logger.info(f"找到 {len(prunable_layers)} 个可剪枝层")
        return prunable_layers

    @staticmethod
    def apply_pruning(
            model: nn.Module,
            pruning_method: str,
            pruning_config: Dict,
            target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    ) -> nn.Module:
        """应用静态剪枝，优化设备内存管理"""
        prunable_layers = ModelPruner.get_prunable_layers(model, target_modules)

        if not prunable_layers:
            logger.warning("未找到可剪枝层，跳过剪枝")
            return model

        if pruning_method == "magnitude":
            for name, module in prunable_layers:
                prune.l1_unstructured(module, name='weight', amount=pruning_config['ratio'])
        elif pruning_method == "random":
            for name, module in prunable_layers:
                prune.random_unstructured(module, name='weight', amount=pruning_config['ratio'])
        elif pruning_method == "structured":
            dim = pruning_config.get('dim', 0)
            for name, module in prunable_layers:
                prune.ln_structured(module, name='weight', amount=pruning_config['ratio'], n=2, dim=dim)
        elif pruning_method == "global":
            parameters_to_prune = [
                (module, 'weight') for name, module in prunable_layers
            ]
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_config['ratio']
            )

        return model

    @staticmethod
    def remove_pruning(model: nn.Module) -> nn.Module:
        """移除剪枝掩码，永久应用剪枝，增加内存清理"""
        for name, module in model.named_modules():
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
        torch.cuda.empty_cache()
        return model

    @staticmethod
    def get_sparsity(model: nn.Module) -> float:
        """计算模型稀疏度，优化计算效率"""
        zero_params = 0
        total_params = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight'):
                    # 只在CPU上计算，避免GPU内存问题
                    weight_cpu = module.weight.cpu()
                    zero_params += torch.sum(weight_cpu == 0).item()
                    total_params += weight_cpu.numel()
        return zero_params / total_params if total_params > 0 else 0.0


class PruningTrainer(Trainer):
    """自定义Trainer实现动态剪枝，优化数据处理流程"""

    def __init__(self, pruning_method: str, pruning_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.pruning_config = pruning_config
        self.pruning_method = pruning_method.lower()
        self.global_step = 0
        self.best_loss = float('inf')
        self.prune_ratio_schedule = self._create_prune_schedule()  # 这里调用
        self.target_modules = None
        self.prunable_layers = []

        # 初始化剪枝
        self._init_pruning()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算损失并更新剪枝

        参数:
            model: 模型实例
            inputs: 输入数据
            return_outputs: 是否返回输出
            **kwargs: 接收所有额外参数以避免版本兼容性问题
        """
        # 过滤掉不需要的参数
        model_inputs = {
            k: v for k, v in inputs.items()
            if k not in ['num_items_in_batch', 'other_unexpected_args']
        }

        # 确保输入在正确的设备上
        for key in model_inputs:
            if isinstance(model_inputs[key], torch.Tensor):
                model_inputs[key] = model_inputs[key].to(model.device)

        outputs = model(**model_inputs)
        loss = outputs.loss

        # 定期更新剪枝掩码
        if (self.pruning_method != 'static' and
                self.global_step % self.pruning_config['update_freq'] == 0 and
                self.global_step > self.pruning_config['warmup_steps']):
            self._apply_pruning()

        self.global_step += 1

        # 记录最佳损失
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_state = deepcopy(model.state_dict())
            for key in self.best_model_state:
                if isinstance(self.best_model_state[key], torch.Tensor):
                    self.best_model_state[key] = self.best_model_state[key].cpu()

        return (loss, outputs) if return_outputs else loss

    def _create_prune_schedule(self) -> List[float]:  # 方法定义
        """创建剪枝比例调度，增加边界检查"""
        total_steps = max(1, self.pruning_config['total_steps'])
        if self.pruning_config['schedule'] == 'linear':
            return np.linspace(0, self.pruning_config['ratio'], total_steps)
        elif self.pruning_config['schedule'] == 'cosine':
            return self.pruning_config['ratio'] * (1 - np.cos(
                np.pi * np.arange(total_steps) / total_steps
            )) / 2
        else:  # constant
            return [self.pruning_config['ratio']] * total_steps

    def _init_pruning(self):
        """初始化剪枝，优化设备管理"""
        self.target_modules = self.pruning_config.get('target_modules',
                                                    ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"])

        # 获取所有可剪枝层
        self.prunable_layers = ModelPruner.get_prunable_layers(self.model, self.target_modules)

        # 初始化动量剪枝的动量缓冲区
        if self.pruning_method == 'movement':
            for name, module in self.prunable_layers:
                if not hasattr(module, 'importance_momentum'):
                    module.register_buffer('importance_momentum', torch.zeros_like(module.weight))

        logger.info(f"初始化剪枝，可剪枝层数: {len(self.prunable_layers)}")

    def _get_current_prune_ratio(self) -> float:
        """获取当前剪枝比例，增加边界处理"""
        total_steps = max(1, self.pruning_config['total_steps'])
        current_step = min(self.global_step, total_steps - 1)
        return self.prune_ratio_schedule[current_step]

    def _apply_pruning(self):
        """应用剪枝策略，优化内存使用"""
        current_ratio = self._get_current_prune_ratio()

        if self.pruning_method == 'magnitude':
            self._magnitude_pruning(current_ratio)
        elif self.pruning_method == 'movement':
            self._movement_pruning(current_ratio)
        elif self.pruning_method == 'structured':
            self._structured_pruning(current_ratio)

        # 清理GPU缓存
        torch.cuda.empty_cache()

    def _magnitude_pruning(self, ratio: float):
        """幅度剪枝，优化循环效率"""
        for name, module in self.prunable_layers:
            prune.l1_unstructured(module, name='weight', amount=ratio)

    def _movement_pruning(self, ratio: float):
        """动态运动剪枝，优化梯度处理"""
        for name, module in self.prunable_layers:
            # 更新重要性分数 (基于梯度动量)
            if module.weight.grad is not None:
                importance_scores = torch.abs(module.weight.grad)
                module.importance_momentum = (
                        self.pruning_config['momentum'] * module.importance_momentum +
                        (1 - self.pruning_config['momentum']) * importance_scores
                )

                # 应用剪枝
                threshold = torch.quantile(
                    module.importance_momentum.flatten(),
                    ratio
                )
                mask = module.importance_momentum > threshold
                if hasattr(module, 'weight_mask'):
                    module.weight_mask.data = mask.to(module.weight_mask.device)
                else:
                    prune.custom_from_mask(module, name='weight', mask=mask)

    def _structured_pruning(self, ratio: float):
        """结构化剪枝，增加维度验证"""
        dim = self.pruning_config.get('dim', 0)
        if dim not in [0, 1]:
            logger.warning(f"无效的结构化剪枝维度 {dim}，使用默认维度 0")
            dim = 0

        for name, module in self.prunable_layers:
            prune.ln_structured(module, name='weight', amount=ratio, n=2, dim=dim)

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     """
    #     完全兼容的损失计算方法
    #     处理所有可能的参数传递情况
    #     """
    #     # 提取真正的模型输入参数
    #     model_inputs = {
    #         k: v for k, v in inputs.items()
    #         if k in ['input_ids', 'attention_mask', 'position_ids', 'token_type_ids']
    #     }
    #
    #     # 确保输入在正确的设备上
    #     model_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
    #                     for k, v in model_inputs.items()}
    #
    #     # 处理不同版本的transformers参数传递
    #     if 'labels' in inputs:
    #         model_inputs['labels'] = inputs['labels'].to(model.device)
    #
    #     outputs = model(**model_inputs)
    #     loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
    #
    #     # 剪枝逻辑
    #     if self.pruning_method != 'static':
    #         if self.global_step % self.pruning_config['update_freq'] == 0:
    #             if self.global_step > self.pruning_config['warmup_steps']:
    #                 self._apply_pruning()
    #         self.global_step += 1
    #
    #     # 记录最佳模型
    #     if loss < self.best_loss:
    #         self.best_loss = loss
    #         self.best_model_state = {
    #             k: v.cpu() if isinstance(v, torch.Tensor) else v
    #             for k, v in model.state_dict().items()
    #         }
    #
    #     return (loss, outputs) if return_outputs else loss

    def evaluate_model(self, eval_dataset=None):
        """评估剪枝模型性能，优化评估流程"""
        original_sparsity = ModelPruner.get_sparsity(self.model)
        results = super().evaluate(eval_dataset=eval_dataset)
        results['sparsity'] = original_sparsity
        return results


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """加载模型和分词器，优化设备分配"""
    logger.info(f"加载模型和分词器: {model_path}")

    # 设置设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("请求CUDA设备但不可用，使用CPU")
        device = "cpu"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    return model, tokenizer


def prepare_dataset(tokenizer, data_path: str, seq_length: int = 512):
    """准备训练数据集，优化数据处理流程"""
    logger.info(f"准备数据集: {data_path}")

    # 加载数据集
    if data_path.endswith(('.json', '.jsonl')):
        try:
            dataset = load_dataset('json', data_files=data_path)
        except:
            # 尝试加载JSONL文件
            dataset = load_dataset('json', data_files={'train': data_path})
    elif data_path.endswith('.txt'):
        dataset = load_dataset('text', data_files=data_path)
    else:
        dataset = load_dataset(data_path) if os.path.isdir(data_path) else load_dataset('json', data_files=data_path)

    # 预处理函数
    def tokenize_function(examples):
        # 确保text字段存在
        if 'text' not in examples:
            # 尝试常见字段
            for field in ['content', 'texts', 'sentence']:
                if field in examples:
                    examples['text'] = examples[field]
                    break
            else:
                # 使用第一个字段作为文本
                examples['text'] = list(examples.values())[0]

        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=seq_length,
            padding="max_length"
        )

    # 应用分词
    try:
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset['train'].features if col != 'text'],
            desc="Tokenizing dataset"
        )
    except Exception as e:
        logger.warning(f"数据集映射失败，尝试替代方法: {str(e)}")
        # 备选方案：手动处理
        all_texts = []
        for split in dataset:
            if 'text' in dataset[split]:
                all_texts.extend(dataset[split]['text'])
            else:
                all_texts.extend(list(dataset[split].values())[0])

        tokenized = tokenizer(
            all_texts,
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors="np"
        )

        # 创建Dataset
        tokenized_dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    return tokenized_dataset


def train(args):
    """执行剪枝训练，优化内存管理和错误处理"""
    set_seed(args.seed)

    # 1. 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return

    # 2. 准备数据集
    try:
        dataset = prepare_dataset(tokenizer, args.train_data, args.seq_length)
    except Exception as e:
        logger.error(f"数据集准备失败: {str(e)}")
        return

    if 'validation' not in dataset:
        dataset = dataset['train'].train_test_split(test_size=0.1)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.eval_data else "no",  # 修改为eval_strategy
        eval_steps=args.eval_steps,
        fp16=args.device == "cuda",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        save_total_limit=2,
        report_to="none"
    )

    # 5. 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 6. 创建Trainer
    pruning_config = {
        'ratio': args.prune_ratio,
        'target_modules': args.target_modules,
        'update_freq': args.update_freq,
        'warmup_steps': args.warmup_steps,
        'momentum': args.momentum,
        'schedule': args.prune_schedule,
        'total_steps': args.total_steps,
        'dim': args.structure_dim
    }
    try:
        trainer = PruningTrainer(
            pruning_method=args.method,
            pruning_config=pruning_config,
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],  # 使用分割出的验证集
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
    except Exception as e:
        logger.error(f"Trainer创建失败: {str(e)}")
        return

    # 7. 训练前评估
    logger.info("=== 训练前评估 ===")
    try:
        eval_results = trainer.evaluate()
        logger.info(f"初始评估结果: {eval_results}")
    except Exception as e:
        logger.warning(f"训练前评估失败: {str(e)}")

    # 8. 执行训练+剪枝
    logger.info("***** 开始剪枝训练 *****")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        return

    # 9. 恢复最佳模型
    if hasattr(trainer, 'best_model_state') and trainer.best_model_state:
        try:
            # 将最佳模型状态加载到当前设备
            for key in trainer.best_model_state:
                if isinstance(trainer.best_model_state[key], torch.Tensor):
                    trainer.best_model_state[key] = trainer.best_model_state[key].to(model.device)
            trainer.model.load_state_dict(trainer.best_model_state)
        except Exception as e:
            logger.warning(f"加载最佳模型状态失败: {str(e)}")

    # 10. 永久应用剪枝
    try:
        ModelPruner.remove_pruning(trainer.model)
    except Exception as e:
        logger.error(f"应用永久剪枝失败: {str(e)}")
        return

    # 11. 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final")
    try:
        os.makedirs(final_model_path, exist_ok=True)
        trainer.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
    except Exception as e:
        logger.error(f"保存模型失败: {str(e)}")
        return

    # 保存剪枝配置和结果
    # 11. 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final")
    try:
        os.makedirs(final_model_path, exist_ok=True)
        trainer.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        # 使用trainer中保存的pruning_config
        pruning_info = {
            'method': trainer.pruning_method,
            'final_sparsity': ModelPruner.get_sparsity(trainer.model),
            'config': trainer.pruning_config  # 从trainer实例获取
        }

        with open(os.path.join(final_model_path, "pruning_info.json"), 'w') as f:
            json.dump(pruning_info, f, indent=2)

    except Exception as e:
        logger.error(f"保存剪枝信息失败: {str(e)}")
        raise  # 重新抛出异常以便调试


def evaluate_pruned_model(model_path: str, eval_data: str, device: str = "auto"):
    """评估剪枝后的模型，优化设备管理和错误处理"""
    logger.info(f"评估剪枝模型: {model_path}")

    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, device)
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return None

    # 准备数据集
    try:
        dataset = prepare_dataset(tokenizer, eval_data)
    except Exception as e:
        logger.error(f"数据集准备失败: {str(e)}")
        return None

    # 计算稀疏度
    try:
        sparsity = ModelPruner.get_sparsity(model)
        logger.info(f"模型稀疏度: {sparsity:.2%}")
    except Exception as e:
        logger.error(f"计算稀疏度失败: {str(e)}")
        return None

    # 评估性能
    try:
        eval_args = TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=8,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=dataset['validation'] if 'validation' in dataset else dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        results = trainer.evaluate()
        logger.info(f"评估结果: {results}")

        return {
            'sparsity': sparsity,
            'eval_results': results
        }
    except Exception as e:
        logger.error(f"评估过程中出错: {str(e)}")
        return None


def parse_args():
    """解析命令行参数，增加参数验证"""
    parser = argparse.ArgumentParser(description="大模型剪枝工具，优化版")

    # 模型参数
    parser.add_argument("--model-path", default=MODEL_PATH, help="原始模型路径")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="计算设备")

    # 数据参数
    parser.add_argument("--train-data", default=DATASET_PATH, help="训练数据路径")
    parser.add_argument("--eval-data", help="评估数据路径")
    parser.add_argument("--seq-length", type=int, default=512, help="序列长度")

    # 剪枝参数
    parser.add_argument("--method", default="magnitude",
                        choices=["magnitude", "movement", "structured", "random"],
                        help="剪枝方法")
    parser.add_argument("--prune-ratio", type=float, default=0.5,
                        help="目标稀疏率(0-1)，建议0.3-0.7")
    parser.add_argument("--target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                        help="剪枝目标模块，Qwen模型常见模块")
    parser.add_argument("--prune-schedule", default="linear",
                        choices=["linear", "cosine", "constant"],
                        help="剪枝比例调度策略")

    # 动态剪枝参数
    parser.add_argument("--momentum", type=float, default=0.99,
                        help="动量系数(动态剪枝)，建议0.9-0.999")
    parser.add_argument("--update-freq", type=int, default=100,
                        help="掩码更新频率(步)，建议50-200")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="预热步数，建议100-1000")
    parser.add_argument("--total-steps", type=int, default=3000,
                        help="总剪枝步数，建议1000-5000")

    # 结构化剪枝参数
    parser.add_argument("--structure-dim", type=int, default=0,
                        choices=[0, 1],
                        help="结构化剪枝维度(0=行,1=列)，Qwen模型建议0")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次，建议1-5")
    parser.add_argument("--batch-size", type=int, default=8, help="批量大小，根据GPU内存调整")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                        help="学习率，建议1e-5-5e-5")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="梯度累积步数，增大可减少内存占用")
    parser.add_argument("--logging-steps", type=int, default=100,
                        help="日志记录步数")
    parser.add_argument("--save-steps", type=int, default=1000,
                        help="模型保存步数")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="评估步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 输出参数
    parser.add_argument("--output-dir", default="./pruned_model",
                        help="输出目录")

    args = parser.parse_args()

    # 参数验证
    if args.prune_ratio < 0 or args.prune_ratio > 1:
        logger.error("--prune-ratio 必须在0到1之间")
        exit(1)

    if args.epochs < 1:
        logger.error("--epochs 必须至少为1")
        exit(1)

    if args.batch_size < 1:
        logger.error("--batch-size 必须至少为1")
        exit(1)

    if not os.path.exists(args.model_path):
        logger.error(f"模型路径不存在: {args.model_path}")
        exit(1)

    if not os.path.exists(args.train_data):
        logger.error(f"训练数据路径不存在: {args.train_data}")
        exit(1)

    return args


def main():
    """主函数，优化错误处理流程"""
    args = parse_args()

    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        return

    # 执行剪枝训练
    train(args)

    # 评估剪枝后的模型
    if args.eval_data:
        evaluate_pruned_model(
            os.path.join(args.output_dir, "final"),
            args.eval_data,
            args.device
        )


if __name__ == "__main__":
    main()
