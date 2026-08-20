import torch
import torch.nn.utils.prune as prune
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, List, Optional, Literal, Union
from collections import defaultdict
from datasets import Dataset
import numpy as np

class ModelPruner:
    """
    Model Pruning Wrapper with sklearn-style API.
    Supports both structured and unstructured pruning.

    Example:
    >>> pruner = ModelPruner(
            model_name="meta-llama/Meta-Llama-3-8B",
            pruning_method="l1_unstructured",
            pruning_ratio=0.3
        )
    >>> pruner.fit(train_dataset)  # 可选：基于训练数据的剪枝
    >>> pruner.prune()  # 执行剪枝
    >>> pruner.save("./pruned_model")
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        pruning_method: Literal[
            "l1_unstructured", 
            "random_unstructured",
            "ln_structured",  # Ln-norm结构化剪枝
            "head_structured",  # 注意力头剪枝
            "layer_structured"  # 整层剪枝
        ] = "l1_unstructured",
        pruning_ratio: float = 0.3,
        global_pruning: bool = True,
        device_map: str = "auto",
    ):
        """
        Initialize model pruner.
        
        :param pruning_method: 
            - "l1_unstructured": 按L1-norm剪枝权重
            - "random_unstructured": 随机剪枝
            - "ln_structured": 按LN-norm剪枝整个通道
            - "head_structured": 剪枝注意力头
            - "layer_structured": 剪枝整个Transformer层
        :param pruning_ratio: 剪枝比例（0.3表示剪掉30%）
        :param global_pruning: 是否全局剪枝（跨所有参数统一阈值）
        """
        self.model_name = model_name
        self.pruning_method = pruning_method
        self.pruning_ratio = pruning_ratio
        self.global_pruning = global_pruning
        
        # 加载模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 存储剪枝掩码
        self.pruning_masks = defaultdict(dict)

    def _get_pruning_parameters(self) -> List[tuple]:
        """获取需要剪枝的参数列表"""
        params = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    params.append((module, "weight"))  # 注意力矩阵优先剪枝
                elif "dense" in name or "fc" in name:
                    params.append((module, "weight"))
        return params

    def _prune_heads(self, head_indices: Dict[str, List[int]]):
        """剪枝注意力头（结构化）"""
        for layer_idx, heads in head_indices.items():
            self.model.prune_heads({int(layer_idx): heads})

    def _prune_layers(self, layer_indices: List[int]):
        """剪枝整个Transformer层（结构化）"""
        if hasattr(self.model.config, "num_hidden_layers"):
            keep_indices = [i for i in range(self.model.config.num_hidden_layers) 
                          if i not in layer_indices]
            self.model.encoder.layer = torch.nn.ModuleList(
                [self.model.encoder.layer[i] for i in keep_indices]
            )
            self.model.config.num_hidden_layers = len(keep_indices)

    def fit(
        self, 
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        epochs: int = 1,
        batch_size: int = 2,
    ):
        """
        基于训练数据计算剪枝重要性（可选）
        :param train_dataset: 用于评估参数重要性的数据
        """
        if train_dataset is None:
            return

        # 数据预处理
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_fn, batched=True)
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # 评估参数重要性（示例：梯度重要性）
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            for batch in train_dataset:
                inputs = self.tokenizer(
                    batch["text"], 
                    return_tensors="pt", 
                    padding=True
                ).to(self.model.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                loss.backward()
                
                # 记录梯度重要性（实际应用可能需要更复杂的评估）
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.pruning_masks[name]["importance"] = (
                            param.grad.abs().mean().item()
                        )
                
                optimizer.zero_grad()

    def prune(self):
        """执行剪枝操作"""
        if self.pruning_method == "head_structured":
            # 示例：剪枝注意力头（需实现head重要性评估）
            head_importance = self._compute_head_importance()  # 伪代码
            head_indices = self._select_heads_to_prune(head_importance)
            self._prune_heads(head_indices)
        elif self.pruning_method == "layer_structured":
            # 示例：剪枝整个层
            layer_importance = self._compute_layer_importance()  # 伪代码
            layer_indices = self._select_layers_to_prune(layer_importance)
            self._prune_layers(layer_indices)
        else:
            # 非结构化或通道剪枝
            params = self._get_pruning_parameters()
            if self.pruning_method == "l1_unstructured":
                prune_method = prune.L1Unstructured
            elif self.pruning_method == "random_unstructured":
                prune_method = prune.RandomUnstructured
            elif self.pruning_method == "ln_structured":
                prune_method = prune.LnStructured
            
            for module, param_name in params:
                if "structured" in self.pruning_method:
                    prune.ln_structured(
                        module,
                        name=param_name,
                        amount=self.pruning_ratio,
                        n=2,  # L2-norm
                        dim=0,  # 剪枝输出通道
                    )
                else:
                    prune.global_unstructured(
                        [(module, param_name)],
                        pruning_method=prune_method,
                        amount=self.pruning_ratio,
                    )
                
                # 保存掩码以便恢复
                mask = getattr(module, f"{param_name}_mask")
                self.pruning_masks[f"{module.__class__.__name__}.{param_name}"] = mask

    def save(self, path: str, remove_masks: bool = False):
        """
        保存剪枝后的模型
        :param remove_masks: 是否永久移除被剪枝的权重（否则只是屏蔽）
        """
        if remove_masks:
            # 永久移除被剪枝的权重
            for module, _ in self._get_pruning_parameters():
                prune.remove(module, "weight")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _compute_head_importance(self) -> Dict[str, float]:
        """计算注意力头重要性（示例）"""
        importance = defaultdict(float)
        for name, param in self.model.named_parameters():
            if "attention.self.query.weight" in name:
                layer_idx = name.split(".")[2]  # 假设格式为model.layers.X.attention...
                # 实际实现需要更精细的重要性评估
                importance[layer_idx] = param.abs().mean().item()
        return importance

    def _compute_layer_importance(self) -> List[float]:
        """计算Transformer层重要性（示例）"""
        return [1.0] * self.model.config.num_hidden_layers  # 伪代码

    @classmethod
    def load(cls, path: str, **kwargs):
        """加载剪枝后的模型"""
        instance = cls(model_name=path, **kwargs)
        return instance