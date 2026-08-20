import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from typing import Optional, Literal, Dict
from torch.quantization import quantize_dynamic, prepare_qat, convert
import bitsandbytes as bnb

class ModelQuantizer:
    """
    Model Quantization Wrapper with sklearn-style API.
    Supports PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training).

    Example:
    >>> quantizer = ModelQuantizer(
            model_name="meta-llama/Meta-Llama-3-8B",
            method="dynamic",  # or "qat", "bnb"
            precision="int8"   # or "int4", "fp4"
        )
    >>> quantizer.quantize()  # PTQ
    >>> # 或 QAT:
    >>> quantizer.fit(train_dataset)
    >>> quantizer.save("./quantized_model")
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B",
        method: Literal["dynamic", "static", "qat", "bnb"] = "dynamic",
        precision: Literal["int8", "int4", "fp4"] = "int8",
        device_map: str = "auto",
        use_double_quant: bool = True,  # 用于bnb的二次量化
    ):
        """
        Initialize quantizer.
        
        :param method: 
            - "dynamic": 动态量化（PTQ）
            - "static": 静态量化（PTQ）
            - "qat": 量化感知训练
            - "bnb": bitsandbytes量化
        :param precision: 量化精度（bnb支持int4/fp4）
        """
        self.model_name = model_name
        self.method = method
        self.precision = precision
        self.use_double_quant = use_double_quant
        
        # 加载原始模型和tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量化配置
        self.quantization_config = None
        if method == "bnb":
            self._setup_bnb_config()

    def _setup_bnb_config(self):
        """配置bitsandbytes量化"""
        if self.precision == "int8":
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif self.precision == "int4":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.precision == "fp4":
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
            )

    def quantize(self, qconfig_spec=None):
        """
        执行训练后量化（PTQ）
        :param qconfig_spec: 自定义量化配置（用于static/qat）
        """
        if self.method == "bnb":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map="auto",
            )
            return
        
        # 动态量化（PTQ）
        if self.method == "dynamic":
            self.model = quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # 量化所有Linear层
                dtype=torch.qint8,
            )
        
        # 静态量化/QAT需要更多步骤
        elif self.method in ["static", "qat"]:
            self.model.eval()
            if qconfig_spec is None:
                qconfig_spec = {
                    torch.nn.Linear: torch.quantization.default_qconfig
                }
            
            # 准备模型
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # 静态量化需要校准数据
            if self.method == "static":
                # 伪代码：需要实际校准数据
                self._calibrate(self.model, calibration_data)
            
            # 转换为量化模型
            self.model = torch.quantization.convert(self.model, inplace=True)

    def fit(
        self,
        train_dataset: Dataset,
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
    ):
        """
        量化感知训练（QAT）
        """
        if self.method != "qat":
            raise ValueError("QAT requires method='qat'")
        
        # 准备QAT模型
        self.model.train()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.model = prepare_qat(self.model, inplace=True)
        
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
        
        # 训练参数
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            output_dir="./qat_output",
            save_strategy="no",
            logging_steps=10,
            bf16=torch.cuda.is_bf16_supported(),
        )
        
        # 训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()
        
        # 转换为最终量化模型
        self.model = convert(self.model, inplace=True)

    def save(self, path: str):
        """保存量化模型"""
        if self.method == "bnb":
            # bnb量化模型直接保存
            self.model.save_pretrained(path)
        else:
            # PTQ/QAT模型需要保存量化状态
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'quantization_config': self.model.qconfig,
            }, path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, method: str, **kwargs):
        """加载量化模型"""
        if method == "bnb":
            # bnb量化模型直接加载
            instance = cls(model_name=path, method="bnb", **kwargs)
        else:
            # 加载PTQ/QAT模型
            instance = cls(model_name=path, method=method, **kwargs)
            checkpoint = torch.load(path)
            instance.model.load_state_dict(checkpoint['model_state_dict'])
            instance.model.qconfig = checkpoint['quantization_config']
        return instance

    def _calibrate(self, model, calibration_data):
        """静态量化校准（示例）"""
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                inputs = self.tokenizer(
                    batch["text"], 
                    return_tensors="pt", 
                    padding=True
                ).to(model.device)
                model(**inputs)