import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from typing import Optional, Literal
from torch.quantization import quantize_dynamic, prepare_qat, convert

from ..utils.OptionalDependencies import require_distribution

QUANTIZED_STATE_FILENAME = "quantized_model.pt"
QUANTIZATION_MANIFEST_FILENAME = "quantization_manifest.json"

__all__ = [
    "QUANTIZATION_MANIFEST_FILENAME",
    "QUANTIZED_STATE_FILENAME",
    "ModelQuantizer",
]


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
        model_name: Optional[str] = None,
        method: Literal["dynamic", "static", "qat", "bnb"] = "dynamic",
        precision: Literal["int8", "int4", "fp4"] = "int8",
        device_map: str = "auto",
        use_double_quant: bool = True,  # 用于bnb的二次量化
        model=None,
        tokenizer=None,
        trust_remote_code: bool = False,
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
        if method not in {"dynamic", "static", "qat", "bnb"}:
            raise ValueError(f"unsupported quantization method: {method}")
        if precision not in {"int8", "int4", "fp4"}:
            raise ValueError(f"unsupported quantization precision: {precision}")
        self.model_name = model_name
        self.method = method
        self.precision = precision
        self.device_map = device_map
        self.use_double_quant = use_double_quant
        self.trust_remote_code = bool(trust_remote_code)
        self.model = model
        self.tokenizer = tokenizer

        if method == "bnb":
            require_distribution(
                "bitsandbytes",
                extra="train,qlora",
                capability="bitsandbytes quantization",
            )

        # bitsandbytes quantization must happen while loading, so defer that
        # model load until quantize(). Other methods retain the historical
        # eager behavior when an explicit model name is provided.
        if self.model is None and self.model_name and method != "bnb":
            self._load_base_model()

        # 量化配置
        self.quantization_config = None
        if method == "bnb":
            self._setup_bnb_config()

    def _load_base_model(self):
        if not self.model_name:
            raise ValueError("model_name or an in-memory model is required")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model

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

    def quantize(
        self,
        model=None,
        bits: Optional[int] = None,
        method: Optional[str] = None,
        qconfig_spec=None,
        calibration_data=None,
    ):
        """
        执行训练后量化（PTQ）
        :param model: 可选的已加载模型；适用于 CPU dynamic/static 量化
        :param bits: 8 自动选择 dynamic，4 自动选择 bitsandbytes
        :param method: 覆盖构造时的量化方法；`auto` 根据 bits 选择
        :param qconfig_spec: 自定义量化配置（用于static/qat）
        :param calibration_data: 静态量化使用的校准样本迭代器
        """
        if bits not in {None, 4, 8}:
            raise ValueError("bits must be 4 or 8")
        resolved_method = self.method if method in {None, "auto"} else method
        if method in {"int4", "int8"}:
            resolved_method = "bnb" if method == "int4" else "dynamic"
        elif method in {None, "auto"} and bits is not None:
            resolved_method = "bnb" if bits == 4 else "dynamic"
        if resolved_method not in {"dynamic", "static", "qat", "bnb"}:
            raise ValueError(f"unsupported quantization method: {resolved_method}")
        self.method = resolved_method
        if bits == 4:
            self.precision = "int4"
        elif bits == 8:
            self.precision = "int8"

        if self.method == "bnb":
            if model is not None:
                raise ValueError(
                    "bitsandbytes quantization cannot convert an already-loaded model; "
                    "construct ModelQuantizer with model_name instead"
                )
            if not self.model_name:
                raise ValueError("bitsandbytes quantization requires model_name")
            require_distribution(
                "bitsandbytes",
                extra="train,qlora",
                capability="bitsandbytes quantization",
            )
            self._setup_bnb_config()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.quantization_config,
                device_map=self.device_map,
                trust_remote_code=self.trust_remote_code,
            )
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code,
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            return self.model

        if model is not None:
            self.model = model
        if self.model is None:
            self._load_base_model()

        # 动态量化（PTQ）
        if self.method == "dynamic":
            if self.precision != "int8":
                raise ValueError(
                    "PyTorch dynamic quantization supports int8 in this wrapper"
                )
            self.model = quantize_dynamic(
                self.model,
                {torch.nn.Linear},  # 量化所有Linear层
                dtype=torch.qint8,
            )

        elif self.method == "static":
            if calibration_data is None:
                raise ValueError("static quantization requires calibration_data")
            self.model.eval()
            if qconfig_spec is None:
                qconfig_spec = torch.quantization.get_default_qconfig("fbgemm")

            # 准备模型
            self.model.qconfig = qconfig_spec or torch.quantization.get_default_qconfig(
                "fbgemm"
            )
            torch.quantization.prepare(self.model, inplace=True)

            self._calibrate(self.model, calibration_data)

            # 转换为量化模型
            self.model = torch.quantization.convert(self.model, inplace=True)
        elif self.method == "qat":
            raise ValueError(
                "QAT requires fit(train_dataset); quantize() cannot skip training"
            )
        return self.model

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
        if self.model is None or self.tokenizer is None:
            raise ValueError("QAT requires both a model and tokenizer")

        # 准备QAT模型
        self.model.train()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
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
        if self.model is None:
            raise ValueError("no quantized model is available to save")
        output_dir = Path(path)
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError("quantized model path must be a directory")
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.method == "bnb":
            # bnb量化模型直接保存
            self.model.save_pretrained(output_dir)
        else:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "method": self.method,
                    "precision": self.precision,
                    "model_name": self.model_name,
                },
                output_dir / QUANTIZED_STATE_FILENAME,
            )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        manifest = {
            "schema_version": 1,
            "method": self.method,
            "precision": self.precision,
            "source_model": self.model_name,
            "state_file": None if self.method == "bnb" else QUANTIZED_STATE_FILENAME,
        }
        (output_dir / QUANTIZATION_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(output_dir)

    @classmethod
    def load(cls, path: str, method: Optional[str] = None, **kwargs):
        """加载量化模型"""
        output_dir = Path(path)
        manifest_path = output_dir / QUANTIZATION_MANIFEST_FILENAME
        manifest = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        resolved_method = method or manifest.get("method")
        if resolved_method not in {"dynamic", "static", "qat", "bnb"}:
            raise ValueError("quantization method is missing or unsupported")
        precision = kwargs.pop("precision", manifest.get("precision", "int8"))
        if resolved_method == "bnb":
            instance = cls(
                model_name=str(output_dir),
                method="bnb",
                precision=precision,
                **kwargs,
            )
            instance.quantize()
            return instance
        if resolved_method != "dynamic":
            raise NotImplementedError(
                "loading static/QAT checkpoints is not supported safely; retrain or export "
                "the converted model with a backend-specific format"
            )
        state_path = output_dir / manifest.get("state_file", QUANTIZED_STATE_FILENAME)
        if not state_path.is_file():
            raise FileNotFoundError(f"quantized state file is missing: {state_path}")
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=True)
        source_model = kwargs.pop("model_name", None) or checkpoint.get("model_name")
        supplied_model = kwargs.get("model")
        if not source_model and supplied_model is None:
            raise ValueError(
                "dynamic checkpoint does not record a source model; pass model=... to load"
            )
        instance = cls(
            model_name=source_model,
            method="dynamic",
            precision=precision,
            **kwargs,
        )
        instance.quantize()
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        return instance

    def _calibrate(self, model, calibration_data):
        """静态量化校准（示例）"""
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                inputs = self.tokenizer(
                    batch["text"], return_tensors="pt", padding=True
                ).to(model.device)
                model(**inputs)
