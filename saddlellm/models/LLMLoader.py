import os
import torch
import warnings
import gc
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    PretrainedConfig
)


@dataclass
class LLMConfig:
    """大模型配置数据类"""
    model_type: str
    model_path: str
    device: str = "auto"
    quantization: str = "none"
    mixed_precision: str = "fp16"
    memory_optimization: str = "balanced"
    use_flash_attention: bool = True
    max_memory: Optional[int] = None
    low_memory_mode: bool = False
    trust_remote_code: bool = True
    local_files_only: bool = False
    torch_dtype: Optional[str] = None

    # 多GPU配置
    multi_gpu: str = "single"
    gpu_ids: Optional[str] = None

    # 升腾专用配置
    ascend_precision_mode: str = "force_fp16"
    ascend_graph_type: str = "static"
    ascend_device_id: int = 0
    use_ascend_optimize: bool = True


class LLMLoader:
    """大模型加载器，支持多种硬件后端"""

    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        """
        初始化加载器

        参数:
            config: 配置字典或LLMConfig对象
        """
        if isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.is_ascend = self.config.device == "ascend"

        # 初始化设备
        self._init_device()

    def _init_device(self):
        """初始化设备设置"""
        if self.config.device == "auto":
            self.config.device = self._auto_select_device()

        if self.is_ascend:
            self._init_ascend_environment()

    def architecture_support(self) -> Dict[str, Any]:
        """检测当前模型路径对应的架构支持级别。"""
        from .Architecture import ArchitectureRegistry

        return ArchitectureRegistry.detect_from_hf_config(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        ).to_dict()

    def _auto_select_device(self) -> str:
        """自动选择最佳设备"""
        device_priority = [
            ("cuda", self._check_cuda_available),
            ("ascend", self._check_ascend_available),
            ("mps", self._check_mps_available),
            ("xpu", self._check_xpu_available),
            ("rocm", self._check_rocm_available),
        ]

        for device, check_func in device_priority:
            if check_func():
                return device

        return "cpu"

    def _check_cuda_available(self) -> bool:
        """检查CUDA是否可用"""
        return torch.cuda.is_available()

    def _check_ascend_available(self) -> bool:
        """检查升腾设备是否可用"""
        try:
            import acl
            return acl.rt.get_device_count() > 0
        except:
            return False

    def _check_mps_available(self) -> bool:
        """检查Apple MPS是否可用"""
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def _check_xpu_available(self) -> bool:
        """检查Intel XPU是否可用"""
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def _check_rocm_available(self) -> bool:
        """检查AMD ROCm是否可用"""
        return torch.version.hip is not None

    def _init_ascend_environment(self):
        """初始化升腾环境"""
        try:
            import acl

            # 设置升腾环境变量
            os.environ["ASCEND_DEVICE_ID"] = str(self.config.ascend_device_id)
            os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
            os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

            # 检查CANN环境
            if not os.environ.get("ASCEND_HOME"):
                warnings.warn("ASCEND_HOME environment variable not set. Ascend NPU may not work properly.")

        except ImportError as e:
            raise RuntimeError("Required Ascend libraries not found. Please install CANN toolkit.") from e

    def _get_compute_dtype(self) -> torch.dtype:
        """获取计算数据类型"""
        if self.config.torch_dtype:
            explicit = getattr(torch, self.config.torch_dtype, None)
            if explicit is not None:
                return explicit

        if self.config.device in ("cpu", "mps"):
            return torch.float32

        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "tf32": torch.float32,  # TF32通过CUDA设置处理
        }

        dtype = precision_map.get(self.config.mixed_precision, torch.float16)

        # 特殊处理TF32
        if self.config.mixed_precision == "tf32":
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        return dtype

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """获取量化配置"""
        if self.config.quantization == "none":
            return None
        if self.config.device != "cuda":
            warnings.warn("bitsandbytes quantization only works reliably on CUDA; disabling quantization.")
            return None

        return BitsAndBytesConfig(
            load_in_4bit=self.config.quantization == "4bit",
            load_in_8bit=self.config.quantization == "8bit",
            bnb_4bit_compute_dtype=self._get_compute_dtype(),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def _get_device_map(self) -> Optional[Dict[str, Any]]:
        """获取设备映射(用于多GPU)"""
        if self.config.device == "cuda" and self.config.multi_gpu == "auto":
            return "auto"
        if self.config.multi_gpu == "single":
            return None

        gpu_ids = self._parse_gpu_ids()
        if not gpu_ids:
            return None

        if self.config.multi_gpu == "data_parallel":
            return {"": gpu_ids[0]}

        # 简化的模型并行实现
        # 实际应用中需要根据具体模型结构调整
        config = PretrainedConfig.from_pretrained(self.config.model_path)
        num_hidden_layers = getattr(config, "num_hidden_layers", 32)

        device_map = {}
        layers_per_gpu = num_hidden_layers // len(gpu_ids)

        for i, gpu_id in enumerate(gpu_ids):
            start_layer = i * layers_per_gpu
            end_layer = (i + 1) * layers_per_gpu if i != len(gpu_ids) - 1 else num_hidden_layers

            device_map.update({
                f"model.layers.{layer}": gpu_id
                for layer in range(start_layer, end_layer)
            })

        # 其他组件放在第一个GPU上
        device_map.update({
            "model.embed_tokens": gpu_ids[0],
            "model.norm": gpu_ids[0],
            "lm_head": gpu_ids[0]
        })

        return device_map

    def _parse_gpu_ids(self) -> List[int]:
        """解析GPU ID字符串"""
        if not self.config.gpu_ids:
            return []

        try:
            return [int(id_str.strip()) for id_str in self.config.gpu_ids.split(",")]
        except Exception as e:
            warnings.warn(f"Failed to parse GPU IDs: {e}")
            return []

    def _get_max_memory(self) -> Optional[Dict[int, str]]:
        """获取最大内存配置"""
        if not self.config.max_memory or self.config.max_memory <= 0:
            return None

        if self.is_ascend:
            return None  # 升腾目前不支持

        gpu_ids = self._parse_gpu_ids() or [0]
        return {gpu_id: f"{self.config.max_memory}MB" for gpu_id in gpu_ids}

    def _standard_device_map_arg(self, device_map):
        if device_map:
            return device_map
        if self.config.device == "cuda" and self.config.multi_gpu != "single":
            return "auto"
        return None

    def _pipeline_device_arg(self, device_map):
        if device_map:
            return None
        if self.config.device == "cuda":
            return 0
        if self.config.device == "mps":
            return "mps"
        return -1

    def _supports_flash_attention(self) -> bool:
        if not self.config.use_flash_attention or self.config.device != "cuda":
            return False
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

    def _load_ascend_model(self):
        """加载升腾模型"""
        try:
            from ais_bench.infer.interface import InferSession

            # 初始化升腾推理会话
            self.model = InferSession(
                self.config.ascend_device_id,
                self.config.model_path
            )

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load Ascend model: {e}")

    def _load_standard_model(self):
        """加载标准模型(HuggingFace格式)"""
        quantization_config = self._get_quantization_config()
        device_map = self._get_device_map()
        max_memory = self._get_max_memory()
        device_map_arg = self._standard_device_map_arg(device_map)

        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": self._get_compute_dtype(),
            "max_memory": max_memory,
            "low_cpu_mem_usage": self.config.low_memory_mode or self.config.device == "cuda",
            "trust_remote_code": self.config.trust_remote_code,
            "local_files_only": self.config.local_files_only,
        }
        if device_map_arg is not None:
            model_kwargs["device_map"] = device_map_arg
        if self._supports_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        if device_map_arg is None:
            self.model = self.model.to(self.config.device)

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 创建pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self._pipeline_device_arg(device_map_arg)
        )

    def load_model(self):
        """加载模型和tokenizer"""
        if self.is_ascend:
            self._load_ascend_model()
        else:
            self._load_standard_model()
        return self.model, self.tokenizer

    def _generate_with_ascend(self, prompt: str, **kwargs) -> str:
        """使用升腾生成文本"""
        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="np")

            # 设置生成参数
            generate_kwargs = {
                "max_length": kwargs.get("max_new_tokens", 512) + len(input_ids[0]),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": kwargs.get("do_sample", True)
            }

            # 执行推理
            outputs = self.model.infer([input_ids], **generate_kwargs)

            # 解码输出
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            raise RuntimeError(f"Ascend generation failed: {e}")

    def _generate_standard(self, prompt: str, **kwargs) -> str:
        """使用标准模型生成文本"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        model_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(model_device)
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": kwargs.get("do_sample", True),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        for key in ("repetition_penalty", "no_repeat_ngram_size", "top_k"):
            if key in kwargs:
                generation_kwargs[key] = kwargs[key]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated[len(prompt):].strip() if generated.startswith(prompt) else generated

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本

        参数:
            prompt: 输入提示
            **kwargs: 生成参数
                - max_new_tokens: 最大新token数
                - temperature: 温度参数
                - top_p: 核采样参数
                - do_sample: 是否采样

        返回:
            生成的文本
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.is_ascend:
            return self._generate_with_ascend(prompt, **kwargs)
        else:
            return self._generate_standard(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """使用 chat_template 进行多轮消息生成。"""
        if not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
            prompt += "\nassistant:"
        return self.generate(prompt, **kwargs)

    def unload(self):
        """释放模型资源。"""
        self.pipeline = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "LLMLoader":
        config = LLMConfig(
            model_type=kwargs.pop("model_type", "causal_lm"),
            model_path=model_path,
            **kwargs,
        )
        loader = cls(config)
        loader.load_model()
        return loader

    def convert_to_ascend_om(self, output_path: str):
        """将模型转换为升腾OM格式

        参数:
            output_path: 输出OM模型路径
        """
        if self.is_ascend:
            warnings.warn("Model is already for Ascend NPU")
            return

        try:
            from ais_bench.tools.ais_bench_utils import ModelConvert

            convert = ModelConvert(
                framework="pytorch",
                model=self.config.model_path,
                output=output_path,
                precision_mode=self.config.ascend_precision_mode,
                device_id=self.config.ascend_device_id,
                dynamic_batch_size=self.config.ascend_graph_type == "dynamic"
            )

            convert.execute()
            print(f"Model successfully converted to OM format at {output_path}")

        except Exception as e:
            raise RuntimeError(f"Model conversion failed: {e}")


def main():
    """测试案例"""

    # 案例1: 升腾NPU配置
    ascend_config = {
        "model_type": "llama3",
        "model_path": "./models/llama3-8b-ascend",
        "device": "ascend",
        "ascend_precision_mode": "force_fp16",
        "ascend_graph_type": "static",
        "ascend_device_id": 0,
        "use_ascend_optimize": True,
        "mixed_precision": "fp16",
        "max_memory": 10240
    }

    # 案例2: 多GPU配置
    multi_gpu_config = {
        "model_type": "llama3",
        "model_path": "./models/llama3-8b",
        "device": "cuda",
        "quantization": "4bit",
        "multi_gpu": "data_parallel",
        "gpu_ids": "0,1",
        "mixed_precision": "bf16",
        "use_flash_attention": True,
        "max_memory": 10240
    }

    # 选择要测试的配置
    test_config = ascend_config  # 或 multi_gpu_config

    print("加载模型配置:", test_config)
    loader = LLMLoader(test_config)

    print("开始加载模型...")
    loader.load_model()
    print("模型加载完成!")

    # 测试生成
    prompt = "请解释一下人工智能的未来发展趋势"
    print(f"\n生成文本 (提示: '{prompt}'):")
    result = loader.generate(prompt, max_new_tokens=200)
    print(result)

    # 如果需要转换为升腾OM格式
    if test_config.get("device") != "ascend":
        loader.convert_to_ascend_om("./converted_model.om")


if __name__ == "__main__":
    main()
