"""
快速蒸馏系统 — 调 API 直接蒸，不下载模型

核心理念: 不下载大模型, 直接调 API, 多家混蒸, 各取所长

教师擅长矩阵:
  GPT-4o    : 综合最强, 推理/创意/写作
  GPT-4o-mini: 性价比高, 简单问答
  Claude    : 安全性, 长上下文, 有帮助
  DeepSeek  : 数学推理 (便宜), 代码
  Gemini    : 多模态, 翻译
  Qwen-Max  : 中文最强

智能路由策略:
  数学题    → DeepSeek (便宜 + math 强)
  安全问题  → Claude (安全性强)
  创意写作  → GPT-4o (最佳)
  代码      → DeepSeek (便宜 + code 强)
  中文问题  → Qwen-Max (中文最强)
  一般问题  → GPT-4o-mini (最便宜)

用法:
  from saddlellm import RapidDistill

  # 一行代码: 从多个 API 蒸馏到你的模型
  RapidDistill.run(
      student_model=my_model,
      student_tokenizer=my_tokenizer,
      total_examples=500,
      use_teachers=["deepseek", "claude", "gpt4o-mini"],
  )
"""
import os
import re
import time
import json
import random
import logging
import threading
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 教师画像: 每个 API 的擅长领域、成本、速度
# ═══════════════════════════════════════════════════════════

@dataclass
class TeacherProfile:
    name: str                    # teacher名称
    provider: str                # openai | anthropic | deepseek | aliyun | google
    model_id: str                # API model ID
    strengths: List[str]         # 擅长领域
    weaknesses: List[str]        # 不擅长领域
    cost_per_1k_input: float     # $/1K input tokens
    cost_per_1k_output: float    # $/1K output tokens
    max_tokens: int = 4096
    rpm: int = 60                # requests per minute
    reliability: float = 0.99    # 成功率
    notes: str = ""


TEACHER_PROFILES: Dict[str, TeacherProfile] = {
    "gpt4o": TeacherProfile(
        name="GPT-4o",
        provider="openai", model_id="gpt-4o",
        strengths=["reasoning", "creative", "writing", "analysis", "code_review"],
        weaknesses=["math_raw", "translation"],
        cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
        max_tokens=4096, rpm=500,
        notes="综合最强, 贵但值。推理/写作/分析类用。",
    ),
    "gpt4o-mini": TeacherProfile(
        name="GPT-4o-mini",
        provider="openai", model_id="gpt-4o-mini",
        strengths=["qa", "summarize", "simple_tasks"],
        weaknesses=["deep_reasoning", "complex_math"],
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
        max_tokens=4096, rpm=500,
        notes="性价比最高。简单问答/总结/分类用。",
    ),
    "claude": TeacherProfile(
        name="Claude Sonnet 4.6",
        provider="anthropic", model_id="claude-sonnet-4-6",
        strengths=["safety", "helpfulness", "long_context", "honesty"],
        weaknesses=["raw_math_speed", "code_execution"],
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
        max_tokens=4096, rpm=50,
        notes="安全性和帮助性最强。长文本/伦理/对话用。",
    ),
    "deepseek": TeacherProfile(
        name="DeepSeek-Chat",
        provider="deepseek", model_id="deepseek-chat",
        strengths=["math", "code", "reasoning"],
        weaknesses=["creative", "safety_sensitive"],
        cost_per_1k_input=0.00014, cost_per_1k_output=0.00028,
        max_tokens=4096, rpm=60,
        notes="数学推理性价比超高。代码/数学必用。便宜!",
    ),
    "deepseek-reasoner": TeacherProfile(
        name="DeepSeek-Reasoner (R1)",
        provider="deepseek", model_id="deepseek-reasoner",
        strengths=["deep_reasoning", "math", "complex_problem"],
        weaknesses=["simple_qa", "chat"],
        cost_per_1k_input=0.00055, cost_per_1k_output=0.00219,
        max_tokens=8192, rpm=60,
        notes="推理最强, 价格低。难题/数学竞赛用。",
    ),
    "qwen-max": TeacherProfile(
        name="Qwen-Max",
        provider="aliyun", model_id="qwen-max",
        strengths=["chinese", "translation", "knowledge"],
        weaknesses=["english_creative"],
        cost_per_1k_input=0.0005, cost_per_1k_output=0.001,
        max_tokens=4096, rpm=60,
        notes="中文最强。中文内容/翻译/知识用。",
    ),
}

# ═══════════════════════════════════════════════════════════
# 可下载的小模型教师 — 几B 级别, 直接下载, 免费
# ═══════════════════════════════════════════════════════════

@dataclass
class DownloadableTeacher:
    name: str                    # 简称
    full_name: str               # 全称
    hf_path: str                 # HuggingFace 路径
    params: str                  # "0.5B" | "1.5B" | "3B"
    size_gb: float               # 下载大小 (GB)
    vram_gb: float               # 推理需要显存 (bf16)
    strengths: List[str]         # 擅长
    best_for_student: str        # 最适合蒸给多大的学生
    notes: str = ""


DOWNLOADABLE_TEACHERS: Dict[str, DownloadableTeacher] = {
    # ──── 0.5B 级别教师 ────
    "qwen2.5-0.5b": DownloadableTeacher(
        name="qwen2.5-0.5b",
        full_name="Qwen2.5 0.5B",
        hf_path="Qwen/Qwen2.5-0.5B",
        params="0.5B", size_gb=1.0, vram_gb=1.5,
        strengths=["chinese", "general", "qa"],
        best_for_student="10M-50M",
        notes="最轻量教师。适合蒸馏给超小模型(10-50M)。中文好。",
    ),
    "smollm2-135m": DownloadableTeacher(
        name="smollm2-135m",
        full_name="SmolLM2 135M",
        hf_path="HuggingFaceTB/SmolLM2-135M",
        params="135M", size_gb=0.3, vram_gb=0.5,
        strengths=["general", "simple_qa", "fast"],
        best_for_student="10M-100M",
        notes="极小但效果不错。135M 参数, 推理极快。",
    ),

    # ──── 1-2B 级别教师 (最实用) ────
    "qwen2.5-1.5b": DownloadableTeacher(
        name="qwen2.5-1.5b",
        full_name="Qwen2.5 1.5B",
        hf_path="Qwen/Qwen2.5-1.5B",
        params="1.5B", size_gb=2.8, vram_gb=3.5,
        strengths=["chinese", "general", "knowledge", "writing"],
        best_for_student="50M-300M",
        notes="中文小模型最强之一。下载快, 能力不错。最推荐!",
    ),
    "smollm2-1.7b": DownloadableTeacher(
        name="smollm2-1.7b",
        full_name="SmolLM2 1.7B",
        hf_path="HuggingFaceTB/SmolLM2-1.7B",
        params="1.7B", size_gb=3.2, vram_gb=4.0,
        strengths=["general", "reasoning", "qa"],
        best_for_student="100M-500M",
        notes="HuggingFace 出品, 英文好。小模型教学的标杆。",
    ),
    "tinyllama-1.1b": DownloadableTeacher(
        name="tinyllama-1.1b",
        full_name="TinyLlama 1.1B",
        hf_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        params="1.1B", size_gb=2.0, vram_gb=2.5,
        strengths=["general", "chat", "fast"],
        best_for_student="50M-300M",
        notes="1.1B Chat 模型, 对话能力好。适合蒸馏聊天能力。",
    ),
    "deepseek-coder-1.3b": DownloadableTeacher(
        name="deepseek-coder-1.3b",
        full_name="DeepSeek-Coder 1.3B",
        hf_path="deepseek-ai/deepseek-coder-1.3b-base",
        params="1.3B", size_gb=2.5, vram_gb=3.0,
        strengths=["code", "reasoning"],
        best_for_student="100M-500M",
        notes="代码专用小模型。蒸馏代码能力的最佳教师。",
    ),

    # ──── 3B 级别教师 ────
    "qwen2.5-3b": DownloadableTeacher(
        name="qwen2.5-3b",
        full_name="Qwen2.5 3B",
        hf_path="Qwen/Qwen2.5-3B",
        params="3B", size_gb=5.5, vram_gb=7,
        strengths=["chinese", "general", "knowledge", "reasoning"],
        best_for_student="300M-1B",
        notes="中文能力很强。适合蒸馏给 300M-1B 学生。",
    ),
    "smollm2-360m": DownloadableTeacher(
        name="smollm2-360m",
        full_name="SmolLM2 360M",
        hf_path="HuggingFaceTB/SmolLM2-360M",
        params="360M", size_gb=0.7, vram_gb=1.0,
        strengths=["general", "fast"],
        best_for_student="10M-150M",
        notes="360M, 小而美。适合给 50M 左右学生当教师。",
    ),

    # ──── Phi 系列 (小模型天花板) ────
    "phi-2": DownloadableTeacher(
        name="phi-2",
        full_name="Phi-2 2.7B",
        hf_path="microsoft/phi-2",
        params="2.7B", size_gb=5.0, vram_gb=6,
        strengths=["reasoning", "knowledge", "code"],
        best_for_student="300M-1B",
        notes="微软出品, 小模型推理天花板。教科书级训练数据。",
    ),
}


# ═══════════════════════════════════════════════════════════
# 下载管理器: 自动拉取小模型
# ═══════════════════════════════════════════════════════════

class ModelDownloader:
    """小模型下载 & 加载。"""

    @staticmethod
    def download(name: str, cache_dir: str = None):
        """下载一个小模型教师。"""
        if name not in DOWNLOADABLE_TEACHERS:
            available = ", ".join(DOWNLOADABLE_TEACHERS.keys())
            raise KeyError(f"未知模型: {name}. 可用: {available}")

        info = DOWNLOADABLE_TEACHERS[name]
        print(f"下载 {info.full_name} ({info.params}, {info.size_gb:.1f}GB)...")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            info.hf_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(info.hf_path, cache_dir=cache_dir)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  ✓ 加载完成 (VRAM: {info.vram_gb:.1f}GB)")
        return model, tokenizer, info

    @staticmethod
    def list_available():
        """列出所有可下载的教师。"""
        print("\n可下载的小模型教师 (免费, 不调API)\n")
        print(f"{'名称':<22} {'参数':>6} {'大小':>6} {'显存':>6} {'适合学生':>12}")
        print("-" * 60)
        for name, info in DOWNLOADABLE_TEACHERS.items():
            print(f"  {name:<20} {info.params:>6} {info.size_gb:>4.1f}GB "
                  f"{info.vram_gb:>4.1f}GB {info.best_for_student:>12}")
        print()

    @staticmethod
    def recommend_for(student_params: int = None, student_size: str = None):
        """根据学生模型大小推荐最佳教师。"""
        if student_size:
            size_map = {"10m": 1e7, "50m": 5e7, "100m": 1e8, "300m": 3e8,
                        "500m": 5e8, "1b": 1e9}
            student_params = size_map.get(student_size.lower(), 1e8)

        if student_params is None:
            student_params = 1e8

        recommendations = []
        for name, info in DOWNLOADABLE_TEACHERS.items():
            # 教师至少比学生大 3-5x
            min_param, max_param = 1e7, 1e9
            if "100M" in info.best_for_student:
                min_param, max_param = 5e7, 3e8
            elif "300M" in info.best_for_student:
                min_param, max_param = 1e8, 1e9

            if min_param <= student_params <= max_param:
                recommendations.append((name, info))

        recommendations.sort(key=lambda x: float(x[1].params.replace("B", "000").replace("M", "").replace("G", "")))
        return recommendations


# ============================================================
# 本地小模型教师接口
# ============================================================

class LocalTeacherPool:
    """管理多个下载的本地小模型教师, 统一调用。"""

    def __init__(self, cache_dir: str = None):
        self.models = {}       # name → (model, tokenizer, info)
        self.cache_dir = cache_dir

    def add(self, name: str):
        """下载并添加一个教师。"""
        model, tokenizer, info = ModelDownloader.download(name, self.cache_dir)
        self.models[name] = (model, tokenizer, info)
        return self

    def add_all(self, names: List[str] = None):
        """批量添加教师。"""
        names = names or list(DOWNLOADABLE_TEACHERS.keys())[:3]  # 默认前3个
        for name in names:
            try:
                self.add(name)
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        return self

    def generate(self, teacher_name: str, prompt: str, max_tokens: int = 1024) -> str:
        """用指定教师生成回复。"""
        import torch
        model, tokenizer, info = self.models[teacher_name]
        model.eval()
        device = next(model.parameters()).device

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        return full[prompt_len:].strip()

    def batch_generate(self, teacher_name: str, prompts: List[str], max_tokens: int = 1024) -> List[str]:
        return [self.generate(teacher_name, p, max_tokens) for p in prompts]

    def remove(self, name: str):
        """释放模型显存。"""
        if name in self.models:
            del self.models[name]
            import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def list_loaded(self):
        print(f"\n已加载的本地教师 ({len(self.models)}):")
        for name, (model, tokenizer, info) in self.models.items():
            print(f"  {name:<22} {info.params:>6} {info.strengths}")


# ═══════════════════════════════════════════════════════════
# 快速本地蒸馏: 下载 → 蒸 → 释放
# ═══════════════════════════════════════════════════════════

def quick_local_distill(
    student_model,
    student_tokenizer,
    teacher_names: List[str] = None,
    prompts: List[str] = None,
    total_examples: int = 300,
    output_dir: str = "./local_distilled_model",
    auto_cleanup: bool = True,
):
    """
    快速本地蒸馏 — 下载小模型教师 → 蒸 → 释放显存。

    不需要任何 API key! 完全免费!

    用法:
        from saddlellm.distillation.RapidDistill import quick_local_distill

        # 自动选教师, 自动生成 prompts
        quick_local_distill(student_model, student_tokenizer)

        # 指定教师
        quick_local_distill(
            student_model, student_tokenizer,
            teacher_names=["qwen2.5-1.5b", "smollm2-360m"],
            total_examples=500,
        )
    """
    import torch

    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"=== 本地蒸馏: 学生 {student_params/1e6:.0f}M params ===")

    # 自动选教师
    if teacher_names is None:
        recs = ModelDownloader.recommend_for(student_params=student_params)
        teacher_names = [r[0] for r in recs[:2]]  # 取最佳的 2 个
        if not teacher_names:
            teacher_names = ["qwen2.5-1.5b"]  # fallback
        print(f"自动选择教师: {teacher_names}")

    # 加载教师
    pool = LocalTeacherPool()
    pool.add_all(teacher_names)

    if not pool.models:
        raise RuntimeError("所有教师下载失败")

    # 生成 prompts
    if prompts is None:
        prompts = RapidDistill._generate_diverse_prompts(total_examples)

    # 对每个教师, 生成训练数据
    all_data = []
    for name in pool.models:
        info = DOWNLOADABLE_TEACHERS[name]
        # 每教师分配一半数据 (或平均分配)
        n = total_examples // len(pool.models)

        print(f"\n教师 {name} ({info.params}): 生成 {n} 条...")
        for i in range(0, n, 5):
            batch = prompts[i:i+5]
            responses = pool.batch_generate(name, batch)
            for p, r in zip(batch, responses):
                if r and len(r) > 20:
                    all_data.append({"instruction": p, "output": r, "teacher": name})

        if len(all_data) >= total_examples * 1.5:
            break

    print(f"\n总生成: {len(all_data)} 条训练数据")

    # 蒸馏训练
    from .UniversalDistiller import DataDistiller, DistillConfig, TeacherInterface

    temp_teacher = TeacherInterface.from_local(
        pool.models[list(pool.models.keys())[0]][0],  # 第一个模型的 model
        pool.models[list(pool.models.keys())[0]][1],  # tokenizer
    )

    config = DistillConfig(num_epochs=3, learning_rate=2e-5, output_dir=output_dir)
    distiller = DataDistiller(temp_teacher, student=(student_model, student_tokenizer), config=config)
    filtered = distiller._filter_by_quality(all_data)
    metrics = distiller._train_student(filtered, 0)

    student_model.save_pretrained(output_dir)
    student_tokenizer.save_pretrained(output_dir)

    # 释放显存
    if auto_cleanup:
        for name in list(pool.models.keys()):
            pool.remove(name)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("教师模型已释放")

    print(f"\n=== 本地蒸馏完成! {output_dir} ===")
    return {"model_path": output_dir, "teachers_used": list(pool.models.keys())[:0] if auto_cleanup else list(pool.models.keys()),
            "training_data": len(filtered), "metrics": metrics}


# ═══════════════════════════════════════════════════════════
# 智能路由器: 根据题目类型自动选最合适的教师
# ═══════════════════════════════════════════════════════════

class SmartRouter:
    """
    智能路由 — 根据 prompt 特征自动选最好的教师。

    策略:
      - 数学/推理 → DeepSeek (便宜, math强)
      - 代码       → DeepSeek (便宜, code强)
      - 安全/伦理  → Claude (安全性强)
      - 创意/写作  → GPT-4o (综合最强)
      - 中文       → Qwen-Max (中文最强)
      - 简单问答   → GPT-4o-mini (最便宜)
      - 复杂推理   → DeepSeek-R1 (推理专用)
      - 默认       → GPT-4o-mini (性价比)
    """

    # 关键词 → 教师映射
    KEYWORD_ROUTES = {
        "math": ["deepseek"],
        "reasoning": ["deepseek-reasoner", "deepseek"],
        "code": ["deepseek"],
        "safety": ["claude"],
        "creative": ["gpt4o"],
        "writing": ["gpt4o"],
        "analysis": ["gpt4o"],
        "chinese": ["qwen-max"],
        "translation": ["qwen-max", "gpt4o-mini"],
        "simple": ["gpt4o-mini"],
        "complex": ["deepseek-reasoner", "gpt4o"],
    }

    # 数学关键词
    MATH_KW = ["计算", "等于", "求解", "证明", "数学", "方程", "概率", "sum", "solve",
                "=?","导数", "积分", "根号", "x²", "x^", "多少", "推理", "逻辑"]
    # 代码关键词
    CODE_KW = ["代码", "函数", "实现", "编程", "Python", "Java", "算法", "class", "def",
                "code", "写一个", "输出", "输入", "API", "bug", "import", "数据库"]
    # 安全关键词
    SAFETY_KW = ["安全", "危险", "合法", "道德", "伦理", "偏见", "歧视"]
    # 创意关键词
    CREATIVE_KW = ["故事", "诗歌", "创意", "设计", "写一", "想象", "假如"]

    def route(self, prompt: str, available_teachers: List[str] = None) -> str:
        """为 prompt 选择最佳教师。"""
        available = available_teachers or list(TEACHER_PROFILES.keys())

        # 1. 关键词匹配
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in self.MATH_KW):
            candidates = ["deepseek-reasoner", "deepseek"] if "deepseek" in available or "deepseek-reasoner" in available else []
            return next((c for c in candidates if c in available), available[0])

        if any(kw in prompt_lower for kw in self.CODE_KW):
            return "deepseek" if "deepseek" in available else available[0]

        if any(kw in prompt_lower for kw in self.SAFETY_KW):
            return "claude" if "claude" in available else available[0]

        if any(kw in prompt_lower for kw in self.CREATIVE_KW):
            return "gpt4o" if "gpt4o" in available else available[0]

        # 2. 中文检测
        cjk_count = len(re.findall(r"[一-鿿]", prompt))
        if cjk_count > len(prompt) * 0.3:
            return "qwen-max" if "qwen-max" in available else available[0]

        # 3. 问题长度判断复杂度
        if len(prompt) > 500:
            return "gpt4o" if "gpt4o" in available else available[0]

        # 4. 默认: 最便宜的
        preference = ["gpt4o-mini", "deepseek", "gpt4o", "claude"]
        return next((p for p in preference if p in available), available[0])


# ═══════════════════════════════════════════════════════════
# 并行 API 调用器
# ═══════════════════════════════════════════════════════════

class ParallelAPICaller:
    """并行批量调用多个 API 教师。"""

    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.teachers = {}
        self._cost_tracker = {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0}
        self._lock = threading.Lock()

    def _get_teacher(self, name: str):
        if name in self.teachers:
            return self.teachers[name]

        profile = TEACHER_PROFILES[name]
        from .UniversalDistiller import TeacherInterface

        key = self.api_keys.get(profile.provider) or os.environ.get(f"{profile.provider.upper()}_API_KEY")

        if profile.provider == "openai":
            teacher = TeacherInterface.from_openai(profile.model_id, key)
        elif profile.provider == "anthropic":
            teacher = TeacherInterface.from_anthropic(profile.model_id, key)
        elif profile.provider == "deepseek":
            teacher = TeacherInterface.from_deepseek(profile.model_id, key)
        else:
            teacher = TeacherInterface(model_type=profile.provider, model_name=profile.model_id, api_key=key)

        # 附加 profile 信息
        teacher.profile = profile
        self.teachers[name] = teacher
        return teacher

    def batch_call(
        self,
        prompts: List[str],
        teacher_names: List[str] = None,
        router: "SmartRouter" = None,
        concurrency: int = 10,
        on_progress: Callable = None,
    ) -> List[Dict]:
        """
        并行调用多个教师 API，为每个 prompt 选择最佳教师。

        返回: [{"prompt": ..., "response": ..., "teacher": ..., "cost": ...}, ...]
        """
        if router is None:
            router = SmartRouter()

        if teacher_names is None:
            teacher_names = [t for t in TEACHER_PROFILES if self._get_teacher(t) is not None]
            if not teacher_names:
                teacher_names = ["gpt4o-mini"]  # 默认最便宜

        results = []

        def process_one(i, prompt):
            """处理单个 prompt。"""
            teacher_name = router.route(prompt, teacher_names)
            teacher = self._get_teacher(teacher_name)

            try:
                response = teacher.generate(prompt)
                # 估算成本
                est_input = len(prompt) // 4  # 粗略: 4 chars ≈ 1 token
                est_output = len(response) // 4
                cost = (teacher.profile.cost_per_1k_input * est_input / 1000 +
                        teacher.profile.cost_per_1k_output * est_output / 1000)

                with self._lock:
                    self._cost_tracker["total_input_tokens"] += est_input
                    self._cost_tracker["total_output_tokens"] += est_output
                    self._cost_tracker["total_cost"] += cost

                return {"prompt": prompt, "response": response, "teacher": teacher_name, "cost": round(cost, 6),
                        "index": i, "success": True}
            except Exception as e:
                logger.warning(f"API 调用失败 [{teacher_name}]: {e}")
                # 回退到备用
                for fallback in ["gpt4o-mini", "deepseek", "gpt4o"]:
                    if fallback != teacher_name and fallback in teacher_names:
                        try:
                            fb = self._get_teacher(fallback)
                            response = fb.generate(prompt)
                            return {"prompt": prompt, "response": response, "teacher": f"{fallback}(fallback)",
                                    "cost": 0, "index": i, "success": True}
                        except Exception:
                            continue
                return {"prompt": prompt, "response": "", "teacher": teacher_name, "cost": 0,
                        "index": i, "success": False, "error": str(e)}

        # 并行执行
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_one, i, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if on_progress:
                    on_progress(len(results), len(prompts))

        # 按原始顺序排序
        results.sort(key=lambda x: x["index"])
        return results

    def get_cost_summary(self) -> Dict:
        return dict(self._cost_tracker)


# ═══════════════════════════════════════════════════════════
# RapidDistill: 一行代码完成 API 蒸馏
# ═══════════════════════════════════════════════════════════

class RapidDistill:
    """
    快速蒸馏 — 调 API，混蒸各家长处。

    三种蒸馏模式:
      "smart_route": 自动路由, 每道题发给最合适的教师 (推荐!)
      "ensemble":    每道题发给多个教师, 取最好的答案 (最贵但最好)
      "divide":      按领域分工, 每个教师负责自己擅长的题目

    用法:
        from saddlellm import RapidDistill

        # 最简用法
        RapidDistill.run(
            student_model=my_model,
            student_tokenizer=my_tokenizer,
            use_teachers=["deepseek", "claude", "gpt4o-mini"],
        )

        # 自定义 prompts
        RapidDistill.run(
            student_model=model, student_tokenizer=tokenizer,
            prompts=my_prompts,
            use_teachers=["deepseek", "gpt4o"],
            mode="smart_route",
        )
    """

    @staticmethod
    def run(
        student_model,
        student_tokenizer,
        prompts: List[str] = None,
        total_examples: int = 500,
        use_teachers: List[str] = None,
        mode: str = "smart_route",
        api_keys: Dict[str, str] = None,
        output_dir: str = "./rapid_distilled_model",
        domains: Dict[str, List[str]] = None,  # 领域 → prompt列表
        concurrency: int = 10,
    ) -> Dict:
        """
        一键快速蒸馏。

        mode:
          "smart_route" — 自动路由, 每道题发给最合适的教师
          "ensemble"    — 多教师共同回答, 选最优
          "divide"      — 按领域分工

        返回: {"generated_pairs": 数量, "cost": 总价, "model_path": 路径}
        """
        # 默认教师: 性价比组合
        if use_teachers is None:
            use_teachers = ["deepseek", "gpt4o-mini", "claude"]

        router = SmartRouter()
        caller = ParallelAPICaller(api_keys)

        logger.info(f"=== RapidDistill: {mode}, teachers={use_teachers} ===")

        # 生成/选择 prompts
        if prompts is None:
            prompts = RapidDistill._generate_diverse_prompts(total_examples, domains)

        # 模式 1: 智能路由
        if mode == "smart_route":
            logger.info(f"Mode: 智能路由 ({len(prompts)} prompts)")
            results = caller.batch_call(prompts, use_teachers, router=router, concurrency=concurrency)

        # 模式 2: Ensemble (多个教师答同一题, 取最好)
        elif mode == "ensemble":
            logger.info(f"Mode: Ensemble ({len(prompts)} prompts × {len(use_teachers)} teachers)")
            all_results = []
            for teacher_name in use_teachers:
                logger.info(f"  Teacher: {teacher_name}")
                batch = caller.batch_call(prompts, [teacher_name], concurrency=concurrency)
                all_results.append(batch)

            # 对每个 prompt, 选最好的回答
            results = []
            for i in range(len(prompts)):
                candidates = [r[i] for r in all_results if r[i]["success"]]
                if candidates:
                    # 简单的质量排序: 长度适中 + 结构好
                    best = max(candidates, key=lambda x: RapidDistill._quality_heuristic(x["response"]))
                    best["ensemble_teachers"] = [r[i]["teacher"] for r in all_results if r[i]["success"]]
                    results.append(best)
                else:
                    results.append({"prompt": prompts[i], "response": "", "teacher": "none", "success": False})

        # 模式 3: 按领域分工
        elif mode == "divide" and domains:
            logger.info(f"Mode: 分工 ({len(domains)} domains)")
            results = []
            for domain, domain_prompts in domains.items():
                # 为每个领域选教师
                domain_teacher = router.route(domain_prompts[0] if domain_prompts else domain, use_teachers)
                logger.info(f"  {domain} → {domain_teacher} ({len(domain_prompts)} prompts)")
                batch = caller.batch_call(domain_prompts, [domain_teacher] if domain_teacher in use_teachers else use_teachers, concurrency=concurrency)
                results.extend(batch)
        else:
            raise ValueError(f"未知模式: {mode}")

        # 统计
        success = [r for r in results if r.get("success")]
        logger.info(f"API 调用: {len(success)}/{len(results)} 成功")

        # 训练学生
        train_data = [{"instruction": r["prompt"], "output": r["response"]} for r in success if r.get("response")]
        logger.info(f"训练数据: {len(train_data)} 条")

        from .UniversalDistiller import DataDistiller, DistillConfig, TeacherInterface
        temp_teacher = caller._get_teacher(use_teachers[0])
        config = DistillConfig(num_epochs=3, learning_rate=2e-5, output_dir=output_dir)
        distiller = DataDistiller(temp_teacher, student=(student_model, student_tokenizer), config=config)
        filtered = distiller._filter_by_quality(train_data)
        metrics = distiller._train_student(filtered, 0)

        # 保存
        student_model.save_pretrained(output_dir)
        student_tokenizer.save_pretrained(output_dir)

        # 成本统计
        cost = caller.get_cost_summary()

        summary = {
            "mode": mode,
            "teachers_used": use_teachers,
            "prompts_total": len(prompts),
            "api_success": len(success),
            "training_data": len(train_data),
            "filtered_data": len(filtered),
            "model_path": output_dir,
            "cost_estimate": {"total_cost_usd": round(cost["total_cost"], 4),
                               "input_tokens": cost["total_input_tokens"],
                               "output_tokens": cost["total_output_tokens"]},
            "teacher_distribution": RapidDistill._count_teachers(success),
        }

        logger.info(f"=== RapidDistill 完成! ===")
        logger.info(f"  模型: {output_dir}")
        logger.info(f"  预估成本: ${summary['cost_estimate']['total_cost_usd']:.4f}")
        logger.info(f"  训练数据: {summary['training_data']} 条")

        return summary

    @staticmethod
    def _generate_diverse_prompts(total: int, domains: Dict[str, List[str]] = None) -> List[str]:
        """生成多样化 prompts。"""
        if domains:
            all_prompts = []
            for domain_prompts in domains.values():
                all_prompts.extend(domain_prompts)
            random.shuffle(all_prompts)
            return all_prompts[:total]

        # 按领域生成
        templates = {
            "math": ["计算: {}", "证明: {}", "求解: {}", "{} 的结果是多少?"],
            "code": ["用 Python 实现: {}", "解释这段代码: {}", "优化以下算法: {}"],
            "knowledge": ["解释: {}", "{} 的原理是什么?", "关于 {} 请详细说明"],
            "creative": ["写一个关于 {} 的故事", "设计一个 {} 的方案"],
            "chinese": ["用中文解释: {}", "{} 的中文翻译是什么?"],
            "analysis": ["分析: {}", "总结: {}", "从多角度评估: {}"],
            "safety": ["如何安全地处理: {}", "{} 存在什么风险?"],
        }

        seeds = {
            "math": ["1+2*3-4/2", "鸡兔同笼", "概率计算", "等差数列求和", "勾股定理证明",
                      "求x²+5x+6=0的解", "100以内质数", "最大公约数", "排列组合"],
            "code": ["快速排序", "二分查找", "二叉树遍历", "REST API", "数据处理pandas",
                      "正则表达式", "多线程", "装饰器", "链表反转"],
            "knowledge": ["机器学习", "黑洞", "光合作用", "区块链", "量子计算",
                          "二战历史", "DNA复制", "气候变化", "古罗马", "神经网络"],
            "creative": ["一个侦探故事", "未来城市的", "一个AI的日记", "时间旅行"],
            "chinese": ["唐诗三百首", "红楼梦人物分析", "中国四大发明", "中医基础理论"],
            "analysis": ["最近的经济趋势", "人工智能对就业的影响", "新能源技术发展"],
            "safety": ["个人信息保护", "网络安全", "食品安全", "心理健康"],
        }

        prompts = []
        domain_keys = list(templates.keys())
        while len(prompts) < total:
            domain = random.choice(domain_keys)
            seed = random.choice(seeds[domain])
            template = random.choice(templates[domain])
            prompts.append(template.format(seed))

        return prompts[:total]

    @staticmethod
    def _quality_heuristic(text: str) -> float:
        if not text: return 0.0
        score = 0.5
        if len(text) > 100: score += 0.15
        if len(text) > 300: score += 0.1
        if any(kw in text for kw in ["首先", "因为", "例如", "1.", "所以"]): score += 0.1
        if any(kw in text for kw in ["答案", "结果", "综上"]): score += 0.1
        if len(text) < 30: score -= 0.3
        return min(1.0, score)

    @staticmethod
    def _count_teachers(results: List[Dict]) -> Dict:
        counts = {}
        for r in results:
            t = r.get("teacher", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    @staticmethod
    def estimate_cost(prompts: List[str], teachers: List[str] = None) -> Dict:
        """预估蒸馏费用。"""
        teachers = teachers or ["deepseek", "gpt4o-mini"]
        total_input_chars = sum(len(p) for p in prompts)
        est_input_tokens = total_input_chars / 4
        est_output_tokens = est_input_tokens * 2

        costs = {}
        total = 0
        for t in teachers:
            if t in TEACHER_PROFILES:
                p = TEACHER_PROFILES[t]
                c = (p.cost_per_1k_input * est_input_tokens + p.cost_per_1k_output * est_output_tokens) / 1000
                costs[t] = round(c, 4)
                total += c

        return {
            "prompts": len(prompts),
            "est_input_tokens": int(est_input_tokens),
            "est_output_tokens": int(est_output_tokens),
            "per_teacher_cost": costs,
            "total_estimated": round(total, 4),
            "note": "实际费用取决于路由结果 (复杂题走贵的, 简单题走便宜的)",
        }
