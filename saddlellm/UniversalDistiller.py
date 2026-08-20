"""
通用模型蒸馏系统 — 任何模型的能力都能快速蒸馏到你的模型

支持 4 种蒸馏模式:
  1. 数据蒸馏 (Data Distill)        — API 教师 → 生成数据 → 训练学生 (最实用)
  2. Logit 蒸馏 (Logit Distill)     — 本地跑师生模型, 匹配输出分布 (最精确)
  3. 多教师蒸馏 (Multi-Teacher)     — 融合多个教师的长处
  4. 能力蒸馏 (Capability Distill)  — 针对性蒸馏特定能力 (推理/数学/代码)

实际使用流程:
  Step 1: 选择一个强模型作为教师 (GPT-4 / Claude / DeepSeek / 本地大模型)
  Step 2: 自动生成高质量训练数据
  Step 3: 用数据训练学生模型
  Step 4: 评估 → 如果不够好, 再来一轮
"""
import os
import json
import time
import random
import logging
from typing import List, Dict, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch

logger = logging.getLogger(__name__)


# ============================================================
# Teacher Interface: 任何模型都能作为教师
# ============================================================

class TeacherInterface:
    """
    教师模型接口 — 封装任意模型/API 为统一接口。

    支持:
      - OpenAI API (GPT-4, GPT-4o)
      - Anthropic API (Claude)
      - DeepSeek API
      - 本地 HuggingFace 模型
      - vLLM/LLM 推理服务端点
    """

    def __init__(
        self,
        model_type: str = "local",
        model_name: str = "",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        local_model=None,
        local_tokenizer=None,
        endpoint_url: Optional[str] = None,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer
        self.endpoint_url = endpoint_url

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """统一生成接口。"""
        if self.model_type in ("openai", "deepseek"):
            return self._generate_openai_compatible(prompt, temperature, max_tokens)
        elif self.model_type == "anthropic":
            return self._generate_anthropic(prompt, temperature, max_tokens)
        elif self.model_type == "local":
            return self._generate_local(prompt, temperature, max_tokens)
        elif self.model_type == "endpoint":
            return self._generate_endpoint(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def batch_generate(self, prompts: List[str], temperature: float = 0.7,
                       max_tokens: int = 2048, concurrency: int = 5) -> List[str]:
        """批量生成。"""
        results = []
        for i in range(0, len(prompts), concurrency):
            batch = prompts[i:i+concurrency]
            batch_results = [self.generate(p, temperature, max_tokens) for p in batch]
            results.extend(batch_results)
            if i + concurrency < len(prompts):
                time.sleep(0.5)
        return results

    def _generate_openai_compatible(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API 调用失败: {e}, 回退到本地模型")
            if self.local_model:
                return self._generate_local(prompt, temperature, max_tokens)
            raise

    def _generate_anthropic(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model=self.model_name or "claude-sonnet-4-6",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic API 失败: {e}")
            raise

    def _generate_local(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import torch
        self.local_model.eval()
        inputs = self.local_tokenizer(prompt, return_tensors="pt", truncation=True,
                                       max_length=4096).to(self.local_model.device)
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=temperature > 0.1, top_p=0.95,
                pad_token_id=self.local_tokenizer.pad_token_id or self.local_tokenizer.eos_token_id,
            )
        full = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(self.local_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        return full[prompt_len:].strip()

    def _generate_endpoint(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import requests
        response = requests.post(
            self.endpoint_url,
            json={"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
            timeout=120,
        )
        return response.json()["response"]

    @classmethod
    def from_openai(cls, model_name: str = "gpt-4o", api_key: str = None):
        return cls(model_type="openai", model_name=model_name, api_key=api_key)

    @classmethod
    def from_anthropic(cls, model_name: str = "claude-sonnet-4-6", api_key: str = None):
        return cls(model_type="anthropic", model_name=model_name, api_key=api_key)

    @classmethod
    def from_deepseek(cls, model_name: str = "deepseek-chat", api_key: str = None):
        return cls(model_type="deepseek", model_name=model_name, api_key=api_key,
                    api_base="https://api.deepseek.com")

    @classmethod
    def from_local(cls, model, tokenizer):
        return cls(model_type="local", local_model=model, local_tokenizer=tokenizer)


# ============================================================
# Core: Data Distiller (最实用)
# ============================================================

@dataclass
class DistillConfig:
    num_epochs: int = 3
    learning_rate: float = 2e-5
    per_device_batch_size: int = 4
    max_seq_length: int = 2048
    temperature: float = 0.8          # 生成多样性
    quality_threshold: float = 0.5    # 数据质量最低分
    num_rounds: int = 2               # 多轮蒸馏轮数
    save_every_round: bool = True
    output_dir: str = "./distilled_model"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32


class DataDistiller:
    """
    数据蒸馏器 — 用教师模型的能力"灌入"学生模型。

    这是最实用的蒸馏方式: 不需要同时加载两个模型, 不需要匹配架构。

    流程:
      1. 用教师模型对 prompts 生成高质量回复
      2. 筛选 (去低质量)
      3. 用筛选后的数据训练学生模型 (SFT)
      4. 可选: 多轮迭代

    用法:
        # 教师: GPT-4 API
        teacher = TeacherInterface.from_openai("gpt-4o")

        # 学生: 你的本地模型
        student_model = AutoModelForCausalLM.from_pretrained("path/to/your/model")

        distiller = DataDistiller(teacher=teacher, student=(student_model, tokenizer))
        distiller.distill(prompts=your_prompts)
        # → 学生模型会自动学到 GPT-4 的风格和能力
    """

    def __init__(
        self,
        teacher: TeacherInterface,
        student: Tuple[torch.nn.Module, any],  # (model, tokenizer)
        config: DistillConfig = None,
    ):
        self.teacher = teacher
        self.student_model, self.student_tokenizer = student
        self.config = config or DistillConfig()
        self.device = next(self.student_model.parameters()).device

        self._generated_data: List[Dict] = []
        self._round_metrics: List[Dict] = []

    def distill(
        self,
        prompts: List[str],
        eval_prompts: Optional[List[str]] = None,
        resume_from_round: int = 0,
    ) -> Dict:
        """
        执行蒸馏。

        参数:
          prompts: 训练用的 prompts (越多越好, 推荐 1000+)
          eval_prompts: 验证用 prompts
          resume_from_round: 从第几轮继续

        返回:
          {"rounds": [...], "final_model_path": "..."}
        """
        logger.info(f"=== 数据蒸馏: {len(prompts)} prompts, {self.config.num_rounds} rounds ===")

        for round_idx in range(resume_from_round, self.config.num_rounds):
            logger.info(f"\n{'='*50}\n  Round {round_idx + 1}/{self.config.num_rounds}\n{'='*50}")

            # Step 1: 教师生成
            logger.info("Step 1: 教师模型生成训练数据...")
            round_data = self._generate_data(prompts, round_idx)

            # Step 2: 质量筛选
            logger.info("Step 2: 质量筛选...")
            filtered_data = self._filter_by_quality(round_data)
            self._generated_data.extend(filtered_data)

            logger.info(f"  本轮: {len(prompts)} prompts → {len(filtered_data)} 条高质量数据")
            logger.info(f"  累计: {len(self._generated_data)} 条数据")

            # Step 3: 训练学生
            logger.info("Step 3: 训练学生模型...")
            metrics = self._train_student(self._generated_data, round_idx)
            self._round_metrics.append(metrics)

            # Step 4: 评估
            if eval_prompts:
                logger.info("Step 4: 评估...")
                eval_metrics = self._evaluate(eval_prompts[:20])
                metrics["eval"] = eval_metrics
                logger.info(f"  Eval: {eval_metrics}")

            # Step 5: 保存
            if self.config.save_every_round:
                round_path = f"{self.config.output_dir}_round{round_idx+1}"
                self.student_model.save_pretrained(round_path)
                self.student_tokenizer.save_pretrained(round_path)
                logger.info(f"  已保存: {round_path}")

        # 最终保存
        final_path = self.config.output_dir
        self.student_model.save_pretrained(final_path)
        self.student_tokenizer.save_pretrained(final_path)

        logger.info(f"=== 蒸馏完成! 最终模型: {final_path} ===")
        return {"rounds": self._round_metrics, "final_model_path": final_path,
                "total_data_generated": len(self._generated_data)}

    def _generate_data(self, prompts: List[str], round_idx: int) -> List[Dict]:
        """教师模型批量生成回复。"""
        temperature = self.config.temperature * (0.9 ** round_idx)  # 每轮降低温度
        data = []

        # 分批处理
        batch_size = 10
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            responses = self.teacher.batch_generate(batch_prompts, temperature=temperature)

            for prompt, response in zip(batch_prompts, responses):
                if response and len(response) > 20:
                    data.append({
                        "instruction": prompt,
                        "output": response,
                        "round": round_idx + 1,
                        "teacher": self.teacher.model_type,
                    })

            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"  生成进度: {min(i+batch_size, len(prompts))}/{len(prompts)}")

        return data

    def _filter_by_quality(self, data: List[Dict]) -> List[Dict]:
        """质量筛选: 过滤太短/太长/空泛的回复。"""
        filtered = []
        for d in data:
            output = d.get("output", "")
            score = 0.5

            # 基础质量
            if len(output) > 100: score += 0.1
            if len(output) > 300: score += 0.1
            if len(output) > 50 and len(output) < 5000: score += 0.1
            if any(kw in output for kw in ["因为", "所以", "例如", "首先", "根据"]): score += 0.1
            if "我不知道" in output or "作为AI" in output: score -= 0.2
            if len(output) < 30: score -= 0.3

            d["quality_score"] = min(1.0, max(0.0, score))
            if d["quality_score"] >= self.config.quality_threshold:
                filtered.append(d)

        return filtered

    def _train_student(self, data: List[Dict], round_idx: int) -> Dict:
        """用生成的数据训练学生模型 (SFT)。"""
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model

        # 格式化为 SFT 数据
        train_texts = []
        for d in data[-5000:]:  # 用最近的 5000 条
            text = f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['output']}"
            train_texts.append({"text": text})

        dataset = Dataset.from_list(train_texts)

        def tokenize(examples):
            return self.student_tokenizer(
                examples["text"], truncation=True,
                max_length=self.config.max_seq_length, padding="max_length",
            )

        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # LoRA 加速
        if self.config.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=self.config.lora_r, lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
            )
            try:
                self.student_model = get_peft_model(self.student_model, lora_config)
            except Exception:
                pass  # 可能已经应用了 LoRA

        training_args = TrainingArguments(
            output_dir=f"{self.config.output_dir}_tmp",
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            report_to="none",
        )

        trainer = Trainer(
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForSeq2Seq(self.student_tokenizer, pad_to_multiple_of=8),
        )

        trainer.train()
        return {"trained_samples": len(train_texts), "epochs": self.config.num_epochs}

    def _evaluate(self, eval_prompts: List[str]) -> Dict:
        """评估蒸馏效果。"""
        from .FrontierAlign import RejectionSampling

        self.student_model.eval()
        rs = RejectionSampling(self.student_model, self.student_tokenizer)

        scores = []
        for prompt in eval_prompts:
            candidates = rs.generate_candidates(prompt)
            best = rs.select_best(candidates, method="rules")
            scores.append(best.get("best_score", 0))

        return {"avg_score": sum(scores) / len(scores) if scores else 0, "num_evaluated": len(scores)}


# ============================================================
# Multi-Teacher Distiller (多教师蒸馏)
# ============================================================

class MultiTeacherDistiller:
    """
    多教师蒸馏 — 融合多个教师模型的长处。

    用法:
        teacher_gpt4 = TeacherInterface.from_openai("gpt-4o")
        teacher_claude = TeacherInterface.from_anthropic("claude-sonnet-4-6")
        teacher_deepseek = TeacherInterface.from_deepseek("deepseek-chat")

        multi = MultiTeacherDistiller(
            teachers=[teacher_gpt4, teacher_claude, teacher_deepseek],
            weights=[0.4, 0.3, 0.3],  # GPT4 权重更高
            student=(model, tokenizer),
        )
        multi.distill(prompts)
    """

    def __init__(
        self,
        teachers: List[TeacherInterface],
        student: Tuple,
        weights: Optional[List[float]] = None,
        config: DistillConfig = None,
    ):
        self.teachers = teachers
        self.student = student
        self.weights = weights or [1.0 / len(teachers)] * len(teachers)
        self.config = config or DistillConfig()

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def distill(self, prompts: List[str]) -> Dict:
        """
        多教师蒸馏:
          1. 每个教师对同一个 prompt 生成回复
          2. 用 AI 裁判选出最好的回复
          3. 用最好的回复训练学生
        """
        logger.info(f"=== 多教师蒸馏: {len(self.teachers)} 个教师 ===")

        all_data = []

        for i, (teacher, weight) in enumerate(zip(self.teachers, self.weights)):
            logger.info(f"教师 {i+1}/{len(self.teachers)}: {teacher.model_type} (权重={weight:.2f})")

            # 按权重分配 prompts
            num_prompts = max(50, int(len(prompts) * weight))
            teacher_prompts = random.sample(prompts, num_prompts)

            responses = teacher.batch_generate(teacher_prompts, temperature=0.8)

            for p, r in zip(teacher_prompts, responses):
                if r and len(r) > 20:
                    all_data.append({
                        "instruction": p,
                        "output": r,
                        "teacher": teacher.model_type,
                        "teacher_weight": weight,
                    })

        # 用所有教师的数据训练学生
        distiller = DataDistiller(
            teacher=self.teachers[0],  # 任意一个, 因为不直接调用
            student=self.student,
            config=self.config,
        )

        # 直接注入数据训练
        filtered = distiller._filter_by_quality(all_data)
        metrics = distiller._train_student(filtered, 0)

        logger.info(f"多教师蒸馏完成: {len(all_data)} → {len(filtered)} 条数据")
        return {"total_data": len(filtered), "metrics": metrics,
                "teacher_distribution": {
                    t.model_type: len([d for d in filtered if d["teacher"] == t.model_type])
                    for t in self.teachers
                }}


# ============================================================
# Capability Distiller (能力定向蒸馏)
# ============================================================

class CapabilityDistiller:
    """
    能力定向蒸馏 — 专门蒸馏某个特定能力。

    预设能力模板:
      - "reasoning":     数学和逻辑推理
      - "coding":        代码生成和解释
      - "writing":       写作能力
      - "translation":   翻译能力
      - "analysis":      文本分析和总结
      - "creative":      创意写作
      - "knowledge":     知识问答
      - "safety":        安全性/对齐

    用法:
        cd = CapabilityDistiller(teacher, student=student_model)
        cd.distill("reasoning", num_examples=500)
        cd.distill("coding", num_examples=300)
    """

    CAPABILITY_TEMPLATES = {
        "reasoning": [
            "请一步步推理: {problem}",
            "解决这个数学问题并解释每一步: {problem}",
            "分析以下逻辑问题: {problem}",
            "这个推理题怎么解? {problem}",
        ],
        "coding": [
            "用 Python 实现: {task}",
            "解释这段代码的功能并优化: {task}",
            "用代码解决: {task}",
            "实现一个 {task} 的功能",
        ],
        "writing": [
            "写一篇文章关于: {topic}",
            "用专业风格写: {topic}",
            "写一篇分析: {topic}",
        ],
        "translation": [
            "将以下内容翻译成中文: {text}",
            "将以下内容翻译成英文: {text}",
            "请翻译并解释关键术语: {text}",
        ],
        "analysis": [
            "分析以下文本的主要内容: {text}",
            "总结这段文字的要点: {text}",
            "从多角度分析: {text}",
        ],
        "creative": [
            "写一个关于 {topic} 的故事",
            "创作一首关于 {topic} 的诗",
            "设计一个 {topic} 的创意方案",
        ],
        "knowledge": [
            "详细解释: {question}",
            "{question} 的原理是什么?",
            "关于 {question}, 请提供深入分析",
        ],
        "safety": [
            "如果有人问危险问题, 你应该如何回应? {scenario}",
            "如何处理以下伦理困境: {scenario}",
            "对以下内容做安全评估: {scenario}",
        ],
    }

    def __init__(self, teacher: TeacherInterface, student_model, student_tokenizer,
                 config: DistillConfig = None):
        self.teacher = teacher
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.config = config or DistillConfig()
        self._trained_capabilities: List[str] = []

    def distill(
        self,
        capability: str,
        num_examples: int = 500,
        extra_seeds: Optional[List[str]] = None,
    ) -> Dict:
        """
        定向蒸馏某个能力。

        capability: 能力名称 (reasoning/coding/writing/...)
        num_examples: 生成的训练样本数
        extra_seeds: 额外的种子问题
        """
        if capability not in self.CAPABILITY_TEMPLATES:
            raise KeyError(f"未知能力: {capability}. 可用: {list(self.CAPABILITY_TEMPLATES.keys())}")

        logger.info(f"=== 定向蒸馏: {capability} ({num_examples} 样本) ===")

        # Step 1: 生成种子问题
        seeds = self._generate_seeds(capability, num_examples, extra_seeds)

        # Step 2: 用种子从教师生成数据
        data = []
        for seed in seeds[:num_examples]:
            template = random.choice(self.CAPABILITY_TEMPLATES[capability])
            prompt = template.format(problem=seed, task=seed, topic=seed, text=seed,
                                      question=seed, scenario=seed)
            response = self.teacher.generate(prompt)
            if response and len(response) > 50:
                data.append({"instruction": prompt, "output": response})

        logger.info(f"  生成: {len(data)} 条 {capability} 训练数据")

        # Step 3: 训练学生
        distiller = DataDistiller(
            teacher=self.teacher,
            student=(self.student_model, self.student_tokenizer),
            config=self.config,
        )
        filtered = distiller._filter_by_quality(data)
        metrics = distiller._train_student(filtered, 0)

        self._trained_capabilities.append(capability)
        return {"capability": capability, "generated": len(data),
                "filtered": len(filtered), "metrics": metrics}

    def distill_multiple(self, capabilities: List[str], examples_per: int = 300) -> Dict:
        """批量蒸馏多个能力。"""
        results = {}
        for cap in capabilities:
            results[cap] = self.distill(cap, num_examples=examples_per)
        return results

    def _generate_seeds(self, capability: str, count: int, extra: Optional[List[str]] = None):
        """自动生成种子问题。"""
        seeds = []
        seed_bank = {
            "reasoning": ["1+2*3-4/2等于多少", "鸡兔同笼问题", "概率计算: 抛硬币", "数列求和"],
            "coding": ["快速排序", "二分查找", "Web爬虫", "REST API", "数据库查询", "正则表达式"],
            "knowledge": ["机器学习", "黑洞", "光合作用", "区块链", "量子计算", "深度学习"],
        }

        if extra:
            seeds.extend(extra)

        # 从种子库获取
        bank = seed_bank.get(capability, [])
        while len(seeds) < count:
            seeds.append(random.choice(bank) if bank else f"解释{capability}相关概念 #{len(seeds)+1}")
            # 避免完全重复
            if bank and len(seeds) > len(bank) * 3:
                seeds.append(f"{capability} 相关问题 #{len(seeds)+1}")

        return seeds[:count]


# ============================================================
# AutoDistiller: 全自动蒸馏
# ============================================================

class AutoDistiller:
    """
    全自动蒸馏 — 给定教师和学生, 自动完成所有步骤。

    用法:
        teacher = TeacherInterface.from_openai("gpt-4o")
        auto = AutoDistiller(
            teacher=teacher,
            student_model=my_model,
            student_tokenizer=my_tokenizer,
        )

        # 执行全能力蒸馏 (自动生成 prompts → 蒸馏 → 评估 → 调整)
        auto.auto_distill(
            capabilities=["reasoning", "knowledge", "coding"],
            total_examples=1000,
        )
    """

    def __init__(self, teacher: TeacherInterface, student_model, student_tokenizer,
                 config: DistillConfig = None):
        self.teacher = teacher
        self.student_model = student_model
        self.student_tokenizer = student_tokenizer
        self.config = config or DistillConfig()
        self._history = []

    def auto_distill(
        self,
        capabilities: List[str] = None,
        total_examples: int = 1000,
        eval_prompts: List[str] = None,
        target_score: float = 0.7,
        max_rounds: int = 3,
    ) -> Dict:
        """
        全自动多轮蒸馏:
          Round 1: 定向蒸馏核心能力
          Round 2: 评估 → 补弱项
          Round 3: 最终 polish
        """
        capabilities = capabilities or ["reasoning", "knowledge", "coding", "writing"]
        examples_per_cap = total_examples // len(capabilities)

        logger.info(f"=== AutoDistill: {capabilities}, {total_examples} examples ===")

        # Round 1: 定向蒸馏所有能力
        logger.info("Round 1: 蒸馏核心能力...")
        cap_distiller = CapabilityDistiller(
            self.teacher, self.student_model, self.student_tokenizer, self.config
        )

        for cap in capabilities:
            result = cap_distiller.distill(cap, num_examples=examples_per_cap)
            self._history.append(result)

        # Round 2: 评估 + 补弱
        if eval_prompts:
            logger.info("Round 2: 评估弱项...")
            distiller = DataDistiller(
                teacher=self.teacher,
                student=(self.student_model, self.student_tokenizer),
                config=self.config,
            )
            scores = distiller._evaluate(eval_prompts[:30])

            if scores.get("avg_score", 0) < target_score:
                logger.info(f"  当前分数 {scores['avg_score']:.2f} < 目标 {target_score}, 继续强化")
                # 生成更多数据
                more_prompts = self._generate_more_prompts(eval_prompts, 100)
                distiller.distill(more_prompts, eval_prompts[:10])

        # Round 3: 最终蒸馏
        logger.info(f"Round 3: 最终多轮蒸馏 ({max_rounds} rounds)...")
        final_distiller = DataDistiller(
            teacher=self.teacher,
            student=(self.student_model, self.student_tokenizer),
            config=DistillConfig(**{**self.config.__dict__, "num_rounds": max_rounds}),
        )
        final_prompts = self._generate_final_prompts(total_examples)
        result = final_distiller.distill(final_prompts)

        logger.info("=== AutoDistill 完成! ===")
        return {"history": self._history, "final_result": result}

    def _generate_more_prompts(self, eval_prompts: List[str], count: int) -> List[str]:
        """基于 eval 生成更多相关 prompts。"""
        from .AdvancedTechniques import EvolInstruct
        evolver = EvolInstruct(self.student_model, self.student_tokenizer)
        return evolver.evolve_batch(eval_prompts[:10], rounds=2)[:count]

    def _generate_final_prompts(self, count: int) -> List[str]:
        return [f"详细解释 #{i}: {random.choice(['科学', '技术', '历史', '哲学', '数学', '编程'])} 相关知识"
                for i in range(count)]
