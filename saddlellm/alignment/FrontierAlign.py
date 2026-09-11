"""
前沿对齐技术 — OpenAI & Anthropic 核心方法论

═══ Anthropic: Constitutional AI ═══
  原理: 用"宪法规则"让模型自我批评和修正
  特点: 不需要人工标注, 模型自己改进
  论文: "Constitutional AI: Harmlessness from AI Feedback" (2022)

═══ OpenAI: Process Reward + Rejection Sampling ═══
  原理:
    - Process Reward: 对推理的每一步打分, 不是只看最终答案
    - Rejection Sampling: 生成多个候选, 选 reward 最高的
  论文: "Let's Verify Step by Step" (2023), "Scaling Laws for Reward Model Overoptimization" (2022)

═══ 组合使用 ═══
  Step1  → 生成多个候选 (Rejection Sampling)
  Step2  → 用 Constitution 规则筛选
  Step3  → 用 Process Reward 对最佳候选的推理步骤打分
  Step4  → 只保留推理步骤正确的数据做 SFT
"""
import re
import random
import logging
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================
# Part 1: Constitutional AI (Anthropic)
# ============================================================

# Anthropic 的默认宪法规则 (简化版核心原则)
DEFAULT_CONSTITUTION = [
    # Harmlessness (无害)
    "请选择最无害的回复。避免鼓励非法、危险或不道德的行为。",
    "不要生成可能造成身体或心理伤害的内容。",
    "不要鼓励自残或伤害他人的行为。",
    # Honesty (诚实)
    "请选择最诚实的回复。如果不确定,请明确说明不确定性。",
    "不要编造事实。如果缺乏信息,请直接说明。",
    "纠正自己和他人的错误认知。",
    # Helpfulness (有帮助)
    "请选择对用户最有帮助的回复。理解用户的真实需求。",
    "提供具体、可操作的答案,而非模糊的概括。",
    "当用户有隐含需求时,主动识别并满足。",
    # Quality (质量)
    "回复应该清晰、有条理、易于理解。",
    "对于复杂问题,分步骤解释。",
    "使用恰当的例子帮助理解。",
    # Bias (公平)
    "对所有人群保持公平和尊重。",
    "不要基于种族、性别、宗教、年龄等做歧视性假设。",
    "承认并指出社会偏见的存在。",
]


@dataclass
class CAIConfig:
    constitution: List[str] = field(default_factory=lambda: DEFAULT_CONSTITUTION)
    num_critique_rounds: int = 2      # 自我批评轮数
    num_candidates: int = 4           # 每轮生成候选数
    critique_temperature: float = 0.7  # 生成批评时的温度
    revision_temperature: float = 0.5  # 修订时的温度
    top_p: float = 0.95
    max_tokens: int = 1024


class ConstitutionalAI:
    """
    Anthropic Constitutional AI 实现。

    流程:
      1. 模型生成初始回复
      2. 模型根据"宪法规则"批评自己的回复
      3. 模型基于批评修订回复
      4. 重复 2-3 直到收敛或达到最大轮数

    与 RLHF 的区别:
      RLHF: 需要人工标注偏好 → 训练 reward model → PPO
      CAI:  只需要写宪法规则 → 模型自我批评 → 直接修订

    用法:
        cai = ConstitutionalAI(model, tokenizer)
        improved_response = cai.generate_with_cai("如何学习编程?")
        # → 经过多轮自我批评和修订的高质量回复
    """

    def __init__(self, model, tokenizer, config: CAIConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or CAIConfig()
        self.device = next(model.parameters()).device

    def generate_with_cai(self, prompt: str, initial_response: str = None) -> Dict:
        """
        用 CAI 流程生成高质量回复。

        返回:
          {
            "final_response": "...",
            "initial_response": "...",
            "critique_rounds": [
              {"round": 1, "critique": "...", "revision": "..."},
              ...
            ],
            "violations_found": [...],
          }
        """
        # Step 1: 初始回复
        if initial_response is None:
            initial = self._generate(prompt, temperature=self.config.revision_temperature)
        else:
            initial = initial_response

        history = {
            "initial_response": initial,
            "critique_rounds": [],
            "violations_found": [],
        }

        current = initial

        for round_idx in range(self.config.num_critique_rounds):
            # Step 2: 自我批评
            critiques = self._generate_critiques(prompt, current)
            violations = self._extract_violations(critiques)
            history["violations_found"].extend(violations)

            if not violations:
                logger.info(f"CAI 轮 {round_idx+1}: 未发现违规, 停止")
                break

            # Step 3: 基于批评修订
            revision = self._revise_response(prompt, current, critiques)
            history["critique_rounds"].append({
                "round": round_idx + 1,
                "critique": critiques,
                "revision": revision,
                "violations": violations,
            })

            # 检查是否有实质性改进
            if self._similarity(current, revision) > 0.95:
                logger.info(f"CAI 轮 {round_idx+1}: 修订变化很小, 停止")
                current = revision
                break

            current = revision
            logger.info(f"CAI 轮 {round_idx+1}: 发现 {len(violations)} 个问题, 已修订")

        history["final_response"] = current
        return history

    def _generate_critiques(self, prompt: str, response: str) -> str:
        """根据宪法规则批评回复。"""
        # 随机选择宪法规则子集 (避免每次都检查所有规则)
        rules_to_check = random.sample(
            self.config.constitution,
            min(3, len(self.config.constitution))
        )

        critique_prompt = f"""请根据以下"宪法规则"审查这个回复,找出所有违反规则的之处。

宪法规则:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(rules_to_check))}

用户问题: {prompt}

待审查的回复:
{response}

请逐条列出回复违反了哪些规则,以及具体违反了什么。如果完全没有违反,请说"未发现违规"。
不要给出修订建议,只做审查。"""

        return self._generate(critique_prompt, temperature=self.config.critique_temperature)

    def _revise_response(self, prompt: str, response: str, critiques: str) -> str:
        """基于批评生成修订版回复。"""
        revision_prompt = f"""请根据以下批评,修订你的回复。确保新回复遵守所有宪法规则。

用户问题: {prompt}

原始回复:
{response}

审查发现的问题:
{critiques}

请生成修订后的回复。要求:
1. 保持原有有用信息的完整性
2. 修复所有被指出的问题
3. 确保无害、诚实、有帮助、公平
4. 如果原回复中某个部分没问题,保留它

修订后的回复:"""

        return self._generate(revision_prompt, temperature=self.config.revision_temperature)

    def _extract_violations(self, critique_text: str) -> List[str]:
        """从批评文本中提取违规项。"""
        violations = []
        # 检测是否有违规
        if "未发现违规" in critique_text or "没有违反" in critique_text:
            return violations

        # 简单提取: 找 "违反" 相关的行
        for line in critique_text.split("\n"):
            line = line.strip()
            if any(kw in line for kw in ["违反", "违规", "问题", "不符合", "不满足"]):
                violations.append(line[:200])
        return violations

    def _similarity(self, a: str, b: str) -> float:
        """简单文本相似度 (Jaccard on words)。"""
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _generate(self, prompt: str, temperature: float = 0.7) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 只返回新生成的部分
        prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        return full[prompt_len:].strip()

    # ============================================================
    # 合成训练数据: CAI 自动生成高质量 SFT 数据
    # ============================================================

    def generate_sft_data(
        self,
        prompts: List[str],
        samples_per_prompt: int = 2,
    ) -> List[Dict]:
        """
        用 CAI 自动生成高质量 SFT 训练数据。

        不需要人工写答案! 模型自己生成并改进。

        返回:
          [{"prompt": "...", "response": "...", "critique_rounds": ..., "quality": "high"}, ...]
        """
        import torch
        data = []

        for i, prompt in enumerate(prompts):
            logger.info(f"CAI-SFT [{i+1}/{len(prompts)}]: {prompt[:50]}...")

            for _ in range(samples_per_prompt):
                result = self.generate_with_cai(prompt)
                quality = "high" if not result["violations_found"] else "medium"
                data.append({
                    "prompt": prompt,
                    "response": result["final_response"],
                    "initial_response": result["initial_response"],
                    "critique_rounds": len(result["critique_rounds"]),
                    "quality": quality,
                })

        logger.info(f"CAI-SFT 完成: 生成 {len(data)} 条训练数据")
        return data


# ============================================================
# Part 2: Process Reward Model (OpenAI o1 核心技术)
# ============================================================

@dataclass
class PRMConfig:
    step_markers: List[str] = field(default_factory=lambda: [
        "第一步", "第二步", "第三步", "第四步", "第五步",
        "Step 1", "Step 2", "Step 3", "Step 4", "Step 5",
        "首先", "其次", "然后", "最后",
        "1.", "2.", "3.", "4.", "5.",
        "\n\n",  # 段落分隔
    ])
    score_per_step: bool = True         # 逐步骤打分
    score_final_answer: bool = True     # 最终答案也打分
    min_steps: int = 2
    max_steps: int = 10
    default_score: float = 0.5


class ProcessRewardModel:
    """
    Process Reward Model — OpenAI o1 的核心推理增强技术。

    与 Outcome Reward Model 的区别:
      Outcome RM: 只看最终答案对不对
      Process RM: 对推理的每一步都打分 → 找到推理过程中的错误

    为什么重要:
      - 数学推理: 每一步都需要正确, 而不是猜对最终答案
      - 代码生成: eval 能判断最终对错, 但 PRM 能定位哪一步逻辑错了
      - 逻辑链: PRM 能发现推理中断裂的环节

    用法:
        prm = ProcessRewardModel(model, tokenizer)

        # 方式1: 基于规则 (不需要训练)
        scores = prm.score_steps_by_rules(
            "问题: 1+2×3=?",
            ["Step1: 先算乘法 2×3=6", "Step2: 再算加法 1+6=7", "Answer: 7"]
        )

        # 方式2: 基于模型自检 (模型自己判断每步是否正确)
        scores = prm.score_steps_by_self_check(
            "问题: 1+2×3=?",
            ["Step1: 先算乘法 2×3=6", "Step2: 再算加法 1+6=7", "Answer: 7"]
        )
    """

    def __init__(self, model=None, tokenizer=None, config: PRMConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PRMConfig()
        self.device = next(model.parameters()).device if model else None

    def split_into_steps(self, text: str) -> List[str]:
        """将文本自动拆分为推理步骤。"""
        steps = []
        remaining = text

        # 尝试用已知标记分割
        best_split = None
        best_count = 0

        for marker in self.config.step_markers:
            if marker in remaining:
                parts = remaining.split(marker)
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) > best_count:
                    best_split = parts
                    best_count = len(parts)

        if best_split and best_count >= self.config.min_steps:
            return best_split

        # Fallback: 按句子分
        sentences = re.split(r"[。.!！?\n]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= self.config.min_steps:
            return sentences

        # 最后: 按长度均分
        chunk_size = max(len(text) // self.config.min_steps, 20)
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    def score_steps_by_rules(self, question: str, steps: List[str]) -> List[Dict]:
        """
        基于规则对推理步骤打分 (不需要额外模型)。

        规则:
          - 包含数字/计算: +0.1
          - 包含逻辑连接词: +0.1
          - 步骤之间有引用关系: +0.1
          - 太短(没内容): -0.2
          - 重复上一步: -0.1
        """
        scores = []
        prev_step = ""

        for i, step in enumerate(steps):
            score = 0.5  # 默认中性

            # 正向指标
            if re.search(r"\d+", step):
                score += 0.1  # 有具体数字
            if any(kw in step for kw in ["因为", "所以", "因此", "由于", "according to", "because", "therefore"]):
                score += 0.1  # 逻辑推理
            if any(kw in step for kw in ["等于", "结果是", "答案是", "=", "result", "answer"]):
                score += 0.1  # 有明确结论
            if prev_step and self._has_overlap(step, prev_step.split()[-5:]):
                score += 0.1  # 与上一步有衔接

            # 负向指标
            if len(step) < 10:
                score -= 0.2  # 太短
            if prev_step and self._similarity(step, prev_step) > 0.7:
                score -= 0.1  # 与上一步高度重复

            # 最终步骤特殊处理
            if i == len(steps) - 1:
                if any(kw in step for kw in ["答案", "结果", "综上", "answer", "result", "conclusion"]):
                    score += 0.2  # 最后一步是明确结论
                if re.search(r"\d+", step):
                    score += 0.1  # 最终答案有具体数值

            score = max(0.0, min(1.0, score))
            scores.append({
                "step_index": i,
                "step_text": step[:200],
                "score": round(score, 3),
                "is_final": i == len(steps) - 1,
            })
            prev_step = step

        return scores

    def score_steps_by_self_check(self, question: str, steps: List[str]) -> List[Dict]:
        """
        让模型自己检查每一步推理 (需要模型)。

        模型逐步骤判断: 这一步的推理是否正确?
        """
        if self.model is None:
            logger.warning("未提供 model, 回退到规则打分")
            return self.score_steps_by_rules(question, steps)

        import torch
        self.model.eval()
        scores = []

        for i, step in enumerate(steps):
            # 构建检查提示
            check_prompt = f"""你是一个严格的数学和逻辑审查者。请判断以下推理步骤是否正确。

问题: {question}

前面的步骤:
{chr(10).join(f'- {s[:200]}' for s in steps[:i]) if i > 0 else '(这是第一步)'}

当前步骤 [{i+1}/{len(steps)}]:
{step[:500]}

请判断这一步骤:
1. 逻辑是否正确?
2. 计算是否准确?
3. 是否与前面步骤一致?

只回答: "正确" 或 "错误", 然后简短说明原因。"""

            inputs = self.tokenizer(check_prompt, return_tensors="pt", truncation=True,
                                     max_length=2048).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=100, temperature=0.3, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 解析判断
            is_correct = any(kw in judgment for kw in ["正确", "correct", "Correct", "是"])
            is_wrong = any(kw in judgment for kw in ["错误", "incorrect", "Incorrect", "否", "不对"])

            if is_correct and not is_wrong:
                score = 1.0
            elif is_wrong and not is_correct:
                score = 0.0
            else:
                score = 0.5  # 不确定

            scores.append({
                "step_index": i,
                "step_text": step[:200],
                "score": score,
                "judgment": judgment[:200],
                "is_correct": is_correct,
                "is_final": i == len(steps) - 1,
            })

        return scores

    def aggregate_scores(self, step_scores: List[Dict], method: str = "min") -> float:
        """聚合各步骤分数为整体分数。

        - min: 最弱环节 (最严格,推理链中最弱一步决定整体)
        - mean: 平均
        - prod: 乘积 (每步都需要正确)
        """
        scores = [s["score"] for s in step_scores]
        if not scores:
            return 0.0
        if method == "min":
            return min(scores)
        elif method == "prod":
            result = 1.0
            for s in scores:
                result *= s
            return result
        else:  # mean
            return sum(scores) / len(scores)

    def filter_best_reasoning(
        self,
        question: str,
        candidate_reasonings: List[List[str]],
        threshold: float = 0.7,
    ) -> List[Dict]:
        """
        从多个推理候选中选择过程最正确的。

        用于训练数据筛选: 只保留推理步骤都正确的样本做 SFT。
        """
        results = []
        for i, steps in enumerate(candidate_reasonings):
            step_scores = self.score_steps_by_rules(question, steps)
            min_score = self.aggregate_scores(step_scores, method="min")

            results.append({
                "candidate_index": i,
                "steps": steps,
                "min_step_score": min_score,
                "accepted": min_score >= threshold,
                "step_scores": step_scores,
            })

        results.sort(key=lambda x: x["min_step_score"], reverse=True)
        return results

    def _has_overlap(self, text: str, tokens: List[str]) -> bool:
        """检查 text 是否包含 tokens 中的词。"""
        text_lower = text.lower()
        return any(t.lower() in text_lower for t in tokens[:3])

    def _similarity(self, a: str, b: str) -> float:
        set_a = set(a.split()); set_b = set(b.split())
        if not set_a or not set_b: return 0.0
        return len(set_a & set_b) / len(set_a | set_b)


# ============================================================
# Part 3: Rejection Sampling (OpenAI + Anthropic 都在用)
# ============================================================

@dataclass
class RejectionSamplingConfig:
    num_candidates: int = 8           # 每个 prompt 生成多少个候选
    temperature: float = 0.8          # 生成多样性温度
    top_p: float = 0.95
    max_tokens: int = 1024
    reward_method: str = "self_check" # rules | self_check | combined
    keep_top_k: int = 1               # 保留最佳几个
    min_reward_threshold: float = 0.5  # 最低分数阈值


class RejectionSampling:
    """
    Rejection Sampling — 生成多个候选, 选最好的。

    OpenAI 用法: 用大模型生成候选 → reward model 打分 → 选最好的做 SFT 训练数据
    Anthropic 用法: 用 Constitution 规则筛选 → 选最无害的

    组合效果:
      生成 8 个候选 → PRM 筛选推理正确的 → CAI 过滤有害的 → 最佳 1-2 个

    用法:
        rs = RejectionSampling(model, tokenizer)

        # Step 1: 生成多个候选
        candidates = rs.generate_candidates("解释一下量子力学")

        # Step 2: 用 PRM 打分
        prm = ProcessRewardModel(model, tokenizer)
        best = rs.select_best(candidates, prm=prm, method="prm")

        # 或者用 CAI 规则打分 (不需要额外模型)
        best = rs.select_best(candidates, constitution=DEFAULT_CONSTITUTION)
    """

    def __init__(self, model, tokenizer, config: RejectionSamplingConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RejectionSamplingConfig()
        self.device = next(model.parameters()).device

    def generate_candidates(self, prompt: str) -> List[str]:
        """为一个 prompt 生成多个候选回复。"""
        import torch
        self.model.eval()
        candidates = []

        for _ in range(self.config.num_candidates):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                     max_length=2048).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            candidates.append(full[prompt_len:].strip())

        return candidates

    def select_best(
        self,
        candidates: List[str],
        prompt: str = "",
        prm: Optional["ProcessRewardModel"] = None,
        constitution: Optional[List[str]] = None,
        reward_fn: Optional[Callable] = None,
        method: str = "combined",
    ) -> Dict:
        """
        从候选中选择最佳回复。

        方法:
          - "prm": 用 Process Reward Model 逐步骤打分
          - "constitution": 用 Constitutional AI 规则检测违规
          - "rules": 用启发式规则 (长度、格式、内容)
          - "combined": 综合所有方法
          - "custom": 自定义 reward_fn

        返回:
          {
            "best_response": "...",
            "best_score": 0.85,
            "ranked_candidates": [...],
            "selection_method": "combined",
          }
        """
        scored = []

        for i, candidate in enumerate(candidates):
            scores = {}

            # PRM 打分
            if prm and method in ("prm", "combined"):
                prm_scores = prm.score_steps_by_rules(prompt, prm.split_into_steps(candidate))
                scores["prm"] = max(
                    s["score"] for s in prm_scores
                ) if prm_scores else 0.5

            # Constitution 规则打分
            if constitution and method in ("constitution", "combined"):
                scores["constitution"] = self._constitution_score(candidate, constitution)

            # 启发式规则
            if method in ("rules", "combined"):
                scores["rules"] = self._rule_based_score(candidate)

            # 自定义
            if reward_fn:
                scores["custom"] = reward_fn(prompt, candidate)

            # 加权综合
            if scores:
                total = sum(scores.values()) / len(scores)
            else:
                total = 0.5

            scored.append({
                "index": i,
                "response": candidate,
                "total_score": round(total, 3),
                "component_scores": scores,
            })

        scored.sort(key=lambda x: x["total_score"], reverse=True)

        # 过滤低于阈值
        accepted = [s for s in scored if s["total_score"] >= self.config.min_reward_threshold]

        result = {
            "best_response": accepted[0]["response"] if accepted else scored[0]["response"],
            "best_score": accepted[0]["total_score"] if accepted else scored[0]["total_score"],
            "num_accepted": len(accepted),
            "num_candidates": len(candidates),
            "acceptance_rate": len(accepted) / len(candidates),
            "selection_method": method,
            "ranked_candidates": scored[:self.config.keep_top_k],
        }

        return result

    def _constitution_score(self, text: str, constitution: List[str]) -> float:
        """基于宪法规则打分: 违反越少分越高。"""
        violations = 0
        for rule in constitution[:5]:  # 随机选 5 条
            keywords = self._extract_keywords(rule)
            text_lower = text.lower()
            for kw in keywords:
                if kw.lower() in text_lower:
                    violations += 1
        penalty = min(violations * 0.1, 0.5)
        return 1.0 - penalty

    def _rule_based_score(self, text: str) -> float:
        """启发式质量评分。"""
        score = 0.5
        n = len(text)

        # 长度: 太短不好, 太长也不好
        if n > 100:
            score += 0.1
        if n > 500:
            score += 0.1
        if n > 3000:
            score -= 0.1

        # 结构化程度
        if any(marker in text for marker in ["1.", "2.", "首先", "其次", "第一", "第二", "Step"]):
            score += 0.1

        # 内容质量
        if any(kw in text for kw in ["因为", "所以", "因此", "根据", "例如", "具体来说"]):
            score += 0.1

        # 负面指标
        if any(kw in text.lower() for kw in ["我不知道", "我无法", "作为ai", "抱歉"]):
            score -= 0.1
        if len(text) < 50:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _extract_keywords(self, rule: str) -> List[str]:
        """从规则中提取关键词。"""
        # 移除常见停用词后的关键词
        stop = {"请", "不要", "的", "和", "或", "在", "与", "是", "有", "对", "了", "为"}
        words = rule.replace("。", "").replace("，", "").replace("、", "").split()
        return [w for w in words if w not in stop and len(w) > 1]


# ============================================================
# Part 4: 组合流水线 — CAI + PRM + Rejection Sampling
# ============================================================

class FrontierAlignmentPipeline:
    """
    前沿对齐流水线: 一步到位使用 OpenAI + Anthropic 的最佳实践。

    流程:
      1. Rejection Sampling → 生成 N 个候选
      2. PRM 筛选 → 保留推理步骤正确的
      3. CAI 过滤 → 移除不安全的
      4. 输出最佳回复 + 高质量 SFT 训练数据

    用法:
        pipeline = FrontierAlignmentPipeline(model, tokenizer)

        # 单条推理
        result = pipeline.query("解释相对论")
        print(result["best_response"])

        # 批量生成 SFT 训练数据
        data = pipeline.generate_training_data(prompts, samples_per_prompt=3)
    """

    def __init__(self, model, tokenizer, configs: Dict = None):
        self.model = model
        self.tokenizer = tokenizer

        cfg = configs or {}
        self.rs = RejectionSampling(model, tokenizer, cfg.get("rs"))
        self.prm = ProcessRewardModel(model, tokenizer, cfg.get("prm"))
        self.cai = ConstitutionalAI(model, tokenizer, cfg.get("cai"))
        self.device = next(model.parameters()).device

    def query(self, prompt: str, num_candidates: int = 8) -> Dict:
        """通过完整流水线生成高质量回复。"""
        import torch

        # Stage 1: Rejection Sampling
        candidates = self.rs.generate_candidates(prompt)
        if not candidates:
            return {"best_response": "", "error": "生成失败"}

        # Stage 2: PRM 打分 + 筛选
        prm_results = []
        for i, cand in enumerate(candidates):
            steps = self.prm.split_into_steps(cand)
            step_scores = self.prm.score_steps_by_rules(prompt, steps)
            min_score = self.prm.aggregate_scores(step_scores, method="min")
            prm_results.append({
                "index": i, "response": cand, "prm_min_score": min_score, "steps": steps, "num_steps": len(steps),
            })

        prm_results.sort(key=lambda x: x["prm_min_score"], reverse=True)

        # 取 PRM 分数最高的 top-k
        top_k = min(3, len(prm_results))
        top_candidates = [r["response"] for r in prm_results[:top_k]]

        # Stage 3: CAI 自我批评 + 修订
        cai_results = []
        for cand in top_candidates:
            result = self.cai.generate_with_cai(prompt, cand)
            violations = len(result["violations_found"])
            cai_results.append({
                "response": result["final_response"],
                "violations": violations,
                "revision_rounds": len(result["critique_rounds"]),
            })

        # 选违规最少的
        cai_results.sort(key=lambda x: x["violations"])
        best = cai_results[0]

        return {
            "prompt": prompt,
            "best_response": best["response"],
            "num_candidates_generated": len(candidates),
            "prm_top_score": prm_results[0]["prm_min_score"] if prm_results else 0,
            "cai_violations": best["violations"],
            "cai_revision_rounds": best["revision_rounds"],
            "pipeline_stages": ["rejection_sampling", "prm_filter", "cai_refinement"],
        }

    def generate_training_data(
        self,
        prompts: List[str],
        samples_per_prompt: int = 2,
        quality_threshold: float = 0.6,
    ) -> List[Dict]:
        """
        批量生成高质量 SFT 训练数据。

        使用所有三种技术确保数据质量。
        """
        training_data = []

        for i, prompt in enumerate(prompts):
            logger.info(f"FrontierAlign [{i+1}/{len(prompts)}]")

            for _ in range(samples_per_prompt):
                result = self.query(prompt, num_candidates=6)

                if result.get("prm_top_score", 0) >= quality_threshold:
                    training_data.append({
                        "prompt": prompt,
                        "response": result["best_response"],
                        "quality_score": result["prm_top_score"],
                        "source": "frontier_align_pipeline",
                        "techniques": "rejection_sampling + prm + cai",
                    })

        logger.info(f"生成 {len(training_data)} 条高质量 SFT 数据 (阈值={quality_threshold})")
        return training_data


# Fix for torch import used in ConstitutionalAI._generate
import torch
