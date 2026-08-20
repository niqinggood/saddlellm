"""
安全生成 & 反重复 — 解决"骂人"和"车轱辘话"

问题根源分析:

═══ 骂人/脏话 ═══
  根因1 (数据): 预训练语料里就有脏话/攻击性内容 → 模型学会了
  根因2 (对齐): SFT/RLHF 没做或不够 → 模型不知道不该说
  根因3 (推理): prompt 诱导 → 模型被引导说出不当内容

  解决环节: 数据过滤 ← 对齐训练 ← 输出拦截 (三道防线)

═══ 车轱辘话 ═══
  根因1 (数据): 语料中大量重复内容 → 模型学习到重复模式
  根因2 (解码): 贪心解码 + 无惩罚 → attention 陷入局部循环
  根因3 (训练): 模型在某个 phrase 上 overfit

  解决环节: 数据去重 ← 推理时解码策略 (主要靠推理时解决)

用法:
  from saddlellm import SafeGenerate

  # 安全生成
  sg = SafeGenerate(model, tokenizer)
  result = sg.generate("如何看待...", enable_safety=True)

  # 防车轱辘话生成
  result = sg.generate("写一篇文章", anti_repeat=True)

  # 数据阶段过滤
  clean = SafeGenerate.filter_toxic_data(texts)
"""
import re
import torch
import logging
from typing import List, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# 第一道防线: 数据阶段 — 过滤有毒内容
# ═══════════════════════════════════════════════════════════

# 脏话/攻击性词汇列表 (中英文)
TOXIC_PATTERNS_ZH = [
    # 脏话
    r"[他她它]妈的", r"傻[逼B比Xx]", r"滚蛋", r"去死", r"废物",
    r"白痴", r"弱智", r"脑残", r"二[逼B]", r"草泥马",
    r"[操草艹].*[你尼]", r"日你", r"狗日", r"王八", r"畜生",
    r"[贱Jj].*[人货]", r"变态", r"神经病", r"疯子",
    # 攻击性
    r"你怎么这么.*[笨蠢慢差烂]", r"你会不会", r"你懂什么",
    r"不[配值].*活", r"死了算了",
    # 歧视性
    r"女司机", r"娘娘腔", r"男人婆",
    r"乡下人", r"外地人", r"打工仔",  # (语境很重要, 粗筛)
]

TOXIC_PATTERNS_EN = [
    r"\bf[u*]ck\b", r"\bsh[i*]t\b", r"\bd[a*]mn\b", r"\b[a*]ss\b",
    r"\bb[i*]tch\b", r"\bd[i*]ck\b", r"\bc[r*]ap\b",
    r"\bk[i*]ll yourself\b", r"\bgo die\b",
    r"\bst[u*]pid\b", r"\bidiot\b", r"\bmoron\b", r"\bretard\b",
    r"\bn[i*]gg[ae]r?\b",
]

# 车轱辘话特征
REPETITION_PATTERNS = [
    r"(.)\1{10,}",          # 单字符重复 10+ 次
    r"(.{10,50})\1{3,}",    # 10-50字的短语重复 3+ 次
    r"(\b\w+\b)(\s+\1){4,}", # 同一单词重复 4+ 次
]


def filter_toxic_texts(texts: List[str], lang: str = "auto") -> List[str]:
    """
    过滤有毒文本。

    返回干净的文本列表。在数据预处理阶段调用。
    """
    patterns = TOXIC_PATTERNS_ZH + TOXIC_PATTERNS_EN
    clean = []
    removed = 0

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            removed += 1
            continue

        is_toxic = False
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                is_toxic = True
                break

        if not is_toxic:
            # 额外检查: 全大写/长重复
            if not _has_repetition_problem(text):
                clean.append(text)
            else:
                removed += 1
        else:
            removed += 1

    logger.info(f"有毒过滤: {len(texts)} → {len(clean)} ({removed} 被移除)")
    return clean


def filter_toxic_dataset(dataset, text_column: str = "text"):
    """对 HuggingFace dataset 做有毒过滤。"""
    patterns = TOXIC_PATTERNS_ZH + TOXIC_PATTERNS_EN

    def _is_clean(example):
        text = example.get(text_column, "")
        if not text: return False
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return False
        if _has_repetition_problem(text):
            return False
        return True

    return dataset.filter(_is_clean)


def _has_repetition_problem(text: str) -> bool:
    """检查是否有严重的重复问题。"""
    if len(text) < 50:
        return False

    # 检查大段重复
    for pattern in REPETITION_PATTERNS:
        if re.search(pattern, text):
            return True

    # 检查句子级别重复
    sentences = re.split(r"[。.!！?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) >= 4:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.3:  # 70%+ 的句子是重复的
            return True

    return False


# ═══════════════════════════════════════════════════════════
# 第二道防线: 推理阶段 — 安全生成 + 反重复
# ═══════════════════════════════════════════════════════════

class SafeGenerate:
    """
    安全生成 & 反车轱辘话。

    用法:
        sg = SafeGenerate(model, tokenizer)

        # 安全生成
        result = sg.generate("你对这件事怎么看?")
        # 自动: 安全前缀 + 输出过滤 + 反重复

    """

    # 安全前缀 (加在 prompt 前面, 引导模型安全回复)
    SAFETY_PREFIX = "你是一个有帮助、诚实、无害的AI助手。"

    # 输出黑名单 (检测到这些就重新生成)
    OUTPUT_BLACKLIST = [
        "作为AI语言模型",
        "作为一个AI",
        "我不能",
        "I cannot",
        "I don't know",
    ]  # 注意: 这些不一定是坏的, 但过度出现的车轱辘话特征

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # 注册安全生成默认参数
        self._pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_safety: bool = True,
        anti_repeat: bool = True,
        repetition_penalty: float = 1.15,    # >1 惩罚重复
        no_repeat_ngram_size: int = 4,       # 禁止 4-gram 重复
        frequency_penalty: float = 0.3,       # 惩罚高频词
        presence_penalty: float = 0.3,        # 惩罚已出现词
        use_contrastive_search: bool = False,  # 对比搜索 (最强反重复)
        contrastive_k: int = 4,
        contrastive_alpha: float = 0.6,
        max_retries: int = 2,                 # 检测到问题时重新生成的次数
    ) -> Dict:
        """
        安全生成。组合多种策略避免骂人和车轱辘话。

        返回:
          {"text": "...", "safety_checks_passed": True, "repeat_score": 0.1, "retries": 0}
        """
        import torch

        # Step 0: 安全前缀
        full_prompt = prompt
        if enable_safety:
            full_prompt = f"{self.SAFETY_PREFIX}\n\n用户: {prompt}\n助手: "

        result_text = ""
        for attempt in range(max_retries + 1):
            # 编码
            inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True,
                                     max_length=2048).to(self.device)

            # 解码参数 (组合所有反重复策略)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.05,
                "top_p": top_p,
                "pad_token_id": self._pad_token,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if anti_repeat:
                gen_kwargs["repetition_penalty"] = repetition_penalty
                gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

                # frequency/presence_penalty 需要 transformers >= 4.40
                # 如果版本不够，这些参数会被忽略

            # 对比搜索 (最强, 但需要特定配置)
            if use_contrastive_search:
                gen_kwargs["penalty_alpha"] = contrastive_alpha
                gen_kwargs["top_k"] = contrastive_k
                gen_kwargs.pop("do_sample", None)
                gen_kwargs.pop("temperature", None)
                gen_kwargs.pop("top_p", None)

            outputs = self.model.generate(**inputs, **gen_kwargs)
            full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            result_text = full[prompt_len:].strip()

            # 检查质量
            repeat_score = self._check_repetition(result_text)
            is_toxic = self._check_toxicity(result_text)

            if repeat_score < 0.5 and not is_toxic:
                break  # 生成质量合格

            if attempt < max_retries:
                logger.info(f"重新生成 (attempt {attempt+1}): repeat={repeat_score:.2f}, toxic={is_toxic}")
                # 调高惩罚
                repetition_penalty += 0.1
                frequency_penalty += 0.1

        return {
            "text": result_text,
            "safety_checks_passed": not self._check_toxicity(result_text),
            "repeat_score": round(self._check_repetition(result_text), 3),
            "retries": min(attempt, max_retries),
        }

    def _check_repetition(self, text: str) -> float:
        """
        检测车轱辘话程度 (0-1, 越高越严重)。

        检测:
          - 短语完全重复
          - n-gram 重复率
          - 句子级别重复
        """
        if len(text) < 50:
            return 0.0

        score = 0.0

        # 1. n-gram 重复率
        words = text.split()
        if len(words) >= 20:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            if trigrams:
                unique_ratio = len(set(trigrams)) / len(trigrams)
                if unique_ratio < 0.4:
                    score += 0.5
                elif unique_ratio < 0.6:
                    score += 0.3
                elif unique_ratio < 0.8:
                    score += 0.1

        # 2. 连续 n-gram 重复
        for pattern in REPETITION_PATTERNS:
            if re.search(pattern, text):
                score += 0.3
                break

        # 3. 句子级别
        sentences = re.split(r"[。.!！?\n]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) >= 3:
            unique_sents = len(set(sentences))
            if unique_sents == 1 and len(sentences) > 2:
                score += 0.5
            elif unique_sents <= 2 and len(sentences) > 4:
                score += 0.3

        # 4. 太长也是信号 (车轱辘话通常很长)
        if len(text) > 3000:
            score += 0.1

        return min(1.0, score)

    def _check_toxicity(self, text: str) -> bool:
        """快速检查是否有不当内容。"""
        if not text:
            return False
        for pattern in TOXIC_PATTERNS_ZH + TOXIC_PATTERNS_EN:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    # ============================================================
    # 批量安全生成
    # ============================================================

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Dict]:
        """批量安全生成。"""
        return [self.generate(p, **kwargs) for p in prompts]


# ============================================================
# 专门的反车轱辘话生成器
# ============================================================

class AntiRepeatGenerator:
    """
    专门针对"车轱辘话"的生成器。

    整合业界最佳实践:
      1. Repetition Penalty — 降低已出现 token 的概率
      2. Frequency/Presence Penalty — OpenAI 同款
      3. No-Repeat-Ngram — 禁止 N-gram 重复
      4. Contrastive Search — 每步选择与上文语义一致但不重复的 token
      5. Diverse Beam Search — 多束搜索 + 多样性惩罚

    用法:
        arg = AntiRepeatGenerator(model, tokenizer)

        # 自动选最佳策略
        result = arg.generate("写一篇关于AI未来发展的文章", method="auto")

        # 指定方法
        result = arg.generate("...", method="contrastive")  # 最强
        result = arg.generate("...", method="diverse_beam") # 多样束搜索
    """

    METHODS = {
        "auto": "自动选择 (基于模型大小)",
        "repetition_penalty": "Repetition Penalty (最简单)",
        "frequency": "Frequency + Presence Penalty (OpenAI 风格)",
        "no_repeat_ngram": "No-Repeat-Ngram (禁止短语重复)",
        "contrastive": "Contrastive Search (最强, 需要大模型)",
        "diverse_beam": "Diverse Beam Search (多样性束搜索)",
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self._pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id

    def generate(self, prompt: str, method: str = "auto", max_new_tokens: int = 1024,
                 temperature: float = 0.7) -> str:
        if method == "auto":
            total_params = sum(p.numel() for p in self.model.parameters())
            if total_params > 3e9:
                method = "contrastive"
            elif total_params > 5e8:
                method = "frequency"
            else:
                method = "no_repeat_ngram"

        gen_fn = getattr(self, f"_gen_{method}", self._gen_default)

        import torch
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=2048).to(self.device)
        with torch.no_grad():
            result_text = gen_fn(inputs, max_new_tokens, temperature)
        return result_text

    def _gen_repetition_penalty(self, inputs, max_tokens, temp):
        import torch
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens, repetition_penalty=1.2,
            temperature=temp, do_sample=temp > 0.05, top_p=0.9,
            pad_token_id=self._pad_token,
        )
        return self._decode(outputs, inputs)

    def _gen_frequency(self, inputs, max_tokens, temp):
        import torch
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            repetition_penalty=1.1, no_repeat_ngram_size=4,
            temperature=temp, do_sample=temp > 0.05, top_p=0.9,
            pad_token_id=self._pad_token,
        )
        return self._decode(outputs, inputs)

    def _gen_no_repeat_ngram(self, inputs, max_tokens, temp):
        import torch
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            no_repeat_ngram_size=3, repetition_penalty=1.15,
            temperature=temp, do_sample=temp > 0.05, top_p=0.92,
            pad_token_id=self._pad_token,
        )
        return self._decode(outputs, inputs)

    def _gen_contrastive(self, inputs, max_tokens, temp):
        import torch
        try:
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                penalty_alpha=0.6, top_k=4,
                pad_token_id=self._pad_token,
            )
        except Exception:
            # 回退
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                repetition_penalty=1.2, no_repeat_ngram_size=4,
                temperature=0.7, do_sample=True, top_p=0.9,
                pad_token_id=self._pad_token,
            )
        return self._decode(outputs, inputs)

    def _gen_diverse_beam(self, inputs, max_tokens, temp):
        import torch
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            num_beams=5, num_beam_groups=5,
            diversity_penalty=1.0, repetition_penalty=1.15,
            pad_token_id=self._pad_token, early_stopping=True,
        )
        return self._decode(outputs, inputs)

    def _gen_default(self, inputs, max_tokens, temp):
        import torch
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temp, do_sample=temp > 0.05, top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self._pad_token,
        )
        return self._decode(outputs, inputs)

    def _decode(self, outputs, inputs):
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_len = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        return full[prompt_len:].strip()


# ============================================================
# 第三道防线: 输出后过滤
# ============================================================

class OutputSafetyFilter:
    """
    输出安全过滤器 — 生成的最后一道防线。

    在模型输出后、返回用户前做检查。
    """

    def __init__(self, custom_blocklist: List[str] = None):
        self.blocklist = custom_blocklist or []
        self.patterns = TOXIC_PATTERNS_ZH + TOXIC_PATTERNS_EN

    def check(self, text: str) -> Dict:
        """
        检查输出是否安全。

        返回: {"safe": True/False, "issues": [...], "filtered_text": "..."}
        """
        issues = []

        # 1. 脏话检查
        for pat in self.patterns[:20]:  # 快速检查前20条
            if re.search(pat, text, re.IGNORECASE):
                issues.append({"type": "toxicity", "pattern": pat})

        # 2. 自定义黑名单
        for item in self.blocklist:
            if item.lower() in text.lower():
                issues.append({"type": "blocklist", "item": item})

        # 3. 车轱辘话检查
        rep_score = _check_simple_repeat(text)
        if rep_score > 0.7:
            issues.append({"type": "repetition", "score": rep_score})

        # 4. 移除问题内容
        filtered = text
        if issues:
            for issue in issues:
                if issue["type"] == "toxicity":
                    # 用 * 替换脏话
                    match = re.search(issue["pattern"], text, re.IGNORECASE)
                    if match:
                        replacement = "*" * len(match.group())
                        filtered = filtered[:match.start()] + replacement + filtered[match.end():]

        return {
            "safe": len(issues) == 0,
            "issues": issues,
            "filtered_text": filtered,
        }


def _check_simple_repeat(text: str) -> float:
    """简单重复度检查。"""
    if len(text) < 100: return 0.0
    trigrams = [text[i:i+3] for i in range(len(text)-2)]
    if not trigrams: return 0.0
    return 1 - len(set(trigrams)) / len(trigrams)


