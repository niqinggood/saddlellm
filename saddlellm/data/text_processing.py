import os
import json
import unicodedata
import logging
import random
import ftfy
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import zipfile
import zstandard as zstd
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import jieba
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import html
import regex as re
from unidecode import unidecode
from simhash import Simhash

# 初始化设置
DetectorFactory.seed = 0  # 确保语言检测结果一致

# 配置日志
logger = logging.getLogger(__name__)

__all__ = ["DataSourceHandler", "TextCleaner", "TextProcessor", "load_config", "main"]


class TextCleaner:
    """文本清洗与规范化处理器"""

    # 常用正则表达式预编译
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    PHONE_PATTERN = re.compile(
        r"\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b"
    )
    SPECIAL_CHARS = re.compile(
        r"[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]", re.UNICODE
    )
    MULTISPACE = re.compile(r"\s+")
    CITATION_PATTERN = re.compile(r"\[(\d+)\]|\([^)]*?\d{4}[^)]*?\)")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    JS_PATTERN = re.compile(
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", re.IGNORECASE
    )
    CSS_PATTERN = re.compile(
        r"<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>", re.IGNORECASE
    )
    TABLE_PATTERN = re.compile(r"<table\b[^>]*>.*?<\/table>", re.DOTALL)
    HEADER_FOOTER_PATTERN = re.compile(
        r"^\s*(header|footer)\b.*?\n|\n\s*(header|footer)\b.*?$",
        re.IGNORECASE | re.MULTILINE,
    )
    PUNCTUATION_MAP = str.maketrans(
        "，。！？；：“”‘’（）【】、％＃＠＆１２３４５６７８９０",
        ",.!?;:\"\"''()[]%#@&1234567890",
    )

    # 网络用语替换表
    SLANG_MAP = {
        "btw": "by the way",
        "lol": "laugh out loud",
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "afaik": "as far as i know",
        "tbh": "to be honest",
        "idk": "i don't know",
        "smh": "shaking my head",
        "rofl": "rolling on the floor laughing",
        "ily": "i love you",
        "thx": "thanks",
        "plz": "please",
        "omg": "oh my god",
        "nvm": "never mind",
        "fyi": "for your information",
        "asap": "as soon as possible",
        "atm": "at the moment",
        "brb": "be right back",
        "gtg": "got to go",
        "irl": "in real life",
        "jk": "just kidding",
        "np": "no problem",
        "tmi": "too much information",
        "yolo": "you only live once",
        "gg": "good game",
        "gl": "good luck",
        "hf": "have fun",
        "wb": "welcome back",
        "nm": "not much",
        "ty": "thank you",
        "yw": "you're welcome",
        "sry": "sorry",
        "sup": "what's up",
        "lmao": "laughing my ass off",
        "ic": "i see",
        "icymi": "in case you missed it",
        "tbf": "to be fair",
        "ttyl": "talk to you later",
        "wbu": "what about you",
        "wya": "where you at",
        "wyd": "what you doing",
        "ykyk": "you know you know",
        "fwiw": "for what it's worth",
    }

    # 中文网络用语替换表
    CN_SLANG_MAP = {
        "酱紫": "这样子",
        "肿么": "怎么",
        "木有": "没有",
        "神马": "什么",
        "有木有": "有没有",
        "灰常": "非常",
        "造": "知道",
        "表": "不要",
        "介个": "这个",
        "内个": "那个",
        "肿么办": "怎么办",
        "好哒": "好的",
        "好滴": "好的",
        "阔以": "可以",
        "孩纸": "孩子",
        "银": "人",
        "森么": "什么",
        "桑心": "伤心",
        "捉急": "着急",
        "辣么": "那么",
        "奏是": "就是",
        "咩": "什么",
        "腻害": "厉害",
        "造吗": "知道吗",
        "造了": "知道了",
        "造啥": "知道什么",
        "造不": "知道不",
        "造呀": "知道呀",
        "造啦": "知道啦",
        "造嘛": "知道嘛",
        "造呗": "知道呗",
        "造咯": "知道咯",
        "造哈": "知道哈",
        "造哟": "知道哟",
        "造喔": "知道喔",
        "造噢": "知道噢",
        "造诶": "知道诶",
        "造嘞": "知道嘞",
        "造咧": "知道咧",
        "造捏": "知道捏",
        "造呐": "知道呐",
        "造呢": "知道呢",
        "造吧": "知道吧",
        "造啊": "知道啊",
    }

    # 中文数字转阿拉伯数字
    CN_NUM_MAP = {
        "零": 0,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "百": 100,
        "千": 1000,
        "万": 10000,
        "亿": 100000000,
        "两": 2,
    }

    def __init__(self, config: Dict):
        self.config = config
        self._init_regex_patterns()
        self.sentence_model = (
            SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            if config.get("extract_main_content", False)
            else None
        )

    def _init_regex_patterns(self):
        """根据配置初始化正则表达式模式"""
        # 自定义特殊字符处理
        if (
            self.config.get("remove_special_chars") == "custom"
            and "custom_special_chars" in self.config
        ):
            custom_chars = re.escape(self.config["custom_special_chars"])
            self.CUSTOM_SPECIAL_CHARS = re.compile(
                f"[^{custom_chars}\\w\\s]", re.UNICODE
            )

        # 初始化黑名单关键词
        if "blacklist_keywords" in self.config:
            self.blacklist_pattern = re.compile(
                "|".join(map(re.escape, self.config["blacklist_keywords"])),
                re.IGNORECASE,
            )

        # 初始化白名单关键词
        if "whitelist_keywords" in self.config:
            self.whitelist_pattern = re.compile(
                "|".join(map(re.escape, self.config["whitelist_keywords"])),
                re.IGNORECASE,
            )

    def clean_text(self, text: str) -> Optional[str]:
        """执行完整的文本清洗流程"""
        if not text or not isinstance(text, str):
            return None

        try:
            # 修复编码问题
            if self.config.get("fix_encoding_errors", True):
                text = ftfy.fix_text(text)
                text = text.encode("utf-8", errors="ignore").decode("utf-8")

            # 移除HTML标签
            if self.config.get("remove_html_tags", True):
                text = self.HTML_TAG_PATTERN.sub(" ", text)
                text = html.unescape(text)

            # 移除JavaScript代码
            if self.config.get("remove_js_code", True):
                text = self.JS_PATTERN.sub(" ", text)

            # 移除CSS代码
            if self.config.get("remove_css_code", True):
                text = self.CSS_PATTERN.sub(" ", text)

            # 移除表格
            if self.config.get("remove_tables", False):
                text = self.TABLE_PATTERN.sub(" ", text)

            # URL/邮箱/电话移除
            if self.config.get("remove_urls", True):
                text = self.URL_PATTERN.sub(" ", text)
            if self.config.get("remove_emails", True):
                text = self.EMAIL_PATTERN.sub(" ", text)
            if self.config.get("remove_phone_numbers", True):
                text = self.PHONE_PATTERN.sub(" ", text)

            # 特殊字符处理
            special_char_option = self.config.get("remove_special_chars", "keep")
            if special_char_option == "common":
                text = self.SPECIAL_CHARS.sub(" ", text)
            elif special_char_option == "remove":
                text = re.sub(r"[^\w\s]", " ", text)
            elif special_char_option == "custom":
                text = self.CUSTOM_SPECIAL_CHARS.sub(" ", text)

            # Unicode标准化
            normalize_form = self.config.get("normalize_unicode", "NFC")
            text = unicodedata.normalize(normalize_form, text)

            # 网络用语替换
            if self.config.get("replace_slang", False):
                for slang, replacement in self.SLANG_MAP.items():
                    text = re.sub(
                        rf"\b{slang}\b", replacement, text, flags=re.IGNORECASE
                    )
                for slang, replacement in self.CN_SLANG_MAP.items():
                    text = text.replace(slang, replacement)

            # 大小写处理
            case_handling = self.config.get("case_handling", "original")
            if case_handling == "lower":
                text = text.lower()
            elif case_handling == "upper":
                text = text.upper()
            elif case_handling == "title":
                text = text.title()
            elif case_handling == "sentence":
                text = self._normalize_sentence_case(text)

            # 数字规范化
            number_option = self.config.get("normalize_numbers", "keep")
            if number_option == "replace":
                text = re.sub(r"\b\d+\b", "[NUM]", text)
            elif number_option == "spellout":
                text = self._spell_out_numbers(text)
            elif number_option == "digits":
                text = self._normalize_numbers_to_digits(text)

            # 标点规范化
            if self.config.get("normalize_punctuation", True):
                text = text.translate(self.PUNCTUATION_MAP)
                text = re.sub(
                    r"[。，；：、]",
                    lambda x: {"。": ".", "，": ",", "；": ";", "：": ":", "、": ","}[
                        x.group()
                    ],
                    text,
                )

            # 缩写扩展
            if self.config.get("expand_contractions", True):
                text = self._expand_contractions(text)

            # 去除重音符号
            if self.config.get("remove_accents", True):
                text = unidecode(text)
                text = "".join(
                    c
                    for c in unicodedata.normalize("NFD", text)
                    if unicodedata.category(c) != "Mn"
                )

            # 空白规范化
            if self.config.get("normalize_whitespace", True):
                text = self.MULTISPACE.sub(" ", text).strip()

            # 移除引用标记
            if self.config.get("remove_citation", True):
                text = self.CITATION_PATTERN.sub(" ", text)

            # 移除页眉页脚
            if self.config.get("remove_header_footer", True):
                text = self.HEADER_FOOTER_PATTERN.sub("\n", text)

            # 提取正文内容
            if self.config.get("extract_main_content", False):
                text = self._extract_main_content(text)

            # 长度过滤
            min_len = self.config.get("min_length", 20)
            max_len = self.config.get("max_length", 2000)
            if len(text) < min_len or len(text) > max_len:
                return None

            # 语言过滤
            if "language_filter" in self.config:
                lang = self.detect_language(text)
                if (
                    lang not in self.config["language_filter"]
                    and "multi" not in self.config["language_filter"]
                ):
                    return None

            # 质量过滤
            if self.config.get("quality_threshold", 0) > 0:
                quality_score = self.assess_quality(text)
                if quality_score < self.config["quality_threshold"]:
                    return None

            # 黑名单过滤
            if hasattr(self, "blacklist_pattern"):
                if self.blacklist_pattern.search(text):
                    return None

            # 白名单过滤
            if hasattr(self, "whitelist_pattern"):
                if not self.whitelist_pattern.search(text):
                    return None

            return text if text.strip() else None

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return None

    def _normalize_sentence_case(self, text: str) -> str:
        """将文本转换为句子格式 (首字母大写)"""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s[0].upper() + s[1:].lower() if s else s for s in sentences]
        return " ".join(sentences)

    def _normalize_numbers_to_digits(self, text: str) -> str:
        """将中文数字转为阿拉伯数字"""

        def cn_num_to_arabic(cn_num):
            if not cn_num:
                return ""

            # 处理简单数字
            if cn_num in self.CN_NUM_MAP:
                return str(self.CN_NUM_MAP[cn_num])

            # 处理复杂数字 (如"一百二十三")
            total = 0
            current = 0
            for char in cn_num:
                if char in self.CN_NUM_MAP:
                    num = self.CN_NUM_MAP[char]
                    if num >= 10:
                        if current == 0:
                            current = 1
                        total += current * num
                        current = 0
                    else:
                        current = num
                else:
                    return cn_num  # 非数字字符，保持原样

            total += current
            return str(total)

        return re.sub(
            r"[零一二三四五六七八九十百千万亿两]+",
            lambda m: cn_num_to_arabic(m.group()),
            text,
        )

    def _extract_main_content(self, text: str) -> str:
        """使用嵌入向量提取主要内容"""
        sentences = self._segment_sentences(text)
        if not sentences:
            return text

        # 计算句子嵌入
        embeddings = self.sentence_model.encode(sentences)

        # 使用K-means聚类找到主要簇
        kmeans = KMeans(n_clusters=2, random_state=0).fit(embeddings)
        main_cluster = np.argmax(np.bincount(kmeans.labels_))

        # 返回主要簇中的句子
        main_sentences = [
            sentences[i]
            for i in range(len(sentences))
            if kmeans.labels_[i] == main_cluster
        ]
        return " ".join(main_sentences)

    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言"""
        try:
            return langdetect.detect(text)
        except LangDetectException:
            return "unknown"

    @staticmethod
    def assess_quality(text: str) -> float:
        """评估文本质量 (0-1)"""
        # 基于重复率、困惑度等指标的实现
        words = text.split()
        if not words:
            return 0

        # 1. 重复率
        unique_words = set(words)
        repeat_ratio = 1 - len(unique_words) / max(1, len(words))

        # 2. 标点符号比例
        punctuation_count = sum(1 for c in text if c in ".!?,;:")
        punctuation_ratio = punctuation_count / max(1, len(words))

        # 3. 大写字母比例 (仅英文)
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))

        # 综合评分
        score = (
            0.7 * (1 - repeat_ratio)
            + 0.2 * min(1, punctuation_ratio * 5)
            + 0.1 * (0.5 - min(0.5, upper_ratio))
        )
        return max(0, min(1, score))

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """扩展英文缩写"""
        contraction_map = {
            "i'm": "i am",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "i've": "i have",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i'll": "i will",
            "you'll": "you will",
            "he'll": "he will",
            "she'll": "she will",
            "we'll": "we will",
            "they'll": "they will",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "won't": "will not",
            "wouldn't": "would not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "what's": "what is",
            "who's": "who is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "how's": "how is",
            "let's": "let us",
            "ma'am": "madam",
            "o'clock": "of the clock",
            "y'all": "you all",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "hafta": "have to",
            "needa": "need to",
            "outta": "out of",
            "kinda": "kind of",
            "sorta": "sort of",
            "lotta": "lot of",
            "lemme": "let me",
            "gimme": "give me",
            "tell'em": "tell them",
            "c'mon": "come on",
            "s'more": "some more",
            "d'you": "do you",
            "e'er": "ever",
            "o'er": "over",
            "shan't": "shall not",
            "needn't": "need not",
            "mightn't": "might not",
            "mustn't": "must not",
            "daren't": "dare not",
            "usedn't": "used not",
        }

        for contraction, expansion in contraction_map.items():
            text = re.sub(rf"\b{contraction}\b", expansion, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _spell_out_numbers(text: str) -> str:
        """将数字转为英文单词 (简单实现)"""
        num_to_words = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "10": "ten",
            "11": "eleven",
            "12": "twelve",
            "13": "thirteen",
            "14": "fourteen",
            "15": "fifteen",
            "16": "sixteen",
            "17": "seventeen",
            "18": "eighteen",
            "19": "nineteen",
            "20": "twenty",
            "30": "thirty",
            "40": "forty",
            "50": "fifty",
            "60": "sixty",
            "70": "seventy",
            "80": "eighty",
            "90": "ninety",
            "100": "hundred",
            "1000": "thousand",
            "1000000": "million",
        }

        def number_to_words(num_str):
            try:
                num = int(num_str)
                if str(num) in num_to_words:
                    return num_to_words[str(num)]
                # 简单处理两位数
                if num < 100:
                    tens = (num // 10) * 10
                    units = num % 10
                    if units == 0:
                        return num_to_words[str(tens)]
                    return f"{num_to_words[str(tens)]}-{num_to_words[str(units)]}"
                return num_str  # 更复杂的数字保持原样
            except ValueError:
                return num_str

        return re.sub(r"\b\d+\b", lambda m: number_to_words(m.group()), text)

    def _segment_sentences(self, text: str) -> List[str]:
        """分句处理"""
        if not self.config.get("segment_sentences", False):
            return [text]

        method = self.config.get("sentence_split_method", "regex")

        if method == "nltk" and self.detect_language(text) == "en":
            try:
                return sent_tokenize(text)
            except LookupError:
                logger.warning(
                    "NLTK punkt data is unavailable; falling back to regex splitting"
                )
        elif method == "jieba" and self.detect_language(text) == "zh":
            return list(jieba.cut(text))
        else:
            # 通用正则分句
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
            return [s.strip() for s in sentences if s.strip()]


class TextProcessor:
    """文本处理管道"""

    def __init__(self, config: Dict):
        self.config = config
        self.cleaner = TextCleaner(config)
        self.duplicate_hashes = set()
        self.simhash_threshold = config.get("simhash_threshold", 3)
        self.chunk_size = config.get("chunk_size", 10000)

    def process_file(self, input_path: str, output_path: str):
        """处理单个文件"""
        file_ext = Path(input_path).suffix.lower()

        try:
            # 读取输入文件
            if file_ext == ".jsonl":
                with open(input_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
            elif file_ext == ".json":
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            elif file_ext in (".txt", ".csv"):
                with open(input_path, "r", encoding="utf-8") as f:
                    data = [{"text": line.strip()} for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported input format: {file_ext}")

            logger.info(f"Loaded {len(data)} items from {input_path}")

            # 处理数据
            processed_data = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                chunks = [
                    data[i : i + self.chunk_size]
                    for i in range(0, len(data), self.chunk_size)
                ]

                for chunk in chunks:
                    futures.append(executor.submit(self._process_chunk, chunk))

                for future in tqdm(futures, desc="Processing chunks"):
                    processed_data.extend(future.result())

            # 移除空结果
            processed_data = [item for item in processed_data if item]

            # 去重
            if self.config.get("remove_duplicates", True):
                processed_data = self._remove_duplicates(processed_data)

            # 打乱顺序
            if self.config.get("shuffle_output", True):
                random.shuffle(processed_data)

            # 数据集拆分
            if self.config.get("split_ratio", 1) < 1:
                split_idx = int(len(processed_data) * self.config["split_ratio"])
                train_data = processed_data[:split_idx]
                valid_data = processed_data[split_idx:]

                train_path = output_path.replace(".", "_train.")
                valid_path = output_path.replace(".", "_valid.")

                self._save_output(train_data, train_path)
                self._save_output(valid_data, valid_path)
                logger.info(
                    f"Split data into train ({len(train_data)}) and valid ({len(valid_data)}) sets"
                )
            else:
                self._save_output(processed_data, output_path)

            logger.info(
                f"Processing completed. Saved {len(processed_data)} items to {output_path}"
            )

        except Exception as e:
            logger.error(f"Error processing file {input_path}: {e}")
            raise

    def _process_chunk(self, chunk: List[Dict]) -> List[Dict]:
        """处理数据块"""
        results = []
        for item in chunk:
            try:
                processed_item = self._process_item(item)
                if processed_item:
                    results.append(processed_item)
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
        return results

    def _process_item(self, item: Dict) -> Optional[Dict]:
        """处理单个文本项"""
        text = item.get("text", "")
        if not text:
            return None

        # 清洗文本
        cleaned_text = self.cleaner.clean_text(text)
        if not cleaned_text:
            return None

        # 保留原始数据中的元字段
        result = {**item, "text": cleaned_text}

        # 保留指定的元字段
        if "metadata_fields" in self.config:
            result = {
                k: v
                for k, v in result.items()
                if k in ["text"] + self.config["metadata_fields"]
            }

        # 分句处理
        if self.config.get("segment_sentences", False):
            sentences = self.cleaner._segment_sentences(cleaned_text)
            result["sentences"] = sentences

        return result

    def _remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """使用Simhash算法去重"""
        unique_data = []

        for item in data:
            text = item.get("text", "")
            if not text:
                continue

            # 计算Simhash
            simhash = Simhash(text)

            # 检查是否与已有内容相似
            is_duplicate = False
            for existing_hash in self.duplicate_hashes:
                if simhash.distance(existing_hash) <= self.simhash_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                self.duplicate_hashes.add(simhash)
                unique_data.append(item)

        logger.info(f"Removed {len(data) - len(unique_data)} duplicates")
        return unique_data

    def _save_output(self, data: List[Dict], output_path: str):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_format = self.config.get("output_format", "jsonl")

        try:
            if output_format == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            elif output_format == "parquet":
                df = pd.DataFrame(data)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path)
            elif output_format == "csv":
                pd.DataFrame(data).to_csv(output_path, index=False, encoding="utf-8")
            elif output_format == "txt":
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in data:
                        f.write(item["text"] + "\n")

            # 压缩输出
            if self.config.get("compress_output") == "gzip":
                with zipfile.ZipFile(
                    output_path + ".zip", "w", zipfile.ZIP_DEFLATED
                ) as zipf:
                    zipf.write(output_path, arcname=Path(output_path).name)
                os.remove(output_path)
                output_path += ".zip"
            elif self.config.get("compress_output") == "zstd":
                with open(output_path, "rb") as f_in:
                    with zstd.open(output_path + ".zst", "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(output_path)
                output_path += ".zst"

        except Exception as e:
            logger.error(f"Error saving output: {e}")
            raise


class DataSourceHandler:
    """处理不同数据源的类"""

    @staticmethod
    def handle_file(input_path: str) -> List[Dict]:
        """处理单个文件"""
        return TextProcessor._load_file(input_path)

    @staticmethod
    def handle_directory(input_dir: str) -> List[Dict]:
        """处理目录中的所有文件"""
        data = []
        for file_path in Path(input_dir).glob("*"):
            if file_path.is_file():
                try:
                    data.extend(DataSourceHandler.handle_file(str(file_path)))
                except Exception as e:
                    logger.warning(f"Error processing {file_path}: {e}")
        return data

    @staticmethod
    def handle_api(endpoint: str, params: Dict) -> List[Dict]:
        """从API获取数据"""
        import requests

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    @staticmethod
    def handle_database(db_config: Dict) -> List[Dict]:
        """从数据库获取数据"""
        db_type = db_config.get("db_type", "mysql")
        try:
            if db_type == "mysql":
                import mysql.connector

                conn = mysql.connector.connect(
                    host=db_config.get("host", "localhost"),
                    user=db_config.get("user"),
                    password=db_config.get("password"),
                    database=db_config.get("database"),
                )
            elif db_type == "mongodb":
                from pymongo import MongoClient

                client = MongoClient(db_config.get("connection_string"))
                db = client[db_config.get("database")]
                collection = db[db_config.get("collection")]
                cursor = collection.find(json.loads(db_config.get("query", "{}")))
                return list(cursor)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            cursor = conn.cursor(dictionary=True)
            cursor.execute(db_config.get("query"))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if db_type == "mysql" and "conn" in locals():
                conn.close()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 处理黑名单/白名单关键词
    if "blacklist_keywords" in config and isinstance(config["blacklist_keywords"], str):
        config["blacklist_keywords"] = [
            k.strip() for k in config["blacklist_keywords"].split("\n") if k.strip()
        ]
    if "whitelist_keywords" in config and isinstance(config["whitelist_keywords"], str):
        config["whitelist_keywords"] = [
            k.strip() for k in config["whitelist_keywords"].split("\n") if k.strip()
        ]

    return config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="大模型文本数据预处理工具")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("output", help="输出文件或目录路径")
    parser.add_argument("--config", help="配置文件路径", default="config.json")
    parser.add_argument(
        "--workers", help="工作线程数", type=int, default=os.cpu_count()
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        processor = TextProcessor(config)

        if os.path.isdir(args.input):
            for file in Path(args.input).glob("*"):
                if file.is_file():
                    output_file = Path(args.output) / (
                        file.stem + "_cleaned" + file.suffix
                    )
                    processor.process_file(str(file), str(output_file))
        else:
            processor.process_file(args.input, args.output)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
