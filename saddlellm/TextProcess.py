import os
import re
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
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from bs4 import BeautifulSoup
from slimit import ast
from slimit.parser import Parser
from slimit.visitors import nodevisitor
import langdetect
from langdetect import DetectorFactory
import jieba
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import html
import emoji
import dateparser
from pandarallel import pandarallel
import chardet
import ftfy
from ftfy import fix_text
import regex as re
from unidecode import unidecode
from rapidfuzz import fuzz
from pypinyin import lazy_pinyin
import zhconv
from simhash import Simhash

# 初始化设置
DetectorFactory.seed = 0  # 确保语言检测结果一致
nltk.download('punkt', quiet=True)
pandarallel.initialize(progress_bar=False)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("text_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextCleaner:
    """文本清洗与规范化处理器"""

    # 常用正则表达式预编译
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b')
    SPECIAL_CHARS = re.compile(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', re.UNICODE)
    MULTISPACE = re.compile(r'\s+')
    CITATION_PATTERN = re.compile(r'\[(\d+)\]|\([^)]*?\d{4}[^)]*?\)')
    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    JS_PATTERN = re.compile(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', re.IGNORECASE)
    CSS_PATTERN = re.compile(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', re.IGNORECASE)
    TABLE_PATTERN = re.compile(r'<table\b[^>]*>.*?<\/table>', re.DOTALL)
    HEADER_FOOTER_PATTERN = re.compile(r'^\s*(header|footer)\b.*?\n|\n\s*(header|footer)\b.*?$', re.IGNORECASE | re.MULTILINE)
    PUNCTUATION_MAP = str.maketrans(
        '，。！？；：“”‘’（）【】、％＃＠＆１２３４５６７８９０',
        ',.!?;:\"\"\'\'()[]%#@&1234567890'
    )

    # 网络用语替换表
    SLANG_MAP = {
        "btw": "by the way",
        "lol": "laugh out loud",
        "imo": "in my opinion",
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
        "wyd": "what are you doing",
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
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tmi": "too much information",
        "smh": "shaking my head",
        "tbh": "to be honest",
        "afaik": "as far as i know",
        "idk": "i don't know",
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
        "wyd": "what are you doing",
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
        "imo": "in my opinion",
        "imho": "in my humble opinion",
        "tmi": "too much information",
        "smh": "shaking my head",
        "tbh": "to be honest",
        "afaik": "as far as i know",
        "idk": "i don't know",
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
        "wyd": "what are you doing",
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
        "fwiw": "for what it's worth"
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
        "造咯": "知道咯",
        "造捏": "知道捏",
        "造呐": "知道呐",
        "造呢": "知道呢",
        "造吧": "知道吧",
        "造啊": "知道啊",
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
        "造咯": "知道咯",
        "造捏": "知道捏",
        "造呐": "知道呐",
        "造呢": "知道呢",
        "造吧": "知道吧",
        "造啊": "知道啊"
    }

    # 中文数字转阿拉伯数字
    CN_NUM_MAP = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '百': 100, '千': 1000, '万': 10000,
        '亿': 100000000, '两': 2
    }

    def __init__(self, config: Dict):
        self.config = config
        self._init_regex_patterns()
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') if config.get('extract_main_content', False) else None

    def _init_regex_patterns(self):
        """根据配置初始化正则表达式模式"""
        # 自定义特殊字符处理
        if self.config.get('remove_special_chars') == 'custom' and 'custom_special_chars' in self.config:
            custom_chars = re.escape(self.config['custom_special_chars'])
            self.CUSTOM_SPECIAL_CHARS = re.compile(f'[^{custom_chars}\\w\\s]', re.UNICODE)

        # 初始化黑名单关键词
        if 'blacklist_keywords' in self.config:
            self.blacklist_pattern = re.compile(
                '|'.join(map(re.escape, self.config['blacklist_keywords'])),
                re.IGNORECASE
            )

        # 初始化白名单关键词
        if 'whitelist_keywords' in self.config:
            self.whitelist_pattern = re.compile(
                '|'.join(map(re.escape, self.config['whitelist_keywords'])),
                re.IGNORECASE
            )

    def clean_text(self, text: str) -> Optional[str]:
        """执行完整的文本清洗流程"""
        if not text or not isinstance(text, str):
            return None

        try:
            # 修复编码问题
            if self.config.get('fix_encoding_errors', True):
                text = ftfy.fix_text(text)
                text = text.encode('utf-8', errors='ignore').decode('utf-8')

            # 移除HTML标签
            if self.config.get('remove_html_tags', True):
                text = self.HTML_TAG_PATTERN.sub(' ', text)
                text = html.unescape(text)

            # 移除JavaScript代码
            if self.config.get('remove_js_code', True):
                text = self.JS_PATTERN.sub(' ', text)

            # 移除CSS代码
            if self.config.get('remove_css_code', True):
                text = self.CSS_PATTERN.sub(' ', text)

            # 移除表格
            if self.config.get('remove_tables', False):
                text = self.TABLE_PATTERN.sub(' ', text)

            # URL/邮箱/电话移除
            if self.config.get('remove_urls', True):
                text = self.URL_PATTERN.sub(' ', text)
            if self.config.get('remove_emails', True):
                text = self.EMAIL_PATTERN.sub(' ', text)
            if self.config.get('remove_phone_numbers', True):
                text = self.PHONE_PATTERN.sub(' ', text)

            # 特殊字符处理
            special_char_option = self.config.get('remove_special_chars', 'keep')
            if special_char_option == 'common':
                text = self.SPECIAL_CHARS.sub(' ', text)
            elif special_char_option == 'remove':
                text = re.sub(r'[^\w\s]', ' ', text)
            elif special_char_option == 'custom':
                text = self.CUSTOM_SPECIAL_CHARS.sub(' ', text)

            # Unicode标准化
            normalize_form = self.config.get('normalize_unicode', 'NFC')
            text = unicodedata.normalize(normalize_form, text)

            # 网络用语替换
            if self.config.get('replace_slang', False):
                for slang, replacement in self.SLANG_MAP.items():
                    text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
                for slang, replacement in self.CN_SLANG_MAP.items():
                    text = text.replace(slang, replacement)

            # 大小写处理
            case_handling = self.config.get('case_handling', 'original')
            if case_handling == 'lower':
                text = text.lower()
            elif case_handling == 'upper':
                text = text.upper()
            elif case_handling == 'title':
                text = text.title()
            elif case_handling == 'sentence':
                text = self._normalize_sentence_case(text)

            # 数字规范化
            number_option = self.config.get('normalize_numbers', 'keep')
            if number_option == 'replace':
                text = re.sub(r'\b\d+\b', '[NUM]', text)
            elif number_option == 'spellout':
                text = self._spell_out_numbers(text)
            elif number_option == 'digits':
                text = self._normalize_numbers_to_digits(text)

            # 标点规范化
            if self.config.get('normalize_punctuation', True):
                text = text.translate(self.PUNCTUATION_MAP)
                text = re.sub(r'[。，；：、]', lambda x: {'。': '.', '，': ',', '；': ';', '：': ':', '、': ','}[x.group()], text)

            # 缩写扩展
            if self.config.get('expand_contractions', True):
                text = self._expand_contractions(text)

            # 去除重音符号
            if self.config.get('remove_accents', True):
                text = unidecode(text)
                text = ''.join(c for c in unicodedata.normalize('NFD', text)
                               if unicodedata.category(c) != 'Mn')

            # 空白规范化
            if self.config.get('normalize_whitespace', True):
                text = self.MULTISPACE.sub(' ', text).strip()

            # 移除引用标记
            if self.config.get('remove_citation', True):
                text = self.CITATION_PATTERN.sub(' ', text)

            # 移除页眉页脚
            if self.config.get('remove_header_footer', True):
                text = self.HEADER_FOOTER_PATTERN.sub('\n', text)

            # 提取正文内容
            if self.config.get('extract_main_content', False):
                text = self._extract_main_content(text)

            # 长度过滤
            min_len = self.config.get('min_length', 20)
            max_len = self.config.get('max_length', 2000)
            if len(text) < min_len or len(text) > max_len:
                return None

            # 语言过滤
            if 'language_filter' in self.config:
                lang = self.detect_language(text)
                if lang not in self.config['language_filter'] and 'multi' not in self.config['language_filter']:
                    return None

            # 质量过滤
            if self.config.get('quality_threshold', 0) > 0:
                quality_score = self.assess_quality(text)
                if quality_score < self.config['quality_threshold']:
                    return None

            # 黑名单过滤
            if hasattr(self, 'blacklist_pattern'):
                if self.blacklist_pattern.search(text):
                    return None

            # 白名单过滤
            if hasattr(self, 'whitelist_pattern'):
                if not self.whitelist_pattern.search(text):
                    return None

            return text if text.strip() else None

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return None

    def _normalize_sentence_case(self, text: str) -> str:
        """将文本转换为句子格式 (首字母大写)"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s[0].upper() + s[1:].lower() if s else s for s in sentences]
        return ' '.join(sentences)

    def _normalize_numbers_to_digits(self, text: str) -> str:
        """将中文数字转为阿拉伯数字"""

        def cn_num_to_arabic(cn_num):
            if not cn_num:
                return ''

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

        return re.sub(r'[零一二三四五六七八九十百千万亿两]+',
                      lambda m: cn_num_to_arabic(m.group()), text)

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
        main_sentences = [sentences[i] for i in range(len(sentences))
                          if kmeans.labels_[i] == main_cluster]
        return ' '.join(main_sentences)

    @staticmethod
    def detect_language(text: str) -> str:
        """检测文本语言"""
        try:
            return langdetect.detect(text)
        except:
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
        punctuation_count = sum(1 for c in text if c in '.!?,;:')
        punctuation_ratio = punctuation_count / max(1, len(words))

        # 3. 大写字母比例 (仅英文)
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))

        # 综合评分
        score = 0.7 * (1 - repeat_ratio) + 0.2 * min(1, punctuation_ratio * 5) + 0.1 * (0.5 - min(0.5, upper_ratio))
        return max(0, min(1, score))

    @staticmethod
    def _expand_contractions(text: str) -> str:
        """扩展英文缩写"""
        contraction_map = {
            "i'm": "i am", "you're": "you are", "he's": "he is",
            "she's": "she is", "it's": "it is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
            "won't": "will not", "wouldn't": "would not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "can't": "cannot",
            "couldn't": "could not", "shouldn't": "should not",
            "that's": "that is", "there's": "there is", "here's": "here is",
            "what's": "what is", "who's": "who is", "where's": "where is",
            "when's": "when is", "why's": "why is", "how's": "how is",
            "let's": "let us", "ma'am": "madam", "o'clock": "of the clock",
            "y'all": "you all", "gonna": "going to", "wanna": "want to",
            "gotta": "got to", "hafta": "have to", "needa": "need to",
            "outta": "out of", "kinda": "kind of", "sorta": "sort of",
            "lotta": "lot of", "lemme": "let me", "gimme": "give me",
            "tell'em": "tell them", "c'mon": "come on", "s'more": "some more",
            "d'you": "do you", "e'er": "ever", "o'er": "over",
            "shan't": "shall not", "needn't": "need not", "mightn't": "might not",
            "mustn't": "must not", "daren't": "dare not", "usedn't": "used not"
        }

        for contraction, expansion in contraction_map.items():
            text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _spell_out_numbers(text: str) -> str:
        """将数字转为英文单词 (简单实现)"""
        num_to_words = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
            '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
            '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
            '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
            '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
            '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
            '100': 'hundred', '1000': 'thousand', '1000000': 'million'
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

        return re.sub(r'\b\d+\b', lambda m: number_to_words(m.group()), text)

    def _segment_sentences(self, text: str) -> List[str]:
        """分句处理"""
        if not self.config.get('segment_sentences', False):
            return [text]

        method = self.config.get('sentence_split_method', 'regex')

        if method == 'nltk' and self.detect_language(text) == 'en':
            return sent_tokenize(text)
        elif method == 'jieba' and self.detect_language(text) == 'zh':
            return list(jieba.cut(text))
        else:
            # 通用正则分句
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            return [s.strip() for s in sentences if s.strip()]


class TextProcessor:
    """文本处理管道"""

    def __init__(self, config: Dict):
        self.config = config
        self.cleaner = TextCleaner(config)
        self.duplicate_hashes = set()
        self.simhash_threshold = config.get('simhash_threshold', 3)
        self.chunk_size = config.get('chunk_size', 10000)

    def process_file(self, input_path: str, output_path: str):
        """处理单个文件"""
        file_ext = Path(input_path).suffix.lower()

        try:
            # 读取输入文件
            if file_ext == '.jsonl':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = [json.loads(line) for line in f]
            elif file_ext == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
            elif file_ext in ('.txt', '.csv'):
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = [{'text': line.strip()} for line in f if line.strip()]
            else:
                raise ValueError(f"Unsupported input format: {file_ext}")

            logger.info(f"Loaded {len(data)} items from {input_path}")

            # 处理数据
            processed_data = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = []
                chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]

                for chunk in chunks:
                    futures.append(executor.submit(self._process_chunk, chunk))

                for future in tqdm(futures, desc="Processing chunks"):
                    processed_data.extend(future.result())

            # 移除空结果
            processed_data = [item for item in processed_data if item]

            # 去重
            if self.config.get('remove_duplicates', True):
                processed_data = self._remove_duplicates(processed_data)

            # 打乱顺序
            if self.config.get('shuffle_output', True):
                random.shuffle(processed_data)

            # 数据集拆分
            if self.config.get('split_ratio', 1) < 1:
                split_idx = int(len(processed_data) * self.config['split_ratio'])
                train_data = processed_data[:split_idx]
                valid_data = processed_data[split_idx:]

                train_path = output_path.replace('.', '_train.')
                valid_path = output_path.replace('.', '_valid.')

                self._save_output(train_data, train_path)
                self._save_output(valid_data, valid_path)
                logger.info(f"Split data into train ({len(train_data)}) and valid ({len(valid_data)}) sets")
            else:
                self._save_output(processed_data, output_path)

            logger.info(f"Processing completed. Saved {len(processed_data)} items to {output_path}")

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
        text = item.get('text', '')
        if not text:
            return None

        # 清洗文本
        cleaned_text = self.cleaner.clean_text(text)
        if not cleaned_text:
            return None

        # 保留原始数据中的元字段
        result = {**item, 'text': cleaned_text}

        # 保留指定的元字段
        if 'metadata_fields' in self.config:
            result = {k: v for k, v in result.items()
                      if k in ['text'] + self.config['metadata_fields']}

        # 分句处理
        if self.config.get('segment_sentences', False):
            sentences = self.cleaner._segment_sentences(cleaned_text)
            result['sentences'] = sentences

        return result

    def _remove_duplicates(self, data: List[Dict]) -> List[Dict]:
        """使用Simhash算法去重"""
        unique_data = []

        for item in data:
            text = item.get('text', '')
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
        output_format = self.config.get('output_format', 'jsonl')

        try:
            if output_format == 'jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            elif output_format == 'parquet':
                df = pd.DataFrame(data)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, output_path)
            elif output_format == 'csv':
                pd.DataFrame(data).to_csv(output_path, index=False, encoding='utf-8')
            elif output_format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(item['text'] + '\n')

            # 压缩输出
            if self.config.get('compress_output') == 'gzip':
                with zipfile.ZipFile(output_path + '.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(output_path, arcname=Path(output_path).name)
                os.remove(output_path)
                output_path += '.zip'
            elif self.config.get('compress_output') == 'zstd':
                with open(output_path, 'rb') as f_in:
                    with zstd.open(output_path + '.zst', 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(output_path)
                output_path += '.zst'

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
        for file_path in Path(input_dir).glob('*'):
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
        db_type = db_config.get('db_type', 'mysql')
        try:
            if db_type == 'mysql':
                import mysql.connector
                conn = mysql.connector.connect(
                    host=db_config.get('host', 'localhost'),
                    user=db_config.get('user'),
                    password=db_config.get('password'),
                    database=db_config.get('database')
                )
            elif db_type == 'mongodb':
                from pymongo import MongoClient
                client = MongoClient(db_config.get('connection_string'))
                db = client[db_config.get('database')]
                collection = db[db_config.get('collection')]
                cursor = collection.find(json.loads(db_config.get('query', '{}')))
                return list(cursor)
            else:
                raise ValueError(f"Unsupported database type: {db_type}")

            cursor = conn.cursor(dictionary=True)
            cursor.execute(db_config.get('query'))
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if db_type == 'mysql' and 'conn' in locals():
                conn.close()


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 处理黑名单/白名单关键词
    if 'blacklist_keywords' in config and isinstance(config['blacklist_keywords'], str):
        config['blacklist_keywords'] = [k.strip() for k in config['blacklist_keywords'].split('\n') if k.strip()]
    if 'whitelist_keywords' in config and isinstance(config['whitelist_keywords'], str):
        config['whitelist_keywords'] = [k.strip() for k in config['whitelist_keywords'].split('\n') if k.strip()]

    return config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="大模型文本数据预处理工具")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("output", help="输出文件或目录路径")
    parser.add_argument("--config", help="配置文件路径", default="config.json")
    parser.add_argument("--workers", help="工作线程数", type=int, default=os.cpu_count())
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        processor = TextProcessor(config)

        if os.path.isdir(args.input):
            for file in Path(args.input).glob('*'):
                if file.is_file():
                    output_file = Path(args.output) / (file.stem + '_cleaned' + file.suffix)
                    processor.process_file(str(file), str(output_file))
        else:
            processor.process_file(args.input, args.output)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()

# import os
# import re
# import json
# import unicodedata
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# import random
# import ftfy
# import pandas as pd
# from tqdm import tqdm
# import langdetect
# from bs4 import BeautifulSoup
# import pyarrow.parquet as pq
# import pyarrow as pa
# import zipfile
# import zstandard as zstd
# import logging
# from concurrent.futures import ThreadPoolExecutor
# from slimit import ast
# from slimit.parser import Parser
# from slimit.visitors import nodevisitor
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
#
# class TextCleaner:
#     """文本清洗与规范化处理器"""
#
#     # 常用正则表达式预编译
#     URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
#     EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
#     PHONE_PATTERN = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b')
#     SPECIAL_CHARS = re.compile(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', re.UNICODE)
#     MULTISPACE = re.compile(r'\s+')
#     CITATION_PATTERN = re.compile(r'\[(\d+)\]|\([^)]*?\d{4}[^)]*?\)')
#
#     # 网络用语替换表
#     SLANG_MAP = {
#         "btw": "by the way",
#         "lol": "laugh out loud",
#         "imo": "in my opinion",
#         # 可扩展更多...
#     }
#
#     @classmethod
#     def clean_text(cls, text: str, config: Dict) -> Optional[str]:
#         """执行完整的文本清洗流程"""
#         if not text or not isinstance(text, str):
#             return None
#
#         # 修复编码问题
#         if config.get('fix_encoding_errors', True):
#             text = ftfy.fix_text(text)
#
#         # 移除HTML标签
#         text = BeautifulSoup(text, 'html.parser').get_text()
#
#         # URL/邮箱/电话移除
#         if config.get('remove_urls', True):
#             text = cls.URL_PATTERN.sub('', text)
#         if config.get('remove_emails', True):
#             text = cls.EMAIL_PATTERN.sub('', text)
#         if config.get('remove_phone_numbers', True):
#             text = cls.PHONE_PATTERN.sub('', text)
#
#         # 特殊字符处理
#         special_char_option = config.get('remove_special_chars', 'keep')
#         if special_char_option == 'common':
#             text = cls.SPECIAL_CHARS.sub(' ', text)
#         elif special_char_option == 'remove':
#             text = re.sub(r'[^\w\s]', '', text)
#
#         # Unicode标准化
#         normalize_form = config.get('normalize_unicode', 'NFC')
#         text = unicodedata.normalize(normalize_form, text)
#
#         # 网络用语替换
#         if config.get('replace_slang', False):
#             for slang, replacement in cls.SLANG_MAP.items():
#                 text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
#
#         # 大小写处理
#         case_handling = config.get('case_handling', 'original')
#         if case_handling == 'lower':
#             text = text.lower()
#         elif case_handling == 'upper':
#             text = text.upper()
#         elif case_handling == 'title':
#             text = text.title()
#
#         # 数字规范化
#         number_option = config.get('normalize_numbers', 'keep')
#         if number_option == 'replace':
#             text = re.sub(r'\b\d+\b', '[NUM]', text)
#         elif number_option == 'spellout':
#             text = cls._spell_out_numbers(text)
#
#         # 缩写扩展
#         if config.get('expand_contractions', True):
#             text = cls._expand_contractions(text)
#
#         # 去除重音符号
#         if config.get('remove_accents', True):
#             text = ''.join(c for c in unicodedata.normalize('NFD', text)
#                            if unicodedata.category(c) != 'Mn')
#
#         # 修剪空白
#         if config.get('trim_whitespace', True):
#             text = cls.MULTISPACE.sub(' ', text).strip()
#
#         # 移除引用标记
#         if config.get('remove_citation', True):
#             text = cls.CITATION_PATTERN.sub('', text)
#
#         # 长度过滤
#         min_len = config.get('min_length', 20)
#         max_len = config.get('max_length', 2000)
#         if len(text) < min_len or len(text) > max_len:
#             return None
#
#         # 语言过滤
#         if 'language_filter' in config:
#             lang = cls.detect_language(text)
#             if lang not in config['language_filter'] and 'multi' not in config['language_filter']:
#                 return None
#
#         # 质量过滤
#         if config.get('quality_threshold', 0) > 0:
#             quality_score = cls.assess_quality(text)
#             if quality_score < config['quality_threshold']:
#                 return None
#
#         # 黑名单过滤
#         if 'blacklist_keywords' in config:
#             if any(keyword.lower() in text.lower()
#                    for keyword in config['blacklist_keywords']):
#                 return None
#
#         return text if text else None
#
#     @staticmethod
#     def detect_language(text: str) -> str:
#         """检测文本语言"""
#         try:
#             return langdetect.detect(text)
#         except:
#             return "unknown"
#
#     @staticmethod
#     def assess_quality(text: str) -> float:
#         """评估文本质量 (0-1)"""
#         # 基于重复率、困惑度等指标的简单实现
#         words = text.split()
#         unique_words = set(words)
#         repeat_ratio = 1 - len(unique_words) / max(1, len(words))
#         return max(0, 1 - repeat_ratio * 2)  # 简单线性映射
#
#     @staticmethod
#     def _expand_contractions(text: str) -> str:
#         """扩展英文缩写"""
#         contraction_map = {
#             "i'm": "i am", "you're": "you are", "he's": "he is",
#             "she's": "she is", "it's": "it is", "we're": "we are",
#             "they're": "they are", "i've": "i have", "you've": "you have",
#             "we've": "we have", "they've": "they have", "i'll": "i will",
#             "you'll": "you will", "he'll": "he will", "she'll": "she will",
#             "we'll": "we will", "they'll": "they will", "isn't": "is not",
#             "aren't": "are not", "wasn't": "was not", "weren't": "were not",
#             "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
#             "won't": "will not", "wouldn't": "would not", "don't": "do not",
#             "doesn't": "does not", "didn't": "did not", "can't": "cannot",
#             "couldn't": "could not", "shouldn't": "should not"
#         }
#         for contraction, expansion in contraction_map.items():
#             text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
#         return text
#
#     @staticmethod
#     def _spell_out_numbers(text: str) -> str:
#         """将数字转为英文单词 (简单实现)"""
#         num_to_words = {
#             '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
#             '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
#         }
#         return re.sub(r'\b\d+\b', lambda m: ' '.join(num_to_words[d] for d in m.group()), text)
#
#
# class TextProcessor:
#     """文本处理管道"""
#
#     def __init__(self, config: Dict):
#         self.config = config
#         self.cleaner = TextCleaner()
#
#     def process_file(self, input_path: str, output_path: str):
#         """处理单个文件"""
#         file_ext = Path(input_path).suffix.lower()
#
#         # 读取输入文件
#         if file_ext == '.jsonl':
#             with open(input_path, 'r', encoding='utf-8') as f:
#                 data = [json.loads(line) for line in f]
#         elif file_ext == '.json':
#             with open(input_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#         elif file_ext in ('.txt', '.csv'):
#             with open(input_path, 'r', encoding='utf-8') as f:
#                 data = [{'text': line.strip()} for line in f if line.strip()]
#         else:
#             raise ValueError(f"不支持的输入格式: {file_ext}")
#
#         # 并行处理文本
#         with ThreadPoolExecutor() as executor:
#             cleaned_data = list(tqdm(
#                 executor.map(lambda x: self._process_item(x), data),
#                 total=len(data),
#                 desc="处理文本"
#             ))
#
#         # 移除空结果
#         cleaned_data = [item for item in cleaned_data if item]
#
#         # 保存结果
#         self._save_output(cleaned_data, output_path)
#
#     def _process_item(self, item: Dict) -> Optional[Dict]:
#         """处理单个文本项"""
#         text = item.get('text', '')
#         cleaned_text = self.cleaner.clean_text(text, self.config)
#
#         if not cleaned_text:
#             return None
#
#         # 保留原始数据中的其他字段
#         result = {**item, 'text': cleaned_text}
#
#         # 分句处理
#         if self.config.get('segment_sentences', False):
#             sentences = self._segment_sentences(cleaned_text)
#             result['sentences'] = sentences
#
#         return result
#
#     def _segment_sentences(self, text: str) -> List[str]:
#         """简单的分句实现 (实际项目建议使用NLTK等专业库)"""
#         sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
#         return [s.strip() for s in sentences if s.strip()]
#
#     def _save_output(self, data: List[Dict], output_path: str):
#         """保存处理后的数据"""
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         output_format = self.config.get('output_format', 'jsonl')
#
#         if output_format == 'jsonl':
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 for item in data:
#                     f.write(json.dumps(item, ensure_ascii=False) + '\n')
#         elif output_format == 'parquet':
#             df = pd.DataFrame(data)
#             table = pa.Table.from_pandas(df)
#             pq.write_table(table, output_path)
#         elif output_format == 'csv':
#             pd.DataFrame(data).to_csv(output_path, index=False)
#         elif output_format == 'txt':
#             with open(output_path, 'w', encoding='utf-8') as f:
#                 for item in data:
#                     f.write(item['text'] + '\n')
#
#         # 压缩输出
#         if self.config.get('compress_output') == 'gzip':
#             with zipfile.ZipFile(output_path + '.zip', 'w') as zipf:
#                 zipf.write(output_path, arcname=Path(output_path).name)
#             os.remove(output_path)
#         elif self.config.get('compress_output') == 'zstd':
#             with open(output_path, 'rb') as f_in:
#                 with zstd.open(output_path + '.zst', 'wb') as f_out:
#                     f_out.write(f_in.read())
#             os.remove(output_path)
#
#         logger.info(f"处理完成，结果已保存到: {output_path}")
#
#
# def load_config(config_path: str) -> Dict:
#     """加载配置文件"""
#     with open(config_path, 'r', encoding='utf-8') as f:
#         return json.load(f)
#
#
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="文本数据预处理工具")
#     parser.add_argument("input", help="输入文件或目录路径")
#     parser.add_argument("output", help="输出文件或目录路径")
#     parser.add_argument("--config", help="配置文件路径", default="config.json")
#     args = parser.parse_args()
#
#     config = load_config(args.config)
#     processor = TextProcessor(config)
#
#     if os.path.isdir(args.input):
#         for file in Path(args.input).glob('*'):
#             output_file = Path(args.output) / (file.stem + '_cleaned' + file.suffix)
#             processor.process_file(str(file), str(output_file))
#     else:
#         processor.process_file(args.input, args.output)
#
#
# if __name__ == "__main__":
#     main()

# import os
# import re
# import json
# import logging
# from typing import List, Dict, Optional
# from pathlib import Path
# from dataclasses import dataclass
# import random
# import datasets
# from datasets import Dataset, DatasetDict
# import pandas as pd
# from tqdm import tqdm
# import argparse
# import langdetect
# from bs4 import BeautifulSoup
# import numpy as np
# from transformers import AutoTokenizer
#
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class DataConfig:
#     """数据预处理配置"""
#     # 数据源配置
#     source_type: str  # file/dataset/database
#     input_path: str
#     file_encoding: str = "utf-8"
#     dataset_name: str = None
#     db_connection: str = None
#     db_query: str = None
#     sample_size: Optional[int] = None
#
#     # 清洗配置
#     remove_duplicates: bool = True
#     remove_urls: bool = True
#     remove_emails: bool = True
#     min_length: int = 20
#     max_length: int = 2000
#     custom_regex: Optional[List[str]] = None
#     language_filter: Optional[List[str]] = None
#     quality_threshold: float = 0.7
#
#     # 格式化配置
#     output_format: str  # sft/pretrain/conversational
#     output_path: str
#     instruction_template: str = "{input}"
#     output_template: str = "{output}"
#     conversation_roles: List[str] = None
#     shuffle_data: bool = True
#     split_ratio: float = 0.9
#
#
# class TextCleaner:
#     """文本清洗工具类"""
#
#     @staticmethod
#     def remove_html_tags(text: str) -> str:
#         return BeautifulSoup(text, "html.parser").get_text()
#
#     @staticmethod
#     def remove_urls(text: str) -> str:
#         return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
#
#     @staticmethod
#     def remove_emails(text: str) -> str:
#         return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
#
#     @staticmethod
#     def apply_custom_regex(text: str, regex_list: List[str]) -> str:
#         for pattern in regex_list:
#             text = re.sub(pattern, '', text)
#         return text
#
#     @staticmethod
#     def detect_language(text: str) -> str:
#         try:
#             return langdetect.detect(text)
#         except:
#             return "unknown"
#
#     @classmethod
#     def clean_text(cls, text: str, config: DataConfig) -> Optional[str]:
#         """执行完整的文本清洗流程"""
#         if not text or not isinstance(text, str):
#             return None
#
#         # 基础清洗
#         text = text.strip()
#         text = cls.remove_html_tags(text)
#
#         if config.remove_urls:
#             text = cls.remove_urls(text)
#         if config.remove_emails:
#             text = cls.remove_emails(text)
#         if config.custom_regex:
#             text = cls.apply_custom_regex(text, config.custom_regex)
#
#         # 长度过滤
#         if len(text) < config.min_length or len(text) > config.max_length:
#             return None
#
#         # 语言过滤
#         if config.language_filter and "multi" not in config.language_filter:
#             lang = cls.detect_language(text)
#             if lang not in config.language_filter:
#                 return None
#
#         return text
#
#
# class DataFormatter:
#     """数据格式化工具类"""
#
#     @staticmethod
#     def format_sft(text: str, config: DataConfig) -> Dict:
#         """格式化为监督微调数据"""
#         return {
#             "instruction": config.instruction_template.format(input=text),
#             "input": "",
#             "output": config.output_template.format(output=text)
#         }
#
#     @staticmethod
#     def format_pretrain(text: str) -> Dict:
#         """格式化为预训练数据"""
#         return {"text": text}
#
#     @staticmethod
#     def format_conversation(text: str, config: DataConfig) -> Dict:
#         """格式化为对话数据"""
#         return {
#             "conversations": [
#                 {"role": config.conversation_roles[0], "content": text},
#                 {"role": config.conversation_roles[1], "content": ""}
#             ]
#         }
#
#     @classmethod
#     def format_data(cls, text: str, config: DataConfig) -> Optional[Dict]:
#         """根据配置格式化数据"""
#         if not text:
#             return None
#
#         if config.output_format == "sft":
#             return cls.format_sft(text, config)
#         elif config.output_format == "pretrain":
#             return cls.format_pretrain(text)
#         elif config.output_format == "conversational":
#             return cls.format_conversation(text, config)
#         else:
#             raise ValueError(f"未知的输出格式: {config.output_format}")
#
#
# class DataPreprocessor:
#     """数据预处理管道"""
#
#     def __init__(self, config: DataConfig):
#         self.config = config
#         self.cleaner = TextCleaner()
#         self.formatter = DataFormatter()
#
#     def load_data(self) -> List[str]:
#         """从不同数据源加载原始数据"""
#         if self.config.source_type == "file":
#             return self._load_from_file()
#         elif self.config.source_type == "dataset":
#             return self._load_from_dataset()
#         elif self.config.source_type == "database":
#             return self._load_from_database()
#         else:
#             raise ValueError(f"未知的数据源类型: {self.config.source_type}")
#
#     def _load_from_file(self) -> List[str]:
#         """从文件系统加载数据"""
#         file_paths = list(Path(self.config.input_path).glob("*"))
#         texts = []
#
#         for file_path in tqdm(file_paths, desc="读取文件"):
#             try:
#                 with open(file_path, 'r', encoding=self.config.file_encoding) as f:
#                     if file_path.suffix == '.jsonl':
#                         texts.extend([json.loads(line)['text'] for line in f])
#                     elif file_path.suffix == '.json':
#                         data = json.load(f)
#                         texts.extend([item['text'] for item in data])
#                     else:  # .txt等
#                         texts.extend(f.read().split('\n'))
#             except Exception as e:
#                 logger.warning(f"读取文件失败: {file_path}, 错误: {e}")
#
#         return texts
#
#     def _load_from_dataset(self) -> List[str]:
#         """从HuggingFace数据集加载数据"""
#         dataset = datasets.load_dataset(self.config.dataset_name)
#         if isinstance(dataset, DatasetDict):
#             dataset = dataset['train']
#         return dataset['text'][:self.config.sample_size]
#
#     def _load_from_database(self) -> List[str]:
#         """从数据库加载数据"""
#         import sqlalchemy
#         engine = sqlalchemy.create_engine(self.config.db_connection)
#         df = pd.read_sql(self.config.db_query, engine)
#         return df['text'].tolist()
#
#     def process(self) -> Dataset:
#         """执行完整的预处理流程"""
#         # 1. 加载原始数据
#         raw_texts = self.load_data()
#         if self.config.sample_size:
#             raw_texts = random.sample(raw_texts, min(self.config.sample_size, len(raw_texts)))
#
#         # 2. 清洗数据
#         cleaned_data = []
#         for text in tqdm(raw_texts, desc="清洗数据"):
#             cleaned = self.cleaner.clean_text(text, self.config)
#             if cleaned:
#                 cleaned_data.append(cleaned)
#
#         # 去重
#         if self.config.remove_duplicates:
#             cleaned_data = list(set(cleaned_data))
#             logger.info(f"去重后剩余数据量: {len(cleaned_data)}")
#
#         # 3. 格式化数据
#         formatted_data = []
#         for text in tqdm(cleaned_data, desc="格式化数据"):
#             formatted = self.formatter.format_data(text, self.config)
#             if formatted:
#                 formatted_data.append(formatted)
#
#         # 4. 拆分数据集
#         if self.config.shuffle_data:
#             random.shuffle(formatted_data)
#
#         split_idx = int(len(formatted_data) * self.config.split_ratio)
#         train_data = formatted_data[:split_idx]
#         val_data = formatted_data[split_idx:]
#
#         # 5. 保存结果
#         os.makedirs(self.config.output_path, exist_ok=True)
#
#         train_dataset = Dataset.from_list(train_data)
#         val_dataset = Dataset.from_list(val_data) if val_data else None
#
#         if val_dataset:
#             dataset_dict = DatasetDict({
#                 "train": train_dataset,
#                 "validation": val_dataset
#             })
#             dataset_dict.save_to_disk(self.config.output_path)
#         else:
#             train_dataset.save_to_disk(self.config.output_path)
#
#         logger.info(f"数据处理完成，已保存到: {self.config.output_path}")
#         return dataset_dict if val_dataset else train_dataset
#
#
# def parse_args():
#     """解析命令行参数"""
#     parser = argparse.ArgumentParser(description="大模型数据预处理工具")
#
#     # 数据源配置
#     parser.add_argument("--source-type", required=True,
#                         choices=["file", "dataset", "database"])
#     parser.add_argument("--input-path", help="文件路径或数据集名称")
#     parser.add_argument("--file-encoding", default="utf-8")
#     parser.add_argument("--dataset-name", help="HuggingFace数据集ID")
#     parser.add_argument("--db-connection", help="数据库连接字符串")
#     parser.add_argument("--db-query", help="数据库查询语句")
#     parser.add_argument("--sample-size", type=int, help="采样数量")
#
#     # 清洗配置
#     parser.add_argument("--remove-duplicates", type=bool, default=True)
#     parser.add_argument("--remove-urls", type=bool, default=True)
#     parser.add_argument("--remove-emails", type=bool, default=True)
#     parser.add_argument("--min-length", type=int, default=20)
#     parser.add_argument("--max-length", type=int, default=2000)
#     parser.add_argument("--custom-regex", nargs="+", help="自定义正则规则")
#     parser.add_argument("--language-filter", nargs="+",
#                         choices=["zh", "en", "ja", "multi"])
#     parser.add_argument("--quality-threshold", type=float, default=0.7)
#
#     # 格式化配置
#     parser.add_argument("--output-format", required=True,
#                         choices=["sft", "pretrain", "conversational"])
#     parser.add_argument("--output-path", required=True)
#     parser.add_argument("--instruction-template",
#                         default="请根据以下内容生成回答：\n{input}")
#     parser.add_argument("--output-template", default="{output}")
#     parser.add_argument("--conversation-roles", nargs="+",
#                         default=["用户", "助手"])
#     parser.add_argument("--shuffle-data", type=bool, default=True)
#     parser.add_argument("--split-ratio", type=float, default=0.9)
#
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     config = DataConfig(
#         source_type=args.source_type,
#         input_path=args.input_path,
#         file_encoding=args.file_encoding,
#         dataset_name=args.dataset_name,
#         db_connection=args.db_connection,
#         db_query=args.db_query,
#         sample_size=args.sample_size,
#
#         remove_duplicates=args.remove_duplicates,
#         remove_urls=args.remove_urls,
#         remove_emails=args.remove_emails,
#         min_length=args.min_length,
#         max_length=args.max_length,
#         custom_regex=args.custom_regex,
#         language_filter=args.language_filter,
#         quality_threshold=args.quality_threshold,
#
#         output_format=args.output_format,
#         output_path=args.output_path,
#         instruction_template=args.instruction_template,
#         output_template=args.output_template,
#         conversation_roles=args.conversation_roles,
#         shuffle_data=args.shuffle_data,
#         split_ratio=args.split_ratio
#     )
#
#     preprocessor = DataPreprocessor(config)
#     preprocessor.process()
#
#
# if __name__ == "__main__":
#     main()