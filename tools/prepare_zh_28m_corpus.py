"""Build the reproducible Chinese-first corpus used by the 28M pretrain run.

The script intentionally materializes a bounded sample instead of mirroring a
multi-terabyte upstream dataset.  Every output row records its source domain
and token count, while ``corpus_manifest.json`` records the source URL,
license, filtering statistics, and content hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


SOURCE_METADATA = {
    "fineweb_zh": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
        "field": "text",
        "language": "zh",
        "license": "odc-by-1.0 (also subject to Common Crawl Terms of Use)",
        "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
    },
    "zh_wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "field": "text",
        "language": "zh",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    "zh_story": {
        "dataset": "Gabrui/multilingual_TinyStories",
        "config": "chinese",
        "field": "story",
        "language": "zh",
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/Gabrui/multilingual_TinyStories",
    },
    "en_story": {
        "dataset": "roneneldan/TinyStories",
        "config": None,
        "field": "text",
        "language": "en",
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
    "zh_poetry": {
        "dataset": "modelscope/chinese-poetry-collection",
        "config": None,
        "field": "text1",
        "language": "zh",
        "license": "apache-2.0",
        "url": "https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection",
    },
}

_URL_ONLY = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
_SPACE = re.compile(r"[\t\x0b\x0c\r ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LETTER_OR_NUMBER = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff]", re.UNICODE)
_LONG_REPEAT = re.compile(r"(.)\1{11,}", re.DOTALL)


def _hf_rows(name: str, seed: int) -> Iterator[dict[str, Any]]:
    meta = SOURCE_METADATA[name]
    dataset = load_dataset(
        meta["dataset"],
        meta["config"],
        split="train",
        streaming=True,
    )
    # IterableDataset.shuffle randomizes shard order and uses a bounded buffer;
    # it does not download or hold the complete upstream corpus.
    yield from dataset.shuffle(seed=seed, buffer_size=10_000)


def _poetry_rows(path: Path, seed: int) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("text1")]
    random.Random(seed).shuffle(rows)
    yield from rows


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    kept_lines: list[str] = []
    seen_lines: set[str] = set()
    for raw_line in text.splitlines():
        line = _SPACE.sub(" ", raw_line).strip()
        if not line or _URL_ONLY.match(line):
            continue
        fingerprint = line.casefold()
        if len(line) >= 20 and fingerprint in seen_lines:
            continue
        seen_lines.add(fingerprint)
        kept_lines.append(line)
    return _BLANK_LINES.sub("\n\n", "\n".join(kept_lines)).strip()


def _is_quality_text(text: str, language: str, minimum_chars: int) -> bool:
    if len(text) < minimum_chars or _LONG_REPEAT.search(text):
        return False
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return False
    if sum(char == "�" for char in visible) / len(visible) > 0.001:
        return False
    useful_ratio = len(_LETTER_OR_NUMBER.findall(text)) / len(visible)
    if useful_ratio < 0.45:
        return False
    if language == "zh":
        cjk_ratio = len(_CJK.findall(text)) / len(visible)
        if cjk_ratio < 0.25:
            return False
    return True


def _chunks(text: str, maximum_chars: int, minimum_chars: int) -> Iterator[str]:
    """Split long documents at paragraphs, with a sentence-safe fallback."""

    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    current: list[str] = []
    size = 0
    for paragraph in paragraphs:
        pieces = (
            [paragraph]
            if len(paragraph) <= maximum_chars
            else [
                paragraph[start : start + maximum_chars]
                for start in range(0, len(paragraph), maximum_chars)
            ]
        )
        for piece in pieces:
            extra = len(piece) + (1 if current else 0)
            if current and size + extra > maximum_chars:
                chunk = "\n".join(current).strip()
                if len(chunk) >= minimum_chars:
                    yield chunk
                current, size = [], 0
            current.append(piece)
            size += len(piece) + (1 if size else 0)
    chunk = "\n".join(current).strip()
    if len(chunk) >= minimum_chars:
        yield chunk


def _source_rows(
    name: str, *, poetry_path: Path, seed: int
) -> Iterator[dict[str, Any]]:
    if name == "zh_poetry":
        yield from _poetry_rows(poetry_path, seed)
    else:
        yield from _hf_rows(name, seed)


def _materialize_source(
    *,
    name: str,
    output: Path,
    target_tokens: int,
    tokenizer: Any,
    poetry_path: Path,
    seed: int,
    maximum_chars: int,
) -> dict[str, Any]:
    meta = SOURCE_METADATA[name]
    language = str(meta["language"])
    # A complete quatrain can be only 20-40 Chinese characters.  It should not
    # inherit the web-document threshold or the poetry source becomes empty.
    minimum_chars = 20 if name == "zh_poetry" else 160
    stats: dict[str, int] = {
        "input_rows": 0,
        "metadata_rejected": 0,
        "quality_rejected": 0,
        "exact_duplicates": 0,
        "records": 0,
        "characters": 0,
        "tokens": 0,
    }
    digest = hashlib.sha256()
    seen: set[bytes] = set()
    pending: list[str] = []

    def write_batch(handle: Any) -> bool:
        if not pending:
            return False
        encoded = tokenizer(
            pending,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
        stop = False
        for text, ids in zip(pending, encoded):
            token_count = len(ids) + 1  # document-ending EOS added by the pipeline
            payload = {
                "text": text,
                "language": language,
                "domain": name,
                "token_count": token_count,
            }
            line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            data = line.encode("utf-8")
            handle.write(line)
            digest.update(data)
            stats["records"] += 1
            stats["characters"] += len(text)
            stats["tokens"] += token_count
            if stats["tokens"] >= target_tokens:
                stop = True
                break
        pending.clear()
        return stop

    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        should_stop = False
        for row in _source_rows(name, poetry_path=poetry_path, seed=seed):
            stats["input_rows"] += 1
            if name == "fineweb_zh":
                if float(row.get("language_score") or 0.0) < 0.95:
                    stats["metadata_rejected"] += 1
                    continue
                if int(row.get("minhash_cluster_size") or 1) > 20:
                    stats["metadata_rejected"] += 1
                    continue
            text = _normalize_text(str(row.get(meta["field"], "")))
            if not _is_quality_text(text, language, minimum_chars):
                stats["quality_rejected"] += 1
                continue
            for chunk in _chunks(text, maximum_chars, minimum_chars):
                key = hashlib.sha1(chunk.encode("utf-8")).digest()
                if key in seen:
                    stats["exact_duplicates"] += 1
                    continue
                seen.add(key)
                pending.append(chunk)
                if len(pending) >= 64 and write_batch(handle):
                    should_stop = True
                    break
            if should_stop:
                break
            if stats["input_rows"] % 10_000 == 0:
                print(
                    f"[{name}] input={stats['input_rows']:,} "
                    f"records={stats['records']:,} tokens={stats['tokens']:,}",
                    flush=True,
                )
        if not should_stop:
            write_batch(handle)
    temporary.replace(output)
    return {
        "name": name,
        "path": str(output.resolve()),
        "sha256": digest.hexdigest(),
        "average_characters": round(
            stats["characters"] / max(stats["records"], 1), 2
        ),
        "average_tokens": round(stats["tokens"] / max(stats["records"], 1), 2),
        "target_tokens": target_tokens,
        **stats,
        **meta,
    }


def _mix_sources(
    files: list[dict[str, Any]], output: Path, seed: int
) -> dict[str, Any]:
    """Uniformly shuffle source records without loading their text into RAM."""

    rng = random.Random(seed)
    handles = [Path(item["path"]).open("r", encoding="utf-8") for item in files]
    remaining = [int(item["records"]) for item in files]
    total = sum(remaining)
    digest = hashlib.sha256()
    written = 0
    temporary = output.with_suffix(output.suffix + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            while total:
                ticket = rng.randrange(total)
                cumulative = 0
                chosen = 0
                for index, count in enumerate(remaining):
                    cumulative += count
                    if ticket < cumulative:
                        chosen = index
                        break
                line = handles[chosen].readline()
                if not line:
                    raise RuntimeError(
                        f"source ended before its manifest count: {files[chosen]['path']}"
                    )
                target.write(line)
                digest.update(line.encode("utf-8"))
                remaining[chosen] -= 1
                total -= 1
                written += 1
                if written % 100_000 == 0:
                    print(f"[mix] records={written:,}", flush=True)
    finally:
        for handle in handles:
            handle.close()
    temporary.replace(output)
    return {
        "path": str(output.resolve()),
        "records": written,
        "tokens": sum(int(item["tokens"]) for item in files),
        "sha256": digest.hexdigest(),
        "seed": seed,
        "method": "record-level weighted-without-replacement shuffle",
    }


def _positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed < 0:
        raise argparse.ArgumentTypeError("token budgets must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--poetry-path",
        type=Path,
        default=Path(r"E:\ml_data\chinese-poetry-collection\train.csv"),
    )
    parser.add_argument("--fineweb-tokens", type=_positive_int, default=380_000_000)
    parser.add_argument("--zh-wikipedia-tokens", type=_positive_int, default=120_000_000)
    parser.add_argument("--zh-story-tokens", type=_positive_int, default=60_000_000)
    parser.add_argument("--en-story-tokens", type=_positive_int, default=30_000_000)
    parser.add_argument("--zh-poetry-tokens", type=_positive_int, default=20_000_000)
    parser.add_argument("--maximum-chars", type=int, default=6_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, use_fast=True
    )
    budgets = {
        "fineweb_zh": args.fineweb_tokens,
        "zh_wikipedia": args.zh_wikipedia_tokens,
        "zh_story": args.zh_story_tokens,
        "en_story": args.en_story_tokens,
        "zh_poetry": args.zh_poetry_tokens,
    }
    files: list[dict[str, Any]] = []
    for offset, (name, target_tokens) in enumerate(budgets.items()):
        if target_tokens == 0:
            continue
        output = args.output_dir / f"{name}.jsonl"
        print(f"materializing {name}: target={target_tokens:,} tokens", flush=True)
        files.append(
            _materialize_source(
                name=name,
                output=output,
                target_tokens=target_tokens,
                tokenizer=tokenizer,
                poetry_path=args.poetry_path,
                seed=args.seed + offset,
                maximum_chars=args.maximum_chars,
            )
        )

    mixed = _mix_sources(files, args.output_dir / "mixed_train.jsonl", args.seed)
    anneal_files = [item for item in files if item["name"] != "fineweb_zh"]
    anneal = _mix_sources(
        anneal_files, args.output_dir / "anneal_train.jsonl", args.seed + 10_000
    )
    manifest = {
        "name": "saddle-zh-28m-pretrain-v3",
        "version": 1,
        "seed": args.seed,
        "tokenizer": str(args.tokenizer.resolve()),
        "filters": {
            "unicode_normalization": "NFC",
            "exact_deduplication": True,
            "fineweb_min_language_score": 0.95,
            "fineweb_max_minhash_cluster_size": 20,
            "maximum_characters_per_record": args.maximum_chars,
        },
        "files": files,
        "mixed": mixed,
        "anneal": anneal,
    }
    manifest_path = args.output_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
