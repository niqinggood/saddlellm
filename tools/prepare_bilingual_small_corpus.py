"""Materialize a reproducible Chinese-first corpus for small-model pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


SOURCES = {
    "zh_story": {
        "dataset": "Gabrui/multilingual_TinyStories",
        "config": "chinese",
        "field": "story",
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/Gabrui/multilingual_TinyStories",
    },
    "zh_wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.zh",
        "field": "text",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
    },
    "en_story": {
        "dataset": "roneneldan/TinyStories",
        "config": None,
        "field": "text",
        "license": "cdla-sharing-1.0",
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories",
    },
}


def _records(name: str) -> Iterable[dict]:
    source = SOURCES[name]
    dataset = load_dataset(
        source["dataset"],
        source["config"],
        split="train",
        streaming=True,
    )
    yield from dataset


def _materialize(
    output: Path,
    source_name: str,
    limit: int,
    *,
    minimum_chars: int,
    maximum_chars: int,
) -> dict:
    source = SOURCES[source_name]
    count = 0
    characters = 0
    digest = hashlib.sha256()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in _records(source_name):
            text = str(row.get(source["field"], "")).strip()
            if len(text) < minimum_chars:
                continue
            if maximum_chars > 0:
                text = text[:maximum_chars]
            payload = json.dumps(
                {"text": text, "language": source_name.split("_", 1)[0], "domain": source_name},
                ensure_ascii=False,
                separators=(",", ":"),
            ) + "\n"
            encoded = payload.encode("utf-8")
            handle.write(payload)
            digest.update(encoded)
            count += 1
            characters += len(text)
            if count >= limit:
                break
    return {
        "path": str(output.resolve()),
        "records": count,
        "characters": characters,
        "average_characters": round(characters / max(count, 1), 2),
        "sha256": digest.hexdigest(),
        **source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zh-stories", type=int, default=100_000)
    parser.add_argument("--zh-wikipedia", type=int, default=20_000)
    parser.add_argument("--en-stories", type=int, default=20_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("zh_story", args.zh_stories, 100, 2_000),
        ("zh_wikipedia", args.zh_wikipedia, 300, 2_000),
        ("en_story", args.en_stories, 100, 4_000),
    )
    files = []
    for name, limit, minimum, maximum in specs:
        files.append(
            _materialize(
                args.output_dir / f"{name}.jsonl",
                name,
                limit,
                minimum_chars=minimum,
                maximum_chars=maximum,
            )
        )

    manifest = {
        "name": "saddle-zh-en-small-pretrain-v2",
        "version": 1,
        "files": files,
    }
    manifest_path = args.output_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
