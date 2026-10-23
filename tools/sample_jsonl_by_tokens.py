"""Copy a deterministic JSONL prefix up to a token budget.

Rows produced by the corpus builders carry ``token_count`` so a bounded pilot
corpus can be made without re-tokenizing or loading the source into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    digest = hashlib.sha256()
    records = 0
    tokens = 0
    with args.input.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            payload = json.loads(line)
            token_count = int(payload.get("token_count") or 0)
            if token_count <= 0:
                raise ValueError(f"row {records + 1} has no positive token_count")
            target.write(line)
            digest.update(line.encode("utf-8"))
            records += 1
            tokens += token_count
            if tokens >= args.tokens:
                break
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "records": records,
                "tokens": tokens,
                "sha256": digest.hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
