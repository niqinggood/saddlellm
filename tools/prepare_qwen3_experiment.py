"""Prepare offline Qwen3-8B non-thinking prompts and an auditable manifest.

Loads only the tokenizer; never a GPU model. Output must be a fresh directory.
--hash-weights fingerprints all shards; it does not verify official provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

QUALITY_CASES = [
    ("arithmetic", "计算 17×23，只输出结果。", "391"),
    (
        "json",
        "把姓名张三、年龄28转成JSON，只输出对象，键为name和age。",
        '{"name":"张三","age":28}',
    ),
    ("extract", "订单A的数量是12，订单B的数量是7。只输出订单B的数量。", "7"),
    ("sort", "将9、2、11、4从小到大排列，只输出逗号分隔的数字。", "2,4,9,11"),
    (
        "logic",
        "所有松树都是植物。庭院里的这棵树是松树。它是否是植物？只回答是或否。",
        "是",
    ),
    (
        "translation",
        "将 The meeting starts at nine. 翻译成中文，只输出译文。",
        "会议九点开始。",
    ),
    (
        "coding",
        "写一个Python函数square(x)，返回x的平方。只输出代码。",
        "def square(x):\n    return x * x",
    ),
    (
        "summary",
        "用不超过20个字概括：小李先备份了文件，然后安装更新，最后检查程序能否正常启动。",
        "备份、更新后检查程序启动。",
    ),
]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(model_path, output, *, hash_weights=False):
    from transformers import AutoTokenizer

    root = Path(model_path).resolve(strict=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3"
        or config.get("hidden_size") != 4096
        or config.get("num_hidden_layers") != 36
    ):
        raise ValueError("Expected the original dense Qwen3-8B architecture")
    if (root / "adapter_config.json").exists() or config.get("quantization_config"):
        raise ValueError(
            "Strict baseline must use original, unquantized weights, not an adapter"
        )
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise ValueError("No safetensors weights found")
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )

    def render(content):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    performance = []
    for i in range(16):
        facts = "\n".join(
            f"记录{j}：样本编号为{i:02d}-{j:03d}，状态为已校验。"
            for j in range((4, 16, 40, 64)[i % 4])
        )
        performance.append(
            render(
                f"实验编号{i:02d}。阅读以下记录并解释如何验证数据质量，至少写150字。\n{facts}"
            )
        )
    quality = [render(prompt) for _, prompt, _ in QUALITY_CASES]
    lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in performance]
    if max(lengths) + 256 > 4096:
        raise ValueError("Prepared prompts exceed the experiment context limit")
    files = {}
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]:
        if (root / name).is_file():
            files[name] = {
                "bytes": (root / name).stat().st_size,
                "sha256": file_hash(root / name),
            }
    for path in shards:
        files[path.name] = {"bytes": path.stat().st_size}
        if hash_weights:
            files[path.name]["sha256"] = file_hash(path)
    manifest = {
        "model_path": str(root),
        "model_id": "Qwen/Qwen3-8B",
        "weight_hashes_computed": hash_weights,
        "official_repository_verified": False,
        "files": files,
        "weight_bytes": sum(path.stat().st_size for path in shards),
        "thinking": False,
        "chat_template_applied_once": True,
        "decoding": "greedy for reproducible deployment comparison, not a general quality recommendation",
        "performance_prompt_tokens": lengths,
        "quality_cases": [
            {"id": name, "prompt": prompt, "reference_example": answer}
            for name, prompt, answer in QUALITY_CASES
        ],
        "quality_scope": "8 toy smoke tasks; reference examples are not exact-match scores or a capability benchmark",
    }
    manifest["files_manifest_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()
    ).hexdigest()
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, body in [
        ("performance.json", performance),
        ("quality.json", quality),
        ("manifest.json", manifest),
    ]:
        (destination / name).write_text(
            json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.model_path, args.output_dir, hash_weights=args.hash_weights)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
