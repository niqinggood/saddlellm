"""Prepare a pinned, decontaminated BANKING77 Choice dataset (stdlib only).

python tools/prepare_decision_banking77.py --output artifacts/decision_model_20261002/data/banking77
Use --raw-dir with train.csv, test.csv, categories.json and LICENSE for offline use.
The official test rows are preserved in their original order, including duplicates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import unicodedata
import urllib.request


COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
REPOSITORY = "https://github.com/PolyAI-LDN/task-specific-datasets"
RAW_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{COMMIT}"
SOURCE_URLS = {name: f"{RAW_BASE}/banking_data/{name}"
               for name in ("train.csv", "test.csv", "categories.json")}
SOURCE_URLS["LICENSE"] = f"{RAW_BASE}/LICENSE"
SOURCE_SHA256 = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
    "LICENSE": "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661",
}
QUESTION = "Which banking customer-service intent best matches this request?"
ATTRIBUTION = (
    "BANKING77 by PolyAI; Casanueva, Temcinas, Gerz, Henderson and Vulic (2020), "
    "Efficient Intent Detection with Dual Sentence Encoders. CC BY 4.0. "
    "https://aclanthology.org/2020.nlp4convai-1.5/"
)
SPLIT_NAMES = ("train", "validation", "calibration")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Used only for grouping; model-facing state retains the original text."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).strip()).casefold()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def download_raw(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCE_URLS.items():
        target = directory / name
        if target.exists():
            data = target.read_bytes()
        else:
            print(f"Downloading pinned official source: {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "SaddleLLM-dataset-preparation/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if digest(data) != SOURCE_SHA256[name]:
            raise ValueError(f"SHA-256 mismatch for {name}; expected pinned official bytes")
        if not target.exists():
            target.write_bytes(data)


def read_rows(raw: bytes, source_split: str, labels: list[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError(f"{source_split}: CSV columns must be text,category")
    rows = []
    for index, row in enumerate(reader):
        state, label = row.get("text"), row.get("category")
        if not isinstance(state, str) or not normalize_text(state):
            raise ValueError(f"{source_split} row {index}: empty text")
        if label not in labels or None in row:
            raise ValueError(f"{source_split} row {index}: unknown label or malformed CSV")
        rows.append({"id": f"banking77:{source_split}:{index:06d}", "state": state, "label": label})
    if not rows:
        raise ValueError(f"{source_split}: empty dataset")
    return rows


def apportion(n: int) -> list[int]:
    """80/10/10 per label, largest remainder; ensure each partition has a row."""
    if n < 3:
        raise ValueError("Each label needs at least 3 clean training rows for three disjoint splits")
    quotas = [n * .8, n * .1, n * .1]
    sizes = [math.floor(q) for q in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - sizes[i]), i))
    for i in order[:n - sum(sizes)]:
        sizes[i] += 1
    for i in range(3):
        if sizes[i] == 0:
            donor = max(range(3), key=lambda j: sizes[j])
            sizes[donor] -= 1
            sizes[i] += 1
    return sizes


def prepare_records(train: list[dict], test: list[dict], labels: list[str], seed: int = 42):
    train_groups, test_groups = defaultdict(list), defaultdict(list)
    for row in train:
        train_groups[normalize_text(row["state"])].append(row)
    for index, row in enumerate(test):
        test_groups[normalize_text(row["state"])].append((index, row))
    overlap = set(train_groups) & set(test_groups)
    conflicts = {key for key, rows in train_groups.items() if len({r["label"] for r in rows}) > 1}
    kept, quarantine, duplicates = [], [], []
    for key, rows in train_groups.items():
        reasons = []
        if key in overlap:
            reasons.append("normalized_text_occurs_in_official_test")
        if key in conflicts:
            reasons.append("conflicting_training_labels")
        if reasons:
            quarantine.extend({**row, "reasons": reasons} for row in rows)
        else:
            kept.append(rows[0])
            duplicates.extend({**row, "kept_id": rows[0]["id"], "reasons": ["same_label_normalized_duplicate"]} for row in rows[1:])
    rng = random.Random(seed)
    partitions = {name: [] for name in SPLIT_NAMES}
    per_label = {}
    for label in labels:
        rows = [r for r in kept if r["label"] == label]
        rng.shuffle(rows)
        sizes = apportion(len(rows))
        offset = 0
        for name, size in zip(SPLIT_NAMES, sizes):
            partitions[name].extend(rows[offset:offset + size])
            offset += size
        per_label[label] = dict(zip(SPLIT_NAMES, sizes))
    for rows in partitions.values():
        rng.shuffle(rows)
    partitions["test"] = list(test)
    normalized_sets = {name: {normalize_text(r["state"]) for r in rows} for name, rows in partitions.items()}
    for i, name in enumerate(partitions):
        for other in list(partitions)[i + 1:]:
            if normalized_sets[name] & normalized_sets[other]:
                raise AssertionError(f"Normalized text leaked between {name} and {other}")
    test_conflicts = {key for key, rows in test_groups.items() if len({r["label"] for _, r in rows}) > 1}
    deduplicated_test = [rows[0][0] for rows in test_groups.values()]
    clean_test = [rows[0][0] for key, rows in test_groups.items() if key not in overlap and key not in test_conflicts]
    leakage = {
        "normalization": "Unicode NFKC; strip; collapse whitespace; casefold; source state unchanged",
        "train_test_overlap_normalized_groups": len(overlap),
        "train_test_overlap_train_row_ids": [r["id"] for key in train_groups if key in overlap for r in train_groups[key]],
        "original_overlap_test_indices": sorted(i for key in overlap for i, _ in test_groups[key]),
        "conflicting_training_groups": [{"normalized_sha256": digest(key.encode()),
                                         "labels": sorted({r["label"] for r in train_groups[key]}),
                                         "row_ids": [r["id"] for r in train_groups[key]]} for key in sorted(conflicts)],
        "same_label_train_duplicate_row_ids": [r["id"] for r in duplicates],
        "quarantined_training_row_count": len(quarantine),
        "test_duplicate_groups": [{"indices": [i for i, _ in rows], "labels": sorted({r["label"] for _, r in rows})}
                                  for rows in test_groups.values() if len(rows) > 1],
        "test_conflicting_normalized_groups": len(test_conflicts),
        "cross_split_normalized_overlap_count": 0,
    }
    indices = {
        "index_base": 0, "source": "test.jsonl (official test order)",
        "deduplicated_test_indices": sorted(deduplicated_test),
        "clean_test_indices": sorted(clean_test),
        "clean_test_policy": "First row of each normalized test text; exclude every text present in ORIGINAL train, even though the training side was removed, and conflicting test-label groups. Official test.jsonl itself is never filtered.",
    }
    return partitions, leakage, indices, quarantine + duplicates, per_label


def prepare_dataset(raw_dir: Path, output: Path, seed: int = 42, *, expected_hashes=None) -> dict:
    """expected_hashes is injectable for small offline fixtures; CLI always pins official hashes."""
    expected_hashes = SOURCE_SHA256 if expected_hashes is None else expected_hashes
    blobs = {name: (raw_dir / name).read_bytes() for name in SOURCE_URLS}
    for name, data in blobs.items():
        if digest(data) != expected_hashes[name]:
            raise ValueError(f"SHA-256 mismatch for {name}")
    labels = json.loads(blobs["categories.json"])
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("categories.json must contain distinct, nonempty string labels")
    official = expected_hashes == SOURCE_SHA256
    if official and len(labels) != 77:
        raise ValueError("Pinned BANKING77 requires 77 categories")
    train, test = (read_rows(blobs[f"{name}.csv"], name, labels) for name in ("train", "test"))
    partitions, leakage, indices, removed, per_label = prepare_records(train, test, labels, seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in partitions.items():
        write_jsonl(output / f"{name}.jsonl", rows)
    write_jsonl(output / "excluded_training_rows.jsonl", removed)
    write_json(output / "test_indices.json", indices)
    (output / "LICENSE").write_bytes(blobs["LICENSE"])
    (output / "ATTRIBUTION.txt").write_text(ATTRIBUTION + "\nChanges: training decontamination, deduplication, stratified splits, category names with underscores replaced by spaces. Original test rows preserved.\n", encoding="utf-8")
    source_records = {name: {"url": SOURCE_URLS[name], "sha256": digest(data), "bytes": len(data)} for name, data in blobs.items()}
    provenance = {
        "dataset": "BANKING77", "repository": REPOSITORY, "commit": COMMIT,
        "license": "CC-BY-4.0", "attribution": ATTRIBUTION,
        "official_source_hashes_verified": official, "sources": source_records,
        "seed": seed, "split_policy": "Per-label 80/10/10 largest-remainder allocation; deterministic label-order tie break; minimum 1 row per partition; shuffled with Python Random(seed).",
        "question_scope": "Fixed known BANKING77 intent inventory. This evaluation does not establish general dynamic-candidate or unseen-intent zero-shot capability.",
        "label_order": "Unchanged categories.json order", "choice_text_policy": "Replace underscores with spaces; no model-generated descriptions.",
        "test_policy": "Original official test text, labels, order, and duplicates retained; never used for fitting, model selection, temperature fitting, or threshold selection. Text-only overlap detection removes matching TRAIN rows.",
        "manifest": "manifest.json", "test_indices": "test_indices.json",
    }
    dataset = {"schema_version": 1, "task_type": "choice", "question": QUESTION,
               "choices": [{"id": label, "text": label.replace("_", " ")} for label in labels],
               "splits": {name: f"{name}.jsonl" for name in partitions}, "provenance": provenance}
    write_json(output / "dataset.json", dataset)
    generated = ["dataset.json", "train.jsonl", "validation.jsonl", "calibration.jsonl", "test.jsonl",
                 "excluded_training_rows.jsonl", "test_indices.json", "LICENSE", "ATTRIBUTION.txt"]
    manifest = {"schema_version": 1, "provenance": provenance,
                "preparation_script_sha256": digest(Path(__file__).read_bytes()),
                "original_counts": {"train": len(train), "test": len(test), "labels": len(labels)},
                "split_counts": {name: len(rows) for name, rows in partitions.items()},
                "per_label_split_counts": per_label, "leakage_audit": leakage,
                "clean_test_count": len(indices["clean_test_indices"]),
                "test_deduplicated_count": len(indices["deduplicated_test_indices"]),
                "output_sha256": {name: digest((output / name).read_bytes()) for name in generated}}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="Read existing pinned source files; never makes network requests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = args.raw_dir if args.raw_dir is not None else args.output / "raw"
    if args.raw_dir is None:
        download_raw(raw)
    result = prepare_dataset(raw, args.output, args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "split_counts": result["split_counts"],
                      "clean_test_count": result["clean_test_count"], "leakage_audit": result["leakage_audit"]}, indent=2))


if __name__ == "__main__":
    main()
