"""Prepare reproducible real-data Choice/Noul/Score training subsets.

Uses pinned Hugging Face files; never uses official held-out examples for training.
Character filtering is only a preliminary filter, NOT tokenizer validation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saddlellm.decision.TypedSchema import read_typed_dataset

SOURCES = {
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4", "data"),
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "data"),
    "yelp": ("Yelp/yelp_review_full", "c1f9ee939b7d05667af864ee1cb066393154bf85", "yelp_review_full"),
}
SPLITS = ("train", "validation", "calibration", "test")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def url(task, split):
    repo, revision, prefix = SOURCES[task]
    name = f"{prefix}/{split}-00000-of-00001.parquet"
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"


def download(task, split, raw):
    target = raw / f"{task}_{split}.parquet"
    if not target.exists():
        partial = target.with_suffix(".parquet.partial")
        with requests.get(url(task, split), stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        partial.replace(target)
    table = pq.read_table(target)
    return table.to_pylist(), {
        "task": task, "source_split": split, "url": url(task, split),
        "local_file": str(target), "rows": table.num_rows,
        "complete_file": True, "size_bytes": target.stat().st_size,
        "sha256": digest(target.read_bytes()),
    }


class RangeFile(io.RawIOBase):
    """Seekable pinned HTTP file, recording hashes of every byte range read."""
    def __init__(self, address):
        super().__init__()
        self.address, self.position, self.ranges = address, 0, []
        self.session = requests.Session()
        with self.session.get(address, headers={"Range": "bytes=0-0"}, stream=True, timeout=30) as response:
            if response.status_code != 206:
                raise RuntimeError("Yelp HTTP server did not honor Range requests")
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        if size < 0: size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0: return b""
        start, end = self.position, self.position + size - 1
        with self.session.get(self.address, headers={"Range": f"bytes={start}-{end}"}, timeout=60) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"HTTP Range mismatch: expected {expected}, got {response.headers.get('Content-Range')}")
            data = response.content
        if len(data) != size: raise RuntimeError("Incomplete HTTP range")
        self.ranges.append({"start": start, "end_inclusive": end, "bytes": len(data), "sha256": digest(data)})
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer)); buffer[:len(data)] = data
        return len(data)


def yelp_train(raw, seed):
    stream = RangeFile(url("yelp", "train"))
    parquet = pq.ParquetFile(stream)
    indices = list(range(parquet.num_row_groups))
    random.Random(seed).shuffle(indices)
    offsets, count = [], 0
    for group in range(parquet.num_row_groups):
        offsets.append(count)
        count += parquet.metadata.row_group(group).num_rows
    rows, used = [], []
    # Multiple deterministic groups give a useful reserve pool without downloading 299 MB.
    for group in indices[:3]:
        table = parquet.read_row_group(group)
        snapshot = raw / f"yelp_train_rowgroup_{group}.parquet"
        pq.write_table(table, snapshot)
        used.append({"row_group": group, "first_source_row": offsets[group], "rows": table.num_rows,
                     "snapshot": str(snapshot), "snapshot_sha256": digest(snapshot.read_bytes())})
        rows.extend((offsets[group] + i, row) for i, row in enumerate(table.to_pylist()))
    return rows, {"task": "yelp", "source_split": "train", "url": url("yelp", "train"),
                  "complete_file": False, "source_size_bytes": stream.size, "source_total_rows": count,
                  "source_whole_file_sha256": None, "row_groups": used, "http_ranges": stream.ranges,
                  "downloaded_range_bytes": sum(r["bytes"] for r in stream.ranges)}


def typed(task, split, index, raw):
    repo, revision, _ = SOURCES[task]
    group_text = raw["passage"] if task == "boolq" else raw["text"]
    group = task + ":" + digest(norm(group_text).encode("utf-8"))
    row = {"id": f"{task}:{split}:{index}", "state_id": group, "task_id": task}
    if task == "ag_news":
        row.update(type="choice", state=raw["text"], question="Which topic best describes this news article?",
                   choices=[{"id": str(i), "text": value} for i, value in enumerate(["World", "Sports", "Business", "Science and Technology"])],
                   label=str(raw["label"]))
    elif task == "boolq":
        row.update(type="noul", state="Passage: " + raw["passage"] + "\nQuestion: " + raw["question"],
                   question="Based on the passage, is the answer to the question yes?",
                   label="true" if raw["answer"] else "false")
    else:
        row.update(type="score", state=raw["text"], question="What star rating did the reviewer assign to this business?",
                   levels=[{"id": str(i), "text": f"{i} star" + ("s" if i != 1 else ""), "value": i} for i in range(1, 6)],
                   label=str(raw["label"] + 1))
    provenance = {"id": row["id"], "repo": repo, "revision": revision, "source_split": split,
                  "row_index_zero_based": index, "raw_label": raw.get("answer", raw.get("label")),
                  "normalized_group_sha256": group.split(":", 1)[1]}
    return row, provenance


def balanced_pick(pool, count, labels, rng, forbidden):
    available = {label: [] for label in labels}
    for row, provenance in pool:
        if row["state_id"] not in forbidden:
            available[row["label"]].append((row, provenance))
    selected, seen = [], set(forbidden)
    for i, label in enumerate(labels):
        wanted = count // len(labels) + int(i < count % len(labels))
        rng.shuffle(available[label])
        chosen = []
        for record in available[label]:
            if record[0]["state_id"] not in seen:
                chosen.append(record); seen.add(record[0]["state_id"])
                if len(chosen) == wanted: break
        if len(chosen) != wanted:
            raise RuntimeError(f"Insufficient independent groups for label {label}: {len(chosen)} < {wanted}")
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected, seen


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prepare(output, seed=42, max_chars=1200, reserve_factor=2):
    root = Path(output).resolve()
    if (root / "dataset.json").exists(): raise FileExistsError("Output already contains a dataset; use a new directory")
    raw = root / "raw"; raw.mkdir(parents=True, exist_ok=True)
    specs = [("ag_news", "train"), ("ag_news", "test"), ("boolq", "train"), ("boolq", "validation"), ("yelp", "test")]
    with ThreadPoolExecutor(max_workers=5) as executor:
        downloaded = list(executor.map(lambda item: download(*item, raw), specs))
    inputs = {item: result[0] for item, result in zip(specs, downloaded)}
    sources = [result[1] for result in downloaded]
    yelp_rows, yelp_source = yelp_train(raw, seed)
    sources.append(yelp_source)
    buckets = {split: [] for split in SPLITS}; reserves = {split: [] for split in SPLITS}
    stats = {}
    sizes = {"train": 512, "validation": 128, "calibration": 128, "test": 256}
    for task in SOURCES:
        rng = random.Random(seed)
        heldout_split = "validation" if task == "boolq" else "test"
        test_rows = inputs[(task, heldout_split)]
        train_rows = yelp_rows if task == "yelp" else list(enumerate(inputs[(task, "train")]))
        labels = ["false", "true"] if task == "boolq" else [str(i) for i in range(1, 6)] if task == "yelp" else ["0", "1", "2", "3"]
        all_heldout_groups = {typed(task, heldout_split, i, item)[0]["state_id"] for i, item in enumerate(test_rows)}
        test_pool, train_pool = [], []
        filtered, overlaps = Counter(), 0
        for source_split, source_rows, destination in [(heldout_split, enumerate(test_rows), test_pool), ("train", train_rows, train_pool)]:
            for index, item in source_rows:
                record = typed(task, source_split, index, item)
                if source_split == "train" and record[0]["state_id"] in all_heldout_groups:
                    overlaps += 1; continue
                if len(record[0]["state"]) > max_chars:
                    filtered[source_split] += 1; continue
                destination.append(record)
        test_selection, used = balanced_pick(test_pool, sizes["test"] * reserve_factor, labels, rng, set())
        pools = {"test": test_selection}
        for split in ("train", "validation", "calibration"):
            pools[split], used = balanced_pick(train_pool, sizes[split] * reserve_factor, labels, rng, used)
        for split in SPLITS:
            selected, _ = balanced_pick(pools[split], sizes[split], labels, rng, set())
            buckets[split].extend(selected)
            reserves[split].extend(pools[split])
        stats[task] = {"official_heldout_split": heldout_split, "training_pool_rows": len(train_pool),
                       "test_pool_rows": len(test_pool), "prefilter_removed_rows": dict(filtered),
                       "train_rows_removed_for_any_official_test_group_overlap": overlaps,
                       "subset_label_balancing": "approximately equal counts per label; not natural distribution"}
    provenance = []
    for split in SPLITS:
        random.Random(seed).shuffle(buckets[split])
        write_jsonl(root / f"{split}.jsonl", [row for row, _ in buckets[split]])
        provenance.extend({**item, "local_split": split} for _, item in buckets[split])
    write_jsonl(root / "provenance.jsonl", provenance)
    pools = root / "candidate_pools"; pools.mkdir(exist_ok=True)
    for split in SPLITS:
        write_jsonl(pools / f"{split}.jsonl", [{"row": row, "provenance": source, "local_split": split} for row, source in reserves[split]])
    descriptor = {"schema_version": 2, "task_type": "typed", "source": "Pinned HF AG News, BoolQ, Yelp Review Full",
                  "splits": {split: f"{split}.jsonl" for split in SPLITS}}
    (root / "dataset.json").write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    _, checked, _ = read_typed_dataset(root)
    outputs = ["dataset.json", "provenance.jsonl"] + [f"{split}.jsonl" for split in SPLITS]
    audit = {"seed": seed, "max_state_characters_prefilter": max_chars,
             "tokenizer_validation_completed": False,
             "length_policy": "Character prefilter only; L20 must validate the exact rendered candidate input with the real tokenizer and refill from candidate_pools if necessary.",
             "reserve_factor": reserve_factor, "group_policy": "NFKC/casefold/whitespace-normalized passage for BoolQ; normalized full text for AG News/Yelp. One selected example per group. Official held-out groups always take precedence over training.",
             "scope": "Selected real-source subsets, not full benchmark evaluation and not a universal Jev capability claim.",
             "sources": sources, "task_statistics": stats,
             "license_notes": {"ag_news": "HF unknown; card says research/non-commercial", "boolq": "CC-BY-SA-3.0", "yelp": "HF other/yelp-licence; preserve original source terms"},
             "schema_read_typed_dataset_passed": True,
             "local_split_counts": {split: len(checked[split]) for split in SPLITS},
             "task_counts": {split: dict(Counter(row["task_id"] for row in checked[split])) for split in SPLITS},
             "label_distributions": {split: {task: dict(Counter(row["label"] for row, _ in buckets[split] if row["task_id"] == task)) for task in SOURCES} for split in SPLITS},
             "output_sha256": {name: digest((root / name).read_bytes()) for name in outputs}}
    (root / "source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"output_sha256": audit["output_sha256"]}, indent=2) + "\n", encoding="utf-8")
    read_typed_dataset(root)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--reserve-factor", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.output, args.seed, args.max_chars, args.reserve_factor)
    print(json.dumps({"schema_passed": result["schema_read_typed_dataset_passed"], "counts": result["local_split_counts"], "tokenizer_validated": False}))
