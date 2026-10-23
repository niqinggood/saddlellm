"""Prepare fixed development, micro, single-task and fresh held-out datasets.

Reuse previously audited real records. The historical test is explicitly a
regression set. A fresh final test is selected from pinned official held-out
files, excluding every normalized original-text group in the old four splits.
New character filtering is a prefilter, never a tokenizer validation claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saddlellm.decision.TypedSchema import read_typed_dataset
from tools.prepare_jev_real_data import (
    SOURCES,
    SPLITS,
    RangeFile,
    balanced_pick,
    typed,
    url,
    write_jsonl,
)

LABELS = {"ag_news": ["0", "1", "2", "3"],
          "boolq": ["false", "true"], "yelp": ["1", "2", "3", "4", "5"]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def source_cache(previous):
    """Verify original file hashes and retain source rows at official indices."""
    audit_path = previous / "data/source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw = previous / "data/raw"
    cache, evidence = {}, []
    for source in audit["sources"]:
        task, split = source["task"], source["source_split"]
        repo, revision, _ = SOURCES[task]
        if revision not in source["url"]:
            raise ValueError(f"Source is not the pinned revision: {task} {split}")
        if source["complete_file"]:
            path = raw / f"{task}_{split}.parquet"
            actual = sha(path)
            if actual != source["sha256"]:
                raise ValueError(f"Raw file hash changed: {path}")
            rows = pq.read_table(path).to_pylist()
            if len(rows) != source["rows"]:
                raise ValueError(f"Source row count changed: {path}")
            cache[task, split] = dict(enumerate(rows))
            evidence.append({"task": task, "repo": repo, "revision": revision,
                             "source_split": split, "path": str(path), "sha256": actual,
                             "rows": len(rows), "complete_file": True})
        else:
            indexed = {}
            for group in source["row_groups"]:
                path = raw / Path(group["snapshot"]).name
                actual = sha(path)
                if actual != group["snapshot_sha256"]:
                    raise ValueError(f"Raw row-group snapshot changed: {path}")
                rows = pq.read_table(path).to_pylist()
                indexed.update((group["first_source_row"] + i, row)
                               for i, row in enumerate(rows))
                evidence.append({"task": task, "repo": repo, "revision": revision,
                                 "source_split": split, "path": str(path), "sha256": actual,
                                 "rows": len(rows), "complete_file": False,
                                 "row_group": group["row_group"],
                                 "first_source_row": group["first_source_row"],
                                 "whole_source_file_sha256": None})
            cache[task, split] = indexed
    return cache, evidence, {"path": str(audit_path), "sha256": sha(audit_path)}


def load_previous(previous, cache):
    formal = previous / "result/data"
    if not (formal / "dataset.json").exists():
        raise FileNotFoundError("Need downloaded final server result/data, not local candidates")
    _, _, hashes = read_typed_dataset(formal)
    # The schema reader normalizes hard labels into target probabilities.
    # Keep original JSONL fields and labels for byte-preserving reuse and sampling.
    splits = {split: read_jsonl(formal / f"{split}.jsonl") for split in SPLITS}
    provenance = {item["id"]: item for item in read_jsonl(formal / "provenance.jsonl")}
    records = {}
    for split in SPLITS:
        records[split] = []
        for row in splits[split]:
            item = provenance[row["id"]]
            if item["revision"] != SOURCES[row["task_id"]][1]:
                raise ValueError("Previous record revision changed")
            raw = cache[row["task_id"], item["source_split"]][item["row_index_zero_based"]]
            rebuilt, _ = typed(row["task_id"], item["source_split"],
                               item["row_index_zero_based"], raw)
            if row != rebuilt:
                raise ValueError(f"Formal typed record no longer matches raw source: {row['id']}")
            enriched = {**item, "raw_fields": raw, "previous_local_split": split}
            records[split].append((row, enriched))
    return formal, records, hashes


def counts(records):
    return {split: {task: dict(Counter(row["label"] for row, _ in records[split]
                                      if row["task_id"] == task))
                    for task in LABELS}
            for split in SPLITS}


def write_dataset(path, records, role, previous_formal=None, copy_splits=()):
    path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        target = path / f"{split}.jsonl"
        if split in copy_splits:
            shutil.copyfile(previous_formal / f"{split}.jsonl", target)
            if sha(target) != sha(previous_formal / f"{split}.jsonl"):
                raise ValueError("Failed to preserve previous formal split bytes")
        else:
            write_jsonl(target, [row for row, _ in records[split]])
    write_jsonl(path / "provenance.jsonl",
                [{**item, "local_split": split} for split in SPLITS
                 for _, item in records[split]])
    write_json(path / "dataset.json", {
        "schema_version": 2, "task_type": "typed",
        "source": "Pinned real HF records; see usage.json and provenance.jsonl",
        "splits": {split: f"{split}.jsonl" for split in SPLITS}})
    write_json(path / "usage.json", role)
    names = ["dataset.json", "provenance.jsonl", "usage.json"] + [f"{s}.jsonl" for s in SPLITS]
    write_json(path / "manifest.json", {"output_sha256": {name: sha(path / name) for name in names}})
    _, checked, _ = read_typed_dataset(path)
    return {"path": str(path), "schema_read_typed_dataset": "passed",
            "split_counts": {s: len(checked[s]) for s in SPLITS},
            "label_distributions": counts(records),
            "manifest_sha256": sha(path / "manifest.json"), **role}


def subset(records, sizes, seed):
    selected = {split: [] for split in SPLITS}
    for split in SPLITS:
        for task, labels in LABELS.items():
            pool = [(row, item) for row, item in records[split] if row["task_id"] == task]
            chosen, _ = balanced_pick(pool, sizes[split], labels,
                                      random.Random(f"{seed}:{split}:{task}"), set())
            selected[split].extend(chosen)
        random.Random(f"{seed}:{split}:shuffle").shuffle(selected[split])
    return selected


def single_task(records, task):
    return {split: [(row, item) for row, item in records[split] if row["task_id"] == task]
            for split in SPLITS}


def final_holdout(cache, previous_groups, seed, max_chars):
    selected, reserves, statistics = [], [], {}
    for task, labels in LABELS.items():
        split = "validation" if task == "boolq" else "test"
        pool, removed = [], Counter()
        for index, raw in cache[task, split].items():
            row, item = typed(task, split, index, raw)
            if row["state_id"] in previous_groups:
                removed["previous_split_group"] += 1
                continue
            if len(row["state"]) > max_chars:
                removed["character_prefilter"] += 1
                continue
            pool.append((row, {**item, "raw_fields": raw,
                               "previous_local_split": None}))
        rng = random.Random(f"{seed}:fresh_final:{task}")
        reserve, _ = balanced_pick(pool, 512, labels, rng, previous_groups)
        chosen, _ = balanced_pick(reserve, 256, labels, rng, set())
        selected.extend(chosen)
        reserves.extend(reserve)
        statistics[task] = {"official_split": split, "eligible_rows": len(pool),
                            "removed": dict(removed), "selected": len(chosen),
                            "reserve_pool": len(reserve)}
    random.Random(seed).shuffle(selected)
    random.Random(seed).shuffle(reserves)
    if any(row["state_id"] in previous_groups for row, _ in reserves):
        raise ValueError("Fresh final reserve leaks a previous original-text group")
    return selected, reserves, statistics


def augment(pool, existing, count, labels, rng, forbidden):
    """Keep all old training records; fill exact per-label targets with new groups."""
    selected = list(existing)
    used = set(forbidden) | {row["state_id"] for row, _ in existing}
    existing_counts = Counter(row["label"] for row, _ in existing)
    for position, label in enumerate(labels):
        wanted = count // len(labels) + int(position < count % len(labels))
        wanted -= existing_counts[label]
        if wanted < 0:
            raise ValueError("Requested expansion cannot retain all existing labels")
        if wanted == 0:
            continue
        available = [record for record in pool if record[0]["label"] == label]
        rng.shuffle(available)
        added = 0
        for record in available:
            if record[0]["state_id"] in used:
                continue
            selected.append(record)
            used.add(record[0]["state_id"])
            added += 1
            if added == wanted:
                break
        if added != wanted:
            raise ValueError(f"Insufficient groups for expanded label {label}: {added}/{wanted}")
    return selected


def extra_yelp_groups(output, cache, previous_evidence, seed, groups=7):
    """Add deterministic pinned train row groups, never claiming a whole-file hash."""
    stream = RangeFile(url("yelp", "train"))
    parquet = pq.ParquetFile(stream)
    order = list(range(parquet.num_row_groups))
    random.Random(seed).shuffle(order)
    offsets, offset = [], 0
    for group in range(parquet.num_row_groups):
        offsets.append(offset)
        offset += parquet.metadata.row_group(group).num_rows
    existing = {item["row_group"] for item in previous_evidence if "row_group" in item}
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    saved = []
    for group in order[:groups]:
        if group in existing:
            continue
        path = raw / f"yelp_train_rowgroup_{group}.parquet"
        if path.exists():
            table = pq.read_table(path)
        else:
            table = parquet.read_row_group(group)
            pq.write_table(table, path)
        cache["yelp", "train"].update((offsets[group] + i, row)
                                     for i, row in enumerate(table.to_pylist()))
        saved.append({"task": "yelp", "repo": SOURCES["yelp"][0],
                      "revision": SOURCES["yelp"][1], "source_split": "train",
                      "row_group": group, "first_source_row": offsets[group],
                      "rows": table.num_rows, "snapshot": str(path),
                      "snapshot_sha256": sha(path), "complete_file": False})
    audit = {"url": url("yelp", "train"), "whole_source_file_sha256": None,
             "source_size_bytes": stream.size, "source_rows": offset,
             "seed": seed, "first_shuffled_row_groups": order[:groups],
             "saved_extra_groups": saved, "http_ranges": stream.ranges,
             "downloaded_range_bytes": sum(r["bytes"] for r in stream.ranges)}
    write_json(output / "expanded_yelp_source_audit.json", audit)
    return audit


def prepare_expanded(previous, output, seed=42, max_chars=1200, train_count=2000):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    first_delivery = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    if first_delivery["seed"] != seed:
        raise ValueError("Expansion seed must match the first delivery")
    if (output / "expanded_preparation.json").exists():
        raise FileExistsError("Expanded data already exists; use a new output directory")
    cache, raw_evidence, _ = source_cache(previous)
    formal, old, _ = load_previous(previous, cache)
    yelp = extra_yelp_groups(output, cache, raw_evidence, seed)
    protected = {row["state_id"] for split in ("validation", "calibration", "test")
                 for row, _ in old[split]}
    # Every official held-out original-text group is excluded, even unselected ones.
    for task in LABELS:
        heldout = "validation" if task == "boolq" else "test"
        protected.update(typed(task, heldout, i, raw)[0]["state_id"]
                         for i, raw in cache[task, heldout].items())
    train, reserve, stats = [], [], {}
    for task, labels in LABELS.items():
        pool, removed = [], Counter()
        for index, raw in cache[task, "train"].items():
            row, item = typed(task, "train", index, raw)
            if row["state_id"] in protected:
                removed["heldout_or_dev_group"] += 1
                continue
            if len(row["state"]) > max_chars:
                removed["character_prefilter"] += 1
                continue
            pool.append((row, {**item, "raw_fields": raw, "previous_local_split": None}))
        existing = [(row, item) for row, item in old["train"] if row["task_id"] == task]
        selected = augment(pool, existing, train_count, labels,
                           random.Random(f"{seed}:expanded:{task}"), protected)
        reserved = augment(pool, selected, train_count + 1000, labels,
                           random.Random(f"{seed}:expanded_reserve:{task}"), protected)
        train.extend(selected)
        reserve.extend(reserved)
        stats[task] = {"source_training_records_read": len(cache[task, "train"]),
                       "eligible_candidates": len(pool), "removed": dict(removed),
                       "old_training_records_retained": len(existing),
                       "selected": len(selected), "reserve": len(reserved)}
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(reserve)
    records = {**old, "train": train}
    role = {"seed": seed, "purpose": "Expanded development, same old validation/calibration",
            "selection_data": "validation_only", "historical_test_used_for_selection": False,
            "test_role": "historical_regression_already_viewed_not_new_final_evidence",
            "tokenizer_validation": "New training rows pending real L20 tokenizer; characters are a prefilter",
            "max_state_characters_prefilter": max_chars,
            "all_previous_training_records_retained": True}
    summary = write_dataset(output / "expanded", records, role, formal,
                            ("validation", "calibration", "test"))
    pools = output / "expanded/candidate_pools"
    pools.mkdir(exist_ok=True)
    for split in SPLITS:
        candidates = reserve if split == "train" else old[split]
        write_jsonl(pools / f"{split}.jsonl", [{"row": row, "provenance": item,
                     "local_split": split} for row, item in candidates])
    final_groups = {row["state_id"] for row in read_jsonl(output / "final_holdout/test.jsonl")}
    if final_groups & {row["state_id"] for row, _ in train}:
        raise ValueError("Expanded training leaks the fresh final holdout")
    audit = {"status": "passed", "dataset": summary, "task_statistics": stats,
             "extra_yelp_source": yelp, "protected_group_count": len(protected),
             "fresh_final_group_overlap": 0, "new_training_tokenizer_validated": False,
             "generator_sha256": sha(__file__),
             "candidate_pool_sha256": {s: sha(pools / f"{s}.jsonl") for s in SPLITS}}
    write_json(output / "expanded_preparation.json", audit)
    return audit


def prepare(previous, output, seed=42, max_chars=1200):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (output / "preparation.json").exists():
        raise FileExistsError("Use a new output directory; prepared datasets are immutable")
    cache, raw_evidence, old_source_audit = source_cache(previous)
    formal, old, old_hashes = load_previous(previous, cache)
    previous_groups = {row["state_id"] for split in SPLITS for row, _ in old[split]}
    common = {"seed": seed, "selection_data": "validation_only",
              "historical_test_used_for_selection": False,
              "test_role": "historical_regression_already_viewed_not_new_final_evidence",
              "tokenizer_validation": "Inherited unchanged state and candidate queries from old audited formal data"}
    datasets = {}
    datasets["development"] = write_dataset(output / "development", old,
        {**common, "purpose": "Same 512/128/128 per-task development control"},
        formal, SPLITS)
    micro = subset(old, {"train": 60, "validation": 20, "calibration": 20, "test": 20}, seed)
    datasets["micro"] = write_dataset(output / "micro", micro,
        {**common, "purpose": "Training-only 60-per-task memorization diagnosis; no final test access",
         "contains_training_subsets_of_development": True})
    for task in ("boolq", "yelp"):
        key = f"single_task/{task}"
        datasets[key] = write_dataset(output / key, single_task(old, task),
            {**common, "purpose": "Weak-task single-task versus joint development comparison"})
        key = f"micro_single_task/{task}"
        datasets[key] = write_dataset(output / key, single_task(micro, task),
            {**common, "purpose": "Single-task tiny-training diagnosis"})
    # Development is available immediately before the independently sealed final set is built.
    write_json(output / "development_ready.json", {"status": "passed", "datasets": datasets})
    print(json.dumps({"stage": "development_ready", "datasets": list(datasets)}), flush=True)
    final_rows, final_reserve, final_stats = final_holdout(cache, previous_groups, seed, max_chars)
    final = {**old, "test": final_rows}
    role = {**common, "purpose": "SEALED new final evaluation; open only after all model/configuration selection is frozen",
            "test_role": "fresh_official_holdout_disjoint_from_all_previous_four_splits",
            "tokenizer_validation": "NEW final test pending real L20 tokenizer validation; max_chars is a prefilter only",
            "max_state_characters_prefilter": max_chars,
            "do_not_train_or_select_using_this_dataset": True}
    datasets["final_holdout"] = write_dataset(output / "final_holdout", final, role,
        formal, ("train", "validation", "calibration"))
    pools = output / "final_holdout/candidate_pools"
    pools.mkdir(exist_ok=True)
    for split in SPLITS:
        candidates = final_reserve if split == "test" else old[split]
        write_jsonl(pools / f"{split}.jsonl", [{"row": row, "provenance": item,
                     "local_split": split} for row, item in candidates])
    audit = {"status": "passed", "stage": "development_micro_single_task_and_fresh_final_ready",
             "seed": seed, "previous_formal_directory": str(formal),
             "previous_formal_hashes": old_hashes, "previous_source_audit": old_source_audit,
             "verified_raw_source_files": raw_evidence, "datasets": datasets,
             "fresh_final_statistics": final_stats,
             "fresh_final_excluded_all_previous_state_groups": len(previous_groups),
             "fresh_final_tokenizer_validated": False,
             "fresh_final_test_counts": dict(Counter(row["task_id"] for row, _ in final_rows)),
             "final_candidate_pool_sha256": {s: sha(pools / f"{s}.jsonl") for s in SPLITS},
             "scope": "Classic English short-text balanced source subsets; pretraining contamination cannot be excluded",
             "expanded_2000_status": "not_prepared_in_first_delivery"}
    write_json(output / "preparation.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", default="artifacts/jev_real_training_20261004")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--stage", choices=["initial", "expanded"], default="initial")
    parser.add_argument("--expanded-train-per-task", type=int, default=2000)
    args = parser.parse_args()
    if args.stage == "expanded":
        audit = prepare_expanded(args.previous, args.output, args.seed, args.max_chars,
                                 args.expanded_train_per_task)
        print(json.dumps({"status": audit["status"], "split_counts": audit["dataset"]["split_counts"],
                          "new_training_tokenizer_validated": False}))
    else:
        audit = prepare(args.previous, args.output, args.seed, args.max_chars)
        print(json.dumps({"status": audit["status"], "datasets": len(audit["datasets"]),
                          "fresh_final_tokenizer_validated": False}))
