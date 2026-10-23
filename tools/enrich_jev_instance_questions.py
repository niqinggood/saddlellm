"""Add only BoolQ's pinned original question as instance_question.

This is an input ablation, not a new sample selection. Original split membership,
record order, labels, state text and state groups are preserved. It performs no
model predictions, including on the sealed final held-out split. Character
lengths do not establish tokenizer safety; L20 must audit the enriched queries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saddlellm.decision.TypedSchema import read_typed_dataset, validate_record
from tools.prepare_jev_optimization_data import source_cache
from tools.prepare_jev_real_data import SOURCES, SPLITS, typed

DATASETS = ("development", "expanded", "final_holdout", "single_task/boolq")
NEW_FIELD = "instance_question"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()
            if line.strip()]


def verify_manifest(directory):
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"Manifest mismatch: {path}")
    return manifest


def raw_cache(previous, source):
    cache, evidence, original_audit = source_cache(previous)
    extra_audit_path = source / "expanded_yelp_source_audit.json"
    extra = read_json(extra_audit_path)
    if SOURCES["yelp"][1] not in extra["url"]:
        raise ValueError("Extra Yelp source revision differs from the pinned version")
    for group in extra["saved_extra_groups"]:
        path = source / "raw" / Path(group["snapshot"]).name
        if sha(path) != group["snapshot_sha256"]:
            raise ValueError(f"Extra raw source snapshot changed: {path}")
        records = pq.read_table(path).to_pylist()
        if len(records) != group["rows"]:
            raise ValueError(f"Extra raw source row count changed: {path}")
        index = cache["yelp", "train"]
        for offset, record in enumerate(records):
            row_index = group["first_source_row"] + offset
            if row_index in index and index[row_index] != record:
                raise ValueError("Conflicting source rows in extra snapshot")
            index[row_index] = record
        evidence.append({**group, "path": str(path), "sha256": sha(path),
                         "whole_source_file_sha256": None})
    return cache, evidence, [original_audit,
                            {"path": str(extra_audit_path), "sha256": sha(extra_audit_path)}]


def schema_status(sample):
    """Distinguish the expected old-schema rejection from any other failure."""
    try:
        parsed = validate_record(sample)
    except ValueError as exc:
        if str(exc) == "unsupported decision row fields":
            return {"status": "pending_framework_support",
                    "reason": str(exc), "instance_question_preserved": False}
        raise
    if parsed.get(NEW_FIELD) != sample[NEW_FIELD]:
        raise ValueError("Schema accepts but drops instance_question")
    return {"status": "passed", "instance_question_preserved": True}


def original_hashes(directory):
    names = ("dataset.json", "manifest.json", "usage.json", "provenance.jsonl")
    return {name: sha(directory / name) for name in names +
            tuple(f"{split}.jsonl" for split in SPLITS)}


def transform_dataset(source, output, cache, verify_only=False):
    verify_manifest(source)
    input_hashes = original_hashes(source)
    provenance_rows = rows(source / "provenance.jsonl")
    provenance = {row["id"]: row for row in provenance_rows}
    if len(provenance) != len(provenance_rows):
        raise ValueError("Duplicate source provenance IDs")
    seen, seen_groups, split_audits = set(), {}, {}
    boolq_sample = None
    if not verify_only:
        if (output / "extra_input_audit.json").exists():
            old = read_json(output / "extra_input_audit.json")
            if old["input_sha256"] != input_hashes:
                raise ValueError("Refusing to replace an output built from other input bytes")
            # Keep a sealed completed copy byte stable. Re-run only its checks.
            verify_only = True
        output.mkdir(parents=True, exist_ok=True)
    if verify_only:
        verify_manifest(output)
    all_non_bool_source, all_non_bool_output = bytearray(), bytearray()
    aggregate_original, aggregate_restored = bytearray(), bytearray()
    total_boolq = 0
    for split in SPLITS:
        original_path, target = source / f"{split}.jsonl", output / f"{split}.jsonl"
        lines = original_path.read_bytes().splitlines(keepends=True)
        target_lines = target.read_bytes().splitlines(keepends=True) if verify_only else []
        if verify_only and len(target_lines) != len(lines):
            raise ValueError("Output physical line count differs from source")
        generated, counts, labels = [], Counter(), Counter()
        checks = {"source_row_mapping": 0, "all_original_fields_exact": 0,
                  "instance_question_matches_pinned_raw": 0,
                  "non_boolq_line_bytes_exact": 0}
        for line_index, line in enumerate(lines):
            if not line.strip():
                candidate = target_lines[line_index] if verify_only else line
                if candidate != line:
                    raise ValueError("Blank-line bytes changed")
                generated.append(candidate)
                continue
            original = json.loads(line)
            if NEW_FIELD in original:
                raise ValueError("Input is already enriched; source must be the original data")
            if original["id"] in seen:
                raise ValueError("Duplicate record ID across splits")
            seen.add(original["id"])
            item = provenance[original["id"]]
            task = original["task_id"]
            repo, revision, _ = SOURCES[task]
            if (item["repo"], item["revision"], item["local_split"]) != (repo, revision, split):
                raise ValueError("Provenance repo, revision or split does not match row")
            index, official_split = item["row_index_zero_based"], item["source_split"]
            raw = cache[task, official_split][index]
            rebuilt, rebuilt_provenance = typed(task, official_split, index, raw)
            if original != rebuilt or item["raw_fields"] != raw:
                raise ValueError(f"Source row does not match original typed record: {original['id']}")
            if any(item[key] != value for key, value in rebuilt_provenance.items()):
                raise ValueError("Original provenance fields differ from pinned raw mapping")
            group = original["state_id"]
            if group in seen_groups and seen_groups[group] != split:
                raise ValueError("Original state group crosses splits")
            seen_groups[group] = split
            checks["source_row_mapping"] += 1
            modified = dict(original)
            if task == "boolq":
                question = raw["question"]
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("Pinned BoolQ question is not nonblank text")
                modified[NEW_FIELD] = question
                newline = b"\r\n" if line.endswith(b"\r\n") else b"\n" if line.endswith(b"\n") else b""
                candidate = (json.dumps(modified, ensure_ascii=False).encode("utf-8") + newline)
                boolq_sample = modified
                checks["instance_question_matches_pinned_raw"] += 1
                total_boolq += 1
            else:
                candidate = line
            if verify_only:
                candidate = target_lines[line_index]
            observed = json.loads(candidate)
            restored = {key: value for key, value in observed.items() if key != NEW_FIELD}
            if restored != original:
                raise ValueError("A field other than instance_question changed")
            if task == "boolq":
                if observed.get(NEW_FIELD) != raw["question"]:
                    raise ValueError("Output instance_question differs from the pinned original question")
            elif NEW_FIELD in observed or candidate != line:
                raise ValueError("Non-BoolQ row field or original line bytes changed")
            else:
                checks["non_boolq_line_bytes_exact"] += 1
                all_non_bool_source.extend(line)
                all_non_bool_output.extend(candidate)
            stable = json.dumps(original, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
            restored_stable = json.dumps(restored, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
            aggregate_original.extend(stable)
            aggregate_restored.extend(restored_stable)
            checks["all_original_fields_exact"] += 1
            counts[task] += 1
            labels[f"{task}:{original['label']}"] += 1
            generated.append(candidate)
        if not verify_only:
            target.write_bytes(b"".join(generated))
        split_audits[split] = {"rows": sum(counts.values()), "by_task": dict(counts),
                               "label_counts": dict(labels), "checks": checks,
                               "input_sha256": sha(original_path), "output_sha256": sha(target)}
    if seen != set(provenance):
        raise ValueError("Source provenance contains missing or unreferenced rows")
    for name in ("dataset.json", "usage.json", "provenance.jsonl"):
        if not verify_only:
            shutil.copyfile(source / name, output / name)
        if sha(source / name) != sha(output / name):
            raise ValueError(f"Original metadata bytes changed: {name}")
    if not verify_only:
        shutil.copyfile(source / "manifest.json", output / "original_manifest.json")
    if sha(output / "original_manifest.json") != input_hashes["manifest.json"]:
        raise ValueError("Original manifest archive does not match source")
    schema = schema_status(boolq_sample)
    audit = {"status": "passed", "dataset_directory": str(output.resolve()),
             "source_directory": str(source.resolve()), "added_field": NEW_FIELD,
             "change_scope": "BoolQ only: instance_question = pinned original raw_fields.question",
             "all_other_original_fields_exact": True,
             "state_label_state_id_task_question_candidates_order_and_split_preserved": True,
             "source_mapping_every_record_verified": True,
             "non_boolq_line_bytes_preserved": True,
             "provenance_bytes_preserved": True,
             "input_sha256": input_hashes, "splits": split_audits,
             "boolq_enriched_rows": total_boolq,
             "original_records_canonical_sha256": digest(aggregate_original),
             "restored_records_canonical_sha256": digest(aggregate_restored),
             "non_boolq_original_line_bytes_sha256": digest(all_non_bool_source),
             "non_boolq_output_line_bytes_sha256": digest(all_non_bool_output),
             "schema_validation": schema,
             "real_tokenizer_validation": "pending L20 max_length<=512 for enriched candidate queries",
             "max_characters_is_not_tokenizer_validation": True,
             "model_predictions_computed": False,
             "fresh_final_test_used_for_selection": False}
    write_json(output / "extra_input_audit.json", audit)
    names = ["dataset.json", "usage.json", "provenance.jsonl", "original_manifest.json",
             "extra_input_audit.json"] + [f"{split}.jsonl" for split in SPLITS]
    write_json(output / "manifest.json", {"output_sha256": {name: sha(output / name) for name in names}})
    if schema["status"] == "passed":
        _, checked, _ = read_typed_dataset(output)
        for split in SPLITS:
            if len(checked[split]) != split_audits[split]["rows"]:
                raise ValueError("Schema-normalized row count mismatch")
            for row in checked[split]:
                if row["task_id"] == "boolq" and row.get(NEW_FIELD) != provenance[row["id"]]["raw_fields"]["question"]:
                    raise ValueError("Dataset reader discarded or changed the instance question")
    return audit


def prepare(source, previous, output, verify_only=False, require_schema=False):
    source, previous, output = (Path(p).resolve() for p in (source, previous, output))
    if source == output or output.is_relative_to(previous):
        raise ValueError("Output must not overwrite input or previous formal artifacts")
    cache, raw_evidence, source_audits = raw_cache(previous, source)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name in DATASETS:
        datasets[name] = transform_dataset(source / name, output / name, cache, verify_only)
    result = {"status": "passed", "tool": str(Path(__file__).resolve()),
              "tool_sha256": sha(__file__), "source_root": str(source), "output_root": str(output),
              "datasets": datasets, "verified_raw_sources": raw_evidence,
              "source_audits": source_audits,
              "schema_all_datasets_passed": all(d["schema_validation"]["status"] == "passed" for d in datasets.values()),
              "real_tokenizer_validation": "pending L20 enriched-query audit; no tokenizer claim from character lengths",
              "model_predictions_computed": False,
              "fresh_final_holdout_sealed": True,
              "no_sample_selection_or_label_changes": True}
    write_json(output / "extra_input_audit.json", result)
    if require_schema and not result["schema_all_datasets_passed"]:
        raise RuntimeError("Instance data built and audited, but framework schema support is pending")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/jev_optimization_20261004/data")
    parser.add_argument("--previous", default="artifacts/jev_real_training_20261004")
    parser.add_argument("--output", default="artifacts/jev_optimization_20261004/data/instance_query")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-schema", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source, args.previous, args.output, args.verify_only, args.require_schema)
    print(json.dumps({"status": result["status"], "datasets": len(result["datasets"]),
                      "schema_all_datasets_passed": result["schema_all_datasets_passed"],
                      "split_counts": {name: {s: d["splits"][s]["rows"] for s in SPLITS}
                                       for name, d in result["datasets"].items()},
                      "model_predictions_computed": False}))
