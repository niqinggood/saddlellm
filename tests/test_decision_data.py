"""Small offline data-integrity checks; no models, GPU, or network."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "tools/prepare_decision_banking77.py"
_SPEC = importlib.util.spec_from_file_location("prepare_decision_banking77", _PATH)
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def record(index, state, label, split="train"):
    return {"id": f"{split}:{index}", "state": state, "label": label}


def base_rows():
    labels = ["zeta_intent", "alpha_intent"]
    train = [record(i * 20 + j, f"training {label} phrase {j}", label)
             for i, label in enumerate(labels) for j in range(20)]
    test = [record(i, f"held out {label}", label, "test") for i, label in enumerate(labels)]
    return train, test, labels


def test_normalization_does_not_rewrite_state():
    original = "  ＣＡＲＤ\t Payment  "
    assert data.normalize_text(original) == "card payment"
    rows = data.read_rows('text,category\n"  ＣＡＲＤ\t Payment  ",card\n'.encode(), "train", ["card"])
    assert rows[0]["state"] == original


def test_overlap_duplicates_conflicts_and_official_test_preservation():
    train, test, labels = base_rows()
    train += [record(100, "  ＨＥＬＤ\tOUT ZETA_INTENT ", labels[0]),
              record(101, train[0]["state"].upper(), labels[0]),
              record(102, "conflict text", labels[0]), record(103, "CONFLICT text", labels[1])]
    test.append(record(2, test[0]["state"].upper(), labels[0], "test"))
    parts, audit, indices, removed, _ = data.prepare_records(train, test, labels)
    assert parts["test"] == test
    assert audit["train_test_overlap_normalized_groups"] == 1
    assert audit["original_overlap_test_indices"] == [0, 2]
    assert audit["quarantined_training_row_count"] == 3
    assert audit["same_label_train_duplicate_row_ids"] == ["train:101"]
    assert indices["deduplicated_test_indices"] == [0, 1]
    assert indices["clean_test_indices"] == [1]
    assert len(removed) == 4
    assert sum(len(parts[x]) for x in data.SPLIT_NAMES) == 40
    assert {row["id"] for row in removed}.isdisjoint(row["id"] for x in data.SPLIT_NAMES for row in parts[x])


def test_stratification_reproducibility_and_no_overlap():
    train, test, labels = base_rows()
    first = data.prepare_records(train, test, labels, 42)
    assert first == data.prepare_records(train, test, labels, 42)
    assert first[0]["train"] != data.prepare_records(train, test, labels, 43)[0]["train"]
    assert first[4] == {label: {"train": 16, "validation": 2, "calibration": 2} for label in labels}
    sets = [{data.normalize_text(r["state"]) for r in rows} for rows in first[0].values()]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i + 1:])


def test_conflicting_test_rows_retained_but_excluded_from_clean_indices():
    train, test, labels = base_rows()
    test.append(record(2, test[0]["state"], labels[1], "test"))
    parts, audit, indices, _, _ = data.prepare_records(train, test, labels)
    assert parts["test"] == test
    assert audit["test_conflicting_normalized_groups"] == 1
    assert indices["clean_test_indices"] == [1]


def make_raw(directory):
    directory.mkdir()
    train, test, labels = base_rows()
    for name, rows in [("train", train), ("test", test)]:
        with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["text", "category"])
            writer.writerows((r["state"], r["label"]) for r in rows)
    (directory / "categories.json").write_text(json.dumps(labels), encoding="utf-8")
    (directory / "LICENSE").write_text("Small offline test fixture", encoding="utf-8")
    return {name: data.digest((directory / name).read_bytes()) for name in data.SOURCE_URLS}


def test_offline_preparation_contract_hashes_and_label_order(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    hashes = make_raw(raw)
    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Offline preparation made a network request"))
    output = tmp_path / "prepared"
    manifest = data.prepare_dataset(raw, output, expected_hashes=hashes)
    descriptor = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor["schema_version"] == 1 and descriptor["task_type"] == "choice"
    assert descriptor["choices"] == [{"id": "zeta_intent", "text": "zeta intent"}, {"id": "alpha_intent", "text": "alpha intent"}]
    assert list(descriptor["splits"]) == ["train", "validation", "calibration", "test"]
    assert manifest["provenance"]["official_source_hashes_verified"] is False
    for name, checksum in manifest["output_sha256"].items():
        assert data.digest((output / name).read_bytes()) == checksum
    for name in descriptor["splits"].values():
        assert all(set(json.loads(line)) == {"id", "state", "label"} for line in (output / name).read_text(encoding="utf-8").splitlines())
    assert manifest == data.prepare_dataset(raw, output, expected_hashes=hashes)


def test_raw_hash_failure_is_not_silently_accepted(tmp_path):
    raw = tmp_path / "raw"
    hashes = make_raw(raw)
    (raw / "test.csv").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch for test.csv"):
        data.prepare_dataset(raw, tmp_path / "out", expected_hashes=hashes)
    assert not (tmp_path / "out").exists()


def test_invalid_label_and_insufficient_clean_rows_fail():
    with pytest.raises(ValueError, match="unknown label"):
        data.read_rows(b"text,category\nhello,unknown\n", "train", ["known"])
    with pytest.raises(ValueError, match="at least 3"):
        data.prepare_records([record(0, "one", "a"), record(1, "two", "a")], [record(0, "test", "a", "test")], ["a"])


@pytest.mark.parametrize("n", [3, 4, 7, 10, 35, 100, 187])
def test_apportion_preserves_every_sample_and_all_partitions(n):
    sizes = data.apportion(n)
    assert sum(sizes) == n and all(x > 0 for x in sizes)
    if n >= 10:
        assert all(abs(actual - ratio * n) <= 1 for actual, ratio in zip(sizes, (.8, .1, .1)))
