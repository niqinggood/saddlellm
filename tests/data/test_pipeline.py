"""Tests for the canonical data-pipeline API and legacy import path."""

import sys
import types

import pytest

import saddlellm.data.pipeline as pipeline_module
from saddlellm.DataPipeline import (
    DataMixConfig,
    DataMixer,
    DataPipeline,
    PipelineConfig,
)


def _fake_datasets_module(monkeypatch, *, load_dataset=None, from_generator=None):
    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset or (lambda *_args, **_kwargs: object())
    module.interleave_datasets = lambda datasets, **_kwargs: datasets
    module.concatenate_datasets = lambda datasets: datasets
    if from_generator is not None:
        module.Dataset = types.SimpleNamespace(from_generator=from_generator)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dedup_threshold": -0.1}, "dedup_threshold"),
        ({"quality_min_score": 1.1}, "quality_min_score"),
        ({"split_ratios": [1.0]}, "split_ratios"),
        ({"split_ratios": [0.7, 0.4]}, "split_ratios"),
        ({"mix_strategy": "unknown"}, "mix_strategy"),
        ({"num_proc": 0}, "num_proc"),
    ],
)
def test_pipeline_config_rejects_invalid_boundaries(overrides, message):
    with pytest.raises(ValueError, match=message):
        DataPipeline(PipelineConfig(**overrides))


def test_source_weight_validation_rejects_mismatch_negative_and_zero_total():
    sources = [{"type": "wikitext"}, {"type": "wikitext"}]
    with pytest.raises(ValueError, match="length"):
        DataPipeline(PipelineConfig(sources=sources, source_weights=[1]))
    with pytest.raises(ValueError, match="non-negative"):
        DataPipeline(PipelineConfig(sources=sources, source_weights=[1, -1]))
    with pytest.raises(ValueError, match="positive total"):
        DataPipeline(PipelineConfig(sources=sources, source_weights=[0, 0]))


def test_wikipedia_source_uses_configured_snapshot_and_cache(monkeypatch):
    calls = []
    dataset = object()

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return dataset

    _fake_datasets_module(monkeypatch, load_dataset=load_dataset)
    pipeline = DataPipeline(
        PipelineConfig(
            sources=[
                {
                    "type": "wikipedia",
                    "lang": "en",
                    "date": "20250101",
                    "split": "validation",
                    "streaming": False,
                }
            ],
            cache_dir="cache-dir",
        )
    )

    result = pipeline.collect()

    assert result is pipeline
    assert pipeline.to_iterable_dataset() is dataset
    assert calls == [
        (
            ("wikipedia", "20250101.en"),
            {
                "split": "validation",
                "streaming": False,
                "cache_dir": "cache-dir",
            },
        )
    ]


def test_unknown_source_type_fails_before_incrementing_stats(monkeypatch):
    _fake_datasets_module(monkeypatch)
    pipeline = DataPipeline(PipelineConfig(sources=[{"type": "mystery"}]))

    with pytest.raises(ValueError, match="unsupported data source type"):
        pipeline.collect()

    assert pipeline.stats.get("source_files", 0) == 0


def test_explicit_zero_dedup_threshold_is_not_replaced_by_default(monkeypatch):
    pipeline = DataPipeline(PipelineConfig(dedup_threshold=0.8))
    pipeline._dataset = object()
    thresholds = []
    monkeypatch.setattr(pipeline, "_dedup_simhash", thresholds.append)

    pipeline.deduplicate(method="simhash", threshold=0)

    assert thresholds == [0]


def test_explicit_zero_min_length_is_honored():
    class FilterDataset:
        def __init__(self):
            self.kept = None

        def filter(self, predicate):
            self.kept = predicate({"text": "abc def"})
            return self

    dataset = FilterDataset()
    pipeline = DataPipeline(
        PipelineConfig(min_text_length=50, quality_min_score=0, filter_urls=False)
    )
    pipeline._dataset = dataset

    pipeline.filter_quality(min_length=0)

    assert dataset.kept is True


def test_split_honors_seed_zero_and_normalizes_override_ratios():
    class SplitDataset:
        def __init__(self):
            self.call = None

        def train_test_split(self, **kwargs):
            self.call = kwargs
            return {"train": "train", "test": "test"}

    dataset = SplitDataset()
    pipeline = DataPipeline(PipelineConfig(streaming=False))
    pipeline._dataset = dataset

    result = pipeline.split(ratios=[3, 1], seed=0)

    assert result == {"train": "train", "val": "test"}
    assert dataset.call == {"test_size": 0.25, "seed": 0, "shuffle": True}


@pytest.mark.parametrize("method", ["to_iterable_dataset", "save_to_disk", "split"])
def test_dataset_consumers_require_collection_first(method, tmp_path):
    pipeline = DataPipeline(PipelineConfig())
    args = [str(tmp_path)] if method == "save_to_disk" else []

    with pytest.raises(RuntimeError, match=r"Call collect\(\)"):
        getattr(pipeline, method)(*args)


def test_pack_generator_is_reentrant(monkeypatch):
    generated = []

    def from_generator(factory):
        generated.append(list(factory()))
        generated.append(list(factory()))
        return generated

    _fake_datasets_module(monkeypatch, from_generator=from_generator)
    pipeline = DataPipeline(PipelineConfig())
    source = [{"input_ids": list(range(13))}]

    pipeline._pack_sequences(source, max_len=10, pad_token_id=0)

    assert generated[0] == generated[1]
    assert generated[0][0]["input_ids"] == list(range(10))


def test_data_mixer_normalizes_copies_without_mutating_callers():
    first = DataMixConfig("first", 3, sources=[{"type": "wikitext"}])
    second = DataMixConfig("second", 1, sources=[{"type": "wikitext"}])

    mixer = DataMixer([first, second])

    assert (first.weight, second.weight) == (3, 1)
    assert [config.weight for config in mixer.configs] == [0.75, 0.25]


def test_data_mixer_rejects_empty_or_zero_weight_configurations():
    with pytest.raises(ValueError, match="at least one"):
        DataMixer([])
    with pytest.raises(ValueError, match="positive total"):
        DataMixer([DataMixConfig("zero", 0)])


def test_oversample_strategy_repeats_higher_weight_sources(monkeypatch):
    concatenations = []
    interleaves = []
    datasets_module = _fake_datasets_module(monkeypatch)

    def concatenate(datasets):
        concatenations.append(list(datasets))
        return tuple(datasets)

    def interleave(datasets, **kwargs):
        interleaves.append((datasets, kwargs))
        return "mixed"

    datasets_module.concatenate_datasets = concatenate
    datasets_module.interleave_datasets = interleave

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def collect(self):
            return self

        def clean(self):
            return self

        def deduplicate(self):
            return self

        def filter_quality(self):
            return self

        def tokenize_and_pack(self, _tokenizer):
            return self

        def to_iterable_dataset(self):
            return self.config.sources[0]["id"]

    monkeypatch.setattr(pipeline_module, "DataPipeline", FakePipeline)
    mixer = DataMixer(
        [
            DataMixConfig("high", 3, sources=[{"id": "high"}]),
            DataMixConfig("low", 1, sources=[{"id": "low"}]),
        ],
        mix_strategy="oversample",
    )

    result = mixer.mix_and_process(tokenizer=object())

    assert result == "mixed"
    assert concatenations == [["high", "high", "high"], ["low"]]
    assert interleaves[0][1] == {"seed": 42}
