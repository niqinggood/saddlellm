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
from saddlellm.data.pipeline import CausalLMDataCollator


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


def test_map_style_stateless_preprocessing_uses_configured_num_proc(monkeypatch):
    class MapDataset:
        column_names = ["text"]
        num_rows = 1

        def __init__(self):
            self.calls = []

        def map(self, _function, **kwargs):
            self.calls.append(("map", kwargs))
            return self

        def filter(self, _predicate, **kwargs):
            self.calls.append(("filter", kwargs))
            return self

    class StreamDataset(MapDataset):
        pass

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = MapDataset
    datasets_module.IterableDataset = StreamDataset

    dataset = MapDataset()
    pipeline = DataPipeline(
        PipelineConfig(
            num_proc=3,
            min_text_length=0,
            quality_min_score=0,
            pack_sequences=False,
        )
    )
    pipeline._dataset = dataset

    pipeline.clean()
    assert dataset.calls == [
        ("map", {"num_proc": 3}),
        ("filter", {"num_proc": 3}),
    ]

    dataset.calls.clear()
    pipeline.filter_quality()
    assert dataset.calls == [("filter", {"num_proc": 3})]

    dataset.calls.clear()
    tokenizer = types.SimpleNamespace(eos_token_id=2, pad_token_id=0)
    pipeline.tokenize_and_pack(tokenizer)
    assert dataset.calls == [
        (
            "map",
            {
                "batched": True,
                "remove_columns": ["text"],
                "num_proc": 3,
            },
        ),
        ("map", {"num_proc": 3}),
    ]


@pytest.mark.parametrize(
    ("dataset_kind", "num_proc"),
    [("stream", 4), ("double", 4), ("map", 1)],
)
def test_num_proc_is_omitted_for_streaming_test_doubles_and_single_worker(
    monkeypatch, dataset_kind, num_proc
):
    class MapDataset:
        column_names = ["text"]
        num_rows = 1

        def __init__(self):
            self.calls = []

        def map(self, _function, **kwargs):
            self.calls.append(("map", kwargs))
            return self

        def filter(self, _predicate, **kwargs):
            self.calls.append(("filter", kwargs))
            return self

    class StreamDataset(MapDataset):
        pass

    class TestDouble:
        column_names = ["text"]
        num_rows = 1

        def __init__(self):
            self.calls = []

        def map(self, _function, **kwargs):
            self.calls.append(("map", kwargs))
            return self

        def filter(self, _predicate, **kwargs):
            self.calls.append(("filter", kwargs))
            return self

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = MapDataset
    datasets_module.IterableDataset = StreamDataset
    dataset = {
        "stream": StreamDataset,
        "double": TestDouble,
        "map": MapDataset,
    }[dataset_kind]()
    pipeline = DataPipeline(
        PipelineConfig(num_proc=num_proc, min_text_length=0, quality_min_score=0)
    )
    pipeline._dataset = dataset

    pipeline.clean()

    assert dataset.calls == [("map", {}), ("filter", {})]


@pytest.mark.parametrize("method", ["exact", "simhash", "minhash"])
def test_stateful_deduplication_never_receives_num_proc(monkeypatch, method):
    class MapDataset:
        num_rows = 1

        def __init__(self):
            self.filter_kwargs = None

        def filter(self, _predicate, **kwargs):
            self.filter_kwargs = kwargs
            return self

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = MapDataset
    datasets_module.IterableDataset = type("StreamDataset", (), {})
    simhash_module = types.ModuleType("simhash")
    simhash_module.Simhash = type("Simhash", (), {"__init__": lambda self, _text: None})
    monkeypatch.setitem(sys.modules, "simhash", simhash_module)
    datasketch_module = types.ModuleType("datasketch")
    datasketch_module.MinHash = type("MinHash", (), {})
    datasketch_module.MinHashLSH = type(
        "MinHashLSH",
        (),
        {"__init__": lambda self, **_kwargs: None},
    )
    monkeypatch.setitem(sys.modules, "datasketch", datasketch_module)
    dataset = MapDataset()
    pipeline = DataPipeline(PipelineConfig(num_proc=8))
    pipeline._dataset = dataset

    pipeline.deduplicate(method=method)

    assert dataset.filter_kwargs == {}


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


def test_quality_filter_accepts_valid_chinese_without_spaces():
    class FilterDataset:
        def __init__(self):
            self.kept = None

        def filter(self, predicate):
            self.kept = predicate(
                {"text": "护士应核对患者身份，评估意识、呼吸和循环情况，并记录生命体征变化。"}
            )
            return self

    dataset = FilterDataset()
    pipeline = DataPipeline(
        PipelineConfig(min_text_length=20, quality_min_score=0, filter_urls=False)
    )
    pipeline._dataset = dataset

    pipeline.filter_quality()

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


def test_streaming_pack_stays_iterable(monkeypatch):
    class StreamDataset(list):
        @classmethod
        def from_generator(cls, factory):
            return cls(factory())

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = types.SimpleNamespace(
        from_generator=lambda _factory: pytest.fail("stream was materialized")
    )
    datasets_module.IterableDataset = StreamDataset
    pipeline = DataPipeline(PipelineConfig())

    packed = pipeline._pack_sequences(
        StreamDataset([{"input_ids": list(range(12))}]),
        max_len=10,
        pad_token_id=0,
    )

    assert isinstance(packed, StreamDataset)
    assert packed[0]["input_ids"] == list(range(10))


def test_packed_tokenization_keeps_long_documents_and_adds_eos(monkeypatch):
    class ListDataset(list):
        column_names = ["text"]

        def map(self, function, batched=False, **_kwargs):
            if batched:
                mapped = function({"text": [row["text"] for row in self]})
                return ListDataset(
                    {"input_ids": ids} for ids in mapped["input_ids"]
                )
            return ListDataset(function(row) for row in self)

    calls = []

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            calls.append(kwargs)
            return {"input_ids": [list(range(1, 24)) for _ in texts]}

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = types.SimpleNamespace(
        from_generator=lambda factory: list(factory())
    )
    pipeline = DataPipeline(
        PipelineConfig(max_seq_length=10, pack_sequences=True)
    )
    pipeline._dataset = ListDataset([{"text": "long document"}])

    pipeline.tokenize_and_pack(Tokenizer())

    assert calls == [
        {"truncation": False, "padding": False, "return_attention_mask": False}
    ]
    packed = pipeline.to_iterable_dataset()
    assert packed[0]["input_ids"] == list(range(1, 11))
    assert packed[1]["input_ids"] == list(range(11, 21))


def test_packed_tokenization_inserts_eos_between_documents(monkeypatch):
    class ListDataset(list):
        column_names = ["text"]

        def map(self, function, batched=False, **_kwargs):
            mapped = function({"text": [row["text"] for row in self]})
            return ListDataset({"input_ids": ids} for ids in mapped["input_ids"])

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

        def __call__(self, _texts, **_kwargs):
            return {"input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]}

    datasets_module = _fake_datasets_module(monkeypatch)
    datasets_module.Dataset = types.SimpleNamespace(
        from_generator=lambda factory: list(factory())
    )
    pipeline = DataPipeline(PipelineConfig(max_seq_length=10))
    pipeline._dataset = ListDataset([{"text": "a"}, {"text": "b"}])

    pipeline.tokenize_and_pack(Tokenizer())

    assert pipeline.to_iterable_dataset()[0]["input_ids"] == [
        1,
        2,
        3,
        4,
        5,
        99,
        6,
        7,
        8,
        9,
    ]


def test_packing_tracks_document_boundaries_resets_positions_and_masks_loss(monkeypatch):
    generated = []

    def from_generator(factory):
        generated.extend(factory())
        return generated

    _fake_datasets_module(monkeypatch, from_generator=from_generator)
    pipeline = DataPipeline(PipelineConfig())
    pipeline._pack_sequences(
        [
            {"input_ids": [1, 2, 99]},
            {"input_ids": [3, 4, 99]},
        ],
        max_len=6,
        pad_token_id=0,
    )

    assert generated == [
        {
            "input_ids": [1, 2, 99, 3, 4, 99],
            # The model shifts labels left.  -100 at the new document's first
            # token prevents token 99 from predicting token 3 across documents.
            "labels": [1, 2, 99, -100, 4, 99],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "document_ids": [0, 0, 0, 1, 1, 1],
            "position_ids": [0, 1, 2, 0, 1, 2],
        }
    ]


def test_long_document_fragments_restart_positions_at_block_boundaries(monkeypatch):
    _fake_datasets_module(
        monkeypatch,
        from_generator=lambda factory: list(factory()),
    )
    pipeline = DataPipeline(PipelineConfig())

    packed = pipeline._pack_sequences(
        [{"input_ids": list(range(24))}],
        max_len=10,
        pad_token_id=0,
    )

    assert len(packed) == 2
    assert packed[0]["position_ids"] == list(range(10))
    assert packed[1]["position_ids"] == list(range(10))
    assert packed[0]["document_ids"] == [0] * 10
    assert packed[1]["document_ids"] == [0] * 10


def test_causal_collator_preserves_real_eos_when_pad_equals_eos():
    tokenizer = types.SimpleNamespace(
        eos_token_id=2, pad_token_id=2, padding_side="right"
    )
    batch = CausalLMDataCollator(tokenizer)(
        [{"input_ids": [1, 2]}, {"input_ids": [3]}]
    )

    assert batch["input_ids"].tolist() == [[1, 2], [3, 2]]
    assert batch["labels"].tolist() == [[1, 2], [3, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_causal_collator_omits_redundant_full_attention_mask():
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=0,
        padding_side="right",
    )
    features = [
        {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]},
        {"input_ids": [4, 5, 6], "labels": [4, 5, 6], "attention_mask": [1, 1, 1]},
    ]

    optimized = CausalLMDataCollator(tokenizer)(features)
    explicit = CausalLMDataCollator(
        tokenizer,
        omit_all_one_attention_mask=False,
    )(features)

    assert "attention_mask" not in optimized
    assert "document_ids" not in optimized
    assert "position_ids" not in optimized
    assert explicit["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]


def test_causal_collator_pads_packed_document_metadata():
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=2,
        padding_side="right",
    )
    features = [
        {
            "input_ids": [1, 2, 3],
            "labels": [1, 2, -100],
            "document_ids": [0, 0, 1],
            "position_ids": [0, 1, 0],
        },
        {
            "input_ids": [4, 5],
            "document_ids": [0, 0],
            "position_ids": [0, 1],
        },
    ]

    batch = CausalLMDataCollator(tokenizer)(features)

    assert batch["document_ids"].tolist() == [[0, 0, 1], [0, 0, -1]]
    assert batch["position_ids"].tolist() == [[0, 1, 0], [0, 1, 0]]
    assert batch["labels"].tolist() == [[1, 2, -100], [4, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_causal_collator_keeps_fast_path_for_single_document_packs():
    tokenizer = types.SimpleNamespace(
        pad_token_id=0,
        eos_token_id=2,
        padding_side="right",
    )
    features = [
        {
            "input_ids": [1, 2, 3],
            "document_ids": [17, 17, 17],
            "position_ids": [0, 1, 2],
        }
    ]

    batch = CausalLMDataCollator(tokenizer)(features)

    assert set(batch) == {"input_ids", "labels"}


def test_streaming_split_is_deterministic_and_leakage_free():
    class StreamDataset(list):
        def filter(self, predicate):
            return StreamDataset(row for row in self if predicate(row))

    source = StreamDataset({"text": f"document-{index}"} for index in range(100))
    pipeline = DataPipeline(PipelineConfig(streaming=True))
    pipeline._dataset = source

    first = pipeline.split(ratios=[0.8, 0.2], seed=7)
    second = pipeline.split(ratios=[0.8, 0.2], seed=7)
    train = {row["text"] for row in first["train"]}
    validation = {row["text"] for row in first["val"]}

    assert train
    assert validation
    assert train.isdisjoint(validation)
    assert train | validation == {row["text"] for row in source}
    assert first == second


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
