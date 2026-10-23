"""Stop-mask reuse must preserve row-local cancellation/EOS/stop lengths."""
import threading
from types import SimpleNamespace

import pytest
import torch

from saddlellm.runtime.InferenceServer import _BatchGenerationControl
from tools.benchmark_batch_control import ReuseControl, benchmark_variant


class LegacyControl(_BatchGenerationControl):
    """Independent pre-optimization per-token host-mask reference."""
    def __call__(self, input_ids, scores):
        stop = torch.tensor([event.is_set() for event in self.cancel_events],
                            dtype=torch.bool, device=input_ids.device)
        for token_id in self.eos_ids:
            stop |= input_ids[:, -1].eq(token_id)
        if self.stop_checker is not None:
            stop |= self.stop_checker(input_ids)
        newly_finished = stop & ~self.finished
        self.lengths = torch.where(newly_finished,
                                  input_ids.shape[1] - self.prompt_tokens, self.lengths)
        self.finished |= stop
        return self.finished


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(str(int(token)) for token in ids)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("eos", [None, 2, [2, 3]])
@pytest.mark.parametrize("with_stop", [False, True])
def test_mask_reuse_matches_legacy_all_steps(device, eos, with_stop):
    rows = [SimpleNamespace(prompt_tokens=2, stop="78" if with_stop else None) for _ in range(4)]
    events = [threading.Event() for _ in rows]
    controls = [kind(Tokenizer(), rows, eos, events, device) for kind in (LegacyControl, ReuseControl)]
    ids = torch.tensor([[4, 5]] * 4, device=device)
    generated = [[6, 6, 7, 6], [2, 6, 8, 3], [0, 7, 6, 6], [0, 8, 9, 6]]
    pointer = controls[1]._stop_flags.data_ptr()
    for step, tokens in enumerate(generated):
        if step == 2:
            events[1].set()
        if step == 3:
            events[1].clear()  # A finished row must never be resurrected.
        ids = torch.cat((ids, torch.tensor(tokens, device=device)[:, None]), dim=1)
        torch.testing.assert_close(controls[0](ids, None), controls[1](ids, None), rtol=0, atol=0)
        torch.testing.assert_close(controls[0].lengths, controls[1].lengths, rtol=0, atol=0)
        assert controls[1]._stop_flags.data_ptr() == pointer


def test_normal_decode_never_constructs_a_new_host_mask(monkeypatch):
    rows = [SimpleNamespace(prompt_tokens=2, stop=None) for _ in range(4)]
    events = [threading.Event() for _ in rows]
    control = ReuseControl(Tokenizer(), rows, [2], events, "cpu")
    ids = torch.tensor([[4, 5, 6]] * 4)
    monkeypatch.setattr(torch, "tensor", lambda *a, **kw: pytest.fail("per-token host tensor construction"))
    assert control(ids, None).tolist() == [False] * 4
    events[2].set()
    assert control(ids, None).tolist() == [False, False, True, False]
    assert control.lengths.tolist() == [0, 0, 1, 0]


def test_native_batch_checks_eos_once_and_preserves_row_lengths(monkeypatch):
    from saddlellm.runtime.InferenceServer import CompletionRequest
    from test_inference_batching import make_engine

    engine = make_engine()
    engine.model.config.eos_token_id = [2, 3]
    rows = [CompletionRequest(prompt="4 5", temperature=0, max_tokens=5, stop=stop)
            for stop in (None, "7", None, None)]
    requests = [engine.prepare_batch_request(row) for row in rows]
    events = [threading.Event() for _ in rows]
    events[2].set()
    seen_eos = []
    create_session = engine.model.create_generation_session

    def recorded(*args, **kwargs):
        session = create_session(*args, **kwargs)
        seen_eos.append(session.eos_ids)
        return session

    def planned(input_ids, **kwargs):
        scores = torch.full((4, 1, 32), -100.0)
        for index, token in enumerate([3, 7, 8, 9]):
            scores[index, :, token] = 10
        return SimpleNamespace(logits=scores, past_key_values=None)

    monkeypatch.setattr(engine.model, "forward", planned)
    monkeypatch.setattr(engine.model, "create_generation_session", recorded)
    with benchmark_variant(engine.model, "reuse"):
        result = engine.generate_batch(requests, events)
    assert seen_eos == [[]]
    assert [row["usage"]["completion_tokens"] for row in result] == [1, 1, 1, 5]
    assert [row["choices"][0]["finish_reason"] for row in result] == ["stop", "stop", "stop", "length"]


def test_candidate_is_benchmark_only_and_restored_after_errors():
    from tools import benchmark_batch_control as benchmark
    from test_inference_batching import make_engine

    engine = make_engine()
    original = benchmark.runtime._BatchGenerationControl
    session = engine.model.create_generation_session
    assert original is _BatchGenerationControl
    with pytest.raises(RuntimeError, match="test failure"):
        with benchmark_variant(engine.model, "reuse"):
            assert benchmark.runtime._BatchGenerationControl is ReuseControl
            raise RuntimeError("test failure")
    assert benchmark.runtime._BatchGenerationControl is original
    assert engine.model.create_generation_session == session


def test_benchmark_speed_gate_rejects_noise_and_output_changes():
    from tools.benchmark_batch_control import summarize

    records = [{"variant": kind, "ms": 100 if kind == "legacy" else 96,
                "signature": "same"} for _ in range(7) for kind in ("legacy", "reuse")]
    assert summarize(records)["development_speed_gate"]
    records[-1]["signature"] = "different"
    assert not summarize(records)["development_speed_gate"]
    records[-1]["signature"] = "same"
    for row in records:
        if row["variant"] == "reuse":
            row["ms"] = 99
    assert not summarize(records)["development_speed_gate"]
