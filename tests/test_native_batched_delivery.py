"""Delivery fusion preserves sampling order, request state and callback fallback."""
from contextlib import ExitStack

import pytest
import torch

from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner
from saddlellm.models.NativeGeneration import NativeGenerationSession, _register_batch_streamer_type
from test_native_batch_decode import model


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA delivery fusion")


class Recorder:
    def __init__(self):
        self.tokens = []
        self.ends = 0

    def put(self, value):
        assert value.device.type == "cpu"
        self.tokens.extend(value.flatten().tolist())

    def end(self):
        self.ends += 1


_register_batch_streamer_type(Recorder)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("vocabulary", [37, 32000, 151936])
@pytest.mark.parametrize("top_k,top_p", [(0, 0.9), (40, 0.9), (0, 1.0)])
def test_batched_filtering_preserves_each_probability(dtype, vocabulary, top_k, top_p):
    instance = model().cuda().to(dtype=dtype)
    with instance.create_generation_session(torch.tensor([[4, 5]], device="cuda"),
                                            temperature=0.7, top_k=top_k, top_p=top_p) as session:
        generator = torch.Generator(device="cuda").manual_seed(720)
        logits = torch.randn((4, vocabulary), dtype=dtype, device="cuda", generator=generator)
        # Quantization-like ties and filtered values exercise nucleus boundaries.
        logits[:, :32] = 0
        expected = torch.cat([session._sampling_probabilities(row[None]) for row in logits])
        actual = session._sampling_probabilities(logits, _preserve_single_row=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rng", ["seed", "global", "shared"])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
def test_delivery_matches_sequential_completion(monkeypatch, rng, streaming, mixed):
    instance = model().cuda().bfloat16()
    prompts = [torch.tensor([[4, 5] + ([6] * index if mixed else [])], device="cuda")
               for index in range(3)]
    with instance.create_generation_session(prompts[0], max_new_tokens=5, seed=301,
                                            top_p=0.9, temperature=0.7, eos_token_id=[]) as probe:
        list(probe)
        eos = int(probe.sequences()[0, -1])

    def run(fused):
        torch.manual_seed(99)
        generator = torch.Generator(device="cuda").manual_seed(99)
        with monkeypatch.context() as patch, ExitStack() as stack:
            if not fused:
                patch.setattr(NativeGenerationSession, "_supports_batched_delivery", lambda self: False)
            recorders = [Recorder() if streaming else None for _ in prompts]
            sessions = [stack.enter_context(instance.create_generation_session(
                prompt, max_new_tokens=7 + index, min_new_tokens=2,
                do_sample=True, top_p=0.9, top_k=13, temperature=0.7,
                repetition_penalty=1.1, eos_token_id=[eos] if index < 2 else [],
                output_scores=True, streamer=recorder,
                **({"seed": 301 + index} if rng == "seed" else {"generator": generator} if rng == "shared" else {}),
            )) for index, (prompt, recorder) in enumerate(zip(prompts, recorders))]
            runner = stack.enter_context(NativeBatchDecodeRunner(instance))
            tick = 0
            while live := [session for session in sessions if not session.done]:
                runner.step(live if tick % 2 else list(reversed(live)))
                tick += 1
            return ([(session.sequences().clone(), [score.clone() for score in session.scores],
                      session.finished.clone(), session.finished_count, session.finish_reason,
                      session.mask_has_padding, session.generated_tokens,
                      None if recorder is None else (recorder.tokens, recorder.ends))
                     for session, recorder in zip(sessions, recorders)], runner.snapshot())

    expected, reference_stats = run(False)
    actual, stats = run(True)
    assert reference_stats["batched_host_transfers"] == 0
    assert stats["batched_host_transfers"] > 0
    assert stats["batched_host_tokens"] >= 2 * stats["batched_host_transfers"]
    for observed, reference in zip(actual, expected):
        assert torch.equal(observed[0], reference[0])
        for left, right in zip(observed[1], reference[1]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert torch.equal(observed[2], reference[2])
        assert observed[3:] == reference[3:]


@pytest.mark.parametrize("unsupported", ["custom_stop", "custom_streamer", "registered_subclass",
                                         "custom_sample", "custom_end", "custom_class_end"])
def test_arbitrary_callbacks_keep_original_order_and_rng(monkeypatch, unsupported):
    instance = model().cuda().bfloat16()

    if unsupported == "custom_class_end":
        original_end = NativeGenerationSession._end_streamer

        def end(self):
            torch.rand((), device="cuda")
            original_end(self)

        monkeypatch.setattr(NativeGenerationSession, "_end_streamer", end)

    class UserStreamer(Recorder):
        # A self-declared attribute must not opt a callback into batching.
        _supports_batched_delivery = True

        def put(self, value):
            super().put(value)
            torch.rand((), device="cuda")

    class RegisteredSubclass(Recorder):
        pass

    def run():
        torch.manual_seed(592)
        with ExitStack() as stack:
            sessions = []
            for index in range(3):
                streamer = (UserStreamer() if unsupported == "custom_streamer"
                            else RegisteredSubclass() if unsupported == "registered_subclass" else Recorder())
                session = stack.enter_context(instance.create_generation_session(
                    torch.tensor([[4 + index, 5]], device="cuda"), max_new_tokens=5,
                    do_sample=True, eos_token_id=[2], min_new_tokens=5,
                    streamer=streamer,
                    stopping_criteria=(lambda ids, scores: bool(torch.rand(()).item() < 0))
                    if unsupported == "custom_stop" else None,
                ))
                if unsupported == "custom_sample":
                    original = session._sample
                    session._sample = lambda logits, out, original=original: original(logits, out)
                if unsupported == "custom_end":
                    original = session._end_streamer

                    def end(original=original):
                        torch.rand((), device="cuda")
                        original()

                    session._end_streamer = end
                sessions.append(session)
            runner = stack.enter_context(NativeBatchDecodeRunner(instance))
            while live := [session for session in sessions if not session.done]:
                runner.step(live)
            return [session.sequences().clone() for session in sessions], runner.snapshot()

    actual, stats = run()
    assert stats["batched_host_transfers"] == 0
    monkeypatch.setattr(NativeGenerationSession, "_supports_batched_delivery", lambda self: False)
    expected, _ = run()
    assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_deferred_streamer_failure_marks_every_advanced_session_failed():
    instance = model().cuda().bfloat16()

    class FailingRecorder(Recorder):
        def put(self, value):
            if len(self.tokens) > 2:
                raise RuntimeError("delivery failed")
            super().put(value)

    _register_batch_streamer_type(FailingRecorder)
    with ExitStack() as stack:
        sessions = [stack.enter_context(instance.create_generation_session(
            torch.tensor([[4 + index, 5]], device="cuda"), max_new_tokens=5,
            do_sample=False, eos_token_id=[], streamer=FailingRecorder(),
        )) for index in range(2)]
        runner = stack.enter_context(NativeBatchDecodeRunner(instance))
        runner.step(sessions)
        with pytest.raises(RuntimeError, match="delivery failed"):
            runner.step(sessions)
        assert all(session._failed for session in sessions)
        assert runner.snapshot()["retained_cohorts"] == 0
