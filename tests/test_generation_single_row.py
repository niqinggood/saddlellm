"""Compare the one-row control fast path against the general row controller."""

import pytest
import torch

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


class Capture:
    def __init__(self):
        self.tokens = []
        self.ended = 0

    def put(self, tokens):
        self.tokens.append(tokens.clone())

    def end(self):
        self.ended += 1


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["eos", "min_tokens", "no_eos", "sample"])
def test_single_row_matches_general_tokens_scores_and_end_state(device, stream, mode):
    torch.manual_seed(572)
    model = SaddleForCausalLM(SaddleModelConfig(
        vocab_size=23, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=32,
        attention_backend="eager",
    )).eval().to(device)
    prompt = torch.tensor([[0, 4, 5]], device=device)
    mask = torch.tensor([[0, 1, 1]], device=device)
    with torch.inference_mode():
        first = int(model(prompt, attention_mask=mask).logits[:, -1].argmax(-1).item())
    results = []
    for fast in (False, True):
        capture = Capture() if stream else None
        with model.create_generation_session(
            prompt, attention_mask=mask, max_new_tokens=5,
            min_new_tokens=2 if mode == "min_tokens" else 0,
            do_sample=mode == "sample", seed=772, top_k=5, top_p=0.9,
            repetition_penalty=1.1, eos_token_id=[] if mode == "no_eos" else [first, (first + 1) % 23],
            streamer=capture, output_scores=True,
        ) as session:
            session._single_row_fastpath = fast
            list(session)
            results.append((session.sequences(), session.scores, session.finish_reason,
                            session.generated_tokens, session.finished_count,
                            session.finished.clone(), capture))
    ref, own = results
    assert torch.equal(ref[0], own[0])
    assert ref[2:5] == own[2:5]
    assert torch.equal(ref[5], own[5])
    assert len(ref[1]) == len(own[1])
    for before, after in zip(ref[1], own[1]):
        assert torch.equal(before, after)
    if stream:
        assert ref[6].ended == own[6].ended == 1
        assert len(ref[6].tokens) == len(own[6].tokens)
        assert all(torch.equal(a, b) for a, b in zip(ref[6].tokens, own[6].tokens))
