import pytest
import torch

from saddlellm.multimodal.MusicCodeModel import (
    ConditionalMusicCodeTransformer,
    MusicCodeConfig,
)


def _model():
    torch.manual_seed(23)
    return ConditionalMusicCodeTransformer(
        MusicCodeConfig(
            num_codebooks=2,
            codebook_size=8,
            max_sequence_length=8,
            condition_dim=6,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
        )
    )


def test_music_code_loss_backpropagates_all_codebooks():
    model = _model()
    codes = torch.randint(0, 8, (2, 2, 5))
    condition = torch.randn(2, 6)
    result = model.compute_loss(codes, condition)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert all(head.weight.grad is not None for head in model.output_heads)


def test_music_code_greedy_generation_shape_and_range():
    model = _model()
    generated = model.generate(torch.zeros(1, 6), max_frames=5, temperature=0)
    assert generated.shape == (1, 2, 5)
    assert generated.min() >= 0
    assert generated.max() < 8


def test_music_code_rejects_bad_codebook_shape():
    model = _model()
    with pytest.raises(ValueError, match="contain 2 codebooks"):
        model(torch.zeros(1, 1, 3, dtype=torch.long), torch.zeros(1, 6))
