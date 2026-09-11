import pytest
import torch

from saddlellm.multimodal.LatentFlowModel import (
    ConditionalLatentFlowTransformer,
    LatentFlowConfig,
)


def _model():
    torch.manual_seed(17)
    return ConditionalLatentFlowTransformer(
        LatentFlowConfig(
            latent_channels=2,
            latent_height=4,
            latent_width=4,
            patch_size=2,
            condition_dim=6,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
        )
    )


def test_latent_flow_forward_loss_and_gradient():
    model = _model()
    target = torch.randn(2, 2, 4, 4)
    condition = torch.randn(2, 6)
    noise = torch.randn_like(target)
    timesteps = torch.tensor([0.2, 0.8])

    result = model.compute_flow_loss(
        target, condition, noise=noise, timesteps=timesteps
    )
    result["loss"].backward()

    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    assert model.output_projection.weight.grad is not None
    assert model.output_projection.weight.grad.abs().sum() > 0


def test_latent_flow_sampling_is_deterministic_with_fixed_noise():
    model = _model()
    condition = torch.randn(1, 6)
    noise = torch.randn(1, 2, 4, 4)

    first = model.sample(condition, num_steps=2, initial_noise=noise)
    second = model.sample(condition, num_steps=2, initial_noise=noise)

    assert first.shape == noise.shape
    assert torch.equal(first, second)


def test_latent_flow_rejects_incompatible_shapes():
    model = _model()
    with pytest.raises(ValueError, match="latents must have shape"):
        model(torch.randn(1, 2, 8, 8), torch.zeros(1), torch.randn(1, 6))
    with pytest.raises(ValueError, match="divisible by patch_size"):
        LatentFlowConfig(latent_height=7, latent_width=8, patch_size=2)
