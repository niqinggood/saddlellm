import torch
import torch.nn.functional as F

from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _tiny_model() -> SaddleForCausalLM:
    config = SaddleModelConfig(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_kind="gqa",
        attention_backend="eager",
    )
    return SaddleForCausalLM(config).eval()


def _legacy_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _capture_layer_attention_masks(model: SaddleForCausalLM):
    captured = []

    def capture(_module, _args, kwargs):
        captured.append(kwargs.get("attention_mask"))

    handles = [
        layer.register_forward_pre_hook(capture, with_kwargs=True)
        for layer in model.layers
    ]
    return captured, handles


def test_causal_lm_loss_matches_legacy_loss_and_logits_gradients_with_ignored_labels():
    torch.manual_seed(211)
    labels = torch.tensor(
        [
            [0, 1, -100, 3, 4],
            [5, -100, 2, 6, -100],
        ],
        dtype=torch.long,
    )
    base_logits = torch.randn(2, 5, 7, dtype=torch.float64)
    legacy_logits = base_logits.clone().requires_grad_(True)
    optimized_logits = base_logits.clone().requires_grad_(True)

    legacy_loss = _legacy_causal_lm_loss(legacy_logits, labels)
    optimized_loss = _tiny_model()._causal_lm_loss(optimized_logits, labels)
    legacy_gradient = torch.autograd.grad(legacy_loss, legacy_logits)[0]
    optimized_gradient = torch.autograd.grad(optimized_loss, optimized_logits)[0]

    assert torch.allclose(optimized_loss, legacy_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        optimized_gradient,
        legacy_gradient,
        atol=1e-12,
        rtol=1e-12,
    )
    assert torch.count_nonzero(optimized_gradient[:, -1]) == 0


def test_all_one_2d_attention_mask_is_folded_once_and_logits_match_no_mask():
    torch.manual_seed(223)
    model = _tiny_model()
    input_ids = torch.tensor(
        [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]],
        dtype=torch.long,
    )
    all_one_mask = torch.ones_like(input_ids)
    captured, handles = _capture_layer_attention_masks(model)

    try:
        with torch.no_grad():
            masked_logits = model(
                input_ids=input_ids,
                attention_mask=all_one_mask,
            ).logits
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        unmasked_logits = model(input_ids=input_ids).logits

    assert captured == [None] * len(model.layers)
    assert torch.allclose(masked_logits, unmasked_logits, atol=1e-6, rtol=1e-5)


def test_padding_attention_mask_with_zeros_is_not_folded():
    torch.manual_seed(227)
    model = _tiny_model()
    input_ids = torch.tensor(
        [[0, 0, 5, 7, 9, 11], [0, 4, 6, 8, 10, 12]],
        dtype=torch.long,
    )
    padding_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    captured, handles = _capture_layer_attention_masks(model)

    try:
        with torch.no_grad():
            padded_logits = model(
                input_ids=input_ids,
                attention_mask=padding_mask,
            ).logits
    finally:
        for handle in handles:
            handle.remove()
    with torch.no_grad():
        unmasked_logits = model(input_ids=input_ids).logits

    assert len(captured) == len(model.layers)
    assert all(mask is padding_mask for mask in captured)
    assert not torch.allclose(padded_logits, unmasked_logits)


def test_all_one_mask_fastpath_preserves_cached_decode_result():
    torch.manual_seed(229)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 3, 5, 7, 9, 11]], dtype=torch.long)
    all_one_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        full_last = model(
            input_ids=input_ids,
            attention_mask=all_one_mask,
        ).logits[:, -1]
        prefix = model(
            input_ids=input_ids[:, :-1],
            attention_mask=all_one_mask[:, :-1],
            use_cache=True,
        )

        captured, handles = _capture_layer_attention_masks(model)
        try:
            cached_last = model(
                input_ids=input_ids[:, -1:],
                attention_mask=all_one_mask,
                past_key_values=prefix.past_key_values,
                use_cache=True,
            ).logits[:, -1]
        finally:
            for handle in handles:
                handle.remove()

    assert captured == [None] * len(model.layers)
    assert torch.allclose(full_last, cached_last, atol=2e-5, rtol=2e-4)
