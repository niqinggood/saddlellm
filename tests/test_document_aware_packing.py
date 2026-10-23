"""End-to-end correctness checks for document-aware packed pretraining."""

import torch

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
        attention_backend="sdpa",
    )
    return SaddleForCausalLM(config).eval()


def test_packed_documents_match_independent_forwards_and_cannot_leak_attention():
    torch.manual_seed(123)
    model = _tiny_model()
    first_document = torch.tensor([[1, 3, 5]])
    second_document = torch.tensor([[7, 9, 11, 13]])
    packed = torch.cat([first_document, second_document], dim=1)
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])

    with torch.no_grad():
        packed_logits = model(
            input_ids=packed,
            document_ids=document_ids,
            position_ids=position_ids,
        ).logits
        independent_logits = model(input_ids=second_document).logits

        # Changing every token in the preceding document must not alter any
        # output belonging to the second document.
        changed_prefix = torch.tensor([[2, 4, 6]])
        changed_logits = model(
            input_ids=torch.cat([changed_prefix, second_document], dim=1),
            document_ids=document_ids,
            position_ids=position_ids,
        ).logits

    second_slice = slice(first_document.shape[1], packed.shape[1])
    assert torch.allclose(
        packed_logits[:, second_slice], independent_logits, atol=1e-6, rtol=1e-5
    )
    assert torch.allclose(
        packed_logits[:, second_slice],
        changed_logits[:, second_slice],
        atol=1e-6,
        rtol=1e-5,
    )


def test_legacy_inputs_match_single_document_metadata_path():
    torch.manual_seed(456)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
    document_ids = torch.zeros_like(input_ids)

    with torch.no_grad():
        legacy_logits = model(input_ids=input_ids).logits
        explicit_logits = model(
            input_ids=input_ids,
            document_ids=document_ids,
            position_ids=position_ids,
        ).logits

    assert torch.allclose(legacy_logits, explicit_logits, atol=1e-6, rtol=1e-5)


def test_packed_metadata_survives_activation_checkpointing_backward():
    torch.manual_seed(789)
    model = _tiny_model().train()
    model.gradient_checkpointing_enable()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[1, 2, 3, -100, 5, 6]])
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])

    output = model(
        input_ids=input_ids,
        labels=labels,
        document_ids=document_ids,
        position_ids=position_ids,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()
