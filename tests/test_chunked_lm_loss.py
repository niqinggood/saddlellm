import copy
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import saddlellm.models.SaddleModeling as modeling
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig


def _tiny_config(*, loss_chunk_size: int = 0, mtp_extra_tokens: int = 0):
    return SaddleModelConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_backend="eager",
        loss_chunk_size=loss_chunk_size,
        multi_token_prediction=mtp_extra_tokens > 0,
        mtp_extra_tokens=mtp_extra_tokens,
    )


@pytest.mark.parametrize("invalid", [True, 1.5, "4"])
def test_loss_chunk_size_requires_an_integer(invalid):
    with pytest.raises(TypeError, match="loss_chunk_size must be an integer"):
        _tiny_config(loss_chunk_size=invalid)


def test_loss_chunk_size_rejects_negative_values():
    with pytest.raises(ValueError, match="loss_chunk_size must be non-negative"):
        _tiny_config(loss_chunk_size=-1)


def test_loss_chunk_size_round_trips_and_old_configs_keep_legacy_default(tmp_path):
    config = _tiny_config(loss_chunk_size=7)
    config.save_pretrained(str(tmp_path))
    payload = json.loads((tmp_path / "saddle_config.json").read_text(encoding="utf-8"))
    assert SaddleModelConfig(**payload).loss_chunk_size == 7

    payload.pop("loss_chunk_size")
    assert SaddleModelConfig(**payload).loss_chunk_size == 0


def test_chunked_training_loss_and_all_parameter_gradients_match_full_logits():
    torch.manual_seed(401)
    full_model = SaddleForCausalLM(_tiny_config()).double().train()
    chunked_model = copy.deepcopy(full_model)
    chunked_model.config.loss_chunk_size = 3
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[0, 2] = -100
    labels[1, 4] = -100

    full_output = full_model(input_ids=input_ids, labels=labels)
    chunked_output = chunked_model(input_ids=input_ids, labels=labels)

    assert full_output.logits.shape == (2, 5, full_model.config.vocab_size)
    assert chunked_output.logits is None
    assert torch.allclose(chunked_output.loss, full_output.loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        chunked_output.loss_items["lm_loss"],
        full_output.loss_items["lm_loss"],
        atol=1e-12,
        rtol=1e-12,
    )

    full_output.loss.backward()
    chunked_output.loss.backward()
    full_parameters = dict(full_model.named_parameters())
    chunked_parameters = dict(chunked_model.named_parameters())
    assert full_parameters.keys() == chunked_parameters.keys()
    for name, full_parameter in full_parameters.items():
        full_gradient = full_parameter.grad
        chunked_gradient = chunked_parameters[name].grad
        assert full_gradient is not None, name
        assert chunked_gradient is not None, name
        assert torch.allclose(
            chunked_gradient,
            full_gradient,
            atol=2e-12,
            rtol=2e-12,
        ), name


def test_chunked_linear_ce_respects_boundaries_and_recomputes_during_backward():
    torch.manual_seed(409)
    model = SaddleForCausalLM(_tiny_config(loss_chunk_size=3)).double()
    hidden = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(19, 8, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(
        [[1, 2, -100, 4, 5], [6, 7, 8, -100, 10]],
        dtype=torch.long,
    )
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_logits = F.linear(reference_hidden.reshape(-1, 8), reference_weight)
    reference_loss = F.cross_entropy(
        reference_logits,
        labels.reshape(-1),
        ignore_index=-100,
    )

    original_linear = modeling.F.linear
    projected_shapes = []
    saved_forward_shapes = []

    def recording_linear(inputs, projection_weight, bias=None):
        output = original_linear(inputs, projection_weight, bias)
        if projection_weight is weight:
            projected_shapes.append(tuple(output.shape))
        return output

    def record_saved_tensor(tensor):
        saved_forward_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(
        record_saved_tensor,
        lambda tensor: tensor,
    ):
        forward_probe = model._chunked_linear_cross_entropy(
            hidden,
            labels,
            weight,
            chunk_size=3,
        )
    del forward_probe

    with mock.patch.object(modeling.F, "linear", side_effect=recording_linear):
        chunked_loss = model._chunked_linear_cross_entropy(
            hidden,
            labels,
            weight,
            chunk_size=3,
        )
        chunked_loss.backward()
    reference_loss.backward()

    assert torch.allclose(chunked_loss, reference_loss, atol=1e-12, rtol=1e-12)
    assert torch.allclose(hidden.grad, reference_hidden.grad, atol=1e-12, rtol=1e-12)
    assert torch.allclose(weight.grad, reference_weight.grad, atol=1e-12, rtol=1e-12)
    # Ten tokens form 3, 3, 3, 1 chunks.  Non-reentrant checkpointing runs the
    # projection once in forward and once again in backward for each chunk.
    assert projected_shapes == [
        (3, 19),
        (3, 19),
        (3, 19),
        (1, 19),
        (1, 19),
        (3, 19),
        (3, 19),
        (3, 19),
    ]
    assert (10, 19) not in projected_shapes
    # Checkpoint saves chunk inputs/weight references, but neither a complete
    # nor a per-chunk vocabulary projection survives the forward pass.
    assert not any(
        len(shape) == 2 and shape[-1] == 19 for shape in saved_forward_shapes
    )


def test_all_ignored_labels_produce_differentiable_zero_in_both_loss_paths():
    torch.manual_seed(419)
    model = SaddleForCausalLM(_tiny_config(loss_chunk_size=2)).double().train()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)

    chunked_output = model(input_ids=input_ids, labels=labels)
    assert chunked_output.logits is None
    assert torch.equal(chunked_output.loss, torch.zeros_like(chunked_output.loss))
    assert torch.isfinite(chunked_output.loss)
    chunked_output.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) == 0

    model.zero_grad(set_to_none=True)
    full_output = model(
        input_ids=input_ids,
        labels=labels,
        return_logits=True,
    )
    assert full_output.logits.shape == (1, 4, model.config.vocab_size)
    assert torch.equal(full_output.loss, torch.zeros_like(full_output.loss))
    full_output.loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.count_nonzero(model.lm_head.weight.grad) == 0


def test_logits_return_contract_is_explicit_and_eval_remains_compatible():
    torch.manual_seed(421)
    model = SaddleForCausalLM(_tiny_config(loss_chunk_size=2))
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    model.train()
    training_tuple = model(
        input_ids=input_ids,
        labels=input_ids,
        return_dict=False,
    )
    assert len(training_tuple) == 2
    assert training_tuple[0].ndim == 0
    assert training_tuple[1] is None

    explicit_logits = model(
        input_ids=input_ids,
        labels=input_ids,
        return_logits=True,
    )
    assert explicit_logits.logits.shape == (1, 4, model.config.vocab_size)

    # Evaluation, generation and distillation keep the legacy default: logits
    # are returned unless a caller explicitly opts out while supplying labels.
    model.eval()
    evaluation_output = model(input_ids=input_ids, labels=input_ids)
    inference_output = model(input_ids=input_ids)
    assert evaluation_output.logits.shape == (1, 4, model.config.vocab_size)
    assert inference_output.logits.shape == (1, 4, model.config.vocab_size)
    with pytest.raises(ValueError, match="return_logits=False requires labels"):
        model(input_ids=input_ids, return_logits=False)


def test_chunked_fast_path_also_avoids_full_mtp_logits():
    torch.manual_seed(431)
    model = SaddleForCausalLM(
        _tiny_config(loss_chunk_size=2, mtp_extra_tokens=2)
    ).double().train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits is None
    assert set(output.loss_items) == {
        "lm_loss",
        "aux_loss",
        "mtp_loss_2",
        "mtp_loss_3",
    }
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(head.weight.grad is not None for head in model.mtp_heads)


def test_native_trainer_uses_chunked_loss_for_loss_only_evaluation(tmp_path):
    from transformers import TrainingArguments

    from saddlellm.training.NativeTrainer import SaddleTrainer

    model = SaddleForCausalLM(_tiny_config(loss_chunk_size=2)).double().eval()
    trainer = SaddleTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            prediction_loss_only=True,
        ),
    )
    assert trainer.model_accepts_loss_kwargs is False
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    loss, outputs = trainer.compute_loss(
        model,
        {"input_ids": input_ids, "labels": input_ids.clone()},
        return_outputs=True,
    )

    assert torch.isfinite(loss)
    assert outputs.logits is None

    wrapped = SimpleNamespace(_orig_mod=SimpleNamespace(module=model))
    assert SaddleTrainer._native_model_config(wrapped) is model.config


def test_native_trainer_keeps_logits_for_external_loss_hooks(tmp_path):
    from transformers import TrainingArguments

    from saddlellm.training.NativeTrainer import SaddleTrainer

    model = SaddleForCausalLM(_tiny_config(loss_chunk_size=2)).double().train()
    observed = {}

    def external_loss(outputs, labels, num_items_in_batch=None):
        del num_items_in_batch
        observed["logits"] = outputs.logits
        shifted_labels = torch.full_like(labels, -100)
        shifted_labels[:, :-1] = labels[:, 1:]
        return F.cross_entropy(
            outputs.logits.reshape(-1, outputs.logits.size(-1)),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )

    trainer = SaddleTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
        ),
        compute_loss_func=external_loss,
    )
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    loss = trainer.compute_loss(
        model,
        {"input_ids": input_ids, "labels": input_ids.clone()},
    )

    assert observed["logits"].shape == (1, 4, model.config.vocab_size)
    assert torch.isfinite(loss)
