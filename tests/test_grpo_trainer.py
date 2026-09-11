import pytest
import torch

from saddlellm.alignment.GRPOTrainer import GRPOConfig, GRPOTrainer


def test_config_validates_sequence_budget_and_optimization_values():
    with pytest.raises(ValueError, match="max_sequence_length"):
        GRPOConfig(
            max_prompt_length=100,
            max_new_tokens=20,
            max_sequence_length=119,
        ).validate()
    with pytest.raises(ValueError, match="learning_rate"):
        GRPOConfig(learning_rate=0).validate()
    with pytest.raises(ValueError, match="prompt_sampling_strategy"):
        GRPOConfig(prompt_sampling_strategy="with_replacement").validate()
    with pytest.raises(ValueError, match="dynamic_sampling_max_attempts"):
        GRPOConfig(dynamic_sampling_max_attempts=0).validate()


def test_partial_gradient_accumulation_is_rescaled_to_actual_mean():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(gradient_accumulation_steps=4)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    parameter = next(trainer.model.parameters())
    # Two micro-loss gradients have each already been divided by target=4.
    parameter.grad = torch.tensor([[0.5]])
    trainer._correct_partial_accumulation(accumulated_steps=2)
    assert parameter.grad.item() == pytest.approx(1.0)


def test_shuffle_cycle_visits_every_prompt_before_repeating():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(prompt_sampling_strategy="shuffle_cycle", seed=42)
    trainer._rng = __import__("random").Random(42)
    trainer._prompt_cycle_key = ()
    trainer._prompt_cycle_indices = []
    prompts = ["a", "b", "c", "d"]

    first_cycle = [trainer._sample_prompts(prompts, 1)[0] for _ in prompts]
    assert set(first_cycle) == set(prompts)
    assert trainer._sample_prompts(prompts, 1)[0] in prompts


def test_dynamic_sampling_discards_flat_group_and_uses_replacement_prompt():
    trainer = GRPOTrainer.__new__(GRPOTrainer)
    trainer.config = GRPOConfig(
        num_generations=2,
        per_device_batch_size=1,
        dynamic_sampling=True,
        dynamic_sampling_max_attempts=2,
    )
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    parameter = next(trainer.model.parameters())

    class FakeOptimizer:
        param_groups = [{"lr": 0.1}]

        @staticmethod
        def zero_grad(set_to_none=True):
            del set_to_none
            parameter.grad = None

        @staticmethod
        def step():
            if parameter.grad is not None:
                with torch.no_grad():
                    parameter.sub_(0.1 * parameter.grad)

    trainer.optimizer = FakeOptimizer()
    trainer.device = torch.device("cpu")
    trainer._adapter_reference = False
    trainer._step = 0
    trainer._optimizer_step = 0
    trainer._metrics_history = []

    selections = iter([["flat"], ["mixed"]])
    trainer._sample_prompts = lambda prompts, count: next(selections)

    def fake_rollout(prompts):
        group_prompt = prompts[0]
        responses = ["wrong", "wrong"] if group_prompt == "flat" else ["right", "wrong"]
        placeholders = [None] * len(responses)
        return responses, placeholders, placeholders, placeholders, placeholders, 0

    trainer._generate_responses_with_log_probs = fake_rollout
    trainer._compute_grpo_loss = lambda **kwargs: (
        trainer.model.weight.sum(),
        0.0,
    )

    metrics = trainer.train(
        ["flat", "mixed"],
        lambda prompt, response: float(response == "right"),
        num_steps=1,
    )

    assert metrics[-1]["sampled_groups"] == 2
    assert metrics[-1]["accepted_groups"] == 1
    assert metrics[-1]["discarded_groups"] == 1
    assert metrics[-1]["sampling_attempts"] == 2
    assert metrics[-1]["optimizer_step"] == 1
