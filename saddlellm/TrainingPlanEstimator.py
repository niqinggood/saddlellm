"""Small deterministic estimates for LLM training plans."""
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class TrainingPlanEstimate:
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_gpus: int
    max_seq_length: int
    max_steps: int
    epochs: Optional[int] = None
    effective_batch_size: int = 0
    tokens_per_step: int = 0
    planned_tokens: Optional[int] = None
    schedule_mode: str = "epochs"

    def to_dict(self) -> Dict:
        return asdict(self)


class TrainingPlanEstimator:
    @staticmethod
    def estimate(
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        num_gpus: int,
        max_seq_length: int,
        max_steps: int,
        epochs: Optional[int] = None,
    ) -> TrainingPlanEstimate:
        effective_batch = max(1, per_device_batch_size) * max(1, gradient_accumulation_steps) * max(1, num_gpus)
        tokens_per_step = effective_batch * max(1, max_seq_length)
        schedule_mode = "steps" if max_steps > 0 else "epochs"
        # An epoch-based token total cannot be calculated without knowing the
        # number (and token lengths) of training examples.  ``None`` is more
        # accurate than reporting a misleading zero-token training run.
        planned_tokens = tokens_per_step * max_steps if max_steps > 0 else None
        return TrainingPlanEstimate(
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_gpus=num_gpus,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            epochs=epochs,
            effective_batch_size=effective_batch,
            tokens_per_step=tokens_per_step,
            planned_tokens=planned_tokens,
            schedule_mode=schedule_mode,
        )


def estimate_training_plan(
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int,
    max_seq_length: int,
    max_steps: int,
    epochs: Optional[int] = None,
) -> Dict:
    return TrainingPlanEstimator.estimate(
        per_device_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_gpus=num_gpus,
        max_seq_length=max_seq_length,
        max_steps=max_steps,
        epochs=epochs,
    ).to_dict()
