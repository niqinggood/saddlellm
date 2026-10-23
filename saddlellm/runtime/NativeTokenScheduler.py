"""Token-budget decisions independent of HTTP, CUDA execution and text delivery."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTokenWork:
    index: int
    stage: str
    tokens: int


class NativeTokenScheduler:
    """Decode first, then bounded prompt chunks in ready-queue order.

    This plans work, not a single flattened GPU batch. Existing equal-context
    decode cohorts still execute separately; mixed prefill/decode attention and
    paged KV are later engine work. The caller requeues unfinished prefills at
    the tail, so long prompts make round-robin progress within the token budget.
    """

    def __init__(self, *, max_sequences, max_step_tokens, prefill_chunk_size):
        for name, value in (("max_sequences", max_sequences), ("max_step_tokens", max_step_tokens),
                            ("prefill_chunk_size", prefill_chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_step_tokens < max_sequences:
            raise ValueError("max_step_tokens must cover max_sequences decode tokens")
        self.max_sequences = max_sequences
        self.max_step_tokens = max_step_tokens
        self.prefill_chunk_size = prefill_chunk_size

    def plan(self, sessions):
        result = []
        remaining = self.max_step_tokens
        for index, session in enumerate(sessions):
            if not session.prefill_remaining_tokens and len(result) < self.max_sequences:
                cost = int(session.generated_tokens > 0)
                result.append(ScheduledTokenWork(index, "decode", cost))
                remaining -= cost
        for index, session in enumerate(sessions):
            if len(result) == self.max_sequences or not remaining:
                break
            count = min(session.prefill_remaining_tokens, self.prefill_chunk_size, remaining)
            if count:
                result.append(ScheduledTokenWork(index, "prefill", count))
                remaining -= count
        return tuple(result)
