"""Native inference, rollout scoring, and lightweight planning for world models."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from ._WorldModel import RSSMState, WorldModelImagination
from .WorldModelBackends import load_world_model


@dataclass
class WorldModelPlannerConfig:
    """Cross-entropy/random-shooting planner settings."""

    horizon: int = 12
    num_candidates: int = 256
    iterations: int = 4
    elite_fraction: float = 0.1
    discount: float = 0.99
    continuation_aware: bool = True
    action_low: Union[float, Sequence[float]] = -1.0
    action_high: Union[float, Sequence[float]] = 1.0
    minimum_std: float = 0.05
    momentum: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("planner.horizon must be positive")
        if self.num_candidates < 2:
            raise ValueError("planner.num_candidates must be at least 2")
        if self.iterations <= 0:
            raise ValueError("planner.iterations must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("planner.elite_fraction must be in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("planner.discount must be in [0, 1]")
        if self.minimum_std < 0:
            raise ValueError("planner.minimum_std cannot be negative")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("planner.momentum must be in [0, 1)")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WorldModelPlannerConfig":
        values = dict(data or {})
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError("Unknown world-model planner fields: " + ", ".join(unknown))
        return cls(**values)


@dataclass
class WorldModelActionEvaluation:
    """Predicted returns and trajectories for candidate action sequences."""

    scores: torch.Tensor
    imagination: WorldModelImagination


@dataclass
class WorldModelPlan:
    """Best action sequence found by :class:`WorldModelRuntime`."""

    actions: torch.Tensor
    score: torch.Tensor
    predicted_observations: torch.Tensor
    predicted_rewards: torch.Tensor
    predicted_continuation: torch.Tensor

    @property
    def action(self) -> torch.Tensor:
        return self.actions[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.detach().cpu().tolist(),
            "actions": self.actions.detach().cpu().tolist(),
            "score": float(self.score.detach().cpu()),
            "predicted_observations": self.predicted_observations.detach().cpu().tolist(),
            "predicted_rewards": self.predicted_rewards.detach().cpu().tolist(),
            "predicted_continuation": self.predicted_continuation.detach().cpu().tolist(),
        }


class WorldModelRuntime:
    """Unified inference facade for every SaddleLLM-native world model."""

    def __init__(self, model: torch.nn.Module, device: str = "auto") -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, os.PathLike],
        device: str = "auto",
    ) -> "WorldModelRuntime":
        model = load_world_model(os.fspath(path), map_location="cpu")
        return cls(model, device=device)

    @property
    def backend(self) -> str:
        return getattr(self.model, "backend_name", "rssm")

    @torch.no_grad()
    def encode_observation(
        self,
        observation: Any,
        deterministic: bool = True,
    ) -> RSSMState:
        """Create a posterior belief from a current observation."""

        value = self._observation_tensor(observation)
        return self.model.initial_state_from_observation(
            value,
            sample_state=not deterministic,
        )

    @torch.no_grad()
    def filter(
        self,
        observations: Any,
        actions: Any,
        deterministic: bool = True,
    ) -> Any:
        """Filter an observed history and return the model's posterior output."""

        observation_tensor = self._observation_tensor(observations)
        observation_rank = len(self.model.config.observation_shape)
        if observation_tensor.ndim == observation_rank + 1:
            observation_tensor = observation_tensor.unsqueeze(0)
            unbatched = True
        elif observation_tensor.ndim == observation_rank + 2:
            unbatched = False
        else:
            raise ValueError(
                "observations must be [time, *observation_shape] or "
                "[batch, time, *observation_shape]"
            )
        action_tensor = torch.as_tensor(actions, device=self.device)
        if unbatched:
            action_tensor = action_tensor.unsqueeze(0)
        return self.model(
            observation_tensor,
            action_tensor,
            sample_state=not deterministic,
        )

    @torch.no_grad()
    def rollout(
        self,
        state: RSSMState,
        actions: Any,
        deterministic: bool = True,
    ) -> WorldModelImagination:
        """Predict observations, rewards, and continuation without new observations."""

        state = state.to(self.device)
        action_tensor = self._rollout_action_tensor(actions, state.deterministic.shape[0])
        return self.model.imagine(
            state,
            action_tensor,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def predict(
        self,
        observations: Any,
        history_actions: Any,
        future_actions: Any,
        deterministic: bool = True,
    ) -> WorldModelImagination:
        """Filter a history and immediately roll it forward under future actions."""

        posterior = self.filter(
            observations,
            history_actions,
            deterministic=deterministic,
        )
        return self.rollout(
            posterior.final_state,
            future_actions,
            deterministic=deterministic,
        )

    @torch.no_grad()
    def score_action_sequences(
        self,
        state: RSSMState,
        candidate_actions: Any,
        discount: float = 0.99,
        continuation_aware: bool = True,
    ) -> WorldModelActionEvaluation:
        """Score candidates by discounted predicted reward in latent imagination."""

        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        candidates = self._candidate_action_tensor(candidate_actions)
        candidate_count = candidates.shape[0]
        expanded_state = self._expand_state(state.to(self.device), candidate_count)
        imagination = self.model.imagine(
            expanded_state,
            candidates,
            deterministic=True,
        )
        horizon = imagination.rewards.shape[1]
        discounts = torch.pow(
            imagination.rewards.new_tensor(discount),
            torch.arange(horizon, device=self.device),
        ).unsqueeze(0)
        weights = discounts
        if continuation_aware:
            survival = torch.cumprod(
                torch.cat(
                    [
                        torch.ones(
                            candidate_count,
                            1,
                            device=self.device,
                            dtype=imagination.continuation.dtype,
                        ),
                        imagination.continuation[:, :-1],
                    ],
                    dim=1,
                ),
                dim=1,
            )
            weights = weights * survival
        scores = (imagination.rewards * weights).sum(dim=1)
        return WorldModelActionEvaluation(scores=scores, imagination=imagination)

    @torch.no_grad()
    def plan(
        self,
        state: RSSMState,
        config: Optional[WorldModelPlannerConfig] = None,
    ) -> WorldModelPlan:
        """Choose an action sequence using native CEM/random-shooting planning."""

        config = config or WorldModelPlannerConfig()
        state = state.to(self.device)
        if state.deterministic.shape[0] != 1:
            raise ValueError("plan() currently expects exactly one posterior belief")
        generator = torch.Generator().manual_seed(config.seed)
        if self.model.config.action_type == "discrete":
            return self._plan_discrete(state, config, generator)
        return self._plan_continuous(state, config, generator)

    def _plan_continuous(
        self,
        state: RSSMState,
        config: WorldModelPlannerConfig,
        generator: torch.Generator,
    ) -> WorldModelPlan:
        action_dim = self.model.config.action_dim
        low = _action_bound(config.action_low, action_dim, "action_low")
        high = _action_bound(config.action_high, action_dim, "action_high")
        if torch.any(low >= high):
            raise ValueError("Every action_low value must be smaller than action_high")
        mean = ((low + high) / 2.0).repeat(config.horizon, 1)
        std = ((high - low) / 2.0).repeat(config.horizon, 1)
        best = None
        elite_count = max(1, int(config.num_candidates * config.elite_fraction))
        for _ in range(config.iterations):
            noise = torch.randn(
                config.num_candidates,
                config.horizon,
                action_dim,
                generator=generator,
            )
            candidates_cpu = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            candidates_cpu = torch.maximum(
                torch.minimum(candidates_cpu, high), low
            )
            candidates = candidates_cpu.to(self.device)
            evaluation = self.score_action_sequences(
                state,
                candidates,
                discount=config.discount,
                continuation_aware=config.continuation_aware,
            )
            best = self._keep_best(best, candidates, evaluation)
            elite_indices = torch.topk(evaluation.scores, elite_count).indices.cpu()
            elites = candidates_cpu[elite_indices]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
            mean = config.momentum * mean + (1.0 - config.momentum) * elite_mean
            std = config.momentum * std + (1.0 - config.momentum) * elite_std
        return self._build_plan(best)

    def _plan_discrete(
        self,
        state: RSSMState,
        config: WorldModelPlannerConfig,
        generator: torch.Generator,
    ) -> WorldModelPlan:
        action_dim = self.model.config.action_dim
        probabilities = torch.full(
            (config.horizon, action_dim), 1.0 / action_dim
        )
        best = None
        elite_count = max(1, int(config.num_candidates * config.elite_fraction))
        for _ in range(config.iterations):
            candidates_cpu = torch.multinomial(
                probabilities,
                config.num_candidates,
                replacement=True,
                generator=generator,
            ).transpose(0, 1)
            candidates = candidates_cpu.to(self.device)
            evaluation = self.score_action_sequences(
                state,
                candidates,
                discount=config.discount,
                continuation_aware=config.continuation_aware,
            )
            best = self._keep_best(best, candidates, evaluation)
            elite_indices = torch.topk(evaluation.scores, elite_count).indices.cpu()
            elite_actions = candidates_cpu[elite_indices]
            frequencies = F.one_hot(elite_actions, action_dim).float().mean(dim=0)
            probabilities = (
                config.momentum * probabilities
                + (1.0 - config.momentum) * frequencies
            )
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return self._build_plan(best)

    @staticmethod
    def _keep_best(
        current: Optional[Tuple[torch.Tensor, ...]],
        candidates: torch.Tensor,
        evaluation: WorldModelActionEvaluation,
    ) -> Tuple[torch.Tensor, ...]:
        index = int(evaluation.scores.argmax())
        score = evaluation.scores[index]
        if current is not None and float(score) <= float(current[0]):
            return current
        imagination = evaluation.imagination
        return (
            score.detach().clone(),
            candidates[index].detach().clone(),
            imagination.observations[index].detach().clone(),
            imagination.rewards[index].detach().clone(),
            imagination.continuation[index].detach().clone(),
        )

    @staticmethod
    def _build_plan(best: Optional[Tuple[torch.Tensor, ...]]) -> WorldModelPlan:
        if best is None:
            raise RuntimeError("Planner did not evaluate any action candidates")
        return WorldModelPlan(
            score=best[0],
            actions=best[1],
            predicted_observations=best[2],
            predicted_rewards=best[3],
            predicted_continuation=best[4],
        )

    def _observation_tensor(self, value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        tensor = tensor.to(dtype=next(self.model.parameters()).dtype)
        if self.model.config.is_image and tensor.numel() and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _rollout_action_tensor(
        self,
        actions: Any,
        state_batch_size: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(actions, device=self.device)
        if self.model.config.action_type == "continuous":
            if tensor.ndim == 1 and self.model.config.action_dim == 1:
                tensor = tensor.reshape(1, -1, 1)
            elif tensor.ndim == 2:
                batched_scalar_actions = (
                    self.model.config.action_dim == 1
                    and tensor.shape[0] == state_batch_size
                    and (state_batch_size > 1 or tensor.shape[-1] != 1)
                )
                tensor = (
                    tensor.unsqueeze(-1)
                    if batched_scalar_actions
                    else tensor.unsqueeze(0)
                )
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif (
            tensor.ndim == 2
            and state_batch_size == 1
            and tensor.shape[-1] == self.model.config.action_dim
            and tensor.dtype.is_floating_point
        ):
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] != state_batch_size:
            raise ValueError(
                f"Action batch {tensor.shape[0]} does not match state batch {state_batch_size}"
            )
        return tensor

    def _candidate_action_tensor(self, actions: Any) -> torch.Tensor:
        tensor = torch.as_tensor(actions, device=self.device)
        if self.model.config.action_type == "continuous":
            if tensor.ndim == 2 and self.model.config.action_dim == 1:
                tensor = tensor.unsqueeze(-1)
            if tensor.ndim != 3:
                raise ValueError(
                    "Continuous candidates must have shape [candidates, horizon, action_dim]"
                )
        elif tensor.ndim != 2:
            raise ValueError(
                "Discrete candidates must have shape [candidates, horizon]"
            )
        return tensor

    @staticmethod
    def _expand_state(state: RSSMState, batch_size: int) -> RSSMState:
        current = state.deterministic.shape[0]
        if current == batch_size:
            return state
        if current != 1:
            raise ValueError(
                f"Cannot expand state batch {current} to candidate batch {batch_size}"
            )
        return RSSMState(
            state.deterministic.expand(batch_size, -1).contiguous(),
            state.stochastic.expand(batch_size, -1).contiguous(),
            (
                state.context.expand(batch_size, *state.context.shape[1:]).contiguous()
                if state.context is not None
                else None
            ),
        )


def run_world_model_inference(
    checkpoint: Union[str, os.PathLike],
    request: Dict[str, Any],
    device: str = "auto",
) -> Dict[str, Any]:
    """Run JSON-compatible native rollout or planning against a checkpoint."""

    runtime = WorldModelRuntime.from_pretrained(checkpoint, device=device)
    deterministic = bool(request.get("deterministic", True))
    if "observations" in request:
        if "actions" not in request:
            raise ValueError("History inference requires request.actions")
        posterior = runtime.filter(
            request["observations"],
            request["actions"],
            deterministic=deterministic,
        )
        state = posterior.final_state
    elif "observation" in request:
        state = runtime.encode_observation(
            request["observation"], deterministic=deterministic
        )
    else:
        raise ValueError("Request must contain observation or observations/actions history")

    result: Dict[str, Any] = {
        "status": "completed",
        "backend": runtime.backend,
        "checkpoint": os.path.abspath(os.fspath(checkpoint)),
    }
    if "future_actions" in request:
        imagined = runtime.rollout(
            state,
            request["future_actions"],
            deterministic=deterministic,
        )
        result["rollout"] = {
            "observations": imagined.observations.detach().cpu().tolist(),
            "rewards": imagined.rewards.detach().cpu().tolist(),
            "continuation": imagined.continuation.detach().cpu().tolist(),
        }
    if "planner" in request:
        planner_config = WorldModelPlannerConfig.from_dict(request.get("planner"))
        result["plan"] = runtime.plan(state, planner_config).to_dict()
    if "rollout" not in result and "plan" not in result:
        raise ValueError("Request must contain future_actions or planner settings")
    return result


def _action_bound(
    value: Union[float, Sequence[float]],
    action_dim: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.repeat(action_dim)
    if tensor.numel() != action_dim:
        raise ValueError(f"planner.{name} must be scalar or contain action_dim values")
    return tensor.reshape(1, action_dim)


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested or "auto").lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device
