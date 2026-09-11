"""SaddleLLM native categorical recurrent state-space world model.

This lightweight PyTorch backend is implemented against SaddleLLM's own data,
training, serialization, and inference interfaces. It combines broadly used
latent-dynamics techniques: categorical stochastic states, uniform probability
mixing, separated dynamics/representation KL terms, signed-log targets, and
grouped recurrent dynamics.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._WorldModel import (
    ObservationDecoder,
    ObservationEncoder,
    RSSMState,
    WorldModelImagination,
)


def symlog(value: torch.Tensor) -> torch.Tensor:
    """Signed logarithm for scale-robust regression."""

    return torch.sign(value) * torch.log1p(torch.abs(value))


def symexp(value: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`, with a safe exponent range."""

    value = value.clamp(-30.0, 30.0)
    return torch.sign(value) * torch.expm1(torch.abs(value))


@dataclass
class CategoricalWorldModelConfig:
    """Network and loss settings for SaddleLLM's categorical RSSM backend."""

    observation_shape: Tuple[int, ...]
    action_dim: int
    action_type: str = "continuous"
    embedding_size: int = 256
    deterministic_size: int = 256
    stochastic_size: int = 16
    stochastic_classes: int = 16
    hidden_size: int = 256
    mlp_layers: int = 2
    recurrent_blocks: int = 8
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 256)
    activation: str = "silu"
    uniform_mix: float = 0.01
    image_output_activation: str = "sigmoid"
    symlog_observations: bool = True
    reward_bins: int = 255
    reward_bin_low: float = -20.0
    reward_bin_high: float = 20.0
    reconstruction_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 1.0
    dynamics_kl_weight: float = 1.0
    representation_kl_weight: float = 0.1
    free_nats: float = 1.0

    def __post_init__(self) -> None:
        self.observation_shape = tuple(int(value) for value in self.observation_shape)
        self.cnn_channels = tuple(int(value) for value in self.cnn_channels)
        self.action_type = str(self.action_type).lower()
        self.image_output_activation = str(self.image_output_activation).lower()
        self._validate()

    @property
    def is_image(self) -> bool:
        return len(self.observation_shape) == 3

    @property
    def categorical_size(self) -> int:
        return self.stochastic_size * self.stochastic_classes

    @property
    def feature_size(self) -> int:
        return self.deterministic_size + self.categorical_size

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["observation_shape"] = list(self.observation_shape)
        data["cnn_channels"] = list(self.cnn_channels)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoricalWorldModelConfig":
        values = dict(data)
        values.pop("backend", None)
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "Unknown categorical world-model config fields: " + ", ".join(unknown)
            )
        return cls(**values)

    def _validate(self) -> None:
        if not self.observation_shape or any(value <= 0 for value in self.observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_type not in {"continuous", "discrete"}:
            raise ValueError("action_type must be 'continuous' or 'discrete'")
        for name in (
            "embedding_size",
            "deterministic_size",
            "stochastic_size",
            "stochastic_classes",
            "hidden_size",
            "recurrent_blocks",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.deterministic_size % self.recurrent_blocks:
            raise ValueError("deterministic_size must be divisible by recurrent_blocks")
        if self.mlp_layers < 1:
            raise ValueError("mlp_layers must be positive")
        if not 0.0 <= self.uniform_mix < 1.0:
            raise ValueError("uniform_mix must be in [0, 1)")
        if self.reward_bins < 3:
            raise ValueError("reward_bins must be at least 3")
        if self.reward_bin_low >= self.reward_bin_high:
            raise ValueError("reward_bin_low must be smaller than reward_bin_high")
        if self.free_nats < 0:
            raise ValueError("free_nats cannot be negative")
        for name in (
            "reconstruction_weight",
            "reward_weight",
            "continuation_weight",
            "dynamics_kl_weight",
            "representation_kl_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class CategoricalWorldModelOutput:
    observations: torch.Tensor
    observation_model: torch.Tensor
    rewards: torch.Tensor
    reward_logits: torch.Tensor
    continue_logits: torch.Tensor
    prior_logits: torch.Tensor
    posterior_logits: torch.Tensor
    deterministic_states: torch.Tensor
    stochastic_states: torch.Tensor
    initial_state: RSSMState
    final_state: RSSMState


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized * self.weight


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported categorical world-model activation: {name}")


def _mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    layers: int,
    activation: str,
) -> nn.Sequential:
    modules = []
    width = input_size
    for _ in range(layers):
        modules.extend(
            [
                nn.Linear(width, hidden_size),
                _RMSNorm(hidden_size),
                _activation(activation),
            ]
        )
        width = hidden_size
    modules.append(nn.Linear(width, output_size))
    return nn.Sequential(*modules)


class _GroupedLinear(nn.Module):
    """Independent linear projections over equally sized recurrent blocks."""

    def __init__(self, groups: int, input_size: int, output_size: int) -> None:
        super().__init__()
        self.groups = groups
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(groups, input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(groups, output_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] != (self.groups, self.input_size):
            raise ValueError(
                f"GroupedLinear expected (..., {self.groups}, {self.input_size}), "
                f"got {tuple(value.shape)}"
            )
        return torch.einsum("bgi,gio->bgo", value, self.weight) + self.bias


class _TwoHotSymlogHead(nn.Module):
    def __init__(self, config: CategoricalWorldModelConfig) -> None:
        super().__init__()
        self.network = _mlp(
            config.feature_size,
            config.reward_bins,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )
        self.register_buffer(
            "bins",
            torch.linspace(config.reward_bin_low, config.reward_bin_high, config.reward_bins),
        )

    def forward(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(feature)
        transformed = (torch.softmax(logits, dim=-1) * self.bins).sum(dim=-1)
        return logits, symexp(transformed)

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        transformed = symlog(target).clamp(float(self.bins[0]), float(self.bins[-1]))
        scale = (len(self.bins) - 1) / float(self.bins[-1] - self.bins[0])
        position = (transformed - self.bins[0]) * scale
        lower = position.floor().long().clamp(0, len(self.bins) - 1)
        upper = (lower + 1).clamp(0, len(self.bins) - 1)
        upper_weight = position - lower.to(position.dtype)
        lower_weight = 1.0 - upper_weight
        log_probabilities = F.log_softmax(logits, dim=-1)
        lower_log_probability = torch.gather(
            log_probabilities, -1, lower.unsqueeze(-1)
        ).squeeze(-1)
        upper_log_probability = torch.gather(
            log_probabilities, -1, upper.unsqueeze(-1)
        ).squeeze(-1)
        return -(
            lower_weight * lower_log_probability + upper_weight * upper_log_probability
        )


class CategoricalRSSMCell(nn.Module):
    """Grouped recurrent dynamics with straight-through categorical latents."""

    def __init__(self, config: CategoricalWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        groups = config.recurrent_blocks
        group_width = config.deterministic_size // groups
        self.group_width = group_width
        self.deterministic_input = nn.Sequential(
            nn.Linear(config.deterministic_size, config.hidden_size),
            _RMSNorm(config.hidden_size),
            _activation(config.activation),
        )
        self.stochastic_input = nn.Sequential(
            nn.Linear(config.categorical_size, config.hidden_size),
            _RMSNorm(config.hidden_size),
            _activation(config.activation),
        )
        self.action_input = nn.Sequential(
            nn.Linear(config.action_dim, config.hidden_size),
            _RMSNorm(config.hidden_size),
            _activation(config.activation),
        )
        grouped_input_size = group_width + 3 * config.hidden_size
        self.dynamic_hidden = _GroupedLinear(groups, grouped_input_size, group_width)
        self.dynamic_norm = _RMSNorm(config.deterministic_size)
        self.gates = _GroupedLinear(groups, group_width, 3 * group_width)
        self.prior = _mlp(
            config.deterministic_size,
            config.categorical_size,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )
        self.posterior = _mlp(
            config.deterministic_size + config.embedding_size,
            config.categorical_size,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )

    def initial_posterior(
        self, embedding: torch.Tensor, sample: bool
    ) -> Tuple[RSSMState, torch.Tensor]:
        deterministic = torch.zeros(
            embedding.shape[0],
            self.config.deterministic_size,
            device=embedding.device,
            dtype=embedding.dtype,
        )
        logits = self._regularized_logits(
            self.posterior(torch.cat([deterministic, embedding], dim=-1))
        )
        stochastic = self._sample(logits, sample)
        return RSSMState(deterministic, stochastic), logits

    def step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        embedding: Optional[torch.Tensor],
        sample: bool,
    ) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]:
        deterministic = self._core(previous, action)
        prior_logits = self._regularized_logits(self.prior(deterministic))
        posterior_logits = (
            prior_logits
            if embedding is None
            else self._regularized_logits(
                self.posterior(torch.cat([deterministic, embedding], dim=-1))
            )
        )
        stochastic = self._sample(posterior_logits, sample)
        return RSSMState(deterministic, stochastic), prior_logits, posterior_logits

    def _core(self, previous: RSSMState, action: torch.Tensor) -> torch.Tensor:
        action_scale = action.detach().abs().clamp_min(1.0)
        normalized_action = action / action_scale
        shared = torch.cat(
            [
                self.deterministic_input(previous.deterministic),
                self.stochastic_input(previous.stochastic),
                self.action_input(normalized_action),
            ],
            dim=-1,
        )
        grouped_previous = previous.deterministic.reshape(
            previous.deterministic.shape[0], self.config.recurrent_blocks, self.group_width
        )
        repeated_shared = shared.unsqueeze(1).expand(
            -1, self.config.recurrent_blocks, -1
        )
        hidden = self.dynamic_hidden(torch.cat([grouped_previous, repeated_shared], dim=-1))
        hidden = hidden.reshape(hidden.shape[0], -1)
        hidden = _activation(self.config.activation)(self.dynamic_norm(hidden))
        reset, candidate, update = torch.chunk(
            self.gates(
                hidden.reshape(hidden.shape[0], self.config.recurrent_blocks, self.group_width)
            ),
            3,
            dim=-1,
        )
        candidate = torch.tanh(torch.sigmoid(reset) * candidate)
        update = torch.sigmoid(update - 1.0)
        deterministic = update * candidate + (1.0 - update) * grouped_previous
        return deterministic.reshape(deterministic.shape[0], -1)

    def _regularized_logits(self, flat_logits: torch.Tensor) -> torch.Tensor:
        logits = flat_logits.reshape(
            flat_logits.shape[0], self.config.stochastic_size, self.config.stochastic_classes
        )
        probabilities = torch.softmax(logits, dim=-1)
        if self.config.uniform_mix:
            probabilities = (
                (1.0 - self.config.uniform_mix) * probabilities
                + self.config.uniform_mix / self.config.stochastic_classes
            )
        return torch.log(probabilities.clamp_min(1e-8))

    def _sample(self, logits: torch.Tensor, sample: bool) -> torch.Tensor:
        probabilities = logits.exp()
        if sample:
            indices = torch.distributions.Categorical(probs=probabilities).sample()
        else:
            indices = probabilities.argmax(dim=-1)
        one_hot = F.one_hot(indices, self.config.stochastic_classes).to(probabilities.dtype)
        if sample:
            one_hot = one_hot + probabilities - probabilities.detach()
        return one_hot.flatten(start_dim=1)


class CategoricalWorldModel(nn.Module):
    """SaddleLLM categorical RSSM with split representation objectives."""

    backend_name = "categorical_rssm"
    CONFIG_NAME = "world_model_config.json"
    WEIGHTS_NAME = "world_model.pt"

    def __init__(self, config: CategoricalWorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.dynamics = CategoricalRSSMCell(config)
        self.decoder = ObservationDecoder(config, self.encoder.conv_channels)
        self.reward_head = _TwoHotSymlogHead(config)
        self.continue_head = _mlp(
            config.feature_size,
            1,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        sample_state: Optional[bool] = None,
    ) -> CategoricalWorldModelOutput:
        self._validate_observations(observations)
        prepared_actions = self._prepare_actions(actions)
        transitions = observations.shape[1] - 1
        if transitions < 1:
            raise ValueError("A categorical world-model batch needs at least one transition")
        if prepared_actions.shape[:2] != (observations.shape[0], transitions):
            raise ValueError("actions must align with observation transitions")
        sample = self.training if sample_state is None else bool(sample_state)
        encoder_input = observations
        if self.config.symlog_observations and not self.config.is_image:
            encoder_input = symlog(encoder_input)
        embeddings = self.encoder(
            encoder_input.reshape(-1, *self.config.observation_shape)
        ).reshape(observations.shape[0], observations.shape[1], -1)
        initial_state, _ = self.dynamics.initial_posterior(embeddings[:, 0], sample)
        state = initial_state

        decoded = []
        decoded_model = []
        rewards = []
        reward_logits = []
        continue_logits = []
        prior_logits = []
        posterior_logits = []
        deterministic_states = []
        stochastic_states = []
        for step in range(transitions):
            state, prior, posterior = self.dynamics.step(
                state, prepared_actions[:, step], embeddings[:, step + 1], sample
            )
            observation_model = self.decoder(state.feature)
            observation = (
                symexp(observation_model)
                if self.config.symlog_observations and not self.config.is_image
                else observation_model
            )
            reward_logit, reward = self.reward_head(state.feature)
            decoded.append(observation)
            decoded_model.append(observation_model)
            rewards.append(reward)
            reward_logits.append(reward_logit)
            continue_logits.append(self.continue_head(state.feature).squeeze(-1))
            prior_logits.append(prior)
            posterior_logits.append(posterior)
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
        return CategoricalWorldModelOutput(
            observations=torch.stack(decoded, dim=1),
            observation_model=torch.stack(decoded_model, dim=1),
            rewards=torch.stack(rewards, dim=1),
            reward_logits=torch.stack(reward_logits, dim=1),
            continue_logits=torch.stack(continue_logits, dim=1),
            prior_logits=torch.stack(prior_logits, dim=1),
            posterior_logits=torch.stack(posterior_logits, dim=1),
            deterministic_states=torch.stack(deterministic_states, dim=1),
            stochastic_states=torch.stack(stochastic_states, dim=1),
            initial_state=initial_state,
            final_state=state,
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        sample_state: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self(batch["observations"], batch["actions"], sample_state)
        mask = batch.get("mask", torch.ones_like(output.rewards)).to(output.rewards.dtype)
        targets = batch["observations"][:, 1 : output.rewards.shape[1] + 1]
        model_targets = (
            symlog(targets)
            if self.config.symlog_observations and not self.config.is_image
            else targets
        )
        reduce_dimensions = tuple(range(2, targets.ndim))
        reconstruction = F.mse_loss(
            output.observation_model, model_targets, reduction="none"
        ).mean(dim=reduce_dimensions)
        reconstruction_loss = self._masked_mean(reconstruction, mask)

        target_rewards = batch.get("rewards", torch.zeros_like(output.rewards))
        target_rewards = target_rewards.to(output.rewards.dtype).reshape_as(output.rewards)
        reward_loss = self._masked_mean(
            self.reward_head.loss(output.reward_logits, target_rewards), mask
        )
        target_dones = batch.get("dones", torch.zeros_like(output.rewards))
        target_dones = target_dones.to(output.rewards.dtype).reshape_as(output.rewards)
        continuation = F.binary_cross_entropy_with_logits(
            output.continue_logits, 1.0 - target_dones.clamp(0.0, 1.0), reduction="none"
        )
        continuation_loss = self._masked_mean(continuation, mask)

        dynamics_kl = self._categorical_kl(
            output.posterior_logits.detach(), output.prior_logits
        )
        representation_kl = self._categorical_kl(
            output.posterior_logits, output.prior_logits.detach()
        )
        if self.config.free_nats:
            dynamics_kl = dynamics_kl.clamp_min(self.config.free_nats)
            representation_kl = representation_kl.clamp_min(self.config.free_nats)
        dynamics_kl_loss = self._masked_mean(dynamics_kl, mask)
        representation_kl_loss = self._masked_mean(representation_kl, mask)
        kl_loss = dynamics_kl_loss + representation_kl_loss
        total = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.reward_weight * reward_loss
            + self.config.continuation_weight * continuation_loss
            + self.config.dynamics_kl_weight * dynamics_kl_loss
            + self.config.representation_kl_weight * representation_kl_loss
        )
        return {
            "loss": total,
            "reconstruction_loss": reconstruction_loss,
            "reward_loss": reward_loss,
            "continuation_loss": continuation_loss,
            "dynamics_kl_loss": dynamics_kl_loss,
            "representation_kl_loss": representation_kl_loss,
            "kl_loss": kl_loss,
            "valid_transitions": mask.sum(),
        }

    def initial_state_from_observation(
        self,
        observation: torch.Tensor,
        sample_state: bool = False,
    ) -> RSSMState:
        """Encode one observation per batch into an initial posterior state."""

        if observation.ndim == len(self.config.observation_shape):
            observation = observation.unsqueeze(0)
        expected_rank = len(self.config.observation_shape) + 1
        if observation.ndim != expected_rank:
            raise ValueError(
                f"observation must have rank {expected_rank}: "
                "[batch, *observation_shape]"
            )
        if tuple(observation.shape[1:]) != self.config.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.config.observation_shape}, got "
                f"{tuple(observation.shape[1:])}"
            )
        encoder_input = observation.to(dtype=self._model_dtype())
        if self.config.symlog_observations and not self.config.is_image:
            encoder_input = symlog(encoder_input)
        embedding = self.encoder(encoder_input)
        state, _ = self.dynamics.initial_posterior(
            embedding, sample=bool(sample_state)
        )
        return state

    def imagine(
        self,
        initial_state: RSSMState,
        actions: torch.Tensor,
        deterministic: bool = True,
    ) -> WorldModelImagination:
        prepared_actions = self._prepare_actions(actions)
        state = initial_state
        observations = []
        rewards = []
        continuation = []
        deterministic_states = []
        stochastic_states = []
        for step in range(prepared_actions.shape[1]):
            state, _, _ = self.dynamics.step(
                state, prepared_actions[:, step], embedding=None, sample=not deterministic
            )
            observation_model = self.decoder(state.feature)
            observation = (
                symexp(observation_model)
                if self.config.symlog_observations and not self.config.is_image
                else observation_model
            )
            _, reward = self.reward_head(state.feature)
            observations.append(observation)
            rewards.append(reward)
            continuation.append(torch.sigmoid(self.continue_head(state.feature).squeeze(-1)))
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
        if not observations:
            raise ValueError("imagine() requires at least one action")
        return WorldModelImagination(
            observations=torch.stack(observations, dim=1),
            rewards=torch.stack(rewards, dim=1),
            continuation=torch.stack(continuation, dim=1),
            deterministic_states=torch.stack(deterministic_states, dim=1),
            stochastic_states=torch.stack(stochastic_states, dim=1),
            final_state=state,
        )

    def save_pretrained(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        config = self.config.to_dict()
        config["backend"] = self.backend_name
        with open(os.path.join(path, self.CONFIG_NAME), "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        torch.save(self.state_dict(), os.path.join(path, self.WEIGHTS_NAME))

    @classmethod
    def from_pretrained(
        cls, path: str, map_location: Optional[Any] = "cpu"
    ) -> "CategoricalWorldModel":
        with open(os.path.join(path, cls.CONFIG_NAME), "r", encoding="utf-8") as file:
            config = CategoricalWorldModelConfig.from_dict(json.load(file))
        model = cls(config)
        try:
            state = torch.load(
                os.path.join(path, cls.WEIGHTS_NAME),
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            state = torch.load(os.path.join(path, cls.WEIGHTS_NAME), map_location=map_location)
        model.load_state_dict(state)
        return model

    def parameter_count(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.config.action_type == "discrete":
            if actions.ndim == 3 and actions.shape[-1] == self.config.action_dim:
                return actions.to(self._model_dtype())
            if actions.ndim == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            if actions.ndim != 2:
                raise ValueError("Discrete actions must have shape [batch, time]")
            indices = actions.long()
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= self.config.action_dim
            ):
                raise ValueError("Discrete action id is outside the configured action space")
            return F.one_hot(indices, self.config.action_dim).to(self._model_dtype())
        if actions.ndim == 2 and self.config.action_dim == 1:
            actions = actions.unsqueeze(-1)
        if actions.ndim < 3:
            raise ValueError("Continuous actions must have shape [batch, time, action_dim]")
        actions = actions.flatten(start_dim=2)
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"Expected action_dim={self.config.action_dim}, got {actions.shape[-1]}"
            )
        return actions.to(self._model_dtype())

    def _validate_observations(self, observations: torch.Tensor) -> None:
        expected_rank = len(self.config.observation_shape) + 2
        if observations.ndim != expected_rank:
            raise ValueError(f"observations must have rank {expected_rank}")
        if tuple(observations.shape[2:]) != self.config.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.config.observation_shape}, got "
                f"{tuple(observations.shape[2:])}"
            )

    def _model_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @staticmethod
    def _categorical_kl(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
        return (log_q.exp() * (log_q - log_p)).sum(dim=(-1, -2))

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (value * mask).sum() / mask.sum().clamp_min(1.0)
