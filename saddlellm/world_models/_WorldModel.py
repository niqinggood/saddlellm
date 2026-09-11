"""Internal implementation of SaddleLLM's Gaussian recurrent state-space model.

The module is intentionally independent from a particular simulator or robot
stack.  It learns a compact latent dynamics model from offline trajectories:

    observation_t --encoder--> posterior state_t
    state_t + action_t --RSSM--> prior state_{t+1}
    state_{t+1} --> next observation, reward, continuation

Vector observations use MLP encoders/decoders.  Channel-first image
observations use convolutional encoders/decoders and are expected in ``[0, 1]``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields, replace
from functools import reduce
from operator import mul
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelConfig:
    """Network and objective configuration for :class:`WorldModel`.

    ``observation_shape`` excludes batch/time dimensions.  Images must be CHW,
    for example ``(3, 64, 64)``.  For discrete actions, ``action_dim`` is the
    number of actions; for continuous actions it is the vector width.
    """

    observation_shape: Tuple[int, ...]
    action_dim: int
    action_type: str = "continuous"  # continuous | discrete
    embedding_size: int = 256
    deterministic_size: int = 256
    stochastic_size: int = 32
    hidden_size: int = 256
    mlp_layers: int = 2
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 256)
    activation: str = "elu"
    min_std: float = 0.1
    image_output_activation: str = "sigmoid"  # sigmoid | none
    reconstruction_weight: float = 1.0
    reward_weight: float = 1.0
    continuation_weight: float = 1.0
    kl_weight: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 1.0
    # Optional spatial auxiliary heads.  They are disabled by default so all
    # existing vector/image checkpoints retain the exact same network.
    spatial_occupancy_channels: int = 0
    spatial_occupancy_weight: float = 0.0
    spatial_occupancy_class_weights: Tuple[float, ...] = ()
    spatial_occupancy_kinematic_prior: bool = False
    spatial_occupancy_residual_scale: float = 3.0
    spatial_occupancy_residual_unknown_only: bool = True
    spatial_ego_motion_dim: int = 0
    spatial_ego_motion_weight: float = 0.0
    spatial_collision_weight: float = 0.0
    spatial_collision_positive_weight: float = 1.0

    def __post_init__(self) -> None:
        self.observation_shape = tuple(int(value) for value in self.observation_shape)
        self.cnn_channels = tuple(int(value) for value in self.cnn_channels)
        self.spatial_occupancy_class_weights = tuple(
            float(value) for value in self.spatial_occupancy_class_weights
        )
        self.action_type = str(self.action_type).lower()
        self.image_output_activation = str(self.image_output_activation).lower()
        self._validate()

    @property
    def is_image(self) -> bool:
        return len(self.observation_shape) == 3

    @property
    def feature_size(self) -> int:
        return self.deterministic_size + self.stochastic_size

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["observation_shape"] = list(self.observation_shape)
        data["cnn_channels"] = list(self.cnn_channels)
        data["spatial_occupancy_class_weights"] = list(
            self.spatial_occupancy_class_weights
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModelConfig":
        if not isinstance(data, dict):
            raise TypeError("WorldModelConfig.from_dict() expects a dictionary")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("Unknown world-model config fields: " + ", ".join(unknown))
        return cls(**dict(data))

    def _validate(self) -> None:
        if not self.observation_shape or any(size <= 0 for size in self.observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_type not in {"continuous", "discrete"}:
            raise ValueError("action_type must be 'continuous' or 'discrete'")
        for name in ("embedding_size", "deterministic_size", "stochastic_size", "hidden_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mlp_layers < 1:
            raise ValueError("mlp_layers must be at least 1")
        if self.is_image and not self.cnn_channels:
            raise ValueError("cnn_channels cannot be empty for image observations")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")
        if not 0.0 <= self.kl_balance <= 1.0:
            raise ValueError("kl_balance must be in [0, 1]")
        if self.free_nats < 0:
            raise ValueError("free_nats cannot be negative")
        if self.image_output_activation not in {"sigmoid", "none"}:
            raise ValueError("image_output_activation must be 'sigmoid' or 'none'")
        if self.spatial_occupancy_channels < 0:
            raise ValueError("spatial_occupancy_channels cannot be negative")
        if self.spatial_occupancy_residual_scale < 0:
            raise ValueError("spatial_occupancy_residual_scale cannot be negative")
        if self.spatial_ego_motion_dim < 0:
            raise ValueError("spatial_ego_motion_dim cannot be negative")
        for name in (
            "spatial_occupancy_weight",
            "spatial_ego_motion_weight",
            "spatial_collision_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.spatial_occupancy_channels:
            if not self.is_image:
                raise ValueError("spatial occupancy prediction requires CHW observations")
            if self.spatial_occupancy_channels > self.observation_shape[0]:
                raise ValueError(
                    "spatial_occupancy_channels cannot exceed observation channels"
                )
        if self.spatial_occupancy_weight and not self.spatial_occupancy_channels:
            raise ValueError(
                "spatial_occupancy_weight requires spatial_occupancy_channels"
            )
        if self.spatial_occupancy_class_weights:
            if len(self.spatial_occupancy_class_weights) != self.spatial_occupancy_channels:
                raise ValueError(
                    "spatial_occupancy_class_weights must match occupancy channels"
                )
            if any(value <= 0 for value in self.spatial_occupancy_class_weights):
                raise ValueError("spatial occupancy class weights must be positive")
        if self.spatial_ego_motion_weight and not self.spatial_ego_motion_dim:
            raise ValueError(
                "spatial_ego_motion_weight requires spatial_ego_motion_dim"
            )
        if self.spatial_collision_positive_weight <= 0:
            raise ValueError("spatial_collision_positive_weight must be positive")


@dataclass
class RSSMState:
    """One recurrent state: deterministic memory plus stochastic latent."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    context: Optional[torch.Tensor] = None

    @property
    def feature(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self.stochastic], dim=-1)

    def detach(self) -> "RSSMState":
        return RSSMState(
            self.deterministic.detach(),
            self.stochastic.detach(),
            self.context.detach() if self.context is not None else None,
        )

    def to(self, *args: Any, **kwargs: Any) -> "RSSMState":
        return RSSMState(
            self.deterministic.to(*args, **kwargs),
            self.stochastic.to(*args, **kwargs),
            self.context.to(*args, **kwargs) if self.context is not None else None,
        )


@dataclass
class WorldModelOutput:
    """Outputs for posterior filtering over an observed trajectory."""

    observations: torch.Tensor
    rewards: torch.Tensor
    continue_logits: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_std: torch.Tensor
    deterministic_states: torch.Tensor
    stochastic_states: torch.Tensor
    initial_state: RSSMState
    final_state: RSSMState
    occupancy_logits: Optional[torch.Tensor] = None
    ego_motion: Optional[torch.Tensor] = None
    collision_logits: Optional[torch.Tensor] = None


@dataclass
class WorldModelImagination:
    """Open-loop predictions produced without future observations."""

    observations: torch.Tensor
    rewards: torch.Tensor
    continuation: torch.Tensor
    deterministic_states: torch.Tensor
    stochastic_states: torch.Tensor
    final_state: RSSMState
    occupancy_logits: Optional[torch.Tensor] = None
    ego_motion: Optional[torch.Tensor] = None
    collision_probability: Optional[torch.Tensor] = None


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


class _MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        layers = []
        current_size = input_size
        for _ in range(max(1, num_layers)):
            layers.extend([nn.Linear(current_size, hidden_size), _activation(activation)])
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ObservationEncoder(nn.Module):
    """Encode vector or CHW image observations into a fixed-width embedding."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.convolutional = config.is_image
        if not self.convolutional:
            input_size = int(reduce(mul, config.observation_shape, 1))
            self.encoder = _MLP(
                input_size,
                config.embedding_size,
                config.hidden_size,
                config.mlp_layers,
                config.activation,
            )
            self.conv_channels: Tuple[int, ...] = ()
            return

        input_channels, height, width = config.observation_shape
        usable_channels = []
        spatial_size = min(height, width)
        for channel in config.cnn_channels:
            if spatial_size < 2:
                break
            usable_channels.append(channel)
            spatial_size = max(1, spatial_size // 2)
        self.conv_channels = tuple(usable_channels)

        layers = []
        current_channels = input_channels
        for channel in self.conv_channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, channel, kernel_size=4, stride=2, padding=1),
                    _activation(config.activation),
                ]
            )
            current_channels = channel
        self.convolutions = nn.Sequential(*layers) if layers else nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(1, *config.observation_shape)
            flat_size = int(self.convolutions(dummy).numel())
        self.projection = nn.Linear(flat_size, config.embedding_size)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if not self.convolutional:
            return self.encoder(observation.flatten(start_dim=1))
        features = self.convolutions(observation)
        return self.projection(features.flatten(start_dim=1))


class ObservationDecoder(nn.Module):
    """Decode an RSSM feature to the configured observation shape."""

    def __init__(self, config: WorldModelConfig, conv_channels: Sequence[int]) -> None:
        super().__init__()
        self.config = config
        self.convolutional = config.is_image
        if not self.convolutional:
            output_size = int(reduce(mul, config.observation_shape, 1))
            self.decoder = _MLP(
                config.feature_size,
                output_size,
                config.hidden_size,
                config.mlp_layers,
                config.activation,
            )
            return

        output_channels, height, width = config.observation_shape
        conv_channels = tuple(conv_channels)
        if not conv_channels:
            output_size = int(reduce(mul, config.observation_shape, 1))
            self.direct_decoder = _MLP(
                config.feature_size,
                output_size,
                config.hidden_size,
                config.mlp_layers,
                config.activation,
            )
            self.base_shape = None
            return

        stages = len(conv_channels)
        scale = 2 ** stages
        base_height = max(1, math.ceil(height / scale))
        base_width = max(1, math.ceil(width / scale))
        base_channels = conv_channels[-1]
        self.base_shape = (base_channels, base_height, base_width)
        self.projection = nn.Linear(
            config.feature_size,
            base_channels * base_height * base_width,
        )

        layers = []
        current_channels = base_channels
        target_channels = list(reversed(conv_channels[:-1])) + [output_channels]
        for index, channel in enumerate(target_channels):
            layers.append(
                nn.ConvTranspose2d(
                    current_channels,
                    channel,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if index < len(target_channels) - 1:
                layers.append(_activation(config.activation))
            current_channels = channel
        self.deconvolutions = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.convolutional:
            value = self.decoder(feature)
            return value.reshape(feature.shape[0], *self.config.observation_shape)
        if self.base_shape is None:
            value = self.direct_decoder(feature)
            value = value.reshape(feature.shape[0], *self.config.observation_shape)
        else:
            value = self.projection(feature).reshape(feature.shape[0], *self.base_shape)
            value = self.deconvolutions(value)
            target_size = self.config.observation_shape[-2:]
            if value.shape[-2:] != target_size:
                value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
        if self.config.image_output_activation == "sigmoid":
            value = torch.sigmoid(value)
        return value


class RSSMCell(nn.Module):
    """Recurrent state-space dynamics with diagonal Gaussian latents."""

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.recurrent_input = nn.Sequential(
            nn.Linear(config.stochastic_size + config.action_dim, config.hidden_size),
            _activation(config.activation),
        )
        self.recurrent = nn.GRUCell(config.hidden_size, config.deterministic_size)
        self.prior = _MLP(
            config.deterministic_size,
            2 * config.stochastic_size,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )
        self.posterior = _MLP(
            config.deterministic_size + config.embedding_size,
            2 * config.stochastic_size,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )

    def zero_deterministic(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.deterministic_size,
            device=device,
            dtype=dtype,
        )

    def initial_posterior(
        self,
        embedding: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[RSSMState, torch.Tensor, torch.Tensor]:
        deterministic = self.zero_deterministic(
            embedding.shape[0], embedding.device, embedding.dtype
        )
        mean, std = self._distribution(self.posterior(torch.cat([deterministic, embedding], dim=-1)))
        stochastic = self._sample(mean, std, sample)
        return RSSMState(deterministic, stochastic), mean, std

    def step(
        self,
        previous: RSSMState,
        action: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
        sample: bool = True,
    ) -> Tuple[RSSMState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recurrent_input = self.recurrent_input(
            torch.cat([previous.stochastic, action], dim=-1)
        )
        deterministic = self.recurrent(recurrent_input, previous.deterministic)
        prior_mean, prior_std = self._distribution(self.prior(deterministic))
        if embedding is None:
            posterior_mean, posterior_std = prior_mean, prior_std
        else:
            posterior_mean, posterior_std = self._distribution(
                self.posterior(torch.cat([deterministic, embedding], dim=-1))
            )
        stochastic = self._sample(posterior_mean, posterior_std, sample)
        state = RSSMState(deterministic, stochastic)
        return state, prior_mean, prior_std, posterior_mean, posterior_std

    def _distribution(self, parameters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = torch.chunk(parameters, 2, dim=-1)
        std = F.softplus(raw_std) + self.config.min_std
        return mean, std

    @staticmethod
    def _sample(mean: torch.Tensor, std: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mean
        return mean + std * torch.randn_like(mean)


class WorldModel(nn.Module):
    """Trainable RSSM world model for vector or image trajectories."""

    backend_name = "rssm"
    CONFIG_NAME = "world_model_config.json"
    WEIGHTS_NAME = "world_model.pt"

    def __init__(self, config: WorldModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.dynamics = RSSMCell(config)
        self.decoder = ObservationDecoder(config, self.encoder.conv_channels)
        self.reward_head = _MLP(
            config.feature_size,
            1,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )
        self.continue_head = _MLP(
            config.feature_size,
            1,
            config.hidden_size,
            config.mlp_layers,
            config.activation,
        )
        self.occupancy_head: Optional[ObservationDecoder] = None
        if config.spatial_occupancy_channels:
            occupancy_config = replace(
                config,
                observation_shape=(
                    config.spatial_occupancy_channels,
                    config.observation_shape[-2],
                    config.observation_shape[-1],
                ),
                image_output_activation="none",
                spatial_occupancy_channels=0,
                spatial_occupancy_weight=0.0,
                spatial_occupancy_class_weights=(),
                spatial_occupancy_kinematic_prior=False,
                spatial_ego_motion_dim=0,
                spatial_ego_motion_weight=0.0,
                spatial_collision_weight=0.0,
            )
            self.occupancy_head = ObservationDecoder(
                occupancy_config, self.encoder.conv_channels
            )
            if config.spatial_occupancy_kinematic_prior:
                self._zero_output_layer(self.occupancy_head)
        self.ego_motion_head: Optional[_MLP] = None
        if config.spatial_ego_motion_dim:
            self.ego_motion_head = _MLP(
                config.feature_size,
                config.spatial_ego_motion_dim,
                config.hidden_size,
                config.mlp_layers,
                config.activation,
            )
        self.collision_head: Optional[_MLP] = None
        if config.spatial_collision_weight:
            self.collision_head = _MLP(
                config.feature_size + config.action_dim,
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
    ) -> WorldModelOutput:
        """Filter a batch of trajectories and predict every next transition.

        Args:
            observations: ``[batch, transitions + 1, *observation_shape]``.
            actions: Continuous ``[batch, transitions, action_dim]`` or discrete
                integer ``[batch, transitions]`` actions.
            sample_state: Use stochastic reparameterized samples.  Defaults to
                ``True`` in training and ``False`` in evaluation.
        """

        self._validate_observations(observations)
        prepared_actions = self._prepare_actions(actions)
        transitions = observations.shape[1] - 1
        if transitions < 1:
            raise ValueError("A world-model batch needs at least one transition")
        if prepared_actions.shape[:2] != (observations.shape[0], transitions):
            raise ValueError(
                "actions must contain exactly one entry for every observation transition"
            )
        sample = self.training if sample_state is None else bool(sample_state)

        flat_observations = observations.reshape(-1, *self.config.observation_shape)
        embeddings = self.encoder(flat_observations).reshape(
            observations.shape[0], observations.shape[1], self.config.embedding_size
        )
        initial_state, _, _ = self.dynamics.initial_posterior(embeddings[:, 0], sample=sample)
        initial_state.context = observations[:, 0]
        state = initial_state

        predicted_observations = []
        predicted_rewards = []
        continue_logits = []
        prior_means = []
        prior_stds = []
        posterior_means = []
        posterior_stds = []
        deterministic_states = []
        stochastic_states = []
        occupancy_logits = []
        ego_motion = []
        collision_logits = []
        for step in range(transitions):
            current_context = state.context
            if self.collision_head is not None:
                collision_input = torch.cat(
                    [state.feature, prepared_actions[:, step]], dim=-1
                )
                collision_logits.append(
                    self.collision_head(collision_input).squeeze(-1)
                )
            state, prior_mean, prior_std, posterior_mean, posterior_std = self.dynamics.step(
                state,
                prepared_actions[:, step],
                embedding=embeddings[:, step + 1],
                sample=sample,
            )
            feature = state.feature
            prior_feature = torch.cat([state.deterministic, prior_mean], dim=-1)
            predicted_observations.append(self.decoder(feature))
            predicted_rewards.append(self.reward_head(feature).squeeze(-1))
            continue_logits.append(self.continue_head(feature).squeeze(-1))
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
            if self.occupancy_head is not None:
                occupancy_logits.append(
                    self._spatial_occupancy_logits(
                        prior_feature, current_context, prepared_actions[:, step]
                    )
                )
            if self.ego_motion_head is not None:
                ego_motion.append(self.ego_motion_head(prior_feature))
            state.context = observations[:, step + 1]

        return WorldModelOutput(
            observations=torch.stack(predicted_observations, dim=1),
            rewards=torch.stack(predicted_rewards, dim=1),
            continue_logits=torch.stack(continue_logits, dim=1),
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            posterior_mean=torch.stack(posterior_means, dim=1),
            posterior_std=torch.stack(posterior_stds, dim=1),
            deterministic_states=torch.stack(deterministic_states, dim=1),
            stochastic_states=torch.stack(stochastic_states, dim=1),
            initial_state=initial_state,
            final_state=state,
            occupancy_logits=(
                torch.stack(occupancy_logits, dim=1) if occupancy_logits else None
            ),
            ego_motion=torch.stack(ego_motion, dim=1) if ego_motion else None,
            collision_logits=(
                torch.stack(collision_logits, dim=1) if collision_logits else None
            ),
        )

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        sample_state: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction, reward, continuation and balanced KL losses."""

        required = {"observations", "actions"}
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError("World-model batch is missing: " + ", ".join(missing))
        observations = batch["observations"]
        output = self(observations, batch["actions"], sample_state=sample_state)
        transitions = output.rewards.shape[1]
        mask = batch.get("mask")
        if mask is None:
            mask = torch.ones_like(output.rewards)
        else:
            mask = mask.to(dtype=output.rewards.dtype)
            if mask.shape != output.rewards.shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} does not match transitions "
                    f"{tuple(output.rewards.shape)}"
                )

        target_observations = observations[:, 1 : transitions + 1]
        reduce_dimensions = tuple(range(2, target_observations.ndim))
        reconstruction_per_step = F.mse_loss(
            output.observations, target_observations, reduction="none"
        ).mean(dim=reduce_dimensions)
        reconstruction_loss = self._masked_mean(reconstruction_per_step, mask)

        target_rewards = batch.get("rewards")
        if target_rewards is None:
            target_rewards = torch.zeros_like(output.rewards)
        target_rewards = target_rewards.to(dtype=output.rewards.dtype).reshape_as(output.rewards)
        reward_per_step = F.mse_loss(output.rewards, target_rewards, reduction="none")
        reward_loss = self._masked_mean(reward_per_step, mask)

        target_dones = batch.get("dones")
        if target_dones is None:
            target_dones = torch.zeros_like(output.rewards)
        target_dones = target_dones.to(dtype=output.rewards.dtype).reshape_as(output.rewards)
        continue_targets = 1.0 - target_dones.clamp(0.0, 1.0)
        continuation_per_step = F.binary_cross_entropy_with_logits(
            output.continue_logits, continue_targets, reduction="none"
        )
        continuation_loss = self._masked_mean(continuation_per_step, mask)

        prior_kl = self._normal_kl(
            output.posterior_mean.detach(),
            output.posterior_std.detach(),
            output.prior_mean,
            output.prior_std,
        )
        posterior_kl = self._normal_kl(
            output.posterior_mean,
            output.posterior_std,
            output.prior_mean.detach(),
            output.prior_std.detach(),
        )
        kl_per_step = (
            self.config.kl_balance * prior_kl
            + (1.0 - self.config.kl_balance) * posterior_kl
        )
        if self.config.free_nats:
            kl_per_step = torch.clamp_min(kl_per_step, self.config.free_nats)
        kl_loss = self._masked_mean(kl_per_step, mask)

        total = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.reward_weight * reward_loss
            + self.config.continuation_weight * continuation_loss
            + self.config.kl_weight * kl_loss
        )
        losses: Dict[str, torch.Tensor] = {
            "loss": total,
            "reconstruction_loss": reconstruction_loss,
            "reward_loss": reward_loss,
            "continuation_loss": continuation_loss,
            "kl_loss": kl_loss,
            "valid_transitions": mask.sum(),
        }
        if output.occupancy_logits is not None:
            occupancy_channels = self.config.spatial_occupancy_channels
            occupancy_target = target_observations[:, :, :occupancy_channels].argmax(
                dim=2
            )
            logits = output.occupancy_logits
            pixel_loss = F.cross_entropy(
                logits.flatten(0, 1),
                occupancy_target.flatten(0, 1),
                weight=(
                    logits.new_tensor(self.config.spatial_occupancy_class_weights)
                    if self.config.spatial_occupancy_class_weights
                    else None
                ),
                reduction="none",
            ).reshape(logits.shape[0], logits.shape[1], -1).mean(dim=-1)
            occupancy_loss = self._masked_mean(pixel_loss, mask)
            total = total + self.config.spatial_occupancy_weight * occupancy_loss
            losses["occupancy_loss"] = occupancy_loss
        if output.ego_motion is not None and self.config.spatial_ego_motion_weight:
            if "ego_motions" not in batch:
                raise KeyError(
                    "Spatial world-model batch requires 'ego_motions' targets"
                )
            ego_target = batch["ego_motions"].to(dtype=output.ego_motion.dtype)
            if ego_target.shape != output.ego_motion.shape:
                raise ValueError(
                    f"ego_motions shape {tuple(ego_target.shape)} does not match "
                    f"prediction {tuple(output.ego_motion.shape)}"
                )
            ego_per_step = F.mse_loss(
                output.ego_motion, ego_target, reduction="none"
            ).mean(dim=-1)
            ego_motion_loss = self._masked_mean(ego_per_step, mask)
            total = total + self.config.spatial_ego_motion_weight * ego_motion_loss
            losses["ego_motion_loss"] = ego_motion_loss
        if output.collision_logits is not None:
            if "collisions" not in batch:
                raise KeyError(
                    "Spatial world-model batch requires 'collisions' targets"
                )
            collision_target = batch["collisions"].to(
                dtype=output.collision_logits.dtype
            ).reshape_as(output.collision_logits)
            collision_per_step = F.binary_cross_entropy_with_logits(
                output.collision_logits,
                collision_target.clamp(0.0, 1.0),
                reduction="none",
                pos_weight=output.collision_logits.new_tensor(
                    self.config.spatial_collision_positive_weight
                ),
            )
            collision_loss = self._masked_mean(collision_per_step, mask)
            total = total + self.config.spatial_collision_weight * collision_loss
            losses["collision_loss"] = collision_loss
        losses["loss"] = total
        return losses

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
        embedding = self.encoder(observation.to(dtype=self._model_dtype()))
        state, _, _ = self.dynamics.initial_posterior(
            embedding, sample=bool(sample_state)
        )
        state.context = observation.to(dtype=self._model_dtype())
        return state

    def imagine(
        self,
        initial_state: RSSMState,
        actions: torch.Tensor,
        deterministic: bool = True,
    ) -> WorldModelImagination:
        """Open-loop latent rollout from a posterior state and future actions."""

        prepared_actions = self._prepare_actions(actions)
        if prepared_actions.shape[0] != initial_state.deterministic.shape[0]:
            raise ValueError("Action batch and initial-state batch must match")
        state = initial_state
        observations = []
        rewards = []
        continuations = []
        deterministic_states = []
        stochastic_states = []
        occupancy_logits = []
        ego_motion = []
        collision_probability = []
        for step in range(prepared_actions.shape[1]):
            current_context = state.context
            if self.collision_head is not None:
                collision_input = torch.cat(
                    [state.feature, prepared_actions[:, step]], dim=-1
                )
                learned_collision_logits = self.collision_head(
                    collision_input
                ).squeeze(-1)
                geometry_collision = self._spatial_geometry_collision(
                    current_context, prepared_actions[:, step]
                )
                collision_probability.append(
                    torch.sigmoid(
                        torch.where(
                            geometry_collision,
                            torch.full_like(learned_collision_logits, 8.0),
                            learned_collision_logits,
                        )
                        if geometry_collision is not None
                        else learned_collision_logits
                    )
                )
            state, _, _, _, _ = self.dynamics.step(
                state,
                prepared_actions[:, step],
                embedding=None,
                sample=not deterministic,
            )
            feature = state.feature
            predicted_observation = self.decoder(feature)
            rewards.append(self.reward_head(feature).squeeze(-1))
            continuations.append(torch.sigmoid(self.continue_head(feature).squeeze(-1)))
            deterministic_states.append(state.deterministic)
            stochastic_states.append(state.stochastic)
            if self.occupancy_head is not None:
                predicted_occupancy = self._spatial_occupancy_logits(
                    feature, current_context, prepared_actions[:, step]
                )
                occupancy_logits.append(predicted_occupancy)
                predicted_observation = predicted_observation.clone()
                predicted_observation[
                    :, : self.config.spatial_occupancy_channels
                ] = torch.softmax(predicted_occupancy, dim=1)
            if self.ego_motion_head is not None:
                ego_motion.append(self.ego_motion_head(feature))
            state.context = predicted_observation
            observations.append(predicted_observation)
        if not observations:
            raise ValueError("imagine() requires at least one action")
        return WorldModelImagination(
            observations=torch.stack(observations, dim=1),
            rewards=torch.stack(rewards, dim=1),
            continuation=torch.stack(continuations, dim=1),
            deterministic_states=torch.stack(deterministic_states, dim=1),
            stochastic_states=torch.stack(stochastic_states, dim=1),
            final_state=state,
            occupancy_logits=(
                torch.stack(occupancy_logits, dim=1) if occupancy_logits else None
            ),
            ego_motion=torch.stack(ego_motion, dim=1) if ego_motion else None,
            collision_probability=(
                torch.stack(collision_probability, dim=1)
                if collision_probability
                else None
            ),
        )

    def _spatial_occupancy_logits(
        self,
        feature: torch.Tensor,
        context: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        if self.occupancy_head is None:
            raise RuntimeError("Spatial occupancy head is not configured")
        residual = self.occupancy_head(feature)
        if not self.config.spatial_occupancy_kinematic_prior or context is None:
            return residual
        residual = (
            torch.tanh(residual) * self.config.spatial_occupancy_residual_scale
        )
        channels = self.config.spatial_occupancy_channels
        occupancy = context[:, :channels].to(dtype=residual.dtype)
        height, width = occupancy.shape[-2:]
        movement = action[:, :2]
        movement = torch.where(
            movement.abs() >= 0.25, movement.sign(), torch.zeros_like(movement)
        )
        collision = self._spatial_geometry_collision(context, action)
        if collision is not None:
            movement = movement * (~collision).to(dtype=movement.dtype).unsqueeze(-1)
        theta = torch.zeros(
            occupancy.shape[0], 2, 3, device=occupancy.device, dtype=occupancy.dtype
        )
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = movement[:, 0] * 2.0 / max(width - 1, 1)
        theta[:, 1, 2] = movement[:, 1] * 2.0 / max(height - 1, 1)
        sampling_grid = F.affine_grid(theta, occupancy.shape, align_corners=True)
        warped = F.grid_sample(
            occupancy,
            sampling_grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        missing = (1.0 - warped.sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        if channels >= 3:
            warped = warped.clone()
            warped[:, 2:3] = warped[:, 2:3] + missing
            if self.config.spatial_occupancy_residual_unknown_only:
                # Preserve deterministic geometry and ask the neural residual
                # only to infer cells that have not yet been observed.
                residual = residual * warped[:, 2:3].clamp(0.0, 1.0)
        else:
            warped = warped + missing / channels
        warped = warped / warped.sum(dim=1, keepdim=True).clamp_min(1e-6)
        # A small uniform component keeps the geometric prior strong while still
        # letting the learned residual correct newly revealed or dynamic cells.
        warped = warped * 0.96 + 0.04 / channels
        return torch.log(warped) + residual

    def _spatial_geometry_collision(
        self,
        context: Optional[torch.Tensor],
        action: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Detect an attempted collision from the agent-centred occupancy crop."""

        if (
            not self.config.spatial_occupancy_kinematic_prior
            or context is None
            or self.config.spatial_occupancy_channels < 2
            or context.ndim != 4
        ):
            return None
        occupancy = context[:, : self.config.spatial_occupancy_channels]
        height, width = occupancy.shape[-2:]
        movement = action[:, :2]
        movement = torch.where(
            movement.abs() >= 0.25, movement.sign(), torch.zeros_like(movement)
        )
        step_x = movement[:, 0].to(dtype=torch.long)
        step_y = movement[:, 1].to(dtype=torch.long)
        center_y, center_x = height // 2, width // 2
        batch = torch.arange(occupancy.shape[0], device=occupancy.device)

        def blocked_at(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
            target_x = (center_x + dx).clamp(0, width - 1)
            target_y = (center_y + dy).clamp(0, height - 1)
            return occupancy[batch, 1, target_y, target_x] >= 0.5

        collision = blocked_at(step_x, step_y)
        diagonal = (step_x != 0) & (step_y != 0)
        collision = collision | (
            diagonal
            & (
                blocked_at(step_x, torch.zeros_like(step_y))
                | blocked_at(torch.zeros_like(step_x), step_y)
            )
        )
        # A no-op samples the centre cell, which may carry map semantics but is
        # never a collision.
        return collision & ((step_x != 0) | (step_y != 0))

    @staticmethod
    def _zero_output_layer(module: nn.Module) -> None:
        output_layer: Optional[nn.Module] = None
        for candidate in module.modules():
            if isinstance(candidate, (nn.Linear, nn.ConvTranspose2d)):
                output_layer = candidate
        if output_layer is not None:
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def save_pretrained(self, path: str) -> None:
        """Save model weights plus a JSON network configuration."""

        os.makedirs(path, exist_ok=True)
        config = self.config.to_dict()
        config["backend"] = self.backend_name
        with open(os.path.join(path, self.CONFIG_NAME), "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        torch.save(self.state_dict(), os.path.join(path, self.WEIGHTS_NAME))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        map_location: Optional[Any] = "cpu",
    ) -> "WorldModel":
        """Load a model written by :meth:`save_pretrained`."""

        with open(os.path.join(path, cls.CONFIG_NAME), "r", encoding="utf-8") as file:
            config_data = json.load(file)
        config_data.pop("backend", None)
        config = WorldModelConfig.from_dict(config_data)
        model = cls(config)
        state = _safe_torch_load(os.path.join(path, cls.WEIGHTS_NAME), map_location)
        # v2.0 spatial checkpoints used a state-only collision head.  Preserve
        # them by padding the first layer for the newly explicit action input.
        current_state = model.state_dict()
        for name, value in list(state.items()):
            expected = current_state.get(name)
            if (
                name.startswith("collision_head.")
                and name.endswith(".weight")
                and expected is not None
                and value.ndim == 2
                and expected.ndim == 2
                and value.shape[0] == expected.shape[0]
                and value.shape[1] < expected.shape[1]
            ):
                migrated = expected.new_zeros(expected.shape)
                migrated[:, : value.shape[1]] = value.to(migrated)
                state[name] = migrated
        model.load_state_dict(state)
        return model

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = (
            parameter for parameter in self.parameters() if not trainable_only or parameter.requires_grad
        )
        return sum(parameter.numel() for parameter in parameters)

    def _validate_observations(self, observations: torch.Tensor) -> None:
        expected_rank = 2 + len(self.config.observation_shape)
        if observations.ndim != expected_rank:
            raise ValueError(
                f"observations must have rank {expected_rank}: "
                "[batch, time, *observation_shape]"
            )
        if tuple(observations.shape[2:]) != self.config.observation_shape:
            raise ValueError(
                f"Expected observation shape {self.config.observation_shape}, got "
                f"{tuple(observations.shape[2:])}"
            )

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.config.action_type == "discrete":
            if actions.ndim == 3 and actions.shape[-1] == self.config.action_dim:
                return actions.to(dtype=self._model_dtype())
            if actions.ndim == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
            if actions.ndim != 2:
                raise ValueError("Discrete actions must have shape [batch, time]")
            indices = actions.to(dtype=torch.long)
            if indices.numel() and (indices.min() < 0 or indices.max() >= self.config.action_dim):
                raise ValueError(
                    f"Discrete action ids must be in [0, {self.config.action_dim - 1}]"
                )
            return F.one_hot(indices, num_classes=self.config.action_dim).to(
                dtype=self._model_dtype()
            )

        if actions.ndim == 2 and self.config.action_dim == 1:
            actions = actions.unsqueeze(-1)
        if actions.ndim < 3:
            raise ValueError("Continuous actions must have shape [batch, time, action_dim]")
        actions = actions.flatten(start_dim=2)
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"Expected action_dim={self.config.action_dim}, got {actions.shape[-1]}"
            )
        return actions.to(dtype=self._model_dtype())

    def _model_dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @staticmethod
    def _normal_kl(
        mean_q: torch.Tensor,
        std_q: torch.Tensor,
        mean_p: torch.Tensor,
        std_p: torch.Tensor,
    ) -> torch.Tensor:
        variance_ratio = (std_q / std_p).pow(2)
        mean_term = ((mean_q - mean_p) / std_p).pow(2)
        kl = torch.log(std_p / std_q) + 0.5 * (variance_ratio + mean_term - 1.0)
        return kl.sum(dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denominator


def _safe_torch_load(path: str, map_location: Optional[Any]) -> Any:
    """Load tensor-only files safely on new Torch and compatibly on old Torch."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
