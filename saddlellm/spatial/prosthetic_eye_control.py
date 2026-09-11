"""Safety-first two-axis prosthetic/robotic eye control utilities.

The learned controller intentionally sits outside the hard real-time actuator
loop.  A native SaddleLLM world model may propose normalized yaw/pitch velocity
commands, while :class:`EyeSafetyEnvelope` remains authoritative for limits,
fault latching, and fallback behavior.

This module is hardware-neutral.  It can generate offline trajectories and run
closed-loop simulation without Torch; Torch is loaded only when a world-model
checkpoint is supplied.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

EYE_CONTROL_SCHEMA_VERSION = "saddlellm.eye-control/v1alpha1"
EYE_OBSERVATION_FIELDS = (
    "yaw",
    "pitch",
    "yaw_velocity",
    "pitch_velocity",
    "target_yaw_error",
    "target_pitch_error",
    "target_visible",
    "yaw_current",
    "pitch_current",
)
EYE_OBSERVATION_SIZE = len(EYE_OBSERVATION_FIELDS)
EYE_ACTION_SIZE = 2


class EyeControlError(ValueError):
    """Raised when an eye-control configuration or sample is invalid."""


def _pair(value: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise EyeControlError(f"{name} must contain exactly yaw and pitch values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise EyeControlError(f"{name} must contain finite values")
    return result


def _positive_pair(value: Sequence[float], name: str) -> tuple[float, float]:
    result = _pair(value, name)
    if any(item <= 0.0 for item in result):
        raise EyeControlError(f"{name} values must be positive")
    return result


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EyeControlError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class EyeTarget:
    """Desired gaze in radians, with visibility supplied by perception."""

    yaw: float
    pitch: float
    visible: bool = True
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        _finite(self.yaw, "target.yaw")
        _finite(self.pitch, "target.pitch")
        _finite(self.timestamp, "target.timestamp")


@dataclass(frozen=True)
class EyeObservation:
    """Canonical nine-value observation plus out-of-band safety telemetry."""

    yaw: float
    pitch: float
    yaw_velocity: float
    pitch_velocity: float
    target_yaw_error: float
    target_pitch_error: float
    target_visible: bool
    yaw_current: float
    pitch_current: float
    temperature_c: float = 25.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "yaw",
            "pitch",
            "yaw_velocity",
            "pitch_velocity",
            "target_yaw_error",
            "target_pitch_error",
            "yaw_current",
            "pitch_current",
            "temperature_c",
            "timestamp",
        ):
            _finite(getattr(self, name), f"observation.{name}")

    def to_vector(self) -> list[float]:
        """Return the exact feature order consumed by the example RSSM."""

        return [
            float(self.yaw),
            float(self.pitch),
            float(self.yaw_velocity),
            float(self.pitch_velocity),
            float(self.target_yaw_error),
            float(self.target_pitch_error),
            float(bool(self.target_visible)),
            float(self.yaw_current),
            float(self.pitch_current),
        ]

    @classmethod
    def from_vector(
        cls,
        value: Sequence[float],
        *,
        temperature_c: float = 25.0,
        timestamp: float = 0.0,
    ) -> EyeObservation:
        if isinstance(value, (str, bytes)) or len(value) != EYE_OBSERVATION_SIZE:
            raise EyeControlError(
                f"Eye observation must contain {EYE_OBSERVATION_SIZE} values"
            )
        items = [float(item) for item in value]
        return cls(
            yaw=items[0],
            pitch=items[1],
            yaw_velocity=items[2],
            pitch_velocity=items[3],
            target_yaw_error=items[4],
            target_pitch_error=items[5],
            target_visible=items[6] >= 0.5,
            yaw_current=items[7],
            pitch_current=items[8],
            temperature_c=temperature_c,
            timestamp=timestamp,
        )

    @property
    def tracking_error(self) -> float:
        return math.hypot(self.target_yaw_error, self.target_pitch_error)


@dataclass
class EyeSafetyConfig:
    """Hard and soft limits that remain independent of learned policy output."""

    yaw_limits: tuple[float, float] = (-0.65, 0.65)
    pitch_limits: tuple[float, float] = (-0.45, 0.45)
    max_velocity: tuple[float, float] = (1.8, 1.4)
    max_acceleration: tuple[float, float] = (10.0, 8.0)
    max_current: tuple[float, float] = (1.5, 1.5)
    max_temperature_c: float = 55.0
    soft_limit_margin: tuple[float, float] = (0.08, 0.06)
    neutral_position: tuple[float, float] = (0.0, 0.0)
    fallback_max_velocity: tuple[float, float] = (0.45, 0.35)
    target_timeout_seconds: float = 0.25

    def __post_init__(self) -> None:
        self.yaw_limits = _pair(self.yaw_limits, "safety.yaw_limits")
        self.pitch_limits = _pair(self.pitch_limits, "safety.pitch_limits")
        self.max_velocity = _positive_pair(self.max_velocity, "safety.max_velocity")
        self.max_acceleration = _positive_pair(
            self.max_acceleration, "safety.max_acceleration"
        )
        self.max_current = _positive_pair(self.max_current, "safety.max_current")
        self.soft_limit_margin = _positive_pair(
            self.soft_limit_margin, "safety.soft_limit_margin"
        )
        self.neutral_position = _pair(self.neutral_position, "safety.neutral_position")
        self.fallback_max_velocity = _positive_pair(
            self.fallback_max_velocity, "safety.fallback_max_velocity"
        )
        self.max_temperature_c = _finite(
            self.max_temperature_c, "safety.max_temperature_c"
        )
        self.target_timeout_seconds = _finite(
            self.target_timeout_seconds, "safety.target_timeout_seconds"
        )
        for name, limits in (
            ("yaw_limits", self.yaw_limits),
            ("pitch_limits", self.pitch_limits),
        ):
            if limits[0] >= limits[1]:
                raise EyeControlError(f"safety.{name} lower bound must be smaller")
        if self.max_temperature_c <= 0.0:
            raise EyeControlError("safety.max_temperature_c must be positive")
        if self.target_timeout_seconds < 0.0:
            raise EyeControlError("safety.target_timeout_seconds cannot be negative")
        for axis, position, limits, margin in zip(
            ("yaw", "pitch"),
            self.neutral_position,
            (self.yaw_limits, self.pitch_limits),
            self.soft_limit_margin,
        ):
            if not limits[0] <= position <= limits[1]:
                raise EyeControlError(
                    f"safety.neutral_position {axis} must be inside its limits"
                )
            if margin * 2.0 >= limits[1] - limits[0]:
                raise EyeControlError(
                    f"safety.soft_limit_margin {axis} is too large for its limits"
                )
        if any(
            fallback > maximum
            for fallback, maximum in zip(self.fallback_max_velocity, self.max_velocity)
        ):
            raise EyeControlError(
                "safety.fallback_max_velocity cannot exceed max_velocity"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> EyeSafetyConfig:
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EyePIDConfig:
    """Fallback visual-servo controller gains."""

    kp: tuple[float, float] = (3.0, 3.0)
    ki: tuple[float, float] = (0.15, 0.15)
    kd: tuple[float, float] = (0.18, 0.18)
    integral_limit: tuple[float, float] = (0.35, 0.35)
    deadband_radians: tuple[float, float] = (0.004, 0.004)

    def __post_init__(self) -> None:
        self.kp = _positive_pair(self.kp, "pid.kp")
        self.ki = _pair(self.ki, "pid.ki")
        self.kd = _pair(self.kd, "pid.kd")
        self.integral_limit = _positive_pair(self.integral_limit, "pid.integral_limit")
        self.deadband_radians = _pair(self.deadband_radians, "pid.deadband_radians")
        if any(value < 0.0 for value in (*self.ki, *self.kd, *self.deadband_radians)):
            raise EyeControlError("PID gains and deadbands cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> EyePIDConfig:
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EyePlantConfig:
    """Small deterministic actuator model used for data generation and tests."""

    dt: float = 0.02
    velocity_time_constant: tuple[float, float] = (0.06, 0.07)
    motor_acceleration_limit: tuple[float, float] = (14.0, 11.0)
    command_deadband: float = 0.015
    idle_current: tuple[float, float] = (0.04, 0.04)
    acceleration_current_gain: tuple[float, float] = (0.045, 0.055)
    velocity_current_gain: tuple[float, float] = (0.06, 0.07)
    ambient_temperature_c: float = 24.0
    thermal_heating_rate: float = 0.12
    thermal_cooling_rate: float = 0.08
    sensor_noise_std: float = 0.001

    def __post_init__(self) -> None:
        self.dt = _finite(self.dt, "plant.dt")
        self.velocity_time_constant = _positive_pair(
            self.velocity_time_constant, "plant.velocity_time_constant"
        )
        self.motor_acceleration_limit = _positive_pair(
            self.motor_acceleration_limit, "plant.motor_acceleration_limit"
        )
        self.command_deadband = _finite(self.command_deadband, "plant.command_deadband")
        self.idle_current = _pair(self.idle_current, "plant.idle_current")
        self.acceleration_current_gain = _positive_pair(
            self.acceleration_current_gain, "plant.acceleration_current_gain"
        )
        self.velocity_current_gain = _positive_pair(
            self.velocity_current_gain, "plant.velocity_current_gain"
        )
        self.ambient_temperature_c = _finite(
            self.ambient_temperature_c, "plant.ambient_temperature_c"
        )
        self.thermal_heating_rate = _finite(
            self.thermal_heating_rate, "plant.thermal_heating_rate"
        )
        self.thermal_cooling_rate = _finite(
            self.thermal_cooling_rate, "plant.thermal_cooling_rate"
        )
        self.sensor_noise_std = _finite(self.sensor_noise_std, "plant.sensor_noise_std")
        if self.dt <= 0.0:
            raise EyeControlError("plant.dt must be positive")
        if not 0.0 <= self.command_deadband < 1.0:
            raise EyeControlError("plant.command_deadband must be in [0, 1)")
        if any(value < 0.0 for value in self.idle_current):
            raise EyeControlError("plant.idle_current cannot be negative")
        if self.thermal_heating_rate < 0.0 or self.thermal_cooling_rate < 0.0:
            raise EyeControlError("plant thermal rates cannot be negative")
        if self.sensor_noise_std < 0.0:
            raise EyeControlError("plant.sensor_noise_std cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> EyePlantConfig:
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EyeSafetyResult:
    normalized_action: tuple[float, float]
    velocity_command: tuple[float, float]
    intervened: bool
    hard_stop: bool
    fault: str | None = None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EyeControlCommand:
    """One auditable command ready for an actuator adapter."""

    normalized_action: tuple[float, float]
    velocity_command: tuple[float, float]
    source: str
    safety_intervened: bool
    hard_stop: bool
    fault: str | None = None
    reasons: tuple[str, ...] = ()
    world_model_score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EyeSafetyEnvelope:
    """Apply non-learned limits and latch safety-critical sensor faults."""

    def __init__(self, config: EyeSafetyConfig | None = None) -> None:
        self.config = config or EyeSafetyConfig()
        self._last_velocity = (0.0, 0.0)
        self._latched_fault: str | None = None

    @property
    def latched_fault(self) -> str | None:
        return self._latched_fault

    def emergency_stop(self, reason: str = "external_emergency_stop") -> None:
        self._latched_fault = str(reason or "external_emergency_stop")
        self._last_velocity = (0.0, 0.0)

    def reset_fault(self, observation: EyeObservation) -> None:
        fault = self._sensor_fault(observation)
        if fault:
            raise EyeControlError(
                f"Cannot clear eye-control fault while condition remains: {fault}"
            )
        self._latched_fault = None
        self._last_velocity = (0.0, 0.0)

    def reset_motion(self) -> None:
        """Reset rate-limit memory without clearing a latched fault."""

        self._last_velocity = (0.0, 0.0)

    def apply(
        self,
        desired_action: Sequence[float],
        observation: EyeObservation,
        *,
        dt: float,
        target_age_seconds: float = 0.0,
    ) -> EyeSafetyResult:
        dt = _finite(dt, "control.dt")
        target_age_seconds = _finite(target_age_seconds, "control.target_age_seconds")
        if dt <= 0.0:
            raise EyeControlError("control.dt must be positive")
        if target_age_seconds < 0.0:
            raise EyeControlError("control.target_age_seconds cannot be negative")

        if self._latched_fault:
            return self._stopped(("latched_fault",), self._latched_fault)

        sensor_fault = self._sensor_fault(observation)
        if sensor_fault:
            self.emergency_stop(sensor_fault)
            return self._stopped((sensor_fault,), sensor_fault)

        try:
            requested = _pair(desired_action, "desired_action")
        except (EyeControlError, TypeError, ValueError):
            fault = "invalid_action"
            self.emergency_stop(fault)
            return self._stopped((fault,), fault)

        reasons: list[str] = []
        normalized = tuple(_clamp(value, -1.0, 1.0) for value in requested)
        if normalized != requested:
            reasons.append("action_clamped")
        velocity = [
            normalized[index] * self.config.max_velocity[index]
            for index in range(EYE_ACTION_SIZE)
        ]

        target_available = (
            observation.target_visible
            and target_age_seconds <= self.config.target_timeout_seconds
        )
        if not target_available:
            reasons.append(
                "target_not_visible"
                if not observation.target_visible
                else "target_stale"
            )
            for index in range(EYE_ACTION_SIZE):
                velocity[index] = _clamp(
                    velocity[index],
                    -self.config.fallback_max_velocity[index],
                    self.config.fallback_max_velocity[index],
                )

        for index in range(EYE_ACTION_SIZE):
            maximum_delta = self.config.max_acceleration[index] * dt
            limited = _clamp(
                velocity[index],
                self._last_velocity[index] - maximum_delta,
                self._last_velocity[index] + maximum_delta,
            )
            if not math.isclose(limited, velocity[index], abs_tol=1e-12):
                reasons.append(f"{('yaw', 'pitch')[index]}_acceleration_limited")
            velocity[index] = limited

        positions = (observation.yaw, observation.pitch)
        limits = (self.config.yaw_limits, self.config.pitch_limits)
        for index, axis in enumerate(("yaw", "pitch")):
            low, high = limits[index]
            position = positions[index]
            margin = self.config.soft_limit_margin[index]
            original = velocity[index]
            if (
                original < 0.0
                and position <= low
                or original > 0.0
                and position >= high
            ):
                velocity[index] = 0.0
                reasons.append(f"{axis}_hard_limit")
            elif original < 0.0 and position < low + margin:
                velocity[index] *= _clamp((position - low) / margin, 0.0, 1.0)
                reasons.append(f"{axis}_soft_limit")
            elif original > 0.0 and position > high - margin:
                velocity[index] *= _clamp((high - position) / margin, 0.0, 1.0)
                reasons.append(f"{axis}_soft_limit")

        currents = (abs(observation.yaw_current), abs(observation.pitch_current))
        for index, axis in enumerate(("yaw", "pitch")):
            current_ratio = currents[index] / self.config.max_current[index]
            if current_ratio > 0.8:
                scale = _clamp((1.0 - current_ratio) / 0.2, 0.0, 1.0)
                velocity[index] *= scale
                reasons.append(f"{axis}_current_derated")

        final_action = tuple(
            _clamp(velocity[index] / self.config.max_velocity[index], -1.0, 1.0)
            for index in range(EYE_ACTION_SIZE)
        )
        final_velocity = (float(velocity[0]), float(velocity[1]))
        self._last_velocity = final_velocity
        return EyeSafetyResult(
            normalized_action=final_action,
            velocity_command=final_velocity,
            intervened=bool(reasons),
            hard_stop=False,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _sensor_fault(self, observation: EyeObservation) -> str | None:
        if observation.temperature_c >= self.config.max_temperature_c:
            return "over_temperature"
        currents = (abs(observation.yaw_current), abs(observation.pitch_current))
        for index, axis in enumerate(("yaw", "pitch")):
            if currents[index] >= self.config.max_current[index]:
                return f"{axis}_over_current"
        tolerance = 1e-3
        if (
            not self.config.yaw_limits[0] - tolerance
            <= observation.yaw
            <= self.config.yaw_limits[1] + tolerance
        ):
            return "yaw_position_out_of_range"
        if (
            not self.config.pitch_limits[0] - tolerance
            <= observation.pitch
            <= self.config.pitch_limits[1] + tolerance
        ):
            return "pitch_position_out_of_range"
        return None

    def _stopped(self, reasons: tuple[str, ...], fault: str | None) -> EyeSafetyResult:
        return EyeSafetyResult(
            normalized_action=(0.0, 0.0),
            velocity_command=(0.0, 0.0),
            intervened=True,
            hard_stop=True,
            fault=fault,
            reasons=reasons,
        )


class EyePIDController:
    """PID fallback that tracks a visible target or slowly returns to neutral."""

    def __init__(self, config: EyePIDConfig | None = None) -> None:
        self.config = config or EyePIDConfig()
        self._integral = [0.0, 0.0]

    def reset(self) -> None:
        self._integral = [0.0, 0.0]

    def compute(
        self,
        observation: EyeObservation,
        safety: EyeSafetyConfig,
        *,
        dt: float,
        target_age_seconds: float = 0.0,
    ) -> tuple[float, float]:
        dt = _finite(dt, "control.dt")
        if dt <= 0.0:
            raise EyeControlError("control.dt must be positive")
        available = (
            observation.target_visible
            and target_age_seconds <= safety.target_timeout_seconds
        )
        if available:
            errors = (
                observation.target_yaw_error,
                observation.target_pitch_error,
            )
        else:
            errors = (
                safety.neutral_position[0] - observation.yaw,
                safety.neutral_position[1] - observation.pitch,
            )
            # Do not carry target-specific integral state into neutral fallback.
            self._integral = [0.0, 0.0]
        velocities = (observation.yaw_velocity, observation.pitch_velocity)
        result = []
        for index in range(EYE_ACTION_SIZE):
            error = errors[index]
            if abs(error) <= self.config.deadband_radians[index]:
                error = 0.0
            self._integral[index] = _clamp(
                self._integral[index] + error * dt,
                -self.config.integral_limit[index],
                self.config.integral_limit[index],
            )
            velocity = (
                self.config.kp[index] * error
                + self.config.ki[index] * self._integral[index]
                - self.config.kd[index] * velocities[index]
            )
            result.append(_clamp(velocity / safety.max_velocity[index], -1.0, 1.0))
        return float(result[0]), float(result[1])


class ProstheticEyeController:
    """Hybrid RSSM+CEM controller with PID fallback and hard safety envelope."""

    def __init__(
        self,
        *,
        safety_config: EyeSafetyConfig | None = None,
        pid_config: EyePIDConfig | None = None,
        world_model_runtime: Any = None,
        planner_config: Any = None,
        history_length: int = 32,
        model_warmup_steps: int = 2,
    ) -> None:
        if history_length < 1:
            raise EyeControlError("history_length must be positive")
        if model_warmup_steps < 0:
            raise EyeControlError("model_warmup_steps cannot be negative")
        self.safety_config = safety_config or EyeSafetyConfig()
        self.pid = EyePIDController(pid_config)
        self.safety = EyeSafetyEnvelope(self.safety_config)
        self.world_model_runtime = world_model_runtime
        self.planner_config = planner_config
        self.history_length = int(history_length)
        self.model_warmup_steps = int(model_warmup_steps)
        self._observations: list[list[float]] = []
        self._actions: list[list[float]] = []
        self._last_action: list[float] | None = None
        self._step = 0
        if self.world_model_runtime is not None:
            self._validate_world_model_runtime(self.world_model_runtime)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | os.PathLike[str],
        *,
        device: str = "auto",
        safety_config: EyeSafetyConfig | None = None,
        pid_config: EyePIDConfig | None = None,
        planner_config: Any = None,
        history_length: int = 32,
        model_warmup_steps: int = 2,
    ) -> ProstheticEyeController:
        from ..world_models.WorldModelInference import WorldModelRuntime

        runtime = WorldModelRuntime.from_pretrained(checkpoint, device=device)
        return cls(
            safety_config=safety_config,
            pid_config=pid_config,
            world_model_runtime=runtime,
            planner_config=planner_config,
            history_length=history_length,
            model_warmup_steps=model_warmup_steps,
        )

    def reset(self) -> None:
        """Reset belief and motion memory without clearing a safety fault."""

        self.pid.reset()
        self.safety.reset_motion()
        self._observations.clear()
        self._actions.clear()
        self._last_action = None
        self._step = 0

    def reset_fault(self, observation: EyeObservation) -> None:
        self.safety.reset_fault(observation)

    def control(
        self,
        observation: EyeObservation | Sequence[float],
        *,
        dt: float,
        target_age_seconds: float = 0.0,
        use_world_model: bool = True,
    ) -> EyeControlCommand:
        if not isinstance(observation, EyeObservation):
            observation = EyeObservation.from_vector(observation)
        self._append_observation(observation)
        diagnostics: dict[str, Any] = {
            "step": self._step,
            "history_transitions": len(self._actions),
        }
        source = "pid_fallback"
        score: float | None = None
        raw_action = self.pid.compute(
            observation,
            self.safety_config,
            dt=dt,
            target_age_seconds=target_age_seconds,
        )
        target_available = (
            observation.target_visible
            and target_age_seconds <= self.safety_config.target_timeout_seconds
        )
        model_ready = (
            use_world_model
            and self.world_model_runtime is not None
            and target_available
            and len(self._actions) >= self.model_warmup_steps
            and self.safety.latched_fault is None
        )
        if model_ready:
            planning_started = time.perf_counter()
            try:
                state = self._world_model_state()
                planner = self._planner_for_step()
                plan = self.world_model_runtime.plan(state, planner)
                value = plan.action
                if hasattr(value, "detach"):
                    value = value.detach().cpu().tolist()
                raw_action = _pair(value, "world_model.action")
                score_value = plan.score
                if hasattr(score_value, "detach"):
                    score_value = score_value.detach().cpu()
                score = float(score_value)
                source = "world_model_cem"
            except Exception as exc:  # noqa: BLE001 - fail closed to PID
                diagnostics["world_model_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                diagnostics["world_model_latency_ms"] = (
                    time.perf_counter() - planning_started
                ) * 1000.0
        elif use_world_model and self.world_model_runtime is not None:
            if not target_available:
                diagnostics["model_bypass"] = "target_unavailable"
            elif len(self._actions) < self.model_warmup_steps:
                diagnostics["model_bypass"] = "history_warmup"
            elif self.safety.latched_fault:
                diagnostics["model_bypass"] = "safety_fault"

        safety_result = self.safety.apply(
            raw_action,
            observation,
            dt=dt,
            target_age_seconds=target_age_seconds,
        )
        if safety_result.hard_stop:
            source = "safety_stop"
        self._last_action = list(safety_result.normalized_action)
        self._step += 1
        return EyeControlCommand(
            normalized_action=safety_result.normalized_action,
            velocity_command=safety_result.velocity_command,
            source=source,
            safety_intervened=safety_result.intervened,
            hard_stop=safety_result.hard_stop,
            fault=safety_result.fault,
            reasons=safety_result.reasons,
            world_model_score=score,
            diagnostics=diagnostics,
        )

    def _append_observation(self, observation: EyeObservation) -> None:
        if self._last_action is not None:
            self._actions.append(list(self._last_action))
        self._observations.append(observation.to_vector())
        while len(self._actions) > self.history_length:
            self._actions.pop(0)
            self._observations.pop(0)
        if len(self._observations) != len(self._actions) + 1:
            raise RuntimeError("Eye controller history alignment invariant failed")

    def _world_model_state(self) -> Any:
        if self._actions:
            output = self.world_model_runtime.filter(
                self._observations,
                self._actions,
                deterministic=True,
            )
            return output.final_state
        return self.world_model_runtime.encode_observation(
            self._observations[-1], deterministic=True
        )

    def _planner_for_step(self) -> Any:
        from ..world_models.WorldModelInference import WorldModelPlannerConfig

        planner = self.planner_config or WorldModelPlannerConfig(
            horizon=10,
            num_candidates=128,
            iterations=4,
            elite_fraction=0.1,
        )
        if isinstance(planner, dict):
            planner = WorldModelPlannerConfig.from_dict(planner)
        return replace(planner, seed=int(planner.seed) + self._step)

    @staticmethod
    def _validate_world_model_runtime(runtime: Any) -> None:
        model = getattr(runtime, "model", None)
        config = getattr(model, "config", None)
        if config is None:
            raise EyeControlError("world_model_runtime must expose model.config")
        shape = tuple(getattr(config, "observation_shape", ()))
        if shape != (EYE_OBSERVATION_SIZE,):
            raise EyeControlError(
                "Eye controller requires world-model observation_shape=(9,), "
                f"got {shape}"
            )
        if int(getattr(config, "action_dim", 0)) != EYE_ACTION_SIZE:
            raise EyeControlError("Eye controller requires world-model action_dim=2")
        if str(getattr(config, "action_type", "")).lower() != "continuous":
            raise EyeControlError(
                "Eye controller requires a continuous-action world model"
            )


class ProstheticEyeSimulator:
    """Simple two-axis plant for safe offline iteration, not a medical twin."""

    def __init__(
        self,
        *,
        safety_config: EyeSafetyConfig | None = None,
        plant_config: EyePlantConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.safety_config = safety_config or EyeSafetyConfig()
        self.config = plant_config or EyePlantConfig()
        self._random = random.Random(seed)
        self.yaw = 0.0
        self.pitch = 0.0
        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.yaw_current = self.config.idle_current[0]
        self.pitch_current = self.config.idle_current[1]
        self.temperature_c = self.config.ambient_temperature_c
        self.time = 0.0
        self.target = EyeTarget(0.0, 0.0, True, 0.0)

    def reset(
        self,
        *,
        yaw: float = 0.0,
        pitch: float = 0.0,
        target: EyeTarget | None = None,
    ) -> EyeObservation:
        self.yaw = _clamp(_finite(yaw, "simulator.yaw"), *self.safety_config.yaw_limits)
        self.pitch = _clamp(
            _finite(pitch, "simulator.pitch"), *self.safety_config.pitch_limits
        )
        self.yaw_velocity = 0.0
        self.pitch_velocity = 0.0
        self.yaw_current = self.config.idle_current[0]
        self.pitch_current = self.config.idle_current[1]
        self.temperature_c = self.config.ambient_temperature_c
        self.time = 0.0
        self.target = target or EyeTarget(0.0, 0.0, True, 0.0)
        return self.observe()

    def observe(self, target: EyeTarget | None = None) -> EyeObservation:
        if target is not None:
            self.target = target
        noise = self.config.sensor_noise_std
        yaw_noise = self._random.gauss(0.0, noise) if noise else 0.0
        pitch_noise = self._random.gauss(0.0, noise) if noise else 0.0
        return EyeObservation(
            yaw=self.yaw + yaw_noise,
            pitch=self.pitch + pitch_noise,
            yaw_velocity=self.yaw_velocity,
            pitch_velocity=self.pitch_velocity,
            target_yaw_error=self.target.yaw - self.yaw,
            target_pitch_error=self.target.pitch - self.pitch,
            target_visible=self.target.visible,
            yaw_current=self.yaw_current,
            pitch_current=self.pitch_current,
            temperature_c=self.temperature_c,
            timestamp=self.time,
        )

    def step(
        self,
        action: Sequence[float],
        *,
        target: EyeTarget | None = None,
    ) -> tuple[EyeObservation, float, bool, dict[str, Any]]:
        normalized = _pair(action, "simulator.action")
        normalized = tuple(_clamp(item, -1.0, 1.0) for item in normalized)
        desired_velocity = [
            normalized[index] * self.safety_config.max_velocity[index]
            if abs(normalized[index]) >= self.config.command_deadband
            else 0.0
            for index in range(EYE_ACTION_SIZE)
        ]
        velocity = [self.yaw_velocity, self.pitch_velocity]
        acceleration = []
        for index in range(EYE_ACTION_SIZE):
            requested = (
                desired_velocity[index] - velocity[index]
            ) / self.config.velocity_time_constant[index]
            acceleration.append(
                _clamp(
                    requested,
                    -self.config.motor_acceleration_limit[index],
                    self.config.motor_acceleration_limit[index],
                )
            )
            velocity[index] += acceleration[index] * self.config.dt

        positions = [self.yaw, self.pitch]
        limits = (self.safety_config.yaw_limits, self.safety_config.pitch_limits)
        hit_limit = [False, False]
        for index in range(EYE_ACTION_SIZE):
            positions[index] += velocity[index] * self.config.dt
            clipped = _clamp(positions[index], *limits[index])
            if not math.isclose(clipped, positions[index], abs_tol=1e-12):
                positions[index] = clipped
                velocity[index] = 0.0
                hit_limit[index] = True

        self.yaw, self.pitch = positions
        self.yaw_velocity, self.pitch_velocity = velocity
        currents = []
        for index in range(EYE_ACTION_SIZE):
            currents.append(
                self.config.idle_current[index]
                + self.config.acceleration_current_gain[index]
                * abs(acceleration[index])
                + self.config.velocity_current_gain[index] * abs(velocity[index])
                + (0.25 if hit_limit[index] else 0.0)
            )
        self.yaw_current, self.pitch_current = currents
        power = sum(current * current for current in currents)
        temperature_delta = (
            self.config.thermal_heating_rate * power
            - self.config.thermal_cooling_rate
            * (self.temperature_c - self.config.ambient_temperature_c)
        )
        self.temperature_c += temperature_delta * self.config.dt
        self.time += self.config.dt
        if target is not None:
            self.target = target
        observation = self.observe()
        error_scale = math.hypot(
            observation.target_yaw_error
            / max(
                abs(self.safety_config.yaw_limits[0]), self.safety_config.yaw_limits[1]
            ),
            observation.target_pitch_error
            / max(
                abs(self.safety_config.pitch_limits[0]),
                self.safety_config.pitch_limits[1],
            ),
        )
        reward = math.exp(-2.5 * error_scale)
        reward -= 0.015 * sum(item * item for item in normalized)
        reward -= 0.01 * sum(
            current / maximum
            for current, maximum in zip(currents, self.safety_config.max_current)
        )
        reward -= 0.1 * float(any(hit_limit))
        fault = None
        if self.temperature_c >= self.safety_config.max_temperature_c:
            fault = "over_temperature"
        elif self.yaw_current >= self.safety_config.max_current[0]:
            fault = "yaw_over_current"
        elif self.pitch_current >= self.safety_config.max_current[1]:
            fault = "pitch_over_current"
        return (
            observation,
            float(reward),
            fault is not None,
            {
                "fault": fault,
                "hit_limit": hit_limit,
                "desired_velocity": desired_velocity,
                "acceleration": acceleration,
            },
        )


def generate_prosthetic_eye_dataset(
    output_path: str | os.PathLike[str],
    *,
    episodes: int = 128,
    steps: int = 100,
    seed: int = 42,
    moving_target_ratio: float = 0.5,
    target_dropout_probability: float = 0.03,
    safety_config: EyeSafetyConfig | None = None,
    pid_config: EyePIDConfig | None = None,
    plant_config: EyePlantConfig | None = None,
) -> dict[str, Any]:
    """Generate PID-expert JSONL trajectories compatible with world-model training."""

    if episodes <= 0 or steps <= 0:
        raise EyeControlError("episodes and steps must be positive")
    if not 0.0 <= moving_target_ratio <= 1.0:
        raise EyeControlError("moving_target_ratio must be in [0, 1]")
    if not 0.0 <= target_dropout_probability < 1.0:
        raise EyeControlError("target_dropout_probability must be in [0, 1)")
    safety = safety_config or EyeSafetyConfig()
    plant = plant_config or EyePlantConfig()
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    transition_count = 0
    for episode_index in range(episodes):
        episode_seed = rng.randrange(2**31)
        simulator = ProstheticEyeSimulator(
            safety_config=safety,
            plant_config=plant,
            seed=episode_seed,
        )
        controller = ProstheticEyeController(
            safety_config=safety,
            pid_config=pid_config,
            model_warmup_steps=0,
        )
        moving = rng.random() < moving_target_ratio
        target_parameters = _sample_target_parameters(rng, safety, moving)
        initial_target = _scheduled_target(
            target_parameters,
            0,
            steps,
            visible=True,
            timestamp=0.0,
            safety=safety,
        )
        observation = simulator.reset(
            yaw=rng.uniform(-0.15, 0.15),
            pitch=rng.uniform(-0.1, 0.1),
            target=initial_target,
        )
        observations = [observation.to_vector()]
        actions: list[list[float]] = []
        rewards: list[float] = []
        dones: list[float] = []
        sources: dict[str, int] = {}
        for step_index in range(steps):
            target_age = (
                0.0
                if observation.target_visible
                else safety.target_timeout_seconds + plant.dt
            )
            command = controller.control(
                observation,
                dt=plant.dt,
                target_age_seconds=target_age,
                use_world_model=False,
            )
            sources[command.source] = sources.get(command.source, 0) + 1
            next_visible = rng.random() >= target_dropout_probability
            next_target = _scheduled_target(
                target_parameters,
                step_index + 1,
                steps,
                visible=next_visible,
                timestamp=simulator.time + plant.dt,
                safety=safety,
            )
            next_observation, reward, faulted, _ = simulator.step(
                command.normalized_action,
                target=next_target,
            )
            final = faulted or step_index == steps - 1
            actions.append(list(command.normalized_action))
            rewards.append(reward)
            dones.append(float(final))
            observations.append(next_observation.to_vector())
            observation = next_observation
            transition_count += 1
            if faulted:
                break
        records.append(
            {
                "episode_id": f"eye-sim-{episode_index:06d}",
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "metadata": {
                    "schema_version": EYE_CONTROL_SCHEMA_VERSION,
                    "observation_fields": list(EYE_OBSERVATION_FIELDS),
                    "action": "normalized_yaw_pitch_velocity",
                    "target_motion": "moving" if moving else "static",
                    "control_hz": 1.0 / plant.dt,
                    "expert": "pid_with_safety_envelope",
                    "command_sources": sources,
                    "simulator_seed": episode_seed,
                },
            }
        )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return {
        "status": "completed",
        "schema_version": EYE_CONTROL_SCHEMA_VERSION,
        "output_path": str(destination),
        "episodes": len(records),
        "transitions": transition_count,
        "observation_shape": [EYE_OBSERVATION_SIZE],
        "action_dim": EYE_ACTION_SIZE,
        "action_type": "continuous",
        "control_hz": 1.0 / plant.dt,
        "seed": seed,
    }


def simulate_prosthetic_eye_control(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    episodes: int = 8,
    steps: int = 150,
    seed: int = 42,
    device: str = "auto",
    safety_config: EyeSafetyConfig | None = None,
    pid_config: EyePIDConfig | None = None,
    plant_config: EyePlantConfig | None = None,
    planner_config: Any = None,
    include_traces: bool = False,
) -> dict[str, Any]:
    """Evaluate the hybrid controller in simulation and return JSON-safe metrics."""

    if episodes <= 0 or steps <= 0:
        raise EyeControlError("episodes and steps must be positive")
    safety = safety_config or EyeSafetyConfig()
    plant = plant_config or EyePlantConfig()
    runtime = None
    checkpoint_path = None
    if checkpoint is not None:
        from ..world_models.WorldModelInference import WorldModelRuntime

        checkpoint_path = str(Path(checkpoint).expanduser().resolve())
        runtime = WorldModelRuntime.from_pretrained(checkpoint_path, device=device)
    rng = random.Random(seed)
    errors: list[float] = []
    final_errors: list[float] = []
    total_reward = 0.0
    interventions = 0
    hard_stops = 0
    sources: dict[str, int] = {}
    planner_latencies: list[float] = []
    traces: list[dict[str, Any]] = []
    completed_episodes = 0
    for episode_index in range(episodes):
        episode_seed = rng.randrange(2**31)
        simulator = ProstheticEyeSimulator(
            safety_config=safety,
            plant_config=plant,
            seed=episode_seed,
        )
        controller = ProstheticEyeController(
            safety_config=safety,
            pid_config=pid_config,
            world_model_runtime=runtime,
            planner_config=planner_config,
        )
        moving = episode_index % 2 == 1
        target_parameters = _sample_target_parameters(rng, safety, moving)
        target = _scheduled_target(
            target_parameters,
            0,
            steps,
            visible=True,
            timestamp=0.0,
            safety=safety,
        )
        observation = simulator.reset(
            yaw=rng.uniform(-0.1, 0.1),
            pitch=rng.uniform(-0.08, 0.08),
            target=target,
        )
        episode_trace: list[dict[str, Any]] = []
        for step_index in range(steps):
            errors.append(observation.tracking_error)
            command = controller.control(observation, dt=plant.dt)
            sources[command.source] = sources.get(command.source, 0) + 1
            latency = command.diagnostics.get("world_model_latency_ms")
            if latency is not None:
                planner_latencies.append(float(latency))
            interventions += int(command.safety_intervened)
            hard_stops += int(command.hard_stop)
            next_target = _scheduled_target(
                target_parameters,
                step_index + 1,
                steps,
                visible=True,
                timestamp=simulator.time + plant.dt,
                safety=safety,
            )
            observation, reward, faulted, info = simulator.step(
                command.normalized_action,
                target=next_target,
            )
            total_reward += reward
            if include_traces:
                episode_trace.append(
                    {
                        "step": step_index,
                        "observation": observation.to_vector(),
                        "command": command.to_dict(),
                        "reward": reward,
                        "plant": info,
                    }
                )
            if faulted or command.hard_stop:
                break
        final_errors.append(observation.tracking_error)
        completed_episodes += 1
        if include_traces:
            traces.append(
                {
                    "episode_id": f"eye-eval-{episode_index:04d}",
                    "target_motion": "moving" if moving else "static",
                    "steps": episode_trace,
                }
            )

    sorted_errors = sorted(errors)
    p95_index = max(0, math.ceil(0.95 * len(sorted_errors)) - 1)
    sorted_latencies = sorted(planner_latencies)
    latency_p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    deadline_ms = plant.dt * 1000.0
    result: dict[str, Any] = {
        "status": "completed",
        "schema_version": EYE_CONTROL_SCHEMA_VERSION,
        "checkpoint": checkpoint_path,
        "controller": "rssm_cem_with_pid_fallback" if runtime else "pid_fallback",
        "episodes": completed_episodes,
        "steps": len(errors),
        "mean_tracking_error_radians": (sum(errors) / len(errors) if errors else None),
        "p95_tracking_error_radians": (
            sorted_errors[p95_index] if sorted_errors else None
        ),
        "mean_final_error_radians": (
            sum(final_errors) / len(final_errors) if final_errors else None
        ),
        "success_rate": (
            sum(error <= 0.05 for error in final_errors) / len(final_errors)
            if final_errors
            else 0.0
        ),
        "mean_reward": total_reward / max(1, len(errors)),
        "safety_interventions": interventions,
        "hard_stops": hard_stops,
        "command_sources": sources,
        "planner_mean_latency_ms": (
            sum(planner_latencies) / len(planner_latencies)
            if planner_latencies
            else None
        ),
        "planner_p95_latency_ms": (
            sorted_latencies[latency_p95_index] if sorted_latencies else None
        ),
        "planner_deadline_miss_rate": (
            sum(latency > deadline_ms for latency in planner_latencies)
            / len(planner_latencies)
            if planner_latencies
            else None
        ),
        "control_deadline_ms": deadline_ms,
        "control_hz": 1.0 / plant.dt,
        "seed": seed,
    }
    if include_traces:
        result["traces"] = traces
    return result


@runtime_checkable
class EyeHardwareAdapter(Protocol):
    """Minimal boundary implemented by a device-specific, reviewed driver."""

    def read_observation(self) -> EyeObservation:
        """Read one calibrated observation and safety telemetry sample."""

    def write_velocity(
        self, yaw_radians_per_second: float, pitch_radians_per_second: float
    ) -> None:
        """Send an already safety-filtered velocity setpoint."""

    def emergency_stop(self, reason: str) -> None:
        """Disable actuator output through an independent hardware path."""


class EyeControlLoop:
    """Single-tick hardware bridge; callers own timing and watchdog scheduling."""

    def __init__(
        self,
        controller: ProstheticEyeController,
        adapter: EyeHardwareAdapter,
        *,
        control_hz: float = 50.0,
    ) -> None:
        control_hz = _finite(control_hz, "control_hz")
        if control_hz <= 0.0:
            raise EyeControlError("control_hz must be positive")
        if not isinstance(adapter, EyeHardwareAdapter):
            raise EyeControlError(
                "adapter must implement read_observation, write_velocity, and emergency_stop"
            )
        self.controller = controller
        self.adapter = adapter
        self.dt = 1.0 / control_hz

    def tick(self, *, target_age_seconds: float = 0.0) -> EyeControlCommand:
        try:
            observation = self.adapter.read_observation()
            if not isinstance(observation, EyeObservation):
                raise EyeControlError(
                    "Hardware adapters must return EyeObservation with safety telemetry"
                )
            command = self.controller.control(
                observation,
                dt=self.dt,
                target_age_seconds=target_age_seconds,
            )
            if command.hard_stop:
                self.adapter.emergency_stop(command.fault or "eye_control_safety_stop")
            else:
                self.adapter.write_velocity(*command.velocity_command)
            return command
        except Exception as exc:  # noqa: BLE001 - hardware bridge must fail closed
            reason = f"control_loop_exception:{type(exc).__name__}"
            self.controller.safety.emergency_stop(reason)
            try:
                self.adapter.emergency_stop(reason)
            finally:
                raise

    def emergency_stop(self, reason: str = "external_emergency_stop") -> None:
        self.controller.safety.emergency_stop(reason)
        self.adapter.emergency_stop(reason)


def _sample_target_parameters(
    rng: random.Random,
    safety: EyeSafetyConfig,
    moving: bool,
) -> dict[str, float | bool]:
    yaw_span = min(abs(safety.yaw_limits[0]), safety.yaw_limits[1])
    pitch_span = min(abs(safety.pitch_limits[0]), safety.pitch_limits[1])
    return {
        "moving": moving,
        "yaw_center": rng.uniform(-0.45 * yaw_span, 0.45 * yaw_span),
        "pitch_center": rng.uniform(-0.45 * pitch_span, 0.45 * pitch_span),
        "yaw_amplitude": rng.uniform(0.08, 0.28) * yaw_span if moving else 0.0,
        "pitch_amplitude": (rng.uniform(0.08, 0.25) * pitch_span if moving else 0.0),
        "phase": rng.uniform(0.0, 2.0 * math.pi),
        "cycles": rng.uniform(0.35, 1.25),
    }


def _scheduled_target(
    parameters: dict[str, float | bool],
    step: int,
    steps: int,
    *,
    visible: bool,
    timestamp: float,
    safety: EyeSafetyConfig,
) -> EyeTarget:
    progress = step / max(1, steps)
    phase = float(parameters["phase"])
    angle = 2.0 * math.pi * float(parameters["cycles"]) * progress + phase
    yaw = float(parameters["yaw_center"])
    pitch = float(parameters["pitch_center"])
    if bool(parameters["moving"]):
        yaw += float(parameters["yaw_amplitude"]) * math.sin(angle)
        pitch += float(parameters["pitch_amplitude"]) * math.cos(angle * 0.8)
    yaw = _clamp(yaw, safety.yaw_limits[0] + 0.02, safety.yaw_limits[1] - 0.02)
    pitch = _clamp(pitch, safety.pitch_limits[0] + 0.02, safety.pitch_limits[1] - 0.02)
    return EyeTarget(yaw=yaw, pitch=pitch, visible=visible, timestamp=timestamp)


__all__ = [
    "EYE_ACTION_SIZE",
    "EYE_CONTROL_SCHEMA_VERSION",
    "EYE_OBSERVATION_FIELDS",
    "EYE_OBSERVATION_SIZE",
    "EyeControlCommand",
    "EyeControlError",
    "EyeControlLoop",
    "EyeHardwareAdapter",
    "EyeObservation",
    "EyePIDConfig",
    "EyePIDController",
    "EyePlantConfig",
    "EyeSafetyConfig",
    "EyeSafetyEnvelope",
    "EyeSafetyResult",
    "EyeTarget",
    "ProstheticEyeController",
    "ProstheticEyeSimulator",
    "generate_prosthetic_eye_dataset",
    "simulate_prosthetic_eye_control",
]
