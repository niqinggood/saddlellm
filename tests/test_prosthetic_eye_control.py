import json
from types import SimpleNamespace

import pytest

from saddlellm.cli import main as cli_main
from saddlellm.spatial.prosthetic_eye_control import (
    EYE_CONTROL_SCHEMA_VERSION,
    EYE_OBSERVATION_FIELDS,
    EyeControlError,
    EyeControlLoop,
    EyeObservation,
    EyePlantConfig,
    EyeSafetyConfig,
    EyeSafetyEnvelope,
    EyeTarget,
    ProstheticEyeController,
    ProstheticEyeSimulator,
    generate_prosthetic_eye_dataset,
    simulate_prosthetic_eye_control,
)
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator
from saddlellm.world_models.WorldModelData import (
    infer_world_model_dimensions,
    load_world_model_trajectories,
)


def _observation(**overrides):
    values = {
        "yaw": 0.0,
        "pitch": 0.0,
        "yaw_velocity": 0.0,
        "pitch_velocity": 0.0,
        "target_yaw_error": 0.2,
        "target_pitch_error": -0.1,
        "target_visible": True,
        "yaw_current": 0.1,
        "pitch_current": 0.1,
        "temperature_c": 25.0,
        "timestamp": 0.0,
    }
    values.update(overrides)
    return EyeObservation(**values)


def test_observation_vector_has_stable_world_model_schema():
    observation = _observation()
    restored = EyeObservation.from_vector(
        observation.to_vector(), temperature_c=31.0, timestamp=1.5
    )

    assert len(observation.to_vector()) == len(EYE_OBSERVATION_FIELDS) == 9
    assert restored.target_visible is True
    assert restored.target_yaw_error == pytest.approx(0.2)
    assert restored.temperature_c == pytest.approx(31.0)
    assert restored.timestamp == pytest.approx(1.5)


def test_safety_envelope_limits_motion_and_latches_sensor_faults():
    safety = EyeSafetyConfig(
        max_velocity=(1.0, 1.0),
        max_acceleration=(2.0, 2.0),
        soft_limit_margin=(0.1, 0.1),
    )
    envelope = EyeSafetyEnvelope(safety)

    limited = envelope.apply(
        (2.0, -2.0),
        _observation(yaw=0.64),
        dt=0.1,
    )
    assert limited.intervened
    assert abs(limited.velocity_command[0]) < 0.03
    assert limited.velocity_command[1] == pytest.approx(-0.2)
    assert "action_clamped" in limited.reasons
    assert "yaw_soft_limit" in limited.reasons

    faulted = envelope.apply(
        (0.2, 0.0),
        _observation(yaw_current=safety.max_current[0]),
        dt=0.1,
    )
    assert faulted.hard_stop
    assert faulted.fault == "yaw_over_current"

    still_stopped = envelope.apply((0.2, 0.0), _observation(), dt=0.1)
    assert still_stopped.hard_stop
    assert still_stopped.fault == "yaw_over_current"

    envelope.reset_fault(_observation())
    recovered = envelope.apply((0.2, 0.0), _observation(), dt=0.1)
    assert not recovered.hard_stop


def test_pid_fallback_tracks_target_in_closed_loop_simulation():
    plant = EyePlantConfig(sensor_noise_std=0.0)
    simulator = ProstheticEyeSimulator(plant_config=plant, seed=3)
    target = EyeTarget(0.30, -0.15)
    observation = simulator.reset(target=target)
    initial_error = observation.tracking_error
    controller = ProstheticEyeController(model_warmup_steps=0)

    for _ in range(150):
        command = controller.control(observation, dt=plant.dt, use_world_model=False)
        observation, _, faulted, _ = simulator.step(
            command.normalized_action, target=target
        )
        assert not faulted

    assert observation.tracking_error < 0.01
    assert observation.tracking_error < initial_error / 20.0


class _FakeWorldModelRuntime:
    def __init__(self, *, fail=False):
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                observation_shape=(9,), action_dim=2, action_type="continuous"
            )
        )
        self.fail = fail
        self.plan_calls = 0

    def encode_observation(self, observation, deterministic=True):
        return {"observation": observation, "deterministic": deterministic}

    def filter(self, observations, actions, deterministic=True):
        return SimpleNamespace(final_state=(observations, actions, deterministic))

    def plan(self, state, planner):
        self.plan_calls += 1
        if self.fail:
            raise RuntimeError("planner unavailable")
        return SimpleNamespace(action=[0.8, -0.6], score=1.25)


def test_world_model_outer_loop_and_fail_closed_pid_fallback():
    runtime = _FakeWorldModelRuntime()
    controller = ProstheticEyeController(
        world_model_runtime=runtime,
        model_warmup_steps=0,
    )
    command = controller.control(_observation(), dt=0.02)

    assert runtime.plan_calls == 1
    assert command.source == "world_model_cem"
    assert command.world_model_score == pytest.approx(1.25)
    assert command.safety_intervened  # acceleration envelope limits first command

    failing = ProstheticEyeController(
        world_model_runtime=_FakeWorldModelRuntime(fail=True),
        model_warmup_steps=0,
    )
    fallback = failing.control(_observation(), dt=0.02)
    assert fallback.source == "pid_fallback"
    assert "RuntimeError" in fallback.diagnostics["world_model_error"]


def test_generated_dataset_is_directly_consumable_by_world_model(tmp_path):
    output = tmp_path / "eye.jsonl"
    result = generate_prosthetic_eye_dataset(
        output,
        episodes=4,
        steps=18,
        seed=7,
        plant_config=EyePlantConfig(sensor_noise_std=0.0),
    )
    trajectories = load_world_model_trajectories(str(output))
    dimensions = infer_world_model_dimensions(trajectories)

    assert result["schema_version"] == EYE_CONTROL_SCHEMA_VERSION
    assert result["episodes"] == 4
    assert result["transitions"] == 72
    assert dimensions == {
        "observation_shape": (9,),
        "action_dim": 2,
        "action_type": "continuous",
    }
    assert trajectories[0]["metadata"]["expert"] == "pid_with_safety_envelope"


def test_simulation_report_is_json_safe_and_fault_free():
    result = simulate_prosthetic_eye_control(
        episodes=3,
        steps=80,
        seed=9,
        plant_config=EyePlantConfig(sensor_noise_std=0.0),
    )

    json.dumps(result)
    assert result["status"] == "completed"
    assert result["controller"] == "pid_fallback"
    assert result["hard_stops"] == 0
    assert result["mean_final_error_radians"] < 0.08


class _Adapter:
    def __init__(self, observation):
        self.observation = observation
        self.velocity = None
        self.fault = None

    def read_observation(self):
        return self.observation

    def write_velocity(self, yaw_radians_per_second, pitch_radians_per_second):
        self.velocity = (yaw_radians_per_second, pitch_radians_per_second)

    def emergency_stop(self, reason):
        self.fault = reason


def test_hardware_adapter_receives_only_safety_filtered_commands():
    adapter = _Adapter(_observation())
    loop = EyeControlLoop(
        ProstheticEyeController(model_warmup_steps=0), adapter, control_hz=50.0
    )
    command = loop.tick()

    assert not command.hard_stop
    assert adapter.velocity == command.velocity_command
    assert max(abs(value) for value in adapter.velocity) <= 0.2

    adapter.observation = _observation(temperature_c=80.0)
    stopped = loop.tick()
    assert stopped.hard_stop
    assert adapter.fault == "over_temperature"

    malformed = _Adapter([0.0] * 9)
    malformed_loop = EyeControlLoop(ProstheticEyeController(), malformed)
    with pytest.raises(EyeControlError, match="must return EyeObservation"):
        malformed_loop.tick()
    assert malformed.fault == "control_loop_exception:EyeControlError"


def test_eye_control_cli_builds_data_and_simulates(tmp_path):
    data_path = tmp_path / "cli-eye.jsonl"
    build_result_path = tmp_path / "build-result.json"
    assert (
        cli_main(
            [
                "build-eye-control-data",
                "--output",
                str(data_path),
                "--episodes",
                "2",
                "--steps",
                "6",
                "--result-output",
                str(build_result_path),
            ]
        )
        == 0
    )
    build_result = json.loads(build_result_path.read_text(encoding="utf-8"))
    assert build_result["transitions"] == 12
    assert data_path.is_file()

    simulation_path = tmp_path / "simulation.json"
    assert (
        cli_main(
            [
                "simulate-eye-control",
                "--episodes",
                "1",
                "--steps",
                "20",
                "--output",
                str(simulation_path),
            ]
        )
        == 0
    )
    simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
    assert simulation["steps"] == 20
    assert simulation["hard_stops"] == 0


def test_world_model_to_eye_control_pipeline_runs_end_to_end(tmp_path):
    data_path = tmp_path / "training-eye.jsonl"
    generate_prosthetic_eye_dataset(
        data_path,
        episodes=4,
        steps=10,
        seed=13,
        plant_config=EyePlantConfig(sensor_noise_std=0.0),
    )
    run_dir = tmp_path / "run"
    orchestrator = TrainingOrchestrator.from_dict(
        {
            "project": "eye-e2e",
            "stages": ["world_model", "eye_control"],
            "pipeline": {
                "dependencies": {
                    "world_model": [],
                    "eye_control": ["world_model"],
                }
            },
            "world_model": {
                "enabled": True,
                "backend": "categorical_rssm",
                "model": {
                    "action_type": "continuous",
                    "embedding_size": 16,
                    "deterministic_size": 16,
                    "stochastic_size": 4,
                    "stochastic_classes": 4,
                    "hidden_size": 16,
                    "recurrent_blocks": 4,
                    "reward_bins": 15,
                    "free_nats": 0.0,
                },
                "data": {
                    "train_path": str(data_path),
                    "sequence_length": 4,
                    "validation_split": 0.25,
                },
                "training": {
                    "output_dir": "world_model",
                    "num_epochs": 1,
                    "batch_size": 2,
                    "max_steps": 1,
                    "device": "cpu",
                    "mixed_precision": "no",
                    "checkpoint_steps": 0,
                },
            },
            "eye_control": {
                "enabled": True,
                "require_world_model": True,
                "output_path": "eye/report.json",
                "episodes": 1,
                "steps": 4,
                "device": "cpu",
                "plant": {"sensor_noise_std": 0.0},
                "planner": {
                    "horizon": 2,
                    "num_candidates": 4,
                    "iterations": 1,
                    "elite_fraction": 0.5,
                    "seed": 2,
                },
            },
            "distributed": {"strategy": "single", "bf16": False, "fp16": False},
            "eval": {"enabled": False},
            "logging": {"output_dir": str(run_dir), "backend": "local"},
        }
    )

    results = orchestrator.run()

    assert results["world_model"]["status"] == "completed"
    assert results["eye_control"]["status"] == "completed"
    assert (
        results["eye_control"]["checkpoint_path"]
        == results["world_model"]["output_dir"]
    )
    assert results["eye_control"]["command_sources"]["world_model_cem"] == 2
    assert results["eye_control"]["planner_mean_latency_ms"] is not None
    assert (run_dir / "eye" / "report.json").is_file()
    assert orchestrator.artifacts["eye_control.report"].kind == "control_report"
