# Unified training pipelines and prosthetic-eye control

## One run, multiple model families

A training config can describe a dependency graph instead of only a linear
list. This makes it possible to train a language/policy branch and a world-model
branch independently, then join them at evaluation or release time.

```yaml
stages: [sft, world_model, preference, eval, export]
pipeline:
  dependencies:
    sft: []
    world_model: []
    preference: [sft]
    eval: [preference, world_model]
    export: [eval]
  resume: true
  rerun: []
```

When `pipeline.dependencies` is absent, the existing stage list remains a
strict linear pipeline. When it is present, omitted stages are graph roots and
the dependency mapping is authoritative. A stable topological sort uses the
original list only to break ties.

Every stage exposes a family and an artifact contract. Runtime outputs are
captured in one typed registry (`language_model`, `control_policy`,
`world_model`, `dataset`, `evaluation`, and `release`) and saved with producer
lineage in `pipeline_state.json` and the normal training summary.

Pipeline recovery is separate from trainer-checkpoint recovery:

- `pipeline.resume: true` skips whole stages that completed under the same
  normalized configuration and graph.
- `training.resume_from_checkpoint` restores optimizer/scheduler/RNG state
  inside a supported trainer stage.
- `pipeline.rerun: [world_model]` reruns that stage and automatically
  invalidates every graph descendant, while retaining independent completed
  branches.
- A configuration or graph fingerprint mismatch fails closed instead of
  silently reusing stale outputs.

See `configs/unified_model_posttrain_world.yaml` for a two-branch example.

## Recommended model for a two-axis mechanical eye

For a pan/tilt prosthetic or animatronic eye whose immediate goal is smooth
target fixation, do not put an LLM in the motor loop. Use a layered controller:

1. A hard real-time PID or constrained MPC inner loop tracks yaw/pitch velocity
   or position commands and enforces angle, speed, acceleration, current, and
   thermal limits.
2. A compact detector/keypoint model plus filtering estimates the target's
   normalized image error and confidence.
3. A learned outer loop predicts or selects short-horizon commands. The current
   SaddleLLM default is `categorical_rssm` plus CEM planning because it handles
   partial observability, continuous actions, latency, and multimodal outcomes
   while fitting the repository's native world-model runtime.

The starter observation vector is:

```text
[yaw, pitch, yaw_velocity, pitch_velocity,
 target_yaw_error, target_pitch_error, target_visible,
 yaw_current, pitch_current]
```

The action is a normalized two-dimensional yaw/pitch velocity command. A useful
reward starts with target-centering error, then penalizes energy, acceleration,
jerk, limit proximity, target loss, and unsafe current:

```text
r = -w1 * image_error
    -w2 * action_squared
    -w3 * jerk
    -w4 * limit_or_current_violation
    +w5 * stable_fixation
```

The repository now includes a hardware-neutral implementation in
`saddlellm/prosthetic_eye_control.py`:

- `ProstheticEyeSimulator` provides a deterministic two-axis actuator plant for
  safe offline iteration.
- `generate_prosthetic_eye_dataset` records PID-expert trajectories in the
  native world-model JSONL schema.
- `ProstheticEyeController` maintains observation/action history, filters a
  latent RSSM belief, requests CEM actions, and fails closed to PID.
- `EyeSafetyEnvelope` owns hard limits, soft-limit slowdown, acceleration
  limiting, current derating, over-current/over-temperature fault latching,
  and stale-target behavior independently of the learned model.
- `EyeControlLoop.tick()` is the only hardware bridge. A device integration
  must explicitly implement `EyeHardwareAdapter`; the library never opens a
  serial/CAN/PWM device on its own.

Generate a larger simulated dataset, train the model, and run the closed-loop
simulator as follows:

```powershell
saddle-llm build-eye-control-data `
  --config configs/prosthetic_eye_runtime.yaml `
  --output data/prosthetic_eye_trajectories.jsonl `
  --episodes 256 --steps 150

# Point world_model.data.train_path at the generated file before production training.
saddle-llm train configs/prosthetic_eye_control.yaml

saddle-llm simulate-eye-control `
  --config configs/prosthetic_eye_runtime.yaml `
  --checkpoint outputs/prosthetic-eye-control/world_model `
  --episodes 20 --steps 250 `
  --output outputs/prosthetic-eye-control/heldout-simulation.json
```

The bundled pipeline is now explicitly `world_model -> eye_control`: after
training, the checkpoint is passed to a closed-loop simulation stage and a
typed `control_report` artifact is published. The report includes tracking
error, success rate, safety interventions, hard stops, planner latency, and
control-deadline miss rate. For a fast structural check:

```powershell
saddle-llm train configs/prosthetic_eye_control.yaml --dry-run
```

The four bundled episodes only validate the data and training path; they are
not enough to learn a deployable policy. Use simulation data for bootstrapping,
then recalibrate the plant and fine-tune on logged real-device trajectories.

For an eye that must actively search a panoramic scene to help a robot perform
a task, the closer target architecture is EyeRobot: a foveated vision
transformer with a behavior-cloning/reinforcement-learning loop. Its eye policy
is rewarded through downstream task performance rather than gaze labels. For
demonstration-rich high-frequency control, ACT-style action chunks are a good
policy baseline. For a future control-centric world-model backend that does not
need pixel reconstruction, TD-MPC2 is the strongest fit to add next.

Primary references:

- [EyeRobot project and CoRL 2025 paper](https://www.eyerobot.net/)
- [DreamerV3 paper](https://arxiv.org/abs/2301.04104)
- [TD-MPC2 project](https://www.tdmpc2.com/)
- [ACT / ALOHA project](https://tonyzhaozh.github.io/aloha/)

## Real-device rollout gate

Before any learned policy can drive hardware, evaluate held-out trajectories
and hardware-in-the-loop scenarios for target error, overshoot, settling time,
command jerk, target-loss recovery, saturation time, and worst-case motor
current. Keep the safety controller outside the learned model and default to a
neutral pose on stale frames, low confidence, process failure, or watchdog
timeout. If the mechanism interfaces with a person, treat this as a medical and
functional-safety system; the example configuration is a research baseline,
not a clinical controller.
