# SaddleLLM Training Guide

This guide covers the stable training flow in this repository.

## 1. Check Environment

Verify dependencies, CUDA, GPU memory, RAM, and disk space:

```bash
python -m saddle_llm.cli doctor
python -m saddle_llm.cli doctor --output build/doctor_report.json
```

## 2. Run Smoke Tests

Run the built-in tiny-model end-to-end checks after changing training code:

```bash
python -m saddle_llm.cli smoke-test --work-dir build/smoke_e2e_auto
```

For faster local iteration, reuse the current Python process:

```bash
python -m saddle_llm.cli smoke-test --in-process --skip-doctor --work-dir build/smoke_e2e_fast
```

Use `--skip-training` for fixture, inspection, and config validation only. The
full smoke suite covers SFT, VLA SFT, DPO, ORPO, KTO, negative VLA action
dimension checks, and negative KTO batch-size checks.

## 3. Inspect Data

Run schema and quality checks before loading a model:

```bash
python -m saddle_llm.cli inspect-data data/sft.jsonl --task sft
python -m saddle_llm.cli inspect-data data/preference.jsonl --task dpo
python -m saddle_llm.cli inspect-vla data/robot.jsonl --image-root data/images --action-dim 7
```

The report includes readiness, schema counts, average prompt/completion length,
duplicate samples, and preference rows where `chosen == rejected`.
For VLA data, the report also checks action dimensions, action ranges, image
paths, and sampled episode step continuity.

## 4. Validate Config

Validate a recipe or compiled orchestrator config without starting training:

```bash
python -m saddle_llm.cli validate-config configs/sft_lora.yaml
python -m saddle_llm.cli validate-config configs/dpo_qlora.yaml --skip-data-inspection
```

The validation report includes blocking issues, warnings, local data inspection,
and deterministic training token estimates.

## 5. Preflight

Create a no-training plan with data checks and token-budget estimates:

```bash
python -m saddle_llm.cli preflight data/sft.jsonl --stage sft --root-dir ./llm_factory
python -m saddle_llm.cli preflight data/preference.jsonl --stage dpo --root-dir ./llm_factory
```

Preflight writes:

- `preflight_plan.json`
- `orchestrator_config.json`
- a recipe YAML

## 6. Use Templates

Starter configs are in `configs/`:

- `configs/preflight.yaml`
- `configs/sft_lora.yaml`
- `configs/dpo_qlora.yaml`
- `configs/vla_sft.yaml`
- `configs/mopd_sft.yaml`

Compile a template and create launch artifacts:

```bash
python -m saddle_llm.cli plan configs/sft_lora.yaml
```

Run training:

```bash
python -m saddle_llm.cli train configs/sft_lora.yaml
```

By default, training failures are reported as structured JSON. Use `--debug`
when you need the full Python traceback:

```bash
python -m saddle_llm.cli train configs/sft_lora.yaml --debug
```

Dry-run without training:

```bash
python -m saddle_llm.cli train configs/sft_lora.yaml --dry-run
```

## 7. Recommended Order

1. `doctor`
2. `smoke-test --skip-training`
3. `inspect-data`
4. `validate-config`
5. `preflight`
6. `plan`
7. `train`
8. `report`

For new datasets, keep `training.preflight_only: true` until the plan and data
inspection are clean.
