import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from saddlellm import cli
from saddlellm.runtime.BackendConfig import (
    RuntimeBackendConfig,
    SGLangBackendConfig,
    validate_runtime_backends,
)
from saddlellm.runtime.RolloutBackend import (
    SGLangRolloutBackend,
    create_rollout_backend,
)
from saddlellm.training.MegatronBackend import (
    MegatronTrainingBackend,
    prepare_sft_jsonl,
)
from saddlellm.training.MegatronWorker import apply_recipe_overrides, run_job
from saddlellm.training.TrainingConfigValidator import TrainingConfigValidator
from saddlellm.training.TrainingOrchestrator import TrainingOrchestrator

RECIPE = "megatron.bridge.recipes.llama:llama32_1b_pretrain_config"


def _megatron_config(tmp_path, stages=None):
    return {
        "stages": stages or ["pretrain"],
        "backends": {"training": "megatron", "megatron": {"recipe": RECIPE}},
        "logging": {"output_dir": str(tmp_path / "run")},
        "eval": {"enabled": False},
    }


def _body(text, version="v1"):
    return {
        "text": text,
        "output_ids": [7, 8],
        "meta_info": {
            "weight_version": version,
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.5, 7, "a"], [-1.0, 8, "b"]],
        },
    }


@pytest.mark.parametrize(
    "block",
    [
        {"training": "misspelled"},
        {"rollout": "vllm"},
        {"oops": True},
        {"sglang": {"mode": "oops"}},
        {"sglang": {"concurrency": True}},
        {"sglang": {"timeout_seconds": float("nan")}},
        {"sglang": {"base_url": "http://secret:password@localhost"}},
        {"sglang": {"base_url": "http://localhost/v1"}},
        {"sglang": {"return_logprob": "false"}},
        {"sglang": {"engine_kwargs": {"model_path": "unexpected"}}},
        {"megatron": {"recipe": "os:system"}},
        {"megatron": {"nnodes": 2}},
        {"megatron": {"node_rank": -1}},
        {"megatron": {"recipes": {"dpo": RECIPE}}},
        {"megatron": {"recipe": False}},
        {"sglang": {"api_key_env": 123}},
        {"sglang": {"model_path": ""}},
        {"sglang": {"engine_kwargs": {"admin_api_key": "secret"}}},
    ],
)
def test_invalid_backend_configuration_is_rejected(block):
    with pytest.raises((ValueError, TypeError)):
        RuntimeBackendConfig.from_dict(block)


def test_optional_backends_and_public_imports_are_lazy():
    source = (
        "import sys; "
        "from saddlellm import RuntimeBackendConfig, create_rollout_backend; "
        "b=create_rollout_backend({'rollout':'sglang'}); b.close(); "
        "assert 'sglang' not in sys.modules; "
        "assert 'megatron.bridge' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_native_rollout_returns_only_generated_ids_and_restores_training():
    import torch
    from transformers import BatchEncoding

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(max_position_embeddings=16)

        def generate(self, input_ids, **kwargs):
            assert not self.training
            assert "temperature" not in kwargs
            assert kwargs["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([[6, 7, 8]])], dim=1)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 3

        def __call__(self, prompt, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[4, 5]])})

        def decode(self, ids, **kwargs):
            return "".join(
                {4: "prompt", 5: "part", 6: "reply", 7: "END", 8: "tail"}[i]
                for i in ids
            )

    model = Model().train()
    with create_rollout_backend({}, model=model, tokenizer=Tokenizer()) as backend:
        result = backend.generate(
            ["prompt"], {"max_new_tokens": 3, "temperature": 0, "stop": "END"}
        )
        assert result[0].text == "reply"
        assert result[0].output_ids == [6, 7, 8]
        assert result[0].backend == "native"
        assert model.training
        with pytest.raises(ValueError, match="context"):
            backend.generate(["prompt"], {"max_new_tokens": 20})
        assert model.training


def test_http_sampling_is_bounded_ordered_and_versioned():
    active = maximum = 0

    async def handle(request):
        nonlocal active, maximum
        assert request.url.path == "/generate"
        payload = json.loads(request.content)
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.002 * (8 - int(payload["text"])))
            return httpx.Response(200, json=_body(payload["text"]))
        finally:
            active -= 1

    backend = create_rollout_backend(
        {
            "rollout": "sglang",
            "sglang": {
                "policy_version": "v1",
                "concurrency": 3,
                "return_logprob": True,
            },
        },
        http_transport=httpx.MockTransport(handle),
    )
    samples = asyncio.run(
        backend.agenerate([str(i) for i in range(8)], {"max_new_tokens": 2})
    )
    assert [sample.text for sample in samples] == [str(i) for i in range(8)]
    assert maximum == 3 and active == 0
    assert len({sample.request_id for sample in samples}) == 8
    assert all(sample.policy_version == "v1" for sample in samples)
    assert all(
        sample.output_ids == [7, 8] and sample.output_logprobs == [-0.5, -1.0]
        for sample in samples
    )
    assert all(sample.logprob_semantics == "backend_reported" for sample in samples)
    backend.close()
    backend.close()


@pytest.mark.parametrize(
    "body",
    [
        _body("wrong", "old"),
        {"text": "missing-version"},
        {"text": 7},
        {"text": "bad", "meta_info": []},
    ],
)
def test_invalid_or_stale_http_responses_fail_without_fallback(body):
    requests = []

    def handle(request):
        requests.append(request.url.path)
        return httpx.Response(200, json=body)

    with SGLangRolloutBackend(
        SGLangBackendConfig(policy_version="v1"),
        http_transport=httpx.MockTransport(handle),
    ) as backend:
        with pytest.raises(RuntimeError):
            backend.generate(["prompt"], {"max_new_tokens": 1})
    assert "/abort_request" in requests


def test_http_error_is_redacted_and_aborts_only_its_request(monkeypatch):
    monkeypatch.setenv("TEST_SGLANG_KEY", "private-secret")
    aborts = []

    def handle(request):
        assert request.headers["Authorization"] == "Bearer private-secret"
        if request.url.path == "/abort_request":
            aborts.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.Response(500, text="private-secret private-prompt")

    with SGLangRolloutBackend(
        SGLangBackendConfig(api_key_env="TEST_SGLANG_KEY"),
        http_transport=httpx.MockTransport(handle),
    ) as backend:
        with pytest.raises(RuntimeError) as caught:
            backend.generate(["private-prompt"], {})
    assert "private-secret" not in str(caught.value)
    assert "private-prompt" not in str(caught.value)
    assert len(aborts) == 1 and "rid" in aborts[0] and not aborts[0].get("abort_all")


def test_http_cancellation_aborts_and_releases_owner():
    aborts = []

    async def scenario():
        started = asyncio.Event()

        async def handle(request):
            if request.url.path == "/abort_request":
                aborts.append(json.loads(request.content)["rid"])
                return httpx.Response(200)
            started.set()
            await asyncio.Future()

        backend = SGLangRolloutBackend(
            SGLangBackendConfig(), http_transport=httpx.MockTransport(handle)
        )
        task = asyncio.create_task(backend.agenerate(["prompt"], {}))
        await started.wait()
        with pytest.raises(RuntimeError, match="busy"):
            backend.close()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        backend.close()
        assert backend._closed

    asyncio.run(scenario())
    assert len(aborts) == 1


def test_policy_update_is_a_barrier_and_failure_blocks_sampling():
    published = "v0"
    allow_update = False

    def handle(request):
        nonlocal published
        payload = json.loads(request.content)
        if request.url.path == "/update_weights_from_disk":
            assert (
                payload["abort_all_requests"] is False and payload["is_async"] is False
            )
            if allow_update:
                published = payload["weight_version"]
            return httpx.Response(200, json={"success": allow_update})
        return httpx.Response(200, json=_body("answer", published))

    backend = SGLangRolloutBackend(
        SGLangBackendConfig(policy_version="v1"),
        http_transport=httpx.MockTransport(handle),
    )
    with pytest.raises(RuntimeError, match="rejected"):
        backend.prepare_policy("/new/model", "v2")
    with pytest.raises(RuntimeError, match="restore policy"):
        backend.generate(["prompt"], {})
    allow_update = True
    backend.prepare_policy("/new/model", "v2")
    assert backend.generate(["prompt"], {})[0].policy_version == "v2"
    backend.close()


def test_pending_limit_and_sync_in_async_are_clear_errors():
    with SGLangRolloutBackend(
        SGLangBackendConfig(concurrency=1, max_pending_requests=1)
    ) as backend:
        with pytest.raises(ValueError, match="max_pending_requests"):
            backend.generate(["one", "two"], {})

        async def scenario():
            with pytest.raises(RuntimeError, match="agenerate"):
                backend.generate(["one"], {})

        asyncio.run(scenario())


def test_local_engine_is_lazy_batched_replaced_and_closed():
    engines = []

    class Engine:
        def __init__(self, **kwargs):
            self.kwargs, self.closed = kwargs, False
            engines.append(self)

        def generate(self, prompt, **kwargs):
            return [_body(item, self.kwargs["weight_version"]) for item in prompt]

        def shutdown(self):
            self.closed = True

    backend = create_rollout_backend(
        {
            "rollout": "sglang",
            "sglang": {
                "mode": "engine",
                "model_path": "/base",
                "policy_version": "v1",
                "engine_kwargs": {"tp_size": 2},
            },
        },
        engine_factory=Engine,
    )
    assert not engines
    assert len(backend.generate(["one", "two"], {})) == 2
    assert engines[0].kwargs["tp_size"] == 2
    backend.prepare_policy("/next", "v2")
    assert engines[0].closed
    assert asyncio.run(backend.agenerate(["three"], {}))[0].policy_version == "v2"
    backend.close()
    assert engines[1].closed


def test_local_cancel_waits_for_running_engine_step():
    started, finish = threading.Event(), threading.Event()

    class Engine:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            started.set()
            assert finish.wait(timeout=5)
            return [_body("answer")]

        def shutdown(self):
            assert finish.is_set()

    async def scenario():
        backend = SGLangRolloutBackend(
            SGLangBackendConfig(mode="engine", model_path="/base"),
            engine_factory=Engine,
        )
        task = asyncio.create_task(backend.agenerate(["one"], {}))
        await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        backend.close()

    asyncio.run(scenario())


def test_megatron_configuration_and_cli_dry_run_do_not_load_runtime(tmp_path, capsys):
    raw = _megatron_config(tmp_path)
    validation = TrainingConfigValidator.validate(raw)
    assert validation.valid, validation.issues
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    assert cli.main(["train", str(config_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backends"]["training"] == "megatron"
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        {"stages": ["preference"]},
        {"distributed": {"strategy": "ddp"}},
        {"model": {"backend": "saddle"}},
        {"training": {"resume_from_checkpoint": True}},
    ],
)
def test_unsupported_megatron_combinations_fail_validation(tmp_path, mutation):
    raw = {**_megatron_config(tmp_path), **mutation}
    assert not TrainingConfigValidator.validate(raw).valid
    with pytest.raises(ValueError):
        validate_runtime_backends(TrainingOrchestrator._parse_config(raw))


def test_megatron_downstream_requires_hf_export(tmp_path):
    raw = _megatron_config(tmp_path, ["pretrain", "mopd"])
    with pytest.raises(ValueError, match="hf_export_source"):
        validate_runtime_backends(TrainingOrchestrator._parse_config(raw))


def _fake_bridge(monkeypatch):
    calls = []

    def recipe():
        return SimpleNamespace(
            checkpoint=SimpleNamespace(
                save=None, load="recipe-old-load", pretrained_checkpoint=None
            ),
            model=SimpleNamespace(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
            ),
            train=SimpleNamespace(
                global_batch_size=1, micro_batch_size=1, train_iters=1
            ),
            dataset=SimpleNamespace(seq_length=32),
        )

    def train(config, **kwargs):
        calls.append(config)
        path = Path(config.checkpoint.save)
        path.mkdir(parents=True)
        (path / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")

    modules = {
        "megatron.bridge.recipes.llama": SimpleNamespace(
            llama32_1b_pretrain_config=recipe
        ),
        "megatron.bridge.training.pretrain": SimpleNamespace(pretrain=train),
        "megatron.bridge.training.finetune": SimpleNamespace(finetune=train),
        "megatron.bridge.training.gpt_step": SimpleNamespace(
            forward_step=lambda *a: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_megatron_worker_executes_entry_and_publishes_verified_manifest(
    tmp_path, monkeypatch
):
    calls = _fake_bridge(monkeypatch)
    config = RuntimeBackendConfig.from_dict(
        {
            "training": "megatron",
            "megatron": {
                "recipe": RECIPE,
                "overrides": {"train": {"train_iters": 2}},
            },
        }
    )
    backend = MegatronTrainingBackend(config.megatron)
    monkeypatch.setattr(backend, "_check_runtime", lambda: None)

    def execute(plan):
        assert plan.command[-2] == "saddlellm.training.MegatronWorker"
        run_job(json.loads(Path(plan.command[-1]).read_text(encoding="utf-8")))

    monkeypatch.setattr(backend, "_execute", execute)
    result = backend.run("pretrain", tmp_path / "worker")
    assert result["status"] == "completed" and "model_path" not in result
    assert calls[0].train.train_iters == 2 and calls[0].checkpoint.load is None
    assert result["checkpoint_format"] == "megatron"


def test_megatron_dry_run_and_failed_launch_do_not_claim_success(tmp_path, monkeypatch):
    backend = MegatronTrainingBackend(
        RuntimeBackendConfig.from_dict({"megatron": {"recipe": RECIPE}}).megatron
    )
    monkeypatch.setattr(
        backend, "_check_runtime", lambda: pytest.fail("runtime loaded in dry-run")
    )
    assert (
        backend.run("pretrain", tmp_path / "planned", dry_run=True)["status"]
        == "planned"
    )
    monkeypatch.setattr(backend, "_check_runtime", lambda: None)
    monkeypatch.setattr(backend, "_execute", lambda _: None)
    with pytest.raises(RuntimeError, match="without a completed"):
        backend.run("pretrain", tmp_path / "failed")


def test_megatron_overrides_are_strict_and_topology_is_explicit(tmp_path):
    config = SimpleNamespace(train=SimpleNamespace(train_iters=2))
    with pytest.raises(ValueError, match="unknown"):
        apply_recipe_overrides(config, {"train": {"train_iterz": 3}})
    with pytest.raises(ValueError, match="invalid"):
        apply_recipe_overrides(config, {"__dict__": {}})
    megatron = RuntimeBackendConfig.from_dict(
        {
            "megatron": {
                "recipe": RECIPE,
                "nnodes": 2,
                "nproc_per_node": 4,
                "node_rank": 1,
                "master_addr": "10.0.0.2",
                "run_id": "shared-unique-test",
            }
        }
    ).megatron
    plan = MegatronTrainingBackend(megatron).launch_plan(tmp_path / "space path.json")
    assert plan.topology["world_size"] == 8
    assert "--node-rank" in plan.command and "--standalone" not in plan.command
    assert plan.command[-1].endswith("space path.json")


def test_megatron_sft_data_conversion_is_atomic_and_preserves_input(tmp_path):
    source, destination = tmp_path / "sft.jsonl", tmp_path / "training.jsonl"
    source.write_text(
        '{"instruction":"question","input":"context","output":"answer"}\n',
        encoding="utf-8",
    )
    assert prepare_sft_jsonl(source, destination) == 1
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "input": "question\ncontext",
        "output": "answer",
    }
    source.write_text('{"messages":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_sft_jsonl(source, destination)
    assert json.loads(destination.read_text(encoding="utf-8"))["output"] == "answer"
    assert not list(tmp_path.glob("*.tmp"))


def test_megatron_hf_export_is_explicit_and_checked(tmp_path, monkeypatch):
    _fake_bridge(monkeypatch)

    class Bridge:
        def export_ckpt(self, source, destination, **kwargs):
            assert kwargs["strict"] and kwargs["source_path"] == "/hf/source"
            path = Path(destination)
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "model.safetensors").write_bytes(b"mock weights")

    monkeypatch.setitem(
        sys.modules,
        "megatron.bridge.models",
        SimpleNamespace(
            AutoBridge=SimpleNamespace(from_hf_pretrained=lambda _: Bridge()),
        ),
    )
    config = RuntimeBackendConfig.from_dict(
        {
            "megatron": {
                "recipe": RECIPE,
                "hf_export_source": "/hf/source",
            }
        }
    )
    backend = MegatronTrainingBackend(config.megatron)
    monkeypatch.setattr(backend, "_check_runtime", lambda: None)
    monkeypatch.setattr(
        backend,
        "_execute",
        lambda plan: run_job(
            json.loads(Path(plan.command[-1]).read_text(encoding="utf-8"))
        ),
    )
    result = backend.run("pretrain", tmp_path / "exported")
    assert result["model_format"] == "hf"
    assert (Path(result["model_path"]) / "config.json").exists()


def test_multi_node_jobs_share_manifest_identity_not_launch_plan(tmp_path):
    raw = {
        "recipe": RECIPE,
        "nnodes": 2,
        "master_addr": "10.0.0.1",
        "run_id": "unique-job",
    }
    one = MegatronTrainingBackend(
        RuntimeBackendConfig.from_dict({"megatron": raw}).megatron
    )
    two = MegatronTrainingBackend(
        RuntimeBackendConfig.from_dict({"megatron": {**raw, "node_rank": 1}}).megatron
    )
    assert one.make_job("pretrain", tmp_path) == two.make_job("pretrain", tmp_path)
    assert (
        one.launch_plan(tmp_path / "job.json").command
        != two.launch_plan(tmp_path / "job.json").command
    )


def test_megatron_timeout_kills_worker_group(tmp_path, monkeypatch):
    from saddlellm.training import MegatronBackend

    calls, signals = [], []

    class Process:
        pid = 173

        def wait(self, timeout=None):
            calls.append(timeout)
            if len(calls) < 3:
                raise subprocess.TimeoutExpired("mock", timeout)
            return 0

    monkeypatch.setattr(MegatronBackend.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(MegatronBackend.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        MegatronBackend.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )
    backend = MegatronTrainingBackend(
        RuntimeBackendConfig.from_dict(
            {
                "megatron": {
                    "recipe": RECIPE,
                    "timeout_seconds": 1,
                }
            }
        ).megatron
    )
    with pytest.raises(subprocess.TimeoutExpired):
        backend._execute(backend.launch_plan(tmp_path / "job.json"))
    assert calls == [1, 5, None]
    assert [item[0] for item in signals] == [173, 173]


@pytest.mark.parametrize(
    "params",
    [
        {"top_k": True},
        {"min_new_tokens": 3, "max_new_tokens": 2},
        {"stop": [7]},
        {"max_new_tokens": 0},
    ],
)
def test_sampling_parameters_fail_before_network(params):
    with SGLangRolloutBackend(SGLangBackendConfig()) as backend:
        with pytest.raises(ValueError):
            backend.generate(["prompt"], params)


def test_http_async_policy_publish_and_aborted_generation():
    def handle(request):
        if request.url.path == "/update_weights_from_disk":
            return httpx.Response(200, json={"success": True})
        body = _body("", "v2")
        body["meta_info"]["finish_reason"] = {"type": "abort"}
        return httpx.Response(200, json=body)

    async def scenario():
        backend = SGLangRolloutBackend(
            SGLangBackendConfig(), http_transport=httpx.MockTransport(handle)
        )
        with pytest.raises(RuntimeError, match="aprepare_policy"):
            backend.prepare_policy("/weights", "v2")
        await backend.aprepare_policy("/weights", "v2")
        with pytest.raises(RuntimeError, match="aborted"):
            await backend.agenerate(["prompt"], {})
        backend.close()

    asyncio.run(scenario())


def test_orchestrator_dispatches_megatron_instead_of_native_trainer(
    tmp_path, monkeypatch
):
    raw = _megatron_config(tmp_path)
    received = []

    def run(self, stage, output_dir, **kwargs):
        received.append((stage, kwargs))
        return {"status": "planned", "backend": "megatron"}

    monkeypatch.setattr(MegatronTrainingBackend, "run", run)
    runner = TrainingOrchestrator.from_dict(raw)
    try:
        runner.run_stage("pretrain")
        assert runner._stage_results["pretrain"]["backend"] == "megatron"
        assert received[0][0] == "pretrain"
    finally:
        runner._close_logging_handlers()


def test_sglang_mopd_avoids_loading_student_and_saves_backend_provenance(
    tmp_path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"one"}\n{"prompt":"two"}\n', encoding="utf-8")
    routes = []

    def handle(request):
        routes.append(request.url.path)
        if request.url.path == "/update_weights_from_disk":
            return httpx.Response(200, json={"success": True})
        return httpx.Response(200, json=_body("student"))

    from saddlellm.runtime import RolloutBackend

    backend = SGLangRolloutBackend(
        SGLangBackendConfig(policy_version="v1"),
        http_transport=httpx.MockTransport(handle),
    )
    monkeypatch.setattr(RolloutBackend, "create_rollout_backend", lambda _: backend)
    raw = {
        "stages": ["mopd"],
        "model": {"name_or_path": "/student"},
        "logging": {"output_dir": str(tmp_path / "run")},
        "backends": {"rollout": "sglang", "sglang": {"policy_version": "v1"}},
        "mopd": {
            "enabled": True,
            "dry_run": False,
            "prompts_path": str(prompts),
            "teachers": [{"type": "endpoint", "model": "teacher"}],
        },
    }
    runner = TrainingOrchestrator.from_dict(raw)
    monkeypatch.setattr(
        runner,
        "_ensure_policy_model_loaded",
        lambda: pytest.fail("local student loaded"),
    )
    monkeypatch.setattr(
        runner,
        "_build_mopd_teacher",
        lambda _: SimpleNamespace(generate=lambda *a, **kw: "improved"),
    )
    try:
        runner.run_stage("mopd")
        result = runner._stage_results["mopd"]
        rows = [
            json.loads(line)
            for line in Path(result["data_path"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert result["rollout_backend"] == "sglang" and len(rows) == 2
        assert rows[0]["student_rollout"]["output_ids"] == [7, 8]
        assert rows[0]["metadata"]["policy_version"] == "v1"
        assert routes[0] == "/update_weights_from_disk"
        assert backend._closed
    finally:
        runner._close_logging_handlers()
