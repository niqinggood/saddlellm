from __future__ import annotations

import asyncio
from copy import deepcopy
import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from saddlellm.decision.Server import DecisionServerSettings, create_decision_app


CHOICES = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}]


class FakeDecisionModel:
    def __init__(self):
        self.metadata = {"question": "Which option?", "choices": deepcopy(CHOICES)}
        self.calibration = {"calibrated": True, "metadata": {"scope": deepcopy(self.metadata)}}
        self.calibration_temperature = 2.0
        self.choice_calls = 0
        self.state_calls = 0
        self.state_threads = []
        self.active = 0
        self.max_active = 0
        self.invalid_logits = False

    def encode_choices(self, descriptions, question, *, device):
        self.choice_calls += 1
        return torch.tensor([[1.0, 0.0] if text.startswith("alpha") else [0.0, 1.0] for text in descriptions])

    def encode_states(self, states, questions, *, device):
        self.state_calls += 1
        self.state_threads.append(threading.get_ident())
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        time.sleep(0.01)
        self.active -= 1
        return torch.tensor([[1.0, 0.0] for _ in states])

    def score(self, features, candidate_features):
        result = 2 * features @ candidate_features.T
        return result * float("nan") if self.invalid_logits else result


def make_app(model=None, **settings):
    return create_decision_app(DecisionServerSettings(model_path="artifact", **settings), model=model or FakeDecisionModel())


def test_defaults_batch_and_calibrated_probabilities():
    model = FakeDecisionModel()
    client = TestClient(make_app(model))
    assert client.get("/health").json()["supported_types"] == ["Choice"]
    assert client.get("/").status_code == 200
    response = client.post("/v1/decisions", json={"states": ["first state", "second state"]})
    assert response.status_code == 200
    data = response.json()
    assert data["calibration_status"] == "calibrated"
    assert data["temperature_used"] == 2.0
    assert len(data["data"]) == 2
    assert model.state_calls == model.choice_calls == 1
    assert all(row["choice_id"] == "a" for row in data["data"])
    probabilities = data["data"][0]["probabilities"]
    assert sum(item["probability"] for item in probabilities) == pytest.approx(1.0)
    assert probabilities[0]["probability"] == pytest.approx(0.7310585786)


def test_candidate_order_preserves_scope_cache_and_id_mapping():
    model = FakeDecisionModel()
    client = TestClient(make_app(model))
    first = client.post("/v1/decisions", json={"state": "x"}).json()
    second = client.post("/v1/decisions", json={"state": "x", "choices": list(reversed(CHOICES))}).json()
    assert second["calibration_status"] == "calibrated"
    assert second["data"][0]["choice_id"] == first["data"][0]["choice_id"]
    assert second["data"][0]["probabilities"] == list(reversed(first["data"][0]["probabilities"]))
    assert model.choice_calls == 1 and model.state_calls == 2


@pytest.mark.parametrize("change", [
    {"question": "Different criterion?"},
    {"choices": [{"id": "a", "text": "alpha changed"}, CHOICES[1]]},
    {"choices": [{"id": "other", "text": "alpha"}, CHOICES[1]]},
    {"choices": CHOICES + [{"id": "c", "text": "gamma"}]},
])
def test_dynamic_task_never_reuses_fitted_temperature(change):
    client = TestClient(make_app())
    data = client.post("/v1/decisions", json={"state": "x", **change}).json()
    assert data["calibration_status"] == "out_of_scope"
    assert data["temperature_used"] == 1.0


def test_uncalibrated_or_missing_scope_never_claims_calibration():
    model = FakeDecisionModel()
    model.calibration["metadata"] = {}
    data = TestClient(make_app(model)).post("/v1/decisions", json={"state": "x"}).json()
    assert data["calibration_status"] == "out_of_scope" and data["temperature_used"] == 1.0
    model.calibration["calibrated"] = False
    data = TestClient(make_app(model)).post("/v1/decisions", json={"state": "x"}).json()
    assert data["calibration_status"] == "uncalibrated" and data["temperature_used"] == 1.0


@pytest.mark.parametrize("payload", [
    {}, {"state": ""}, {"state": "   "}, {"state": 123}, {"states": []},
    {"states": ["x"] * 17}, {"state": "x", "states": ["y"]},
    {"state": "x", "type": "Number"}, {"state": "x", "type": []},
    {"state": "x", "question": ""}, {"state": "x", "temperature": 2},
    {"state": "x", "choices": CHOICES[:1]},
    {"state": "x", "choices": [CHOICES[0], CHOICES[0]]},
    {"state": "x", "choices": [{"id": "a", "text": "alpha", "score": 1}, CHOICES[1]]},
    {"state": "x", "choices": [{"id": str(i), "text": "x"} for i in range(101)]},
])
def test_request_validation_rejects_invalid_input_before_encoding(payload):
    model = FakeDecisionModel()
    response = TestClient(make_app(model)).post("/v1/decisions", json=payload)
    assert response.status_code == 422
    assert model.state_calls == model.choice_calls == 0


def test_nonfinite_request_and_model_values_are_rejected():
    client = TestClient(make_app())
    response = client.post("/v1/decisions", content='{"state":"x","temperature":NaN}', headers={"content-type": "application/json"})
    assert response.status_code == 422
    model = FakeDecisionModel()
    model.invalid_logits = True
    assert TestClient(make_app(model)).post("/v1/decisions", json={"state": "x"}).status_code == 500
    model.calibration_temperature = float("inf")
    with pytest.raises(ValueError, match="finite"):
        make_app(model)


def test_authentication_limits_and_optional_web():
    client = TestClient(make_app(api_key="test-secret", max_state_chars=4, max_body_bytes=100, web_ui=False))
    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200
    assert client.get("/v1/decisions/schema").status_code == 401
    assert client.post("/v1/decisions", json={"state": "x"}).status_code == 401
    headers = {"Authorization": "Bearer test-secret"}
    assert client.post("/v1/decisions", json={"state": "12345"}, headers=headers).status_code == 422
    assert client.post("/v1/decisions", json={"state": "x" * 101}, headers=headers).status_code == 413
    assert client.post("/v1/decisions", json={"state": "x"}, headers=headers).status_code == 200


def test_compute_runs_off_event_loop_and_serializes_shared_encoder():
    import httpx

    model = FakeDecisionModel()
    app = make_app(model)

    async def send():
        event_thread = threading.get_ident()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(*(client.post("/v1/decisions", json={"state": f"state{i}"}) for i in range(4)))
        assert all(response.status_code == 200 for response in responses)
        assert all(thread != event_thread for thread in model.state_threads)

    asyncio.run(send())
    assert model.state_calls == 4 and model.choice_calls == 1
    assert model.max_active == 1


def test_candidate_cache_is_bounded_and_scoped_to_engine_instance():
    model = FakeDecisionModel()
    client = TestClient(make_app(model, candidate_cache_size=1))
    for question in ["first", "second", "first"]:
        assert client.post("/v1/decisions", json={"state": "x", "question": question}).status_code == 200
    assert model.choice_calls == 3
    assert TestClient(make_app(model)).post("/v1/decisions", json={"state": "x", "question": "first"}).status_code == 200
    assert model.choice_calls == 4


@pytest.mark.parametrize("kwargs", [{"max_batch": 17}, {"max_candidates": 101}, {"max_batch": True}, {"max_state_chars": float("nan")}, {"candidate_cache_size": 0}])
def test_settings_enforce_hard_bounds(kwargs):
    with pytest.raises(ValueError):
        DecisionServerSettings(model_path="artifact", **kwargs)
