import json
import threading
import time

from saddlellm.distillation.OnPolicyDistillation import (
    MOPDConfig,
    MultiTeacherOnPolicyDistiller,
    OnPolicyTeacherSpec,
)
from saddlellm.distillation.UniversalDistiller import TeacherInterface


def test_endpoint_teacher_uses_openai_compatible_contract(monkeypatch) -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {"choices": [{"message": {"content": "reviewed answer"}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    teacher = TeacherInterface(
        model_type="endpoint",
        model_name="nursing-teacher",
        endpoint_url="http://127.0.0.1:8001",
        api_key="secret",
    )

    answer = teacher.generate("question", temperature=0.2, max_tokens=32)

    assert answer == "reviewed answer"
    assert captured["url"] == "http://127.0.0.1:8001/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "nursing-teacher"
    assert captured["body"]["messages"] == [{"role": "user", "content": "question"}]
    assert captured["body"]["stream"] is False


def test_on_policy_teacher_requests_run_concurrently_and_keep_order() -> None:
    lock = threading.Lock()

    class Teacher:
        active = 0
        max_active = 0

        def generate(self, prompt, **_kwargs):
            with lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with lock:
                self.active -= 1
            original = prompt.split("Prompt:\n", 1)[1].split("\n\n", 1)[0]
            return f"answer-{original}"

    teacher = Teacher()
    distiller = MultiTeacherOnPolicyDistiller(
        teachers=[OnPolicyTeacherSpec(name="teacher", teacher=teacher)],
        rollout_fn=lambda prompts, _params: ["draft" for _ in prompts],
        config=MOPDConfig(teacher_concurrency=4),
    )
    prompts = ["q1", "q2", "q3", "q4"]

    records = distiller.collect(prompts)

    assert teacher.max_active > 1
    assert [record["instruction"] for record in records] == prompts
    assert [record["output"] for record in records] == [
        "answer-q1",
        "answer-q2",
        "answer-q3",
        "answer-q4",
    ]
