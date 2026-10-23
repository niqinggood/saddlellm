import pytest

from saddlellm.runtime.StreamingText import StreamingTextBuffer


@pytest.mark.parametrize("stop", [None, "", [], [""]])
def test_without_stops_emits_immediately(stop):
    buffer = StreamingTextBuffer(stop)
    assert buffer.push("你好") == "你好"
    assert buffer.push(" world") == " world"
    assert buffer.pending == ""
    assert buffer.finish() == ""


@pytest.mark.parametrize("split", range(1, 15))
def test_stop_spanning_any_chunk_boundary_is_never_leaked(split):
    text = "hello<STOP>ignored"
    buffer = StreamingTextBuffer("<STOP>")
    actual = buffer.push(text[:split]) + buffer.push(text[split:]) + buffer.finish()
    assert actual == "hello"
    assert buffer.stopped
    assert buffer.push("later text") == ""


def test_multiple_unicode_stops_and_earliest_match():
    buffer = StreamingTextBuffer(["结束", "<STOP>"])
    assert buffer.push("你好 结") == ""
    assert buffer.push("束 trailing<STOP>") == "你好 "
    assert buffer.finish() == ""


def test_incomplete_stop_prefix_is_flushed_at_natural_end():
    buffer = StreamingTextBuffer("<STOP>")
    text = buffer.push("hello<ST") + buffer.finish()
    assert text == "hello<ST"
    assert not buffer.stopped
    assert buffer.finish() == ""


def test_pending_storage_is_bounded_by_stop_length_not_response_length():
    buffer = StreamingTextBuffer("<STOP>")
    output = []
    for _ in range(1000):
        output.append(buffer.push("abcdefghij"))
        assert len(buffer.pending) <= 5
    output.append(buffer.finish())
    assert "".join(output) == "abcdefghij" * 1000
