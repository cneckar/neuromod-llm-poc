"""
Unit tests for the RunPod handler's pure request/response logic (Deploy D0).

GPU-free and torch-free: the handler's marshalling functions are tested with a fake model,
so the ChatRequest/ChatResponse contract and pack pass-through are verified without loading
a real model. Loaded by file path (the deploy dir is not a package).
"""

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "..", "deploy", "runpod", "handler.py")


def _load():
    spec = importlib.util.spec_from_file_location("runpod_handler", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load()


class FakeModel:
    """Records the generate_text call and returns a canned result."""

    def __init__(self, text="Hello from the model.", emotions=None):
        self.text = text
        self.emotions = emotions or {"joy": 0.5}
        self.calls = []

    def generate_text(self, prompt, max_tokens, temperature, top_p, pack_name):
        self.calls.append(dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                               top_p=top_p, pack_name=pack_name))
        return {"text": self.text, "emotions": self.emotions, "tokens_generated": 5}


def test_messages_to_prompt_roles():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "how are you"}]
    p = h.messages_to_prompt(msgs)
    assert "User: hi" in p and "Assistant: yo" in p
    assert p.endswith("Assistant:")


def test_parse_event_input_wrapper_and_defaults():
    ev = {"input": {"messages": [{"role": "user", "content": "hi"}], "pack_name": "lsd"}}
    parsed = h.parse_event(ev)
    assert parsed["pack_name"] == "lsd"
    assert parsed["max_tokens"] == 128 and parsed["temperature"] == 1.0
    assert parsed["model"] == h.DEFAULT_MODEL
    assert parsed["prompt"].endswith("Assistant:")


def test_parse_event_raw_prompt_and_intensity_clamp():
    ev = {"prompt": "just this", "intensity": 5.0, "pack_name": "cocaine", "max_tokens": 32}
    parsed = h.parse_event(ev)
    assert parsed["prompt"] == "just this"
    assert parsed["intensity"] == 1.0  # clamped to [0, 1]
    assert parsed["max_tokens"] == 32


def test_parse_event_bad_numbers_fall_back():
    parsed = h.parse_event({"prompt": "x", "max_tokens": "not-a-number"})
    assert parsed["max_tokens"] == 128


def test_format_response_shape_and_token_count():
    parsed = {"pack_name": "dmt", "intensity": 0.7, "model": "m"}
    r = h.format_response("one two three", parsed, generation_time=1.234)
    assert r["response"] == "one two three"
    assert r["pack_applied"] == "dmt" and r["intensity"] == 0.7
    assert r["tokens_generated"] == 3  # word-count fallback
    assert r["model_type"] == "local" and r["model"] == "m"


def test_run_inference_passes_pack_through():
    model = FakeModel()
    parsed = h.parse_event({"prompt": "hi", "pack_name": "lsd", "intensity": 0.8, "max_tokens": 64})
    r = h.run_inference(parsed, model=model)
    assert model.calls[0]["pack_name"] == "lsd"
    assert model.calls[0]["max_tokens"] == 64
    assert r["response"] == "Hello from the model."
    assert r["emotions"] == {"joy": 0.5}
    assert r["tokens_generated"] == 5


def test_handler_empty_prompt_errors():
    r = h.handler({"input": {"messages": []}})
    # messages_to_prompt still appends "Assistant:", so empty-messages yields a prompt;
    # a truly empty payload with no messages/prompt should error.
    r2 = h.handler({"input": {"pack_name": "lsd", "messages": None, "prompt": ""}})
    assert "error" in r2


def test_handler_end_to_end_with_fake_model():
    model = FakeModel(text="ok")
    r = h.handler({"input": {"prompt": "hello", "pack_name": "mdma"}}, model=model)
    assert r["response"] == "ok"
    assert r["pack_applied"] == "mdma"


# ---------------------------------------------------------------- streaming (Deploy D1)


class FakeStreamModel:
    """Model exposing a native token stream."""

    def __init__(self, chunks):
        self.chunks = chunks

    def generate_text_stream(self, prompt, max_tokens, temperature, top_p, pack_name):
        for c in self.chunks:
            yield c


def test_chunk_text_preserves_content():
    pieces = h.chunk_text("one two three")
    assert "".join(pieces) == "one two three"
    assert len(pieces) == 3


def test_stream_native_model_yields_chunks_then_done():
    model = FakeStreamModel(["Hel", "lo ", "world"])
    events = list(h.run_inference_stream(h.parse_event({"prompt": "hi"}), model=model))
    chunks = [e["chunk"] for e in events if "chunk" in e]
    assert chunks == ["Hel", "lo ", "world"]
    final = events[-1]
    assert final.get("done") is True
    assert final["response"] == "Hello world"


def test_stream_fallback_chunks_full_text():
    model = FakeModel(text="alpha beta gamma")
    events = list(h.run_inference_stream(h.parse_event({"prompt": "hi"}), model=model))
    chunks = [e["chunk"] for e in events if "chunk" in e]
    assert "".join(chunks) == "alpha beta gamma"
    assert events[-1]["response"] == "alpha beta gamma"


def test_stream_handler_empty_prompt_errors():
    out = list(h.stream_handler({"input": {"prompt": "", "messages": None}}))
    assert out and "error" in out[0]


def test_stream_handler_passes_pack():
    model = FakeStreamModel(["x", "y"])
    out = list(h.stream_handler({"input": {"prompt": "hi", "pack_name": "lsd"}}, model=model))
    assert out[-1]["pack_applied"] == "lsd"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
