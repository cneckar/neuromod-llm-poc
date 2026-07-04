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

    def generate_text(self, prompt, max_tokens, temperature, top_p, pack_name, intensity=0.5,
                      custom_pack=None):
        self.calls.append(dict(prompt=prompt, max_tokens=max_tokens, temperature=temperature,
                               top_p=top_p, pack_name=pack_name, intensity=intensity,
                               custom_pack=custom_pack))
        # A custom pack "applies" in this fake so the honest-label path can be exercised.
        return {"text": self.text, "emotions": self.emotions, "tokens_generated": 5,
                "neuromod_applied": True if (pack_name or custom_pack) else None}


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


def test_parse_event_raw_prompt_and_intensity_overload():
    # Intensity is a multiplier now: > 1.0 is allowed (overload), bounded by the safety ceiling.
    ev = {"prompt": "just this", "intensity": 3.0, "pack_name": "cocaine", "max_tokens": 32}
    parsed = h.parse_event(ev)
    assert parsed["prompt"] == "just this"
    assert parsed["intensity"] == 3.0  # passes through (not clamped to 1.0)
    assert parsed["max_tokens"] == 32


def test_parse_event_intensity_clamped_to_ceiling(monkeypatch):
    monkeypatch.setenv("NEUROMOD_MAX_INTENSITY", "5.0")
    assert h.parse_event({"prompt": "x", "intensity": 99.0})["intensity"] == 5.0  # capped
    assert h.parse_event({"prompt": "x", "intensity": -2.0})["intensity"] == 0.0  # floor


def test_parse_event_bad_numbers_fall_back():
    parsed = h.parse_event({"prompt": "x", "max_tokens": "not-a-number"})
    assert parsed["max_tokens"] == 128


def test_parse_event_custom_pack_passthrough_and_validation():
    cp = {"name": "MyDrug", "effects": [{"effect": "steering", "weight": 0.5, "direction": "up"}]}
    assert h.parse_event({"prompt": "x", "custom_pack": cp})["custom_pack"] == cp
    # A custom pack with no effects (or a non-dict) is rejected -> None.
    assert h.parse_event({"prompt": "x", "custom_pack": {"name": "n", "effects": []}})["custom_pack"] is None
    assert h.parse_event({"prompt": "x", "custom_pack": "nope"})["custom_pack"] is None


def test_run_inference_applies_custom_pack_and_labels_it():
    model = FakeModel(text="ok")
    cp = {"name": "MyDrug", "effects": [{"effect": "steering", "weight": 0.5, "direction": "up"}]}
    r = h.run_inference(h.parse_event({"prompt": "hi", "custom_pack": cp, "intensity": 0.9}), model=model)
    assert model.calls[0]["custom_pack"] == cp          # reached the model
    assert r["pack_applied"] == "MyDrug"                # labeled by the custom name
    assert r["neuromod_applied"] is True


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
    assert model.calls[0]["intensity"] == 0.8  # dose must reach the model (was silently dropped)
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

    def generate_text_stream(self, prompt, max_tokens, temperature, top_p, pack_name, intensity=0.5,
                             custom_pack=None):
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


# ---------------------------------------------------------------- model registry (D2)


def test_resolve_model_aliases_and_allowlist(monkeypatch):
    monkeypatch.delenv("ALLOW_ANY_MODEL", raising=False)
    assert h.resolve_model("gpt-oss-120b") == "openai/gpt-oss-120b"
    assert h.resolve_model("gemma-3-27b") == "google/gemma-3-27b-it"
    # An already-allowed HF id passes through untouched.
    assert h.resolve_model("openai/gpt-oss-20b") == "openai/gpt-oss-20b"
    # Empty -> the endpoint default.
    assert h.resolve_model(None) == h.DEFAULT_MODEL
    assert h.resolve_model("") == h.DEFAULT_MODEL


def test_resolve_model_rejects_unvetted_id(monkeypatch):
    monkeypatch.delenv("ALLOW_ANY_MODEL", raising=False)
    # An untrusted id is NOT loaded; it falls back to the default.
    assert h.resolve_model("some/huge-unvetted-model-680b") == h.DEFAULT_MODEL
    # ...unless the dev override is set.
    monkeypatch.setenv("ALLOW_ANY_MODEL", "1")
    assert h.resolve_model("some/other-model") == "some/other-model"


def test_parse_event_resolves_model_alias():
    parsed = h.parse_event({"prompt": "hi", "model": "gpt-oss-120b"})
    assert parsed["model"] == "openai/gpt-oss-120b"


# ---------------------------------------------------------------- cost logging (D4)


def test_billing_record_fields():
    parsed = {"model": "openai/gpt-oss-20b", "pack_name": "lsd"}
    rec = h.billing_record(parsed, generation_time=2.5, cold_start=None)
    assert rec["gpu_seconds"] == 2.5
    assert rec["model"] == "openai/gpt-oss-20b" and rec["pack"] == "lsd"
    assert rec["cold_start_seconds"] is None
    rec2 = h.billing_record(parsed, 2.5, cold_start=12.0)
    assert rec2["cold_start_seconds"] == 12.0


def test_run_inference_emits_gpu_seconds():
    model = FakeModel()
    parsed = h.parse_event({"prompt": "hi", "pack_name": "lsd"})
    r = h.run_inference(parsed, model=model)
    assert "gpu_seconds" in r and isinstance(r["gpu_seconds"], float)
    # No cold start on an injected model.
    assert r["cold_start_seconds"] is None


def test_stream_final_carries_gpu_seconds():
    model = FakeStreamModel(["a", "b"])
    events = list(h.run_inference_stream(h.parse_event({"prompt": "hi"}), model=model))
    assert "gpu_seconds" in events[-1]


# ---------------------------------------------------------------- server-side task routing


def test_parse_event_defaults_task_to_generate():
    assert h.parse_event({"prompt": "hi"})["task"] == "generate"
    assert h.parse_event({"input": {"task": "steering"}})["task"] == "steering"


def test_build_job_command_steering():
    parsed = h.parse_event({"task": "steering", "model": "gpt-oss-120b", "layer": -1})
    cmd, out = h.build_job_command("steering", parsed)
    assert "generate_steering_vectors.py" in " ".join(cmd)
    assert "--model" in cmd and "openai/gpt-oss-120b" in cmd  # alias resolved
    assert "--layer" in cmd and "-1" in cmd
    assert out == h.STEERING_DIR


def test_build_job_command_endpoints():
    parsed = h.parse_event({"task": "endpoints", "pack_name": "cocaine", "model": "gpt2"})
    cmd, out = h.build_job_command("endpoints", parsed)
    assert "calculate_endpoints.py" in " ".join(cmd)
    assert cmd[cmd.index("--pack") + 1] == "cocaine"
    assert "--skip-completed" in cmd
    assert out == h.ENDPOINTS_DIR


def test_dispatch_unknown_task_errors():
    parsed = h.parse_event({"task": "not_a_task", "prompt": "x"})
    out = h.dispatch_task(parsed)
    assert "error" in out


def test_handler_routes_task_not_generate(monkeypatch):
    # A task event bypasses the generate path entirely (no prompt needed).
    monkeypatch.setattr(h, "dispatch_task", lambda parsed, model=None: {"ok": True, "task": parsed["task"]})
    out = h.handler({"input": {"task": "warmup"}})
    assert out == {"ok": True, "task": "warmup"}


def test_stream_handler_yields_single_result_for_task(monkeypatch):
    monkeypatch.setattr(h, "dispatch_task", lambda parsed, model=None: {"ok": True, "task": "steering"})
    out = list(h.stream_handler({"input": {"task": "steering"}}))
    assert out == [{"ok": True, "task": "steering"}]


def test_diag_task_reports_runtime(monkeypatch):
    out = h.dispatch_task(h.parse_event({"task": "diag", "prompt": "x"}))
    assert out["task"] == "diag" and out["ok"] is True
    # Reports package versions (or "(not installed)") + volume presence, without loading a model.
    assert "transformers" in out and "torch" in out and "volume_mounted" in out
    assert h.parse_event({"input": {"task": "diag"}})["task"] == "diag"


def test_run_warmup_uses_get_model(monkeypatch):
    calls = {}
    monkeypatch.setattr(h, "_get_model", lambda name: calls.setdefault("name", name))
    monkeypatch.setattr(h, "_pop_cold_start", lambda: 3.0)
    res = h.run_warmup(h.parse_event({"task": "warmup", "model": "gpt2"}))
    assert res["ok"] and res["task"] == "warmup" and res["cold_start_seconds"] == 3.0
    assert calls["name"] == "gpt2"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
