"""
Unit tests for model-aware steering-vector resolution.

Steering vectors are model-specific (they live in a model's hidden space and carry its
hidden_size), so one repo/volume must hold vectors for several models side by side under
per-model subdirectories: ``<vector_dir>/<model_slug>/<type>_layer-1.pt``. These tests cover
``steering_model_slug`` and ``resolve_steering_vector_path`` from ``neuromod.effects``.

The helpers are pure (os/pathlib only), but ``neuromod.effects`` imports torch at module top.
To keep the tests runnable in a torch-free environment we fall back to exec'ing just the two
helper functions out of the source file.
"""

import importlib.util
import os

import pytest


def _load_helpers():
    """Return (steering_model_slug, resolve_steering_vector_path).

    Prefer a real import so the test exercises the shipped module; if torch is absent,
    slice the two pure helpers out of the source and exec them in isolation.
    """
    try:
        spec = importlib.util.find_spec("neuromod.effects")
        if spec is not None:
            from neuromod.effects import (  # noqa: WPS433 (local import is intentional)
                resolve_steering_vector_path,
                steering_model_slug,
            )
            return steering_model_slug, resolve_steering_vector_path
    except Exception:  # torch or another heavy dep missing — fall through to source slice
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "..", "neuromod", "effects.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("def steering_model_slug(")
    end = src.index("class SteeringEffect(BaseEffect):")
    from typing import Optional

    ns = {"os": os, "Optional": Optional}
    exec(compile(src[start:end], src_path, "exec"), ns)  # noqa: S102 (trusted repo source)
    return ns["steering_model_slug"], ns["resolve_steering_vector_path"]


slug, resolve = _load_helpers()


def test_slug_is_deterministic_and_fs_safe():
    assert slug("openai/gpt-oss-120b") == "openai__gpt-oss-120b"
    assert slug("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama__Llama-3.1-8B-Instruct"
    assert slug("gpt2") == "gpt2"  # no slash -> unchanged
    assert slug(None) is None
    assert slug("  openai/gpt-oss-120b  ") == "openai__gpt-oss-120b"  # trimmed


def _seed(tmp_path):
    (tmp_path / "openai__gpt-oss-120b").mkdir()
    (tmp_path / "openai__gpt-oss-120b" / "salient_layer-1.pt").write_bytes(b"x")
    (tmp_path / "visionary_layer-1.pt").write_bytes(b"x")  # flat / legacy
    (tmp_path / "ego_thin.pt").write_bytes(b"x")  # no layer suffix


def test_prefers_per_model_subdir(tmp_path):
    _seed(tmp_path)
    p = resolve(str(tmp_path), "salient", model_name="openai/gpt-oss-120b")
    assert p is not None and os.path.join("openai__gpt-oss-120b", "salient_layer-1.pt") in p


def test_falls_back_to_flat_when_no_per_model_file(tmp_path):
    _seed(tmp_path)
    p = resolve(str(tmp_path), "visionary", model_name="openai/gpt-oss-120b")
    assert p is not None and p.endswith("visionary_layer-1.pt")
    assert "openai__gpt-oss-120b" not in p


def test_no_model_only_sees_flat(tmp_path):
    _seed(tmp_path)
    # per-model-only 'salient' is invisible without a model hint
    assert resolve(str(tmp_path), "salient", model_name=None) is None
    # flat 'visionary' still resolves
    assert resolve(str(tmp_path), "visionary", model_name=None).endswith("visionary_layer-1.pt")


def test_model_name_env_fallback(tmp_path, monkeypatch):
    _seed(tmp_path)
    monkeypatch.setenv("MODEL_NAME", "openai/gpt-oss-120b")
    p = resolve(str(tmp_path), "salient")  # no explicit model_name -> uses env
    assert p is not None and "openai__gpt-oss-120b" in p


def test_no_layer_suffix_fallback(tmp_path):
    _seed(tmp_path)
    p = resolve(str(tmp_path), "ego_thin")
    assert p is not None and p.endswith("ego_thin.pt")


def test_missing_returns_none(tmp_path):
    assert resolve(str(tmp_path), "nonexistent", model_name="openai/gpt-oss-120b") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
