"""
Tests that SteeringEffect actually hooks the model and shifts the residual stream.

Regression guard for the bug where SteeringEffect.apply() was a no-op, so steering vectors did
NOTHING during served generation. Requires torch; skipped otherwise.
"""

import os
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from neuromod.effects import SteeringEffect  # noqa: E402


class _Layer(nn.Module):
    """A decoder-style layer that returns (hidden_states,) like HF layers do."""
    def forward(self, x):
        return (x,)


class _Inner(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x


class _Model(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.model = _Inner(n)

    def forward(self, x):
        return self.model(x)


def _effect_with_vector(hidden=8, weight=1.0, n_layers=1):
    eff = SteeringEffect(weight=weight, steering_type="assoc")
    # Inject a known unit vector directly so no disk load is needed.
    eff.vector = torch.ones(hidden)
    eff._vector_cache["/fake"] = eff.vector
    eff._steer_last_n = n_layers
    return eff


def test_apply_registers_hooks_and_shifts_output():
    model = _Model(n=3)
    eff = _effect_with_vector(hidden=8, weight=1.0, n_layers=1)

    x = torch.zeros(1, 4, 8)
    base = model(x)
    assert torch.allclose(base, torch.zeros_like(base))  # no steering yet

    eff.apply(model)
    assert len(eff._handles) == 1  # only the last layer hooked by default

    steered = model(x)
    # The hook adds effective_strength * steering_vector to the last layer's output.
    assert not torch.allclose(steered, torch.zeros_like(steered)), "steering had no effect"
    assert (steered != 0).any()


def test_cleanup_removes_hooks():
    model = _Model(n=3)
    eff = _effect_with_vector(hidden=8)
    eff.apply(model)
    assert eff._handles

    eff.cleanup()
    assert eff._handles == []
    # After cleanup, output is back to the unsteered baseline.
    x = torch.zeros(1, 4, 8)
    assert torch.allclose(model(x), torch.zeros_like(model(x)))


def test_steer_last_n_env(monkeypatch):
    monkeypatch.setenv("NEUROMOD_STEER_LAYERS", "2")
    model = _Model(n=4)
    eff = SteeringEffect(weight=1.0, steering_type="assoc")
    eff.vector = torch.ones(8)
    eff.apply(model)
    assert len(eff._handles) == 2  # last two layers hooked


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
