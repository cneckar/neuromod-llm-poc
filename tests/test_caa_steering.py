"""
Tests for CAASteeringJSpaceEffect (the J-space-restricted CAA steering effect).

The key behavior is the projection of the CAA vector by ``mode``:
  * full       -> the whole vector
  * jspace     -> only its component inside the workspace (project_in is idempotent)
  * complement -> only its component outside the workspace (project_in ~ 0)
plus ``match_magnitude`` making all three the same norm, and that the legacy
SteeringEffect is unaffected. A head-to-head integration check confirms the three
modes actually produce different residual states through a real forward pass.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neuromod.effects import EffectRegistry, CAASteeringJSpaceEffect, SteeringEffect
from neuromod.jspace import JacobianLens, JLensConfig


# --- hook-compatible toy LM (blocks are modules; model.model.layers) --------- #
class _Cfg:
    def __init__(self, name):
        self._name_or_path = name


class _Block(nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(d, d)
        with torch.no_grad():
            self.lin.weight.normal_(generator=g)
            self.lin.bias.normal_(generator=g)

    def forward(self, h):
        return (h + torch.tanh(self.lin(h)),)


class _Inner(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = blocks


class _Out:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _ToyLM(nn.Module):
    def __init__(self, vocab=64, d=24, n_layers=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = _Cfg(f"toy-{id(self)}")
        self.embed = nn.Embedding(vocab, d)
        blocks = nn.ModuleList([_Block(d, seed + i + 1) for i in range(n_layers)])
        self.model = _Inner(blocks)
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.normal_(generator=g)
            self.unembed.weight.normal_(generator=g)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None,
                output_hidden_states=False, use_cache=False, **kw):
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hs = [h]
        for b in self.model.layers:
            h = b(h)[0]
            hs.append(h)
        return _Out(self.unembed(h), tuple(hs) if output_hidden_states else None)


class _Tok:
    def __init__(self, vocab=64):
        self.vocab = vocab

    def _id(self, w):
        return 1 + (sum((i + 1) * ord(ch) for i, ch in enumerate(w)) % (self.vocab - 1))

    def _ids(self, text):
        return [self._id(w) for w in text.split()] or [1]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = self._ids(text)
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


CONCEPTS = ["alpha", "bravo", "charlie"]
PROBE = "the engineers review the plan"


@pytest.fixture
def env():
    torch.manual_seed(0)
    model = _ToyLM(seed=0)
    tok = _Tok()
    basis = JacobianLens(model, tok, JLensConfig(layers="workspace_band")).fit(CONCEPTS)
    return model, tok, basis


def _mixed_vector(basis, jl, d, seed=1):
    """A CAA vector with both an in-workspace and an out-of-workspace component."""
    g = torch.Generator().manual_seed(seed)
    inside = basis.matrix(layer=jl).sum(0)            # in the J-space span
    noise = torch.randn(d, generator=g)
    outside = noise - basis.project_in(noise, layer=jl)  # orthogonal to the span
    return (inside / inside.norm() + outside / outside.norm())


# --------------------------------------------------------------------------- #
# Projection semantics
# --------------------------------------------------------------------------- #
def test_registry_constructs_caa_steering():
    reg = EffectRegistry()
    assert "caa_steering" in reg.list_effects()
    eff = reg.get_effect("caa_steering", concepts=CONCEPTS, mode="jspace")
    assert isinstance(eff, CAASteeringJSpaceEffect)


def test_mode_full_returns_whole_vector(env):
    _, _, basis = env
    jl = basis.layer_indices[-1]
    v = _mixed_vector(basis, jl, basis.hidden_size)
    eff = CAASteeringJSpaceEffect(mode="full", concepts=CONCEPTS, caa_vector=v,
                                  match_magnitude=False)
    eff.basis = basis
    veff = eff._veff_for_layer(jl, basis.hidden_size)
    assert torch.allclose(veff, v, atol=1e-6)


def test_mode_jspace_lies_in_workspace(env):
    _, _, basis = env
    jl = basis.layer_indices[-1]
    v = _mixed_vector(basis, jl, basis.hidden_size)
    eff = CAASteeringJSpaceEffect(mode="jspace", concepts=CONCEPTS, caa_vector=v,
                                  match_magnitude=False)
    eff.basis = basis
    veff = eff._veff_for_layer(jl, basis.hidden_size)
    # idempotent: projecting an already-in-workspace vector changes nothing
    assert torch.allclose(basis.project_in(veff, layer=jl), veff, atol=1e-5)
    # and it is strictly smaller than the full vector (it dropped the outside part)
    assert veff.norm() < v.norm()


def test_mode_complement_is_orthogonal_to_workspace(env):
    _, _, basis = env
    jl = basis.layer_indices[-1]
    v = _mixed_vector(basis, jl, basis.hidden_size)
    eff = CAASteeringJSpaceEffect(mode="complement", concepts=CONCEPTS, caa_vector=v,
                                  match_magnitude=False)
    eff.basis = basis
    veff = eff._veff_for_layer(jl, basis.hidden_size)
    # the complement has ~no component inside the workspace
    assert basis.project_in(veff, layer=jl).norm() < 1e-4


def test_match_magnitude_equalizes_norm(env):
    _, _, basis = env
    jl = basis.layer_indices[-1]
    v = _mixed_vector(basis, jl, basis.hidden_size)
    for mode in ("jspace", "complement"):
        eff = CAASteeringJSpaceEffect(mode=mode, concepts=CONCEPTS, caa_vector=v,
                                      match_magnitude=True)
        eff.basis = basis
        veff = eff._veff_for_layer(jl, basis.hidden_size)
        assert abs(float(veff.norm()) - float(v.norm())) < 1e-4


# --------------------------------------------------------------------------- #
# Head-to-head: the three modes produce different residual states
# --------------------------------------------------------------------------- #
def _hidden(model, tok, jl):
    with torch.no_grad():
        out = model(input_ids=tok(PROBE)["input_ids"], output_hidden_states=True)
    return out.hidden_states[jl][0, -1, :].clone()


def test_three_modes_diverge_and_are_reversible(env):
    model, tok, basis = env
    jl = basis.layer_indices[-1]
    v = _mixed_vector(basis, jl, basis.hidden_size)
    baseline = _hidden(model, tok, jl)

    outs = {}
    for mode in ("full", "jspace", "complement"):
        eff = CAASteeringJSpaceEffect(weight=0.8, mode=mode, concepts=CONCEPTS,
                                      caa_vector=v, match_magnitude=True)
        eff.basis = basis
        eff.apply(model, tokenizer=tok)
        try:
            outs[mode] = _hidden(model, tok, jl)
            assert torch.isfinite(outs[mode]).all()
        finally:
            eff.cleanup()
        # reversible: cleanup restores baseline
        assert torch.allclose(_hidden(model, tok, jl), baseline, atol=1e-6)

    # the three arms genuinely differ from each other and from baseline
    assert not torch.allclose(outs["full"], baseline, atol=1e-4)
    assert not torch.allclose(outs["jspace"], outs["full"], atol=1e-4)
    assert not torch.allclose(outs["jspace"], outs["complement"], atol=1e-4)


# --------------------------------------------------------------------------- #
# Legacy steering is untouched
# --------------------------------------------------------------------------- #
def test_legacy_steering_effect_unchanged():
    # Constructs and exposes apply_steering exactly as before; no basis/concepts needed.
    eff = SteeringEffect(steering_type="associative")
    assert hasattr(eff, "apply_steering")
    assert not isinstance(eff, CAASteeringJSpaceEffect)
