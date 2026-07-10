"""
Tests for the P1 J-space effects (in neuromod.effects):

    JLensSteerEffect, JSpaceAblationEffect, WorkspaceGainEffect, LensCoordinateSwapEffect

These are validated end-to-end against a tiny hook-compatible toy causal LM (no network):
the effects fit a JSpaceBasis from the toy, install forward hooks on the workspace-band
decoder layers, and we then read the residual stream back through the lens to confirm each
intervention did what the paper says -- while checking generation stays finite (the §3.5
"fluency preserved under workspace intervention" spirit).
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neuromod import effects as fx
from neuromod.effects import (
    EffectRegistry,
    JLensSteerEffect,
    JSpaceAblationEffect,
    WorkspaceGainEffect,
    LensCoordinateSwapEffect,
)
from neuromod.jspace import JacobianLens, JLensConfig
from neuromod.pack_system import Pack, PackManager


# --------------------------------------------------------------------------- #
# Hook-compatible toy causal LM: blocks are real modules returning tuples, and
# are discoverable via ``model.model.layers`` (what _resolve_transformer_layers
# looks for), so forward hooks land on the residual stream.
# --------------------------------------------------------------------------- #
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
        return (h + torch.tanh(self.lin(h)),)   # tuple output, like HF decoder layers


class _Inner(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = blocks


class _Out:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _ToyModel(nn.Module):
    def __init__(self, vocab=64, d=24, n_layers=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # per-instance name so the effect's basis cache never collides across tests
        self.config = _Cfg(f"toy-{id(self)}")
        self.embed = nn.Embedding(vocab, d)
        blocks = nn.ModuleList([_Block(d, seed + i + 1) for i in range(n_layers)])
        self.model = _Inner(blocks)             # -> model.model.layers
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
        return _Out(self.unembed(h), tuple(hs))


class _Tok:
    """Deterministic whitespace tokenizer (no Python hash salt)."""

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


CONCEPTS = ["alpha", "bravo"]         # distinct single tokens under _Tok
PROBE = "the engineers review the plan"


@pytest.fixture(autouse=True)
def _clear_basis_cache():
    fx._JSPACE_BASIS_CACHE.clear()
    yield
    fx._JSPACE_BASIS_CACHE.clear()


@pytest.fixture
def toy():
    torch.manual_seed(0)
    model = _ToyModel(seed=0)
    tok = _Tok()
    # sanity: the two concept tokens are distinct
    assert tok.encode(" alpha")[0] != tok.encode(" bravo")[0]
    return model, tok


def _hidden_at(model, tok, text, jl):
    ids = tok(text)["input_ids"]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[jl][0, -1, :].detach().clone()


def _logits(model, tok, text):
    ids = tok(text)["input_ids"]
    with torch.no_grad():
        return model(input_ids=ids).logits


def _ref_basis(model, tok, concepts=CONCEPTS, layers="workspace_band"):
    return JacobianLens(model, tok, JLensConfig(layers=layers)).fit(concepts)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_registry_exposes_jspace_effects():
    reg = EffectRegistry()
    for name, cls in [
        ("jlens_steer", JLensSteerEffect),
        ("jspace_ablation", JSpaceAblationEffect),
        ("workspace_gain", WorkspaceGainEffect),
        ("lens_coordinate_swap", LensCoordinateSwapEffect),
    ]:
        assert name in reg.list_effects()
        eff = reg.get_effect(name, concepts=CONCEPTS)
        assert isinstance(eff, cls)


# --------------------------------------------------------------------------- #
# Steering (runtime-CRT primitive): raises the target concept, surgically
# --------------------------------------------------------------------------- #
def test_steer_raises_target_readout_surgically(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]

    h0 = _hidden_at(model, tok, PROBE, jl)
    s0 = dict(ref.readout(h0, layer=jl))

    eff = JLensSteerEffect(weight=0.8, direction="up", concepts=["alpha"])
    eff.apply(model, tokenizer=tok)
    try:
        h1 = _hidden_at(model, tok, PROBE, jl)
        s1 = dict(ref.readout(h1, layer=jl))
        assert torch.isfinite(_logits(model, tok, PROBE)).all()  # fluency proxy
    finally:
        eff.cleanup()

    d_target = s1["alpha"] - s0["alpha"]
    d_control = s1["bravo"] - s0["bravo"]
    assert d_target > 0, (s0, s1)
    # surgical: the steered concept moves more than the untouched one
    assert d_target > abs(d_control)


def test_steer_down_suppresses(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]
    h0 = _hidden_at(model, tok, PROBE, jl)
    s0 = dict(ref.readout(h0, layer=jl))

    eff = JLensSteerEffect(weight=0.8, direction="down", concepts=["alpha"])
    eff.apply(model, tokenizer=tok)
    try:
        h1 = _hidden_at(model, tok, PROBE, jl)
        s1 = dict(ref.readout(h1, layer=jl))
    finally:
        eff.cleanup()
    assert s1["alpha"] < s0["alpha"]


# --------------------------------------------------------------------------- #
# Ablation: removes the concept from the workspace
# --------------------------------------------------------------------------- #
def test_ablation_removes_concept(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]

    h0 = _hidden_at(model, tok, PROBE, jl)
    u = ref.vector("alpha", layer=jl)
    u = u / u.norm()
    proj0 = float(h0 @ u)

    eff = JSpaceAblationEffect(weight=1.0, direction="up", concepts=["alpha"])
    eff.apply(model, tokenizer=tok)
    try:
        h1 = _hidden_at(model, tok, PROBE, jl)
    finally:
        eff.cleanup()
    proj1 = float(h1 @ u)
    # the final workspace-band hook removes the concept's component at this layer
    assert abs(proj1) < 1e-3
    assert abs(proj1) < abs(proj0)


def test_ablation_top_k_runs_and_is_finite(toy):
    model, tok = toy
    eff = JSpaceAblationEffect(weight=1.0, direction="up",
                              concepts=["alpha", "bravo"], top_k=1)
    eff.apply(model, tokenizer=tok)
    try:
        assert torch.isfinite(_logits(model, tok, PROBE)).all()
    finally:
        eff.cleanup()


# --------------------------------------------------------------------------- #
# Workspace gain: scales the workspace component up / down
# --------------------------------------------------------------------------- #
def test_workspace_gain_changes_occupancy(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]
    base_occ = ref.occupancy(_hidden_at(model, tok, PROBE, jl), layer=jl)

    up = WorkspaceGainEffect(weight=0.9, direction="up", concepts=CONCEPTS)
    up.apply(model, tokenizer=tok)
    try:
        occ_up = ref.occupancy(_hidden_at(model, tok, PROBE, jl), layer=jl)
    finally:
        up.cleanup()

    down = WorkspaceGainEffect(weight=0.9, direction="down", concepts=CONCEPTS)
    down.apply(model, tokenizer=tok)
    try:
        occ_down = ref.occupancy(_hidden_at(model, tok, PROBE, jl), layer=jl)
    finally:
        down.cleanup()

    assert occ_up > base_occ
    assert occ_down < base_occ


# --------------------------------------------------------------------------- #
# Coordinate swap: exchanges two concepts, complement untouched (single layer)
# --------------------------------------------------------------------------- #
def test_swap_exchanges_coordinates_single_layer(toy):
    model, tok = toy
    # Hook only the final decoder layer so the recorded state is exactly swap(baseline).
    eff = LensCoordinateSwapEffect(src="alpha", tgt="bravo", layers=[4])
    # baseline at that single layer, before hooks
    ref = JacobianLens(model, tok, JLensConfig(layers=[4])).fit(["alpha", "bravo"])
    jl = ref.layer_indices[-1]
    h0 = _hidden_at(model, tok, PROBE, jl)

    eff.apply(model, tokenizer=tok)
    try:
        h1 = _hidden_at(model, tok, PROBE, jl)
    finally:
        eff.cleanup()

    V = torch.stack([ref.vector("alpha", jl), ref.vector("bravo", jl)], dim=-1)  # [d,2]
    Vp = torch.linalg.pinv(V)
    c0, c1 = Vp @ h0, Vp @ h1
    assert torch.allclose(c1, c0.flip(-1), atol=1e-4), (c0, c1)


# --------------------------------------------------------------------------- #
# Cleanup fully restores baseline behavior
# --------------------------------------------------------------------------- #
def test_cleanup_restores_baseline(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]
    h0 = _hidden_at(model, tok, PROBE, jl)

    eff = JLensSteerEffect(weight=0.8, direction="up", concepts=["alpha"])
    eff.apply(model, tokenizer=tok)
    eff.cleanup()
    h_after = _hidden_at(model, tok, PROBE, jl)
    assert torch.allclose(h_after, h0, atol=1e-6)


# --------------------------------------------------------------------------- #
# Integration through PackManager + the j_space pack field
# --------------------------------------------------------------------------- #
def test_packmanager_applies_jlens_pack(toy):
    model, tok = toy
    ref = _ref_basis(model, tok)
    jl = ref.layer_indices[-1]
    s0 = dict(ref.readout(_hidden_at(model, tok, PROBE, jl), layer=jl))

    pack = Pack.from_dict({
        "name": "principle_test",
        "description": "runtime-CRT style principle steering",
        "effects": [{
            "effect": "jlens_steer", "weight": 0.7, "direction": "up",
            "parameters": {"concepts": ["alpha"], "layers": "workspace_band"},
        }],
        "j_space": {"concepts": ["alpha"], "band": [0.25, 0.9]},
    })
    assert pack.j_space is not None

    pm = PackManager()
    result = pm.apply_pack(pack, model, tokenizer=tok)
    try:
        assert not result["errors"], result["errors"]
        s1 = dict(ref.readout(_hidden_at(model, tok, PROBE, jl), layer=jl))
    finally:
        pm.clear_effects()
    assert s1["alpha"] > s0["alpha"]
