"""
Tests for neuromod.jspace (P0: the Jacobian lens and the J-space).

Two layers of validation:

1. The intervention algebra (steer / ablate / ablate_span / swap) is checked
   against its defining mathematical properties on random vectors -- no model
   required, fully deterministic.

2. The Jacobian-lens *fitter* is exercised end-to-end against a tiny toy causal
   language model that mimics the HuggingFace interface. Because the toy model's
   final layer maps to logits through a known unembedding ``U``, we can assert
   the fitted last-layer lens vector for a concept recovers ``U[token]`` (up to
   the corpus-averaged position reduction), that ``readout`` ranks the intended
   concept first, and that save/load round-trips.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from neuromod import jspace
from neuromod.jspace import (
    JacobianLens,
    JLensConfig,
    JSpaceBasis,
    ablate,
    ablate_span,
    steer,
    swap,
)


# --------------------------------------------------------------------------- #
# 1. Intervention algebra
# --------------------------------------------------------------------------- #
def _rand(d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=g, dtype=torch.float64)


def test_steer_raises_alignment():
    h, v = _rand(16, 1), _rand(16, 2)
    u = v / v.norm()
    before = float(h @ u)
    after = float(steer(h, v, alpha=2.0) @ u)
    assert after > before
    assert math.isclose(after - before, 2.0, rel_tol=1e-6)


def test_ablate_removes_component():
    h, v = _rand(16, 3), _rand(16, 4)
    u = v / v.norm()
    # residual carries an O(eps) bias from the unit-normalization epsilon
    assert abs(float(ablate(h, v) @ u)) < 1e-6


def test_ablate_span_removes_both():
    h = _rand(16, 5)
    M = torch.stack([_rand(16, 6), _rand(16, 7)])          # [2, d]
    out = ablate_span(h, M)
    Q = jspace._orthonormal_rows(M)
    assert torch.allclose(out @ Q.transpose(-1, -2),
                          torch.zeros(2, dtype=torch.float64), atol=1e-9)


def test_swap_preserves_complement_and_exchanges_coords():
    h = _rand(16, 8)
    v, w = _rand(16, 9), _rand(16, 10)
    V = torch.stack([v, w], dim=-1)                        # [d, 2]
    P = V @ torch.linalg.pinv(V)                           # orthogonal projector
    comp_before = h - P @ h
    hs = swap(h, V)
    comp_after = hs - P @ hs
    assert torch.allclose(comp_before, comp_after, atol=1e-9)

    Vp = torch.linalg.pinv(V)
    c_before, c_after = Vp @ h, Vp @ hs
    assert torch.allclose(c_after, c_before.flip(-1), atol=1e-9)
    # involutive
    assert torch.allclose(swap(hs, V), h, atol=1e-8)


def test_interventions_are_batched():
    H = _rand(5 * 16, 11).reshape(5, 16)
    v = _rand(16, 12)
    assert ablate(H, v).shape == (5, 16)
    assert steer(H, v, 0.5).shape == (5, 16)


# --------------------------------------------------------------------------- #
# 2. Toy causal LM to exercise the fitter without downloading a model
# --------------------------------------------------------------------------- #
class _ToyOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class _ToyConfig:
    _name_or_path = "toy/tiny-causal-lm"


class _ToyCausalLM(torch.nn.Module):
    """A minimal causal-LM-shaped model.

    Architecture: embedding -> N residual MLP blocks -> unembedding. It returns
    an object with ``.logits`` and ``.hidden_states`` (tuple of length N+1),
    matching what JacobianLens.fit consumes. The final hidden state maps to
    logits linearly via ``self.unembed``, so lens vectors at the last layer are
    analytically known.
    """

    def __init__(self, vocab=32, d=24, n_layers=4, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.config = _ToyConfig()
        self.embed = torch.nn.Embedding(vocab, d)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(n_layers)])
        self.unembed = torch.nn.Linear(d, vocab, bias=False)
        # Deterministic-ish init
        with torch.no_grad():
            self.embed.weight.normal_(generator=g)
            for b in self.blocks:
                b.weight.normal_(generator=g)
                b.bias.normal_(generator=g)
            self.unembed.weight.normal_(generator=g)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None,
                output_hidden_states=False, use_cache=False, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        h = inputs_embeds
        hs = [h]
        for b in self.blocks:
            h = h + torch.tanh(b(h))          # residual block
            hs.append(h)
        logits = self.unembed(h)              # [1, seq, vocab]
        return _ToyOutput(logits, tuple(hs))


class _ToyTokenizer:
    """Whitespace tokenizer over a fixed vocab, HF-ish call signature."""

    def __init__(self, vocab=32):
        self.vocab = vocab

    def _ids(self, text):
        # Map words to stable ids via hashing into the vocab range (>=1).
        ids = [1 + (abs(hash(w)) % (self.vocab - 1)) for w in text.split()]
        return ids or [1]

    def encode(self, text, add_special_tokens=False):
        return self._ids(text)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = self._ids(text)
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


@pytest.fixture
def toy():
    torch.manual_seed(0)
    model = _ToyCausalLM(vocab=32, d=24, n_layers=4, seed=0)
    tok = _ToyTokenizer(vocab=32)
    corpus = [
        "the quick brown fox",
        "a slow green turtle walks",
        "bright stars over the valley",
        "engineers review the careful plan",
    ]
    return model, tok, corpus


def test_fit_shapes_and_layers(toy):
    model, tok, corpus = toy
    lens = JacobianLens(model, tok, JLensConfig(layers="all"))
    concepts = ["alpha", "beta", "gamma"]
    tids = [3, 7, 11]
    basis = lens.fit(concepts, corpus=corpus, token_ids=tids)
    assert isinstance(basis, JSpaceBasis)
    assert basis.num_concepts == 3
    assert basis.hidden_size == 24
    # "all" -> layers 1..4 (excludes embedding index 0)
    assert basis.layer_indices == [1, 2, 3, 4]
    assert basis.vectors.shape == (3, 4, 24)


def test_last_layer_lens_recovers_unembedding(toy):
    """At the final layer, logits = h @ U^T, so d(sum_q logit_{q,c})/dh_p = U[c].

    Averaged over positions this is exactly U[c]; the fitted lens vector should
    therefore be (anti-)parallel to the unembedding row for the concept token.
    """
    model, tok, corpus = toy
    lens = JacobianLens(model, tok, JLensConfig(layers="all", position_reduce="mean"))
    tids = [5, 9]
    basis = lens.fit(["c0", "c1"], corpus=corpus, token_ids=tids)
    U = model.unembed.weight.detach().to(torch.float32)   # [vocab, d]
    for concept, tid in zip(["c0", "c1"], tids):
        v = basis.vector(concept, layer=4).to(torch.float32)
        cos = float(torch.nn.functional.cosine_similarity(v, U[tid], dim=0))
        assert cos > 0.999, f"{concept}: cos={cos}"


def test_readout_ranks_intended_concept(toy):
    model, tok, corpus = toy
    lens = JacobianLens(model, tok, JLensConfig(layers="all"))
    tids = [5, 9, 13]
    concepts = ["c0", "c1", "c2"]
    basis = lens.fit(concepts, corpus=corpus, token_ids=tids)
    U = model.unembed.weight.detach().to(torch.float32)
    # A state aligned with U[tid] for c1 should read out c1 on top at last layer.
    h = U[tids[1]].clone()
    ranked = basis.readout(h, layer=4)
    assert ranked[0][0] == "c1", ranked


def test_interventions_on_fitted_basis(toy):
    model, tok, corpus = toy
    lens = JacobianLens(model, tok, JLensConfig(layers="all"))
    basis = lens.fit(["c0", "c1"], corpus=corpus, token_ids=[5, 9])
    h = torch.randn(24)
    # steer raises the readout score of the steered concept
    s0 = dict(basis.readout(h, layer=4, normalize=True))["c0"]
    s1 = dict(basis.readout(basis.steer(h, "c0", layer=4, alpha=5.0),
                            layer=4, normalize=True))["c0"]
    assert s1 > s0
    # ablate drives the (unnormalized) projection onto that concept to ~0
    ha = basis.ablate(h, "c0", layer=4)
    u = basis.vector("c0", layer=4)
    u = u / u.norm()
    assert abs(float(ha @ u)) < 1e-5
    # occupancy is a fraction in [0, 1]
    occ = basis.occupancy(h, layer=4)
    assert 0.0 <= occ <= 1.0 + 1e-6


def test_save_load_roundtrip(toy, tmp_path):
    model, tok, corpus = toy
    lens = JacobianLens(model, tok, JLensConfig(layers="workspace_band"))
    basis = lens.fit(["c0", "c1"], corpus=corpus, token_ids=[5, 9])
    p = basis.save(tmp_path / "basis.pt")
    reloaded = JSpaceBasis.load(p)
    assert reloaded.concepts == basis.concepts
    assert reloaded.layer_indices == basis.layer_indices
    assert torch.allclose(reloaded.vectors, basis.vectors)
    assert reloaded.model_name == "toy/tiny-causal-lm"


def test_workspace_band_excludes_edges(toy):
    model, tok, corpus = toy   # n_layers = 4  -> L = 4
    lens = JacobianLens(model, tok, JLensConfig(layers="workspace_band", band=(0.25, 0.9)))
    basis = lens.fit(["c0"], corpus=corpus, token_ids=[5])
    # band (0.25,0.9) of L=4 -> lo=1, hi=4 here (small L); must be non-empty and valid
    assert basis.layer_indices
    assert all(1 <= i <= 4 for i in basis.layer_indices)
