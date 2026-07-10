"""
Tests for P2: grounded J-lens telemetry.

Covers:
  * JLensProbe -- token-grounded workspace readout on the ProbeBus.
  * EmotionSystem ingestion of workspace_occupancy + concept scores (raw signals
    and probe events), and the grounded EmotionState fields.
  * The WORKSPACE_CONCEPT target type.
  * ProbeEvaluator: the previously *simulated* (np.random) internal-state signals
    are gone -- only real signals (entropy/surprisal + J-lens workspace telemetry)
    flow into the emotion system.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neuromod.jspace import JacobianLens, JLensConfig
from neuromod.probes import ProbeBus, ProbeListener, ProbeEvent, JLensProbe, create_jlens_probe
from neuromod.emotion_system import EmotionSystem
from neuromod.optimization.targets import (
    BehavioralTarget, TargetType, OptimizationObjective,
)


# --------------------------------------------------------------------------- #
# Minimal causal LM (HF-shaped) + deterministic tokenizer
# --------------------------------------------------------------------------- #
class _Cfg:
    def __init__(self, name):
        self._name_or_path = name


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
        self.blocks = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        self.unembed = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.normal_(generator=g)
            for b in self.blocks:
                b.weight.normal_(generator=g)
                b.bias.normal_(generator=g)
            self.unembed.weight.normal_(generator=g)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None,
                output_hidden_states=False, use_cache=False, **kw):
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        hs = [h]
        for b in self.blocks:
            h = h + torch.tanh(b(h))
            hs.append(h)
        return _Out(self.unembed(h), tuple(hs) if output_hidden_states else None)


class _Enc(dict):
    """Dict that also supports attribute access, like HF's BatchEncoding."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _Tok:
    eos_token_id = -1  # never matches a real (>=1) id, so generation runs to the cap

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
        return _Enc(input_ids=torch.tensor([ids], dtype=torch.long))


CONCEPTS = ["alpha", "bravo", "charlie"]
PROBE = "the engineers review the plan"


@pytest.fixture
def basis_and_model():
    torch.manual_seed(0)
    model = _ToyLM(seed=0)
    tok = _Tok()
    basis = JacobianLens(model, tok, JLensConfig(layers="workspace_band")).fit(CONCEPTS)
    return basis, model, tok


def _hidden_states(model, tok, text):
    with torch.no_grad():
        out = model(input_ids=tok(text)["input_ids"], output_hidden_states=True)
    return out.hidden_states


# --------------------------------------------------------------------------- #
# JLensProbe
# --------------------------------------------------------------------------- #
def test_jlens_probe_emits_workspace_telemetry(basis_and_model):
    basis, model, tok = basis_and_model
    bus = ProbeBus()
    bus.register_probe(create_jlens_probe(basis))
    listener = ProbeListener("J_LENS")
    bus.add_listener("J_LENS", listener)

    bus.process_signals(hidden_states=_hidden_states(model, tok, PROBE))

    assert listener.events, "J_LENS probe should fire every token"
    ev = listener.events[-1]
    assert "workspace_occupancy" in ev.metadata
    assert 0.0 <= ev.metadata["workspace_occupancy"] <= 1.0 + 1e-6
    assert set(ev.metadata["concept_scores"]) == set(CONCEPTS)
    assert ev.metadata["top_concepts"][0] in CONCEPTS
    # raw_signals carry the grounded, concept-named readouts
    assert "workspace_occupancy" in ev.raw_signals
    assert all(f"concept::{c}" in ev.raw_signals for c in CONCEPTS)


def test_jlens_probe_watch_concepts_sets_intensity(basis_and_model):
    basis, model, tok = basis_and_model
    hs = _hidden_states(model, tok, PROBE)
    jl = basis.layer_indices[-1]
    h = hs[jl][0, -1, :]
    scores = dict(basis.readout(h, layer=jl))

    probe = JLensProbe(basis, watch_concepts=["alpha"])
    listener = ProbeListener("J_LENS")
    probe.add_listener(listener)
    probe.update_position(1)
    probe.process_signals(hidden_states=hs)

    assert listener.events
    got = listener.events[-1].intensity
    expected = max(0.0, min(1.0, scores["alpha"]))
    assert abs(got - expected) < 1e-4


# --------------------------------------------------------------------------- #
# EmotionSystem ingestion
# --------------------------------------------------------------------------- #
def test_emotion_system_ingests_raw_workspace_signals():
    es = EmotionSystem(window_size=16)
    es.update_raw_signals({
        "entropy": 1.0, "surprisal": 2.0,
        "workspace_occupancy": 0.4,
        "concept::honest": 0.7, "concept::careful": 0.2,
    })
    state = es.update_emotion_state(0)
    assert abs(state.workspace_occupancy - 0.4) < 1e-6
    assert "honest" in state.workspace_concepts
    assert abs(state.workspace_concepts["honest"] - 0.7) < 1e-6


def test_emotion_system_ingests_probe_event_metadata():
    es = EmotionSystem(window_size=16)
    ev = ProbeEvent(
        probe_name="J_LENS", timestamp=0, intensity=0.5,
        metadata={"workspace_occupancy": 0.6, "concept_scores": {"honest": 0.3}},
        raw_signals={},
    )
    es.update_probe_statistics(ev)
    assert abs(es.get_workspace_occupancy() - 0.6) < 1e-6
    assert "honest" in es.get_workspace_concepts()


def test_occupancy_refinement_is_noop_without_telemetry():
    # With no workspace signals fed, the occupancy refinement must not change axes.
    es = EmotionSystem(window_size=16)
    es.update_raw_signals({"entropy": 0.5, "surprisal": 1.0})
    axes = es.compute_latent_axes()
    assert len(es.workspace_occupancy_buffer) == 0
    assert -1.0 <= axes["integration"] <= 1.0


# --------------------------------------------------------------------------- #
# WORKSPACE_CONCEPT target
# --------------------------------------------------------------------------- #
def test_workspace_concept_target():
    assert TargetType.WORKSPACE_CONCEPT.value == "workspace_concept"
    t = BehavioralTarget(name="hold_honest", description="keep honesty in the workspace")
    t.add_workspace_concept_target("honest", target_value=0.8, weight=2.0)
    spec = t.targets[0]
    assert spec.name == "workspace_honest"
    assert spec.target_type == TargetType.WORKSPACE_CONCEPT
    # loss is zero when actual hits target, positive otherwise
    assert t.compute_loss({"workspace_honest": 0.8}) == pytest.approx(0.0)
    assert t.compute_loss({"workspace_honest": 0.3}) > 0.0


# --------------------------------------------------------------------------- #
# ProbeEvaluator: real signals only (no np.random fabrication)
# --------------------------------------------------------------------------- #
def test_probe_evaluator_feeds_real_signals_not_random(basis_and_model):
    basis, model, tok = basis_and_model
    from neuromod.optimization.probe_evaluator import ProbeEvaluator

    # Dummy manager so __init__ doesn't build the real (model-loading) one; we call
    # the per-token routine directly with our toy model.
    evaluator = ProbeEvaluator(model_manager=object(), jspace_basis=basis)
    evaluator.emotion_system = EmotionSystem(window_size=64)
    bus = ProbeBus()
    bus.register_probe(create_jlens_probe(basis))

    result = evaluator._generate_with_probe_monitoring(model, tok, bus, PROBE, 0)
    assert result is not None

    es = evaluator.emotion_system
    # Real signals flowed:
    assert len(es.surprisal_buffer) > 0
    assert len(es.entropy_buffer) > 0
    assert len(es.workspace_occupancy_buffer) > 0
    assert len(es.workspace_concept_buffers) > 0
    # The previously fabricated (np.random) channels must stay empty -- nothing fakes them now.
    assert len(es.kl_buffer) == 0
    assert len(es.lr_attention_buffer) == 0
    assert len(es.prosocial_alignment_buffer) == 0
    assert len(es.anti_cliche_buffer) == 0
    assert len(es.risk_bend_buffer) == 0
