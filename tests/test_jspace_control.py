"""
Tests for P3: closed-loop workspace control and self-exploration.

Drives a hook-compatible toy causal LM (no network):
  * WorkspaceController tunes per-concept J-lens steering until the measured
    workspace matches a target, and is reversible.
  * SelfExplorationSession runs the full hypothesize -> apply -> observe -> tune ->
    verify loop, including the Goodhart hold-out and placebo-arm guards.
  * The optimizer's outer-loop bridge scores a WORKSPACE_CONCEPT target from the
    evaluator's grounded workspace telemetry.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from neuromod.jspace import JacobianLens, JLensConfig
from neuromod.jspace_control import (
    WorkspaceController, WorkspaceTarget, SelfExplorationSession, measure_workspace,
)


# --------------------------------------------------------------------------- #
# Hook-compatible toy causal LM (blocks are modules; model.model.layers)
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
    eos_token_id = -1

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


# --------------------------------------------------------------------------- #
# Inner loop
# --------------------------------------------------------------------------- #
def test_controller_converges_to_target(env):
    model, tok, basis = env
    base_scores, _, _ = measure_workspace(model, tok, basis, PROBE)
    tgt_val = base_scores["alpha"] + 0.15
    target = WorkspaceTarget(concepts={"alpha": tgt_val})

    ctrl = WorkspaceController(model, tok, basis, target, PROBE)
    try:
        result = ctrl.run()
    finally:
        ctrl.close()

    assert result["converged"], result["trajectory"]
    assert abs(result["final_scores"]["alpha"] - tgt_val) <= 0.03
    assert ctrl.weight_bounds[0] <= result["weights"]["alpha"] <= ctrl.weight_bounds[1]
    # the tuned concept moved toward target; an untouched concept moved far less
    d_alpha = result["final_scores"]["alpha"] - base_scores["alpha"]
    d_bravo = result["final_scores"]["bravo"] - base_scores["bravo"]
    assert d_alpha > abs(d_bravo)


def test_controller_is_reversible(env):
    model, tok, basis = env
    base_scores, base_occ, _ = measure_workspace(model, tok, basis, PROBE)
    target = WorkspaceTarget(concepts={"alpha": base_scores["alpha"] + 0.2})
    ctrl = WorkspaceController(model, tok, basis, target, PROBE)
    ctrl.run()
    ctrl.close()
    after_scores, after_occ, _ = measure_workspace(model, tok, basis, PROBE)
    assert abs(after_scores["alpha"] - base_scores["alpha"]) < 1e-5
    assert abs(after_occ - base_occ) < 1e-5


# --------------------------------------------------------------------------- #
# Self-exploration harness + guards
# --------------------------------------------------------------------------- #
def test_self_exploration_converges_with_guards(env):
    model, tok, basis = env
    base_scores, _, _ = measure_workspace(model, tok, basis, PROBE)
    target = WorkspaceTarget(concepts={"alpha": base_scores["alpha"] + 0.15})

    session = SelfExplorationSession(model, tok, basis, PROBE)
    report = session.run(target, with_placebo=True)

    assert report.converged
    # honest metadata present
    assert report.coverage_note
    assert set(report.holdout) == {"alpha"}
    # Goodhart hold-out: steering along the lens vector also moved the real output
    # distribution, so no Goodhart is suspected here.
    assert report.holdout["alpha"]["delta"] > 0
    assert not report.goodhart_suspected
    # trajectory retained for audit; effects were cleaned up (reversible)
    assert report.trajectory
    assert not session_effects_active(model)


def test_self_exploration_placebo_is_specific(env):
    model, tok, basis = env
    base_scores, _, _ = measure_workspace(model, tok, basis, PROBE)
    target = WorkspaceTarget(concepts={"alpha": base_scores["alpha"] + 0.15})
    report = SelfExplorationSession(model, tok, basis, PROBE).run(target, with_placebo=True)

    assert report.placebo is not None
    p = report.placebo
    # the placebo arm moves the concept it targets, not the real target concept
    assert p["placebo_gain"] > abs(p["real_concept_drift"])


def session_effects_active(model) -> bool:
    """True if any forward hooks remain on the model's layers."""
    for layer in model.model.layers:
        if getattr(layer, "_forward_hooks", None):
            return True
    return False


# --------------------------------------------------------------------------- #
# Outer-loop bridge: WORKSPACE_CONCEPT target scored from workspace telemetry
# --------------------------------------------------------------------------- #
def test_optimizer_scores_workspace_concept_target(env):
    from neuromod.optimization.pack_optimizer import PackOptimizer, OptimizationConfig
    from neuromod.optimization.probe_evaluator import ProbeEvaluationResult
    from neuromod.optimization.targets import BehavioralTarget
    from neuromod.pack_system import Pack, EffectConfig

    opt = PackOptimizer(model_manager=object(), config=OptimizationConfig())

    # Stub the (model-loading) evaluator with a fixed workspace telemetry result.
    def fake_eval(pack_name=None, test_prompts=None, model_name=None):
        return ProbeEvaluationResult(
            emotions={}, latent_axes={}, probe_stats={}, text_metrics={},
            overall_score=0.0, workspace={"workspace_alpha": 0.8},
        )
    opt.probe_evaluator.evaluate_with_pack = fake_eval

    target = BehavioralTarget(name="hold_alpha", description="keep alpha in workspace")
    target.add_workspace_concept_target("alpha", target_value=0.8, weight=1.0)
    pack = Pack(name="p", description="", effects=[EffectConfig(effect="jlens_steer")])

    loss_hit = opt._evaluate_pack(pack, target, ["prompt"])
    assert loss_hit == pytest.approx(0.0, abs=1e-9)

    # a mismatched target yields positive loss
    target2 = BehavioralTarget(name="hold_alpha2", description="")
    target2.add_workspace_concept_target("alpha", target_value=0.2, weight=1.0)
    loss_miss = opt._evaluate_pack(pack, target2, ["prompt"])
    assert loss_miss > 0.0
