"""
Closed-loop workspace control and self-exploration (P3).

This module closes the loop opened by P0-P2: it lets an agent steer its *internal
workspace state* (which concepts it holds in mind, per the Jacobian lens) toward a
desired target in real time, and packages that into a self-exploration session --
the model hypothesizes a pack from a desired internal state, applies it to itself,
observes its own workspace via the lens, and tunes the pack toward intent.

Two layers:

* :class:`WorkspaceController` -- the inner loop. Given a live model and a
  :class:`WorkspaceTarget`, it installs one :class:`JLensSteerEffect` per target
  concept and adjusts their intensities with a gradient-free proportional update
  until the measured workspace matches the target. Fully reversible (``close()``
  removes every effect).

* :class:`SelfExplorationSession` -- the outer harness. It hypothesizes a pack
  from a target, runs the controller, and then *verifies* with guards designed to
  keep the loop honest:

    - **Goodhart hold-out.** Convergence is scored against the lens, a linear
      approximation. We additionally check an independent signal -- the concept
      token's probability in the model's actual output distribution, which the
      lens does not directly optimize. If the lens score rose but the hold-out did
      not, we flag it. Optimizing the instrument is not optimizing the state.
    - **Coverage honesty.** Per the paper, not all cognition routes through the
      J-space; a concept's absence from the readout is weak evidence. Convergence
      means "as measured in the workspace," and we say so.
    - **Containment.** Bounded iterations, fully reversible effects, a complete
      trajectory returned for audit, and an optional placebo arm that steers an
      *untargeted* concept to show the loop moves what it targets rather than
      everything (i.e. it is not merely fooling itself).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .effects import JLensSteerEffect

logger = logging.getLogger(__name__)


COVERAGE_NOTE = (
    "Convergence is measured in the J-space (a linear, corpus-averaged lens). Per "
    "the source paper, not all cognition routes through the workspace, so this is a "
    "necessary-not-sufficient signal and a concept's absence is weak evidence. The "
    "Goodhart hold-out below is an independent behavioral check."
)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _input_ids(tokenizer, prompt, device):
    enc = tokenizer(prompt, return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "__getitem__") else enc.input_ids
    return ids.to(device)


def measure_workspace(model, tokenizer, basis, prompt, layer: Optional[int] = None):
    """Run one forward pass (with any applied effects active) and read the workspace.

    Returns ``(scores, occupancy, last_logits)`` where ``scores`` maps each basis
    concept to its normalized lens readout at ``layer`` (deepest fitted layer by
    default) for the last token, ``occupancy`` is the J-space occupancy, and
    ``last_logits`` is the raw output logits at the last position.
    """
    jl = layer if layer is not None else basis.layer_indices[-1]
    device = _model_device(model)
    input_ids = _input_ids(tokenizer, prompt, device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    hs = out.hidden_states
    h = (hs[jl] if jl < len(hs) else hs[-1])[0, -1, :].detach().to(torch.float32).cpu()
    scores = {c: float(s) for c, s in basis.readout(h, layer=jl)}
    occ = float(basis.occupancy(h, layer=jl))
    return scores, occ, out.logits[0, -1, :].detach()


@dataclass
class WorkspaceTarget:
    """A desired internal workspace state.

    ``concepts`` maps concept -> target normalized lens score (roughly [-1, 1]).
    ``occupancy`` optionally targets the overall J-space occupancy (0-1).
    """
    concepts: Dict[str, float] = field(default_factory=dict)
    occupancy: Optional[float] = None

    def concept_error(self, measured: Dict[str, float]) -> Dict[str, float]:
        return {c: (t - measured.get(c, 0.0)) for c, t in self.concepts.items()}

    def max_abs_error(self, measured_scores: Dict[str, float],
                      measured_occ: Optional[float] = None) -> float:
        errs = [abs(t - measured_scores.get(c, 0.0)) for c, t in self.concepts.items()]
        if self.occupancy is not None and measured_occ is not None:
            errs.append(abs(self.occupancy - measured_occ))
        return max(errs) if errs else 0.0


class WorkspaceController:
    """Inner loop: tune per-concept J-lens steering so the model's measured
    workspace matches ``target``. Gradient-free proportional control; reversible.
    """

    def __init__(self, model, tokenizer, basis, target: WorkspaceTarget,
                 probe_prompt: str, layer: Optional[int] = None,
                 step: float = 0.3, max_iters: int = 25, tolerance: float = 0.03,
                 weight_bounds=(0.0, 5.0), alpha_max: float = 0.35):
        self.model = model
        self.tokenizer = tokenizer
        self.basis = basis
        self.target = target
        self.probe_prompt = probe_prompt
        self.layer = layer if layer is not None else basis.layer_indices[-1]
        self.step = step                 # bootstrap probe size (secant thereafter)
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.weight_bounds = weight_bounds
        self.alpha_max = alpha_max
        self.effects: Dict[str, JLensSteerEffect] = {}
        self.trajectory: List[Dict[str, Any]] = []
        self._prev: Dict[str, tuple] = {}  # concept -> (weight, score) from last step
        self._install()

    def _install(self):
        for concept in self.target.concepts:
            if concept not in self.basis.concepts:
                logger.warning("WorkspaceController: concept %r not in basis; skipping", concept)
                continue
            eff = JLensSteerEffect(weight=0.0, direction="up", concepts=[concept],
                                   alpha_max=self.alpha_max)
            eff.basis = self.basis  # share the fitted basis; skip re-fitting
            eff.apply(self.model, tokenizer=self.tokenizer)
            self.effects[concept] = eff

    def measure(self):
        return measure_workspace(self.model, self.tokenizer, self.basis,
                                 self.probe_prompt, self.layer)

    def _update(self, scores: Dict[str, float]):
        """Secant (Newton) update: estimate d(score)/d(weight) from the last two
        measurements and step straight to the target; bootstrap with a fixed probe
        in the error's direction when no slope is known yet. Robust to the (large,
        model-dependent) gain that a fixed-step controller would over/undershoot."""
        lo, hi = self.weight_bounds
        for concept, target_val in self.target.concepts.items():
            eff = self.effects.get(concept)
            if eff is None:
                continue
            cur_w = eff.weight
            cur_s = scores.get(concept, 0.0)
            err = target_val - cur_s
            prev = self._prev.get(concept)
            if prev is not None and abs(cur_w - prev[0]) > 1e-9:
                slope = (cur_s - prev[1]) / (cur_w - prev[0])
            else:
                slope = None
            if slope is not None and abs(slope) > 1e-6:
                new_w = cur_w + err / slope                 # Newton step
            else:
                new_w = cur_w + self.step * (1.0 if err >= 0 else -1.0)  # bootstrap probe
            self._prev[concept] = (cur_w, cur_s)
            eff.weight = float(min(hi, max(lo, new_w)))

    def run(self) -> Dict[str, Any]:
        converged = False
        for i in range(self.max_iters):
            scores, occ, _ = self.measure()
            max_err = self.target.max_abs_error(scores, occ)
            self.trajectory.append({
                "iter": i, "scores": scores, "occupancy": occ,
                "max_error": max_err,
                "weights": {c: e.weight for c, e in self.effects.items()},
            })
            if max_err <= self.tolerance:
                converged = True
                break
            self._update(scores)

        final_scores, final_occ, _ = self.measure()
        converged = converged or (self.target.max_abs_error(final_scores, final_occ) <= self.tolerance)
        return {
            "converged": converged,
            "iterations": len(self.trajectory),
            "final_scores": final_scores,
            "final_occupancy": final_occ,
            "weights": {c: e.weight for c, e in self.effects.items()},
            "trajectory": self.trajectory,
        }

    def close(self):
        for eff in self.effects.values():
            try:
                eff.cleanup()
            except Exception:  # pragma: no cover - defensive
                pass
        self.effects = {}


@dataclass
class SelfExplorationReport:
    target: WorkspaceTarget
    converged: bool
    iterations: int
    baseline_scores: Dict[str, float]
    final_scores: Dict[str, float]
    weights: Dict[str, float]
    holdout: Dict[str, Dict[str, float]]   # concept -> {baseline_prob, final_prob, delta}
    goodhart_flags: Dict[str, bool]
    coverage_note: str
    trajectory: List[Dict[str, Any]]
    placebo: Optional[Dict[str, Any]] = None

    @property
    def goodhart_suspected(self) -> bool:
        return any(self.goodhart_flags.values())


class SelfExplorationSession:
    """Outer harness: hypothesize -> apply-to-self -> observe -> tune -> verify."""

    def __init__(self, model, tokenizer, basis, probe_prompt: str,
                 goodhart_lens_threshold: float = 0.02, **controller_kw):
        self.model = model
        self.tokenizer = tokenizer
        self.basis = basis
        self.probe_prompt = probe_prompt
        self.goodhart_lens_threshold = goodhart_lens_threshold
        self.controller_kw = controller_kw

    def _token_id(self, concept: str) -> Optional[int]:
        try:
            return self.basis.token_ids[self.basis.concepts.index(concept)]
        except (ValueError, IndexError):
            return None

    def _holdout_prob(self, concept: str, logits: torch.Tensor) -> float:
        tid = self._token_id(concept)
        if tid is None or tid < 0 or tid >= logits.shape[-1]:
            return 0.0
        return float(torch.softmax(logits.to(torch.float32), dim=-1)[tid])

    def run(self, target: WorkspaceTarget, with_placebo: bool = True) -> SelfExplorationReport:
        # Baseline: measure with no effects applied.
        base_scores, _, base_logits = measure_workspace(
            self.model, self.tokenizer, self.basis, self.probe_prompt)
        base_holdout = {c: self._holdout_prob(c, base_logits) for c in target.concepts}

        # Hypothesize (one steer effect per concept) + tune.
        ctrl = WorkspaceController(self.model, self.tokenizer, self.basis, target,
                                   self.probe_prompt, **self.controller_kw)
        try:
            result = ctrl.run()
            final_scores = result["final_scores"]
            _, _, final_logits = ctrl.measure()
        finally:
            ctrl.close()  # reversible: effects removed regardless of outcome

        # Goodhart hold-out: did the *real output* move, not just the lens?
        holdout, goodhart = {}, {}
        for c in target.concepts:
            fp = self._holdout_prob(c, final_logits)
            bp = base_holdout[c]
            holdout[c] = {"baseline_prob": bp, "final_prob": fp, "delta": fp - bp}
            lens_gain = final_scores.get(c, 0.0) - base_scores.get(c, 0.0)
            goodhart[c] = (lens_gain > self.goodhart_lens_threshold) and ((fp - bp) <= 0.0)

        placebo = self._placebo_arm(target, base_scores) if with_placebo else None

        return SelfExplorationReport(
            target=target,
            converged=result["converged"],
            iterations=result["iterations"],
            baseline_scores=base_scores,
            final_scores=final_scores,
            weights=result["weights"],
            holdout=holdout,
            goodhart_flags=goodhart,
            coverage_note=COVERAGE_NOTE,
            trajectory=result["trajectory"],
            placebo=placebo,
        )

    def _placebo_arm(self, target: WorkspaceTarget,
                     base_scores: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Steer an *untargeted* concept and confirm the loop moves what it targets
        (the placebo concept), not the real target concept -- evidence of specificity
        rather than the loop congratulating itself."""
        untargeted = [c for c in self.basis.concepts if c not in target.concepts]
        if not untargeted or not target.concepts:
            return None
        placebo_concept = untargeted[0]
        watched = next(iter(target.concepts))
        ptarget = WorkspaceTarget(concepts={placebo_concept: base_scores.get(placebo_concept, 0.0) + 0.3})
        ctrl = WorkspaceController(self.model, self.tokenizer, self.basis, ptarget,
                                   self.probe_prompt, **self.controller_kw)
        try:
            res = ctrl.run()
            final = res["final_scores"]
        finally:
            ctrl.close()
        return {
            "placebo_concept": placebo_concept,
            "watched_real_concept": watched,
            "placebo_gain": final.get(placebo_concept, 0.0) - base_scores.get(placebo_concept, 0.0),
            "real_concept_drift": final.get(watched, 0.0) - base_scores.get(watched, 0.0),
        }
