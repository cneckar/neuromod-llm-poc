"""
Jacobian Lens (J-lens) and the J-space.

This module implements the interpretability technique from Gurnee, Lindsey et
al., *Verbalizable Representations Form a Global Workspace in Language Models*
(Transformer Circuits, 2026), and translates it into a runtime instrument for
this project. It is the foundation ("P0") for the J-space integration: the
surgical steering effects, the telemetry probe, and the self-exploration loop
all consume the objects defined here.

Background
----------
The paper identifies a privileged subspace of a model's representations -- the
concepts it is *poised to verbalize* -- and shows this subspace behaves like a
cognitive "global workspace": it supports verbal report, directed modulation,
internal reasoning, flexible generalization, and selectivity (the model can
parse text and recall facts with it suppressed, but struggles to reason).

For a concept token ``c`` and a layer ``l``, the J-lens vector is the *average
linearized effect* of the layer-``l`` residual activation on the model's
log-likelihood of producing ``c`` now or at any later position, averaged over a
corpus of contexts::

    v_lens(c, l) = E_context [ d/d h_l ( sum_q logit_{q, c} ) ]

The ``sum_q`` (over positions) captures "now or in the future" via the causal
structure of the network: the activation at position ``p`` can only influence
logits at positions ``q >= p``. Averaging over contexts is what distinguishes a
representation that is *verbalizable* (poised to be spoken, should the occasion
arise) from one that merely happens to be verbalized in a single context.

The span of the ``v_lens`` vectors over a concept set is the **J-space**. This
module exposes:

* :class:`JacobianLens` -- fits ``v_lens`` vectors from a model + corpus.
* :class:`JSpaceBasis` -- the fitted, serializable artifact (pure tensors, no
  model needed). Provides ``readout`` (rank the concepts a state is poised to
  verbalize) and the three faithful interventions from the paper:

  - **steer / write**: ``h <- h + alpha * v_lens(c)``
  - **ablate**: project out a concept (negative steer / span removal)
  - **swap**: exchange two concepts' lens coordinates via the pseudoinverse,
    leaving the orthogonal complement of the activation untouched (Fig. 4C).

The interventions are also available as backend-light module-level functions so
that the P1 ``Effect`` subclasses and the unit tests share exactly one
implementation of the linear algebra.

Notes on faithfulness / limitations
------------------------------------
* The lens is a *linear, corpus-averaged* approximation. It is a principled
  refinement of the logit lens (it corrects for cross-layer representational
  drift, which is why it reads meaning in earlier/middle layers where the logit
  lens is uninterpretable), but it can still misreport. Anything that optimizes
  *against* the lens must hold out an independent behavioral check.
* Per the paper's own disclaimer, not all cognition routes through the J-space;
  sufficiently automatic or well-practiced computation can proceed beneath it.
  A concept's *absence* from the readout is therefore weak evidence.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# A small, generic corpus used when the caller does not supply one. The lens is
# meant to average over *contexts*; a handful of neutral sentences is enough for
# a smoke test / CPU run, but real use should pass a larger, task-relevant
# corpus (see JacobianLens.fit).
DEFAULT_CORPUS: List[str] = [
    "The weather today is calm and the streets are quiet.",
    "She opened the book and began to read the first chapter.",
    "The engineers discussed the plan before starting the project.",
    "A river runs through the valley toward the distant sea.",
    "He carefully measured the ingredients and mixed them together.",
    "The teacher explained the problem step by step to the class.",
    "Music drifted through the open window on a summer evening.",
    "They walked along the shore, watching the waves come in.",
    "The report summarized the findings in three short paragraphs.",
    "An old clock ticked softly in the corner of the room.",
    "The committee reviewed the proposal and asked several questions.",
    "Bright stars appeared as the sky darkened over the mountains.",
    "The mechanic inspected the engine and replaced a worn part.",
    "Children played in the park while their parents talked nearby.",
    "The scientist recorded the temperature every hour of the day.",
    "A gentle breeze moved the leaves of the tall oak tree.",
]


# --------------------------------------------------------------------------- #
# Backend-light intervention primitives.
#
# All operate on the last dimension (= hidden size). ``h`` may be a single
# vector ``[d]`` or a batch ``[..., d]``; concept vectors are ``[d]``. These are
# the single source of truth for the linear algebra; JSpaceBasis and the P1
# effects call these, and the unit tests assert their mathematical properties.
# --------------------------------------------------------------------------- #
def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def steer(h: torch.Tensor, v: torch.Tensor, alpha: float = 1.0,
          normalize: bool = True) -> torch.Tensor:
    """Write along a concept direction: ``h + alpha * v`` (v unit-normalized)."""
    d = _unit(v) if normalize else v
    return h + alpha * d


def ablate(h: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project out the component of ``h`` along a single concept direction."""
    u = _unit(v, eps)
    coeff = (h * u).sum(dim=-1, keepdim=True)
    return h - coeff * u


def _orthonormal_rows(M: torch.Tensor) -> torch.Tensor:
    """Return orthonormal rows spanning the row space of ``M`` ([k, d], k<=d)."""
    # QR on M^T gives Q with orthonormal columns spanning col-space(M^T)=row-space(M).
    q, _ = torch.linalg.qr(M.transpose(-1, -2))     # q: [d, k]
    return q.transpose(-1, -2).contiguous()          # [k, d]


def ablate_span(h: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Project out the component of ``h`` in the span of the rows of ``M``."""
    if M.ndim != 2 or M.shape[0] == 0:
        return h
    Q = _orthonormal_rows(M)                          # [k, d]
    proj = (h @ Q.transpose(-1, -2)) @ Q             # [..., d]
    return h - proj


def swap(h: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Exchange two concepts' lens coordinates (paper Fig. 4C).

    ``V`` is ``[d, 2]`` holding the two lens vectors as columns. We read the
    activation's coordinates in that basis via the Moore-Penrose pseudoinverse,
    swap them, and write the result back. Because ``V @ pinv(V)`` is the
    orthogonal projector onto ``span(V)``, everything orthogonal to the two
    vectors is left exactly unchanged.
    """
    Vp = torch.linalg.pinv(V)                         # [2, d]
    c = h @ Vp.transpose(-1, -2)                      # [..., 2]
    c_swapped = c.flip(-1)                            # exchange the two coords
    delta = (c_swapped - c) @ V.transpose(-1, -2)    # [..., d], lies in span(V)
    return h + delta


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class JLensConfig:
    """Configuration for fitting a Jacobian lens.

    Attributes
    ----------
    layers:
        Which residual-stream layers to fit. ``"workspace_band"`` selects the
        middle band the paper associates with workspace behavior (see ``band``);
        ``"mid"``, ``"all"``, or an explicit list of hidden-state indices are
        also accepted. Indices refer to ``output_hidden_states`` positions,
        where 0 is the embedding output and ``L`` is the final layer.
    band:
        Fractional ``(lo, hi)`` of the layer stack used when
        ``layers == "workspace_band"``. The default drops the earliest layers
        (still parsing tokens) and the very last (collapsing to output tokens).
    position_reduce:
        How to aggregate the per-position gradient into one vector: ``"mean"``
        or ``"sum"`` over context positions.
    max_context_tokens:
        Truncate contexts to this many tokens (keeps CPU fitting tractable).
    normalize_readout:
        If True, ``readout`` ranks by cosine-like scores (unit lens vectors).
    """
    layers: Union[str, Sequence[int]] = "workspace_band"
    band: Tuple[float, float] = (0.25, 0.9)
    position_reduce: str = "mean"
    max_context_tokens: int = 64
    normalize_readout: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layers": list(self.layers) if not isinstance(self.layers, str) else self.layers,
            "band": list(self.band),
            "position_reduce": self.position_reduce,
            "max_context_tokens": self.max_context_tokens,
            "normalize_readout": self.normalize_readout,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JLensConfig":
        d = dict(d or {})
        if "band" in d and d["band"] is not None:
            d["band"] = tuple(d["band"])
        return cls(**d)


# --------------------------------------------------------------------------- #
# The fitted artifact
# --------------------------------------------------------------------------- #
class JSpaceBasis:
    """A fitted J-space: concept lens vectors over a set of layers.

    This is a pure-tensor artifact -- it does not hold a reference to the model,
    so it can be serialized and reloaded for use by effects/probes at runtime.
    """

    def __init__(self,
                 concepts: Sequence[str],
                 token_ids: Sequence[int],
                 layer_indices: Sequence[int],
                 vectors: torch.Tensor,
                 model_name: str = "",
                 normalize_readout: bool = True,
                 meta: Optional[Dict[str, Any]] = None):
        # vectors: [num_concepts, num_layers, hidden_size]
        if vectors.ndim != 3:
            raise ValueError(f"vectors must be [C, L, d], got shape {tuple(vectors.shape)}")
        C, L, _ = vectors.shape
        if len(concepts) != C or len(token_ids) != C:
            raise ValueError("concepts/token_ids length must match vectors dim 0")
        if len(layer_indices) != L:
            raise ValueError("layer_indices length must match vectors dim 1")

        self.concepts: List[str] = list(concepts)
        self.token_ids: List[int] = [int(t) for t in token_ids]
        self.layer_indices: List[int] = [int(i) for i in layer_indices]
        self.vectors: torch.Tensor = vectors.detach().to(torch.float32)
        self.model_name = model_name
        self.normalize_readout = normalize_readout
        self.meta: Dict[str, Any] = dict(meta or {})

        self._concept_ix = {c: i for i, c in enumerate(self.concepts)}
        self._layer_ix = {l: i for i, l in enumerate(self.layer_indices)}

    # -- shape / lookup helpers -------------------------------------------- #
    @property
    def num_concepts(self) -> int:
        return self.vectors.shape[0]

    @property
    def num_layers(self) -> int:
        return self.vectors.shape[1]

    @property
    def hidden_size(self) -> int:
        return self.vectors.shape[2]

    def has_layer(self, layer: int) -> bool:
        return layer in self._layer_ix

    def _resolve_layer(self, layer: Optional[int]) -> int:
        """Map an absolute hidden-state index to a row in ``vectors``.

        ``None`` selects the deepest fitted layer (most abstract workspace
        content). An exact match is required otherwise, except that we snap to
        the nearest fitted layer to be forgiving of off-by-one callers.
        """
        if layer is None:
            return self.num_layers - 1
        if layer in self._layer_ix:
            return self._layer_ix[layer]
        nearest = min(self.layer_indices, key=lambda l: abs(l - layer))
        logger.debug("Layer %s not fitted; snapping to nearest fitted layer %s",
                     layer, nearest)
        return self._layer_ix[nearest]

    def vector(self, concept: str, layer: Optional[int] = None) -> torch.Tensor:
        if concept not in self._concept_ix:
            raise KeyError(f"concept {concept!r} not in basis; have {self.concepts}")
        return self.vectors[self._concept_ix[concept], self._resolve_layer(layer)]

    def matrix(self, layer: Optional[int] = None,
               concepts: Optional[Sequence[str]] = None) -> torch.Tensor:
        """All (or a subset of) concept vectors at ``layer`` as ``[k, d]``."""
        li = self._resolve_layer(layer)
        if concepts is None:
            return self.vectors[:, li, :]
        idx = [self._concept_ix[c] for c in concepts]
        return self.vectors[idx, li, :]

    # -- readout ("what is it poised to verbalize?") ----------------------- #
    def readout(self, h: torch.Tensor, layer: Optional[int] = None,
                top_k: Optional[int] = None,
                normalize: Optional[bool] = None) -> List[Tuple[str, float]]:
        """Rank concepts by how strongly ``h`` is poised to verbalize them.

        ``h`` is a single hidden state ``[d]`` at ``layer``. Returns
        ``(concept, score)`` pairs sorted descending.
        """
        if normalize is None:
            normalize = self.normalize_readout
        M = self.matrix(layer)                     # [C, d]
        if normalize:
            M = _unit(M)
        h = h.to(M.dtype)
        scores = (M @ h).tolist()                  # [C]
        pairs = sorted(zip(self.concepts, scores), key=lambda p: p[1], reverse=True)
        return pairs[:top_k] if top_k else pairs

    def occupancy(self, h: torch.Tensor, layer: Optional[int] = None) -> float:
        """Fraction of ``h``'s norm that lies inside the J-space at ``layer``.

        A scalar "how much of what it's doing right now is workspace content"
        readout, useful as a grounded telemetry axis (P2).
        """
        inside = self.project_in(h, layer)
        denom = float(h.norm()) + 1e-8
        return float(inside.norm() / denom)

    # -- geometry ---------------------------------------------------------- #
    def project_in(self, h: torch.Tensor, layer: Optional[int] = None) -> torch.Tensor:
        """Component of ``h`` inside the J-space span at ``layer``."""
        M = self.matrix(layer)
        Q = _orthonormal_rows(M)
        return (h @ Q.transpose(-1, -2)) @ Q

    # -- interventions ----------------------------------------------------- #
    def steer(self, h: torch.Tensor, concept: str, layer: Optional[int] = None,
              alpha: float = 1.0, normalize: bool = True) -> torch.Tensor:
        return steer(h, self.vector(concept, layer), alpha=alpha, normalize=normalize)

    def ablate(self, h: torch.Tensor, concept: str,
               layer: Optional[int] = None) -> torch.Tensor:
        return ablate(h, self.vector(concept, layer))

    def ablate_concepts(self, h: torch.Tensor, concepts: Sequence[str],
                        layer: Optional[int] = None) -> torch.Tensor:
        if not concepts:
            return h
        return ablate_span(h, self.matrix(layer, concepts))

    def ablate_top_k(self, h: torch.Tensor, k: int,
                     layer: Optional[int] = None) -> torch.Tensor:
        """Suppress the ``k`` concepts the state is currently most poised to say.

        This is the paper's "suppress the top-k J-space contents" knob -- the
        surgical analogue of a reasoning-dampening intervention.
        """
        ranked = self.readout(h, layer=layer, top_k=k)
        top = [c for c, _ in ranked]
        return self.ablate_concepts(h, top, layer)

    def swap(self, h: torch.Tensor, src: str, tgt: str,
             layer: Optional[int] = None) -> torch.Tensor:
        V = torch.stack([self.vector(src, layer), self.vector(tgt, layer)], dim=-1)  # [d,2]
        return swap(h, V)

    # -- (de)serialization ------------------------------------------------- #
    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "jspace_basis_v1",
            "concepts": self.concepts,
            "token_ids": self.token_ids,
            "layer_indices": self.layer_indices,
            "vectors": self.vectors,
            "model_name": self.model_name,
            "normalize_readout": self.normalize_readout,
            "meta": self.meta,
        }, path)
        logger.info("Saved J-space basis (%d concepts x %d layers, d=%d) to %s",
                    self.num_concepts, self.num_layers, self.hidden_size, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JSpaceBasis":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") != "jspace_basis_v1":
            raise ValueError(f"unrecognized J-space basis file: {path}")
        return cls(
            concepts=blob["concepts"],
            token_ids=blob["token_ids"],
            layer_indices=blob["layer_indices"],
            vectors=blob["vectors"],
            model_name=blob.get("model_name", ""),
            normalize_readout=blob.get("normalize_readout", True),
            meta=blob.get("meta", {}),
        )

    def __repr__(self) -> str:
        return (f"JSpaceBasis(model={self.model_name!r}, concepts={self.num_concepts}, "
                f"layers={self.layer_indices}, d={self.hidden_size})")


# --------------------------------------------------------------------------- #
# The fitter
# --------------------------------------------------------------------------- #
class JacobianLens:
    """Fits Jacobian-lens vectors for a set of concepts from a causal LM.

    Only relies on the standard HuggingFace causal-LM interface -- a call
    ``model(inputs_embeds=..., output_hidden_states=True)`` returning an object
    with ``.logits`` ``[1, seq, vocab]`` and ``.hidden_states`` (a tuple of
    ``L+1`` tensors) -- so it works uniformly across GPT-2, Llama, Qwen, etc.
    without architecture-specific block surgery.
    """

    def __init__(self, model, tokenizer, config: Optional[JLensConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or JLensConfig()

    # -- device / dtype ---------------------------------------------------- #
    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    # -- layer selection --------------------------------------------------- #
    def _select_layers(self, n_hidden_states: int) -> List[int]:
        L = n_hidden_states - 1  # index L == final layer output; 0 == embeddings
        cfg = self.config
        if isinstance(cfg.layers, str):
            if cfg.layers == "all":
                idx = list(range(1, L + 1))
            elif cfg.layers == "mid":
                idx = list(range(max(1, L // 3), max(2, (2 * L) // 3) + 1))
            elif cfg.layers == "workspace_band":
                lo = max(1, int(round(cfg.band[0] * L)))
                hi = min(L, int(round(cfg.band[1] * L)))
                idx = list(range(lo, hi + 1))
            else:
                raise ValueError(f"unknown layers spec {cfg.layers!r}")
        else:
            idx = [int(i) for i in cfg.layers if 1 <= int(i) <= L]
        if not idx:
            idx = [L]
        return idx

    # -- concept -> token id ---------------------------------------------- #
    def resolve_token_id(self, concept: str) -> int:
        """Map a concept string to a single vocabulary token id.

        Prefers the space-prefixed form (how the token usually appears
        mid-sentence) and warns when a concept is multi-token (we then use the
        first token, which is a lossy but standard choice).
        """
        candidates = [" " + concept, concept]
        best: Optional[List[int]] = None
        for cand in candidates:
            try:
                ids = self.tokenizer.encode(cand, add_special_tokens=False)
            except TypeError:
                ids = self.tokenizer.encode(cand)
            if not ids:
                continue
            if len(ids) == 1:
                return int(ids[0])
            if best is None:
                best = ids
        if best is None:
            raise ValueError(f"could not tokenize concept {concept!r}")
        logger.warning("concept %r is multi-token %s; using first token %d",
                       concept, best, best[0])
        return int(best[0])

    # -- input embedding path (robust to frozen params) -------------------- #
    def _inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.model.get_input_embeddings()
        inputs_embeds = emb(input_ids)
        inputs_embeds.requires_grad_(True)
        inputs_embeds.retain_grad()
        return inputs_embeds

    def _encode(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_context_tokens,
        )
        return tok["input_ids"].to(self._device())

    # -- fit --------------------------------------------------------------- #
    def fit(self,
            concepts: Sequence[str],
            corpus: Optional[Sequence[str]] = None,
            token_ids: Optional[Sequence[int]] = None) -> JSpaceBasis:
        """Fit lens vectors for ``concepts`` by averaging over ``corpus``.

        Parameters
        ----------
        concepts:
            Concept strings (e.g. ``["honest", "careful", "deception"]``).
        corpus:
            Contexts to average the linearized effect over. Defaults to
            :data:`DEFAULT_CORPUS`. Use a larger, representative corpus for real
            work -- a handful of sentences is only enough for a smoke test.
        token_ids:
            Optional explicit vocabulary ids, one per concept, overriding
            :meth:`resolve_token_id` (useful for multi-token concepts).
        """
        concepts = list(concepts)
        if not concepts:
            raise ValueError("need at least one concept")
        corpus = list(corpus) if corpus is not None else list(DEFAULT_CORPUS)
        if token_ids is None:
            token_ids = [self.resolve_token_id(c) for c in concepts]
        token_ids = [int(t) for t in token_ids]

        was_training = self.model.training
        self.model.eval()

        sums: Optional[torch.Tensor] = None       # [C, L_sel, d]
        layer_indices: Optional[List[int]] = None
        count = 0

        try:
            for text in corpus:
                input_ids = self._encode(text)
                if input_ids.numel() == 0:
                    continue
                with torch.enable_grad():
                    inputs_embeds = self._inputs_embeds(input_ids)
                    out = self.model(inputs_embeds=inputs_embeds,
                                     output_hidden_states=True, use_cache=False)
                    logits = out.logits            # [1, seq, V]
                    hs = out.hidden_states         # tuple len L+1

                    if layer_indices is None:
                        layer_indices = self._select_layers(len(hs))
                        hidden_size = hs[0].shape[-1]
                        sums = torch.zeros(len(concepts), len(layer_indices),
                                           hidden_size, dtype=torch.float32)
                    selected = [hs[i] for i in layer_indices]

                    for ci, tid in enumerate(token_ids):
                        # S = total tendency to emit this concept anywhere in the
                        # context. grad wrt h_l at position p is the summed effect
                        # over all reachable future positions (causal) -> the
                        # "now or in future" linearized effect.
                        S = logits[0, :, tid].sum()
                        grads = torch.autograd.grad(
                            S, selected,
                            retain_graph=(ci < len(token_ids) - 1),
                            create_graph=False, allow_unused=False,
                        )
                        for li, g in enumerate(grads):
                            gp = g[0]              # [seq, d]
                            red = gp.mean(0) if self.config.position_reduce == "mean" else gp.sum(0)
                            sums[ci, li] += red.detach().to(torch.float32).cpu()
                count += 1
        finally:
            if was_training:
                self.model.train()

        if sums is None or count == 0:
            raise RuntimeError("fit produced no data (empty corpus?)")

        vectors = sums / count
        meta = {
            "corpus_size": count,
            "corpus_hash": _hash_corpus(corpus),
            "config": self.config.to_dict(),
        }
        model_name = getattr(getattr(self.model, "config", None), "_name_or_path", "") or ""
        basis = JSpaceBasis(
            concepts=concepts,
            token_ids=token_ids,
            layer_indices=layer_indices,
            vectors=vectors,
            model_name=model_name,
            normalize_readout=self.config.normalize_readout,
            meta=meta,
        )
        logger.info("Fitted %s over %d contexts", basis, count)
        return basis


def _hash_corpus(corpus: Sequence[str]) -> str:
    h = hashlib.sha256()
    for line in corpus:
        h.update(line.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
