"""
Tests for the audit follow-up fixes: tokenizer-derived logit bias, real per-head attention
intervention, dynamic-size soft projection, and the KV-decay cache. torch-gated.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from neuromod.effects import (  # noqa: E402
    token_ids_for_words, ConceptLogitBiasProcessor,
    StyleAffectLogitBiasEffect, HeadReweightingEffect, HeadMaskingDropoutEffect,
    SoftProjectionEffect, make_decaying_cache,
)


class _FakeTok:
    def encode(self, s, add_special_tokens=False):
        s = s.strip()
        return [ord(s[0])] if s else []


# ---- item 2: tokenizer-derived concept logit bias ---------------------------------------

def test_concept_bias_is_additive_and_targets_words():
    eff = StyleAffectLogitBiasEffect(weight=1.0, bias_type="prosocial", sentiment="positive")
    eff.apply(model=object(), tokenizer=_FakeTok())
    assert eff._boost_ids, "no concept tokens derived"
    proc = eff.get_logits_processor()
    scores = torch.zeros(1, 300)
    before = scores.clone()
    out = proc(torch.tensor([[1]]), scores)
    # boosted ids got a positive additive bias
    for tid in eff._boost_ids:
        if tid < 300:
            assert out[0, tid] > before[0, tid]


def test_concept_bias_processor_additive_math():
    proc = ConceptLogitBiasProcessor(boost_ids=[2], suppress_ids=[5], bias=3.0)
    scores = torch.zeros(1, 10)
    out = proc(torch.tensor([[0]]), scores)
    assert torch.isclose(out[0, 2], torch.tensor(3.0))
    assert torch.isclose(out[0, 5], torch.tensor(-1.5))


# ---- item 3b: real per-head attention intervention --------------------------------------

class _Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
    def forward(self, x):
        return self.o_proj(x)


class _Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Attn(hidden)
    def forward(self, x):
        return self.self_attn(x)


class _Inner(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(hidden) for _ in range(n)])


class _Model(nn.Module):
    def __init__(self, n=2, hidden=8, heads=4):
        super().__init__()
        self.model = _Inner(n, hidden)
        self.config = type("cfg", (), {"num_attention_heads": heads, "hidden_size": hidden})()
    def forward_attn(self, x):
        # run through the first attn (o_proj) to observe the per-head hook effect
        return self.model.layers[0].self_attn.o_proj(x)


def test_head_reweighting_hooks_o_proj_and_changes_input():
    torch.manual_seed(0)
    model = _Model(n=2, hidden=8, heads=4)  # head_dim=2
    eff = HeadReweightingEffect(weight=1.0, routing_type="stylistic")  # heads [0,2,4,6]->[0,2]
    eff.apply(model)
    assert eff._handles, "no o_proj hooks registered"
    # The pre-hook scales heads 0 and 2 of the o_proj input; check o_proj sees a modified input.
    x = torch.ones(1, 1, 8)
    o_proj = model.model.layers[0].self_attn.o_proj
    seen = {}
    h = o_proj.register_forward_pre_hook(lambda m, a: seen.setdefault("x", a[0].clone()))
    o_proj(x)  # baseline (our effect's pre-hook runs first, then this observer)
    h.remove()
    # heads 0 and 2 (dims 0-1 and 4-5) should be boosted above 1.0; heads 1,3 stay at 1.0
    xin = seen["x"][0, 0]
    assert xin[0] > 1.0 and xin[4] > 1.0
    assert torch.isclose(xin[2], torch.tensor(1.0)) and torch.isclose(xin[6], torch.tensor(1.0))
    eff.cleanup()
    assert eff._handles == []


def test_head_masking_registers_and_cleans_up():
    model = _Model(n=3, hidden=8, heads=4)
    eff = HeadMaskingDropoutEffect(weight=1.0, dropout_type="alternating")
    eff.apply(model)
    assert len(eff._handles) == 3  # one per attention block
    eff.cleanup()
    assert eff._handles == []


# ---- item 3a: soft projection sized to the model ----------------------------------------

def test_soft_projection_dynamic_hidden_size():
    eff = SoftProjectionEffect(weight=1.0, projection_type="creative")
    P = eff._get_projection(2880, torch.device("cpu"), torch.float32)  # gpt-oss hidden size
    assert P.shape == (2880, 2880)
    # deterministic per (type, size)
    P2 = eff._get_projection(2880, torch.device("cpu"), torch.float32)
    assert torch.equal(P, P2)


# ---- item 1: KV decay cache -------------------------------------------------------------

def test_decaying_cache_scales_value_cache():
    DynamicCache = pytest.importorskip("transformers").DynamicCache
    cache = make_decaying_cache(0.5)
    assert cache is not None
    k = torch.ones(1, 2, 1, 4)
    v = torch.ones(1, 2, 1, 4)
    cache.update(k, v, 0)          # seq len 1 -> no decay (guard: >1)
    cache.update(k, v, 0)          # seq len 2 -> decay applied
    # oldest position should be decayed more than the newest
    vc = cache.value_cache[0]
    assert vc[0, 0, 0, 0] < 1.0

    assert make_decaying_cache(1.0) is None  # no decay -> no cache
    assert make_decaying_cache(None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
