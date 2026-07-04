"""Harmony (gpt-oss) channel parsing — pure, torch-free, unit-testable.

Harmony/reasoning models emit multiple channels in ONE token stream::

    <|channel|>analysis<|message|>{chain-of-thought}<|end|>...<|channel|>final<|message|>{answer}

The ``final`` channel is the user-facing answer; the ``analysis`` channel is the model's
chain-of-thought. Decoding with ``skip_special_tokens=True`` strips the ``<|channel|>`` markers
but leaves the analysis text glued in front of the answer (the "analysis...final..." leak), so
we split the channels explicitly instead.

Kept free of torch / transformers so it imports on any machine and can be tested without a GPU.
``api.model_manager`` re-exports :func:`split_harmony_channels`.
"""

from __future__ import annotations

import re

# A channel's message runs from its <|message|> up to the next control token (or, for a partial
# streaming buffer, end-of-string). Group 1 is the channel content.
_CHANNEL_RE = (
    r"<\|channel\|>\s*{channel}\b[^<]*<\|message\|>(.*?)"
    r"(?=<\|(?:end|return|call|channel|start)\|>|$)"
)


def _grab(raw: str, channel: str) -> str:
    m = re.search(_CHANNEL_RE.format(channel=channel), raw, re.DOTALL)
    return m.group(1).strip() if m else ""


def split_harmony_channels(raw: str):
    """Split a harmony raw decode into ``(final_text, analysis_text)``.

    Tolerant of a **partial/streaming** buffer: an unterminated channel runs to end-of-string,
    so this can be applied incrementally as tokens arrive. ``final`` is ``""`` when the final
    channel hasn't appeared yet (so a streaming caller never mistakes analysis for the answer);
    ``analysis`` is ``None`` when absent. Text with no harmony markers returns
    ``(raw.strip(), None)``.
    """
    if "<|channel|>" not in raw and "<|message|>" not in raw:
        return raw.strip(), None
    return _grab(raw, "final"), (_grab(raw, "analysis") or None)
