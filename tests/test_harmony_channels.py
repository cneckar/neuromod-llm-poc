"""Torch-free tests for harmony (gpt-oss) channel splitting.

Locks in the behavior the streaming path relies on: only the FINAL channel is ever surfaced as
the answer, the ANALYSIS channel is captured as reasoning, and a partial/streaming buffer never
leaks analysis text into the answer.
"""

import re

from api.harmony import split_harmony_channels

FULL = ("<|channel|>analysis<|message|>Let me think about trees.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>A tree is a plant.<|return|>")


def test_plain_text_has_no_channels():
    assert split_harmony_channels("Hello world.") == ("Hello world.", None)


def test_full_stream_splits_final_and_analysis():
    final, reasoning = split_harmony_channels(FULL)
    assert final == "A tree is a plant."
    assert reasoning == "Let me think about trees."


def test_partial_analysis_only_never_leaks_as_answer():
    # Mid-analysis, before the final channel exists: final must be empty so the streaming
    # caller doesn't emit chain-of-thought as the answer.
    final, reasoning = split_harmony_channels("<|channel|>analysis<|message|>Let me think")
    assert final == ""
    assert reasoning == "Let me think"


def test_partial_final_grows_incrementally():
    # Buffer cut in the middle of the final channel (unterminated) -> final runs to end-of-buffer.
    cut = FULL[: FULL.index("A tree is a") + 8]  # "...A tree i"
    final, reasoning = split_harmony_channels(cut)
    assert final == "A tree i"
    assert reasoning == "Let me think about trees."


def test_streaming_loop_emits_clean_answer_and_captures_reasoning():
    # Faithfully replicate generate_text_stream's harmony emit loop (feed FULL one char at a
    # time; strip a trailing partial control-token fragment; emit only the growth of the final
    # channel). The assembled answer must equal the final channel exactly, with no marker leakage,
    # and the last-seen reasoning must be the analysis channel.
    emitted = 0
    answer_pieces = []
    last_reasoning = None
    buf = ""
    for ch in FULL:
        buf += ch
        final, reasoning = split_harmony_channels(buf)
        last_reasoning = reasoning if reasoning is not None else last_reasoning
        safe = re.sub(r"<(\|[a-z]*\|?)?$", "", final)   # same guard the streamer applies
        if len(safe) > emitted:
            answer_pieces.append(safe[emitted:])
            emitted = len(safe)
    answer = "".join(answer_pieces)
    assert answer == "A tree is a plant."
    assert "<|" not in answer and "analysis" not in answer  # no CoT / marker leak
    assert last_reasoning == "Let me think about trees."


def test_safe_guard_strips_partial_markers_but_keeps_prose_lt():
    # The streamer's trailing-marker guard: strip only an INCOMPLETE harmony marker at the end,
    # never a legitimate "<" inside prose.
    guard = lambda s: re.sub(r"<(\|[a-z]*\|?)?$", "", s)
    assert guard("A tree is a plant.<") == "A tree is a plant."      # start of <|return|>
    assert guard("hello<|") == "hello"
    assert guard("hello<|retur") == "hello"
    assert guard("hello<|return|") == "hello"
    assert guard("if a < b then") == "if a < b then"                 # prose "<" kept (not trailing)
    assert guard("compare a < b") == "compare a < b"                 # trailing prose, no marker
