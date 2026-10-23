import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer

from mortgageapplication.probe_locations import create_probes, find_probe_indices


@pytest.fixture(scope="module")
def model():
    """Stand-in for HookedSAETransformer: create_probes only needs the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def to_str_tokens(text, prepend_bos=True):
        ids = tokenizer(text)["input_ids"]
        return [tokenizer.decode(i) for i in ids]

    return SimpleNamespace(tokenizer=tokenizer, to_str_tokens=to_str_tokens)


# --- find_probe_indices -----------------------------------------------------

def test_no_markers():
    assert find_probe_indices("plain text") == ("plain text", [])


def test_point_index_is_in_clean_text():
    clean, probes = find_probe_indices("Chicago<:> is")
    assert clean == "Chicago is"
    assert probes == [len("Chicago")]


def test_span_indices_are_in_clean_text():
    clean, probes = find_probe_indices("a <:big dog:> ran")
    assert clean == "a big dog ran"
    assert clean[slice(*probes[0])] == "big dog"


# --- create_probes ----------------------------------------------------------

def test_no_markers_gives_no_probes(model):
    assert create_probes("nothing to see", model) == []


def test_point_marks_token_before_marker(model):
    (p,) = create_probes("visiting Chicago<:> is great", model)
    doc = p.doc
    assert doc.clean_text == "visiting Chicago is great"
    assert doc.tokens[p.start:p.end] == [" Chicago"]
    assert p.end - p.start == 1


def test_point_char_indices_include_leading_space(model):
    (p,) = create_probes("visiting Chicago<:> is great", model)
    assert p.doc.clean_text[p.start_char_index:p.end_char_index] == " Chicago"


def test_span_covers_all_its_tokens(model):
    (p,) = create_probes("try the <:Chicago-style hotdog:>.", model)
    assert p.doc.tokens[p.start:p.end] == [" Chicago", "-", "style", " hot", "dog"]


def test_span_char_indices_inhale_leading_space(model):
    (p,) = create_probes("try the <:Chicago-style hotdog:>.", model)
    text = p.doc.clean_text
    assert text[p.start_char_index:p.end_char_index] == " Chicago-style hotdog"


def test_span_at_start_of_text(model):
    (p,) = create_probes("<:The cat:> sat", model)
    assert p.start == 0
    assert p.doc.tokens[p.start:p.end] == ["The", " cat"]
    assert p.start_char_index == 0


def test_multiple_probes_in_order_share_one_doc(model):
    point, span = create_probes(
        "visiting Chicago<:> is the <:Chicago-style hotdog:>.", model
    )
    assert point.doc is span.doc
    assert point.end <= span.start
    assert point.doc.text == "visiting Chicago<:> is the <:Chicago-style hotdog:>."


def test_marker_inside_word_raises(model):
    with pytest.raises(ValueError):
        create_probes("visiting Chi<:>cago", model)


def test_span_start_inside_word_raises(model):
    with pytest.raises(ValueError):
        create_probes("visiting Chi<:cago:> now", model)
