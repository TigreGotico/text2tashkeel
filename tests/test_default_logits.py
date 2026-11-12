"""The default model exposes its per-character class distribution.

A caller such as arbtok's fusion scores readings with ``logits()`` and needs the
argmax of that distribution to be the decision ``diacritize()`` makes.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from text2tashkeel import DEFAULT_MODEL, Diacritizer
from text2tashkeel._models import build_backend

# 200 sentences of the val split of TigreGotico/arabic_diacritized_text, with
# the harakat removed. The benchmark scores the test split, not this one.
SENTENCES = (Path(__file__).parent / "data" / "val_sentences_200.txt") \
    .read_text(encoding="utf-8").splitlines()


def _skeleton(text):
    """Letters only: the hamza and madda marks go too, so no mark constrains."""
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn"))


def test_fixture_has_200_sentences():
    assert len(SENTENCES) == 200 == len(set(SENTENCES))


def test_argmax_of_logits_is_the_default_diacritize():
    d = Diacritizer()
    for text in map(_skeleton, SENTENCES):
        bare, logits, classes = d.logits(text)
        assert logits.shape == (len(bare), len(classes))
        assert d.decode(bare, logits.argmax(-1)) == d.diacritize(text), text


def test_argmax_of_logits_is_the_unconstrained_diacritize():
    d = Diacritizer(preserve_orthography=False, respect_existing=False)
    for text in SENTENCES:
        bare, logits, _ = d.logits(text)
        assert d.decode(bare, logits.argmax(-1)) == d.diacritize(text), text


def test_logit_rows_are_the_rawi_v3_value_head():
    """Second engine: the separate rawi-v3 and rawi-v2 INT8 models."""
    value = build_backend("rawi-v3-int8")
    gate = build_backend("rawi-v2-int8")
    d = Diacritizer(DEFAULT_MODEL)
    for text in SENTENCES[:50]:
        bare, logits, _ = d.logits(text)
        ids = np.array([[value.c2i.get(c, value.unk) for c in bare]], dtype=np.int64)
        v = value.sess.run(["value"], {"input": ids})[0][0]
        g = gate.sess.run(["output"], {"input": ids})[0][0].argmax(-1)
        np.testing.assert_array_equal(logits[:, 1:], v[:, 1:])
        open_ = g != 0
        np.testing.assert_array_equal(logits[open_, 0], v[open_, 0])
        assert (logits[~open_].argmax(-1) == 0).all()


@pytest.mark.parametrize("text", ["", "ًٌٍَُِّْ"])
def test_logits_on_nothing_to_mark(text):
    bare, logits, classes = Diacritizer().logits(text)
    assert bare == ""
    assert logits is None
    assert len(classes) == 75
