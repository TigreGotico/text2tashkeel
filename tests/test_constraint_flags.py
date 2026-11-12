"""A constraint flag the model cannot honour raises; it is never dropped.

rawi-v3, bilstm, libtashkeel and the ensembles whose value model is rawi-v3
decode without the orthography constraints. Before this rule they accepted the
flags and ignored them, so ``respect_existing=True`` still overwrote a human's
kasra on كِتاب.
"""
import unicodedata

import pytest

from text2tashkeel import Diacritizer
from text2tashkeel._models import build_backend

IGNORING = ["rawi-v3-int8", "bilstm-int8", "rawi-v2-int8+rawi-v3-int8"]
HONOURING = ["rawi-ensemble", "rawi-v2-int8", "rawi-v2-int8+rawi-int8"]


@pytest.mark.parametrize("model", IGNORING)
@pytest.mark.parametrize("flag", ["preserve_orthography", "respect_existing", "allow_restoration"])
@pytest.mark.parametrize("value", [True, False])
def test_flag_on_a_model_that_ignores_it_raises(model, flag, value):
    with pytest.raises(ValueError, match=flag):
        Diacritizer(model, **{flag: value})
    with pytest.raises(ValueError, match=flag):
        build_backend(model, **{flag: value})


@pytest.mark.parametrize("model", IGNORING)
def test_model_without_flags_still_builds(model):
    assert Diacritizer(model).diacritize("") == ""


@pytest.mark.parametrize("model", HONOURING)
def test_flag_on_a_model_that_honours_it_is_applied(model):
    # The source spells a madda and a human's kasra; both are kept.
    d = Diacritizer(model, preserve_orthography=True, respect_existing=True)
    assert d.diacritize("آبد").startswith("آ")
    assert d.diacritize("كِتاب").startswith("كِ")


GATED_BY_BILSTM_OR_LIBTASHKEEL = [
    "bilstm+rawi", "bilstm+rawi-int8", "bilstm-int8+rawi-int8",
    "libtashkeel+rawi", "libtashkeel+rawi-int8",
    "bilstm+libtashkeel+rawi", "bilstm+libtashkeel+rawi-int8",
]


def _written(text):
    """(base character, set of combining marks on it), by Unicode decomposition."""
    pairs = []
    for c in unicodedata.normalize("NFD", text):
        if unicodedata.category(c) == "Mn":
            pairs[-1][1].add(c)
        else:
            pairs.append((c, set()))
    return pairs


@pytest.mark.parametrize("text", ["آبد القُرآن مُحَمَّد", "ذهب الوَلد"])
@pytest.mark.parametrize("model", GATED_BY_BILSTM_OR_LIBTASHKEEL)
def test_gate_keeps_the_marks_the_writing_fixed(model, text):
    # Every madda and every human mark in the source is in the output, unchanged.
    d = Diacritizer(model, preserve_orthography=True, respect_existing=True)
    src, out = _written(text), _written(d.diacritize(text))
    assert [b for b, _ in out] == [b for b, _ in src]
    expected = [(i, marks) for i, (_, marks) in enumerate(src) if marks]
    assert [(i, out[i][1]) for i, _ in expected] == expected
