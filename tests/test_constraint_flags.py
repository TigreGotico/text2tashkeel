"""A constraint flag the model cannot honour raises; it is never dropped.

rawi-v3, bilstm, libtashkeel and the ensembles whose value model is rawi-v3
decode without the orthography constraints. Before this rule they accepted the
flags and ignored them, so ``respect_existing=True`` still overwrote a human's
kasra on كِتاب.
"""
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
