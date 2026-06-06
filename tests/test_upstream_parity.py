"""Pin the ported backends to their upstream reference outputs.

These strings come directly from each project's own published example, so if a
port ever drifts (wrong vocab, wrong normalization, wrong decode), it breaks
here. Compared under NFC since combining-mark order is cosmetic.
"""

import unicodedata

from text2tashkeel import Diacritizer


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def test_libtashkeel_matches_readme_example():
    # From mush42/libtashkeel README / Rust test `test_basic_tashkeel`.
    d = Diacritizer("libtashkeel")
    got = d.diacritize("بسم الله الرحمن الرحيم")
    expected = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم"
    assert _nfc(got) == _nfc(expected)


def test_rawi_matches_readme_sample():
    # From TigreGotico/rawi model card, "Sample 1" predicted output.
    d = Diacritizer("rawi")
    undiacritized = (
        "رواه البخارى فى الصحيح عن ادم بن ابى اياس واخرجه مسلم "
        "من وجهين اخرين عن شعبة ."
    )
    expected = (
        "رَوَاَهُ أَلْبُخَأْرِىُّ فِىِ أَلْصَّحِيْحِ عَنْ آدَمَ بْنِ أَبِىِّ "
        "إِيَأْسٍ وَأَخْرَجَهُ مُسْلِمٌ مِنْ وَجْهَيْنِ آخَرِيْنِ عَنْ شُعْبَةَ ."
    )
    assert _nfc(d.diacritize(undiacritized)) == _nfc(expected)
