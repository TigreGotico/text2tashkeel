"""text2tashkeel — tiny standalone Arabic diacritizer (tashkeel).

Adds the missing vowel marks to Arabic text. Pure onnxruntime + numpy — no
torch, no network. Ships several interchangeable models, each self-contained:

    bilstm        BiLSTM + attention, 15 classes        (default; best here)
    bilstm-int8   8-bit quantized bilstm (~4× smaller, less accurate)
    rawi          BiLSTM, 73 rich classes
    libtashkeel   char+hint encoder, length-masked

    >>> from text2tashkeel import Diacritizer, available_models
    >>> available_models()
    ['bilstm', 'bilstm-int8', 'rawi', 'libtashkeel']
    >>> Diacritizer().diacritize("بسم الله الرحمن الرحيم")
    'بِسمِ اللَّهِ الرَّحمَنِ الرَّحِيمِ'
    >>> Diacritizer("libtashkeel").diacritize("بسم الله الرحمن الرحيم")
    'بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم'

Every model adds marks to the *same* base letters; they only differ in which
marks they predict. See docs/ for the linguistics and architecture, and
benchmarks/ for an accuracy comparison.

Model and architecture credits: see docs/07-credits-and-license.md.
"""

from __future__ import annotations

from functools import lru_cache

from ._models import DEFAULT_MODEL, available_models, build_backend

__all__ = ["Diacritizer", "diacritize", "available_models", "DEFAULT_MODEL"]


class Diacritizer:
    """A reusable diacritizer for one model.

    Args:
        model: a name from `available_models()` (default ``"bilstm"``).
        providers: onnxruntime execution providers
            (default ``["CPUExecutionProvider"]``).

    The onnxruntime session is built lazily on first use, so constructing a
    ``Diacritizer`` is cheap. Process **one sentence per call** — see
    docs/04-inference-pipeline.md on why padded batching is unsafe.
    """

    def __init__(self, model: str = DEFAULT_MODEL, providers=None) -> None:
        if model not in available_models():
            raise ValueError(f"unknown model {model!r}; choose from {available_models()}")
        self.model = model
        self._providers = providers
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = build_backend(self.model, self._providers)
        return self._backend

    def diacritize(self, text: str) -> str:
        """Return ``text`` with predicted diacritics applied."""
        return self.backend.diacritize(text)

    __call__ = diacritize


@lru_cache(maxsize=None)
def _default(model: str) -> Diacritizer:
    return Diacritizer(model)


def diacritize(text: str, model: str = DEFAULT_MODEL) -> str:
    """Diacritize using a shared default model (convenience wrapper)."""
    return _default(model).diacritize(text)
