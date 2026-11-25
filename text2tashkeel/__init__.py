"""text2tashkeel — tiny standalone Arabic diacritizer (tashkeel).

Adds the missing vowel marks to Arabic text. Pure onnxruntime + numpy — no
torch, no network. A model picker over interchangeable models: singles (rawi-v2,
bilstm, rawi, libtashkeel, + INT8 variants) and gated ensembles named
`gate(+gate)+value` (e.g. rawi-v2+rawi, bilstm+libtashkeel+rawi).

    >>> from text2tashkeel import Diacritizer, available_models
    >>> Diacritizer().diacritize("بسم الله الرحمن الرحيم")  # default rawi-ensemble (2.03% DER)
    'بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ'
    >>> Diacritizer("rawi-v2-int8").diacritize("بسم الله الرحمن الرحيم")  # lean single model
    'بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ'

Every model adds marks to the *same* base letters; they differ in which marks
they predict, and in accuracy / latency / size. See docs/10 for the full
benchmark report and docs/ for the linguistics and architecture.

Model and architecture credits: see docs/07-credits-and-license.md.
"""

from __future__ import annotations

from functools import lru_cache

from ._models import (
    DEFAULT_MODEL, BUNDLED, available_models, build_backend, register_model,
)
from .waqf import pausal
from .version import __version__

__all__ = [
    "Diacritizer", "diacritize", "available_models", "register_model",
    "pausal", "DEFAULT_MODEL", "BUNDLED", "__version__",
]


class Diacritizer:
    """A reusable diacritizer for one model.

    Args:
        model: a name from `available_models()`
            (default ``"rawi-ensemble"`` — the flagship, 2.03% DER, single 4.9 MB ONNX).
        providers: onnxruntime execution providers
            (default ``["CPUExecutionProvider"]``).
        waqf: drop the case/mood endings (iʿrāb) from the result, leaving the
            pausal form that is spoken. The models restore the *full* endings,
            which is right for a pedagogical text and stilted for speech; a TTS
            frontend almost always wants ``waqf=True``.
        preserve_orthography: forbid the model from changing a letter the writing
            already spells. These models normalize to NFD and treat the hamza as
            a *mark on a bare alif*, so 29 of their classes carry a hamza or a
            madda — which means they can, and do, rewrite ⟨آ⟩ as ⟨أ⟩ and destroy
            the long /aː/ it stands for (آبد /ʔaːbid/ → أَبْد /ʔabd/; 6.5% of a
            17.5k-word WikiPron sweep). The offending classes are masked out
            *before* the argmax, so the model picks the best reading the writing
            permits. On by default: this is correctness, not preference. See
            :mod:`text2tashkeel.orthography`.
        respect_existing: never overwrite a mark a human already wrote. Partial
            vocalization is common — a text marks the words it thinks are
            ambiguous — and each of those marks is a person telling us the
            answer. Unconstrained, كِتَاب comes back as كَتَاب. On by default.
        allow_restoration: let the model ADD a hamza to a letter that carries
            none — the defective-spelling fix (bare ⟨ا⟩ typed for ⟨أ⟩) these
            models are built to do. This is why the rule is asymmetric: dropping
            a written mark is always wrong, adding one to a silent letter is the
            job. On by default.

    The three constraints apply to the models that decode with them: ``rawi``,
    ``rawi-v2``, ``rawi-ensemble``, their INT8 variants, and the ensembles whose
    value model is one of them. Left at ``None``, a constraint is on where the
    model supports it. Set on any other model, it raises :class:`ValueError`.

    The onnxruntime session is built lazily on first use, so constructing a
    ``Diacritizer`` is cheap. A constraint set explicitly builds the session at
    once, to check the model honours it. Process **one sentence per call** — see
    docs/04-inference-pipeline.md on why padded batching is unsafe.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        providers=None,
        waqf: bool = False,
        preserve_orthography: bool | None = None,
        respect_existing: bool | None = None,
        allow_restoration: bool | None = None,
    ) -> None:
        if model not in available_models():
            raise ValueError(f"unknown model {model!r}; choose from {available_models()}")
        self.model = model
        self.waqf = waqf
        self.preserve_orthography = preserve_orthography
        self.respect_existing = respect_existing
        self.allow_restoration = allow_restoration
        self._providers = providers
        self._backend = None
        if (preserve_orthography, respect_existing, allow_restoration) != (None, None, None):
            # Built now, so a model that ignores a constraint raises here.
            self._backend = build_backend(
                model, providers,
                preserve_orthography=preserve_orthography,
                respect_existing=respect_existing,
                allow_restoration=allow_restoration,
            )

    @property
    def backend(self):
        if self._backend is None:
            self._backend = build_backend(
                self.model, self._providers,
                preserve_orthography=self.preserve_orthography,
                respect_existing=self.respect_existing,
                allow_restoration=self.allow_restoration,
            )
        return self._backend

    def logits(self, text: str):
        """The raw per-character class distribution, before any decision.

        Returns ``(bare, logits, classes)`` — the NFD base characters the model
        actually saw, a ``[len(bare), n_classes]`` array, and the diacritic
        string each class stands for. ``bare`` is empty and ``logits`` is
        ``None`` when there is nothing to mark.

        This is for a caller that **knows something the model does not**: that
        this alif is written with a madda and not a hamza, that this letter
        already carries a human's fatḥa, that a variety's orthography does not
        admit some mark sequence at all. Such a caller can mask the classes that
        contradict what it knows and take the argmax of what remains — which is
        strictly better than correcting the output afterwards, because a masked
        class can never be chosen in the first place.

        Only the single-head architectures (``rawi``, ``rawi-v2``) expose this;
        anything else raises :class:`NotImplementedError`.
        """
        backend = self.backend
        if not hasattr(backend, "_logits"):
            raise NotImplementedError(
                f"model {self.model!r} does not expose per-class logits; "
                f"use a single-head rawi model (rawi, rawi-v2, + INT8 variants)"
            )
        bare, logits = backend._logits(text)
        return bare, logits, backend.classes

    def decode(self, bare: str, classes_per_char) -> str:
        """Recompose *bare* with one class id per character — the counterpart to
        :meth:`logits`, so a caller that masked and re-argmaxed can render the
        result the same way the model would have."""
        backend = self.backend
        if not hasattr(backend, "decode"):
            raise NotImplementedError(f"model {self.model!r} cannot decode class ids")
        return backend.decode(bare, classes_per_char)

    def diacritize(self, text: str) -> str:
        """Return ``text`` with predicted diacritics applied.

        With ``waqf=True`` the case and mood endings are then dropped, giving
        the pausal form that is actually spoken rather than the fully-parsed
        form the models restore — see :mod:`text2tashkeel.waqf`.
        """
        out = self.backend.diacritize(text)
        return pausal(out) if self.waqf else out

    __call__ = diacritize


@lru_cache(maxsize=None)
def _default(model: str) -> Diacritizer:
    return Diacritizer(model)


def diacritize(text: str, model: str = DEFAULT_MODEL, waqf: bool = False) -> str:
    """Diacritize using a shared default model (convenience wrapper).

    ``waqf=True`` returns the spoken (pausal) form — see
    :mod:`text2tashkeel.waqf`.
    """
    out = _default(model).diacritize(text)
    return pausal(out) if waqf else out
