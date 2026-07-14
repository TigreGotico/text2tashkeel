"""text2tashkeel — tiny standalone Arabic diacritizer (tashkeel).

Adds the missing vowel marks to Arabic text. Pure onnxruntime + numpy — no
torch, no network. A model picker over interchangeable models: singles (rawi-v2,
bilstm, rawi, libtashkeel, + INT8 variants) and gated ensembles named
`gate(+gate)+value` (e.g. rawi-v2+rawi, bilstm+libtashkeel+rawi).

    >>> from text2tashkeel import Diacritizer, available_models
    >>> Diacritizer().diacritize("بسم الله الرحمن الرحيم")  # default rawi-ensemble (2.04% DER)
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
    # per-model wrapper classes (syntactic sugar)
    "Bilstm", "BilstmInt8", "Rawi", "RawiInt8", "RawiV2", "RawiV2Int8", "RawiV3", "RawiV3Int8", "Libtashkeel",
    "BilstmRawi", "BilstmRawiInt8", "LibtashkeelRawi", "LibtashkeelRawiInt8",
    "BilstmInt8RawiInt8", "BilstmLibtashkeelRawi", "BilstmLibtashkeelRawiInt8",
    "RawiV2Rawi", "RawiV2RawiInt8", "RawiV2Int8RawiInt8", "RawiV2RawiV3", "RawiV2Int8RawiV3Int8",
    "RawiEnsemble",
]


class Diacritizer:
    """A reusable diacritizer for one model.

    Args:
        model: a name from `available_models()`
            (default ``"rawi-ensemble"`` — the flagship, 2.04% DER, single 4.9 MB ONNX).
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

    The onnxruntime session is built lazily on first use, so constructing a
    ``Diacritizer`` is cheap. Process **one sentence per call** — see
    docs/04-inference-pipeline.md on why padded batching is unsafe.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        providers=None,
        waqf: bool = False,
        preserve_orthography: bool = True,
        respect_existing: bool = True,
        allow_restoration: bool = True,
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


# ── Per-model wrapper classes ───────────────────────────────────────────────
# Thin Diacritizer subclasses that fix the model name — pure syntactic sugar so
# `BilstmLibtashkeelRawi()` reads/ autocompletes. Gated-ensemble names spell out
# the combo: `gate(+gate)+value`, last model = value. Numbers are in docs/08.

# singles
class Bilstm(Diacritizer):
    """BiLSTM + Bahdanau attention. Best single — DER 4.95%, ~188 sent/s, 18 MB."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm", providers)


class BilstmInt8(Diacritizer):
    """INT8-quantized bilstm — lossy (DER 12.89%); attention is quant-sensitive."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm-int8", providers)


class Rawi(Diacritizer):
    """TigreGotico rawi (V1). Fastest (~437 sent/s) but over-marks alone
    (DER 18.49%); shines as an ensemble value model. See docs/09 §9.5."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi", providers)


class RawiInt8(Diacritizer):
    """INT8 rawi — essentially lossless (no attention), ~2.5 MB."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-int8", providers)


class RawiV2(Diacritizer):
    """TigreGotico rawi V2 — same BiLSTM, retrained with the V1 over-marking
    bug fixed (pad with -100, default ignore_index). Usable standalone."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2", providers)


class RawiV2Int8(Diacritizer):
    """INT8 rawi V2 — lossless (no attention), ~2.5 MB."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2-int8", providers)


class RawiV3(Diacritizer):
    """rawi V3 — two-head gated model (presence head decides WHERE, value head
    WHICH, in one network). The ensemble, internalized. See docs/09 §9.11."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v3", providers)


class RawiV3Int8(Diacritizer):
    """INT8 rawi V3 — lossless (no attention), ~2.5 MB."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v3-int8", providers)


class Libtashkeel(Diacritizer):
    """mush42/libtashkeel char+hint encoder. Smallest good single — DER 6.89%."""
    def __init__(self, providers=None) -> None:
        super().__init__("libtashkeel", providers)


# gated ensembles (gate decides WHERE, value decides WHICH mark)
class BilstmRawi(Diacritizer):
    """Gate bilstm → value rawi. DER 4.38%."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm+rawi", providers)


class BilstmRawiInt8(Diacritizer):
    """Gate bilstm → value rawi-int8 (smaller, lossless value)."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm+rawi-int8", providers)


class LibtashkeelRawi(Diacritizer):
    """Gate libtashkeel → value rawi. DER 4.12%."""
    def __init__(self, providers=None) -> None:
        super().__init__("libtashkeel+rawi", providers)


class LibtashkeelRawiInt8(Diacritizer):
    """Gate libtashkeel → value rawi-int8."""
    def __init__(self, providers=None) -> None:
        super().__init__("libtashkeel+rawi-int8", providers)


class BilstmInt8RawiInt8(Diacritizer):
    """Fully-int8 2-model ensemble: gate bilstm-int8 → value rawi-int8."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm-int8+rawi-int8", providers)


class BilstmLibtashkeelRawi(Diacritizer):
    """Best accuracy: gate (bilstm OR libtashkeel) → value rawi. DER 3.99%,
    ~12 sent/s (3 models). See docs/09."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm+libtashkeel+rawi", providers)


class BilstmLibtashkeelRawiInt8(Diacritizer):
    """Best accuracy with the lossless int8 value — matches the fp32 best at a
    smaller footprint. Gate (bilstm OR libtashkeel) → value rawi-int8."""
    def __init__(self, providers=None) -> None:
        super().__init__("bilstm+libtashkeel+rawi-int8", providers)


class RawiV2Rawi(Diacritizer):
    """Most accurate: rawi-v2 gates WHERE, rawi-v1 picks WHICH mark. DER 2.19%
    (full test), ~2 ms. Beats rawi-v2 alone. See docs/09 §9.11."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2+rawi", providers)


class RawiV2RawiInt8(Diacritizer):
    """Same combo with the lossless int8 value (rawi-int8). DER 2.20%, smaller."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2+rawi-int8", providers)


class RawiV2Int8RawiInt8(Diacritizer):
    """Fully-int8 top-tier combo (gate rawi-v2-int8, value rawi-int8) — ~5 MB,
    smallest of the most-accurate options."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2-int8+rawi-int8", providers)


class RawiV2RawiV3(Diacritizer):
    """Flagship: rawi-v2 gates WHERE, rawi-v3's value head supplies WHICH (best
    DER*). Ships as a single stitched ONNX (TigreGotico/rawi-ensemble)."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2+rawi-v3", providers)


class RawiV2Int8RawiV3Int8(Diacritizer):
    """Flagship, fully int8 (~5 MB stitched)."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-v2-int8+rawi-v3-int8", providers)


class RawiEnsemble(Diacritizer):
    """The flagship (default): rawi-v2+rawi-v3 as a single stitched ONNX — one
    session.run, 4.9 MB, DER 2.04%. Same output as RawiV2Int8RawiV3Int8()."""
    def __init__(self, providers=None) -> None:
        super().__init__("rawi-ensemble", providers)
