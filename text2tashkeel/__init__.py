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
from .version import __version__

__all__ = [
    "Diacritizer", "diacritize", "available_models", "register_model",
    "DEFAULT_MODEL", "BUNDLED", "__version__",
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
