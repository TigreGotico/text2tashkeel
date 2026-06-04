"""Model backends. One class per architecture, each replicating its source's
exact pre/post-processing in pure Python. No dependencies beyond numpy +
onnxruntime; vocabularies live in models/ as small JSON (or inline below).

Backends
--------
bilstm        — BiLSTM + Bahdanau attention, 15 classes, char vocab of 54.
                Source: github.com/Z-Mahmood/arabic-diacritizer-public-release
rawi          — BiLSTM, 73 classes (rich combos), full char vocab.
                Source: huggingface.co/TigreGotico/rawi
libtashkeel   — char+hint encoder, 15 targets, length-masked.
                Source: github.com/mush42/libtashkeel (Musharraf Omer)

Every backend exposes `.diacritize(text) -> str`.
"""

from __future__ import annotations

import json
import os
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODELS_DIR = Path(__file__).parent / "models"


def _session(onnx_path, providers):
    """Build an InferenceSession, honoring TT_ORT_THREADS (set it to 1 when you
    parallelize with your own process pool, to avoid thread oversubscription)."""
    opts = ort.SessionOptions()
    threads = os.environ.get("TT_ORT_THREADS")
    if threads:
        opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = int(threads)
    return ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=_providers(providers)
    )

# The Arabic tashkeel combining marks we strip from inputs (so the model always
# predicts from a bare skeleton). U+064B–U+0652 plus superscript alef U+0670.
_STRIP = {chr(c) for c in range(0x64B, 0x653)} | {"ٰ"}


def _strip(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text) if c not in _STRIP)


def _is_arabic_letter(c: str) -> bool:
    return "ء" <= c <= "ي"


def _providers(p):
    return p or ["CPUExecutionProvider"]


# ── bilstm ─────────────────────────────────────────────────────────────────
# Vocab + label map are inlined (they are the trained tokenizer; a mismatch
# would silently corrupt every prediction — see tests/test_vocab.py).
_BILSTM_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغ" "ـفقكلمنهوي"
_BILSTM_PUNCT = " .,،؛:؟!()-\"'\n"
_BILSTM_UNK = 1
_BILSTM_C2I = {c: 4 + i for i, c in enumerate(_BILSTM_LETTERS)}
_BILSTM_C2I.update(
    {c: 4 + len(_BILSTM_LETTERS) + i for i, c in enumerate(_BILSTM_PUNCT)}
)
_F, _D, _K, _SU = "َ", "ُ", "ِ", "ْ"
_SH = "ّ"
_FN, _DN, _KN = "ً", "ٌ", "ٍ"
_BILSTM_ID2LABEL = {
    0: "", 1: _F, 2: _D, 3: _K, 4: _SU, 5: _SH, 6: _FN, 7: _DN, 8: _KN,
    9: _SH + _F, 10: _SH + _D, 11: _SH + _K,
    12: _SH + _FN, 13: _SH + _DN, 14: _SH + _KN,
}


class _BilstmBackend:
    def __init__(self, onnx_path: Path, providers=None) -> None:
        self.sess = _session(onnx_path, providers)

    def _predict(self, text: str):
        """Return (stripped_text, predicted_class_per_char)."""
        s = _strip(text)
        if not s:
            return s, None
        ids = np.array([[_BILSTM_C2I.get(c, _BILSTM_UNK) for c in s]], dtype=np.int64)
        return s, self.sess.run(["logits"], {"input_ids": ids})[0][0].argmax(-1)

    def diacritize(self, text: str) -> str:
        s, pred = self._predict(text)
        if not s:
            return s
        out = "".join(ch + _BILSTM_ID2LABEL[int(p)] for ch, p in zip(s, pred))
        return unicodedata.normalize("NFC", out)


# ── rawi ───────────────────────────────────────────────────────────────────
class _RawiBackend:
    """Port of rawi's notebook decode. Key quirks vs the others:

    * Normalization is **NFD** (so أ → ا + hamza) with Unicode 'So' (symbol)
      characters dropped; the input is the NFD base sequence with all
      combining marks (category Mn) removed.
    * Diacritic *classes* therefore include hamza / superscript-alef, i.e. rawi
      restores those as part of "diacritization". Output is recomposed to NFC.
    * Marks are applied only to letters (Unicode L*); whitespace/punctuation are
      preserved untouched.
    """

    def __init__(self, onnx_path: Path, vocab_path: Path, providers=None) -> None:
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.c2i = dict(v["char_to_idx"])
        self.i2d = {i: s for s, i in dict(v["diac_to_idx"]).items()}
        self.unk = self.c2i.get("<UNK>", 1)
        self.sess = _session(onnx_path, providers)

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "So"
        )

    def _predict(self, text: str):
        """Return (bare_chars, predicted_class_per_char) on the NFD base seq."""
        bare = "".join(
            c for c in self._normalize(text) if unicodedata.category(c) != "Mn"
        )
        if not bare:
            return bare, None
        ids = np.array([[self.c2i.get(c, self.unk) for c in bare]], dtype=np.int64)
        return bare, self.sess.run(["output"], {"input": ids})[0][0].argmax(-1)

    def diacritize(self, text: str) -> str:
        bare, pred = self._predict(text)
        if not bare:
            return text
        out = "".join(
            ch + (self.i2d[int(p)] if unicodedata.category(ch).startswith("L") else "")
            for ch, p in zip(bare, pred)
        )
        return unicodedata.normalize("NFC", out)


# ── libtashkeel ────────────────────────────────────────────────────────────
class _LibtashkeelBackend:
    """Port of mush42/libtashkeel's pure-prediction path (no hints, no taskeen)."""

    _NUMERALS = set("0123456789٠١٢٣٤٥٦٧٨٩")
    _DIAC = {chr(c) for c in (1618, 1617, 1614, 1615, 1616, 1611, 1612, 1613)}

    def __init__(self, onnx_path: Path, maps_path: Path, providers=None) -> None:
        m = json.loads(Path(maps_path).read_text(encoding="utf-8"))
        self.inmap = m["input"]
        self.hintmap = m["hint"]
        self.id2target = {v: k for k, v in m["target"].items()}
        self.pad_id = self.inmap["_"]
        self.sess = _session(onnx_path, providers)

    def _to_valid(self, text: str):
        valid, removed = [], set()
        for c in text:
            if c in self.inmap or c in self._DIAC:
                valid.append(c)
            elif c in self._NUMERALS:
                valid.append("#")
            else:
                removed.add(c)
        return "".join(valid), removed

    def _split_chars_diac(self, text: str):
        text = text.lstrip("".join(self._DIAC))
        clean, diac, pend = [], [], ""
        for c in list(text) + [" "]:
            if c in self._DIAC:
                pend += c
            else:
                clean.append(c)
                diac.append(pend)
                pend = ""
        clean.pop()
        diac.pop(0)
        diac = [d if d in self.hintmap else "" for d in diac]
        return "".join(clean), diac

    def diacritize(self, text: str) -> str:
        text = _strip(text)  # unified "from scratch" semantics; hints unused
        valid, removed = self._to_valid(text)
        clean, hints = self._split_chars_diac(valid)
        if not clean:
            return text
        ci = np.array([[self.inmap[c] for c in clean]], dtype=np.int64)
        di = np.array([[self.hintmap[h] for h in hints]], dtype=np.int64)
        il = np.array([len(clean)], dtype=np.int64)
        pred = self.sess.run(
            ["predictions"],
            {"char_inputs": ci, "diac_inputs": di, "input_lengths": il},
        )[0][0]
        diacs = [self.id2target[int(p)] for p in pred if int(p) != self.pad_id]
        out, it = [], iter(diacs)
        for c in text:
            if c in self._DIAC:
                continue
            if c in removed:
                out.append(c)
            else:
                out.append(c + next(it, ""))
        return unicodedata.normalize("NFC", "".join(out))


# ── ensemble: agreement gating ──────────────────────────────────────────────
_MARK_SET = {chr(c) for c in range(0x64B, 0x653)}


def _marks_of(text: str) -> np.ndarray:
    """Per base character of `text`: True if it carries any tashkeel mark.
    Walks the NFC string, attaching trailing marks to the preceding base char."""
    flags: list[bool] = []
    has = False
    for ch in unicodedata.normalize("NFC", text):
        if ch in _MARK_SET:
            has = True
        else:
            if flags:
                flags[-1] = has
            flags.append(False)
            has = False
    if flags:
        flags[-1] = has
    return np.array(flags, dtype=bool)


class _EnsembleBackend:
    """Agreement-gated ensemble — the most accurate option (see
    docs/09-combining-models.md).

    The models split the job by their strengths:
      * one or more **gate** models decide WHERE a mark goes — they are
        well-calibrated on the mark / no-mark decision;
      * the **value** model (rawi) decides WHICH mark — it has the best per-marked
        position accuracy, but over-marks on its own.

    We keep the value model's predicted mark only at positions the gate approves,
    which suppresses the value model's confident over-marking. With several gates,
    ``combine='any'`` approves a position if ANY gate marks it (permissive — best
    on our benchmark), ``'all'`` requires every gate to mark it (strict). The
    result beats every single model. Costs one ONNX run per model per sentence.

    Alignment: rawi normalizes NFD (أ→ا), the gates NFC, but both yield one
    position per base letter, so we index-align with a length-check fallback (a
    gate whose length disagrees on a given sentence is skipped for it).
    """

    def __init__(self, gates, value: str = "rawi", combine: str = "any",
                 providers=None) -> None:
        self.gates = [build_backend(g, providers) for g in gates]
        self.value = build_backend(value, providers)
        self.combine = combine

    def diacritize(self, text: str) -> str:
        bare, cls = self.value._predict(text)
        if not bare:
            return text
        n = len(bare)
        masks = [m for g in self.gates
                 if len(m := _marks_of(g.diacritize(text))) == n]
        gate = None
        if masks:
            gate = masks[0].copy()
            for m in masks[1:]:
                gate = gate | m if self.combine == "any" else gate & m
        out = []
        for i, (ch, c) in enumerate(zip(bare, cls)):
            if unicodedata.category(ch).startswith("L"):
                mark = self.value.i2d[int(c)]
                if mark and gate is not None and not gate[i]:
                    mark = ""
                out.append(ch + mark)
            else:
                out.append(ch)
        return unicodedata.normalize("NFC", "".join(out))


# ── registry ───────────────────────────────────────────────────────────────
_REGISTRY = {
    "bilstm": lambda p: _BilstmBackend(_MODELS_DIR / "bilstm.onnx", p),
    "bilstm-int8": lambda p: _BilstmBackend(_MODELS_DIR / "bilstm.int8.onnx", p),
    "rawi": lambda p: _RawiBackend(
        _MODELS_DIR / "rawi.onnx", _MODELS_DIR / "rawi.vocab.json", p
    ),
    "libtashkeel": lambda p: _LibtashkeelBackend(
        _MODELS_DIR / "libtashkeel.onnx", _MODELS_DIR / "libtashkeel.maps.json", p
    ),
    # agreement-gated ensembles (gate decides WHERE, rawi decides WHICH mark)
    "bilstm+rawi": lambda p: _EnsembleBackend(["bilstm"], "rawi", "any", p),
    "libtashkeel+rawi": lambda p: _EnsembleBackend(["libtashkeel"], "rawi", "any", p),
    # champion: gate = bilstm OR libtashkeel, value = rawi (best on the benchmark)
    "ensemble": lambda p: _EnsembleBackend(["bilstm", "libtashkeel"], "rawi", "any", p),
}

DEFAULT_MODEL = "bilstm"


def available_models() -> list[str]:
    """Names accepted by `Diacritizer(model=...)`."""
    return list(_REGISTRY)


@lru_cache(maxsize=None)
def _build(model: str, providers_key):
    if model not in _REGISTRY:
        raise ValueError(
            f"unknown model {model!r}; choose from {available_models()}"
        )
    return _REGISTRY[model](list(providers_key) if providers_key else None)


def build_backend(model: str, providers=None):
    return _build(model, tuple(providers) if providers else None)
