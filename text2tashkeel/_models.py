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
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

_MODELS_DIR = Path(__file__).parent / "models"

# The wheel bundles the INT8 models and the stitched flagship (small, offline by
# default). The fp32 weights of the rawi family live in their own Hugging Face
# repos and are fetched on first use when `huggingface_hub` is installed
# (`pip install text2tashkeel[hf]`). Vocab/maps JSON are always bundled. Map a
# local filename → (hf_repo_id, remote_filename) for the files we host on HF.
_HF_SOURCES = {
    "rawi.onnx":          ("TigreGotico/rawi",                   "diacritization_model_lstm.onnx"),
    "rawi_v2.onnx":       ("TigreGotico/rawi-v2",                "rawi_v2.onnx"),
    "rawi_v3.onnx":       ("TigreGotico/rawi-v3",                "rawi_v3.onnx"),
    "rawi_ensemble.onnx": ("TigreGotico/rawi-ensemble",         "rawi_ensemble.onnx"),
    "bilstm.onnx":        ("TigreGotico/bilstm-diacritizer",     "bilstm.onnx"),
    "libtashkeel.onnx":   ("TigreGotico/libtashkeel-diacritizer", "libtashkeel.onnx"),
    # third-party baselines (re-exported, not bundled — fetched on first use)
    "shakkala.onnx":          ("TigreGotico/shakkala-diacritizer", "shakkala_v3.onnx"),
    "shakkala.int8.onnx":     ("TigreGotico/shakkala-diacritizer", "shakkala_v3.int8.onnx"),
    "shakkala.in_vocab.json": ("TigreGotico/shakkala-diacritizer", "input_vocab_to_int.json"),
    "shakkala.out_vocab.json":("TigreGotico/shakkala-diacritizer", "output_int_to_vocab.json"),
    "catt.onnx":              ("TigreGotico/catt-diacritizer",     "catt_eo.onnx"),
    "catt.int8.onnx":         ("TigreGotico/catt-diacritizer",     "catt_eo.int8.onnx"),
}


def _model_path(filename: str) -> str:
    """Resolve a model file: bundled in the wheel if present, else downloaded (and
    cached) from its Hugging Face repo. Raises a clear, actionable error otherwise.

    To use a model trained on your own corpus, see `register_model()` — you point at
    your own ONNX + vocab and never touch this resolver."""
    local = _MODELS_DIR / filename
    if local.exists():
        return str(local)
    src = _HF_SOURCES.get(filename)
    if src is None:
        raise FileNotFoundError(
            f"{filename!r} is not bundled and has no Hugging Face source. The wheel "
            f"ships the INT8 models and the stitched flagship; this fp32 weight is "
            f"not hosted. Use its INT8 variant, or load your own file with "
            f"text2tashkeel.register_model(...)."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        repo, remote = src
        raise FileNotFoundError(
            f"{filename!r} is not bundled in the wheel (only the INT8 models and the "
            f"stitched flagship ship inside it, to keep it small). Install the "
            f"download extra to fetch it automatically:\n"
            f"    pip install text2tashkeel[hf]\n"
            f"or download {remote!r} from https://huggingface.co/{repo}"
        ) from None
    repo, remote = src
    return hf_hub_download(repo, remote)


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

    def mark_mask(self, text: str) -> np.ndarray:
        """Per base char: True if this position gets a mark (class != 0). Used as
        a gate — equivalent to _marks_of(diacritize(text)) but without building or
        re-parsing the output string."""
        _, cls = self._predict(text)
        return cls != 0 if cls is not None else np.zeros(0, dtype=bool)

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

    def mark_mask(self, text: str) -> np.ndarray:
        """Per base char: True where this position gets a mark (class != 0).
        Lets a rawi model serve as an ensemble *gate* without rebuilding/parsing
        the output string. Both rawi V1/V2 share the NFD base sequence, so the
        mask index-aligns with another rawi backend's `_predict`."""
        _, pred = self._predict(text)
        return pred != 0 if pred is not None else np.zeros(0, dtype=bool)

    def diacritize(self, text: str) -> str:
        bare, pred = self._predict(text)
        if not bare:
            return text
        out = "".join(
            ch + (self.i2d[int(p)] if unicodedata.category(ch).startswith("L") else "")
            for ch, p in zip(bare, pred)
        )
        return unicodedata.normalize("NFC", out)


# ── rawi V3 (two-head gated) ─────────────────────────────────────────────────
class _RawiV3Backend:
    """rawi V3 — one network that internalizes the gate. A shared BiLSTM feeds a
    *presence head* (1 logit → sigmoid: does this letter carry a mark? = WHERE) and
    a *value head* (75 logits → argmax: which mark? = WHICH). A mark is applied only
    where presence > threshold. Same NFD vocab as V2 (reused). See docs/09 §9.11."""

    def __init__(self, onnx_path: Path, vocab_path: Path, threshold: float = 0.5,
                 providers=None) -> None:
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.c2i = dict(v["char_to_idx"])
        self.i2d = {i: s for s, i in dict(v["diac_to_idx"]).items()}
        self.unk = self.c2i.get("<UNK>", 1)
        self.threshold = threshold
        self.sess = _session(onnx_path, providers)

    def _predict(self, text: str):
        bare = "".join(
            c for c in _RawiBackend._normalize(text)
            if unicodedata.category(c) != "Mn"
        )
        if not bare:
            return bare, None, None
        ids = np.array([[self.c2i.get(c, self.unk) for c in bare]], dtype=np.int64)
        pres, val = self.sess.run(["presence", "value"], {"input": ids})
        gate = (1.0 / (1.0 + np.exp(-pres[0]))) > self.threshold
        return bare, gate, val[0].argmax(-1)

    def mark_mask(self, text: str) -> np.ndarray:
        """Presence gate as a bool mask — lets V3 act as an ensemble gate too."""
        _, gate, _ = self._predict(text)
        return gate if gate is not None else np.zeros(0, dtype=bool)

    def diacritize(self, text: str) -> str:
        bare, gate, cls = self._predict(text)
        if not bare:
            return text
        out = []
        for ch, g, c in zip(bare, gate, cls):
            out.append(ch)
            if g and unicodedata.category(ch).startswith("L"):
                out.append(self.i2d[int(c)])
        return unicodedata.normalize("NFC", "".join(out))


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

    def mark_mask(self, text: str) -> np.ndarray:
        """Per base char of the stripped text: True if it gets a mark. Used as a
        gate — equivalent to _marks_of(diacritize(text)) without building/normalizing
        the output string. (id2target is "" exactly when no mark is placed.)"""
        text = _strip(text)
        valid, removed = self._to_valid(text)
        clean, hints = self._split_chars_diac(valid)
        if not clean:
            return np.zeros(sum(1 for c in text if c not in self._DIAC), dtype=bool)
        ci = np.array([[self.inmap[c] for c in clean]], dtype=np.int64)
        di = np.array([[self.hintmap[h] for h in hints]], dtype=np.int64)
        il = np.array([len(clean)], dtype=np.int64)
        pred = self.sess.run(
            ["predictions"],
            {"char_inputs": ci, "diac_inputs": di, "input_lengths": il},
        )[0][0]
        marks = (self.id2target[int(p)] != "" for p in pred if int(p) != self.pad_id)
        flags, it = [], marks
        for c in text:
            if c in self._DIAC:
                continue
            flags.append(False if c in removed else next(it, False))
        return np.array(flags, dtype=bool)


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
        # Value model supplies WHICH mark via its class-per-char prediction. Both
        # rawi backends expose `_predict`; V1/V2 return (bare, cls) and V3 returns
        # (bare, gate, cls) — taking [0]/[-1] uses V3's *value* head and ignores its
        # own presence gate (the ensemble's gates decide WHERE).
        pred = self.value._predict(text)
        bare, cls = pred[0], pred[-1]
        if not bare:
            return text
        n = len(bare)
        # Fast path: gates expose mark_mask (argmax → bool, no string build/parse).
        # Identical result to _marks_of(gate.diacritize(text)), ~5× cheaper.
        masks = []
        for g in self.gates:
            mm = getattr(g, "mark_mask", None)
            m = mm(text) if mm is not None else _marks_of(g.diacritize(text))
            if len(m) == n:
                masks.append(m)
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


# ── stitched single-ONNX ensemble ────────────────────────────────────────────
class _StitchedEnsembleBackend:
    """The flagship gated ensemble as ONE ONNX graph: rawi-v2 gates WHERE and
    rawi-v3's value head supplies WHICH, with the gating math folded in (see
    tools/build_ensemble_v2v3_onnx.py). One `session.run` instead of two — byte-
    identical to the `rawi-v2+rawi-v3` Python ensemble. Input = NFD-bare char ids;
    output `gated_cls` = one diacritic-class id per position (0 = no mark)."""

    def __init__(self, onnx_path: Path, vocab_path: Path, providers=None) -> None:
        v = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.c2i = dict(v["char_to_idx"])
        self.i2d = {i: s for s, i in dict(v["diac_to_idx"]).items()}
        self.unk = self.c2i.get("<UNK>", 1)
        self.sess = _session(onnx_path, providers)

    def diacritize(self, text: str) -> str:
        bare = "".join(
            c for c in _RawiBackend._normalize(text)
            if unicodedata.category(c) != "Mn"
        )
        if not bare:
            return text
        ids = np.array([[self.c2i.get(c, self.unk) for c in bare]], dtype=np.int64)
        cls = self.sess.run(["gated_cls"], {"input": ids})[0][0]
        out = "".join(
            ch + (self.i2d[int(c)] if unicodedata.category(ch).startswith("L") else "")
            for ch, c in zip(bare, cls)
        )
        return unicodedata.normalize("NFC", out)


# ── shakkala (third-party baseline) ──────────────────────────────────────────
class _ShakkalaBackend:
    """Ahmad Barqawi's Shakkala v3, re-exported to ONNX (MIT). Fixed 315-length: the
    input is char-id-padded to 315 and the model emits one of 28 harakat classes per
    position. Input longer than 315 characters is diacritized up to the limit and the
    tail returned bare. See TigreGotico/shakkala-diacritizer."""

    _MAXLEN = 315

    def __init__(self, onnx_path: Path, in_vocab_path, out_vocab_path, providers=None) -> None:
        self.i2v = json.loads(Path(in_vocab_path).read_text(encoding="utf-8"))
        out = json.loads(Path(out_vocab_path).read_text(encoding="utf-8"))
        self.o2v = {int(k): v for k, v in out.items()}
        self.unk = self.i2v["<UNK>"]
        self.sess = _session(onnx_path, providers)

    def diacritize(self, text: str) -> str:
        s = _strip(text)
        if not s:
            return text
        n = min(len(s), self._MAXLEN)
        ids = [self.i2v.get(c, self.unk) for c in s[:n]] + [0] * (self._MAXLEN - n)
        name = self.sess.get_inputs()[0].name
        logits = self.sess.run(None, {name: np.array([ids], dtype=np.float32)})[0][0]
        har = [self.o2v[int(a)] for a in logits.argmax(-1) if self.o2v[int(a)] != "<PAD>"]
        har += [""] * (n - len(har))
        out = "".join(c + ("" if h in ("<UNK>", "ـ") else h) for c, h in zip(s[:n], har))
        return unicodedata.normalize("NFC", out + s[n:])


# ── CATT (third-party baseline) ──────────────────────────────────────────────
class _CattBackend:
    """Abjad AI's encoder-only CATT (Apache-2.0), with the transformer encoder and its
    non-autoregressive classifier head stitched into one ONNX. Buckwalter-tokenized;
    emits one of 18 tashkeel tags per character. The tokenizer is vendored
    (``text2tashkeel._vendor.catt``). See TigreGotico/catt-diacritizer."""

    # one or more Arabic words (with internal whitespace) — non-Arabic spans
    # (Latin, digits, punctuation) are left untouched.
    _AR = r"؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
    _RUN = re.compile(rf"[{_AR}]+(?:\s+[{_AR}]+)*")

    def __init__(self, onnx_path: Path, providers=None) -> None:
        from text2tashkeel._vendor.catt import TashkeelTokenizer
        self.tok = TashkeelTokenizer()
        self.pad = self.tok.letters_map["<PAD>"]
        self.space = self.tok.letters_map[" "]
        self.nt = self.tok.tashkeel_map[self.tok.no_tashkeel_tag]
        self.sess = _session(onnx_path, providers)

    def diacritize(self, text: str) -> str:
        return self._RUN.sub(lambda m: self._diac_run(m.group(0)), text)

    def _diac_run(self, s: str) -> str:
        s = _strip(s)
        if not s.strip():
            return s
        input_ids, _ = self.tok.encode(s, test_match=False)
        src = input_ids[None, :].astype(np.int64)                      # (1, T)
        keep = src != self.pad
        src_mask = keep[:, None, :, None] & keep[:, None, None, :]     # (1, 1, T, T)
        logits = self.sess.run(None, {"src": src, "src_mask": src_mask})[0][0]
        pred = logits.argmax(-1)
        pred[src[0] == self.space] = self.nt
        # encode() wraps the sequence in <BOS>/<EOS>; the model doesn't always emit
        # those tags at the boundary positions, so pin them so decode()'s value-based
        # filter drops exactly the boundaries (otherwise the marks shift by one).
        pred[0] = self.tok.tashkeel_map["<BOS>"]
        pred[-1] = self.tok.tashkeel_map["<EOS>"]
        return self.tok.decode([input_ids], [pred])[0]


# ── registry ───────────────────────────────────────────────────────────────
_REGISTRY = {
    "bilstm": lambda p: _BilstmBackend(_model_path("bilstm.onnx"), p),
    "bilstm-int8": lambda p: _BilstmBackend(_model_path("bilstm.int8.onnx"), p),
    "rawi": lambda p: _RawiBackend(
        _model_path("rawi.onnx"), _model_path("rawi.vocab.json"), p
    ),
    # INT8-quantized rawi — essentially lossless (rawi has no attention), ~2.5 MB
    "rawi-int8": lambda p: _RawiBackend(
        _model_path("rawi.int8.onnx"), _model_path("rawi.vocab.json"), p
    ),
    # rawi V2 — same architecture, retrained with the V1 over-marking bug fixed
    # (pad with -100, default ignore_index); usable standalone, unlike V1.
    "rawi-v2": lambda p: _RawiBackend(
        _model_path("rawi_v2.onnx"), _model_path("rawi_v2.vocab.json"), p
    ),
    "rawi-v2-int8": lambda p: _RawiBackend(
        _model_path("rawi_v2.int8.onnx"), _model_path("rawi_v2.vocab.json"), p
    ),
    # rawi V3 — two-head gated model (presence + value heads in one network),
    # shares V2's vocab. Internalizes the rawi-v2+rawi gating into a single pass.
    "rawi-v3": lambda p: _RawiV3Backend(
        _model_path("rawi_v3.onnx"), _model_path("rawi_v2.vocab.json"), 0.5, p
    ),
    "rawi-v3-int8": lambda p: _RawiV3Backend(
        _model_path("rawi_v3.int8.onnx"), _model_path("rawi_v2.vocab.json"), 0.5, p
    ),
    "libtashkeel": lambda p: _LibtashkeelBackend(
        _model_path("libtashkeel.onnx"), _model_path("libtashkeel.maps.json"), p
    ),
    # ── third-party baselines (fetched from HF, not bundled) ────────────────
    "shakkala": lambda p: _ShakkalaBackend(
        _model_path("shakkala.onnx"), _model_path("shakkala.in_vocab.json"),
        _model_path("shakkala.out_vocab.json"), p
    ),
    "shakkala-int8": lambda p: _ShakkalaBackend(
        _model_path("shakkala.int8.onnx"), _model_path("shakkala.in_vocab.json"),
        _model_path("shakkala.out_vocab.json"), p
    ),
    "catt": lambda p: _CattBackend(_model_path("catt.onnx"), p),
    "catt-int8": lambda p: _CattBackend(_model_path("catt.int8.onnx"), p),
    # ── agreement-gated ensembles ──────────────────────────────────────────
    # Naming: `gate(+gate…)+value` — the LAST model is the value (decides WHICH
    # mark); the preceding model(s) are gates (decide WHERE, OR-combined).
    # int8 value (rawi-int8) is lossless, so int8 variants match fp32 accuracy at
    # a smaller footprint.
    "bilstm+rawi": lambda p: _EnsembleBackend(["bilstm"], "rawi", "any", p),
    "bilstm+rawi-int8": lambda p: _EnsembleBackend(["bilstm"], "rawi-int8", "any", p),
    "libtashkeel+rawi": lambda p: _EnsembleBackend(["libtashkeel"], "rawi", "any", p),
    "libtashkeel+rawi-int8": lambda p: _EnsembleBackend(["libtashkeel"], "rawi-int8", "any", p),
    "bilstm-int8+rawi-int8": lambda p: _EnsembleBackend(["bilstm-int8"], "rawi-int8", "any", p),
    # best on the benchmark: gate = bilstm OR libtashkeel, value = rawi
    "bilstm+libtashkeel+rawi": lambda p: _EnsembleBackend(["bilstm", "libtashkeel"], "rawi", "any", p),
    "bilstm+libtashkeel+rawi-int8": lambda p: _EnsembleBackend(["bilstm", "libtashkeel"], "rawi-int8", "any", p),
    # in-family gate: rawi-v2 decides WHERE (best calibrated), rawi-v1 decides WHICH
    # (best per-marked accuracy). Beats rawi-v2 alone — the most accurate option.
    "rawi-v2+rawi": lambda p: _EnsembleBackend(["rawi-v2"], "rawi", "any", p),
    "rawi-v2+rawi-int8": lambda p: _EnsembleBackend(["rawi-v2"], "rawi-int8", "any", p),
    # fully-int8 combo — both models quantized (gate rawi-v2-int8, value rawi-int8);
    # smallest top-tier option (~5 MB), int8 is lossless for these attention-free LSTMs
    "rawi-v2-int8+rawi-int8": lambda p: _EnsembleBackend(["rawi-v2-int8"], "rawi-int8", "any", p),
    # flagship: rawi-v2 gates WHERE, rawi-v3's value head supplies WHICH (best DER*).
    # Same vocab → ships as a single stitched ONNX (TigreGotico/rawi-ensemble,
    # tools/build_ensemble_v2v3_onnx.py). Here it composes the two bundled models.
    "rawi-v2+rawi-v3": lambda p: _EnsembleBackend(["rawi-v2"], "rawi-v3", "any", p),
    "rawi-v2-int8+rawi-v3-int8": lambda p: _EnsembleBackend(["rawi-v2-int8"], "rawi-v3-int8", "any", p),
    # flagship as a single stitched ONNX (one session.run) — the default. Same
    # output as rawi-v2-int8+rawi-v3-int8, in one 4.9 MB file.
    "rawi-ensemble": lambda p: _StitchedEnsembleBackend(
        _model_path("rawi_ensemble.int8.onnx"), _model_path("rawi_v2.vocab.json"), p
    ),
}

DEFAULT_MODEL = "rawi-ensemble"

# Models whose every weight ships inside the wheel — they work fully offline.
# The rest fetch their fp32 weights from Hugging Face on first use (see
# `_HF_SOURCES`) or need a bring-your-own file (see `register_model`).
BUNDLED = frozenset({
    "bilstm-int8", "rawi-int8", "rawi-v2-int8", "rawi-v3-int8", "rawi-ensemble",
    "bilstm-int8+rawi-int8", "rawi-v2-int8+rawi-int8", "rawi-v2-int8+rawi-v3-int8",
})

# Architecture name → backend factory for register_model(). A model trained on a
# different corpus typically reuses the rawi single-head ("rawi") or two-head
# ("rawi-v3") decode with its own ONNX + vocab.
_ARCHS = {
    "rawi": lambda onnx, vocab, thr, p: _RawiBackend(onnx, vocab, p),
    "rawi-v3": lambda onnx, vocab, thr, p: _RawiV3Backend(onnx, vocab, thr, p),
    "two-head": lambda onnx, vocab, thr, p: _RawiV3Backend(onnx, vocab, thr, p),
    "bilstm": lambda onnx, vocab, thr, p: _BilstmBackend(onnx, p),
    "libtashkeel": lambda onnx, vocab, thr, p: _LibtashkeelBackend(onnx, vocab, p),
    "stitched": lambda onnx, vocab, thr, p: _StitchedEnsembleBackend(onnx, vocab, p),
    "catt": lambda onnx, vocab, thr, p: _CattBackend(onnx, p),
    # for shakkala, `vocab` is a directory holding input_vocab_to_int.json +
    # output_int_to_vocab.json
    "shakkala": lambda onnx, vocab, thr, p: _ShakkalaBackend(
        onnx, str(Path(vocab) / "input_vocab_to_int.json"),
        str(Path(vocab) / "output_int_to_vocab.json"), p
    ),
}


def available_models(bundled_only: bool = False) -> list[str]:
    """Names accepted by `Diacritizer(model=...)`. With ``bundled_only=True``,
    only the models whose weights ship in the wheel (work offline)."""
    names = list(_REGISTRY)
    return [n for n in names if n in BUNDLED] if bundled_only else names


def register_model(name: str, onnx_path, vocab_path=None, *, arch: str = "rawi",
                   threshold: float = 0.5) -> None:
    """Register a model from your own files — e.g. one trained on a different
    corpus — so `Diacritizer(name)` can use it.

    Args:
        name: the model name to register (overrides an existing name if reused).
        onnx_path: path to your exported ONNX.
        vocab_path: path to the matching vocab JSON (``char_to_idx`` / ``diac_to_idx``
            for ``rawi``/``rawi-v3``/``stitched``; maps JSON for ``libtashkeel``).
            Not needed for ``bilstm`` (its vocab is inlined).
        arch: which decode to use — ``"rawi"`` (single head), ``"rawi-v3"`` /
            ``"two-head"``, ``"stitched"``, ``"bilstm"``, ``"libtashkeel"``,
            ``"catt"`` (Buckwalter transformer; ``vocab_path`` ignored), or
            ``"shakkala"`` (``vocab_path`` = a directory holding
            ``input_vocab_to_int.json`` + ``output_int_to_vocab.json``).
        threshold: presence threshold for two-head models.

    Example::

        from text2tashkeel import register_model, Diacritizer
        register_model("my-rawi", "my_model.onnx", "my_vocab.json", arch="rawi")
        Diacritizer("my-rawi").diacritize("نص عربي")
    """
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")
    onnx_path = str(onnx_path)
    vocab_path = str(vocab_path) if vocab_path is not None else None
    factory = _ARCHS[arch]
    _REGISTRY[name] = lambda p: factory(onnx_path, vocab_path, threshold, p)
    _build.cache_clear()


@lru_cache(maxsize=None)
def _build(model: str, providers_key):
    if model not in _REGISTRY:
        raise ValueError(
            f"unknown model {model!r}; choose from {available_models()}"
        )
    return _REGISTRY[model](list(providers_key) if providers_key else None)


def build_backend(model: str, providers=None):
    return _build(model, tuple(providers) if providers else None)
