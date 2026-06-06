"""CLI smoke tests (argument mode and stdin mode)."""

import subprocess
import sys
import unicodedata


def _run(args, stdin=None):
    return subprocess.run(
        [sys.executable, "-m", "text2tashkeel", *args],
        input=stdin, capture_output=True, text=True, timeout=120,
    )


def _has_marks(s):
    return any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", s))


def test_cli_argument_mode():
    r = _run(["بسم الله الرحمن الرحيم"])
    assert r.returncode == 0
    assert _has_marks(r.stdout)


def test_cli_stdin_mode():
    r = _run([], stdin="محمد رسول الله\n")
    assert r.returncode == 0
    assert _has_marks(r.stdout)


def test_cli_model_flag():
    r = _run(["-m", "libtashkeel", "هذا كتاب مفيد"])
    assert r.returncode == 0
    assert _has_marks(r.stdout)


def test_cli_blank_line_passthrough():
    r = _run([], stdin="\n\n")
    assert r.returncode == 0
