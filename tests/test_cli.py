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


# The globs of [tool.setuptools.package-data] in pyproject.toml: what the wheel ships.
_WHEEL_MODEL_GLOBS = ("*.int8.onnx", "*.json")


def test_cli_default_runs_from_a_wheel_without_hf(tmp_path, monkeypatch, capsys):
    """A plain install has the wheel's models and no huggingface_hub. The CLI's
    default model must run there, and it is the library's default."""
    import text2tashkeel.__main__ as cli
    from text2tashkeel import _models

    for pattern in _WHEEL_MODEL_GLOBS:
        for src in _models._MODELS_DIR.glob(pattern):
            (tmp_path / src.name).symlink_to(src)
    monkeypatch.setattr(_models, "_MODELS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setattr(sys, "argv", ["text2tashkeel", "بسم الله الرحمن الرحيم"])
    _models._build.cache_clear()
    try:
        cli.main()
    finally:
        _models._build.cache_clear()
    # The output the README and the package docstring give for the default model.
    assert capsys.readouterr().out == "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ\n"
