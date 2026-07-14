"""Vendored CATT tokenizer (Buckwalter encoding + 18-tag tashkeel scheme).

Original: Abjad AI — https://github.com/abjadai/catt (Apache-2.0). Unmodified except
for package-relative imports, so the `catt` model can run without an external dep.
"""
from .tashkeel_tokenizer_onnx import TashkeelTokenizer  # noqa: F401
