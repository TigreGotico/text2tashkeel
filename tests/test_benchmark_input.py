"""The benchmark scripts must score a model on bare input.

The models keep the marks they are given (``respect_existing``), so a script
that feeds the gold sentence scores the answer key against itself. An identity
model makes the leak visible: on bare input it returns no marks, so every
marked letter of the gold sentence counts as wrong.
"""
import importlib.util
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"

# بِسْمِ اللَّهِ — counted by hand: 8 base characters (the space included),
# of which 5 carry a mark (ب kasra, س sukun, م kasra, ل shadda+fatha, ه kasra),
# and 2 words, both with a mark.
GOLD = "بِسْمِ اللَّهِ"
EXPECTED = {"dw": 5, "dt": 8, "mw": 5, "mt": 5, "ww": 2, "wt": 2}


class Identity:
    def diacritize(self, text):
        return text


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_strips_gold_before_the_model(monkeypatch):
    bench = _load("benchmark")
    monkeypatch.setattr(bench, "split_lines", lambda split: iter([GOLD]))
    result = bench.run_split("test", {"identity": Identity()}, limit=0, max_len=400)
    assert result["n"] == 1
    assert result["stats"]["identity"] == EXPECTED


@pytest.mark.parametrize("script", ["parallel_benchmark", "bench_v2_gate", "bench_v3_value"])
def test_parallel_scripts_strip_gold_before_the_model(monkeypatch, script):
    bench = _load(script)
    monkeypatch.setattr(bench, "_MODELS", {"identity": Identity()})
    n, stats = bench._work([GOLD])
    assert n == 1
    assert stats["identity"] == EXPECTED
