"""The docs match the code: figure data and the printed values of examples."""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
#: `expr  # 'value'`: an example line whose comment is the repr of its result
EXAMPLE = re.compile(r"^(?P<expr>\S.*?)\s+#\s+(?P<want>(['\"\[(]).*?\3|\[.*?\]|\(.*?\)|-?\d+|None|True|False)(\s|$)")


@pytest.mark.parametrize("script", [p for p in sorted((ROOT / "docs" / "images").glob("plot_*.py")) if p.name != "plot_style.py"], ids=lambda p: p.name)
def test_figure_data_matches_the_result_file(script, tmp_path):
    stem = re.search(r'parent / "([^"]+)"', script.read_text(encoding="utf-8"))[1]
    committed = script.with_name(f"{stem}.json")
    before = committed.read_bytes()
    try:
        subprocess.run([sys.executable, str(script)], check=True,
                       env={**os.environ, "DOCS_FIGURES_DATA_ONLY": "1"})
        assert committed.read_bytes() == before, f"rerun plot_{stem}: {committed.name} is stale"
    finally:
        committed.write_bytes(before)


def _example(line):
    """(expr, value) when *line* is an example whose comment is a Python literal, else None."""
    m = EXAMPLE.match(line)
    if not m:
        return None
    try:
        return m["expr"], ast.literal_eval(m["want"])
    except (ValueError, SyntaxError):
        return None


def _example_pages():
    """Pages with example lines; each page's blocks share one namespace, in order."""
    for page in DOCS:
        blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)
        # Blocks before the first import are code sketches, not runnable examples.
        first = next((i for i, b in enumerate(blocks) if "import " in b), len(blocks))
        blocks = [b for b in blocks[first:] if any(_example(line) for line in b.splitlines())]
        if blocks:
            yield page.relative_to(ROOT).as_posix(), blocks


@pytest.mark.parametrize("page, blocks", list(_example_pages()), ids=lambda v: v if isinstance(v, str) else "")
def test_example_prints_its_comment(page, blocks):
    def check(got, want, expr):
        assert got == want, f"{page}: {expr} gives {got!r}, the comment says {want!r}"

    namespace = {"_check": check}
    for block in blocks:
        lines = []
        for line in block.splitlines():
            example = _example(line)
            if example:
                expr, want = example
                line = f"_check({expr}, {want!r}, {expr!r})"
            lines.append(line)
        exec(compile("\n".join(lines), page, "exec"), namespace)  # noqa: S102 - runs the docs examples


def _results():
    """model → every (DER, DER*, WER) triple in benchmarks/results_*.txt, as printed."""
    rows = {}
    for path in (ROOT / "benchmarks").glob("results_*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"(\S+)\s+([0-9.]+)%\s+([0-9.]+)%\s+([0-9.]+)%", line)
            if m:
                rows.setdefault(m[1], set()).add((m[2], m[3], m[4]))
    return rows


def test_benchmark_table_cells_come_from_results():
    results = _results()
    checked = 0
    for page in DOCS:
        for line in page.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\| `([^`]+)`†? \| ([0-9.]+)% \| ([0-9.]+)% \| ([0-9.]+)% \|", line)
            if m:
                checked += 1
                assert (m[2], m[3], m[4]) in results.get(m[1], set()), \
                    f"{page.name}: {m[1]} {m[2]}/{m[3]}/{m[4]} is not a row of benchmarks/results_*.txt"
    assert checked >= 21
