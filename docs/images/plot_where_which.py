"""DER on all positions and DER* on marked positions for the rawi family, from benchmarks/results_*.txt."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import save

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
SOURCES = {
    "rawi": "results_rawi_v2_fulltest.txt",
    "rawi-v2": "results_rawi_v2_fulltest.txt",
    "rawi-v3": "results_v3_fulltest.txt",
    "rawi-v2+rawi": "results_v2_gate_fulltest.txt",
    "rawi-v2+rawi-v3": "results_v3value_fulltest.txt",
    "rawi-ensemble": "results_rawi_ensemble_fulltest.txt",
}


def row(model, fname):
    for line in (BENCH / fname).read_text(encoding="utf-8").splitlines():
        m = re.match(rf"{re.escape(model)}\s+([0-9.]+)%\s+([0-9.]+)%", line)
        if m:
            return float(m[1]), float(m[2])
    raise ValueError(f"{model} not in {fname}")


rows = [(m, *row(m, f)) for m, f in SOURCES.items()]


def draw(fig, theme):
    ax = fig.subplots()
    y = range(len(rows))
    h = 0.38
    ax.barh([i - h / 2 for i in y], [r[1] for r in rows], h, label="DER, all positions (where + which)",
            color=theme["series"][0])
    ax.barh([i + h / 2 for i in y], [r[2] for r in rows], h, label="DER*, marked positions (which)",
            color=theme["series"][1])
    for i, r in zip(y, rows):
        ax.text(r[1] + 0.2, i - h / 2, f"{r[1]:.2f}", va="center", fontsize=8, color=theme["text"])
        ax.text(r[2] + 0.2, i + h / 2, f"{r[2]:.2f}", va="center", fontsize=8, color=theme["text"])
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("error rate (%), lower is better")
    ax.legend(loc="lower right")


save(str(Path(__file__).parent / "where-which"), draw)
