"""Generate the benchmark plots for docs/ from the all-combos results.

    python benchmarks/generate_plots.py [results_all_combos.txt]

DER is parsed from the results file; latency (single-thread) and on-disk size are
measured constants (see docs/08). Writes PNGs to docs/images/.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results_all_combos.txt")
OUT = Path("docs/images"); OUT.mkdir(parents=True, exist_ok=True)

# latency ms/sentence (1 thread, ~58-char sentences) and on-disk MB
LAT = {
    "bilstm": 6.5, "bilstm-int8": 5.5, "rawi": 1.6, "rawi-int8": 1.1, "libtashkeel": 3.3,
    "bilstm+rawi": 8.6, "bilstm+rawi-int8": 7.8, "libtashkeel+rawi": 6.6,
    "libtashkeel+rawi-int8": 7.0, "bilstm-int8+rawi-int8": 8.7,
    "bilstm+libtashkeel+rawi": 21.0, "bilstm+libtashkeel+rawi-int8": 18.4,
}
SIZE = {
    "bilstm": 17.9, "bilstm-int8": 4.5, "rawi": 9.8, "rawi-int8": 2.5, "libtashkeel": 4.8,
    "bilstm+rawi": 27.7, "bilstm+rawi-int8": 20.4, "libtashkeel+rawi": 14.6,
    "libtashkeel+rawi-int8": 7.3, "bilstm-int8+rawi-int8": 7.0,
    "bilstm+libtashkeel+rawi": 32.5, "bilstm+libtashkeel+rawi-int8": 25.2,
}
SINGLES = {"bilstm", "bilstm-int8", "rawi", "rawi-int8", "libtashkeel"}
THREE = {"bilstm+libtashkeel+rawi", "bilstm+libtashkeel+rawi-int8"}

der, derm = {}, {}
for ln in RESULTS.read_text().splitlines():
    m = re.match(r"(\S+)\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\d+\.\d+)%", ln)
    if m and m.group(1) in LAT:
        der[m.group(1)] = float(m.group(2)); derm[m.group(1)] = float(m.group(3))
assert der, f"no rows parsed from {RESULTS}"

def cat(m):
    if m in SINGLES: return "single", "#888888", "o"
    if m in THREE: return "3-model ensemble", "#d62728", "*"
    return "2-model ensemble", "#1f77b4", "s"

def label(m):
    return m.replace("bilstm", "B").replace("libtashkeel", "L").replace("rawi", "R") \
            .replace("-int8", "₈").replace("+", "+")

def scatter(xkey, xlabel, fname, title, logx=True):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    seen = set()
    for m in der:
        x = xkey[m]; y = der[m]; name, color, mk = cat(m)
        ax.scatter(x, y, c=color, marker=mk, s=140 if mk == "*" else 80,
                   edgecolors="black", linewidths=0.6, zorder=3,
                   label=name if name not in seen else None, alpha=.9)
        seen.add(name)
        dx = 1.03 if not logx else 1.04
        ax.annotate(label(m), (x, y), fontsize=7.5, xytext=(x * dx, y + 0.12),
                    textcoords="data")
    if logx: ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("DER (all) %  — lower is better")
    ax.set_title(title); ax.grid(True, alpha=.3, zorder=0); ax.legend(loc="upper right")
    # highlight the sweet spot
    ss = "libtashkeel+rawi-int8"
    ax.annotate("sweet spot", (xkey[ss], der[ss]), fontsize=8, color="green",
                xytext=(xkey[ss] * (0.45 if logx else 0.8), der[ss] + 1.6),
                arrowprops=dict(arrowstyle="->", color="green"))
    fig.text(0.012, 0.012, "B=bilstm  L=libtashkeel  R=rawi  ₈=int8",
             fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    print("wrote", OUT / fname)

scatter(LAT, "latency  (ms / sentence, 1 thread — log scale)", "der_vs_latency.png",
        "Accuracy vs latency — text2tashkeel models  (before-TTS budget)")
scatter(SIZE, "on-disk size  (MB — log scale)", "der_vs_size.png",
        "Accuracy vs size — text2tashkeel models")

# DER(all) vs DER*(marked) — the over/under-marking diagnostic
order = sorted(der, key=lambda m: der[m])
fig, ax = plt.subplots(figsize=(9, 5.5))
import numpy as np
x = np.arange(len(order)); w = 0.4
ax.bar(x - w/2, [der[m] for m in order], w, label="DER (all)", color="#1f77b4")
ax.bar(x + w/2, [derm[m] for m in order], w, label="DER* (marked only)", color="#ff7f0e")
ax.set_xticks(x); ax.set_xticklabels([label(m) for m in order], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("error %"); ax.set_title("DER(all) vs DER*(marked) — over-marking shows as a big gap")
ax.legend(); ax.grid(True, axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "der_all_vs_marked.png", dpi=130); plt.close(fig)
print("wrote", OUT / "der_all_vs_marked.png")
