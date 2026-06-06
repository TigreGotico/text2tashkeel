"""02 — The bundled models, side by side.

Each model adds marks to the SAME base letters; they differ in which marks they
predict and how they were trained. There is no single "correct" diacritization
of bare text — several readings can be valid — so the models legitimately differ.

Run:  python examples/02_choose_a_model.py
"""

from text2tashkeel import Diacritizer, available_models

print("available models:", available_models())
print()

sentences = [
    "بسم الله الرحمن الرحيم",
    "العلم نور والجهل ظلام",   # "Knowledge is light, ignorance is darkness"
]

# Build each once.
models = {name: Diacritizer(name) for name in available_models()}

for s in sentences:
    print("input:", s)
    for name, d in models.items():
        print(f"  {name:<12} {d.diacritize(s)}")
    print()

print(
    "Notes (numbers in docs/10-benchmark-report.md):\n"
    " - 'rawi-ensemble' is the default and most accurate: rawi-v2 decides where a\n"
    "   mark goes, rawi-v3's value head decides which, in one stitched ONNX (2.04% DER).\n"
    " - 'rawi-v2-int8' is the lean single model (2.30% DER, ~1 ms, 2.5 MB).\n"
    " - 'rawi' (V1) chooses the right mark best but over-marks on its own — a one-line\n"
    "   training bug (docs/09 §9.3); it serves as an ensemble value model.\n"
    " - int8 variants are lossless for these attention-free LSTMs, and smaller.\n"
    "See benchmarks/README.md and docs/10 for measured accuracy/latency + the caveat."
)
