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
    "Notes:\n"
    " - 'ensemble' is the MOST accurate: it gates rawi's marks by bilstm+libtashkeel\n"
    "   (decides where to mark) — see docs/09-combining-models.md.\n"
    " - 'bilstm' is the best single model and the default.\n"
    " - 'bilstm-int8' is 4x smaller but noticeably noisier (quantization).\n"
    " - 'rawi' knows the vowels best but over-marks on its own.\n"
    " - 'libtashkeel' is a separate project with its own training data.\n"
    "See benchmarks/README.md for measured accuracy and the contamination caveat."
)
