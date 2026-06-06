"""01 — Quickstart. The whole library in three lines.

Run:  python examples/01_quickstart.py
"""

from text2tashkeel import Diacritizer

# Build once, reuse. Defaults to the 'bilstm' model.
d = Diacritizer()

for sentence in [
    "بسم الله الرحمن الرحيم",   # "In the name of God, the Most Gracious..."
    "الحمد لله رب العالمين",    # "Praise be to God, Lord of the worlds"
    "هذا كتاب مفيد",            # "This is a useful book"
]:
    print("in :", sentence)
    print("out:", d.diacritize(sentence))   # d(sentence) also works
    print()

# One-off convenience (uses a shared default model):
from text2tashkeel import diacritize
print(diacritize("محمد رسول الله"))
