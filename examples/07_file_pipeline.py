"""07 — Diacritize a text file, line by line.

Reads examples/sample.txt and prints each line diacritized. Swap in your own
file, or use the CLI for the same effect:

    text2tashkeel < examples/sample.txt
    python -m text2tashkeel -m libtashkeel < examples/sample.txt

Run:  python examples/07_file_pipeline.py
"""

from pathlib import Path

from text2tashkeel import Diacritizer

d = Diacritizer()  # try Diacritizer("libtashkeel") to compare
path = Path(__file__).parent / "sample.txt"

for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    print(d.diacritize(line))
