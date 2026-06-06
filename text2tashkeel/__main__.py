"""CLI: `text2tashkeel [-m MODEL] "نص"` or pipe via stdin."""

from __future__ import annotations

import argparse
import sys

from . import Diacritizer, available_models


def main() -> None:
    ap = argparse.ArgumentParser(prog="text2tashkeel", description="Arabic diacritizer")
    ap.add_argument("text", nargs="*", help="text to diacritize (else read stdin)")
    ap.add_argument("-m", "--model", default="bilstm", choices=available_models())
    args = ap.parse_args()

    d = Diacritizer(args.model)
    if args.text:
        print(d.diacritize(" ".join(args.text)))
        return
    for line in sys.stdin:
        line = line.rstrip("\n")
        print(d.diacritize(line) if line else "")


if __name__ == "__main__":
    main()
