"""Prepare CATT's test set for an apples-to-apples bench.

72% of CATT's test appears in our train (= rawi-v2's training data), so a naive
rawi-v2 run on CATT's test would be hugely inflated. We emit two files:

  catt_test_full.txt       — CATT's test as-is
  catt_test_rawiclean.txt  — CATT's test minus any sentence whose bare form is in
                             our train split → provably unseen by rawi-v2

We also check how much of (each) CATT test file is in CATT's *own* train, to
confirm the clean set is also fair for CATT (its held-out test).
"""
import unicodedata
from pathlib import Path

OUR = Path.home() / (
    ".cache/huggingface/hub/datasets--TigreGotico--arabic_diacritized_text/"
    "snapshots/b9ac17960a5d01bd110267a0b736ab844ad30f71"
)
CATT = Path("/home/miro/AgentWorkspaces/.scratch/catt/dataset")
OUT = Path(__file__).resolve().parent

_STRIP = {chr(c) for c in range(0x64B, 0x653)} | {"ٰ"}


def bare(line: str) -> str:
    s = unicodedata.normalize("NFC", line.strip())
    return " ".join("".join(c for c in s if c not in _STRIP).split())


def main():
    print("loading our train keys ...", flush=True)
    our_train = set()
    with open(OUR / "train.txt", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                our_train.add(bare(ln))
    print(f"  our train unique: {len(our_train)}")

    catt_test = [ln.rstrip("\n") for ln in open(CATT / "test" / "test.txt", encoding="utf-8") if ln.strip()]
    print(f"CATT test lines: {len(catt_test)}")

    clean = [g for g in catt_test if bare(g) not in our_train]
    (OUT / "catt_test_full.txt").write_text("\n".join(catt_test) + "\n", encoding="utf-8")
    (OUT / "catt_test_rawiclean.txt").write_text("\n".join(clean) + "\n", encoding="utf-8")
    print(f"  full:       {len(catt_test)}")
    print(f"  rawi-clean: {len(clean)}  (removed {len(catt_test)-len(clean)} "
          f"= {100*(len(catt_test)-len(clean))/len(catt_test):.1f}% seen by rawi)")

    # is CATT test held out from CATT's own train? stream the 1 GB once.
    catt_test_keys = {bare(g) for g in catt_test}
    clean_keys = {bare(g) for g in clean}
    in_ct = in_ct_clean = 0
    seen = set()
    print("streaming CATT train to check CATT-test exposure ...", flush=True)
    for ln in open(CATT / "train" / "train.txt", encoding="utf-8"):
        if not ln.strip():
            continue
        b = bare(ln)
        if b in seen:
            continue
        seen.add(b)
        if b in catt_test_keys:
            in_ct += 1
        if b in clean_keys:
            in_ct_clean += 1
    print(f"  CATT test    in CATT train: {in_ct}/{len(catt_test)} "
          f"({100*in_ct/len(catt_test):.1f}%)")
    print(f"  rawi-clean   in CATT train: {in_ct_clean}/{len(clean)} "
          f"({100*in_ct_clean/max(1,len(clean)):.1f}%)  "
          f"<- CATT's exposure to the fair set")


if __name__ == "__main__":
    main()
