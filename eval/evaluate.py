"""Reproduce the before/after fine-tuning evaluation from results/results.csv.

Usage:
    python eval/evaluate.py
"""
import csv
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def main() -> None:
    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    before = sum(1 for r in rows if r["verdict_before"].strip().lower() == "correct")
    after = sum(1 for r in rows if r["verdict_after"].strip().lower() == "correct")

    print(f"Cases evaluated : {n}")
    print(f"Before fine-tuning: {before}/{n} correct ({before / n:.1%})")
    print(f"After fine-tuning : {after}/{n} correct ({after / n:.1%})")
    print("\nFine-tuned model response distribution (mode collapse check):")
    for resp, cnt in Counter(r["response_after"] for r in rows).most_common():
        print(f"  {cnt:>2}x  {resp[:70]}")


if __name__ == "__main__":
    main()
