"""Per-row comparison of two result folders (guide Phase 8.2 spot-checks and Phase 13 clean-room rerun).

Usage: python scripts/compare_rows.py <folder A> <folder B> [--subset] [--show N]
Rows are matched on (cell, replicate) and compared by SHA-256 of their canonical JSON (rows hold no timing, so
equal runs give equal rows). Default: the two key sets must be equal. With --subset every row of B must exist in A
and match (A may hold more rows), as for --only-row spot-checks.
Exit code 0 if no problem was found, 1 otherwise.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def load(folder):
    rows, duplicates = {}, []
    for path in sorted(Path(folder).glob("grid_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            key = (row["cell"], row["replicate"])
            if key in rows:
                duplicates.append(key)
            rows[key] = row
    return rows, duplicates


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest()


def compare(a, b, subset=False):
    problems = []
    if not a or not b:
        problems.append("a folder has no rows (A %d, B %d)" % (len(a), len(b)))
    for key in sorted(b):
        if key not in a:
            problems.append("only in B: %s:%s" % key)
        elif row_hash(a[key]) != row_hash(b[key]):
            problems.append("row differs: %s:%s" % key)
    if not subset:
        for key in sorted(a):
            if key not in b:
                problems.append("only in A: %s:%s" % key)
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_a")
    parser.add_argument("folder_b")
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--show", type=int)
    args = parser.parse_args()
    a, dup_a = load(args.folder_a)
    b, dup_b = load(args.folder_b)
    problems = ["duplicate key in A: %s:%s" % k for k in dup_a] + ["duplicate key in B: %s:%s" % k for k in dup_b]
    problems += compare(a, b, args.subset)
    print("rows A %d, rows B %d, matched %d" % (len(a), len(b), len(set(a) & set(b))))
    shown = problems if args.show is None else problems[:args.show]
    for p in shown:
        print("PROBLEM: " + p)
    if len(shown) < len(problems):
        print("... %d more problems" % (len(problems) - len(shown)))
    print("FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
