"""Row sanity check (guide Phase 8.2).

Usage: python scripts/check_rows.py --glob "results/grid_*.jsonl" --n 100
Reads every row matching the glob. The first --n rows get the content checks (required fields, finite numbers,
damage non-negative, conservation_ok true); (cell, replicate) keys must be unique over all rows read.
Exit code 0 if no problem was found, 1 otherwise.
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

REQUIRED = ("cell", "replicate", "event_id", "dataset", "variant", "damage", "conservation_ok")


def load_rows(pattern):
    rows, bad = [], 0
    files = sorted(glob.glob(pattern))
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad, files


def check(rows, n, bad=0):
    problems = []
    if bad:
        problems.append("%d unreadable lines" % bad)
    if not rows:
        problems.append("no rows found")
    for i, row in enumerate(rows[:n]):
        where = "row %d (%s:%s)" % (i, row.get("cell"), row.get("replicate"))
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            problems.append("%s: missing fields %s" % (where, missing))
            continue
        for key, val in row.items():
            if val is None:
                problems.append("%s: %s is null" % (where, key))
            elif isinstance(val, float) and not math.isfinite(val):
                problems.append("%s: %s is not finite" % (where, key))
        if isinstance(row["damage"], (int, float)) and row["damage"] < 0:
            problems.append("%s: damage %r is negative" % (where, row["damage"]))
        if row["conservation_ok"] is not True:
            problems.append("%s: conservation_ok is not true" % where)
    seen = set()
    for row in rows:
        key = (row.get("cell"), row.get("replicate"))
        if key in seen:
            problems.append("duplicate row key %s:%s" % key)
        seen.add(key)
    return problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--n", type=int, required=True)
    args = parser.parse_args()
    rows, bad, files = load_rows(args.glob)
    problems = check(rows, args.n, bad)
    print("rows read %d from %d files, content checks on the first %d" % (len(rows), len(files), min(args.n, len(rows))))
    for p in problems[:args.n]:
        print("PROBLEM: " + p)
    if len(problems) > args.n:
        print("... %d more problems" % (len(problems) - args.n))
    print("FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
