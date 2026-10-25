"""Render configs/tuned_ext.yaml from the P4a tuning jsonl (A29).

python scripts/render_tuned_ext.py --config configs/default.yaml --jsonl results/tune_ext/P4a.jsonl --out configs/tuned_ext.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.tuned_ext import render_tuned_ext  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_config(args.config)
    rows = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line]
    summaries = [r for r in rows if r["type"] == "summary"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tuned_ext(summaries, DATASETS), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
