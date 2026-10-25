"""Render configs/tuned_layers.yaml from the layers-on tuning jsonl files (A32).

python scripts/render_tuned_layers.py --config configs/default.yaml --results-dir results/tune_layers --out configs/tuned_layers.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.tuned_layers import NAMES, render_tuned_layers  # noqa: E402

DATASETS = ("D1", "D2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_config(args.config)
    summaries = []
    for name in NAMES:
        path = Path(args.results_dir) / (name + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        summaries += [r for r in rows if r["type"] == "summary"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_tuned_layers(summaries, DATASETS), encoding="utf-8", newline="\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
