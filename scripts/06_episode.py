"""Episode figures FA2 and review Figure 3 (A104). Regenerates one rule-selected replicate per dataset from the
frozen results and draws it only if every drawn policy reproduces its frozen row. Exit status 1 if a dataset
did not reproduce."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.report_episode import make_episodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results_frozen")
    ap.add_argument("--tuned", default="configs/tuned.yaml")
    ap.add_argument("--tuned-ext", default="configs/tuned_ext.yaml")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--style", default="configs/report_style.json")
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths, bad = make_episodes(cfg, a.results, a.tuned, a.tuned_ext, a.out, a.style)
    for p in paths:
        print(p)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
