"""Figures of the report (A94). Group 1: FA3, FA4, FA5, FA7. Reads recorded tables only."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.report_figures import make_group1  # noqa: E402
from src.report_figures2 import make_group2  # noqa: E402
from src.report_figures3 import make_group3  # noqa: E402

GROUPS = {"1": make_group1, "2": make_group2, "3": make_group3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--style", default="configs/report_style.json")
    ap.add_argument("--group", default="1", choices=sorted(GROUPS))
    a = ap.parse_args()
    cfg = load_config(a.config)
    for p in GROUPS[a.group](cfg, a.tables, a.out, a.style):
        print(p)


if __name__ == "__main__":
    main()
