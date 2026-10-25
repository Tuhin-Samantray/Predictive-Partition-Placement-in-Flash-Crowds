"""Write tables/queue_status.csv: planned versus completed cells per queue (guide Phase 13.1, A83).

Usage: python scripts/queue_status.py --config configs/default.yaml --out tables
The planned counts are read from the first line of each queue's own dry run. Nothing is padded or hidden.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import queuestatus  # noqa: E402
from src.analysis_data import load_folder  # noqa: E402


def dry_run_counts(config, queue):
    command = [sys.executable, "scripts/04_run_grid.py", "--config", config, "--tuned", "configs/tuned.yaml",
               "--tuned-ext", "configs/tuned_ext.yaml", "--tuned-layers", "configs/tuned_layers.yaml", "--queue", queue, "--dry-run"]
    done = subprocess.run(command, capture_output=True, text=True)
    return queuestatus.parse_planned(done.stdout + "\n" + done.stderr) if done.returncode == 0 else None


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    planned = {queue: dry_run_counts(args.config, queue) for queue in queuestatus.QUEUE_FOLDER}
    frames = {queue: load_folder(cfg["result_folders"][folder]) for queue, folder in queuestatus.QUEUE_FOLDER.items()}
    table_rows = {name: len(pd.read_csv(Path("tables") / table)) for name, (_, table) in queuestatus.NO_GRID.items()}
    table = queuestatus.build_table(cfg["min_replicates"], planned, frames, table_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "queue_status.csv", index=False, lineterminator="\n")
    print(table.drop(columns=["folder", "note"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
