"""Report outputs, part 1 (Phase 14, A91): TA1, TA2, numbers.json, macros.tex, ledger.md, index.csv.

python scripts/06_report.py --config configs/default.yaml --results results_frozen --out figures
Stages: tables, numbers, ledger, index, all (default); show prints a compact view of the outputs.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import report_numbers as rn  # noqa: E402
from src.config import load_config  # noqa: E402
from src.report_tables import build_ta1, build_ta2  # noqa: E402

STAGES = ("tables", "numbers", "ledger", "index")
DATA_LOG = "DATA_LOG.md"
PREREG_FILE = "PREREG.sha256"
MANIFEST = "manifest.json"
RT_TABLES = {"rt_policy": "rt_policy_metrics.csv", "rt_rv1": "rt_rv1_metrics.csv"}
SHOW_LINES = 12


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_text(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def write_csv(path, df):
    df.to_csv(path, index=False, lineterminator="\n")


def report(path, rows=None):
    note = "" if rows is None else f"  rows={rows}"
    print(f"{Path(path).name}{note}  sha256={sha256(path)[:16]}")


def inputs(cfg, args):
    ta2 = build_ta2(args.config)
    ta1 = build_ta1(cfg, DATA_LOG, ta2)
    return ta1, ta2


def stage_tables(cfg, args, out):
    ta1, ta2 = inputs(cfg, args)
    for name, df in (("ta1_datasets.csv", ta1), ("ta2_parameters.csv", ta2)):
        write_csv(out / name, df)
        report(out / name, len(df))


def read_text_sig(path):
    return Path(path).read_text(encoding="utf-8-sig")


def stage_numbers(cfg, args, out):
    ta1, ta2 = inputs(cfg, args)
    tdir = Path(args.tables)
    tables = {name: (pd.read_csv(tdir / f), f"tables/{f}") for name, f in RT_TABLES.items()}
    entries = rn.build_numbers(
        ta1=ta1, ta2=ta2,
        queue=pd.read_csv(tdir / "queue_status.csv"),
        ledger=pd.read_csv(tdir / "ledger.csv"),
        manifest=json.loads(read_text_sig(Path(args.results) / MANIFEST)),
        amendments_text=read_text_sig(cfg["amendments_file"]),
        prereg_text=read_text_sig(PREREG_FILE),
        tables=tables,
        sources={"config": args.config, "data": "ta1_datasets.csv", "queue": "tables/queue_status.csv",
                 "manifest": f"{args.results}/{MANIFEST}", "amendments": cfg["amendments_file"],
                 "prereg": PREREG_FILE, "ledger": "tables/ledger.csv"})
    write_text(out / "numbers.json", rn.numbers_json_text(entries))
    write_text(out / "macros.tex", rn.macros_text(entries))
    report(out / "numbers.json", len(entries))
    report(out / "macros.tex", len(rn.macros_text(entries).splitlines()) - 1)


def stage_ledger(cfg, args, out):
    tdir = Path(args.tables)
    text = rn.render_ledger_md(pd.read_csv(tdir / "ledger.csv"), pd.read_csv(tdir / "queue_status.csv"))
    write_text(out / "ledger.md", text)
    report(out / "ledger.md", len(text.splitlines()))


def stage_index(cfg, args, out):
    index = rn.build_index(out, args.tables)
    write_csv(out / "index.csv", index)
    report(out / "index.csv", len(index))


def stage_show(cfg, args, out):
    ta1 = pd.read_csv(out / "ta1_datasets.csv")
    ta2 = pd.read_csv(out / "ta2_parameters.csv", keep_default_na=False)
    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    macros = (out / "macros.tex").read_text(encoding="utf-8").splitlines()
    print("== ta1 (no text columns)")
    print(ta1.drop(columns=["name", "source_url", "licence", "licence_check", "citation"]).to_csv(index=False))
    print("== ta1 text")
    print(ta1[["dataset", "name", "source_url", "licence", "citation"]].to_csv(index=False))
    print("== ta2 rows by tag")
    print(ta2["real_or_assumed"].value_counts().to_string())
    print(ta2.head(SHOW_LINES).to_csv(index=False))
    print(f"== numbers.json {len(numbers)} entries; macros.tex {len(macros) - 1} macros")
    for key in ("queue_n_grids", "queue_total_completed_cells", "queue_total_completed_rows",
                "queue_n_not_completed", "amendments_count", "amendments_last_id", "manifest_last_amendment_id",
                "ledger_n_hypotheses", "ledger_n_tests_total", "params_n_total", "cfg_n_boot", "prereg_sha256"):
        print(key, numbers[key]["value"])
    print("\n".join(macros[: SHOW_LINES + 1]))
    print("== ledger.md head")
    print("\n".join((out / "ledger.md").read_text(encoding="utf-8").splitlines()[: SHOW_LINES * 2]))
    print("== index.csv")
    print((out / "index.csv").read_text(encoding="utf-8"))


RUNNERS = {"tables": stage_tables, "numbers": stage_numbers, "ledger": stage_ledger, "index": stage_index}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default="results_frozen")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--stage", default="all", choices=STAGES + ("all", "show"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "show":
        stage_show(cfg, args, out)
        return
    for name in STAGES if args.stage == "all" else (args.stage,):
        RUNNERS[name](cfg, args, out)


if __name__ == "__main__":
    main()
