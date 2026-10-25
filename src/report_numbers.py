"""Numbers, macros, ledger rendering and index for the report (Phase 14, A91).

Every number in the paper comes from numbers.json or macros.tex, and both are
built here from recorded files; nothing is typed by hand.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MACRO_PREFIX = "pa"
NA_TEXT = "n/a"
SIG = 4
SIG_P = 3
SMALL = 10 ** -3
P_KEY = "p_holm"
STRING_PREFIXES = ("ledger_", "manifest_", "prereg_")
DIGIT_WORDS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
LATEX_SPECIAL = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                 "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
AMENDMENT_ROW = re.compile(r"^\| A(\d+) \|", re.M)
QUEUE_COLUMNS = ("planned_cells", "planned_runs", "completed_cells", "completed_rows", "replicates_min",
                 "replicates_max", "min_replicates", "cells_below_min", "duplicate_keys", "conservation_false")
LEDGER_COLUMNS = ("id", "hypothesis", "quantity", "threshold", "result", "label", "n_tests", "holm_family",
                  "p_holm_min", "status", "exploratory_columns", "note")
NOT_COMPLETED = "not completed"


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def _plain(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _whole(value):
    value = _plain(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _add(entries, key, value, source):
    if key in entries:
        raise ValueError(f"duplicate number key {key}")
    entries[key] = {"value": _plain(value), "source": source}


def camel(key):
    out = []
    for piece in (p for p in re.split(r"[^A-Za-z0-9]+", key) if p):
        spelled = "".join(DIGIT_WORDS[int(c)] if c.isdigit() else c for c in piece)
        out.append(spelled[:1].upper() + spelled[1:])
    return "".join(out)


def macro_name(key):
    return MACRO_PREFIX + camel(key)


def latex_escape(text):
    return "".join(LATEX_SPECIAL.get(c, c) for c in str(text))


def format_value(value, key=""):
    """Text of a number as printed in the paper."""
    value = _plain(value)
    if value is None:
        return NA_TEXT
    if isinstance(value, bool):
        raise TypeError("booleans have no macro")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return NA_TEXT
        small = P_KEY in key or (value != 0 and abs(value) < SMALL)
        text = f"{value:.{SIG_P if small else SIG}g}"
        if "e" in text:
            mant, exp = text.split("e")
            return rf"\ensuremath{{{mant}\times 10^{{{int(exp)}}}}}"
        return text
    return latex_escape(value)


def config_entries(entries, ta2, source):
    for _, r in ta2.iterrows():
        parent = slug(r["parent"]) + "_" if r["parent"] else ""
        _add(entries, f"cfg_{parent}{r['key']}", yaml.safe_load(r["value"]), source)
    for tag, n in ta2["real_or_assumed"].value_counts().sort_index().items():
        _add(entries, f"params_n_{tag}", int(n), source + " (count of tags)")
    _add(entries, "params_n_total", len(ta2), source + " (count of tags)")


def data_entries(entries, ta1, source):
    for _, r in ta1.iterrows():
        for col in ta1.columns:
            val = _plain(r[col])
            if col == "dataset" or val is None or val == "" or isinstance(val, bool):
                continue
            _add(entries, f"data_{r['dataset']}_{col}", val, source)


def queue_entries(entries, queue, source):
    grid = queue["planned_cells"].fillna(0) > 0
    for _, r in queue.iterrows():
        base = f"queue_{slug(r['queue'])}"
        for col in QUEUE_COLUMNS:
            if col in queue.columns and _plain(r[col]) is not None:
                _add(entries, f"{base}_{col}", _whole(r[col]), source)
        _add(entries, f"{base}_status", r["status"], source)
    _add(entries, "queue_n_grids", int(grid.sum()), source + " (rows with planned_cells above 0)")
    for col in ("completed_cells", "completed_rows"):
        _add(entries, f"queue_total_{col}", _whole(queue.loc[grid, col].sum()), source + " (grid rows)")
    done = queue["status"].astype(str).str.lower() == NOT_COMPLETED
    _add(entries, "queue_n_not_completed", int(done.sum()), source)


def manifest_entries(entries, manifest, source):
    _add(entries, "manifest_created_utc", manifest["created_utc"], source)
    _add(entries, "manifest_tree_sha256", manifest["tree_sha256"], source)
    _add(entries, "manifest_last_amendment_id", int(manifest["last_amendment_id"]), source)
    _add(entries, "manifest_python", manifest["python"], source)
    _add(entries, "manifest_n_files", len(manifest["files"]), source)
    _add(entries, "manifest_n_locked", len(manifest["locked"]), source)


def amendment_entries(entries, text, source):
    ids = sorted({int(x) for x in AMENDMENT_ROW.findall(text)})
    _add(entries, "amendments_count", len(ids), source)
    _add(entries, "amendments_last_id", ids[-1], source)


def prereg_entry(entries, text, source):
    _add(entries, "prereg_sha256", text.split()[0], source)


def ledger_entries(entries, ledger, source):
    _add(entries, "ledger_n_hypotheses", len(ledger), source)
    tests = pd.to_numeric(ledger["n_tests"], errors="coerce").fillna(0)
    _add(entries, "ledger_n_tests_total", _whole(float(tests.sum())), source)
    for status, grp in ledger.groupby("status"):
        _add(entries, f"ledger_n_status_{slug(status)}", len(grp), source)
        _add(entries, f"ledger_n_tests_status_{slug(status)}", _whole(float(tests[grp.index].sum())), source)
    for label, n in ledger["label"].value_counts().sort_index().items():
        _add(entries, f"ledger_n_label_{slug(label)}", int(n), source)
    for _, r in ledger.iterrows():
        base = f"ledger_{slug(r['id'])}"
        _add(entries, f"{base}_label", r["label"], source)
        for col in ("n_tests", "p_holm_min"):
            val = _whole(pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0])
            if _plain(val) is not None:
                _add(entries, f"{base}_{col}", val, source)
        _add(entries, f"{base}_holm_family", r["holm_family"], source)


def split_columns(df):
    keys = [c for c in df.columns if not is_numeric_dtype(df[c]) or is_bool_dtype(df[c])]
    if not keys:
        raise ValueError("table has no key column")
    ints = [c for c in df.columns if c not in keys and pd.api.types.is_integer_dtype(df[c])]
    while df[keys].astype(str).duplicated().any():
        if not ints:
            raise ValueError("rows of the table are not unique by their key columns")
        keys.append(ints.pop(0))
    return keys, [c for c in df.columns if c not in keys and not is_bool_dtype(df[c])]


def table_entries(entries, name, df, source):
    keys, values = split_columns(df)
    for _, r in df.iterrows():
        ident = "_".join(slug(r[k]) for k in keys)
        for col in values:
            _add(entries, f"{name}_{ident}_{col}", r[col], source)


def build_numbers(*, ta1, ta2, queue, ledger, manifest, amendments_text, prereg_text, tables, sources):
    entries = {}
    config_entries(entries, ta2, sources["config"])
    data_entries(entries, ta1, sources["data"])
    queue_entries(entries, queue, sources["queue"])
    manifest_entries(entries, manifest, sources["manifest"])
    amendment_entries(entries, amendments_text, sources["amendments"])
    prereg_entry(entries, prereg_text, sources["prereg"])
    ledger_entries(entries, ledger, sources["ledger"])
    for name, (df, source) in tables.items():
        table_entries(entries, name, df, source)
    return entries


def numbers_json_text(entries):
    return json.dumps(entries, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def macros_text(entries):
    lines = ["% generated by scripts/06_report.py from numbers.json; do not edit"]
    seen = {}
    for key, e in entries.items():
        val = e["value"]
        if isinstance(val, (bool, list, dict)):
            continue
        if isinstance(val, str) and not key.startswith(STRING_PREFIXES):
            continue
        name = macro_name(key)
        if name in seen:
            raise ValueError(f"macro name clash: {key} and {seen[name]} give {name}")
        seen[name] = key
        lines.append(rf"\newcommand{{\{name}}}{{{format_value(val, key)}}}")
    return "\n".join(lines) + "\n"


def _cell(value):
    value = _plain(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.{SIG_P}g}"
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_ledger_md(ledger, queue):
    cols = [c for c in LEDGER_COLUMNS if c in ledger.columns]
    if "id" not in cols:
        raise ValueError("ledger has no id column")
    tests = pd.to_numeric(ledger["n_tests"], errors="coerce").fillna(0)
    out = ["# Hypothesis ledger", "",
           "Source: tables/ledger.csv, generated by scripts/06_report.py (do not edit).", "",
           f"Hypotheses: {len(ledger)}; tests in the ledger: {_cell(float(tests.sum()))}.", ""]
    for status, grp in ledger.groupby("status"):
        out.append(f"- {status}: {len(grp)} hypotheses, {_cell(float(tests[grp.index].sum()))} tests")
    out += ["", "Not completed (no result is claimed):", ""]
    todo = queue[queue["status"].astype(str).str.lower() == NOT_COMPLETED]
    for _, r in todo.iterrows():
        note = _plain(r["note"]) if "note" in queue.columns else None
        out.append(f"- {r['queue']}" + (f": {note}" if note else ""))
    out += ["", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in ledger.iterrows():
        out.append("| " + " | ".join(_cell(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


TABLE_ROWS = (
    ("TA1", "", "Datasets", "ta1_datasets.csv", "figures"),
    ("TA2", "", "Parameters with real, assumed or design tag", "ta2_parameters.csv", "figures"),
    ("TA3", "", "Forecast accuracy", "", "tables:ta3_"),
    ("TA4", "", "Main results", "", "tables:ta4_"),
    ("TA5", "", "Regime map", "", "tables:ta5_"),
    ("TA6", "", "Ablations", "", "tables:ta6_"),
    ("TA7", "", "Cross-dataset results", "", "tables:ta7_"),
    ("TA8", "", "Cross-layer and scenarios", "", "tables:ta8_"),
    ("TA9", "", "Statistics and ledger", "", "tables:ta9_"),
    ("RT1", "5", "Contention levels per policy (descriptive)", "rt_rv1_metrics.csv", "tables"),
    ("RT2", "", "Damage, violation fraction, backlog and moves per policy (descriptive)", "rt_policy_metrics.csv", "tables"),
)
INFERRED = "mapping inferred from table names (A91), confirm against the guide when the figure is made"
FIGURE_ROWS = (
    ("FA2", "3", "Episode: events and policy trajectories", "", "fa2", "regenerated under the A88 rule", ""),
    ("FA3", "", "Forecast accuracy", "ta3_forecast_accuracy.csv", "fa3", INFERRED, ""),
    ("FA4", "", "Early warning", "forecast_accuracy_early_warning.csv", "fa4", INFERRED, ""),
    ("FA5", "", "Lag tax", "c1_lag_tax.csv", "fa5", INFERRED, ""),
    ("FA6", "4", "Regime map", "regime_map_q2.csv;regime_map_must.csv", "fa6", INFERRED, ""),
    ("FA7", "1;2", "Main comparison", "rt_policy_metrics.csv", "fa7", INFERRED, ""),
    ("n/a", "5", "Main comparison per dataset", "rt_policy_metrics.csv", "rev5", INFERRED, ""),
    ("FA8", "", "Cross-dataset transfer", "c5_transfer_detail.csv;xa1_transfer_matrix.csv", "fa8", INFERRED, ""),
    ("FA9", "", "Cross-dataset ranking", "c6_ranking.csv", "fa9", INFERRED, ""),
    ("FA10", "", "Ablations", "aa_ablations.csv", "fa10", INFERRED, ""),
    ("FA11", "", "Damage against moves", "rt_policy_metrics.csv", "fa11", INFERRED, ""),
    ("FA12", "7", "MASE against placement benefit", "c4_accuracy_benefit_cells.csv", "fa12", INFERRED, ""),
    ("FA13", "6", "Cross-layer contention", "rt_rv1_metrics.csv;rv1_contention.csv", "fa13", INFERRED, ""),
    ("FA14", "8", "Conformal pooled coverage and width", "rv4_logging.csv", "fa14", INFERRED, ""),
    ("FA15", "10", "Placement and throttle", "rv2_placement.csv;rv3_budget.csv", "fa15", INFERRED, ""),
    ("n/a", "9", "Sensitivity to migration duration", "regime_map_must.csv", "rev9", INFERRED, ""),
    ("FA16", "", "Sensitivity tornado", "", "fa16", "not made (A88)", NOT_COMPLETED),
)
INDEX_COLUMNS = ["kind", "guide_id", "review_number", "title", "source_tables", "status", "note"]


def build_index(out_dir, tables_dir, figures_dir=None):
    out_dir, tables_dir = Path(out_dir), Path(tables_dir)
    figures_dir = Path(figures_dir) if figures_dir else out_dir
    rows = []
    for gid, review, title, source, where in TABLE_ROWS:
        if where.startswith("tables:"):
            hits = sorted(p.name for p in tables_dir.glob(where.split(":", 1)[1] + "*.csv"))
            names, found = ";".join(hits), bool(hits)
        else:
            folder = out_dir if where == "figures" else tables_dir
            names, found = source, (folder / source).exists()
        rows.append(["table", gid, review, title, names, "made" if found else "to make", ""])
    for gid, review, title, source, prefix, note, forced in FIGURE_ROWS:
        names = [n for n in source.split(";") if n]
        missing = [n for n in names if not (tables_dir / n).exists()]
        made = bool(list(figures_dir.glob(prefix + "_*.pdf"))) or (figures_dir / (prefix + ".pdf")).exists()
        status = forced or ("made" if made else ("source missing" if missing else "to make"))
        extra = note + ("; missing: " + ";".join(missing) if missing else "")
        rows.append(["figure", gid, review, title, ";".join(names), status, extra])
    return pd.DataFrame(rows, columns=INDEX_COLUMNS)
