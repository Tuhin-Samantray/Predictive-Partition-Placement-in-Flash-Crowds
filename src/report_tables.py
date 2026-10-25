"""TA1 (datasets) and TA2 (parameters) builders for the report (Phase 14, A91).

Pure functions over files: nothing is typed by hand, every value comes from
configs/default.yaml, DATA_LOG.md, the processed data files or the events files.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_numeric_dtype

from src.config import split_points, step_seconds

TAGS = ("real", "assumed", "design")
SPLITS = ("train", "tune", "eval")
DATASETS = ("D1", "D2", "D1_D3", "D2_D3")
ROLES = {
    "D1": "primary real trace",
    "D2": "second real trace",
    "D1_D3": "D1 with injected events",
    "D2_D3": "D2 with injected events",
}
BYTES_KEY = {"D1": "d1_bytes_sigma", "D2": "d2_vsize_col"}
ONSET_COLUMNS = ("onset_class", "onset")
ALL_CLASS = "all"
SPLIT_COLUMN = "split"
ID_COLUMN = "event_id"
LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z_]\w*):(?P<rest>.*)$")
TAG_RE = re.compile(r"\s#\s*\[(?P<tag>real|assumed|design)\]\s*(?P<why>.*)$")


def parse_config_tags(path):
    """One row per key that carries a value; block headers are not rows."""
    text = Path(path).read_text(encoding="utf-8-sig")
    rows, stack = [], []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = LINE_RE.match(line)
        if m is None:
            raise ValueError(f"{path}:{number}: cannot parse line")
        indent = len(m["indent"])
        while stack and stack[-1][0] >= indent:
            stack.pop()
        rest = m["rest"]
        t = TAG_RE.search(rest)
        value = (rest[: t.start()] if t else rest).strip()
        if value == "" or value.startswith("#"):
            stack.append((indent, m["key"]))
            continue
        if t is None:
            raise ValueError(f"{path}:{number}: key {m['key']} has no [real], [assumed] or [design] tag")
        parent = ".".join(k for _, k in stack)
        rows.append({"key": m["key"], "parent": parent, "value": value,
                     "real_or_assumed": t["tag"], "justification": t["why"].strip()})
    return pd.DataFrame(rows, columns=["key", "parent", "value", "real_or_assumed", "justification"])


def _leaves(node, parent=""):
    out = []
    for key, val in node.items():
        if isinstance(val, dict):
            out += _leaves(val, f"{parent}.{key}" if parent else key)
        else:
            out.append((parent, key))
    return out


def build_ta2(path):
    """TA2: parsed rows, checked against the leaves that yaml itself reads."""
    rows = parse_config_tags(path)
    if rows.duplicated(["parent", "key"]).any():
        raise ValueError("duplicate keys in the config")
    parsed = set(zip(rows["parent"], rows["key"]))
    loaded = set(_leaves(yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))))
    if parsed != loaded:
        raise ValueError(f"tag parser and yaml disagree: only parsed {sorted(parsed - loaded)}, only yaml {sorted(loaded - parsed)}")
    return rows


def tag_of(ta2, key):
    hit = ta2.loc[(ta2["key"] == key) & (ta2["parent"] == ""), "real_or_assumed"]
    if len(hit) != 1:
        raise ValueError(f"config key {key} not found exactly once in TA2")
    return hit.iloc[0]


def _find_lines(text, pattern, what):
    m = re.search(pattern, text, flags=re.M)
    if m is None:
        raise ValueError(f"DATA_LOG.md: no line for {what} (pattern {pattern})")
    return m


def _section(text, prefix):
    parts = re.split(r"^## ", text, flags=re.M)
    hits = [p for p in parts if p.startswith(prefix)]
    if len(hits) != 1:
        raise ValueError(f"DATA_LOG.md: section {prefix} found {len(hits)} times")
    return hits[0]


def parse_data_log(path):
    """Facts of D1 and D2 as written in DATA_LOG.md; raises when a line is missing."""
    text = Path(path).read_text(encoding="utf-8-sig")
    out = {}
    for ds, source_key, raw_key in (("D1", "Page", "D1 archive"), ("D2", "Repository", "D2 file")):
        sec = _section(text, f"{ds}:")
        title = sec.splitlines()[0].split(":", 1)[1].strip()
        url = _find_lines(sec, rf"^- {source_key}: (\S+)", f"{ds} source").group(1)
        licence = _find_lines(sec, r"^- Licence: (.*)$", f"{ds} licence").group(1).strip()
        cite = _find_lines(sec, r"^- Citation[^:]*: (.*)$", f"{ds} citation").group(1).strip()
        if ds == "D1":
            check = _find_lines(text, r"^- D1 licence confirmed [^:]*: (.*)$", "D1 licence check").group(1).strip()
        else:
            check = licence
        raw = _find_lines(text, rf"^- {raw_key}: (\S+), (\d+) bytes, SHA-256 (\w+)\s*$", f"{ds} download")
        if ds == "D1":
            rows = _find_lines(text, r"^- D1: (\d+) invocation-file rows", "D1 rows").group(1)
        else:
            rows = _find_lines(text, r"^- D2 \S+: (\d+) rows", "D2 rows").group(1)
        out[ds] = {"name": title, "source_url": url, "licence": licence, "licence_check": check,
                   "citation": cite, "raw_file": raw.group(1), "raw_bytes": int(raw.group(2)),
                   "raw_sha256": raw.group(3), "raw_rows": int(rows)}
    return out


def _find_file(cfg, name):
    root = Path(cfg["data_root"])
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    preferred = [h for h in hits if cfg["processed_dirname"] in h.parts]
    return (preferred or hits)[0]


def _event_counts(events):
    if SPLIT_COLUMN not in events.columns:
        raise ValueError(f"events file lacks a split column: {list(events.columns)}")
    uniq = events.drop_duplicates(ID_COLUMN) if ID_COLUMN in events.columns else events
    onset = next((c for c in ONSET_COLUMNS if c in uniq.columns and not is_numeric_dtype(uniq[c])), None)
    classes = uniq[onset] if onset else pd.Series(ALL_CLASS, index=uniq.index)
    counts = {}
    for (split, cls), n in uniq.groupby([uniq[SPLIT_COLUMN], classes]).size().items():
        counts[f"events_{split}_{cls}"] = int(n)
    return len(events), len(uniq), counts


def build_ta1(cfg, data_log_path, ta2, *, split_fn=split_points, step_fn=step_seconds):
    """TA1: one row per dataset, all values read from files."""
    facts = parse_data_log(data_log_path)
    rows = []
    for ds in DATASETS:
        base = ds.split("_")[0]
        load = np.load(_find_file(cfg, f"{ds}_L.npy"), mmap_mode="r")
        events = pd.read_csv(_find_file(cfg, f"{ds}_events.csv"))
        n_steps, n_parts = int(load.shape[0]), int(load.shape[1])
        train_end, tune_end = split_fn(cfg, base, n_steps)
        step = step_fn(cfg, base)
        n_rows, n_events, counts = _event_counts(events)
        row = {"dataset": ds, "role": ROLES[ds], "step_seconds": step, "n_steps": n_steps,
               "span_days": n_steps * step / cfg["day_seconds"], "n_partitions": n_parts,
               "train_end": int(train_end), "tune_end": int(tune_end),
               "eval_steps": n_steps - int(tune_end), "n_event_rows": n_rows, "n_events": n_events}
        row.update(counts)
        fact = facts[base]
        if ds == base:
            row.update(fact)
        else:
            row.update({"name": fact["name"] + " with injected events", "source_url": fact["source_url"],
                        "licence": "derived from " + base + ": " + fact["licence"],
                        "licence_check": fact["licence_check"], "citation": fact["citation"]})
        key = BYTES_KEY[base]
        row["bytes_provenance"] = tag_of(ta2, key)
        row["bytes_tag_key"] = key
        rows.append(row)
    ta1 = pd.DataFrame(rows)
    for col in ("raw_bytes", "raw_rows"):
        ta1[col] = ta1[col].astype("Int64")
    ordered = [f"events_{s}_{c}" for s in SPLITS for c in sorted({k.split("_", 2)[2] for k in ta1.columns if k.startswith(f"events_{s}_")})]
    other = [c for c in ta1.columns if not c.startswith("events_")]
    ta1[ordered] = ta1[ordered].fillna(0).astype(int)
    first = ["dataset", "role", "name"]
    return ta1[first + [c for c in other if c not in first] + ordered]
