"""Result loaders and pairing for scripts/05_analyze.py (Phase 12, A57, A59). Pure functions; every parameter is an argument.

Rows are the jsonl rows of the grid runner. Pairing is strict: duplicate keys, rows without a partner and
rows whose partners carry different event ids raise an error instead of being dropped.
"""
import glob
import json
from pathlib import Path

import pandas as pd


def load_folder(folder):
    """All rows of one result folder (grid_*.jsonl), sorted by cell and replicate; duplicate keys raise."""
    files = sorted(glob.glob(str(Path(folder) / "grid_*.jsonl")))
    if not files:
        raise FileNotFoundError("no grid_*.jsonl files in " + str(folder))
    rows = []
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line:
                rows.append(json.loads(line))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["cell", "replicate"]).any():
        raise ValueError("duplicate (cell, replicate) keys in " + str(folder))
    return frame.sort_values(["cell", "replicate"]).reset_index(drop=True)


def load_queues(folders):
    """folders maps a queue name to its result folder; returns {name: frame} with a source column."""
    out = {}
    for name, folder in folders.items():
        frame = load_folder(folder)
        frame["source"] = name
        out[name] = frame
    return out


def select_variant(frame, variant_by_dataset):
    """Rows whose variant is the one named for their dataset (for example the best tuned reactive policy)."""
    wanted = frame["dataset"].map(variant_by_dataset)
    return frame[wanted == frame["variant"]].reset_index(drop=True)


def pair_rows(rows_a, rows_b, keys, value, check="event_id"):
    """One-to-one join of two row sets on keys. Returns the keys, the shared event id and value_a, value_b.
    Raises on duplicate keys, on rows without a partner and on partners with different event ids."""
    keys = list(keys)
    for name, rows in (("a", rows_a), ("b", rows_b)):
        if rows.duplicated(keys).any():
            raise ValueError("duplicate pairing keys in set " + name)
    left = rows_a[keys + [check, value]].rename(columns={check: check + "_a", value: value + "_a"})
    right = rows_b[keys + [check, value]].rename(columns={check: check + "_b", value: value + "_b"})
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("%d rows have no partner (a %d, b %d)" % (
            int((merged["_merge"] != "both").sum()), len(rows_a), len(rows_b)))
    if (merged[check + "_a"] != merged[check + "_b"]).any():
        raise ValueError("paired rows carry different " + check)
    merged = merged.drop(columns=["_merge", check + "_b"]).rename(columns={check + "_a": check})
    return merged.sort_values(keys).reset_index(drop=True)


def event_frame(frame, cluster, columns):
    """Per-event means of several columns (A57 item 2): one row per cluster."""
    return frame.groupby(list(cluster), sort=True)[list(columns)].mean().reset_index()


def resolve_folders(folders, root, base_key):
    """Re-root the folder mapping: the folder named by base_key becomes root and the others keep their place below it
    (so the same configuration reads results, a frozen copy or a clean-room copy)."""
    base = Path(folders[base_key])
    return {name: str(Path(root) / Path(folder).relative_to(base)) for name, folder in folders.items()}
