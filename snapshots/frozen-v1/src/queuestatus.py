"""Planned versus completed cells per queue (guide Phase 13.1, A83).

The planned counts come from the runner's own dry run (`04_run_grid.py --queue <q> --dry-run`, first line "cells N, runs M"),
the completed counts from the result rows. Cells with fewer replicates than `min_replicates` are flagged, never padded.
Only the numbers 0, 1 and 2 appear here (guide rule G1).
"""
import re

import pandas as pd

PLANNED_LINE = re.compile(r"cells (\d+), runs (\d+)")
QUEUE_FOLDER = {"must": "must", "q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4", "q6": "q6", "q8": "q8",
                "q11": "q11_rv0", "q11b": "q11_rv123"}
# forecast-only studies without a cell grid: name -> (what, table holding their rows)
NO_GRID = {"Q5": ("forecast-only cross-data (A37)", "forecast_cross.csv"),
           "Q10": ("Chronos accuracy only (A50, A51)", "forecast_accuracy_chronos.csv")}
NOT_COMPLETED = [("Q9", "FA16 sensitivity", "A52"), ("Q8-XA6", "XA6 cross-resolution", "A52"), ("Q7", "AA13 scale", "A52"),
                 ("D4", "Azure 2021 trace", "A52"), ("XA3", "XA3", "A52"), ("pairwise", "interaction pairwise queue", "A61")]
COLUMNS = ["queue", "folder", "kind", "planned_cells", "planned_runs", "completed_cells", "completed_rows", "replicates_min",
           "replicates_max", "min_replicates", "cells_below_min", "duplicate_keys", "conservation_false", "status", "note"]


def parse_planned(text):
    """(cells, runs) from the first dry-run line, or None."""
    match = PLANNED_LINE.search(text or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def folder_counts(frame, min_replicates):
    per_cell = frame.groupby("cell")["replicate"].nunique()
    conservation = int((~frame["conservation_ok"].astype(bool)).sum()) if "conservation_ok" in frame.columns else 0
    return {"completed_cells": int(per_cell.size), "completed_rows": int(len(frame)),
            "replicates_min": int(per_cell.min()) if per_cell.size else 0,
            "replicates_max": int(per_cell.max()) if per_cell.size else 0,
            "cells_below_min": int((per_cell < min_replicates).sum()),
            "duplicate_keys": int(frame.duplicated(["cell", "replicate"]).sum()),
            "conservation_false": conservation}


def grid_row(queue, folder, counts, planned, min_replicates):
    row = {"queue": queue, "folder": folder, "kind": "cell grid", "min_replicates": min_replicates}
    row.update(counts)
    if planned is None:
        row.update({"planned_cells": None, "planned_runs": None, "status": "planned count unavailable",
                    "note": "the dry run did not print a cell count; counts below are the completed rows"})
        return row
    row["planned_cells"], row["planned_runs"] = planned
    full = (counts["completed_cells"] == planned[0] and counts["completed_rows"] == planned[1]
            and counts["cells_below_min"] == 0 and counts["duplicate_keys"] == 0)
    row["status"] = "complete" if full else "short"
    row["note"] = "" if full else "completed counts differ from the planned counts or a cell has fewer replicates than min_replicates"
    return row


def build_table(min_replicates, planned, frames, table_rows):
    """planned: queue -> (cells, runs) or None; frames: queue -> DataFrame of result rows; table_rows: study -> row count."""
    rows = []
    for queue, folder in QUEUE_FOLDER.items():
        rows.append(grid_row(queue, folder, folder_counts(frames[queue], min_replicates), planned.get(queue), min_replicates))
    for name, (what, table) in NO_GRID.items():
        rows.append({"queue": name, "folder": "tables/" + table, "kind": "no cell grid", "completed_rows": int(table_rows[name]),
                     "status": "complete, no cell grid", "note": what})
    for name, what, amendment in NOT_COMPLETED:
        rows.append({"queue": name, "folder": "", "kind": "not completed", "completed_cells": 0, "completed_rows": 0,
                     "status": "not completed", "note": what + " (" + amendment + ")"})
    return pd.DataFrame(rows, columns=COLUMNS).convert_dtypes()
