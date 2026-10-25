import pandas as pd

from src import queuestatus as qs


def frame(cells, reps, conservation=True):
    rows = [{"cell": c, "replicate": r, "conservation_ok": conservation} for c in cells for r in range(reps)]
    return pd.DataFrame(rows)


def test_parse_planned_reads_the_first_dry_run_line():
    assert qs.parse_planned("cells 40, runs 1200 (30.0 runs per cell), tasks 120\nmore") == (40, 1200)
    assert qs.parse_planned("no pilot timing") is None and qs.parse_planned(None) is None


def test_complete_and_short_queues_are_told_apart():
    full = qs.grid_row("q1", "q1", qs.folder_counts(frame(["a", "b"], 3), 3), (2, 6), 3)
    assert full["status"] == "complete" and full["cells_below_min"] == 0
    missing = qs.grid_row("q1", "q1", qs.folder_counts(frame(["a"], 3), 3), (2, 6), 3)
    assert missing["status"] == "short"
    thin = qs.grid_row("q1", "q1", qs.folder_counts(frame(["a", "b"], 2), 3), (2, 4), 3)
    assert thin["status"] == "short" and thin["cells_below_min"] == 2


def test_duplicates_and_missing_plan_are_reported():
    dup = pd.concat([frame(["a"], 2), frame(["a"], 1)])
    counts = qs.folder_counts(dup, 2)
    assert counts["duplicate_keys"] == 1 and counts["completed_rows"] == 3 and counts["completed_cells"] == 1
    row = qs.grid_row("q2", "q2", qs.folder_counts(frame(["a"], 2, conservation=False), 2), None, 2)
    assert row["status"] == "planned count unavailable" and row["conservation_false"] == 2


def test_build_table_lists_every_queue_and_the_not_completed_studies():
    frames = {q: frame(["a", "b"], 2) for q in qs.QUEUE_FOLDER}
    planned = {q: (2, 4) for q in qs.QUEUE_FOLDER}
    table = qs.build_table(2, planned, frames, {"Q5": 468, "Q10": 7})
    assert list(table.columns) == qs.COLUMNS
    assert set(table["queue"]) == set(qs.QUEUE_FOLDER) | set(qs.NO_GRID) | {n for n, _, _ in qs.NOT_COMPLETED}
    status = dict(zip(table["queue"], table["status"]))
    assert status["q11b"] == "complete" and status["Q5"] == "complete, no cell grid" and status["Q9"] == "not completed"
    assert int(table.loc[table["queue"] == "Q5", "completed_rows"].iloc[0]) == 468
    assert table.loc[table["queue"] == "Q9", "planned_cells"].isna().all()
