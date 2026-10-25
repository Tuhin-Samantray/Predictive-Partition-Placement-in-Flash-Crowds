import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_rows = _load("check_rows")
compare_rows = _load("compare_rows")


def _row(replicate=0, damage=1.5, ok=True, cell="D1-all-m5-h1.6-P0"):
    return {"cell": cell, "replicate": replicate, "event_id": "D1-1", "dataset": "D1", "variant": "P0",
            "damage": damage, "conservation_ok": ok, "moves": 3}


def _write(folder, rows, name="grid_1.jsonl"):
    folder.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
    (folder / name).write_text(text, encoding="utf-8")


def test_check_clean_rows_pass():
    rows = [_row(0), _row(1), _row(2)]
    assert check_rows.check(rows, len(rows)) == []


def test_check_flags_nan_negative_conservation_missing_and_duplicate():
    assert check_rows.check([_row(0, damage=float("nan"))], 1)
    assert check_rows.check([_row(0, damage=-1.0)], 1)
    assert check_rows.check([_row(0, ok=False)], 1)
    broken = _row(0)
    del broken["damage"]
    assert check_rows.check([broken], 1)
    assert check_rows.check([_row(0), _row(0)], 2)
    assert check_rows.check([_row(0)], 1, bad=1)


def test_check_only_inspects_first_n_rows_for_content():
    rows = [_row(0), _row(1, damage=-1.0)]
    assert check_rows.check(rows, 1) == []
    assert check_rows.check(rows, 2)


def test_load_rows_counts_unreadable_lines(tmp_path):
    _write(tmp_path, [_row(0), _row(1)])
    with open(tmp_path / "grid_1.jsonl", "a", encoding="utf-8") as handle:
        handle.write('{"cell": "broken\n')
    rows, bad, files = check_rows.load_rows(str(tmp_path / "grid_*.jsonl"))
    assert len(rows) == 2 and bad == 1 and len(files) == 1


def test_compare_identical_folders_pass(tmp_path):
    _write(tmp_path / "a", [_row(0), _row(1)])
    _write(tmp_path / "b", [_row(1), _row(0)], name="grid_2.jsonl")
    a, _ = compare_rows.load(tmp_path / "a")
    b, _ = compare_rows.load(tmp_path / "b")
    assert compare_rows.compare(a, b) == []


def test_compare_detects_changed_row_and_subset_rule(tmp_path):
    _write(tmp_path / "a", [_row(0), _row(1)])
    _write(tmp_path / "b", [_row(0, damage=2.5)])
    _write(tmp_path / "c", [_row(0)])
    a, _ = compare_rows.load(tmp_path / "a")
    b, _ = compare_rows.load(tmp_path / "b")
    c, _ = compare_rows.load(tmp_path / "c")
    assert compare_rows.compare(a, b, subset=True)
    assert compare_rows.compare(a, c, subset=True) == []
    assert compare_rows.compare(a, c)


def test_compare_flags_empty_folder(tmp_path):
    _write(tmp_path / "a", [_row(0)])
    (tmp_path / "b").mkdir()
    a, _ = compare_rows.load(tmp_path / "a")
    b, _ = compare_rows.load(tmp_path / "b")
    assert compare_rows.compare(a, b)
