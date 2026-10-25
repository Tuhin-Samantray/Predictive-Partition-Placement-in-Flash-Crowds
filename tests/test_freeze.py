import json
import os
import stat

import pytest

from src import freeze


def make_tree(root):
    (root / "q1").mkdir(parents=True)
    (root / "grid_1.jsonl").write_text('{"cell": "a", "replicate": 0}\n{"cell": "a", "replicate": 1}\n\n', encoding="utf-8")
    (root / "q1" / "grid_2.jsonl").write_text('{"cell": "b", "replicate": 0}\n', encoding="utf-8")
    (root / "notes.txt").write_text("plain", encoding="utf-8")
    (root / "q1" / "junk.pyc").write_text("skip me", encoding="utf-8")


def test_freeze_tree_copies_hashes_and_counts_rows(tmp_path):
    make_tree(tmp_path / "src")
    files = freeze.freeze_tree(tmp_path / "src", tmp_path / "out", ["*.pyc", "__pycache__"], 7)
    assert sorted(files) == ["grid_1.jsonl", "notes.txt", "q1/grid_2.jsonl"]
    assert files["grid_1.jsonl"]["rows"] == 2 and files["q1/grid_2.jsonl"]["rows"] == 1 and "rows" not in files["notes.txt"]
    for rel, entry in files.items():
        assert freeze.file_sha256(tmp_path / "out" / rel, 5) == entry["sha256"]
    assert not (tmp_path / "out" / "q1" / "junk.pyc").exists()


def test_freeze_tree_refuses_a_non_empty_output_and_a_missing_source(tmp_path):
    make_tree(tmp_path / "src")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        freeze.freeze_tree(tmp_path / "src", tmp_path / "out", [], 8)
    with pytest.raises(FileNotFoundError):
        freeze.freeze_tree(tmp_path / "missing", tmp_path / "out2", [], 8)
    assert (tmp_path / "out" / "x.txt").read_text(encoding="utf-8") == "old"


def test_tree_hash_changes_with_content_and_not_with_order(tmp_path):
    a = {"x": {"sha256": "1"}, "y": {"sha256": "2"}}
    b = {"y": {"sha256": "2"}, "x": {"sha256": "1"}}
    c = {"x": {"sha256": "1"}, "y": {"sha256": "3"}}
    assert freeze.tree_sha256(a) == freeze.tree_sha256(b) != freeze.tree_sha256(c)


def test_read_only_and_last_amendment_id(tmp_path):
    make_tree(tmp_path / "t")
    freeze.make_read_only(tmp_path / "t")
    mode = (tmp_path / "t" / "grid_1.jsonl").stat().st_mode
    assert not mode & stat.S_IWUSR
    for path in sorted((tmp_path / "t").rglob("*")) + [tmp_path / "t"]:
        os.chmod(path, path.stat().st_mode | stat.S_IWUSR)
    text = "| A9 | x |\n| A10 | y |\n| A2 | z |\nnot a row\n"
    assert freeze.last_amendment_id(text) == 10 and freeze.last_amendment_id("") == 0


def test_env_section_has_marker_and_both_freezes():
    text = freeze.env_section("2026-01-01T00:00:00+00:00", "3.14.3", "Windows", 16, "numpy==2\npandas==3\n", "")
    assert text.startswith(freeze.ENV_MARKER) and "numpy==2" in text and "not available at freeze time" in text
    assert json.dumps(text)
