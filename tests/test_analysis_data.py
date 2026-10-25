import json

import pandas as pd
import pytest

from src import analysis_data as ad

KEYS = ["dataset", "onset_class", "m", "headroom", "replicate"]


def write(folder, rows, name="grid_1.jsonl"):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def row(variant, rep, damage, event="e", dataset="D1", phi=None):
    return {"cell": "%s-abrupt-m5-h1.6-%s" % (dataset, variant), "replicate": rep, "dataset": dataset,
            "onset_class": "abrupt", "m": 5, "headroom": 1.6, "variant": variant, "event_id": "%s%d" % (event, rep),
            "damage": damage, "phi": phi}


def test_load_folder_reads_all_files_sorts_and_keeps_nulls(tmp_path):
    write(tmp_path, [row("P1", 1, 2.0), row("P1", 0, 1.0)], "grid_1.jsonl")
    write(tmp_path, [row("P5", 0, 0.5, phi=1.0)], "grid_2.jsonl")
    frame = ad.load_folder(tmp_path)
    assert len(frame) == 3 and frame["replicate"].tolist() == [0, 1, 0]
    assert frame["cell"].is_monotonic_increasing and frame["phi"].isna().sum() == 2


def test_load_folder_rejects_empty_folder_and_duplicate_keys(tmp_path):
    with pytest.raises(FileNotFoundError):
        ad.load_folder(tmp_path)
    write(tmp_path, [row("P1", 0, 1.0), row("P1", 0, 1.0)])
    with pytest.raises(ValueError):
        ad.load_folder(tmp_path)


def test_load_queues_adds_the_source_column(tmp_path):
    write(tmp_path / "a", [row("P1", 0, 1.0)])
    write(tmp_path / "b", [row("P2", 0, 2.0)])
    out = ad.load_queues({"qa": tmp_path / "a", "qb": tmp_path / "b"})
    assert out["qa"]["source"].tolist() == ["qa"] and out["qb"]["source"].tolist() == ["qb"]


def test_select_variant_picks_the_variant_named_per_dataset():
    frame = pd.DataFrame([row("P1", 0, 1.0), row("P2", 0, 2.0), row("P1", 0, 3.0, dataset="D2"),
                          row("P2", 0, 4.0, dataset="D2")])
    out = ad.select_variant(frame, {"D1": "P1", "D2": "P2"})
    assert sorted(zip(out["dataset"], out["variant"], out["damage"])) == [("D1", "P1", 1.0), ("D2", "P2", 4.0)]


def test_pair_rows_joins_one_to_one_with_the_shared_event_id():
    a = pd.DataFrame([row("P1", 1, 2.0), row("P1", 0, 1.0)])
    b = pd.DataFrame([row("P5", 0, 0.5), row("P5", 1, 1.5)])
    out = ad.pair_rows(a, b, KEYS, "damage")
    assert out["damage_a"].tolist() == [1.0, 2.0] and out["damage_b"].tolist() == [0.5, 1.5]
    assert out["event_id"].tolist() == ["e0", "e1"] and "event_id_b" not in out.columns


def test_pair_rows_fails_loudly_on_duplicates_missing_partners_and_event_mismatch():
    a = pd.DataFrame([row("P1", 0, 1.0), row("P1", 1, 2.0)])
    with pytest.raises(ValueError):
        ad.pair_rows(pd.concat([a, a]), a, KEYS, "damage")
    with pytest.raises(ValueError):
        ad.pair_rows(a, a.iloc[:1], KEYS, "damage")
    other = pd.DataFrame([row("P5", 0, 0.5, event="x"), row("P5", 1, 1.5, event="x")])
    with pytest.raises(ValueError):
        ad.pair_rows(a, other, KEYS, "damage")


def test_event_frame_averages_replicates_of_the_same_event():
    frame = pd.DataFrame({"dataset": ["D3", "D3", "D3"], "event_id": ["e1", "e1", "e2"],
                          "damage_a": [2.0, 4.0, 5.0], "damage_b": [1.0, 1.0, 3.0]})
    out = ad.event_frame(frame, ["dataset", "event_id"], ["damage_a", "damage_b"])
    assert out["damage_a"].tolist() == [3.0, 5.0] and out["damage_b"].tolist() == [1.0, 3.0]
