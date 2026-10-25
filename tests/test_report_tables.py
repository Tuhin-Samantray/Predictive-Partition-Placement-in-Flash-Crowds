from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.report_tables import build_ta1, build_ta2, parse_config_tags, parse_data_log, tag_of

ROOT = Path(__file__).parent.parent

YAML = """layers:
  enabled: false # [design]
n_partitions: 512 # [real]
headroom: 1.6 # [assumed] why it is assumed
horizons: [1, 2, 5, 10] # [design]
key_sep: '|' # [design]
pattern: 'x.d{day:02d}.csv' # [real]
q2:  # free text without a tag
  D1: P3_naive # [design] 14.9 against P4
blk: # [design] header with a tag
  a: 'u' # [design]
"""

LOG = """## D1: Azure Trace
- Page: https://example.org/a.md
- Licence: CC-BY page.
- Citation (dataset page): Author A. Title. Venue 2020.
- D1 licence confirmed 2026-10-23: LICENSE opens with Attribution.
## D2: Twitter
- Repository: https://example.org/t (README read)
- Licence: CC BY 4.0. Full text read.
- Citation (README): Yang, Yue. Title. OSDI 2020.
## Downloaded
- D1 archive: a.tar.xz, 100 bytes, SHA-256 abc123
- D2 file: c.zst, 200 bytes, SHA-256 def456
## Phase 3
- D1: 77 invocation-file rows over days 1 to 7
- D2 c.zst: 88 rows; first timestamp 0
"""


def test_parse_config_tags_planted(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML, encoding="utf-8")
    rows = parse_config_tags(p)
    assert list(rows["key"]) == ["enabled", "n_partitions", "headroom", "horizons", "key_sep", "pattern", "D1", "a"]
    by = rows.set_index("key")
    assert by.loc["enabled", "parent"] == "layers"
    assert by.loc["headroom", "real_or_assumed"] == "assumed"
    assert by.loc["headroom", "justification"] == "why it is assumed"
    assert by.loc["horizons", "value"] == "[1, 2, 5, 10]"
    assert by.loc["key_sep", "value"] == "'|'"
    assert by.loc["pattern", "value"] == "'x.d{day:02d}.csv'"
    assert by.loc["D1", "parent"] == "q2" and by.loc["a", "parent"] == "blk"
    assert len(build_ta2(p)) == 8


def test_missing_tag_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1 # [design]\nb: 2 # no tag here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_config_tags(p)


def test_duplicate_or_mismatch_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1 # [design]\na: 2 # [design]\n", encoding="utf-8")
    with pytest.raises(Exception):
        build_ta2(p)


def test_real_config_is_fully_tagged():
    ta2 = build_ta2(ROOT / "configs" / "default.yaml")
    assert set(ta2["real_or_assumed"]) <= {"real", "assumed", "design"}
    assert len(ta2) > 100
    assert tag_of(ta2, "n_partitions") == "real"


def test_parse_data_log_planted(tmp_path):
    p = tmp_path / "L.md"
    p.write_text(LOG, encoding="utf-8")
    f = parse_data_log(p)
    assert f["D1"]["raw_bytes"] == 100 and f["D1"]["raw_sha256"] == "abc123" and f["D1"]["raw_rows"] == 77
    assert f["D2"]["raw_rows"] == 88 and f["D2"]["source_url"] == "https://example.org/t"
    assert f["D1"]["licence_check"].startswith("LICENSE opens") and f["D2"]["licence_check"] == f["D2"]["licence"]
    p.write_text(LOG.replace("- Licence: CC BY 4.0. Full text read.\n", ""), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_data_log(p)


def test_real_data_log_parses():
    f = parse_data_log(ROOT / "DATA_LOG.md")
    assert f["D1"]["raw_bytes"] == 142968140 and f["D2"]["raw_rows"] == 246508262


def test_build_ta1_planted(tmp_path):
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    for ds, n in (("D1", 100), ("D2", 50), ("D1_D3", 100), ("D2_D3", 50)):
        np.save(d / f"{ds}_L.npy", np.zeros((n, 3), dtype="float32"))
    def ev(rows):
        return pd.DataFrame(rows, columns=["event_id", "onset", "split"])
    ev([(0, "abrupt", "train"), (1, "ramped", "train"), (2, "abrupt", "eval"), (3, "abrupt", "eval")]).to_csv(d / "D1_events.csv", index=False)
    ev([(0, "ramped", "eval")]).to_csv(d / "D2_events.csv", index=False)
    ev([(0, "abrupt", "eval"), (0, "abrupt", "eval"), (1, "abrupt", "eval")]).to_csv(d / "D1_D3_events.csv", index=False)
    ev([(0, "ramped", "eval")]).to_csv(d / "D2_D3_events.csv", index=False)
    log = tmp_path / "L.md"
    log.write_text(LOG, encoding="utf-8")
    cfg = {"data_root": str(tmp_path / "data"), "processed_dirname": "processed", "day_seconds": 86400}
    ta2 = pd.DataFrame({"key": ["d1_bytes_sigma", "d2_vsize_col"], "parent": ["", ""], "value": ["0.5", "'v'"],
                        "real_or_assumed": ["assumed", "real"], "justification": ["", ""]})
    ta1 = build_ta1(cfg, log, ta2, split_fn=lambda c, ds, n: (n // 2, n * 3 // 4), step_fn=lambda c, ds: 60 if ds == "D1" else 10)
    r = ta1.set_index("dataset")
    assert list(ta1["dataset"]) == ["D1", "D2", "D1_D3", "D2_D3"]
    assert not [c for c in ta1.columns if c.startswith("events_") and any(ch.isdigit() for ch in c)]
    assert r.loc["D1", "n_steps"] == 100 and r.loc["D1", "train_end"] == 50 and r.loc["D1", "tune_end"] == 75
    assert r.loc["D1", "eval_steps"] == 25 and r.loc["D1", "n_partitions"] == 3
    assert r.loc["D1", "events_train_abrupt"] == 1 and r.loc["D1", "events_eval_abrupt"] == 2 and r.loc["D1", "n_events"] == 4
    assert r.loc["D1_D3", "n_event_rows"] == 3 and r.loc["D1_D3", "n_events"] == 2
    assert r.loc["D1_D3", "events_eval_abrupt"] == 2 and r.loc["D1_D3", "events_train_abrupt"] == 0
    assert r.loc["D2", "events_eval_ramped"] == 1 and r.loc["D1", "events_eval_ramped"] == 0
    assert abs(r.loc["D2", "span_days"] - 50 * 10 / 86400) < 1e-12
    assert r.loc["D1", "raw_bytes"] == 100 and r.loc["D2", "raw_rows"] == 88
    assert r.loc["D1", "bytes_provenance"] == "assumed" and r.loc["D2_D3", "bytes_provenance"] == "real"
    assert pd.isna(r.loc["D1_D3", "raw_bytes"]) and r.loc["D2_D3", "licence"].startswith("derived from D2")


def test_numeric_onset_is_not_a_class(tmp_path):
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    for ds, n in (("D1", 100), ("D2", 50), ("D1_D3", 100), ("D2_D3", 50)):
        np.save(d / f"{ds}_L.npy", np.zeros((n, 3), dtype="float32"))
    cls = pd.DataFrame([(0, "abrupt", "eval"), (1, "ramped", "eval")], columns=["event_id", "onset", "split"])
    num = pd.DataFrame([(0, 4325, "eval"), (0, 4325, "eval"), (1, 9180, "eval"), (2, 9180, "eval")], columns=["event_id", "onset", "split"])
    cls.to_csv(d / "D1_events.csv", index=False)
    cls.to_csv(d / "D2_events.csv", index=False)
    num.to_csv(d / "D1_D3_events.csv", index=False)
    num.to_csv(d / "D2_D3_events.csv", index=False)
    log = tmp_path / "L.md"
    log.write_text(LOG, encoding="utf-8")
    cfg = {"data_root": str(tmp_path / "data"), "processed_dirname": "processed", "day_seconds": 86400}
    ta2 = pd.DataFrame({"key": ["d1_bytes_sigma", "d2_vsize_col"], "parent": ["", ""], "value": ["0.5", "'v'"],
                        "real_or_assumed": ["assumed", "real"], "justification": ["", ""]})
    ta1 = build_ta1(cfg, log, ta2, split_fn=lambda c, ds, n: (n // 2, n * 3 // 4), step_fn=lambda c, ds: 60)
    r = ta1.set_index("dataset")
    assert r.loc["D1_D3", "events_eval_all"] == 3 and r.loc["D1_D3", "n_event_rows"] == 4
    assert r.loc["D1", "events_eval_abrupt"] == 1 and r.loc["D1", "events_eval_ramped"] == 1 and r.loc["D1", "events_eval_all"] == 0
    assert not [c for c in ta1.columns if c.startswith("events_") and any(ch.isdigit() for ch in c)]
