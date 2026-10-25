"""G1: configs/default.yaml loads and every value line has a justification comment with a tag."""
import re
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
HEADER = re.compile(r"^\s*[\w\-]+:\s*(#.*)?$")
TAG = re.compile(r"\[(real|assumed|design)\]")


def test_config_loads():
    with open(CONFIG, "r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    assert data["layers"]["enabled"] is False


def test_every_value_line_is_tagged():
    missing = []
    text = CONFIG.read_text(encoding="utf-8-sig")
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or HEADER.match(raw):
            continue
        comment = raw.split(" #", 1)[1] if " #" in raw else ""
        if not TAG.search(comment):
            missing.append(f"line {number}: {line}")
    assert not missing, "value lines without a [real|assumed|design] tag:\n" + "\n".join(missing)
