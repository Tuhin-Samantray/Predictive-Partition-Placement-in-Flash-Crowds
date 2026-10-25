import importlib.util
import inspect
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_grid", Path(__file__).resolve().parents[1] / "scripts" / "04_run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def test_timing_key_drops_the_label_suffix():
    lines = [line for line in inspect.getsource(rg.run_task).splitlines() if "timing.append" in line]
    assert len(lines) == 1
    assert 'variant.split("+", 1)[0]' in lines[0]
    assert "variant.split" in inspect.getsource(rg.dry_run) or 'v.split("+", 1)[0]' in inspect.getsource(rg.dry_run)
