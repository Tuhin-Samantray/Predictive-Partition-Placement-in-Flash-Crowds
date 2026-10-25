
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CONFIG = "configs/default.yaml"
TUNED = "configs/tuned.yaml"
DATASETS = ("D1", "D2")

TARGETS = {
    "test": [[PY, "-m", "pytest", "-q"]],
    "smoke": [[PY, "-m", "pytest", "-q", "tests/test_smoke.py"]],
    "data": [
        [PY, "scripts/01_prepare.py", "--config", CONFIG, "--dataset", ds, "--stage", "all"]
        for ds in DATASETS
    ],
    "forecast": [
        [PY, "scripts/02_forecast.py", "--config", CONFIG, "--dataset", ds, "--forecaster", "all"]
        for ds in DATASETS
    ],
    "tune": [
        [PY, "scripts/03_tune.py", "--config", CONFIG, "--policy", "all",
         "--split", "tune", "--out", TUNED]
    ],
    "grid": [
        [PY, "scripts/04_run_grid.py", "--config", CONFIG, "--tuned", TUNED,
         "--queue", "all", "--out", "results"]
    ],
    "analyze": [
        [PY, "scripts/05_analyze.py", "--config", CONFIG, "--results", "results",
         "--out", "tables", "--stage", "all"]
    ],
    "freeze": [
        [PY, "scripts/07_freeze.py", "--config", CONFIG, "--results", "results",
         "--out", "results_frozen"]
    ],
    "report": [
        [PY, "scripts/06_report.py", "--config", CONFIG, "--results", "results_frozen",
         "--out", "figures"]
    ],
}

ORDER_ALL = ("test", "data", "forecast", "tune", "grid", "analyze", "freeze", "report")


def run_target(name):
    for cmd in TARGETS[name]:
        script = next((a for a in cmd if a.endswith(".py")), None)
        if script and not (ROOT / script).exists():
            print(f"[{name}] missing {script}: its phase has not been executed yet",
                  file=sys.stderr)
            return 2
        print("+", " ".join(cmd), flush=True)
        code = subprocess.run(cmd, cwd=ROOT).returncode
        if code:
            return code
    return 0


def main(argv):
    valid = sorted(TARGETS) + ["all"]
    if len(argv) != 2 or argv[1] not in valid:
        print("usage: python tasks.py <" + "|".join(valid) + ">", file=sys.stderr)
        return 2
    names = ORDER_ALL if argv[1] == "all" else (argv[1],)
    for name in names:
        code = run_target(name)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
