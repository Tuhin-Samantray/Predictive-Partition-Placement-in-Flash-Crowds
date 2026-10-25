"""Freeze the results (guide Phase 13.2).

Usage: python scripts/07_freeze.py --config configs/default.yaml --results results --out results_frozen
Copies the whole results tree into the (empty or missing) output folder, verifies every copy by SHA-256, writes
manifest.json (files, bytes, rows, tree hash, locked configs, tables and hash files, last amendment id), appends the
environment section to ENV.md, writes logs/frozen_results.sha256 and makes the frozen folder read-only.
Nothing is overwritten: a non-empty output folder or an ENV.md that already has the section stops the run.
Afterwards run: python scripts/snapshot.py frozen-v1
"""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import freeze  # noqa: E402


def pip_freeze(python):
    try:
        done = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True)
    except OSError:
        return ""
    return done.stdout if done.returncode == 0 else ""


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    chunk = cfg["compute"]["io_chunk_bytes"]
    snap = cfg["snapshot"]
    out = Path(args.out)
    env_path = Path("ENV.md")
    env_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if freeze.ENV_MARKER in env_text:
        print("ENV.md already has the frozen environment section; stop", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        files = freeze.freeze_tree(args.results, out, snap["ignore_patterns"], chunk)
    except (FileExistsError, FileNotFoundError) as error:
        print("stop: " + str(error), file=sys.stderr)
        return 2
    locked_paths = [Path(args.config), Path("configs/tuned.yaml"), Path("configs/tuned_ext.yaml"), Path("configs/tuned_layers.yaml"),
                    Path("requirements.txt"), Path("PREREG.md"), Path("PREREG.sha256")]
    locked_paths += sorted(Path("tables").glob("*.csv")) + sorted(Path("logs").glob("*.sha256"))
    amendments = Path("AMENDMENTS.md").read_text(encoding="utf-8") if Path("AMENDMENTS.md").exists() else ""
    folders = {}
    for name, folder in cfg["result_folders"].items():
        prefix = Path(folder).relative_to(Path(args.results)).as_posix() if Path(folder) != Path(args.results) else ""
        rows = sum(entry.get("rows", 0) for rel, entry in files.items()
                   if (Path(rel).parent.as_posix() == (prefix or ".")))
        folders[name] = rows
    manifest = {
        "created_utc": stamp,
        "python": sys.version.split()[0],
        "tree_sha256": freeze.tree_sha256(files),
        "result_folder_rows": folders,
        "last_amendment_id": freeze.last_amendment_id(amendments),
        "locked": freeze.hash_files(locked_paths, chunk),
        "files": files,
    }
    manifest_path = out / snap["manifest_name"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = freeze.file_sha256(manifest_path, chunk)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/frozen_results.sha256").write_text(
        "%s  %s tree (%d files, manifest.json lists them)\n%s  %s/%s\n" % (
            manifest["tree_sha256"], out.as_posix(), len(files), manifest_hash, out.as_posix(), snap["manifest_name"]),
        encoding="utf-8", newline="\n")
    backup = Path("logs/phase7_backup")
    backup.mkdir(parents=True, exist_ok=True)
    (backup / "ENV.md.before_freeze").write_text(env_text, encoding="utf-8", newline="\n")
    chronos_python = Path(".venv_chronos/Scripts/python.exe")
    section = freeze.env_section(stamp, sys.version.split()[0], platform.platform(), os.cpu_count(),
                                 pip_freeze(sys.executable), pip_freeze(chronos_python) if chronos_python.exists() else "")
    env_path.write_text(env_text.rstrip("\n") + "\n\n" + section, encoding="utf-8", newline="\n")
    freeze.make_read_only(out)
    print("frozen %d files into %s; tree sha256 %s; manifest sha256 %s" % (len(files), out, manifest["tree_sha256"], manifest_hash))
    print("result folder rows: " + json.dumps(folders))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
