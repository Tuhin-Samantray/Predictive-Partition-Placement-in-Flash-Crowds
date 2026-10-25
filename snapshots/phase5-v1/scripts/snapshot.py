"""Create a frozen, read-only snapshot of src/, scripts/ and configs/ (guide rule G4).

Usage: python scripts/snapshot.py <tag> [--config configs/default.yaml]
Writes snapshots/<tag>/ with a per-file SHA-256 manifest.json, then makes the copy read-only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle)


def file_sha256(path, chunk_bytes):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(dest, tag, chunk_bytes):
    files = {}
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        files[path.relative_to(dest).as_posix()] = {
            "sha256": file_sha256(path, chunk_bytes),
            "bytes": path.stat().st_size,
        }
    tree = hashlib.sha256()
    for rel in sorted(files):
        tree.update(f"{rel}\t{files[rel]['sha256']}\n".encode("utf-8"))
    return {
        "tag": tag,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "tree_sha256": tree.hexdigest(),
        "files": files,
    }


def make_read_only(root):
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    entries = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for entry in entries + [root]:
        os.chmod(entry, entry.stat().st_mode & ~write_bits)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    snap = cfg["snapshot"]
    if not re.fullmatch(snap["tag_pattern"], args.tag):
        print(f"invalid tag {args.tag!r}; allowed pattern {snap['tag_pattern']}", file=sys.stderr)
        return 2
    dest = Path(cfg["paths"]["snapshots"]) / args.tag
    if dest.exists():
        print(f"{dest} already exists; snapshots are immutable", file=sys.stderr)
        return 2
    missing = [name for name in snap["sources"] if not Path(name).is_dir()]
    if missing:
        print(f"missing source directories: {missing}", file=sys.stderr)
        return 2
    ignore = shutil.ignore_patterns(*snap["ignore_patterns"])
    for name in snap["sources"]:
        shutil.copytree(Path(name), dest / name, ignore=ignore)
    manifest = build_manifest(dest, args.tag, cfg["compute"]["io_chunk_bytes"])
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    (dest / snap["manifest_name"]).write_text(manifest_text, encoding="utf-8")
    make_read_only(dest)
    print(f"snapshot {args.tag}: {len(manifest['files'])} files, tree sha256 {manifest['tree_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

