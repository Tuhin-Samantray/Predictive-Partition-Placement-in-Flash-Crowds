"""Freeze helpers for Phase 13 (A83): copy a results tree, hash every file, lock files, write the environment section.

Only the numbers 0, 1 and 2 appear here (guide rule G1); sizes come from the config.
"""
import hashlib
import os
import re
import stat
from pathlib import Path

ENV_MARKER = "## Frozen environment"


def file_sha256(path, chunk_bytes):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path):
    """Number of non-empty lines of a jsonl file."""
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def tree_sha256(files):
    """Tree hash over (relative path, sha256) pairs, the same rule as scripts/snapshot.py."""
    tree = hashlib.sha256()
    for rel in sorted(files):
        tree.update(f"{rel}\t{files[rel]['sha256']}\n".encode("utf-8"))
    return tree.hexdigest()


def is_ignored(path, patterns):
    return any(Path(path).match(pattern) or any(Path(part).match(pattern) for part in Path(path).parts) for pattern in patterns)


def freeze_tree(src, dst, ignore_patterns, chunk_bytes):
    """Copy every file under src to dst (dst must be missing or empty), verify each copy by hash and return the file table."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(str(src))
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(str(dst) + " is not empty; a frozen folder is written once")
    dst.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = path.relative_to(src)
        if is_ignored(rel, ignore_patterns):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        before, after = file_sha256(path, chunk_bytes), file_sha256(target, chunk_bytes)
        if before != after:
            raise RuntimeError("copy differs from source: " + rel.as_posix())
        entry = {"sha256": after, "bytes": target.stat().st_size}
        if target.suffix == ".jsonl":
            entry["rows"] = count_rows(target)
        files[rel.as_posix()] = entry
    return files


def hash_files(paths, chunk_bytes):
    out = {}
    for path in paths:
        path = Path(path)
        if path.is_file():
            out[path.as_posix()] = file_sha256(path, chunk_bytes)
    return out


def last_amendment_id(text):
    ids = [int(m) for m in re.findall(r"(?m)^\| A(\d+) \|", text)]
    return max(ids) if ids else 0


def make_read_only(root):
    """Remove every write bit under root (files first, then folders)."""
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    root = Path(root)
    entries = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for entry in entries + [root]:
        os.chmod(entry, entry.stat().st_mode & ~write_bits)


def env_section(stamp, python_version, platform_text, cores, freeze_main, freeze_chronos):
    lines = [ENV_MARKER + " (Phase 13, written by scripts/07_freeze.py at " + stamp + ")", "",
             "- Python (main environment): " + python_version,
             "- Platform: " + platform_text,
             "- Logical processors seen by Python: " + str(cores),
             "", "### pip freeze, main environment (.venv)", "", "```"]
    lines += freeze_main.strip().splitlines()
    lines += ["```", "", "### pip freeze, Chronos environment (.venv_chronos)", "", "```"]
    lines += (freeze_chronos.strip().splitlines() if freeze_chronos.strip() else ["not available at freeze time"])
    lines += ["```", ""]
    return "\n".join(lines)
