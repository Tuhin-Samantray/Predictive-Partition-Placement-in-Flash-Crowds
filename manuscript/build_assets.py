#!/usr/bin/env python
"""Build manuscript assets (LaTeX and Markdown tables, figure floats) from the Project A repo.

Reads only. Writes only below <root>/manuscript. Never edits figures/, tables/, results_frozen/ or logs/.
Every table cell is copied from a recorded CSV and only formatted; no number is computed or typed here.
"""
import argparse
import csv
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
INTEGER = re.compile(r"^[+-]?\d+$")
MISSING = {"", "nan", "NaN", "None", "NA", "N/A", "<NA>"}
WATCHED = ("figures", "tables")
TEX_ESCAPES = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}
FIGURE_NAME = re.compile(r"^fa(\d+)([a-z]?)_", re.IGNORECASE)
TA_NAME = re.compile(r"^ta(\d+)_", re.IGNORECASE)


def tex_escape(text):
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in text)


def read_text(path):
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- preflight

def last_amendment(root):
    path = root / "AMENDMENTS.md"
    if not path.exists():
        return None
    ids = [int(x) for x in re.findall(r"\|\s*A(\d+)\s*\|", read_text(path))]
    return max(ids) if ids else None


def verify_hashes(root):
    """Every 'hash  path' line in logs/*.sha256 that points into figures, tables or configs."""
    expected = {}
    for log in sorted((root / "logs").glob("*.sha256")):
        for line in read_text(log).splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or not HEX64.match(parts[0]):
                continue
            rel = parts[1].strip().lstrip("*").replace("\\", "/")
            if rel.split("/")[0] not in WATCHED:
                continue
            expected.setdefault(rel, {}).setdefault(parts[0].lower(), []).append(log.name)
    bad, absent = [], []
    for rel, hashes in sorted(expected.items()):
        target = root / rel
        if not target.exists():
            absent.append(rel)
            continue
        actual = sha256_of(target)
        if actual not in hashes:
            bad.append((rel, sorted({n for names in hashes.values() for n in names})))
    return len(expected), bad, absent


# ---------------------------------------------------------------- formatting

def sig_text(value, digits):
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent)}", rf"${mantissa}\times10^{{{int(exponent)}}}$"
    return text, text


def format_cell(raw, header, digits, p_digits):
    """Return (plain, tex). Text stays text; numbers keep 4 significant digits (3 for p values and tiny values)."""
    text = raw.replace("\n", " ").strip()
    if text in MISSING:
        return "n/a", "n/a"
    hashlike = re.search(r"sha|hash", header, re.IGNORECASE) is not None
    if not hashlike and INTEGER.match(text):
        return text, text
    if not hashlike and NUMBER.match(text) and len(text) <= 24:
        value = float(text)
        if math.isnan(value) or math.isinf(value):
            return text, tex_escape(text)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value)), str(int(value))
        if abs(value) >= 1e4:
            return f"{value:.0f}", f"{value:.0f}"
        pcol = re.search(r"(^|_)(p|holm|pval)(_|$)|p_value|holm", header, re.IGNORECASE) is not None
        digits_here = p_digits if (pcol or (value != 0 and abs(value) < 1e-3)) else digits
        return sig_text(value, digits_here)
    if hashlike and len(text) > 12:
        return text[:12], r"\texttt{" + tex_escape(text[:12]) + "}"
    return text, tex_escape(text)


def header_tex(name):
    return tex_escape(name.replace("_", " "))


def is_numeric_column(values):
    seen = [v.strip() for v in values if v.strip() not in MISSING]
    return bool(seen) and all(NUMBER.match(v) for v in seen)


# ---------------------------------------------------------------- captions

def load_caption_overrides(root):
    path = root / "manuscript" / "captions.csv"
    table = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("id") and row.get("caption"):
                    table[row["id"].strip().lower()] = row["caption"].strip()
    return table


def load_index_captions(root):
    path = root / "figures" / "index.csv"
    found = {}
    if not path.exists():
        return found
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return found
    header = rows[0]
    cap_cols = [i for i, h in enumerate(header) if re.search(r"caption|title|description", h, re.IGNORECASE)]
    if not cap_cols:
        return found
    for row in rows[1:]:
        text = next((row[i].strip() for i in cap_cols if i < len(row) and row[i].strip()), "")
        if not text:
            continue
        for cell in row:
            found.setdefault(cell.strip().lower(), text)
    return found


def load_column_choices(root):
    path = root / "manuscript" / "table_columns.csv"
    choice = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("id") and row.get("columns"):
                    choice[row["id"].strip().lower()] = [c.strip() for c in row["columns"].split(";") if c.strip()]
    return choice


# ---------------------------------------------------------------- tables

def read_table(path, wanted=None):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    header, body = rows[0], rows[1:]
    if wanted:
        missing = [c for c in wanted if c not in header]
        if missing:
            raise ValueError(f"{path.name}: columns {missing} not found; available: {header}")
        keep = [header.index(c) for c in wanted]
        header = [header[i] for i in keep]
        body = [[row[i] if i < len(row) else "" for i in keep] for row in body]
    width = len(header)
    body = [(row + [""] * width)[:width] for row in body]
    return header, body


def render_table(table_id, header, body, caption, args):
    formatted = []
    for row in body:
        formatted.append([format_cell(cell, header[i], args.digits, args.p_digits) for i, cell in enumerate(row)])
    columns = [[row[i] for row in body] for i in range(len(header))]
    align = "".join("r" if is_numeric_column(col) else "l" for col in columns)
    head = " & ".join(header_tex(h) for h in header) + r" \\"
    lines = [r"% generated by build_assets.py from the recorded CSV; do not edit"]
    label = f"tab:{table_id.lower()}"
    cap = rf"\caption{{{caption}}}\label{{{label}}}"
    if len(body) > args.long_rows:
        lines += [r"{\scriptsize", r"\setlength{\tabcolsep}{3pt}", rf"\begin{{longtable}}{{{align}}}",
                  cap + r"\\", r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule",
                  r"\endhead", r"\bottomrule", r"\endfoot"]
        lines += [" & ".join(c[1] for c in row) + r" \\" for row in formatted]
        lines += [r"\end{longtable}", "}"]
    else:
        lines += [r"\begin{table}[tbp]", r"\centering", r"\scriptsize", cap,
                  r"\adjustbox{max width=\linewidth}{%", rf"\begin{{tabular}}{{{align}}}", r"\toprule", head, r"\midrule"]
        lines += [" & ".join(c[1] for c in row) + r" \\" for row in formatted]
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    md = ["| " + " | ".join(h.replace("|", "\\|") for h in header) + " |",
          "| " + " | ".join("---" for _ in header) + " |"]
    md += ["| " + " | ".join(c[0].replace("|", "\\|") for c in row) + " |" for row in formatted]
    return "\n".join(lines) + "\n", "\n".join(md) + "\n"


def render_figure(stem, caption, prefix):
    return "\n".join([
        r"% generated by build_assets.py from the recorded PDF and caption; do not edit",
        r"\begin{figure}[tbp]", r"\centering",
        rf"\includegraphics[width=\linewidth]{{{prefix}/figures/{stem}.pdf}}",
        rf"\caption{{{caption}}}\label{{fig:{stem.lower()}}}", r"\end{figure}", ""])


# ---------------------------------------------------------------- discovery

def discover_tables(root, extra):
    found = {}
    for folder in (root / "figures", root / "tables"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            stem = path.stem
            match = TA_NAME.match(stem)
            if match:
                key = f"TA{int(match.group(1))}"
            elif stem.startswith("rt_") or stem.startswith("review_table") or stem in extra:
                key = stem
            else:
                continue
            found.setdefault(key, []).append(path)
    return found


def discover_figures(root):
    found = {}
    folder = root / "figures"
    for path in sorted(folder.glob("*.pdf")) if folder.exists() else []:
        match = FIGURE_NAME.match(path.stem)
        key = f"FA{int(match.group(1))}" if match else path.stem
        found.setdefault(key, []).append(path)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="manuscript/assets")
    parser.add_argument("--min-amendment", type=int, default=108)
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--not-completed", nargs="*", default=[])
    parser.add_argument("--undefined", nargs="*", default=[])
    parser.add_argument("--tex-prefix", default="assets")
    parser.add_argument("--long-rows", type=int, default=25)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--p-digits", type=int, default=3)
    parser.add_argument("--required-figures", default="2-15")
    parser.add_argument("--required-tables", default="1-9")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    if not str(out).startswith(str((root / "manuscript").resolve())):
        print("refusing: output folder must be inside <root>/manuscript")
        return 2

    problems = []
    last = last_amendment(root)
    print(f"last amendment id in AMENDMENTS.md: {last}")
    if last is None or last < args.min_amendment:
        problems.append(f"last amendment is {last}, expected at least A{args.min_amendment}: stop and tell the assistant")
    if not args.skip_hash_check:
        n_paths, bad, absent = verify_hashes(root)
        print(f"hash check: {n_paths} recorded paths under {', '.join(WATCHED)}; {len(bad)} changed, {len(absent)} absent")
        problems += [f"hash differs from every recorded value: {rel} (logs: {', '.join(logs)})" for rel, logs in bad]
        problems += [f"recorded but absent: {rel}" for rel in absent]
    for name in ("numbers.json", "macros.tex", "ledger.md"):
        if not (root / "figures" / name).exists():
            problems.append(f"figures/{name} is missing")
    if problems:
        print("PREFLIGHT FAILED, nothing written:")
        for line in problems:
            print("  - " + line)
        return 1

    tables = discover_tables(root, set(args.extra))
    figures = discover_figures(root)
    ambiguous = [k for k, v in tables.items() if len(v) > 1]
    if ambiguous:
        for key in ambiguous:
            print(f"ambiguous table id {key}: " + "; ".join(str(p.relative_to(root)) for p in tables[key]))
        print("stop: one source file per table id is required")
        return 1

    def span(text):
        lo, hi = (int(x) for x in text.split("-"))
        return range(lo, hi + 1)

    coverage = []
    for number in span(args.required_figures):
        key = f"FA{number}"
        files = [p.name for p in figures.get(key, [])]
        status = "ok" if files else ("not completed" if key in args.not_completed else
                                     ("undefined in the guide" if key in args.undefined else "MISSING"))
        coverage.append((key, "figure", status, ";".join(files)))
    for number in span(args.required_tables):
        key = f"TA{number}"
        files = [p.name for p in tables.get(key, [])]
        status = "ok" if files else ("not completed" if key in args.not_completed else "MISSING")
        coverage.append((key, "table", status, ";".join(files)))
    for key in args.undefined + args.not_completed:
        if not any(c[0] == key for c in coverage):
            coverage.append((key, "item", "not completed" if key in args.not_completed else "undefined in the guide", ""))
    for key in sorted(tables):
        if not key.startswith("TA"):
            coverage.append((key, "table", "ok", tables[key][0].name))
    for key in sorted(figures):
        if not re.fullmatch(r"FA\d+", key):
            coverage.append((key, "figure", "ok", ";".join(p.name for p in figures[key])))
    print("coverage:")
    for row in coverage:
        print("  {:<28} {:<7} {:<24} {}".format(*row))
    missing = [c for c in coverage if c[2] == "MISSING"]

    if args.check:
        print(f"check only: {len(missing)} missing, nothing written")
        return 1 if missing else 0

    overrides = load_caption_overrides(root)
    index_caps = load_index_captions(root)
    choices = load_column_choices(root)

    for sub in ("tables", "md", "figures"):
        target = out / sub
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    out.mkdir(parents=True, exist_ok=True)

    index_rows, warnings = [], []
    order = []
    ids = sorted(tables, key=lambda k: (0, int(k[2:])) if re.fullmatch(r"TA\d+", k) else (1, k))
    for key in ids:
        path = tables[key][0]
        header, body = read_table(path, choices.get(key.lower()))
        caption_raw, source = None, ""
        for label, table in (("captions.csv", overrides), ("figures/index.csv", index_caps)):
            hit = table.get(key.lower()) or table.get(path.stem.lower())
            if hit:
                caption_raw, source = hit, label
                break
        if caption_raw is None:
            caption_raw, source = path.name, "file name (no caption recorded)"
            warnings.append(f"{key}: no caption in manuscript/captions.csv or figures/index.csv; file name used")
        tex, md = render_table(key, header, body, tex_escape(caption_raw), args)
        (out / "tables" / f"{key.lower()}.tex").write_text(tex, encoding="utf-8", newline="\n")
        (out / "md" / f"{key.lower()}.md").write_text(md, encoding="utf-8", newline="\n")
        index_rows.append((key, "table", str(path.relative_to(root)).replace("\\", "/"), sha256_of(path),
                           f"tables/{key.lower()}.tex", source, len(body), len(header)))
        order.append(f"tables/{key.lower()}.tex")

    fig_keys = sorted(figures, key=lambda k: (0, int(k[2:])) if re.fullmatch(r"FA\d+", k) else (1, k))
    for key in fig_keys:
        for path in figures[key]:
            caption_file = path.with_suffix(".caption.txt")
            if caption_file.exists():
                caption_raw, source = " ".join(read_text(caption_file).split()), caption_file.name
            else:
                caption_raw, source = path.name, "file name (no caption file)"
                warnings.append(f"{path.stem}: no {caption_file.name}; file name used")
            shutil.copyfile(path, out / "figures" / path.name)
            (out / "figures" / f"{path.stem}.tex").write_text(
                render_figure(path.stem, tex_escape(caption_raw), args.tex_prefix), encoding="utf-8", newline="\n")
            index_rows.append((path.stem, "figure", str(path.relative_to(root)).replace("\\", "/"), sha256_of(path),
                               f"figures/{path.stem}.tex", source, "", ""))
            order.append(f"figures/{path.stem}.tex")

    for name in ("macros.tex", "numbers.json", "ledger.md"):
        shutil.copyfile(root / "figures" / name, out / name)

    with open(out / "assets_index.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "kind", "source", "source_sha256", "tex", "caption_source", "rows", "columns"])
        writer.writerows(index_rows)
    with open(out / "coverage.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "kind", "status", "files"])
        writer.writerows(coverage)
    quick = ["% generated by build_assets.py: quick-view appendix, \\input{" + args.tex_prefix + "/assets} once",
             "% needs: booktabs, longtable, adjustbox, graphicx"]
    quick += [rf"\input{{{args.tex_prefix}/{name}}}" for name in order]
    (out / "assets.tex").write_text("\n".join(quick) + "\n", encoding="utf-8", newline="\n")

    print(f"wrote {len(ids)} tables, {sum(len(v) for v in figures.values())} figures, index, coverage, quick-view file to {out}")
    for line in warnings:
        print("warning: " + line)
    if missing:
        print(f"{len(missing)} required items missing (see coverage above)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())