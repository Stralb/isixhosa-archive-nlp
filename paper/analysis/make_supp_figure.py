#!/usr/bin/env python3
"""Reproduce the supplementary page-position figure from the original archive.

From repository root:
  cd paper
  python analysis/make_supp_figure.py --archive ../data/TxtLangFiles1870-1894_TB.zip --outdir .

Creates paper/supp_page_position_compact.pdf and
paper/analysis/supp_page_position_summary.csv.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_SHA256 = "375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5"
FILE_RE = re.compile(r"/XhTxt/(IMVO|ISIG)_(1885|1888)-\d\d-\d\d_p(\d\d\d)\.txt$")
WORD_RE = re.compile(r"[a-z]+")


def load_source(archive: Path):
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != EXPECTED_SHA256:
        raise ValueError("Source archive checksum mismatch")

    counts = defaultdict(lambda: [0, 0, 0])
    with ZipFile(archive) as z:
        for filename in z.namelist():
            m = FILE_RE.search(filename)
            if not m or "/._" in filename:
                continue
            source, year, page = m.groups()
            words = WORD_RE.findall(z.read(filename).decode("utf-8", errors="replace").lower())
            positions = sum(max(0, len(word) - 3) for word in words)
            if len(words) < 40 or positions == 0:
                continue
            key = (source, int(year), min(int(page), 5))
            row = counts[key]
            row[0] += 1
            row[1] += len(words)
            row[2] += positions

    rows = []
    for (source, year, position), (pages, words, positions) in sorted(counts.items()):
        rows.append(dict(
            source=source, year=year,
            position=str(position) if position < 5 else "5+",
            pages=pages, eligible_words=words,
            eligible_positions=positions,
            mean_positions_per_page=round(positions/pages, 2)))
    firsts = [(r["pages"], r["eligible_positions"]) for r in rows
              if r["source"] == "IMVO" and r["year"] == 1885 and r["position"] in ("1", "2")]
    if firsts != [(49, 97071), (49, 411492)]:
        raise AssertionError("Source exposure sanity check failed")
    return rows


def plot(rows, outdir: Path):
    def ratio(source, year, position):
        values = {r["position"]: r["mean_positions_per_page"] for r in rows
                  if r["source"] == source and r["year"] == year}
        return values[str(position)]/values["1"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.labelsize": 8.7, "legend.fontsize": 7.6,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(3.38, 2.54))
    for src, color, name in [("IMVO", "#253f57", "Imvo"),
                             ("ISIG", "#a44f31", "Isigidimi")]:
        for year, style, marker in [(1885, "-", "o"), (1888, "--", "s")]:
            ax.plot([1, 2, 3, 4], [ratio(src, year, p) for p in (1, 2, 3, 4)],
                    color=color, linestyle=style, marker=marker, markersize=4,
                    linewidth=1.6, label=f"{name} {year}")
    ax.set(xlim=(.94, 4.06), ylim=(.67, 4.55), xticks=[1, 2, 3, 4],
           xlabel="Newspaper page position", ylabel="Ratio to position 1")
    ax.grid(axis="y", alpha=.17, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.015),
              frameon=False, columnspacing=.9, handlelength=1.55, borderaxespad=.1)
    fig.subplots_adjust(left=.17, right=.985, bottom=.23, top=.78)
    fig.savefig(outdir/"supp_page_position_compact.pdf",
                bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=Path("../data/TxtLangFiles1870-1894_TB.zip"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = load_source(args.archive)
    summary = args.outdir/"analysis"/"supp_page_position_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.outdir)
    print("PASS: input ZIP verified; supplementary figure reproduced")


if __name__ == "__main__":
    main()
