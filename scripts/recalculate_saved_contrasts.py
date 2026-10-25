#!/usr/bin/env python3
"""Recalculate all 64 factorial point effects from saved per-example scores.

This audit is independent of the original Phase 6 script. It pairs the
segmentation and bigram tables by document and model identity, reconstructs
character-normalised domain metrics, averages the five seeds, and checks
all published 2 by 2 factorial contrasts. It does not retrain models.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "coursework" / "isiXhosa_Excellence_Sprint"
ARMS = ("PP", "PW", "WP", "WW")
FAMILIES = ("bpe", "unigram")
PERIODS = ("1888", "1889-94")
VIEWS = ("en", "xh")
SEEDS = (441, 442, 443, 444, 445)
KEYS = ("family", "arm", "seed", "filename", "view", "issue", "year", "period")


def read_csv(name: str):
    with (ROOT / name).open(newline="", encoding="utf-8") as file:
        yield from csv.DictReader(file)


def key(record: dict[str, str]) -> tuple[str, ...]:
    return tuple(record[col] for col in KEYS)


def contrast(values: dict[str, float]) -> dict[str, float]:
    return {
        "allocation": 0.5 * ((values["WP"] - values["PP"]) + (values["WW"] - values["PW"])),
        "within_view": 0.5 * ((values["PW"] - values["PP"]) + (values["WW"] - values["WP"])),
        "interaction": (values["WW"] - values["WP"]) - (values["PW"] - values["PP"]),
        "WW_minus_PP": values["WW"] - values["PP"],
    }


def recalculate():
    by_key = {}
    pageviews = set()
    issues = {period: set() for period in PERIODS}
    rows3 = 0
    for row in read_csv("Phase3/outputs/heldout_scores.csv"):
        row_key = key(row)
        if row_key in by_key:
            raise AssertionError("Duplicate segmentation row: " + repr(row_key))
        by_key[row_key] = row
        pageviews.add((row["filename"], row["view"]))
        issues[row["period"]].add(row["issue"])
        rows3 += 1
    if rows3 != 99720 or len(pageviews) != 2493:
        raise AssertionError(f"Unexpected segmentation support: {rows3}, {len(pageviews)}")
    if {p: len(v) for p, v in issues.items()} != {"1888": 52, "1889-94": 306}:
        raise AssertionError("Wrong historical issue support")

    sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    observed = set()
    rows5 = 0
    for row in read_csv("Phase5/outputs/heldout_lm_scores.csv"):
        row_key = key(row)
        if row_key not in by_key or row_key in observed:
            raise AssertionError("Unmatched or duplicate LM scoring row")
        observed.add(row_key)
        prior = by_key[row_key]
        for field in ("words", "chars", "text_sha256", "pieces", "unknown"):
            if prior[field] != row[field]:
                raise AssertionError(f"Phase 3/5 disagreement in {field}: {row_key}")
        chars = int(row["chars"])
        pieces = int(row["pieces"])
        bits = float(row["bits"])
        if chars <= 0 or pieces <= 0 or int(row["unknown"]) != 0 or not math.isfinite(bits):
            raise AssertionError(f"Invalid score in {row_key}")
        if not math.isclose(bits / chars, float(row["bpc"]), abs_tol=1e-12, rel_tol=0):
            raise AssertionError(f"Bits/character mismatch: {row_key}")
        bucket = sums[(row["family"], row["period"], row["view"], row["arm"], int(row["seed"]))]
        bucket[0] += chars
        bucket[1] += pieces
        bucket[2] += bits
        rows5 += 1
    if rows5 != rows3 or len(observed) != rows3:
        raise AssertionError("Incomplete paired score evidence")
    if len(sums) != 160:
        raise AssertionError(f"Incomplete 40-model, two-period, two-view cells: {len(sums)}")

    written = {}
    for phase, name in (
        ("segmentation", "Phase3/outputs/factorial_contrasts.csv"),
        ("language_model", "Phase5/outputs/bpc_factorial_contrasts.csv"),
    ):
        for row in read_csv(name):
            idx = (phase, row["family"], row["period"], row["view"], row["contrast"])
            if idx in written:
                raise AssertionError("Duplicate reported contrast")
            written[idx] = float(row["mean"])
    if len(written) != 64:
        raise AssertionError(f"Expected 64 contrasts, found {len(written)}")

    measured = {}
    for family in FAMILIES:
        for period in PERIODS:
            for view in VIEWS:
                for phase in ("segmentation", "language_model"):
                    effects = []
                    for seed in SEEDS:
                        by_arm = {}
                        for arm in ARMS:
                            chars, pieces, bits = sums[(family, period, view, arm, seed)]
                            by_arm[arm] = 100.0 * pieces / chars if phase == "segmentation" else bits / chars
                        effects.append(contrast(by_arm))
                    for label in ("allocation", "within_view", "interaction", "WW_minus_PP"):
                        idx = (phase, family, period, view, label)
                        effect = sum(each[label] for each in effects) / len(SEEDS)
                        if not math.isclose(effect, written[idx], abs_tol=1e-10, rel_tol=0.0):
                            raise AssertionError(f"Contrast mismatch at {idx}: {effect} vs {written[idx]}")
                        measured["|".join(idx)] = effect

    return {
        "segmentation_rows": rows3,
        "lm_rows": rows5,
        "pageviews": len(pageviews),
        "issues": {x: len(y) for x, y in issues.items()},
        "effects": measured,
        "max_difference": max(abs(v - written[tuple(k.split("|"))]) for k, v in measured.items()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-out", type=Path, help="Optional machine-readable audit summary")
    a = p.parse_args()
    result = recalculate()
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS: {result['segmentation_rows']} paired segmentation rows and "
        f"{result['lm_rows']} LM rows, {result['pageviews']} excerpts, "
        f"{len(result['effects'])} independently recalculated point effects"
    )
    print(f"Maximum absolute difference: {result['max_difference']:.3g}")


if __name__ == "__main__":
    main()
