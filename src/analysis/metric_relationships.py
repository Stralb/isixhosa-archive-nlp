#!/usr/bin/env python3
"""Exploratory paired-metric and length-heterogeneity analysis for NLP Assignment 2.

This is a POST-HOC SECONDARY analysis. No models are retrained, no hypotheses
are retroactively amended, and all original score files remain unchanged.

Run (from the repository root):
  python src/analysis/metric_relationships.py --root .
  python src/analysis/metric_relationships.py --root . --verify

Primary published contrasts remain issue/character weighted across evaluation
rows, with five fixed trained seeds. For subgroup estimates, source text length
quartiles are defined within period and archive view BEFORE viewing outcomes;
resampling publication issues preserves page clustering and seed pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("PP", "PW", "WP", "WW")
SEEDS = (441, 442, 443, 444, 445)
QUARTILES = ("Q1", "Q2", "Q3", "Q4")
N_BOOT = 5000
BOOT_SEED = 772026
INPUTS = (
    "results/raw/segmentation.csv",
    "results/raw/bigram.csv",
    "results/segmentation.csv",
    "results/bigram.csv",
)
OUTPUTS = (
    "issue_allocation_effects.csv",
    "metric_coupling_summary.csv",
    "length_quartile_effects.csv",
    "robustness_summary.csv",
    "analysis_provenance.json",
)
COLUMNS_KEY = ["family", "arm", "seed", "filename", "view", "issue", "year", "period"]
GROUP = ["family", "period", "view"]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xx = x - x.mean()
    yy = y - y.mean()
    den = np.sqrt((xx * xx).sum() * (yy * yy).sum())
    return float((xx * yy).sum() / den) if den > 0 else float("nan")


def _boot_corr(x: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    # Publication issues are the sampling unit. Seeds are averaged BEFORE bootstrap.
    idx = rng.integers(0, len(x), size=(n, len(x)))
    xx = x[idx]
    yy = y[idx]
    xc = xx - xx.mean(axis=1, keepdims=True)
    yc = yy - yy.mean(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sum(xc * yc, axis=1) / np.sqrt(
            np.sum(xc * xc, axis=1) * np.sum(yc * yc, axis=1)
        )
    good = r[np.isfinite(r)]
    if len(good) < .98 * n:
        raise AssertionError("Too many degenerate issue-level bootstrap samples")
    return tuple(map(float, np.quantile(good, [.025, .975])))


def load_and_pair(root: Path) -> pd.DataFrame:
    seg = pd.read_csv(root / INPUTS[0], dtype={"period": str})
    lm = pd.read_csv(root / INPUTS[1], dtype={"period": str})
    assert len(seg) == len(lm) == 99720
    assert not seg.duplicated(COLUMNS_KEY).any()
    assert not lm.duplicated(COLUMNS_KEY).any()
    paired = seg.merge(
        lm,
        how="inner",
        on=COLUMNS_KEY,
        suffixes=("_seg", "_lm"),
        validate="one_to_one",
        sort=False,
    )
    assert len(paired) == 99720
    for col in ["chars", "words", "pieces", "unknown", "text_sha256"]:
        if not (paired[f"{col}_seg"].to_numpy() == paired[f"{col}_lm"].to_numpy()).all():
            raise AssertionError("Frozen segmentation and LM records disagree: " + col)
    assert paired["unknown_seg"].eq(0).all()
    assert paired["chars_seg"].gt(0).all()
    assert np.isfinite(paired["bits"]).all()
    assert np.allclose(
        paired["bpc"], paired["bits"] / paired["chars_seg"], rtol=0, atol=1e-12
    )
    paired = paired[COLUMNS_KEY + ["chars_seg", "pieces_seg", "bits"]].rename(
        columns={"chars_seg": "chars", "pieces_seg": "pieces"}
    )
    assert set(paired.arm.unique()) == set(ARMS)
    assert set(paired.seed.unique()) == set(SEEDS)
    assert set(paired.family.unique()) == {"bpe", "unigram"}
    assert set(paired.period.unique()) == {"1888", "1889-94"}
    assert set(paired.view.unique()) == {"en", "xh"}
    assert len(paired[["filename", "view"]].drop_duplicates()) == 2493
    return paired


def make_document_effects(paired: pd.DataFrame) -> pd.DataFrame:
    idx = ["family", "period", "view", "year", "issue", "filename", "seed", "chars"]
    pivot = paired.pivot(index=idx, columns="arm", values=["pieces", "bits"])
    assert len(pivot) == 24930  # 2493 excerpts x 2 families x 5 seeds
    if pivot.isna().any().any():
        raise AssertionError("Missing model-arm score for an evaluated excerpt")
    chars = pivot.index.get_level_values("chars").to_numpy(dtype=float)
    out = pivot.index.to_frame(index=False).reset_index(drop=True)
    for measure, unit in (("pieces", 100.0), ("bits", 1.0)):
        v = {arm: pivot[(measure, arm)].to_numpy(dtype=float) for arm in ARMS}
        out[f"{measure}_allocation"] = (
            .5 * ((v["WP"] - v["PP"]) + (v["WW"] - v["PW"])) * unit / chars
        )
        out[f"{measure}_within_view"] = (
            .5 * ((v["PW"] - v["PP"]) + (v["WW"] - v["WP"])) * unit / chars
        )
        out[f"{measure}_whole_policy"] = (v["WW"] - v["PP"]) * unit / chars
    # Exact factorial identity holds at every sample, not only in aggregate.
    for measure in ("pieces", "bits"):
        assert np.allclose(
            out[f"{measure}_allocation"] + out[f"{measure}_within_view"],
            out[f"{measure}_whole_policy"], rtol=0, atol=1e-11,
        )
    return out


def issue_effects(doc: pd.DataFrame) -> pd.DataFrame:
    effect_columns = [f"{m}_{c}" for m in ("pieces", "bits")
                      for c in ("allocation", "within_view", "whole_policy")]
    q = doc[["family", "period", "view", "year", "issue", "seed", "chars"] + effect_columns].copy()
    for col in effect_columns:
        q[col] *= q["chars"]
    grouped = q.groupby(GROUP + ["year", "issue", "seed"], sort=True, observed=True)
    num = grouped[effect_columns + ["chars"]].sum().reset_index()
    for col in effect_columns:
        num[col] /= num["chars"]
    by_issue = num.groupby(GROUP + ["year", "issue"], sort=True).agg(
        {**{name: "mean" for name in effect_columns}, "chars": "first"}
    ).reset_index()
    pages = doc[["family", "period", "view", "issue", "filename"]].drop_duplicates().groupby(
        GROUP + ["issue"], sort=True
    ).size().rename("page_views").reset_index()
    by_issue = by_issue.merge(pages, on=GROUP + ["issue"], validate="one_to_one")
    assert len(by_issue) > 0 and by_issue.issue.nunique() == 358
    return by_issue


def check_against_frozen(by_issue: pd.DataFrame, root: Path) -> float:
    # Published effect pools characters across issues; simple issue means differ.
    all_errors = []
    for measure, path in (("pieces", "results/segmentation.csv"),
                          ("bits", "results/bigram.csv")):
        published = pd.read_csv(root / path, dtype={"period": str})
        for (family, period, view), subset in by_issue.groupby(GROUP):
            for contrast, col in (
                ("allocation", f"{measure}_allocation"),
                ("within_view", f"{measure}_within_view"),
                ("WW_minus_PP", f"{measure}_whole_policy"),
            ):
                rec = published[
                    (published.family == family) & (published.period == period)
                    & (published.view == view) & (published.contrast == contrast)
                ]
                assert len(rec) == 1
                observed = np.average(subset[col], weights=subset["chars"])
                delta = abs(observed - rec.iloc[0]["mean"])
                all_errors.append(delta)
                if delta > 1e-10:
                    raise AssertionError(f"Mismatch with frozen {path}, {family} {period} {view} {contrast}: {observed} vs {rec.iloc[0]['mean']}")
    return float(max(all_errors))


def associations(by_issue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (key, group) in enumerate(by_issue.groupby(GROUP, sort=True)):
        x = group["pieces_allocation"].to_numpy()
        y = group["bits_allocation"].to_numpy()
        rng = np.random.default_rng(BOOT_SEED + i)
        low, high = _boot_corr(x, y, N_BOOT, rng)
        rows.append(dict(zip(GROUP, key)) | {
            "n_issues": len(group),
            "pearson_r": _corr(x, y),
            "pearson_ci_low": low,
            "pearson_ci_high": high,
            "spearman_rho": _corr(pd.Series(x).rank(method="average").to_numpy(),
                                  pd.Series(y).rank(method="average").to_numpy()),
            "same_direction_count": int((np.sign(x) == np.sign(y)).sum()),
            "discordant_count": int((np.sign(x) != np.sign(y)).sum()),
            "seg_better_lm_worse_count": int(((x < 0) & (y > 0)).sum()),
            "seg_worse_lm_better_count": int(((x > 0) & (y < 0)).sum()),
            "both_better_count": int(((x < 0) & (y < 0)).sum()),
            "both_worse_count": int(((x > 0) & (y > 0)).sum()),
            "issue_mean_seg_allocation": float(np.mean(x)),
            "issue_mean_bpc_allocation": float(np.mean(y)),
        })
    return pd.DataFrame(rows)


def robustness(by_issue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in by_issue.groupby(GROUP, sort=True):
        weights = g["chars"].to_numpy(dtype=float)
        effects = {}
        for measure, col in (("seg", "pieces_allocation"), ("bpc", "bits_allocation")):
            vals = g[col].to_numpy(dtype=float)
            total = (vals * weights).sum() / weights.sum()
            loo = ((vals * weights).sum() - vals * weights) / (weights.sum() - weights)
            effects[measure + "_full_weighted"] = float(total)
            effects[measure + "_leave_one_issue_min"] = float(loo.min())
            effects[measure + "_leave_one_issue_max"] = float(loo.max())
            effects[measure + "_issues_positive"] = int((vals > 0).sum())
            effects[measure + "_issues_negative"] = int((vals < 0).sum())
        rows.append(dict(zip(GROUP, key)) | {"n_issues": len(g)} | effects)
    return pd.DataFrame(rows)


def _boot_weighted_effect(by_issue_seed: pd.DataFrame, metric: str, seed: int) -> tuple[float, float, float]:
    pivot_num = by_issue_seed.pivot(index="issue", columns="seed", values="numer").sort_index()
    pivot_char = by_issue_seed.pivot(index="issue", columns="seed", values="chars").sort_index()
    assert pivot_num.shape[1] == pivot_char.shape[1] == 5
    assert tuple(pivot_num.columns) == SEEDS
    assert np.allclose(pivot_char.to_numpy(), pivot_char.to_numpy()[:, :1])
    numer = pivot_num.to_numpy(dtype=float)
    chars = pivot_char.to_numpy(dtype=float)[:, 0]
    point = float(np.mean(numer.sum(axis=0) / chars.sum()))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(chars), [1 / len(chars)] * len(chars), size=N_BOOT)
    reweighted = (draws @ numer) / (draws @ chars)[:, None]
    boot = np.mean(reweighted, axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return point, float(low), float(high)


def length_quartiles(doc: pd.DataFrame) -> pd.DataFrame:
    unique = doc[["period", "view", "issue", "filename", "chars"]].drop_duplicates()
    assert len(unique) == 2493
    unique["quartile"] = unique.groupby(["period", "view"])["chars"].transform(
        lambda v: pd.qcut(v, 4, labels=QUARTILES).astype(str)
    )
    q = doc.merge(unique, on=["period", "view", "issue", "filename", "chars"],
                  how="left", validate="many_to_one")
    results = []
    for i, (key, g) in enumerate(q.groupby(GROUP + ["quartile"], sort=True)):
        family, period, view, quartile = key
        src = unique[(unique.period == period) & (unique.view == view)
                     & (unique.quartile == quartile)]
        record = dict(zip(GROUP + ["quartile"], key)) | {
            "page_views": len(src), "n_issues": int(src.issue.nunique()),
            "min_chars": int(src.chars.min()), "max_chars": int(src.chars.max()),
            "median_chars": float(src.chars.median()),
        }
        for j, (metric, col) in enumerate((("seg", "pieces_allocation"),
                                           ("bpc", "bits_allocation"))):
            subset = g[["issue", "seed", "chars", col]].copy()
            subset["numer"] = subset["chars"] * subset[col]
            per_issue = subset.groupby(["issue", "seed"], sort=True)[["numer", "chars"]].sum().reset_index()
            point, low, high = _boot_weighted_effect(per_issue, metric, BOOT_SEED + 100 + 2 * i + j)
            record[f"{metric}_allocation"] = point
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        results.append(record)
    out = pd.DataFrame(results)
    assert len(out) == 32
    return out


def make_figures(issues: pd.DataFrame, assoc: pd.DataFrame, length: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 8.0), layout="constrained")
    for i, family in enumerate(("bpe", "unigram")):
        for j, view in enumerate(("en", "xh")):
            ax = axs[i, j]
            dd = issues[(issues.period == "1888") & (issues.family == family) & (issues.view == view)]
            info = assoc[(assoc.period == "1888") & (assoc.family == family) & (assoc.view == view)].iloc[0]
            ax.scatter(dd.pieces_allocation, dd.bits_allocation, alpha=.55, s=23)
            ax.axhline(0, color="gray", linewidth=.8, linestyle="--")
            ax.axvline(0, color="gray", linewidth=.8, linestyle="--")
            ax.set_title(f"{family.upper()} | {'English' if view == 'en' else 'isiXhosa'} archive view")
            ax.set_xlabel("Allocation effect: pieces / 100 alphabetic characters")
            ax.set_ylabel("Allocation effect: bigram bits / character")
            ax.annotate(f"Issue-level r = {info.pearson_r:+.2f} [{info.pearson_ci_low:+.2f}, {info.pearson_ci_high:+.2f}]\n"
                        f"Discordant signs: {int(info.discordant_count)}/{int(info.n_issues)} issues",
                        xy=(.02, .98), xycoords="axes fraction", ha="left", va="top",
                        fontsize=8, bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    fig.suptitle("Exploratory: issue-level coupling between segmentation and downstream loss (1888)", fontsize=13)
    fig.savefig(out / "metric_coupling_1888.png", bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.5), layout="constrained")
    for row, view in enumerate(("en", "xh")):
        for col, (metric, unit) in enumerate((("seg", "pieces / 100 characters"),
                                            ("bpc", "bits / character"))):
            ax = axs[row, col]
            for fam in ("bpe", "unigram"):
                s = length[(length.period == "1888") & (length.view == view) &
                           (length.family == fam)].set_index("quartile").loc[list(QUARTILES)]
                xs = np.arange(1, 5)
                ys = s[f"{metric}_allocation"].to_numpy()
                err = np.array([ys - s[f"{metric}_ci_low"].to_numpy(),
                                s[f"{metric}_ci_high"].to_numpy() - ys])
                ax.errorbar(xs, ys, yerr=err, marker="o" if fam == "bpe" else "s",
                            linewidth=1.2, markersize=4.5, capsize=3, label=fam.upper())
            ax.axhline(0, color="gray", linewidth=.8, linestyle="--")
            ax.set_xticks([1,2,3,4], ["Q1", "Q2", "Q3", "Q4"])
            ax.set_xlabel("Excerpt length (shortest to longest within view)")
            ax.set_ylabel("Allocation effect: " + unit)
            ax.set_title(("English" if view == "en" else "isiXhosa") +
                         " archive view | " + ("Segmentation" if metric == "seg" else "Bigram"))
            if row == 0 and col == 0: ax.legend(loc="best", frameon=False)
    fig.suptitle("Exploratory: allocation effects by source excerpt length (1888)", fontsize=13)
    fig.savefig(out / "length_quartiles_1888.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None,
                        help="Defaults to root/results/exploratory")
    parser.add_argument("--figures", type=Path, default=None,
                        help="Defaults to root/results/figures")
    parser.add_argument("--verify", action="store_true",
                        help="Do not write; compare regenerated CSV/JSON with existing frozen output files")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out if args.out is not None else root / "results" / "exploratory").resolve()
    figures = (args.figures if args.figures is not None else root / "results" / "figures").resolve()
    paired = load_and_pair(root)
    doc = make_document_effects(paired)
    issues = issue_effects(doc)
    max_mismatch = check_against_frozen(issues, root)
    assoc = associations(issues)
    quartile = length_quartiles(doc)
    robust = robustness(issues)
    metadata = {
        "analysis_kind": "POST_HOC_EXPLORATORY_NON_CAUSAL_ASSOCIATION_AND_HETEROGENEITY",
        "source_repository": "Stralb/isixhosa-archive-nlp",
        "source_input_sha256": {name: _sha256(root / name) for name in INPUTS},
        "original_scores_untouched": True,
        "sampling_unit_for_correlations_and_bootstraps": "publication issue, five tokenizer seeds averaged before association",
        "full_sample_aggregator": "character weighted within publication issue, then across issues, then five seed means",
        "excerpt_length_groups": "within evaluation period and archive language view, using frozen source alphabetic-character count",
        "number_of_paired_model_score_rows": len(paired),
        "number_of_original_page_view_excerpts": int(len(paired[["filename", "view"]].drop_duplicates())),
        "number_of_seed_specific_document_arm_quartets": len(doc),
        "number_of_issue_cells": len(issues),
        "number_of_exploratory_association_cells": len(assoc),
        "number_of_length_strata": len(quartile),
        "paired_issue_bootstrap_draws": N_BOOT,
        "bootstrap_rng_seed": BOOT_SEED,
        "max_abs_difference_vs_published_full_sample_allocation_within_whole_effects": max_mismatch,
        "limitations": [
            "Previously inspected historical evaluation years; conditional exploratory follow-up only",
            "Correlations are descriptive, not causal; segmentation and bigram loss share underlying encoded text and character denominators",
            "No human-adjudicated view language, genre, topic, meaning or societal fairness labels",
            "Within-view length quartiles are post-hoc and different across English/isiXhosa views",
            "Issue-level effects weight issues equally; published primary effects weight pooled characters",
            "Bootstrap intervals condition on the observed archive/issues and five trained tokenizer seeds",
            "No multiple-comparison adjustment or confirmatory significance claims for these new analyses",
            "Training/test exact text shingle overlap noted in independent audit",
        ],
    }
    # Fixed decimal rendering, stable sorting, no new outcomes fed into existing protocols.
    frames = {
        "issue_allocation_effects.csv": issues.sort_values(GROUP + ["issue"]),
        "metric_coupling_summary.csv": assoc.sort_values(GROUP),
        "length_quartile_effects.csv": quartile.sort_values(GROUP + ["quartile"]),
        "robustness_summary.csv": robust.sort_values(GROUP),
    }
    outputs = {name: frame.to_csv(index=False, float_format="%.15g", lineterminator="\n")
               for name, frame in frames.items()}
    outputs["analysis_provenance.json"] = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.verify:
        for name, data in outputs.items():
            if (out / name).read_text(encoding="utf-8") != data:
                raise AssertionError("Exploratory artifact differs from independent fresh calculation: " + name)
        print("PASS: 5 exploratory artifacts match fresh independent calculations")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            (out / name).write_text(data, encoding="utf-8")
        if not args.no_figures:
            figures.mkdir(parents=True, exist_ok=True)
            make_figures(issues, assoc, quartile, figures)
    print(f"PASS: matched {len(paired):,} per-arm P3/P5 scores, {len(doc):,} paired document-seed quads, "
          f"{len(issues)} issue effects, {len(quartile)} length strata")
    print(f"PASS: original published contrasts recovered (max error {max_mismatch:.3g})")
    print(assoc[(assoc.period == "1888")][["family", "view", "n_issues", "pearson_r",
             "discordant_count", "same_direction_count"]].to_string(index=False))
    print("Exploratory: do NOT relabel new associations as preregistered findings.")


if __name__ == "__main__":
    main()
