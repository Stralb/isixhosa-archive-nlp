#!/usr/bin/env python3
"""Reproduce the 2x2 tokenizer sampling experiment from the original IsiXIT data.

This is a readable independent implementation of the original frozen design.
It is not the original preregistration and its model .model hashes will differ
if SentencePiece embeds a different training path. The sampled text and learned
token pieces can be compared with the archived implementation.

The original archival TXT ZIP must be supplied by a licensed reader. No data
is downloaded or checked into the public repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARCHIVE_SHA256 = "375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5"
ISSUE = re.compile(r"IMVO_(\d{4})-(\d{2})-(\d{2})_p\d{3}\.txt$")
WORDS = re.compile(r"[a-z]+")
ARMS = ("PP", "PW", "WP", "WW")
FAMILIES = ("bpe", "unigram")
SEEDS = (441, 442, 443, 444, 445)
VIEWS = ("en", "xh")
ALPHAS = (1.0, 10.0, 100.0)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write to " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_pages(archive: Path, *, verify_hash: bool = True) -> tuple[list, list, list]:
    """Parse only IMVO EnTxt/XhTxt files and enforce disjoint issue dates."""
    if verify_hash and digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Course archive does not match pinned SHA256")
    pages = {}
    with zipfile.ZipFile(archive) as src:
        for name in sorted(src.namelist()):
            fields = name.split("/")
            if len(fields) < 2 or fields[-2] not in ("EnTxt", "XhTxt"):
                continue
            match = ISSUE.fullmatch(fields[-1])
            if not match:
                continue
            year, month, day = map(int, match.groups())
            if year not in {1885, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894}:
                continue
            view = "en" if fields[-2] == "EnTxt" else "xh"
            key = (fields[-1], view)
            if key in pages:
                raise ValueError("Duplicate source page/view: " + str(key))
            words = WORDS.findall(src.read(name).decode("utf-8-sig", "replace").lower())
            pages[key] = dict(
                filename=fields[-1],
                issue=f"IMVO_{year:04d}-{month:02d}-{day:02d}",
                year=year, month=month, view=view, words=words,
            )
    ordered = sorted(pages.values(), key=lambda r: (r["filename"], r["view"]))
    train = [p for p in ordered if p["year"] == 1885 or (p["year"] == 1887 and p["month"] <= 9)]
    validation = [p for p in ordered if p["year"] == 1887 and p["month"] >= 10]
    heldout = [p for p in ordered if p["year"] >= 1888]
    if not (train and validation and heldout):
        raise ValueError("Missing required train, validation or heldout period")
    groups = [set(p["issue"] for p in part) for part in (train, validation, heldout)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("Overlapping publication issues across splits")
    return train, validation, heldout


def eligible_units(pages: list[dict]) -> list[dict]:
    return [{**p, "nwords": len(p["words"])} for p in pages if len(p["words"]) >= 16]


def distribution(units: list[dict], arm: str):
    """Frozen page P versus word W allocation, crossed with selection P/W."""
    if arm not in ARMS:
        raise ValueError("Unknown arm: " + arm)
    page = defaultdict(lambda: {"en": 0, "xh": 0})
    for unit in units:
        page[unit["filename"]][unit["view"]] = unit["nwords"]
    bypage = list(page.values())
    if not bypage:
        raise ValueError("No eligible training pages")
    p_page = sum(x["en"] / (x["en"] + x["xh"]) for x in bypage) / len(bypage)
    p_word = sum(x["en"] for x in bypage) / sum(sum(x.values()) for x in bypage)
    english_share = p_page if arm[0] == "P" else p_word
    indices = {v: [i for i, u in enumerate(units) if u["view"] == v] for v in VIEWS}
    probabilities = {}
    for view in VIEWS:
        if not indices[view]:
            raise ValueError("No eligible " + view + " training view")
        weights = np.array(
            [
                units[i]["nwords"] / sum(page[units[i]["filename"]].values())
                if arm[1] == "P" else units[i]["nwords"]
                for i in indices[view]
            ], dtype=float
        )
        probabilities[view] = weights / weights.sum()
    return indices, probabilities, english_share


def draw_sample(units: list[dict], arm: str, seed: int, budget: int = 500_000):
    """Exactly reproduce the saved factorial sampler's RNG and text budget."""
    indices, probabilities, english_share = distribution(units, arm)
    rng = np.random.default_rng(int(seed) * 1000 + {"PP": 11, "PW": 12, "WP": 13, "WW": 14}[arm])
    result, total, en_total = [], 0, 0
    while total < budget - 30:
        view = "en" if rng.random() < english_share else "xh"
        ids = indices[view]
        unit = units[ids[int(rng.choice(len(ids), p=probabilities[view]))]]
        nstarts = max(unit["nwords"] - 32 + 1, 1)
        start = int(rng.integers(nstarts))
        block = unit["words"][start:start + 32]
        words, size = [], 0
        for word in block:
            if size + len(word) > budget - total:
                break
            words.append(word)
            size += len(word)
        if not words:
            break
        result.append(" ".join(words))
        total += size
        if view == "en":
            en_total += size
    if not budget - 30 <= total <= budget:
        raise AssertionError("Invalid fixed-character-budget sample")
    sample_text = ("\n".join(result) + "\n").encode("utf-8")
    return result, {
        "arm": arm, "seed": seed, "characters": total,
        "en_characters": en_total, "en_share": en_total / total,
        "sample_sha256": hashlib.sha256(sample_text).hexdigest(),
    }


def lm_training_docs(train: list[dict]) -> list[dict]:
    """Same first-128-token document window for each eligible page and view."""
    return [
        {"text": " ".join(p["words"][:128]), "view": p["view"], "issue": p["issue"]}
        for p in train if len(p["words"]) >= 40
    ]


def clean_validation_windows(train: list[dict], validation: list[dict]) -> list[dict]:
    """Exclude every 32-word window containing a train 13-gram in its view."""
    owner = {v: set() for v in VIEWS}
    for record in train:
        words = record["words"]
        owner[record["view"]].update(
            tuple(words[i:i + 13]) for i in range(len(words) - 12)
        )
    windows = []
    for record in validation:
        words = record["words"]
        if len(words) < 32:
            continue
        flags = np.fromiter(
            (tuple(words[i:i + 13]) in owner[record["view"]]
             for i in range(len(words) - 12)),
            dtype=np.uint8, count=len(words) - 12,
        )
        pref = np.concatenate(([0], np.cumsum(flags)))
        starts = np.flatnonzero(pref[20:] - pref[:-20] == 0)
        if len(starts) > 5:
            starts = starts[np.round(np.linspace(0, len(starts) - 1, 5)).astype(int)]
        for start in starts:
            selected = words[int(start):int(start) + 32]
            windows.append({
                "view": record["view"], "issue": record["issue"],
                "text": " ".join(selected), "chars": sum(map(len, selected)),
            })
    if not windows:
        raise ValueError("No uncontaminated validation windows")
    return windows


def test_examples(heldout: list[dict]) -> list[dict]:
    pages = []
    for record in heldout:
        if len(record["words"]) < 40:
            continue
        words = record["words"][:128]
        pages.append({
            "filename": record["filename"], "issue": record["issue"],
            "year": record["year"], "view": record["view"],
            "period": "1888" if record["year"] == 1888 else "1889-94",
            "chars": sum(map(len, words)), "text": " ".join(words),
        })
    return pages


class SmoothedBigram:
    """Token bigram LM with the frozen unigram backoff formula."""
    def __init__(self, tokenizer, documents: list[dict]):
        self.vocab = tokenizer.vocab_size()
        self.bigram = Counter()
        self.previous = Counter()
        self.unigram = Counter()
        self.ntokens = 0
        for doc in documents:
            before = -1
            for token in tokenizer.encode(doc["text"], out_type=int):
                self.bigram[(before, token)] += 1
                self.previous[before] += 1
                self.unigram[token] += 1
                self.ntokens += 1
                before = token
        if not self.ntokens:
            raise ValueError("Empty language model training set")

    def bits(self, tokens: list[int], alpha: float) -> float:
        before, bits = -1, 0.0
        den = self.ntokens + 0.1 * self.vocab
        for token in tokens:
            backoff = (self.unigram[token] + 0.1) / den
            prob = (self.bigram[(before, token)] + alpha * backoff) / (self.previous[before] + alpha)
            if not 0 < prob <= 1:
                raise AssertionError("Invalid token conditional probability")
            bits -= math.log2(prob)
            before = token
        return bits


def contrast(arms: dict) -> dict:
    """Main effects and difference-in-differences for the 2 by 2 design."""
    a = arms
    return {
        "allocation": 0.5 * ((a["WP"] - a["PP"]) + (a["WW"] - a["PW"])),
        "within_view": 0.5 * ((a["PW"] - a["PP"]) + (a["WW"] - a["WP"])),
        "interaction": (a["WW"] - a["WP"]) - (a["PW"] - a["PP"]),
        "WW_minus_PP": a["WW"] - a["PP"],
    }


def train_and_select(train, validation, out: Path, families, seeds, budget: int, vocab_size: int):
    """Train tokenizers, fit count bigram, select alpha from validation only."""
    import sentencepiece as spm

    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    units = eligible_units(train)
    docs = lm_training_docs(train)
    val = clean_validation_windows(train, validation)
    manifest, scores = [], []
    for family in families:
        for seed in seeds:
            for arm in ARMS:
                name = f"{family}_{arm}_{seed}"
                sample, sample_metadata = draw_sample(units, arm, seed, budget)
                prefix = models / name
                training = prefix.with_suffix(".training.tmp")
                training.write_text("\n".join(sample) + "\n", encoding="utf-8")
                try:
                    spm.SentencePieceTrainer.train(
                        input=str(training), model_prefix=str(prefix),
                        model_type=family, vocab_size=vocab_size,
                        character_coverage=1.0, hard_vocab_limit=False,
                        normalization_rule_name="identity",
                        bos_id=-1, eos_id=-1, pad_id=-1, unk_id=0,
                        num_threads=1, max_sentence_length=4096,
                        input_sentence_size=0, shuffle_input_sentence=False,
                        split_by_whitespace=True, add_dummy_prefix=True,
                        remove_extra_whitespaces=True, minloglevel=2,
                    )
                finally:
                    training.unlink(missing_ok=True)
                sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
                lm = SmoothedBigram(sp, docs)
                manifest.append({
                    "name": name, "family": family, "arm": arm, "seed": seed,
                    **sample_metadata, "vocab": sp.vocab_size(),
                    "model_sha256": digest(prefix.with_suffix(".model")),
                })
                for view in VIEWS:
                    rows = [r for r in val if r["view"] == view]
                    chars = sum(r["chars"] for r in rows)
                    bits = {a: 0.0 for a in ALPHAS}
                    for record in rows:
                        tokens = sp.encode(record["text"], out_type=int)
                        for alpha in ALPHAS:
                            bits[alpha] += lm.bits(tokens, alpha)
                    for alpha in ALPHAS:
                        scores.append({
                            "name": name, "view": view, "alpha": alpha,
                            "bpc": bits[alpha] / chars,
                        })
                print("TRAINED", name, flush=True)
    write_csv(out / "training_manifest.csv", manifest)
    write_csv(out / "validation_scores.csv", scores)
    means = {a: float(np.mean([r["bpc"] for r in scores if r["alpha"] == a])) for a in ALPHAS}
    optimum = min(means.values())
    tied = [a for a in ALPHAS if means[a] <= optimum + 1e-12]
    selected = sorted(tied, key=lambda a: (a != 10.0, a))[0]
    (out / "selection.json").write_text(
        json.dumps({"selected_alpha": selected, "grid": ALPHAS,
                    "validation_scores": means, "heldout_used_for_selection": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected, manifest


def evaluate(train, test, out, manifest, alpha):
    """Score each heldout page/view under each fitted tokenizer and LM."""
    import sentencepiece as spm

    pages = test_examples(test)
    docs = lm_training_docs(train)
    scores = []
    for model in manifest:
        path = out / "models" / (model["name"] + ".model")
        if digest(path) != model["model_sha256"]:
            raise AssertionError("Corrupted tokenizer " + model["name"])
        sp = spm.SentencePieceProcessor(model_file=str(path))
        lm = SmoothedBigram(sp, docs)
        for page in pages:
            tokens = sp.encode(page["text"], out_type=int)
            scores.append({
                "family": model["family"], "arm": model["arm"], "seed": model["seed"],
                "filename": page["filename"], "view": page["view"],
                "issue": page["issue"], "year": page["year"], "period": page["period"],
                "chars": page["chars"], "pieces": len(tokens),
                "unknown": sum(t == sp.unk_id() for t in tokens),
                "bits": lm.bits(tokens, alpha),
            })
        print("EVALUATED", model["name"], flush=True)
    write_csv(out / "heldout_scores.csv", scores)
    return scores


def summarize(scores, draws=5000):
    """Paired issue-bootstrap intervals, with tokenizer seeds treated as fixed."""
    output = []
    periods = sorted({r["period"] for r in scores})
    families = sorted({r["family"] for r in scores})
    seeds = sorted({r["seed"] for r in scores})
    for period in periods:
        issues = sorted({r["issue"] for r in scores if r["period"] == period})
        n = len(issues)
        rng = np.random.default_rng(343401 + (0 if period == "1888" else 1))
        weights = rng.multinomial(n, [1 / n] * n, size=draws)
        for family in families:
            for view in VIEWS:
                group = [r for r in scores if r["period"] == period and
                         r["family"] == family and r["view"] == view]
                if not group:
                    continue
                for metric in ("pieces", "bits"):
                    by_seed, sampled = [], []
                    for seed in seeds:
                        cells = {}
                        for arm in ARMS:
                            rows = [r for r in group if r["arm"] == arm and r["seed"] == seed]
                            if not rows:
                                raise AssertionError("Missing factorial cell")
                            numer = defaultdict(float)
                            denom = defaultdict(float)
                            for row in rows:
                                numer[row["issue"]] += float(row[metric])
                                denom[row["issue"]] += float(row["chars"])
                            x = np.array([numer[i] for i in issues])
                            y = np.array([denom[i] for i in issues])
                            scale = 100.0 if metric == "pieces" else 1.0
                            point = scale * x.sum() / y.sum()
                            boots = scale * (weights @ x) / (weights @ y)
                            cells[arm] = point, boots
                        by_seed.append(contrast({a: cells[a][0] for a in ARMS}))
                        sampled.append(contrast({a: cells[a][1] for a in ARMS}))
                    for label in ("allocation", "within_view", "interaction", "WW_minus_PP"):
                        points = [x[label] for x in by_seed]
                        bs = np.mean(np.stack([x[label] for x in sampled]), axis=0)
                        lower, upper = np.quantile(bs, [0.025, 0.975])
                        output.append({
                            "family": family, "period": period, "view": view,
                            "metric": "pieces_per_100_chars" if metric == "pieces" else "bits_per_char",
                            "contrast": label, "mean": float(np.mean(points)),
                            "ci_low": float(lower), "ci_high": float(upper),
                            "seeds": len(seeds), "issues": n,
                        })
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("full", "smoke"), default="full")
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()
    if args.bootstrap < 10:
        p.error("--bootstrap must be >= 10")
    if args.out.exists() and any(args.out.iterdir()):
        p.error("Refusing to overwrite an existing output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    complete = args.mode == "full"
    train, validation, heldout = load_pages(args.archive, verify_hash=complete)
    families = FAMILIES if complete else ("bpe",)
    seeds = SEEDS if complete else (441,)
    budget = 500_000 if complete else 4000
    vocab = 2000 if complete else 100
    alpha, models = train_and_select(train, validation, args.out, families, seeds, budget, vocab)
    scores = evaluate(train, heldout, args.out, models, alpha)
    results = summarize(scores, args.bootstrap)
    write_csv(args.out / "contrasts.csv", results)
    (args.out / "run_manifest.json").write_text(json.dumps({
        "source_sha256": digest(args.archive), "mode": args.mode,
        "training_models": len(models), "test_scores": len(scores),
        "validation_selected_alpha": alpha, "bootstrap_draws": args.bootstrap,
        "warning": "New implementation, not a fresh untouched historical evaluation.",
    }, indent=2) + "\n", encoding="utf-8")
    print("PASS:", len(models), "trained tokenizers,", len(scores),
          "heldout rows,", len(results), "contrast estimates", flush=True)


if __name__ == "__main__":
    main()
