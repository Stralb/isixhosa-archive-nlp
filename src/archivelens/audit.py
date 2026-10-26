#!/usr/bin/env python3
"""ArchiveLens: counterfactual estimand audit of historical XhTxt patterns.

Frozen plan: reports/ARCHIVELENS_PRECOMMIT_2026-10-26.md.
This is conditional/post-hoc archival research, NOT a social fairness detector.

Example:
 python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip \
   --out results/archivelens --figures results/figures \
   --report reports/ARCHIVELENS_RESULTS_2026-10-26.md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import numpy as np

EXPECTED_SHA256 = '375c5864e180717f34403287e502cf8812b038baf5a8fb4ae8ee8cdb2be321d5'
YEARS = (1885, 1887, 1888)
SOURCES = ('IMVO', 'ISIG')
ESTIMANDS = ('word_mass', 'page_equal', 'issue_equal', 'month_equal')
FEATURES_K = 120
N_BOOT = 3000
BOOT_SEED = 20261026
SCALE = 10000.0
MIN_WORDS = 40
RE_NAME = re.compile(r'/XhTxt/(IMVO|ISIG)_(1885|1887|1888)-(\d{2})-(\d{2})_p(\d{3})\.txt$')
RE_WORD = re.compile(r'[a-z]+')


@dataclass(frozen=True)
class Page:
    source: str
    year: int
    month: int
    issue: str
    page_no: int
    n_words: int
    denom: int
    counts: np.ndarray


@dataclass
class Units:
    source: str
    year: int
    ids: list[str]
    months: np.ndarray
    counts: np.ndarray
    positions: np.ndarray
    page_rate_sums: np.ndarray
    n_pages: np.ndarray
    rates: np.ndarray
    month_indices: dict[int, np.ndarray]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def names_in_archive(z: ZipFile):
    candidates = []
    for name in z.namelist():
        if name.startswith('__MACOSX') or '/._' in name or not name.endswith('.txt'):
            continue
        match = RE_NAME.search(name)
        if match:
            src, yr, month, day, page = match.groups()
            candidates.append((name, src, int(yr), int(month), int(day), int(page)))
    candidates.sort(key=lambda row: (row[1], row[2], row[3], row[4], row[5], row[0]))
    return candidates


def word_grams(raw: bytes, candidate_index: dict[str, int] | None):
    text = raw.decode('utf-8', errors='replace').lower()
    word_list = RE_WORD.findall(text)
    words = len(word_list)
    positions = sum(max(0, len(w) - 3) for w in word_list)
    if candidate_index is None:
        counts: Counter | np.ndarray = Counter()
        for w in word_list:
            for i in range(len(w) - 3):
                counts[w[i:i + 4]] += 1
    else:
        counts = np.zeros(len(candidate_index), dtype=np.int64)
        for w in word_list:
            for i in range(len(w) - 3):
                j = candidate_index.get(w[i:i + 4])
                if j is not None:
                    counts[j] += 1
    return words, positions, counts, text.count('\ufffd')


def get_candidates(z: ZipFile, names: list[tuple]):
    totals = {s: Counter() for s in SOURCES}
    denoms = {s: 0 for s in SOURCES}
    coverage = {s: {'pages': 0, 'issues': set(), 'excluded_short': 0,
                    'words': 0, 'positions': 0, 'replacement_chars': 0} for s in SOURCES}
    for name, src, yr, month, day, no in names:
        if yr != 1887:
            continue
        n, d, grams, replaced = word_grams(z.read(name), None)
        if n < MIN_WORDS or d <= 0:
            coverage[src]['excluded_short'] += 1
            continue
        coverage[src]['pages'] += 1
        coverage[src]['issues'].add(f'{src}_{yr}-{month:02}-{day:02}')
        coverage[src]['words'] += n
        coverage[src]['positions'] += d
        coverage[src]['replacement_chars'] += replaced
        totals[src].update(grams)
        denoms[src] += d
    if any(denoms[s] == 0 for s in SOURCES):
        raise ValueError('Missing 1887 candidate-selection corpus for one title')
    grams = set(totals['IMVO']) | set(totals['ISIG'])
    rank = sorted(grams, key=lambda g: (-.5 * (
        totals['IMVO'][g] / denoms['IMVO'] + totals['ISIG'][g] / denoms['ISIG']), g))
    selected = rank[:FEATURES_K]
    if len(selected) != FEATURES_K:
        raise AssertionError('Not enough 1887 patterns')
    candidate_rates = []
    for rank_id, gram in enumerate(selected, 1):
        a = totals['IMVO'][gram] / denoms['IMVO'] * SCALE
        b = totals['ISIG'][gram] / denoms['ISIG'] * SCALE
        candidate_rates.append({'rank': rank_id, 'pattern': gram,
                                'imvo_1887_per_10k': a, 'isig_1887_per_10k': b,
                                'source_mean_1887_per_10k': (a + b) / 2})
    for src in SOURCES:
        coverage[src]['issues'] = len(coverage[src]['issues'])
    return selected, candidate_rates, coverage


def eval_pages(z: ZipFile, names: list[tuple], selected: list[str]):
    index = {x: i for i, x in enumerate(selected)}
    rows = defaultdict(list)
    eligibility = defaultdict(lambda: {'seen': 0, 'eligible': 0, 'excluded_short': 0,
                                       'words': 0, 'positions': 0, 'replacement_chars': 0})
    for name, src, year, month, day, page_no in names:
        if year not in (1885, 1888):
            continue
        key = (src, year)
        eligibility[key]['seen'] += 1
        words, positions, counts, replaced = word_grams(z.read(name), index)
        if words < MIN_WORDS or positions == 0:
            eligibility[key]['excluded_short'] += 1
            continue
        eligibility[key]['eligible'] += 1
        eligibility[key]['words'] += words
        eligibility[key]['positions'] += positions
        eligibility[key]['replacement_chars'] += replaced
        rows[key].append(Page(src, year, month, f'{src}_{year}-{month:02}-{day:02}',
                              page_no, words, positions, counts))
    return rows, eligibility


def make_units(source: str, year: int, pages: list[Page], only_months: set[int] | None = None) -> Units:
    by_issue = defaultdict(list)
    for page in pages:
        if only_months is None or page.month in only_months:
            by_issue[page.issue].append(page)
    if not by_issue:
        raise ValueError(f'No issue support for {source} {year}')
    ids, months, cnt, pos, page_sums, n_pages = [], [], [], [], [], []
    for issue in sorted(by_issue):
        views = by_issue[issue]
        months.append(views[0].month)
        if any(v.month != views[0].month for v in views):
            raise AssertionError('Issue spans multiple date months')
        ids.append(issue)
        cc = np.sum([p.counts for p in views], axis=0)
        dd = sum(p.denom for p in views)
        pp = np.sum([p.counts / p.denom for p in views], axis=0)
        cnt.append(cc)
        pos.append(dd)
        page_sums.append(pp)
        n_pages.append(len(views))
    c = np.asarray(cnt, dtype=np.float64)
    d = np.asarray(pos, dtype=np.float64)
    p = np.asarray(page_sums, dtype=np.float64)
    n = np.asarray(n_pages, dtype=np.float64)
    m = np.asarray(months, dtype=np.int16)
    return Units(source, year, ids, m, c, d, p, n, c / d[:, None],
                 {int(mm): np.where(m == mm)[0] for mm in sorted(set(m))})


def estimates(u: Units, months: list[int]) -> np.ndarray:
    if set(u.month_indices) != set(months):
        raise AssertionError('Unmatched common-month support')
    word = u.counts.sum(axis=0) / u.positions.sum()
    page = u.page_rate_sums.sum(axis=0) / u.n_pages.sum()
    issue = u.rates.mean(axis=0)
    month = np.mean([u.rates[u.month_indices[mm]].mean(axis=0) for mm in months], axis=0)
    return np.stack((word, page, issue, month)) * SCALE


def resample_estimates(u: Units, sampled_months: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    # The sampling unit is the issue: all page-view records of a sampled issue move together.
    selected_indices = []
    month_rates = []
    for month in sampled_months:
        idx = u.month_indices[int(month)]
        drawn = rng.choice(idx, size=len(idx), replace=True)
        selected_indices.extend(drawn.tolist())
        month_rates.append(u.rates[drawn].mean(axis=0))
    sampled = np.asarray(selected_indices, dtype=int)
    cc = u.counts[sampled]
    positions = u.positions[sampled]
    page_sums = u.page_rate_sums[sampled]
    n_pages = u.n_pages[sampled]
    word = cc.sum(axis=0) / positions.sum()
    page = page_sums.sum(axis=0) / n_pages.sum()
    issue = u.rates[sampled].mean(axis=0)
    month = np.mean(month_rates, axis=0)
    return np.stack((word, page, issue, month)) * SCALE


def bootstrap_pair(a: Units, b: Units, months: list[int], seed: int, draws: int = N_BOOT):
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(ESTIMANDS), a.counts.shape[1]), dtype=np.float64)
    for i in range(draws):
        drawn_months = rng.choice(months, size=len(months), replace=True)
        before = resample_estimates(a, drawn_months, rng)
        after = resample_estimates(b, drawn_months, rng)
        samples[i] = after - before
    return np.quantile(samples, [.025, .975], axis=0)


def delta_support(value: float, lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def summarize_claims(claims: list[dict], source: str) -> dict:
    rows = [r for r in claims if r['source'] == source]
    return {'candidate_patterns': len(rows),
            'sign_reversal_any': sum(r['sign_reversal_any'] for r in rows),
            'supported_word_mass': sum(r['supported_word_mass'] for r in rows),
            'support_loss_any': sum(r['support_loss_any'] for r in rows),
            'strong_conflict_any': sum(r['strong_conflict_any'] for r in rows),
            'stable_supported_all': sum(r['stable_supported_all'] for r in rows),
            'cover_page_word_mass_sign_change': sum(r['cover_page_word_mass_sign_change'] for r in rows),
            'pages_excluded_p001': None}


def perform(data_path: Path, draws: int = N_BOOT):
    digest = sha256(data_path)
    if digest != EXPECTED_SHA256:
        raise ValueError(f'Original ZIP hash mismatch: {digest}')
    with ZipFile(data_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Corrupt ZIP entry')
        names = names_in_archive(archive)
        selected, ranked, training_coverage = get_candidates(archive, names)
        pages, eligibility = eval_pages(archive, names, selected)

    records, long_rows, units, summaries, months_by_title, first_page_coverage = [], [], {}, {}, {}, {}
    for i, src in enumerate(SOURCES):
        initial = {yr: make_units(src, yr, pages[(src, yr)]) for yr in (1885, 1888)}
        common = sorted(set(initial[1885].month_indices) & set(initial[1888].month_indices))
        if not common:
            raise AssertionError('No common calendar month support for ' + src)
        months_by_title[src] = common
        u = {yr: make_units(src, yr, pages[(src, yr)], set(common)) for yr in (1885, 1888)}
        units[src] = u
        point = {yr: estimates(u[yr], common) for yr in (1885, 1888)}
        d = point[1888] - point[1885]
        intervals = bootstrap_pair(u[1885], u[1888], common, BOOT_SEED + i, draws=draws)
        first = {yr: [p for p in pages[(src, yr)] if p.page_no != 1]
                 for yr in (1885, 1888)}
        first_page_coverage[src] = {str(yr): {'removed': len(pages[(src, yr)]) - len(first[yr]),
                                                    'retained': len(first[yr])} for yr in (1885, 1888)}
        if any(len(first[yr]) == 0 for yr in (1885, 1888)):
            raise AssertionError('No p001-excluded pages in ' + src)
        first_units = {yr: make_units(src, yr, first[yr], set(common)) for yr in (1885, 1888)}
        if set(first_units[1885].month_indices) == set(first_units[1888].month_indices) == set(common):
            first_delta = estimates(first_units[1888], common) - estimates(first_units[1885], common)
        else:
            surviving = sorted(set(first_units[1885].month_indices) & set(first_units[1888].month_indices))
            if not surviving:
                raise AssertionError('No shared months after p001 removal')
            first_units = {yr: make_units(src, yr, first[yr], set(surviving)) for yr in (1885,1888)}
            first_delta = estimates(first_units[1888], surviving) - estimates(first_units[1885], surviving)
            first_page_coverage[src]['common_months_after_removal'] = surviving
        for j, gram in enumerate(selected):
            base = float(d[0, j])
            base_supported = delta_support(base, *intervals[:, 0, j])
            alt_reversed = [ESTIMANDS[k] for k in range(1, 4) if base * d[k, j] < 0]
            support_losses = [ESTIMANDS[k] for k in range(1, 4) if base_supported and not delta_support(
                float(d[k,j]), *intervals[:, k, j])]
            strong_conflicts = [ESTIMANDS[k] for k in range(1, 4) if base*d[k,j] < 0
                and base_supported and delta_support(float(d[k,j]), *intervals[:,k,j])
                and abs(base) >= 2 and abs(float(d[k,j])) >= 2]
            baseline_flip = bool(base * first_delta[0,j] < 0)
            stable_supported = bool(base_supported and all(
                base * float(d[k,j]) > 0 and delta_support(float(d[k,j]), *intervals[:,k,j])
                for k in range(1,4)))
            rec = {'source': src, 'pattern': gram, 'rank_from_1887': j + 1,
                   'word_mass_delta': base, 'word_mass_ci_low': float(intervals[0,0,j]),
                   'word_mass_ci_high': float(intervals[1,0,j]),
                   'page_equal_delta': float(d[1,j]), 'issue_equal_delta': float(d[2,j]),
                   'month_equal_delta': float(d[3,j]),
                   'sign_reversal_any': bool(alt_reversed), 'sign_reversal_estimands': ';'.join(alt_reversed),
                   'supported_word_mass': base_supported,
                   'support_loss_any': bool(support_losses), 'support_loss_estimands': ';'.join(support_losses),
                   'strong_conflict_any': bool(strong_conflicts),
                   'strong_conflict_estimands': ';'.join(strong_conflicts),
                   'stable_supported_all': stable_supported,
                   'p001_removed_word_mass_delta': float(first_delta[0, j]),
                   'cover_page_word_mass_sign_change': baseline_flip}
            records.append(rec)
            for k, est in enumerate(ESTIMANDS):
                long_rows.append({'source':src, 'pattern':gram, 'rank_from_1887':j+1,
                                  'estimand':est, 'rate_1885':float(point[1885][k,j]),
                                  'rate_1888':float(point[1888][k,j]),
                                  'delta':float(d[k,j]),
                                  'ci_low':float(intervals[0,k,j]),
                                  'ci_high':float(intervals[1,k,j]),
                                  'unadjusted_interval_excludes_zero':delta_support(float(d[k,j]),*intervals[:,k,j])})
    for src in SOURCES:
        summaries[src] = summarize_claims(records, src)
        summaries[src]['pages_excluded_p001'] = first_page_coverage[src]
        summaries[src]['common_months'] = months_by_title[src]
        for yr in (1885, 1888):
            rows = pages[(src, yr)]
            summaries[src][f'year_{yr}'] = {
                **eligibility[(src,yr)], 'issues': len(set(p.issue for p in rows)),
                'months': len(set(p.month for p in rows)),
                'first_pages': sum(p.page_no == 1 for p in rows),
            }
        summaries[src]['training_1887'] = training_coverage[src]
    dicts = {(r['source'],r['pattern']):r for r in records}
    cross_source_disagreement = sum(
        dicts[('IMVO',p)]['word_mass_delta'] * dicts[('ISIG',p)]['word_mass_delta'] < 0
        for p in selected)
    metadata = {'analysis': 'ArchiveLens historical surface-pattern claim sensitivity',
                'analysis_date': '2026-10-26', 'source_sha256': digest,
                'input_file': 'TxtLangFiles1870-1894_TB.zip', 'archive_view': 'XhTxt',
                'sources': list(SOURCES), 'year_contrast': [1885,1888],
                'candidate_selection_year': 1887, 'candidate_patterns': FEATURES_K,
                'within_word_4grams_only':True, 'normalization': 'lowercase ASCII [a-z]+',
                'minimum_words_per_view_page': MIN_WORDS,
                'rate_unit': 'per 10000 within-word 4-gram opportunities',
                'estimands':list(ESTIMANDS), 'bootstrap_replicates':draws,
                'bootstrap_seed': BOOT_SEED, 'bootstrap': 'paired calendar months and nested publication issues',
                'status': 'post-hoc conditional exploratory (test years previously examined)',
                'multiple_comparison_correction':'none; correlated candidates and intervals descriptive',
                'source_comparison': 'both titles from the SAME historical archive collection, not independent external validation',
                'warnings': ['Archive text is corrected transcription rather than verified raw OCR.',
                             'XhTxt is a block view, not human-validated word-level language identity.',
                             'Historical year changes are not evidence of semantic or cultural evolution.',
                             'Topic, genre, newspaper formatting and source preservation are not controlled.',
                             'First-page removal does not uniquely identify or remove mastheads.',
                             'The selected 1885/1888 years have been examined in earlier research.',
                             'Bootstrap intervals condition on observed years, issues and feature selection.',
                             'No multiple-testing correction across 120 highly correlated character patterns.',
                             'The same digitized archive supplies both newspaper titles.',
                             'No independent modern corpus, historical genre annotations or human evaluation.']}
    return {'metadata':metadata, 'selection_1887':ranked,
            'summary':{**summaries,'cross_source_word_mass_sign_disagreement':cross_source_disagreement},
            'claims':records,'estimands':long_rows}


def _fmt(value: float) -> str:
    return f'{value:.8f}'


def dump_csv(rows: list[dict]):
    out = io.StringIO(newline='')
    columns = list(rows[0])
    writer = csv.DictWriter(out, columns, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (_fmt(v) if isinstance(v, float) else v) for k,v in row.items()})
    return out.getvalue()


def choose_witnesses(claims: list[dict], source: str, n: int = 5):
    records = [r for r in claims if r['source'] == source]
    # Prespecified non-outcome-specific prioritization of sensitivity categories;
    # ranking within each category by magnitude difference, never by cultural interpretation.
    return sorted(records, key=lambda r: (
        -int(r['strong_conflict_any']), -int(r['sign_reversal_any']),
        -int(r['support_loss_any']),
        -max(abs(r[f'{e}_delta'] - r['word_mass_delta']) for e in ESTIMANDS[1:]),
        r['rank_from_1887']))[:n]


def make_markdown(audit: dict) -> str:
    summ = audit['summary']
    result = ['# ArchiveLens: counterfactual historical-language claim stability',
              '', '**26 October 2026 — post-hoc, conditional research evidence.**',
              '', '## Research question', '',
              'Would statements about 1885-to-1888 changes in the **surface character patterns** of historical isiXhosa archive views change under defensible alternative page-, issue-, month- or text-mass sampling estimands?',
              '', '## Methods (frozen before this run)', '',
              '- From the original verified source ZIP, used XhTxt block views of *Imvo Zabantsundu* and *Isigidimi samaXosa*; at least 40 lowercased ASCII words per view-page.',
              '- Selected exactly **120 within-word character 4-grams using 1887 only**, ranking by the mean of source-normalised frequencies across both newspaper titles.',
              '- Contrasted **1888 minus 1885** frequency per 10,000 eligible within-word four-character positions, under **word/character-mass, equal-page, equal-issue and equal-month** estimands.',
              f'- Calculated **{N_BOOT:,} paired calendar-month / nested publication-issue bootstrap replicates** per title, keeping all pages of an issue together. Reported percentile 95% intervals are descriptive and **not multiple-comparison-adjusted**.',
              '- Independently recomputed point effects after excluding page 1, to probe cover-page sensitivity (not a validated masthead removal).',
              '', '## Main observed results', '',
              '| Newspaper | 1885 eligible pages/issues | 1888 eligible pages/issues | Change sign flips, any alternative | Word-mass supported patterns | Support lost for some alternative | Strong opposite-support conflicts |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for src in SOURCES:
        s=summ[src]
        result.append(f"| {src} | {s['year_1885']['eligible']}/{s['year_1885']['issues']} | {s['year_1888']['eligible']}/{s['year_1888']['issues']} | **{s['sign_reversal_any']}/{FEATURES_K}** | {s['supported_word_mass']}/{FEATURES_K} | {s['support_loss_any']}/{FEATURES_K} | {s['strong_conflict_any']}/{FEATURES_K} |")
    result += ['',f"The two newspaper titles have opposite **word-mass change directions** for **{summ['cross_source_word_mass_sign_disagreement']}/{FEATURES_K}** preselected patterns. This is observational disagreement between two newspaper corpora from one archive, not independent validation of a historical linguistic trajectory.", '',
               '## Concrete sensitivity witnesses', '',
               'The witnesses below are chosen automatically by the preregistered stability categories; **four-character sequences are not translated words or semantic concepts**.', '']
    for src in SOURCES:
        result += [f'### {src}', '', '| Pattern | Word-mass year change | Equal-page change | Equal-issue change | Equal-month change | Classification |', '|---|---:|---:|---:|---:|---|']
        for rec in choose_witnesses(audit['claims'], src):
            cats=[]
            if rec['strong_conflict_any']: cats.append('opposing supported signs')
            elif rec['sign_reversal_any']: cats.append('sign reversal')
            if rec['support_loss_any']: cats.append('support loss')
            if not cats: cats.append('no sign reversal')
            result.append(f"| `{rec['pattern']}` | {rec['word_mass_delta']:+.2f} | {rec['page_equal_delta']:+.2f} | {rec['issue_equal_delta']:+.2f} | {rec['month_equal_delta']:+.2f} | {'; '.join(cats)} |")
        result.append('')
    result += ['## Additional diagnostics', '']
    for src in SOURCES:
        s=summ[src]
        result.append(f"- **{src}:** excluding first pages changed the word-mass direction of **{s['cover_page_word_mass_sign_change']}/{FEATURES_K}** candidate patterns. There were **{s['year_1885']['excluded_short']}** excluded short views in 1885 and **{s['year_1888']['excluded_short']}** in 1888. Common calendar months: {len(s['common_months'])}/12.")
    result += ['', '## Responsible NLP interpretation', '',
               'The estimand choices describe **different exposure distributions**, not competing estimates of one uniquely correct societal fact. A conclusion that changes direction or loses conditional bootstrap support under reasonable exposure choices should be reported with the sensitivity, not elevated to a historical claim without further validation. Stability does not by itself establish historical semantic truth or social fairness.', '',
               'This instrument provides a **reproducible claim-sensitivity warning**, not a language-identity detector, cultural-change detector, verified topic model, moral fairness score, or general solution to biased historical digitisation.', '',
               '### Methodological limits', '',
               '- The original text is corrected transcription; page views are not human-adjudicated languages or genres.',
               '- The years 1885 and 1888 had been explored in earlier phases: this is a declared **post-hoc conditional** follow-up.',
               '- 120 correlated string patterns are tested; bootstrap intervals are **unadjusted**, not 120 independent discoveries.',
               '- The month/issue bootstrap measures conditional archive variability, not uncertainty about speakers, societies or missing publications.',
               '- Source and year confound genre, editorial changes, preservation, mastheads and OCR/transcription characteristics.',
               '- Two titles originate in the **same IsiXIT archive**; there is **no new independent modern or external validation**.',
               '', '## Reproduction and artefacts', '',
               '```bash',
               'python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md',
               'python src/archivelens/audit.py --data data/TxtLangFiles1870-1894_TB.zip --out results/archivelens --figures results/figures --report reports/ARCHIVELENS_RESULTS_2026-10-26.md --verify',
               '```', '',
               'Machine-auditable output: `results/archivelens/audit.json`, `claims.csv`, `estimands.csv`, `candidate_1887.csv`, `explorer.html`, and `results/figures/archivelens_claim_sensitivity.png` (plus vector `.svg`).', '',
               'Original precommit: `reports/ARCHIVELENS_PRECOMMIT_2026-10-26.md`. Prior art/background: `reports/related_work_and_archivelens_proposal_2026-10-26.md`.', '']
    return '\n'.join(result)


def make_html(audit: dict):
    encoded=json.dumps({'summary': audit['summary'], 'claims': audit['claims'],
                        'metadata':audit['metadata']},ensure_ascii=True,separators=(',', ':')).replace('</', '<\\/')
    head='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ArchiveLens · Historical NLP Audit</title><style>
:root{--bg:#f8fafc;--ink:#152238;--muted:#526177;--border:#d7e2ed;--deep:#124363;--card:#fff;--warning:#80510a}
*{box-sizing:border-box}body{background:var(--bg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--ink);margin:0}header{background:var(--deep);color:#fff;padding:27px max(20px,calc((100vw - 1100px)/2))}h1{margin:0;font-size:27px;letter-spacing:-.5px}header p{margin:5px 0 0;color:#e2f2fa}main{max-width:1100px;margin:24px auto;padding:0 16px 50px}section{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;margin-bottom:16px}h2{margin:0 0 12px;font-size:19px}p{margin:8px 0 12px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{padding:14px;background:#f0f5fb;border-radius:8px}.metric strong{display:block;font-size:24px}.metric span{color:var(--muted);font-size:12px}.filters{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}select,input{font:inherit;padding:8px;border:1px solid #9bb3c8;border-radius:6px;background:#fff;color:var(--ink)}input{min-width:175px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}th{background:#f2f7fb;white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}.pill{display:inline-block;padding:2px 7px;border-radius:12px;background:#ebeff4;font-size:11px}.warn{color:var(--warning);font-weight:600}.footer{font-size:12px;color:var(--muted)}@media(max-width:700px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}table{font-size:11px}}
</style></head><body><header><h1>ArchiveLens</h1><p>Counterfactual claim-sensitivity auditor · Historical isiXhosa newspapers · 1885–1888</p></header><main>
<section><h2>Interpretation guardrail</h2><p>This tool tests whether changes in <b>surface character patterns</b> depend on the unit of archival sampling. It does not measure language identity, cultural meaning, human sentiment, or social fairness. Year comparisons were previously explored; uncertainty intervals are descriptive and unadjusted for multiple comparisons.</p></section>
<section><div class="filters"><label>Newspaper <select id="source"><option value="IMVO">Imvo Zabantsundu (IMVO)</option><option value="ISIG">Isigidimi samaXosa (ISIG)</option></select></label></div><div id="metrics" class="metrics" aria-live="polite"></div></section>
<section><h2>Inspect the 120 preselected patterns</h2><p>Change is 1888 minus 1885, per 10,000 within-word four-character positions. A positive value means a pattern was more frequent in the later year's corrected XhTxt view. These are fragments, not translated words.</p><div class="filters"><label>Filter <select id="filter"><option value="all">All 120</option><option value="reversal">Sign reversal</option><option value="support">Support loss</option><option value="strong">Opposing supported signs</option><option value="stable">Supported under every estimand</option></select></label><label>Search pattern <input id="search" type="search" placeholder="e.g. ntun" maxlength="4"></label></div><p id="count" class="footer" aria-live="polite"></p><div class="scroll"><table><thead><tr><th>Pattern</th><th>Word mass Δ</th><th>95% interval</th><th>Equal page Δ</th><th>Equal issue Δ</th><th>Equal month Δ</th><th>Flag</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>Methods and cautions</h2><p>Patterns selected using 1887 only, balancing the two newspaper sources. Both years were compared on exactly the same eligible corrected XhTxt page views under four estimands. Percentile intervals use 3,000 paired month-block resamples with publication issues clustered within months.</p><p>Limitations: historical source and editorial/genre confounding; unknown raw-OCR quality; the archive's XhTxt view is not a gold-standard language label; no multiple-testing adjustment; no independently held-out external dataset.</p></section><p class="footer">Source: IsiXIT, course-supplied ZIP; analysis precommit and code in the project's GitHub repository. No original full text is embedded in this viewer.</p></main><script>const data='''
    tail=''';const S=document.getElementById('source'),F=document.getElementById('filter'),Q=document.getElementById('search'),R=document.getElementById('rows'),M=document.getElementById('metrics'),C=document.getElementById('count');const f=(v)=>Number(v).toFixed(2);function render(){const src=S.value,s=data.summary[src];M.innerHTML=[['Candidates',s.candidate_patterns],['Any sign reversal',s.sign_reversal_any],['Support lost',s.support_loss_any],['Strong opposite-support',s.strong_conflict_any]].map(([k,v])=>`<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join('');let arr=data.claims.filter(r=>r.source===src&&r.pattern.includes(Q.value.toLowerCase()));if(F.value==='reversal')arr=arr.filter(r=>r.sign_reversal_any);if(F.value==='support')arr=arr.filter(r=>r.support_loss_any);if(F.value==='strong')arr=arr.filter(r=>r.strong_conflict_any);if(F.value==='stable')arr=arr.filter(r=>r.stable_supported_all);C.textContent=`Showing ${arr.length} of 120 patterns · 1887 rank order`;R.replaceChildren();for(const r of arr){const tr=document.createElement('tr');const labels=r.strong_conflict_any?'Opposing support':r.sign_reversal_any?'Reversal':r.support_loss_any?'Support loss':r.stable_supported_all?'Stable support':'No qualifying flag';const nums=[r.pattern,f(r.word_mass_delta),`[${f(r.word_mass_ci_low)}, ${f(r.word_mass_ci_high)}]`,f(r.page_equal_delta),f(r.issue_equal_delta),f(r.month_equal_delta),labels];nums.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i>0&&i<6)td.className='num';if(i===6&&labels!=='No qualifying flag')td.className='warn';tr.appendChild(td)});R.appendChild(tr)}}[S,F,Q].forEach(el=>el.addEventListener('input',render));render();</script></body></html>'''
    return head + encoded + tail


def make_figure(audit: dict, prefix: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by={(r['source'],r['pattern']):r for r in audit['claims']}
    fig, axs = plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
    for src,mark in (('IMVO','o'),('ISIG','^')):
        rr=[r for r in audit['claims'] if r['source']==src]
        x=[r['word_mass_delta'] for r in rr]
        y=[r['page_equal_delta'] for r in rr]
        axs[0].scatter(x,y,alpha=.64,s=20 if src=='IMVO' else 26,marker=mark,label=src)
    lim = max(abs(k) for rr in audit['claims'] for k in (rr['word_mass_delta'],rr['page_equal_delta']))*1.05
    axs[0].plot([-lim,lim],[-lim,lim],ls=':',linewidth=1,color='grey')
    axs[0].axhline(0,color='grey',ls='--',linewidth=.7)
    axs[0].axvline(0,color='grey',ls='--',linewidth=.7)
    axs[0].set_xlabel('1888−1885 change, character-mass weighting')
    axs[0].set_ylabel('1888−1885 change, equal-page weighting')
    axs[0].set_title('A. Do sampling units change the direction?')
    axs[0].legend(frameon=False)
    witnesses=[]
    for src in SOURCES:
        witnesses.extend(choose_witnesses(audit['claims'],src,n=2))
    rows={(r['source'],r['pattern'],r['estimand']):r for r in audit['estimands']}
    ys=[];labs=[]
    for j,claim in enumerate(witnesses):
        src,pat=claim['source'],claim['pattern']
        for i,est in enumerate(ESTIMANDS):
            row=rows[(src,pat,est)]
            v=row['delta']; low=row['ci_low'];high=row['ci_high']
            yy=j*5+(3-i)
            axs[1].errorbar(v,yy,xerr=[[max(0,v-low)],[max(0,high-v)]],fmt='o',capsize=2,markersize=4)
            ys.append(yy);labs.append(f'{src}  {pat} · {est.replace("_"," ")}')
    axs[1].set_yticks(ys,labs,fontsize=7)
    axs[1].axvline(0,color='grey',ls='--',linewidth=.7)
    axs[1].set_xlabel('1888−1885 change (95% month/issue bootstrap interval)')
    axs[1].set_title('B. Automatically selected sensitivity examples')
    fig.suptitle('ArchiveLens | historical surface-form claim sensitivity (per 10,000 4-gram positions)',fontsize=12)
    fig.savefig(prefix.with_suffix('.png'),dpi=190,bbox_inches='tight')
    fig.savefig(prefix.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':'2026-10-26'})
    plt.close(fig)


def serialize(audit: dict):
    packed={k:audit[k] for k in ('metadata','selection_1887','summary','claims')}
    return json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=False)+'\n'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('results/archivelens'))
    ap.add_argument('--figures',type=Path,default=Path('results/figures'))
    ap.add_argument('--report',type=Path,default=Path('reports/ARCHIVELENS_RESULTS_2026-10-26.md'))
    ap.add_argument('--verify',action='store_true',help='Recompute all results; compare byte-for-byte without overwriting')
    args=ap.parse_args()
    audit=perform(args.data)
    contents={args.out/'audit.json':serialize(audit),
              args.out/'candidate_1887.csv':dump_csv(audit['selection_1887']),
              args.out/'claims.csv':dump_csv(audit['claims']),
              args.out/'estimands.csv':dump_csv(audit['estimands']),
              args.out/'explorer.html':make_html(audit),
              args.report:make_markdown(audit)}
    if args.verify:
        for path,expect in contents.items():
            if not path.exists() or path.read_text(encoding='utf-8') != expect:
                raise AssertionError('Research artifact does not match fresh computation: '+str(path))
        for ext in ('.png','.svg'):
            if not (args.figures/('archivelens_claim_sensitivity'+ext)).is_file():
                raise AssertionError('Missing paper figure '+ext)
        print('PASS: all six ArchiveLens source-dependent research artifacts match fresh recomputation')
    else:
        for path,content in contents.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding='utf-8')
        args.figures.mkdir(parents=True,exist_ok=True)
        make_figure(audit,args.figures/'archivelens_claim_sensitivity')
    for src in SOURCES:
        print(src, json.dumps(audit['summary'][src],sort_keys=True))
    print('Cross-newspaper opposing temporal directions:',audit['summary']['cross_source_word_mass_sign_disagreement'],'of',FEATURES_K)
    print('PASS: exact original ZIP, fixed 1887 candidates, all four estimands, paired month/issue bootstrap and diagnostics')


if __name__=='__main__':
    main()