#!/usr/bin/env python3
"""ArchiveLens calibration and archival-structure attribution follow-up.

Post-hoc conditional research; see ARCHIVELENS_FOLLOWUP_PRECOMMIT_2026-10-26.md.
Does not modify the original ArchiveLens outputs or tokenizer experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

try:
    from archivelens.audit import (
        EXPECTED_SHA256, ESTIMANDS, FEATURES_K, MIN_WORDS, RE_WORD, SOURCES,
        SCALE, Page, Units, estimates, make_units, names_in_archive, sha256,
    )
except ModuleNotFoundError:
    from audit import (
        EXPECTED_SHA256, ESTIMANDS, FEATURES_K, MIN_WORDS, RE_WORD, SOURCES,
        SCALE, Page, Units, estimates, make_units, names_in_archive, sha256,
    )

PERMUTATIONS = 4999
RNG_SEED = 20261027
SHINGLE_N = 13
DATE = '2026-10-26'


def adjust_pvalues(pvals: np.ndarray, method: str) -> np.ndarray:
    """Exact deterministic BH FDR or Holm FWER corrections, including ties."""
    p = np.asarray(pvals, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('p-values must be a finite vector in [0,1]')
    n = len(p)
    order = np.argsort(p, kind='stable')
    sp = p[order]
    if method == 'bh':
        adj = np.minimum.accumulate((sp * n / np.arange(1, n + 1))[::-1])[::-1]
    elif method == 'holm':
        adj = np.maximum.accumulate(sp * np.arange(n, 0, -1))
    else:
        raise ValueError(method)
    ret = np.empty(n, dtype=float)
    ret[order] = np.clip(adj, 0, 1)
    return ret


def sign_reversal(delta: np.ndarray) -> np.ndarray:
    """One flag per pattern for changes in direction across four estimands."""
    base = delta[0]
    return np.any((base[None, :] * delta[1:]) < 0, axis=0)


def sensitivity_gap(delta: np.ndarray) -> np.ndarray:
    """Absolute maximum difference from character-mass across other estimands."""
    return np.max(np.abs(delta[1:] - delta[0][None, :]), axis=0)


def slice_units(u: Units, idx: np.ndarray) -> Units:
    idx = np.asarray(idx, dtype=np.int64)
    if not len(idx):
        raise ValueError('Cannot take empty issue support')
    mm = u.months[idx]
    return Units(u.source, u.year, [u.ids[int(i)] for i in idx], mm,
                 u.counts[idx], u.positions[idx], u.page_rate_sums[idx],
                 u.n_pages[idx], u.rates[idx],
                 {int(m): np.flatnonzero(mm == m) for m in np.unique(mm)})


def paired_permutation(u0: Units, u1: Units, rng: np.random.Generator) -> tuple[Units, Units]:
    """Shuffle whole ISSUES within common month, keeping original monthly group sizes."""
    if set(u0.month_indices) != set(u1.month_indices):
        raise ValueError('Monthly support must match')
    base = Units(u0.source, -1, u0.ids + u1.ids,
                 np.concatenate([u0.months, u1.months]),
                 np.concatenate([u0.counts, u1.counts], axis=0),
                 np.concatenate([u0.positions, u1.positions]),
                 np.concatenate([u0.page_rate_sums, u1.page_rate_sums], axis=0),
                 np.concatenate([u0.n_pages, u1.n_pages]),
                 np.concatenate([u0.rates, u1.rates], axis=0), {})
    ix0, ix1 = [], []
    offset = len(u0.ids)
    for m in sorted(u0.month_indices):
        old = np.concatenate([u0.month_indices[m], u1.month_indices[m] + offset])
        perm = rng.permutation(old)
        n0 = len(u0.month_indices[m])
        ix0.extend(perm[:n0].tolist())
        ix1.extend(perm[n0:].tolist())
    return slice_units(base, np.asarray(ix0)), slice_units(base, np.asarray(ix1))


def run_permutations(u0: Units, u1: Units, n: int, seed: int) -> dict:
    months = sorted(u0.month_indices)
    obs = estimates(u1, months) - estimates(u0, months)
    obs_abs = np.abs(obs[0])
    obs_gap = sensitivity_gap(obs)
    actual_flags = int(sign_reversal(obs).sum())
    rng = np.random.default_rng(seed)
    exceed_year = np.zeros(obs.shape[1], dtype=np.int64)
    exceed_gap = np.zeros(obs.shape[1], dtype=np.int64)
    null_sign_counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        a, b = paired_permutation(u0, u1, rng)
        null_delta = estimates(b, months) - estimates(a, months)
        exceed_year += np.abs(null_delta[0]) >= obs_abs - 1e-12
        exceed_gap += sensitivity_gap(null_delta) >= obs_gap - 1e-12
        null_sign_counts[i] = int(sign_reversal(null_delta).sum())
    p_year = (1 + exceed_year) / (n + 1)
    p_gap = (1 + exceed_gap) / (n + 1)
    rows = {}
    for group, p in (('temporal_mass', p_year), ('estimand_gap', p_gap)):
        rows[group] = {'p': p, 'bh': adjust_pvalues(p, 'bh'),
                       'holm': adjust_pvalues(p, 'holm')}
    null_info = {'observed': actual_flags,
                 'null_mean': float(null_sign_counts.mean()),
                 'null_median': float(np.median(null_sign_counts)),
                 'null_95_lo': float(np.quantile(null_sign_counts, .025)),
                 'null_95_hi': float(np.quantile(null_sign_counts, .975)),
                 'p_upper_tail': float((1 + (null_sign_counts >= actual_flags).sum()) / (n + 1)),
                 'p_lower_tail': float((1 + (null_sign_counts <= actual_flags).sum()) / (n + 1)),
                 'null_min': int(null_sign_counts.min()),
                 'null_max': int(null_sign_counts.max())}
    return {'observed': obs, 'observed_gap': obs_gap,
            'p': rows, 'counts': null_info}


def leave_one_out(u0: Units, u1: Units) -> tuple[np.ndarray, list[dict]]:
    months = sorted(u0.month_indices)
    original = estimates(u1, months) - estimates(u0, months)
    records=[]
    for year, u in ((1885, u0), (1888, u1)):
        for i, issue_id in enumerate(u.ids):
            kept = np.asarray([j for j in range(len(u.ids)) if j != i], dtype=np.int64)
            if not len(kept):
                continue
            sub = slice_units(u, kept)
            # The three non-month estimands are still defined if month support shrinks.
            if len(sub.month_indices) != len(months):
                sampled = estimates_ignoring_missing_months(sub)
            else:
                sampled = estimates(sub, months)
            d = (estimates(u1, months) - sampled) if year == 1885 else (sampled - estimates(u0, months))
            # d has shape 4 x K; set month-only to nan when month is dropped.
            change = d - original
            records.append({'year': year, 'issue': issue_id, 'issue_index':i,
                            'source':u.source, 'page_count':int(u.n_pages[i]),
                            'share_of_year_char_positions': float(u.positions[i]/u.positions.sum()),
                            'deltas':d, 'change':change})
    return original, records


def estimates_ignoring_missing_months(u: Units) -> np.ndarray:
    char_mass = u.counts.sum(axis=0)/u.positions.sum()
    page = u.page_rate_sums.sum(axis=0)/u.n_pages.sum()
    issue = u.rates.mean(axis=0)
    month = np.full(char_mass.shape, np.nan)
    return SCALE*np.stack((char_mass, page, issue, month))


def issue_influence_rows(u0: Units, u1: Units, selected: list[str], flagged: set[str]):
    original, loo = leave_one_out(u0, u1)
    main_rows=[]; top_rows=[]
    for j, pat in enumerate(selected):
        if pat not in flagged:
            continue
        shifts = sorted(loo, key=lambda row:(-abs(float(row['change'][0,j])), row['issue']))
        mass_flips=sum(np.isfinite(row['deltas'][0,j]) and
                       original[0,j]*row['deltas'][0,j] < 0 for row in loo)
        page_flips=sum(np.isfinite(row['deltas'][1,j]) and
                       original[1,j]*row['deltas'][1,j] < 0 for row in loo)
        max_page = min(loo, key=lambda row:(-abs(float(row['change'][1,j])),row['issue']))
        max_mass = shifts[0]
        main_rows.append({'source':u0.source,'pattern':pat,
                          'n_issues_tested':len(loo),
                          'mass_baseline_delta':float(original[0,j]),
                          'page_baseline_delta':float(original[1,j]),
                          'loo_mass_sign_flip_n':int(mass_flips),
                          'loo_page_sign_flip_n':int(page_flips),
                          'max_abs_mass_shift':abs(float(max_mass['change'][0,j])),
                          'max_mass_shift_issue':max_mass['issue'],
                          'max_mass_shift_issue_year':max_mass['year'],
                          'max_mass_shift_sign_flip': bool(original[0,j]*max_mass['deltas'][0,j]<0),
                          'max_abs_page_shift':abs(float(max_page['change'][1,j])),
                          'max_page_shift_issue':max_page['issue'],
                          'max_page_shift_issue_year':max_page['year']})
        for rank,r in enumerate(shifts[:3],1):
            top_rows.append({'source':u0.source,'pattern':pat,'rank':rank,'issue':r['issue'],
                             'year':r['year'],'pages_in_issue':r['page_count'],
                             'issue_share_of_year_positions':r['share_of_year_char_positions'],
                             'mass_effect_change':float(r['change'][0,j]),
                             'mass_effect_without_issue':float(r['deltas'][0,j]),
                             'page_effect_change':float(r['change'][1,j])})
    return main_rows,top_rows


def shingle_digest(words: list[str], at: int, width: int = SHINGLE_N) -> bytes:
    # Length-separated text to prevent ambiguity; BLAKE2b-128 makes accidental collision negligible.
    return hashlib.blake2b(' '.join(words[at:at+width]).encode('utf-8'), digest_size=16).digest()


@dataclass
class SourceWords:
    row: Page
    words: list[str]


def collect_words(original_zip: Path, selected: list[str]):
    # Independently recounting pages would be much more costly, but keeps full text
    # available only in memory for repeated-passage verification.
    out=defaultdict(list)
    with ZipFile(original_zip) as z:
        for name,src,year,month,day,page_no in names_in_archive(z):
            if year not in (1885, 1888):
                continue
            words=RE_WORD.findall(z.read(name).decode('utf-8',errors='replace').lower())
            if len(words)<MIN_WORDS:
                continue
            denom=sum(max(0,len(w)-3) for w in words)
            if denom==0:
                continue
            page=Page(src,year,month,f'{src}_{year}-{month:02}-{day:02}',page_no,len(words),denom,
                      np.zeros(len(selected),dtype=np.int64))
            out[src].append(SourceWords(page,words))
    return out


def discover_repetitions(rows: list[SourceWords]) -> set[bytes]:
    seen={}
    repeat=set()
    for row in rows:
        words=row.words
        for i in range(len(words)-SHINGLE_N+1):
            key=shingle_digest(words,i)
            old=seen.get(key)
            if old is None:
                seen[key]=row.row.issue
            elif old!=row.row.issue:
                repeat.add(key)
    return repeat


def repeated_mask(words: list[str], repeated: set[bytes]) -> np.ndarray:
    n=len(words)
    if not repeated or n<SHINGLE_N:
        return np.zeros(n,dtype=bool)
    events=np.zeros(n+1,dtype=np.int32)
    for i in range(n-SHINGLE_N+1):
        if shingle_digest(words,i) in repeated:
            events[i]+=1
            events[i+SHINGLE_N]-=1
    return np.cumsum(events[:-1])>0


def recount_unmasked_words(row: SourceWords, mask: np.ndarray, selected: dict[str,int]) -> Page | None:
    if len(mask)!=len(row.words):
        raise ValueError('Wrong repeated text mask')
    keep=[w for w,flag in zip(row.words, mask) if not flag]
    positions=sum(max(0,len(w)-3) for w in keep)
    if len(keep)<MIN_WORDS or positions<=0:
        return None
    counts=np.zeros(len(selected),dtype=np.int64)
    for word in keep:
        for i in range(len(word)-3):
            ix=selected.get(word[i:i+4])
            if ix is not None:
                counts[ix]+=1
    p=row.row
    return Page(p.source,p.year,p.month,p.issue,p.page_no,len(keep),positions,counts)


def duplicates_sensitivity(original_zip: Path, selected: list[str], original:dict,
                           flagged: dict[str,set[str]]):
    extracted=collect_words(original_zip,selected)
    idx={s:i for i,s in enumerate(selected)}
    stats=[]; effects={}; page_position=[]; mask_audit=[]
    for src in SOURCES:
        rows=extracted[src]
        repeated=discover_repetitions(rows)
        filtered=defaultdict(list)
        counters=defaultdict(Counter)
        # Count 4gram opportunities and occurrences in the original, descriptive bins only.
        pos=defaultdict(lambda:{'pages':0,'words':0,'opportunities':0,
                                'candidate_counts':np.zeros(len(selected),dtype=np.int64)})
        for row in rows:
            p=row.row
            mask=repeated_mask(row.words,repeated)
            retained=recount_unmasked_words(row,mask,idx)
            n_removed=int(mask.sum())
            key=(src,p.year)
            counters[key]['original_words']+=len(row.words)
            counters[key]['masked_words']+=n_removed
            counters[key]['original_pages']+=1
            if retained is not None:
                filtered[p.year].append(retained)
                counters[key]['retained_pages']+=1
                counters[key]['retained_words']+=retained.n_words
            else:
                counters[key]['lost_pages']+=1
            pgrp=p.page_no if p.page_no<5 else 5
            g=pos[(p.year,pgrp)]
            g['pages']+=1;g['words']+=len(row.words);g['opportunities']+=p.denom
            # Recount selected within-word 4grams independently of already existing score arrays
            for w in row.words:
                for k in range(len(w)-3):
                    j=idx.get(w[k:k+4])
                    if j is not None:g['candidate_counts'][j]+=1
            mask_audit.append({'source':src,'year':p.year,'issue':p.issue,
                               'page_no':p.page_no,'total_words':len(row.words),
                               'masked_words':n_removed,'remaining_words':len(row.words)-n_removed,
                               'retained_after_mask':int(retained is not None)})
        for year in (1885,1888):
            c=counters[(src,year)]
            stats.append({'source':src,'year':year,'exact_repeated_13gram_types_in_source':len(repeated),
                          **dict(c),'masked_word_fraction':c['masked_words']/c['original_words'],
                          'retained_issues':len(set(p.issue for p in filtered[year]))})
            for grp in range(1,6):
                obj=pos.get((year,grp),{'pages':0,'words':0,'opportunities':0,
                                      'candidate_counts':np.zeros(len(selected),dtype=np.int64)})
                page_position.append({'source':src,'year':year,
                                      'page_position':f'p{grp:03}' if grp<5 else 'p005_plus',
                                      'pages':obj['pages'],'words':obj['words'],
                                      'within_word_fourgram_positions':obj['opportunities']})
        if any(not filtered[yr] for yr in (1885,1888)):
            raise AssertionError('Duplicate-filtered archive contains no eligible pages')
        u0=make_units(src,1885,filtered[1885]);u1=make_units(src,1888,filtered[1888])
        common=sorted(set(u0.month_indices)&set(u1.month_indices))
        if not common:raise AssertionError('No common month after exact-duplicate masking')
        u0=make_units(src,1885,filtered[1885],set(common))
        u1=make_units(src,1888,filtered[1888],set(common))
        dd=estimates(u1,common)-estimates(u0,common)
        for j,pat in enumerate(selected):
            base=original[(src,pat)]
            original_delta=np.array([base[e+'_delta'] for e in ESTIMANDS])
            had_reversal=bool(base['sign_reversal_any'])
            has_reversal=bool(sign_reversal(dd[:,j:j+1])[0])
            effects[(src,pat)]={'original_mass_delta':float(original_delta[0]),
                                'duplicate_masked_mass_delta':float(dd[0,j]),
                                'original_page_delta':float(original_delta[1]),
                                'duplicate_masked_page_delta':float(dd[1,j]),
                                'mass_direction_changed':bool(original_delta[0]*dd[0,j]<0),
                                'original_sign_reversal':had_reversal,
                                'masked_sign_reversal':has_reversal,
                                'reversal_warning_disappeared':bool(had_reversal and not has_reversal),
                                'flagged_original':pat in flagged[src]}
        # A detailed page-position diagnostic for flagged patterns: contribution
        # to year-specific pattern counts, not an inference about article genre.
        for year in (1885,1888):
            total=np.sum([pos[(year,g)]['candidate_counts'] for g in range(1,6)],axis=0)
            for pat in selected:
                if pat not in flagged[src]:continue
                j=idx[pat]
                for g in range(1,6):
                    obj=pos[(year,g)]
                    count=int(obj['candidate_counts'][j]);whole=int(total[j])
                    page_position.append({'source':src,'year':year,'pattern':pat,
                                          'page_position':f'p{g:03}' if g<5 else 'p005_plus',
                                          'pages':obj['pages'],'words':obj['words'],
                                          'within_word_fourgram_positions':obj['opportunities'],
                                          'pattern_occurrences':count,
                                          'share_of_pattern_occurrences':count/whole if whole else 0})
    effect_rows=[{'source':src,'pattern':pat,**stats} for (src,pat),stats in effects.items()]
    return effect_rows,stats_from_duplicate(stats), page_position, mask_audit


def stats_from_duplicate(rows):
    # Keep the original four source-year summaries in reproducible order.
    return sorted(rows, key=lambda r:(r['source'],r['year']))


def _float(v):
    return float(v) if isinstance(v,np.floating) else v


def fmt_float(v):
    return f'{v:.10f}' if isinstance(v,(float,np.floating)) else v


def csv_output(rows: list[dict]) -> str:
    import io
    if not rows:return ''
    buffer=io.StringIO(newline='')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(buffer,fieldnames=keys,lineterminator='\n')
    writer.writeheader()
    for r in rows:
        writer.writerow({k:fmt_float(v) for k,v in r.items()})
    return buffer.getvalue()


def make_figure(calibration, attribution, stats, figures:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(1,2,figsize=(11.7,4.8),layout='constrained')
    for i,src in enumerate(SOURCES):
        x=calibration[src]['counts']
        ax[0].bar(i-.15,x['observed'],width=.3,label='Observed' if i==0 else None)
        ax[0].bar(i+.15,x['null_mean'],width=.3,label='Year-label permutation mean' if i==0 else None)
        ax[0].plot([i+.15,i+.15],[x['null_95_lo'],x['null_95_hi']],color='black',lw=1.5)
    ax[0].set_xticks([0,1],['Imvo','Isigidimi'])
    ax[0].set_ylabel('4-gram sign-reversal warnings (of 120)')
    ax[0].set_title('A. Are warning counts unusual?')
    ax[0].legend(frameon=False,fontsize=8)
    labels=[]; vals=[]
    for src in SOURCES:
        sub=[r for r in attribution if r['source']==src]
        if not sub:continue
        labels.append(src+' | flagged patterns')
        vals.append(sum(r['loo_mass_sign_flip_n']>0 for r in sub))
    ax[1].barh(labels,vals)
    ax[1].set_xlabel('Flagged patterns whose mass trend flips\nwhen one issue is removed')
    ax[1].set_title('B. Publication-issue influence')
    ax[1].invert_yaxis()
    fig.suptitle('ArchiveLens | Calibration and attribution (conditional historical surface patterns)',fontsize=12)
    base=figures/'archivelens_followup_calibration_attribution'
    fig.savefig(base.with_suffix('.png'),dpi=190,bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':DATE})
    plt.close(fig)


def make_report(meta,calibration,attr,dup,cover):
    a=['# ArchiveLens: calibration and attribution follow-up',
       '', '**26 October 2026 | Post-hoc conditional follow-up to an already-inspected historical archive.**',
       '', '## Research questions', '',
       '1. Are ArchiveLens surface-pattern direction warnings unusual under a month-stratified publication-issue permutation benchmark, with multiplicity adjustment?',
       '2. Are warnings sensitive to individual issues, exact repeated passages or page-position exposure?',
       '', '## Experiment 1 — Permutation calibration', '',
       'Exactly 4,999 within-calendar-month label permutations preserve whole publication issues and year/month issue counts. Pattern selection remains fixed from 1887, and reference years are 1885/1888. Both the year-effect test and the maximum sampling-estimand gap are corrected across 120 patterns per title with Holm FWER and BH FDR. The previously reported 3,000-replicate bootstraps are unchanged and unadjusted.',
       '', '| Newspaper | Original sign warnings | Mean null warnings [central 95%] | Upper-tail p | Lower-tail p | Holm year effects | BH year effects | Holm sensitivity gaps | BH sensitivity gaps |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for src in SOURCES:
        c=calibration[src]['counts'];s=calibration[src]['multiplicity']
        a.append(f"| {src} | **{c['observed']}/120** | {c['null_mean']:.2f} [{c['null_95_lo']:.0f},{c['null_95_hi']:.0f}] | {c['p_upper_tail']:.4f} | {c['p_lower_tail']:.4f} | {s['holm_year']} | {s['bh_year']} | {s['holm_gap']} | {s['bh_gap']} |")
    a += ['', 'The permutation distribution is a **reference benchmark**, not an unconditional estimate of false-alarm rate. Exchanging historical year labels can break real temporal structure, and 120 four-grams are correlated. The count test asks if the *number* of point-sign flags is unusually high or low; it does **not** calibrate all existing bootstrap support-loss warnings.',
          '', '## Experiment 2 — Issue, exact repetition and page-position attribution', '',
          '| Newspaper | Original flagged union | Flags with baseline year-effect sign changed by removing one issue | Exact 13-gram repeated-word share 1885 / 1888 | Flagged mass signs changed after masking repeated passages | Original reversal warnings disappearing after masking |',
          '|---|---:|---:|---:|---:|---:|']
    for src in SOURCES:
        sub=[r for r in attr if r['source']==src]
        d=[r for r in dup if r['source']==src and r['flagged_original']]
        s0=cover[(src,1885)];s1=cover[(src,1888)]
        a.append(f"| {src} | **{len(sub)}** | **{sum(r['loo_mass_sign_flip_n']>0 for r in sub)}** | {s0['masked_word_fraction']:.1%} / {s1['masked_word_fraction']:.1%} | {sum(r['mass_direction_changed'] for r in d)} | {sum(r['reversal_warning_disappeared'] for r in d)} |")
    a += ['', 'Leave-one-out calculations remove entire publication issues and report the maximal shift per warning; these identify influential *sampling units*, not historical causes. Exact repeated-passage masking removes words covered by normalized 13-word sequences recurring across distinct issues **within the same newspaper**, then reruns all four estimands on the retained eligible views. Masked-text results change the eligible sample if pages fall below 40 remaining words, so they are a **sensitivity test**, not a controlled causal effect of duplication.', '',
          'Page-position contribution tables use page numbers 1–4 and >=5. Position alone is not an annotated article type, newspaper topic, editor, OCR fidelity or masthead detection.', '',
          '## Responsible NLP conclusion', '',
          'ArchiveLens can expose when surface-pattern trends are sensitive to sampling unit, and its warnings can be checked against an explicit randomisation reference and traced to influential archive structure. A warning is **not** a linguistic or cultural discovery, a moral fairness score, or proof that a year trend is spurious. Multiple-comparison-adjusted permutation findings and unsupported results are reported rather than cherry-picked.', '',
          '## Limitations', '',
          '- The 1885/1888 source was examined repeatedly before this follow-up; there is no fresh external independent holdout.',
          '- These patterns are within-word four-character fragments, not validated morphemes, sentiment, named entities or historical ideas.',
          '- The two newspapers are in the same archive; year and newspaper source can co-vary with genre, preservation and editorial practices.',
          '- Synthetic controls and permutation diagnostics do not guarantee calibration under arbitrary historical domain shifts.',
          '- Holm controls a family of 120 tests per title and family under the reference null, but different null families were separately adjusted; BH statements require dependence caution.',
          '- Existing bootstrap support-loss warnings are descriptive, not covered by the new permutation benchmark.',
          '- Exact 13-token repeat masking is incomplete for paraphrases, short boilerplate, formatting artifacts and cross-source repeats.',
          '', '## Reproduce', '', '```bash',
          'python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root .',
          'python src/archivelens/followup.py --data data/TxtLangFiles1870-1894_TB.zip --root . --verify',
          '```', '', 'The main evidence is in `results/archivelens/followup/`: `calibration.csv`, `attribution.csv`, `influential_issues.csv`, `duplicate_sensitivity.csv`, `duplicate_coverage.csv`, `page_positions.csv`, `summary.json`, plus figure outputs. The old ArchiveLens result files remain unchanged.', '']
    return '\n'.join(a)


def run(data:Path,root:Path,n_perm:int=PERMUTATIONS):
    if sha256(data) != EXPECTED_SHA256:
        raise AssertionError('Original archival ZIP changed; refusing to score')
    old=json.loads((root/'results/archivelens/audit.json').read_text(encoding='utf-8'))
    claims={(r['source'],r['pattern']):r for r in old['claims']}
    selected=[r['pattern'] for r in old['selection_1887']]
    assert len(selected)==FEATURES_K==len(set(selected))
    # Load without repeating 3000 original bootstrap draws; this is a new experiment.
    from zipfile import ZipFile
    with ZipFile(data) as z:
        names=names_in_archive(z)
        from_arch=defaultdict(list)
        ix={s:i for i,s in enumerate(selected)}
        for name,src,year,month,day,page_no in names:
            if year not in (1885,1888):continue
            words=RE_WORD.findall(z.read(name).decode('utf-8',errors='replace').lower())
            if len(words)<MIN_WORDS:continue
            den=sum(max(0,len(w)-3) for w in words)
            if den<=0:continue
            cc=np.zeros(len(selected),dtype=np.int64)
            for w in words:
                for k in range(len(w)-3):
                    i=ix.get(w[k:k+4])
                    if i is not None:cc[i]+=1
            from_arch[(src,year)].append(Page(src,year,month,f'{src}_{year}-{month:02}-{day:02}',page_no,len(words),den,cc))
    calibration={};cal_rows=[];attr_rows=[];top_rows=[]
    for k,src in enumerate(SOURCES):
        initial={yr:make_units(src,yr,from_arch[(src,yr)]) for yr in (1885,1888)}
        months=sorted(set(initial[1885].month_indices)&set(initial[1888].month_indices))
        if len(months)!=12:raise AssertionError('Unexpected calendar-month support')
        u0=make_units(src,1885,from_arch[(src,1885)],set(months))
        u1=make_units(src,1888,from_arch[(src,1888)],set(months))
        permutation=run_permutations(u0,u1,n_perm,RNG_SEED+k)
        # Crosscheck old published four observed deltas to numerical tolerance.
        d=permutation['observed']
        for j,pat in enumerate(selected):
            oldrow=claims[(src,pat)]
            for m,est in enumerate(ESTIMANDS):
                if abs(float(d[m,j])-float(oldrow[est+'_delta']))>1.1e-8:
                    raise AssertionError('Mismatch with frozen ArchiveLens: '+src+'/'+pat+'/'+est)
            cal_rows.append({'source':src,'pattern':pat,'rank_1887':j+1,
                             'original_sign_reversal':bool(oldrow['sign_reversal_any']),
                             'original_support_loss':bool(oldrow['support_loss_any']),
                             'temporal_mass_delta':float(d[0,j]),
                             'max_estimand_gap':float(permutation['observed_gap'][j]),
                             'temporal_perm_p':float(permutation['p']['temporal_mass']['p'][j]),
                             'temporal_bh_q':float(permutation['p']['temporal_mass']['bh'][j]),
                             'temporal_holm_p':float(permutation['p']['temporal_mass']['holm'][j]),
                             'estimand_gap_perm_p':float(permutation['p']['estimand_gap']['p'][j]),
                             'estimand_gap_bh_q':float(permutation['p']['estimand_gap']['bh'][j]),
                             'estimand_gap_holm_p':float(permutation['p']['estimand_gap']['holm'][j])})
        cal=permutation['counts']
        cal['observed_matches_prior']=(cal['observed']==sum(bool(claims[(src,p)]['sign_reversal_any']) for p in selected))
        if not cal['observed_matches_prior']:raise AssertionError('Old warning count changed')
        multip={'bh_year':int((permutation['p']['temporal_mass']['bh']<=.05).sum()),
                'holm_year':int((permutation['p']['temporal_mass']['holm']<=.05).sum()),
                'bh_gap':int((permutation['p']['estimand_gap']['bh']<=.05).sum()),
                'holm_gap':int((permutation['p']['estimand_gap']['holm']<=.05).sum())}
        calibration[src]={'counts':cal,'multiplicity':multip}
        flagged={p for p in selected if claims[(src,p)]['sign_reversal_any'] or
                 claims[(src,p)]['support_loss_any']}
        a,b=issue_influence_rows(u0,u1,selected,flagged)
        attr_rows+=a;top_rows+=b
    flagged={src:{p for p in selected if claims[(src,p)]['sign_reversal_any'] or
                   claims[(src,p)]['support_loss_any']} for src in SOURCES}
    dupe_rows,coverage_rows,pos_rows,mask_rows=duplicates_sensitivity(data,selected,claims,flagged)
    provenance={'input_zip_sha256':EXPECTED_SHA256,'source_date':DATE,
                'original_analysis_sha256':hashlib.sha256((root/'results/archivelens/audit.json').read_bytes()).hexdigest(),
                'research_design':'post-hoc conditional follow-up; protocol frozen before these new results',
                'permutations':n_perm,'permutation_seed':RNG_SEED,
                'permutation_unit':'whole publication issue, within original calendar month, year-group sizes held fixed',
                'multiplicity':'Holm FWER and BH FDR, each applied separately across 120 patterns per source/permutation test family',
                'null_test_families':['abs(1888-1885 character-mass rate difference)',
                                      'max absolute difference in year change between mass and other estimands'],
                'global_null':'number of point-sign reversal flags across the 120 features',
                'repetition':'exact normalized 13-word shingles across distinct issue IDs within source and both evaluation years',
                'words_min_eligible_after_mask':MIN_WORDS,
                'frozen_features':selected,
                'no_historical_semantic_or_fairness_claim':True,
                'prior_archivelens_outputs_unchanged':True}
    stats_map={(r['source'],r['year']):r for r in coverage_rows}
    report=make_report(provenance,calibration,attr_rows,dupe_rows,stats_map)
    summary={'provenance':provenance,'calibration':calibration,
             'source_year_repeat_coverage':coverage_rows,
             'attribution_counts':{src:{'flagged_n':sum(r['source']==src for r in attr_rows),
                      'loo_mass_sign_change_patterns':sum(r['source']==src and r['loo_mass_sign_flip_n']>0 for r in attr_rows),
                      'duplicate_mask_mass_sign_change_flagged':sum(r['source']==src and r['flagged_original'] and r['mass_direction_changed'] for r in dupe_rows),
                      'duplicate_mask_reversal_warning_disappeared_flagged':sum(r['source']==src and r['flagged_original'] and r['reversal_warning_disappeared'] for r in dupe_rows)} for src in SOURCES}}
    return {'summary':summary,'report':report,'tables':{
        'calibration.csv':cal_rows,'attribution.csv':attr_rows,
        'influential_issues.csv':top_rows,
        'duplicate_sensitivity.csv':dupe_rows,
        'duplicate_coverage.csv':coverage_rows,
        'page_positions.csv':pos_rows,
        'page_duplicate_masks.csv':mask_rows}}


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data',type=Path,required=True)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--permutations',type=int,default=PERMUTATIONS)
    a.add_argument('--verify',action='store_true')
    a.add_argument('--no-figure',action='store_true')
    args=a.parse_args()
    if args.permutations!=PERMUTATIONS:
        raise ValueError('Frozen protocol requires exactly 4999 permutation replicates')
    root=args.root.resolve()
    data=args.data.resolve()
    result=run(data,root,args.permutations)
    out=root/'results/archivelens/followup'
    content={out/'summary.json':json.dumps(result['summary'],indent=2,sort_keys=True)+'\n',
             root/'reports/ARCHIVELENS_FOLLOWUP_2026-10-26.md':result['report']}
    for name,rows in result['tables'].items():
        content[out/name]=csv_output(rows)
    if args.verify:
        for path,txt in content.items():
            if not path.exists() or path.read_text(encoding='utf-8')!=txt:
                raise AssertionError('Follow-up artifact does not match recomputation: '+str(path))
        print('PASS: byte-identical follow-up artifacts from fresh recalculation')
    else:
        for path,txt in content.items():
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(txt,encoding='utf-8')
        if not args.no_figure:
            figs=root/'results/figures'
            figs.mkdir(parents=True,exist_ok=True)
            make_figure(result['summary']['calibration'],result['tables']['attribution.csv'],
                        result['summary']['source_year_repeat_coverage'],figs)
    for src in SOURCES:
        print(src, json.dumps({'calibration':result['summary']['calibration'][src],
                               'attribution':result['summary']['attribution_counts'][src]},sort_keys=True))
    print('PASS: full calibration/attribution follow-up complete; original science unchanged')


if __name__=='__main__':main()
