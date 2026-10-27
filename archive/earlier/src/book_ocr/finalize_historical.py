"""Create a figure and Phase historical checkpoint from frozen results; no refitting."""
from pathlib import Path
import json, hashlib, collections, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent
D=json.loads((R/'HISTORICAL_BOOK_RESULTS.json').read_text())
raw={r['book']:r for r in D['scores'] if r['variant']=='raw'}
joined={r['book']:r for r in D['scores'] if r['variant']=='joined_line_hyphens'}
modern=json.loads((R/'isiXhosa_external_validation/modern_seed_uncertainty.json').read_text())
overlap=json.loads((R/'EXCLUSION_SENSITIVITY.json').read_text())
source=json.loads((R/'ocr/OCR_MANIFEST.json').read_text())
seeds=[11,23,37,47,59]
series={'Modern Vuk’uzenzele':[-v for v in modern['unseen']['per_seed_diff']],
        'Ityala Lamawele':raw['ityala_lamawele']['primary_parity_minus_opportunity']['unseen_by_seed'],
        'Zemk’inkomo':raw['zemkinkomo_magwalandini']['primary_parity_minus_opportunity']['unseen_by_seed']}
xx=list(range(3));fig,ax=plt.subplots(figsize=(9.3,5.4))
for j,seed in enumerate(seeds):
    y=[series[k][j] for k in series]
    ax.plot(xx,y,marker='o',markersize=6,linewidth=1.6,alpha=.65,label=f'Seed {seed}')
mean=[sum(v)/len(v) for v in series.values()]
ax.plot(xx,mean,color='black',marker='D',linewidth=3,markersize=8,label='5-seed mean')
ax.axhline(0,color='grey',linestyle='--',linewidth=1)
ax.set_xticks(xx,list(series.keys()));ax.set_ylabel('Parity minus opportunity, unseen-form pieces / 100 chars\nPositive favours opportunity selector')
ax.set_title('The mean advantage is smaller on historical books and unstable by seed')
ax.legend(loc='upper right',ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(R/'historical_frozen_seed_comparison.png',dpi=190);plt.close(fig)

rawrows=[]
for b,title in [('ityala_lamawele','Ityala Lamawele'),('zemkinkomo_magwalandini',"Zemk’inkomo Magwalandini")]:
    a=raw[b];j=joined[b];p=a['primary_parity_minus_opportunity'];m=a['by_mix'];
    rawrows.append({'book':title,'file':b,'ocr_word_occurrences':a['all_word_occurrences'],
                    'ocr_supported_char_percent':100*a['supported_char_fraction'],
                    'novel_occurrences_percent':100*a['unseen_occurrence_fraction'],
                    'exact_12word_overlap_percent':100*a['detected_12word_overlap_fraction'],
                    'unseen_0':m['0']['unseen_E_mean'],'unseen_50':m['0.5']['unseen_E_mean'],'unseen_100':m['1']['unseen_E_mean'],
                    'all_0':m['0']['all_E_mean'],'all_50':m['0.5']['all_E_mean'],'all_100':m['1']['all_E_mean'],
                    'parity_minus_opportunity_unseen':p['unseen_mean'],'relative_unseen_benefit_percent':100*p['unseen_mean']/m['0.5']['unseen_E_mean'],
                    'parity_minus_opportunity_all':p['all_mean'],'positive_seeds':p['unseen_positive_seeds'],
                    'seed_ci95':p['descriptive_seed_bootstrap_ci95'],
                    'line_join_delta':j['primary_parity_minus_opportunity']['unseen_mean'],
                    'overlap_exclusion_delta':overlap[b+'::raw']['parity_minus_opportunity_unseen_mean'],
                    'seed_effects':p['unseen_by_seed']})

ocrflags={}
for b in raw:
    text=(R/f'ocr/{b}_raw.txt').read_text()
    digit_candidates=re.findall(r'(?i)(?<![a-z])[0-9][a-z]{2,}|[a-z]{2,}[0-9][a-z]*',text)
    ocrflags[b]={'digit_adjacent_alphabetic_strings':len(digit_candidates),'hyphen_linebreaks':len(re.findall(r'-\s*\n',text)),
                 'examples_digit_strings':digit_candidates[:5]}

payload={'date':'2026-10-23','status':'Negative/weak external historical result: robust improvement NOT demonstrated',
         'data_precommitted':True,'books_as_units':2,'historical_results':rawrows,'ocr_diagnostic_flags':ocrflags,
         'modern_unseen_parity_minus_opportunity':-modern['unseen']['mean_difference'],
         'modern_seed_positive':sum(v<0 for v in modern['unseen']['per_seed_diff']),
         'cross_corpus_seed_correlation_descriptive':{f'{a} vs {b}':float(np.corrcoef(series[a],series[b])[0,1]) for i,a in enumerate(series) for b in list(series)[i+1:]}}
(R/'HISTORICAL_COMPACT_SUMMARY.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))

def fmt(x,n=3):return f'{x:.{n}f}'
parts=["# Historical isiXhosa external-transfer evaluation — Phase checkpoint",
       "**Date:** 23 October 2026. **Decision:** the proposed 'opportunity-aware' tokenizer-selection method is **not validated**.",
       "\n## Experimental contract and provenance",
       "The 15 original reference BPE models, 1887 newspaper training data, and two selection rules were frozen on 22 October **before** these two books were supplied. The 50/50 Imvo–Isigidimi tokenizer minimises original newspaper-average parity disparity; the 100% Imvo model minimises worst-domain specialist-relative unseen-form regret in 1887 validation. Five matched seeds: 11, 23, 37, 47, 59. Vocab size 2000, fixed training text budget 421,696 alphabetic characters. No models were modified or retrained.",
       f"- Original freeze file SHA-256: `{D['frozen_selection_sha256']}`",
       f"- Frozen protocol SHA-256: `{hashlib.sha256((R/'HISTORICAL_EVAL_FROZEN_PROTOCOL.md').read_bytes()).hexdigest()}`",
       f"- Ityala scanned PDF SHA-256: `{source['input_pdfs']['ityala_lamawele']['sha256']}`",
       f"- Zem scanned PDF SHA-256: `{source['input_pdfs']['zemkinkomo_magwalandini']['sha256']}`",
       "\n## OCR and edition caveats",
       "Both PDFs contain only scanned page images, so their contents were processed by Tesseract's English OCR model, not a historical isiXhosa OCR model. Ityala: 90 PDF pages; 154 physical page-side OCR segments used (the main printed text; introductory and terminal glossary/index excluded). Zem: 571 PDF pages; 510 main-text pages used (introductory matter and end notes excluded). Extracted text is **not** a verified scholarly transcription. The supplied Ityala scan includes a 1931-dated foreword despite the work's 1914 origin. Neither scan should be silently equated with the exact electronic editions described in the archive paper.",
       f"- OCR diagnostic: digit-touching-letter strings: Ityala **{ocrflags['ityala_lamawele']['digit_adjacent_alphabetic_strings']:,}**, Zem **{ocrflags['zemkinkomo_magwalandini']['digit_adjacent_alphabetic_strings']:,}**. These are potential OCR/typographic ambiguities; not all are verified errors.",
       "- Primary scoring retains historical spelling/typography without manual repairs. The precommitted sensitivity rejoins OCR line-end hyphen splits. Every model uses an identical common-character mask; scores apply only to encodable forms.",
       "\n## Main result (unseen-form cost; lower is better)",
       "Cost = mean across 5 seeds of subword pieces per 100 alphabetic characters for OCR-derived surface forms absent from all Jan–Sep 1887 newspaper training text. One book is one independent evaluation unit.",
       "| Book | Isigidimi-only (0%) | Parity (50%) | Opportunity (100%) | Parity − opportunity | Positive seeds |",
       "|---|---:|---:|---:|---:|---:|"]
for r in rawrows:
    parts.append(f"| {r['book']} | {fmt(r['unseen_0'])} | {fmt(r['unseen_50'])} | {fmt(r['unseen_100'])} | +{fmt(r['parity_minus_opportunity_unseen'])} | {r['positive_seeds']}/5 |")
parts.extend(["\n**Interpretation:** Both book-level five-seed means favour the opportunity rule, but only by small amounts and with 3/5 winning seeds on each source. Neither book's seed-level resampling interval excludes zero. On Zem, the 0% Imvo baseline is **strictly better** on the unseen-form mean than both proposed selectors. **No reliable or dominant improvement has been demonstrated.**",
              "\n## Strength of evidence and explicit trade-offs",
              "| Book | Relative unseen saving vs parity | All-form parity minus opportunity (positive is better) | 95% descriptive seed-bootstrap interval, unseen contrast | Supported alphabetic characters |",
              "|---|---:|---:|---:|---:|"])
for r in rawrows:
    parts.append(f"| {r['book']} | {fmt(r['relative_unseen_benefit_percent'],2)}% | {fmt(r['parity_minus_opportunity_all'])} | [{fmt(r['seed_ci95'][0])}, {fmt(r['seed_ci95'][1])}] | {fmt(r['ocr_supported_char_percent'],2)}% |")
parts.extend(["\nThe full-form result reverses the apparent advantage: the parity-selected tokenizer is better for both books on overall mean subword cost. The Zem effect is particularly small (+0.078 pieces / 100 characters, approximately 0.16% of parity cost), far smaller than the uncertainty from only five tokenizer training seeds, with substantive OCR caveats.",
              "\n## Frozen robustness and overlap checks",
              "| Book | Original unseen benefit | Join OCR hyphen line breaks | Exclude detected exact 12-word training matches | Exact training-overlap word coverage |",
              "|---|---:|---:|---:|---:|"])
for r in rawrows:
    parts.append(f"| {r['book']} | {fmt(r['parity_minus_opportunity_unseen'])} | {fmt(r['line_join_delta'])} | {fmt(r['overlap_exclusion_delta'])} | {fmt(r['exact_12word_overlap_percent'],3)}% |")
parts.extend(["\nZem contains **96 exact matching 12-word windows across 14 scanned PDF pages**, including an extract corresponding to `ISIG_1887-09-01_p007.txt` in tokenizer training. They cover 349 word positions (about 0.62%). This is real detected train/external passage reuse, so the collection is not a fully independent text at passage level. Excluding these windows does not change the unseen-form result **by construction**: words in exactly matched training passages are necessarily *seen* forms already excluded by the primary metric. Approximate/OCR-distorted reused text may still survive.",
              "\n## Relation to existing modern transfer",
              "The earlier pre-external Vuk'uzenzele test showed 50/50 − 100% = **+0.721** pieces / 100 characters on unseen surface forms (article-weighted), again **3/5 seeds**. Each external source favours seeds 23/47/59 and disfavors seeds 11/37. This pattern indicates substantial dependence on tokenizer-training randomness rather than a stable selection-rule effect. Modern article weighting differs from within-book character weighting, and genres and transcription quality differ; do not pool these into a single statistical test.",
              "\n## Scientific conclusion and gate decisions",
              "1. **Independent source acquisition:** PASS. Two historical scanned books successfully processed and provenance preserved.",
              "2. **Original metric-disagreement:** PASS on 1887 validation (as already frozen); this is not a claim of metric correctness.",
              "3. **Mean historical novel-form direction:** weakly positive on both OCR-derived books; the one stronger 0%-Imvo baseline wins on Zem.",
              "4. **Training-seed robustness:** FAIL. 3/5 seeds per book and descriptive intervals crossing zero; exactly the same seeds win on modern and historical corpora.",
              "5. **Full-text efficiency:** FAIL for opportunity selector relative to parity on both historical books.",
              "6. **OCR/lexical ground truth:** NOT SATISFIED. Scan artifacts and historical typography affect the definition of 'unseen forms'; 98.18% of Zem alphabetic characters are covered under the shared 26-letter alphabet mask.",
              "7. **Downstream language modelling:** NOT TESTED with these tokenizers on the historical books. Tokenization cost does not demonstrate better semantic understanding or predictive utility.",
              "\n**Overall go/no-go:** Do **not** claim a novel, fairness-improving selection method or reliable unseen-historical-writing gain. The result is useful as a reproducible *negative test*, and as evidence that tokenizer selection objectives, source composition, stochastic training and OCR must be separated carefully. Further work should prioritise strong OCR/transcription controls and comparison to standard objective functions; additional training on these now-inspected books cannot create fresh confirmation.",
              "\n## Reproduction",
              "Place the user's PDFs at the paths expected by `extract_scanned_books.py`; provide the supplied newspaper XhTxt ZIP. Run `python extract_scanned_books.py`, then `python evaluate_historical_frozen.py`, `python evaluate_overlap_exclusion.py`, `python audit_book_overlap.py`, and `python finalize_historical.py`. The extracted raw OCR and page IDs are retained under `ocr/`; full numeric results are in `HISTORICAL_BOOK_RESULTS.json`, source hashes in `ocr/OCR_MANIFEST.json`. The original `score_external.py` independently reproduced raw aggregate figures for both books exactly in `BOOKS_ORIGINAL_SCRIPT_CROSSCHECK.json`. OCR texts are machine-extracted and need proofreading before linguistic claims."])
(R/'HISTORICAL_RESEARCH_CHECKPOINT.md').write_text('\n'.join(parts)+'\n')
print('Wrote checkpoint, summary and figure')
