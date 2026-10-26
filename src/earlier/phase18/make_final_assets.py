from pathlib import Path
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path('/mnt/data/isiXhosa_phase18_evidence_routing')
U=json.load(open(P/'FINAL_COMPARATIVE_UNCERTAINTY.json'))
cohorts=[('clean','Clean'),('char_only','1-character'),('original_char_boundary','Substitution + boundary'),('char_plus_boundary','Fresh boundary variation'),('two_char_plus_boundary','2-character + boundary')]
methods=[('raw_fuzzy','Raw BM25 + fuzzy'),('hard_fuzzy','Hard language routing'),('gated_fuzzy','Gated candidate expansion')]
x=np.arange(len(cohorts));width=.25
fig,ax=plt.subplots(figsize=(11.2,5.6))
for j,(method,label) in enumerate(methods):
 values=[U[c]['top1_macro'][method]*100 for c,_ in cohorts]
 ax.bar(x+(j-1)*width,values,width,label=label)
ax.set_xticks(x,[s for _,s in cohorts]);ax.set_ylim(0,105);ax.set_ylabel('Any-matching-page Top-1 (%)')
ax.set_title('Matched issue-held-out historical isiXhosa retrieval\n408 paired English- and isiXhosa-view queries')
ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',ncol=3,fontsize=9)
fig.tight_layout()
fig.savefig(P/'phase18_policies_vs_noise.png',dpi=180);plt.close(fig)

G=json.load(open(P/'ISSUE_DISJOINT_RESULTS.json'))
cohorts2=[('clean','Clean'),('original_char_boundary','Original noise'),('char_only','Single-character'),('char_plus_boundary','Boundary-change'),('two_char_plus_boundary','Stronger noise')]
raw=[100*G['summary'][key]['test_macro_raw'] for key,_ in cohorts2]
corrected=[100*G['summary'][key]['test_macro_gated'] for key,_ in cohorts2]
fig,ax=plt.subplots(figsize=(9.2,5.1))
ax.plot(range(len(cohorts2)),raw,marker='o',linewidth=2,label='Raw BM25 + fuzzy')
ax.plot(range(len(cohorts2)),corrected,marker='o',linewidth=2,label='Language-gated candidate expansion')
ax.set_xticks(range(len(cohorts2)),[x[1] for x in cohorts2],rotation=15)
ax.set_ylabel('Macro Top-1 (%)');ax.set_ylim(0,105);ax.grid(alpha=.22);ax.legend()
ax.set_title('No source-view subgroup loses accuracy in these issue-held-out tests')
fig.tight_layout();fig.savefig(P/'phase18_issue_disjoint_gate.png',dpi=180);plt.close(fig)

# Human-readable experimental checkpoint, not manuscript.
notes=f'''# Phase 18 - research checkpoint (experiments, not a paper)

Date: 2026-10-23. **No extended abstract was written.** All protocols, raw per-query decisions and trained-retrieval replay inputs are retained.

## The scientific question and what changed

Can a historical bilingual newspaper retrieval system **use short-query language predictions to add candidates rather than exclude entire source-language views**? Initial evidence-score hard routing appeared to work on English-only queries, but was rejected: it chose English for all 204 held-out English examples. The follow-up paired an English-view and an isiXhosa-view five-word query from each of 466 identical source pages. Every candidate had to occur within a raw page paragraph; every raw page containing the original phrase counted as a valid source. Queries were grouped into 212 development issues and 169 held-out issues; 408 held-out paired queries total (159 pairs Imvo, 45 pairs Isigidimi).

## Controlled retrieval comparison

20 candidate pages per query and the same RapidFuzz partial-ratio reranker; methods differ only in candidate selection:

- RAW: top 20 raw-page BM25 candidates.
- HARD: one language field chosen by the short-query classifier, top 20 field BM25 candidates.
- GATED: use RAW unless the classifier predicts English; when English, take 15 raw-page + 5 English-view candidates (backfill to 20).
- ORACLE VIEW: queries are retrieved through the original generating language view (methodological upper reference, not an oracle for the best possible result).

A second classifier training run **excluded every held-out issue** and included 1,566 training excerpts from 110 development Imvo issues. The retrieval candidates are still indexed over the complete archive, as appropriate for an archive retrieval test.

| Condition | RAW macro Top-1 | HARD macro Top-1 | GATED macro Top-1 | GATED-RAW (issue-bootstrap 95% CI) |
|---|---:|---:|---:|---:|
'''
for key,label in [('clean','Clean'),('char_only','One character substitution'),('original_char_boundary','Original substitution plus boundary'),('char_plus_boundary','Fresh substitution plus boundary'),('two_char_plus_boundary','Two substitutions plus boundary')]:
 rec=U[key];a=rec['top1_macro'];dd=rec['comparisons']['gated_fuzzy_minus_raw_fuzzy']
 notes+=f"| {label} | {a['raw_fuzzy']*100:.1f}% | {a['hard_fuzzy']*100:.1f}% | **{a['gated_fuzzy']*100:.1f}%** | +{dd['point']*100:.1f} pp [{dd['ci95'][0]*100:+.1f}, {dd['ci95'][1]*100:+.1f}] |\n"
notes+='''
Source-view equal-weighted macro accuracy. All five results were obtained on **the same issues** under five query noise conditions. Thus repeated positive intervals are not five independent datasets. Confidence intervals resample source issues, not OCR engines or human query-writing behaviour.

## Mechanism and negative controls

1. The 408 query benchmark has 204 English-view and 204 isiXhosa-view queries, paired within 204 pages. At equal source/view weighting, raw Top-1 is 93.6% on clean queries. The same fixed 20-candidate fuzzy reranker is used in all interventions.
2. Hard routing reduces clean Top-1 to 88.0%. Its candidate exclusion is detrimental on Isigidimi English-view excerpts and on some Xhosa-view excerpts despite reasonable overall classifier-to-view agreement.
3. Fuzzy reranking of raw BM25 alone is an exceptionally strong baseline; a word-internal character-trigram TFIDF alternative performs substantially worse in this setup.
4. Always adding five English-view candidates improves synthetic-noise retrieval but can harm Xhosa-view source recovery by 2-4 percentage points. Classifier-gated augmentation preserves the RAW Top-1 for Xhosa-view queries in this held-out sample, while improving English-view retrieval. **No observed harm is not a universal guarantee.**
5. Field-score hard routing only appeared good on English-only evaluation, where it effectively learned to select English for everyone. On balanced queries it significantly underperformed raw fuzzy retrieval.
6. Candidate gain comes from adding English-view pages to the top-20 pool, not from an original fuzzy-matching algorithm. Hard language routing is not the same as nonexclusive candidate enrichment.

## Publication and scientific validity gate

- **What we have:** a source-anchored, issue-grouped empirical demonstration that query-level hard language-index exclusion can suppress evidence access; routing the same language prediction into candidate *augmentation* rather than *exclusion* improves short-query passage-location retrieval under synthetic OCR noise without measured Xhosa-view degradation.
- **What is NOT supported:** a new general-purpose retrieval algorithm, word-level language-id correctness, fairness for real language communities, results for real user intent, OCR-error robustness on original newspaper scans, or out-of-archive generalisation.
- **Related prior work:** multilingual search / cross-language access and field routing are established; historical newspaper fuzzy search is established (Järvelin et al. 2016), and interfaces like impresso already incorporate language and OCR-quality metadata. A publishable *resource/evaluation* contribution would need an independently assessed relevance set, preferably human-authored short searches, and a real OCR alignment or cross-archive replication. Do not call this a method invention.
- **Next hard gate:** collect a small but independent set of user-authored source queries or real newspaper OCR/page pairs, and compare the fixed GATED method against raw fuzzy, HARD fuzzy, no-LID English enrichment, and a competitive established multilingual retriever. No more tuning on the current 408-query held-out split.

## Reproducibility

Run in the same filesystem layout with the source `corpus_cache.pkl` and Phase17/Phase15 score files. See included `run_*.py`, `*PROTOCOL.md`, `*RESULTS.json`, and per-query JSONL files. Major relevant scripts: `build_balanced.py`, `run_balanced.py`, `candidate_pool.py`, `noisy_transfer.py`, `noise_sensitivity.py`, `gated_pool.py`, `lid_disjoint.py`, `hard_fuzzy_control.py`, `final_uncertainty.py`. Their protocols record the successive negative controls, including the rejected English-only routing result.
'''
(P/'RESEARCH_CHECKPOINT_PHASE18.md').write_text(notes)
print('created',P/'RESEARCH_CHECKPOINT_PHASE18.md')
