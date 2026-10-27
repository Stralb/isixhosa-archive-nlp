from pathlib import Path
import json, collections, csv, zipfile, hashlib, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/mnt/data/isiXhosa_phase13_ocr_certification')
D=json.loads((BASE/'results.json').read_text())
E=json.loads((BASE/'edit_distance_results.json').read_text())
V=json.loads((BASE/'posthoc_two_view_audit.json').read_text())
summary={}
for source in ('ityala','zemkinkomo','modern'):
    orig=D['sources'][source]['pair_comparisons'];ext=E['sources'][source]['pairs']
    table={}
    for index,rate in enumerate([0,.001,.005,.01,.02]):
        # Fix numerical boundary where float bisection yields spurious sign on exact ties
        substitution=[('A_better' if r['budgets'][index]['upper_E'] < -1e-8 else ('B_better' if r['budgets'][index]['lower_E'] > 1e-8 else 'abstain')) for r in orig]
        edit=[r['bounds'][index]['certified'] for r in ext]
        table[str(rate)]={'substitution_certified':sum(s!='abstain' for s in substitution),
                          'edit_certified':sum(s!='abstain' for s in edit),'total':len(edit),
                          'substitution_abstain':substitution.count('abstain'),
                          'edit_abstain':edit.count('abstain')}
    summary[source]={'coverage':D['sources'][source]['coverage'],
                      'unseen_fraction':D['sources'][source]['eligible_occur_unseen_fraction'],
                      'rates':table}
(BASE/'summary_checked.json').write_text(json.dumps(summary,indent=2))

labels={'ityala':'Ityala Lamawele','zemkinkomo':'Zemk’inkomo Magwalandini','modern':'Vuk’uzenzele (modern)'}
fig,ax=plt.subplots(figsize=(9.2,4.9),dpi=170)
style={'ityala':'o','zemkinkomo':'s','modern':'^'}
for src in ['ityala','zemkinkomo','modern']:
    rates=[0,.001,.005,.01,.02]
    vals=[summary[src]['rates'][str(r)]['edit_certified'] for r in rates]
    ax.plot([x*100 for x in rates],vals,marker=style[src],lw=1.9,markersize=6,label=labels[src])
ax.set(xlabel='Maximum edited word occurrences (%)',ylabel='Provably stable pairwise rankings (out of 15)',
       title='Tokenizer ranking certificates under one OCR edit per affected word')
ax.set_ylim(-0.5,15.5)
ax.set_xticks([0,.1,.5,1,2]);ax.set_yticks([0,5,10,15]);ax.grid(alpha=.2)
ax.legend(frameon=False,loc='upper right')
fig.tight_layout();fig.savefig(BASE/'phase13_certificate_curve.png');plt.close(fig)

p12=Path('/mnt/data/isiXhosa_phase12_data_integrity')
p12r=json.loads((p12/'results.json').read_text())
(p12/'PHASE12_CHECKPOINT.md').write_text('''# Phase 12 outcome — targeted 1% training text concentration\n\nProtocol: `PRECOMMIT.md` frozen before scoring. Four training-source arms, 12 independent matched draws/arm, five treatments per draw, 200,000 alphabetic characters and identical 2,000 BPE vocabulary. Relative cost is unseen-form pieces per 100 alphabetic characters. Results are retrospective on previously inspected source periods.\n\n| Source | Targeted 1% minus distributed baseline | Paired 95% bootstrap CI |\n|:--|--:|:--|\n| Imvo 1887 | -0.0071 | [-0.7490, +0.7337] |\n| Isigidimi 1887 | +0.0352 | [+0.0225, +0.0485] |\n| Imvo 1889 | +0.0235 | [-0.0563, +0.0942] |\n| Modern isiXhosa | -0.0401 | [-0.1352, +0.0217] |\n\nPredefined material-effect threshold: >= +0.25. Generalisation rule: at least 3/4 arms with threshold, CI positive and targeted > random concentrated control. **0/4 arms passed; reject this candidate general method.** Isigidimi has a small positive effect, not a practically large one. No source editing or deployed-model effect is established. Implementation-only fixes are recorded in `EXECUTION_NOTE.md`.\n''')

report=f'''# Phase 13 research checkpoint — certifying OCR-uncertain tokenizer comparisons

Date: 2026-10-23. **Status: viable technical prototype; publishability and novelty NOT established.**

## Why this experiment exists
Phases 1-11 showed a real but non-generalizable BPE first-merge bifurcation. Phase 12 tested 1% targeted data concentration under a strong multi-corpus negative control and failed in all four source arms (see separate checkpoint). Further tokenizer-optimization heuristics are not justified.

The new question is whether, when historical text is only known via OCR, **a tokenizer evaluation winner can be certified against an explicit budget of transcription uncertainty**. This is a low-resource / Responsible NLP EVALUATION question, not an attempt to improve BPE training.

## Exact mathematical result (conditional on the uncertainty model)
For fixed tokenizer pair A,B, tokenizer-training vocabulary T, and OCR-observed word occurrences w_i, define

    d_i(v) = 1[v not in T] * (pieces_A(v)-pieces_B(v)).

The unseen-form cost gap is D = 100 * sum_i d_i(v_i) / sum_i 1[v_i not in T] * |v_i|. As both models use the same observed/altered unseen-word subset, the denominator is positive and shared, so **the ranking sign depends only on sum_i d_i(v_i)**.

For a specified edit set V_i and at most m independently edited word occurrences, the exact minimum achievable numerator is

    S_min(m) = S_obs + sum of the m most negative values of min(0, min_(v in V_i) d_i(v) - d_i(w_i)).

Maximum similarly uses the m greatest positive changes. If S_min(m)>0, B remains better under all permitted edits; if S_max(m)<0, A remains better; otherwise ABSTAIN. This is a tight, deterministic robustness certificate, not a Monte-Carlo confidence interval. The separate substitution-only script obtains tight *numerical gap intervals* using a fractional-programming bisection that recomputes unseen-group denominators for every candidate variant.

## Source material and corpus status
- Ityala Lamawele: 30 hash-selected OCR page/side records, 3,000 alphabetic word occurrences, {summary['ityala']['unseen_fraction']*100:.1f}% unseen.
- Zemk'inkomo Magwalandini: 30 records, 2,697 word occurrences, {summary['zemkinkomo']['unseen_fraction']*100:.1f}% unseen.
- Modern Vuk'uzenzele: 30 articles, 3,000 word occurrences, {summary['modern']['unseen_fraction']*100:.1f}% unseen.
- Fixed 1887 Jan–Sep newspaper training vocabulary; all 15 previously frozen SentencePiece BPEs (three source mixtures, five training-sample draws); 15 matched comparisons per corpus, **45 paired comparisons total**. Common ASCII letter support was 100% on selected words.
- All three external texts and the original tokenizer models were previously studied. These are **retrospective method-development tests**. They are not fresh external confirmation.

## Outcome: number of certified rankings out of 15

| Source | 0.1% substitutions | 0.5% substitutions | 0.5% edit-distance-one | 1% edit-distance-one | 2% edit-distance-one |
|:--|--:|--:|--:|--:|--:|
'''
for src in ['ityala','zemkinkomo','modern']:
    s=summary[src]['rates']
    report+=f"| {labels[src]} | {s[str(.001)]['substitution_certified']} | {s[str(.005)]['substitution_certified']} | {s[str(.005)]['edit_certified']} | {s[str(.01)]['edit_certified']} | {s[str(.02)]['edit_certified']} |\n"
report+='''
Numerical correction: a nearly tied Zemk'inkomo comparison was erroneously assigned a winning sign by a floating-point threshold in the first substitution-only status tally. Reclassification with 1e-8 tolerance changes its 0.5% count from 12 to **11**, without altering calculated numerical bounds. The edit-distance-one 0.5% count is 10.

The practical precommit gate is satisfied: at least one certificate and at least one abstention at 0.5% in every corpus. BOTH toy exhaustive verifications passed, at most two edited words under substitution and edit-distance variant families respectively. Budget monotonicity, subset-consistency, unchanged training lexicon and positive denominator assumptions verified in code.

## Independent real-OCR-setting check — important limitation
On a separate 24-page sample of each book, two Tesseract settings generated different text. Comparing unseen-form tokenization across all 15 matched pairs produced 0/15 ranking reversals on Ityala and **2/15 on Zemk'inkomo**. For the two Zemk'inkomo flips (50/50 minus Imvo-only BPE):

- Seed 23: +0.2953 pieces/100 chars on primary OCR; -0.2691 on alternative OCR.
- Seed 47: +0.2215 on primary OCR; -0.1409 on alternative OCR.

These *real* OCR-view differences include word boundary, splitting and layout changes, which violate the certificate's one-edit-per-word uncertainty model. The certificate therefore **does not guarantee stability between arbitrary real OCR transcriptions**. The two views are not ground truth, and their agreement is not a gold-standard OCR accuracy rate.

## Novelty/literature caveat
- Jia et al. (EMNLP 2019) and CERT-ED (Findings EMNLP 2024) already establish certified text robustness. Wu et al. (CIKM 2022) already study certified robustness for ranking models. Other work studies OCR effect on downstream tasks (ACL Industry 2026).
- Our specialized exact partial-identification formula for a dynamic unseen-word tokenizer evaluation subset is **not automatically an algorithmic breakthrough**: it is an application of separable budgeted optimization to this metric. A broader survey and stronger baselines are required before claiming novelty.

## What would raise this to a credible paper
1. Extend the uncertainty set to **word split/join and multiple edits**, with a tractable exact or conservative certificate; those are material real OCR operations.
2. Compare with naive score ranking, simple empirical OCR perturbation, multiple-OCR-output consensus, generic edit-distance bounds and selective abstention. Demonstrate that a certificate prevents unsupported model-selection claims or flags real OCR-view reversals.
3. Calibrate the uncertainty budgets using actual page-level OCR/verified transcriptions or a clean officially published isiXhosa reference corpus subjected to controlled degradations; no need for an isiXhosa expert to run the synthetic control, but claims about real OCR error need validation.
4. Run this frozen procedure on an **uninspected independent source** and report its selective decision quality, coverage and computational costs.

**Decision:** This is the first formally testable robustness *evaluation* idea in this continuation with a working certificate and nontrivial abstention. It is stronger than the failed repetition/merge heuristics, but is still a prototype, **not yet a publishable new method**.

## Files
- `PRECOMMIT.md`, `EDIT_DISTANCE_EXTENSION_PRECOMMIT.md`, and SHA256 hashes: frozen protocols.
- `certify.py`: exact substitution-only numeric ranking-gap envelopes and toy exhaustive proof check.
- `edit_distance_certificate.py`: exact sign guarantee under substitution/insertion/deletion.
- `paired_ocr_check.py`: post-hoc real two-setting OCR instability check.
- `results.json`, `edit_distance_results.json`, `posthoc_two_view_audit.json`, `summary_checked.json`: complete results.
- `phase13_certificate_curve.png`: certificate fractions versus corruption budget.
'''
(BASE/'PHASE13_RESEARCH_CHECKPOINT.md').write_text(report)
(BASE/'README.md').write_text('''# Phase 13 reproducible research package\n\nRun requirements: Python >=3.11, numpy, sentencepiece, matplotlib; original isiXhosa newspaper language ZIP, previous 15 fitted BPE models, OCR-derived book page JSONL and xho.jsonl. All paths in the original scripts reference the working project layout under `/mnt/data`. Change constants if moved.\n\n1. `python certify.py` — exact numerical substitution-only gaps and exhaustive toy check.\n2. `python edit_distance_certificate.py` — stricter edit-distance-one sign certificate and exhaustive toy check.\n3. `python paired_ocr_check.py` — exploratory two-Tesseract-setting comparison.\n4. `python build_report.py` — build checked result tables, plot and research checkpoint.\n\n**Important:** The two books were already inspected, so this is retrospective development evidence. The current bounds cover at most one word-internal edit on any affected word, and cannot certify arbitrary historical OCR outputs. The method is not a replacement for human transcription verification.\n''')
print('Created',BASE/'PHASE13_RESEARCH_CHECKPOINT.md',BASE/'phase13_certificate_curve.png')
