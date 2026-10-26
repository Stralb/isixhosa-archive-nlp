#!/usr/bin/env python3
"""Summarize three Phase-7 precommitted diagnostics without refitting/reselecting models."""
import json, statistics as st, random, hashlib, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sim=json.loads((ROOT/'simulation_results.json').read_text())
tv=json.loads((ROOT/'two_view_results.json').read_text())
fac=json.loads((ROOT/'sampling_factorial_results.json').read_text())
res={'synthetic':{'baseline':{},'by_condition':{}},'two_view':{},'factorial':{}}

for src in ('IMVO','ISIG'):
    base=sim['baseline'][src]
    res['synthetic']['baseline'][src]={
        'unseen_pct':100*base['observed_unseen_fraction'],
        'eligible_word_positions':base['eligible_original_occs'],
        'total_original_words':base['source_word_occurrences'],
        'parity_minus_opportunity_mean':base['parity_minus_opportunity_mean'],
        'by_seed':base['parity_minus_opportunity_by_seed']}
    for mode in ('confusable','random','split'):
        for rate in (0.01,0.03,0.05,0.1):
            v=[z['by_source'][src] for z in sim['rows'] if z['mode']==mode and z['target_fraction']==rate]
            assert len(v)==20
            unseen=[x['observed_unseen_fraction'] for x in v]
            artificial=[x['artificial_share_of_observed_unseen'] for x in v]
            gaps=[x['parity_minus_opportunity_mean'] for x in v]
            base_sign=base['parity_minus_opportunity_mean']>0
            res['synthetic']['by_condition'][f'{src}:{mode}:{rate}']={
                'observed_unseen_pct_mean':100*st.mean(unseen),
                'increase_pp':100*(st.mean(unseen)-base['observed_unseen_fraction']),
                'artificial_of_observed_unseen_pct_mean':100*st.mean(artificial),
                'rank_contrast_mean':st.mean(gaps),
                'rank_contrast_2p5_97p5_simulation_range':sorted(gaps)[0:1]+sorted(gaps)[-1:],
                'mean_rank_reversals_out_of20':sum((x>0)!=base_sign for x in gaps)}

for book in ['ityala_lamawele','zemkinkomo_magwalandini']:
    p=[x for x in tv['per_page'] if x['book']==book]
    assert len(p)==24
    s=tv['book_summary'][book]
    rng=random.Random(20261023)
    delta=[]
    for _ in range(20000):
        sample=[p[rng.randrange(len(p))] for i in p]
        am=sum(x['unseen_psm3_matched'] for x in sample)
        an=sum(x['matched_words'] for x in sample)
        bm=sum(x['unseen_psm3_unmatched'] for x in sample)
        bn=sum(x['psm3_words']-x['matched_words'] for x in sample)
        if an and bn:delta.append(bm/bn-am/an)
    delta.sort()
    res['two_view'][book]={
        'pages':len(p),'agree_pct':100*s['agree']['word_occurrences']/s['all']['word_occurrences'],
        'unseen_agree_pct':100*s['agree']['unseen_fraction'],
        'unseen_disagree_pct':100*s['disagree']['unseen_fraction'],
        'unseen_diff_pp':100*(s['disagree']['unseen_fraction']-s['agree']['unseen_fraction']),
        'descriptive_page_bootstrap_ci95_pp':[100*delta[int(.025*len(delta))],100*delta[int(.975*len(delta))]],
        'parity_minus_opportunity_unseen_all':s['all']['parity_minus_opportunity_unseen_mean'],
        'parity_minus_opportunity_unseen_agree':s['agree']['parity_minus_opportunity_unseen_mean'],
        'rank_reversed_by_agreement_filter':(s['all']['parity_minus_opportunity_unseen_mean']>0)!=(s['agree']['parity_minus_opportunity_unseen_mean']>0)}

for mix in (0,0.5,1):
    entries=[r for r in fac['runs'] if r['mix']==mix]
    membership={}
    for m in sorted({r['membership_seed'] for r in entries}):
        rows=[r for r in entries if r['membership_seed']==m]
        membership[str(m)]={'vocab_sha_count_across_orders':len({r['vocab_sha256'] for r in rows}),
                            'membership_multiset_sha_count':len({r['membership_multiset_sha256'] for r in rows}),
                            'unique_training_surface_forms':rows[0]['membership_unique_types'],
                            'unseen_E_IMVO':st.mean(r['validation']['IMVO']['unseen_E'] for r in rows),
                            'unseen_E_ISIG':st.mean(r['validation']['ISIG']['unseen_E'] for r in rows),
                            'original_byte_for_byte_replayed':all(r.get('byte_for_byte_original_vocab_reproduced') is True for r in rows if 'byte_for_byte_original_vocab_reproduced' in r)}
    res['factorial'][str(mix)]={'number_of_models':len(entries),'membership':membership,
                               'vocab_sha_count_across_memberships':len({r['vocab_sha256'] for r in entries})}

res['gates']={'H1_3pct_confusable_unseen_increase_ge_1pp_either_source':any(res['synthetic']['by_condition'][f'{s}:confusable:0.03']['increase_pp']>=1 for s in ['IMVO','ISIG']),
              'H2_3pct_any_mode_ranking_reversal_ge_10of20':any(res['synthetic']['by_condition'][f'{s}:{m}:0.03']['mean_rank_reversals_out_of20']>=10 for s in ['IMVO','ISIG'] for m in ['confusable','random','split']),
              'two_view_rank_reversal_either_book':any(v['rank_reversed_by_agreement_filter'] for v in res['two_view'].values()),
              'fixed_member_order_has_any_vocabulary_change':any(m['vocab_sha_count_across_orders']>1 for x in res['factorial'].values() for m in x['membership'].values())}

(ROOT/'PHASE7_SUMMARY.json').write_text(json.dumps(res,indent=2,ensure_ascii=False))
print('GATES',res['gates'])
for book,r in res['two_view'].items():print(book,r)
for mix,r in res['factorial'].items():print(mix,r)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for src in ('IMVO','ISIG'):
        b=res['synthetic']['baseline'][src]['unseen_pct']
        rates=[0,.01,.03,.05,.1]
        for mode in ('confusable','split'):
            y=[b]+[res['synthetic']['by_condition'][f'{src}:{mode}:{r}']['observed_unseen_pct_mean'] for r in rates[1:]]
            ax.plot([100*r for r in rates],y,marker='o',label=f'{src} / {mode}')
    ax.set_xlabel('Fraction of eligible input words synthetically corrupted (%)')
    ax.set_ylabel('Observed unseen word occurrences (%)')
    ax.set_title('Synthetic OCR perturbation: apparent lexical novelty')
    ax.legend(frameon=False,fontsize=9,ncol=2)
    ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(ROOT/'ocr_unseen_inflation.png',dpi=190);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for src in ('IMVO','ISIG'):
        xs=[];ys=[];labels=[]
        for mix in (0,0.5,1):
            for k,v in res['factorial'][str(mix)]['membership'].items():
                xs.append(100*mix+(-2 if k=='11' else +2 if k=='23' else 0))
                ys.append(v['unseen_E_'+src]);labels.append((mix,k))
        ax.scatter(xs,ys,s=80,label=src)
        for x,y,(mix,k) in zip(xs,ys,labels):
            ax.annotate(f'm{k}',(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xticks([0,50,100],['0% Imvo','50% Imvo','100% Imvo'])
    ax.set_ylabel('Unseen-form pieces / 100 letters (validation)')
    ax.set_title('Fixed word-order factorial: training sample membership changes outcomes')
    ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(ROOT/'sampling_membership_effect.png',dpi=190);plt.close(fig)
except Exception as exc:print('FIGURE_ERROR',str(exc))
