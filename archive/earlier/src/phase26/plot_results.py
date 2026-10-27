#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib.pyplot as plt
D=Path(__file__).parent
p=json.load(open(D/'phase26_abc_results.json'))
t=json.load(open(D/'tokenizer_link_results.json'))
# Separate figures, no subplots.
r=[x for x in p['temporal'] if x['feature']=='char4']
fig,ax=plt.subplots(figsize=(7.2,4.2))
x=list(range(2));w=.34
ax.bar([a-w/2 for a in x],[a['observed_js_bits'] for a in r],width=w,label='Observed 1885-1888')
ax.bar([a+w/2 for a in x],[a['permutation_95'] for a in r],width=w,label='95th percentile: month-shuffled issues')
ax.set_xticks(x,[a['source'] for a in r]);ax.set_ylim(0,.26);ax.set_ylabel('Within-word character-4-gram JSD (bits)')
ax.set_title('Temporal lexical-pattern evidence: mixed across newspapers');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(D/'temporal_character_drift.png',dpi=175);plt.close(fig)
r=next(a for a in p['sentiment'] if a['source']=='IMVO')
fig,ax=plt.subplots(figsize=(6.8,4.2))
ax.bar(['English view','isiXhosa view'],[100*r['en_dictionary_coverage'],100*r['xh_dictionary_coverage']]);ax.set_ylabel('Exact dictionary hits / 100 tokens (%)');ax.set_title('English sentiment lexicon coverage on paired pages');fig.tight_layout();fig.savefig(D/'english_sentiment_coverage.png',dpi=175);plt.close(fig)
c=p['newspaper']
fig,ax=plt.subplots(figsize=(7.2,4.2))
ax.bar([a['model'] for a in c],[100*a['balanced_accuracy_issue'] for a in c])
ax.axhline(50,linestyle='--',color='gray',linewidth=1,label='Balanced chance')
ax.set_ylabel('Issue-balanced held-out 1888 accuracy (%)');ax.set_ylim(0,100);ax.set_title('Newspaper signatures transfer across years');ax.legend();fig.tight_layout();fig.savefig(D/'heldout_newspaper_classifier.png',dpi=175);plt.close(fig)
b=t['within_source_benefits']
fig,ax=plt.subplots(figsize=(6.9,4.2))
labels=['Imvo','Isigidimi'];source=['IMVO','ISIG'];val=[b[s]['mean'] for s in source];ci=[b[s]['bootstrap_95ci'] for s in source]
ax.bar(labels,val);ax.errorbar(labels,val,yerr=[[v-a for v,(a,c) in zip(val,ci)],[c-v for v,(a,c) in zip(val,ci)]],fmt='none',ecolor='black',capsize=5)
ax.axhline(0,color='gray',linewidth=.75);ax.set_ylabel('Within-source specialist BPE savings (pieces / 100 chars)');ax.set_title('New pilot: matched corpus, unseen 1888 passages');fig.tight_layout();fig.savefig(D/'bpe_source_specific_savings.png',dpi=175);plt.close(fig)
