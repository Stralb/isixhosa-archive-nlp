"""Conditional issue-grouped uncertainty and figures for the three-track pilot."""
from pathlib import Path
from collections import defaultdict
import json,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('/mnt/data/isiXhosa_three_track_pilot')
audit=[json.loads(x) for x in (O/'cross_view_adjacency_pages.jsonl').open()]
q=[json.loads(x) for x in (O/'query_repaired.jsonl').open()]
base=json.loads((O/'retrieval_repaired_summary.json').read_text())
length=json.loads((O/'query_length_summary.json').read_text())
cond={};rng=np.random.default_rng(20261023)
for src in ('IMVO','ISIG'):
 for lang in ('xh','en'):
  byissue=defaultdict(lambda:[0,0])
  for d in audit:
   if d['source']!=src:continue
   byissue[d['issue']][0]+=d[f'{lang}_nonverbatim'];byissue[d['issue']][1]+=d[f'{lang}_windows']
  pairs=np.array([z for z in byissue.values() if z[1]>0],dtype=float)
  ix=rng.integers(0,len(pairs),size=(6000,len(pairs)))
  samples=pairs[ix,0].sum(axis=1)/pairs[ix,1].sum(axis=1)
  cond[src+'_'+lang]={'issues':len(pairs),'rate':float(pairs[:,0].sum()/pairs[:,1].sum()),
       'issue_bootstrap_95':[float(x) for x in np.quantile(samples,[.025,.975])]}
byissue=defaultdict(list)
for z in q:byissue[z['issue']].append(int(z['num_exact_match_pages']>1))
pairs=np.array([[sum(a),len(a)] for a in byissue.values()],dtype=float)
ix=rng.integers(0,len(pairs),size=(6000,len(pairs)))
samples=pairs[ix,0].sum(axis=1)/pairs[ix,1].sum(axis=1)
cond['multi_page_query_rate']={'queries':len(q),'issues':len(pairs),'rate':float(pairs[:,0].sum()/pairs[:,1].sum()),
 'issue_bootstrap_95':[float(x) for x in np.quantile(samples,[.025,.975])]}
(O/'issue_uncertainty.json').write_text(json.dumps(cond,indent=2))
print(json.dumps(cond,indent=2),flush=True)

# Plot 1: query length sensitivity
x=np.array([5,10,20]);fig,ax=plt.subplots(figsize=(7.8,4.6));
for label,key in [('Multiple valid pages','multiple_page_frac'),('Exact sampled page, top 1','source_R1'),('Any matching page, top 1','any_R1')]:
 y=[length[str(n)]['all'][key]*100 for n in x]
 ax.plot(x,y,marker='o',label=label)
 for xi,yi in zip(x,y):ax.annotate(f'{yi:.1f}%',(xi,yi),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set(xlabel='Contiguous words in query',ylabel='Percentage of the same 589 queries',title='Repeated source text changes retrieval evaluation');ax.set_xticks(x);ax.set_ylim(0,105);ax.grid(alpha=.2);ax.legend(loc='center right',fontsize=9);fig.tight_layout();fig.savefig(O/'figure_query_length.png',dpi=180);plt.close(fig)

# Plot 2: language visibility
fig,ax=plt.subplots(figsize=(8.4,4.4));k=['IMVO_en','ISIG_en','All English'];a=[base['raw_filter_en_majority']['clean_'+g]['exact_r1']*100 for g in ['IMVO_en','ISIG_en']] + [sum(base['raw_filter_en_majority']['clean_'+g]['exact_r1']*base['raw_filter_en_majority']['clean_'+g]['n'] for g in ['IMVO_en','ISIG_en'])/229*100]
b=[base['raw_filter_en_multitag']['clean_'+g]['exact_r1']*100 for g in ['IMVO_en','ISIG_en']] + [sum(base['raw_filter_en_multitag']['clean_'+g]['exact_r1']*base['raw_filter_en_multitag']['clean_'+g]['n'] for g in ['IMVO_en','ISIG_en'])/229*100]
xx=np.arange(len(k));w=.33;p=ax.bar(xx-w/2,a,w,label='English-majority page filter');s=ax.bar(xx+w/2,b,w,label='English evidence tag (50+ words)');
ax.bar_label(p,fmt='%.1f%%',padding=3,fontsize=9);ax.bar_label(s,fmt='%.1f%%',padding=3,fontsize=9)
ax.set(xticks=xx,xticklabels=k,ylabel='Source-page top-1 retrieval (%)',title='Page-majority filtering obscures minority-language content',ylim=(0,100));ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(O/'figure_language_filtering.png',dpi=180);plt.close(fig)

# Plot 3 corpuswide nonverbatim
fig,ax=plt.subplots(figsize=(8.2,4.5));labs=['IMVO isiXhosa','IMVO English','ISIG isiXhosa','ISIG English'];v=[cond[g]['rate']*100 for g in ('IMVO_xh','IMVO_en','ISIG_xh','ISIG_en')]
bars=ax.bar(labs,v);ax.bar_label(bars,labels=[f'{f:.2f}%' for f in v],padding=4);ax.set(ylabel='Share of language-view 10-word windows (%)',title='Sequences absent from the original raw page',ylim=(0,30));ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(O/'figure_cross_view_adjacency.png',dpi=180);plt.close(fig)

# Plot 4: Imvo source page stratum multi-match
boiler=json.loads((O/'boilerplate_control_summary.json').read_text());fig,ax=plt.subplots(figsize=(8.2,4.5))
x=np.arange(3);z=['p001','p002_p003','p004_plus'];w=.35
for j,(lan,lab) in enumerate([('xh','isiXhosa queries'),('en','English queries')]):
 val=[100*boiler['IMVO_'+lan+'_'+s]['multi_frac'] for s in z];b=ax.bar(x+(-w/2 if j==0 else w/2),val,w,label=lab);ax.bar_label(b,fmt='%.1f%%',padding=3,fontsize=9)
ax.set(xticks=x,xticklabels=['Page 1','Pages 2–3','Page 4+'],ylabel='Queries with multiple matching pages (%)',title='Much of Imvo passage ambiguity is page-position dependent',ylim=(0,105));ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(O/'figure_page_position.png',dpi=180);plt.close(fig)
