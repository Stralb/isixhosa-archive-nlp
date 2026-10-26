#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
base=json.loads((root/'first_merge_all_seed_results.json').read_text())
data={r['seed']:r for r in base['paired']}
seeds=np.array(sorted(data))
nat=np.array([data[s]['natural_E'] for s in seeds])
ng=np.array([data[s]['ng_E'] for s in seeds])
wn=np.array([data[s]['word_start_n_E'] for s in seeds])
plt.figure(figsize=(11,5))
plt.plot(seeds,nat,'o-',label='Standard greedy BPE (observed first merge)',linewidth=1.4,markersize=5)
plt.plot(seeds,ng,'o-',label='Force first merge n + g',linewidth=1.4,markersize=4)
plt.plot(seeds,wn,'o-',label='Force first merge word-start + n',linewidth=1.4,markersize=4)
plt.xlabel('Independent Imvo-only training-sample seed')
plt.ylabel('Worst-newspaper unseen-form pieces / 100 characters')
plt.title('Identical sampled training text, different first BPE merge')
plt.xticks(seeds,rotation=60,fontsize=8)
plt.legend(fontsize=9)
plt.grid(axis='y',alpha=.3)
plt.tight_layout()
plt.savefig(root/'first_merge_causal_modes.png',dpi=190)
plt.close()

fresh=json.loads((root/'fresh_draw_validation_results.json').read_text())
r=sorted([z for z in fresh['records'] if z['mix']==1.],key=lambda z:z['seed'])
seeds=np.array([z['seed'] for z in r]);nat=np.array([z['natural_E'] for z in r]);sel=np.array([z['selected_E'] for z in r])
plt.figure(figsize=(11,5))
plt.plot(seeds,nat,'o--',label='Unmodified greedy BPE',markersize=4)
plt.plot(seeds,sel,'o-',label='Training-compression two-branch selection',markersize=4)
plt.axhline(39.5,linewidth=1,linestyle=':',label='Earlier high-cost diagnostic level (39.5)')
ix=np.argmax(sel)
plt.annotate('Residual high-cost sample',xy=(seeds[ix],sel[ix]),xytext=(seeds[ix]+1.5,39.45),arrowprops={'arrowstyle':'->'},fontsize=9)
plt.xticks(seeds,rotation=60,fontsize=8)
plt.xlabel('Previously unused training-sample seed (Imvo-only)')
plt.ylabel('Worst-newspaper unseen-form pieces / 100 characters')
plt.title('New-draw replication: 30 samples, training-only decisions')
plt.grid(axis='y',alpha=.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(root/'fresh_rule_new_draws.png',dpi=190)
plt.close()

ext=json.loads((root/'fresh_external_retro_results.json').read_text())
labels=['Ityala Lamawele','Zemk\'inkomo','Modern Vuk\'uzenzele']
keys=['ityala','zem','modern']
x=np.arange(len(keys));width=.25
nat=np.array([ext['corpora'][k]['random_natural_mean_E'] for k in keys]);sel=np.array([ext['corpora'][k]['training_only_selected_mean_E'] for k in keys]);bal=np.array([ext['corpora'][k]['balanced_mean_E'] for k in keys])
fig,ax=plt.subplots(figsize=(10,5))
for xx,vals,name in [(-width,nat,'Unmodified greedy BPE'),(0,sel,'Training-only selected'),(width,bal,'50/50 balanced BPE')]:
 bars=ax.bar(x+xx,vals,width,label=name)
 ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
ax.set_xticks(x,labels)
ax.set_ylabel('Unseen-form pieces / 100 alphabetic characters')
ax.set_title('Retrospective external transfer (three previously examined corpora)')
ax.set_ylim(40,49)
ax.legend(fontsize=9,loc='upper right')
ax.grid(axis='y',alpha=.25)
fig.tight_layout()
fig.savefig(root/'retrospective_external_comparison.png',dpi=190)
plt.close(fig)
print('PLOTS SAVED')
