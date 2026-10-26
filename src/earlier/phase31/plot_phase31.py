"""Regenerate Phase31 scientific figures from saved results; no rerunning models."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parent
r=json.loads((base/'RESULTS.json').read_text())
groups=['EARLY_ISIG','LATE_IMVO']
labels=['Isigidimi 1870–1884','Imvo 1889–1894']
y=[r['groups'][g]['specialist_E_gain_pieces_per_100_chars'] for g in groups]
lo=[y[i]-r['groups'][g]['specialist_E_gain_95CI'][0] for i,g in enumerate(groups)]
hi=[r['groups'][g]['specialist_E_gain_95CI'][1]-y[i] for i,g in enumerate(groups)]
fig,ax=plt.subplots(figsize=(8.2,4.1))
ax.bar(labels,y,yerr=[lo,hi],capsize=5,width=.45)
ax.axhline(0,color='0.4',lw=.7)
ax.set_ylim(0,.7);ax.set_ylabel('Specialist BPE advantage (pieces / 100 characters)')
ax.set_title('Same frozen BPE models on independent publication years')
fig.tight_layout();fig.savefig(base/'phase31_out_of_period_specialization.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.2,4.3))
for group,label in zip(groups,labels):
 p=r['groups'][group]
 xs=[192,224,256,288,320]
 ys=[p['secondary_budgets'][str(x)]['specialist_gain'] if x!=256 else p['specialist_retained_gain_words_B256'] for x in xs]
 ax.plot(xs,ys,marker='o',label=label)
ax.axhline(0,color='0.4',lw=.7)
ax.axvline(256,color='0.5',ls='--',lw=1,label='Precommitted primary budget')
ax.set_xlabel('Model context budget (BPE pieces)')
ax.set_ylabel('Extra complete words retained (out of 90)')
ax.set_title('Operational effect depends on context budget')
ax.legend(frameon=False,fontsize=9)
fig.tight_layout();fig.savefig(base/'phase31_context_budget_effect.png',dpi=160);plt.close(fig)
