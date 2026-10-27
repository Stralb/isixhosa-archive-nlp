from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent
r=json.loads((D/'RESULTS.json').read_text())
mask=json.loads((D/'cue_mask_matched.json').read_text())
names=['Full 1888','Month + lexical match','Matched + cue removal']
values=[100*r['classification']['char_wb_3_5']['full']['macro'],100*r['classification']['char_wb_3_5']['matched']['macro'],100*mask['macro']]
fig,ax=plt.subplots(figsize=(8,4.5))
ax.bar(names,values)
ax.axhline(50,ls='--',lw=1,label='Balanced-chance reference')
for i,v in enumerate(values):ax.text(i,v+1.1,f'{v:.1f}%',ha='center',fontsize=10)
ax.set_ylabel('Issue-balanced newspaper-source accuracy (%)')
ax.set_ylim(0,100)
ax.set_title('Phase 28: source signatures after lexical matching')
ax.tick_params(axis='x',labelsize=9)
ax.legend(loc='upper right',fontsize=9)
fig.tight_layout();fig.savefig(D/'phase28_source_classification.png',dpi=160);plt.close(fig)
rng=r['bpe'];sources=['Imvo','Isigidimi'];full=[rng['full_savings'][s] for s in ('IMVO','ISIG')];matched=[rng['matched_savings'][s] for s in ('IMVO','ISIG')]
import numpy as np
x=np.arange(2);width=.36
fig,ax=plt.subplots(figsize=(7,4.4))
ax.bar(x-width/2,full,width,label='Full 1888')
ax.bar(x+width/2,matched,width,label='Lexically matched')
ax.axhline(0,color='grey',lw=.8)
for i in range(2):
 ax.text(x[i]-width/2,full[i]+.035,f'{full[i]:.2f}',ha='center',fontsize=9)
 ax.text(x[i]+width/2,matched[i]+.035,f'{matched[i]:.2f}',ha='center',fontsize=9)
ax.set_ylabel('Matching-source BPE advantage (pieces / 100 letters)')
ax.set_title('Phase 28: tokenizer source-specialization persists')
ax.set_xticks(x,sources);ax.legend();ax.set_ylim(0,1.4)
fig.tight_layout();fig.savefig(D/'phase28_bpe_savings.png',dpi=160);plt.close(fig)
