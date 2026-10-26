#!/usr/bin/env python3
"""Regenerate Phase27 figures from evaluated tables; no new results."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).parent
T=pd.read_csv(D/'transfer_summary.csv')
S=T[T.masked==0].set_index('model').loc[['char','word','bpe0','bpe50','bpe100']]
labels=['Character','Word','BPE: 0% Imvo','BPE: 50% Imvo','BPE: 100% Imvo']
x=np.arange(len(S))
width=.35
fig,ax=plt.subplots(figsize=(10,5.4))
ax.bar(x-width/2,100*S.within,width,label='Within newspaper')
ax.bar(x+width/2,100*S.cross,width,label='Across newspapers')
ax.axhline(50,linestyle='--',linewidth=1,alpha=.7,label='Chance (50%)')
ax.set(ylim=(0,81),ylabel='Issue-balanced year accuracy (%)',
       title='Period classification: same-newspaper versus cross-newspaper transfer')
ax.set_xticks(x,labels,rotation=13,ha='right')
ax.legend(loc='upper right',frameon=False,ncols=3)
for i,r in enumerate(S.itertuples()):
    ax.text(i-width/2,100*r.within+1,f'{100*r.within:.1f}%',ha='center',fontsize=9)
    ax.text(i+width/2,100*r.cross+1,f'{100*r.cross:.1f}%',ha='center',fontsize=9)
fig.tight_layout()
fig.savefig(D/'phase27_period_transfer.png',dpi=180)
plt.close(fig)
Y=pd.read_csv(D/'future_yearly_scores.csv')
Y=Y[Y.masked==0]
S=Y.groupby('year').agg(mean=('avg_advantage','mean'),
                        lowest=('avg_advantage','min'),
                        highest=('avg_advantage','max')).reset_index()
fig,ax=plt.subplots(figsize=(8.5,4.9))
ax.plot(S.year,S['mean'],marker='o',linewidth=2,label='Mean across 3 tokenizer seeds')
ax.fill_between(S.year,S.lowest,S.highest,alpha=.17,label='Range across seeds')
ax.axvline(1891.5,linestyle='--',linewidth=1,alpha=.7,label='Predefined period split')
ax.axhline(0,linestyle=':',linewidth=1)
ax.set(xlabel='Imvo publication year',ylabel='Additional pieces per 100 characters\nwith Isigidimi-only versus Imvo-only BPE',
       title='Future-year test: no sustained growth in specialist BPE advantage',
       xticks=list(range(1889,1895)),ylim=(-.1,2.0))
ax.legend(frameon=False,loc='upper right')
fig.tight_layout()
fig.savefig(D/'phase27_future_tokenizer_drift.png',dpi=180)
plt.close(fig)
print('PLOTS_OK')
