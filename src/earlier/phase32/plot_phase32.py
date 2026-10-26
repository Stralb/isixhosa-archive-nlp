import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).parent
y=pd.read_csv(p/'yearly_visibility.csv')
fig,ax=plt.subplots(figsize=(10,4.8))
for key,label,style in [('IMVO','Imvo (archive English-view word share)','o-'),('ISIG','Isigidimi (archive English-view word share)','s-')]:
 a=y[y.source==key]
 ax.plot(a.year,a.word_share*100,style,label=label,linewidth=2,markersize=4)
ax.axvspan(1885,1888,alpha=.09,color='grey',label='Shared publication years (missing Isigidimi 1886)')
ax.set_xlabel('Edition year');ax.set_ylabel('English-view share of classified word mass (%)')
ax.set_title('Historical newspaper text allocations: 1870–1894')
ax.set_xlim(1869,1895);ax.set_ylim(bottom=0);ax.grid(True,axis='y',alpha=.2)
ax.legend(loc='upper right',fontsize=8);fig.tight_layout();fig.savefig(p/'phase32_historical_visibility.png',dpi=180);plt.close(fig)
e=pd.read_csv(p/'sampling_estimands.csv');d=e[e.period=='overlap_1885_1887_1888']
fig,ax=plt.subplots(figsize=(8,4.8))
labels=['Imvo','Isigidimi']; xx=range(2);w=.28
ax.bar([i-w for i in xx],d.word_weighted.values*100,width=w,label='Word-weighted')
ax.bar(xx,d.page_weighted.values*100,width=w,label='Page-weighted')
ax.bar([i+w for i in xx],d.issue_weighted.values*100,width=w,label='Issue-weighted')
ax.set_xticks(list(xx),labels);ax.set_ylabel('English-view text (%)');ax.set_ylim(0,36)
ax.set_title('What “language balance” means depends on the sampling unit')
ax.legend();ax.grid(True,axis='y',alpha=.2);fig.tight_layout();fig.savefig(p/'phase32_sampling_estimands.png',dpi=180);plt.close(fig)
