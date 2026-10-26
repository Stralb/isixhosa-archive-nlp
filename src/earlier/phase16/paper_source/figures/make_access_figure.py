import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out=Path(__file__).with_name('language_filter_access.png')
# Shares are exact Phase 15 archived analysis, not observations of a deployed interface.
names=['Imvo','Isigidimi']
ret=[6.647,20.774]
hidden=[93.353,79.226]
fig,ax=plt.subplots(figsize=(6.3,2.05),dpi=210)
ax.barh(names,ret,label='Retained',height=.48)
ax.barh(names,hidden,left=ret,label='Excluded',height=.48)
for y,(r,h) in enumerate(zip(ret,hidden)):
 ax.text(r+h*.50,y,f'{h:.1f}% excluded',ha='center',va='center',fontsize=10.2,color='black')
ax.set_xlim(0,103)
ax.invert_yaxis()
ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Share of English word occurrences (%)',fontsize=9.5)
ax.tick_params(axis='both',labelsize=9.5)
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=9.5)
for side in ['top','right','left']:ax.spines[side].set_visible(False)
fig.tight_layout(pad=.5)
fig.savefig(out,bbox_inches='tight')
print(out)
