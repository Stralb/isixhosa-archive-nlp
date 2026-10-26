"""Reproducible paper figure; numeric constants verified against results/segmentation.csv and ARCHIVELENS_FOLLOWUP_2026-10-26.md."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
folder=Path(__file__).resolve().parent/'figures'
folder.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9.4,'axes.labelsize':8.2,'xtick.labelsize':7.5,'ytick.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'axes.spines.right':False,'axes.spines.top':False})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.2,3.0),gridspec_kw={'width_ratios':[1.05,1]})
labels=['BPE · English','Unigram · English','BPE · isiXhosa','Unigram · isiXhosa']
means=np.array([-1.877536696,-1.794836583,0.452589327,0.554912154])
lo=np.array([-1.971592296,-1.908135913,0.406844702,0.512723306])
hi=np.array([-1.781812237,-1.670614681,0.497374599,0.600412931])
y=np.array([3,2,1,0]); lims=np.vstack((means-lo,hi-means))
for j,(x,v,lim) in enumerate(zip(means,y,lims.T)):
 ax1.errorbar(x,v,xerr=lim.reshape(2,1),fmt='o',capsize=3,color=('0.15' if j<2 else '0.50'),markersize=4.6,elinewidth=1.5)
ax1.axvline(0,color='0.25',ls=':',lw=1)
ax1.set_yticks(y,labels)
ax1.set_xlim(-2.35,1.0)
ax1.set_xlabel('Allocation effect on pieces / 100 alphabetic characters')
ax1.set_title('A  |  Controlled tokenizer trade-off (1888)',loc='left',fontweight='bold')
ax1.set_ylim(-.3,3.6)
# Randomisation count reference display: min/max central 95% perm bands, observed and null centers.
labels2=['Imvo', 'Isigidimi']
nullmeans=[29.015203,8.301060]
los=[19,3];his=[40,16];observed=[23,4]
y2=np.array([1,0])
for yy,l,h,mu,obs in zip(y2,los,his,nullmeans,observed):
 ax2.hlines(yy,l,h,color='0.70',lw=8,zorder=1)
 ax2.plot(mu,yy,'o',color='0.42',ms=5,zorder=4)
 ax2.plot(obs,yy,'D',markerfacecolor='white',markeredgecolor='0.10',markeredgewidth=1.3,ms=5.6,zorder=5)
 ax2.text(h+1.3,yy, f'{obs} observed',ha='left',va='center',fontsize=7.5)
ax2.set_xlim(-1,49)
ax2.set_yticks(y2,labels2)
ax2.set_xlabel('Number of direction-reversal warnings / 120')
ax2.set_title('B  |  Audit warnings versus permutation reference',loc='left',fontweight='bold')
ax2.set_ylim(-.3,1.6)
fig.subplots_adjust(left=.15,right=.96,bottom=.22,top=.91,wspace=.49)
fig.savefig(folder/'main_evidence.pdf',bbox_inches='tight',pad_inches=.07)
fig.savefig(folder/'main_evidence.png',dpi=200,bbox_inches='tight',pad_inches=.07)
print('Saved',list(folder.iterdir()))
