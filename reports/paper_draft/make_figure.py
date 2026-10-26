"""One integrated, three-panel scientific figure generated solely from verified results.
Source: original sampling contrasts in results/{segmentation,bigram}.csv and
ArchiveLens follow-up results in results/archivelens/followup/summary.json.
The numeric constants are explicitly pinned and checked in tests.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

base = Path(__file__).resolve().parent
out = base / 'figures'
out.mkdir(parents=True, exist_ok=True)
# Paper-ready restrained colours; grayscale distinction retained via marker shapes.
navy='#175b78'; rust='#a15c3e'; ink='#252b30'; grey='#bfc5c9'
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':7.6, 'axes.titlesize':8.8,
    'axes.labelsize':7.4, 'xtick.labelsize':6.9, 'ytick.labelsize':7.4,
    'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.right':False,
    'axes.spines.top':False, 'axes.linewidth':.7,
})
fig=plt.figure(figsize=(9.5,2.75), facecolor='white')
# Exactly one figure with three connected evidence panels, not overlapping axes.
gs=fig.add_gridspec(1,3,left=.085,right=.988,bottom=.20,top=.79,wspace=.66,width_ratios=[1.18,1.1,1.04])
a,b,c=[fig.add_subplot(gs[0,i]) for i in range(3)]
fig.text(.265,.965,'CONTROLLED COMPUTATIONAL INTERVENTION',ha='center',va='top',size=7.0,weight='bold',color=navy)
fig.text(.820,.965,'OBSERVATIONAL CLAIM AUDIT',ha='center',va='top',size=7.0,weight='bold',color=rust)
fig.lines.append(Line2D([.09,.60],[.917,.917],color=navy,lw=.8,transform=fig.transFigure))
fig.lines.append(Line2D([.676,.987],[.917,.917],color=rust,lw=.8,transform=fig.transFigure))

# A: source-window allocation A main effect (signed pieces / 100 alphabetic chars)
labels=['BPE / EN', 'Unigram / EN', 'BPE / XH', 'Unigram / XH']
x=np.array([-1.8775366962678,-1.7948365833562, .4525893268408, .5549121537146])
lo=np.array([-1.9715922955449,-1.9081359125033,.4068447015743,.5127233062483])
hi=np.array([-1.7818122365909,-1.6706146811067,.4973745994284,.6004129314635])
y=np.arange(4)[::-1]
a.axvline(0,lw=.8,ls=':',color=ink,zorder=0)
for j in range(4):
    color=navy if j<2 else rust
    a.errorbar([x[j]],[y[j]],xerr=np.array([[x[j]-lo[j]],[hi[j]-x[j]]]),
            fmt='o',ms=4.1,capsize=2.3,lw=1.4,color=color,zorder=4)
a.set_xlim(-2.28,.95);a.set_ylim(-.55,3.55)
a.set_yticks(y,labels)
a.tick_params(axis='y',length=0,pad=4)
a.set_xticks([-2,-1,0,1]);a.tick_params(axis='x',length=2.5)
a.set_xlabel('Allocation effect: pieces / 100 chars',labelpad=6)
a.set_title('A  Segmentation trade-off',loc='left',pad=10,weight='bold')
# B: real downstream mechanism of unequal factor contributions + year cluster sensitivity
labels_b=['Allocation / EN', 'Page choice / EN', 'Whole policy / 1888', 'Whole / 1889–94']
vals=np.array([-.0372166548186,.019378108714,-.017838546105,.00450329190034])
lo_b=np.array([-.0423644430624,.0114589338819,-.0256331291705,-.00268175909286])
hi_b=np.array([-.0321188374919,.0276860522382,-.0101906519834,.01305019747896])
yb=np.arange(4)[::-1]
b.axvline(0,lw=.8,ls=':',color=ink,zorder=0)
for i in range(4):
    color=navy if vals[i]<0 else rust
    marker='s' if i==3 else 'o'
    b.errorbar([vals[i]],[yb[i]],xerr=np.array([[vals[i]-lo_b[i]],[hi_b[i]-vals[i]]]),
               fmt=marker,ms=3.9,capsize=2.2,lw=1.3,color=color,zorder=4)
b.set_xlim(-.052,.038);b.set_ylim(-.55,3.55)
b.set_yticks(yb,labels_b);b.tick_params(axis='y',length=0,pad=4)
b.set_xticks([-.04,-.02,0,.02],['−.04','−.02','0','+.02']);b.tick_params(axis='x',length=2.5)
b.set_xlabel('BPE bigram effect: bits / char',labelpad=6)
b.set_title('B  Downstream check',loc='left',pad=10,weight='bold')
# C: permutation count interval vs observed, and separate Holm estimator-gap count.
labels_c=['Imvo', 'Isigidimi']
cy=np.array([1,0]); null_lo=np.array([19,3]); null_hi=np.array([40,16]); null_avg=np.array([29.015203,8.301060]); obs=np.array([23,4]); holm=np.array([29,0])
for j in range(2):
    yy=cy[j]
    c.plot([null_lo[j],null_hi[j]],[yy,yy],color=grey,lw=7,zorder=1,solid_capstyle='round')
    c.plot([null_avg[j]],[yy],marker='|',ms=11, mew=1.6,color='#697178',ls='none',zorder=3)
    c.plot([obs[j]],[yy],marker='D',ms=4.6,mec=ink,mfc='white',mew=1.1,ls='none',zorder=5)
    c.plot([holm[j]],[yy-.23],marker='s',ms=4.7,mec=rust,mfc=rust,ls='none',zorder=6)
c.set_xlim(-4,48);c.set_ylim(-.45,1.55)
c.set_yticks(cy,labels_c);c.tick_params(axis='y',length=0,pad=3)
c.set_xticks([0,10,20,30,40]);c.tick_params(axis='x',length=2.5)
c.set_xlabel('Counts / 120 preselected 4-grams',labelpad=6)
c.set_title('C  Calibrated claim audit',loc='left',pad=10,weight='bold')

fig.savefig(out/'integrated_evidence.pdf',bbox_inches='tight',pad_inches=.05)
fig.savefig(out/'integrated_evidence.png',dpi=240,bbox_inches='tight',pad_inches=.05)
plt.close(fig)
print('wrote integrated figure:',[(p.name,p.stat().st_size) for p in out.glob('integrated_evidence.*')])
