#!/usr/bin/env python3
"""Verifiable, editable vector plots for Whose Words Count? (2026).

All values are copied without rounding from saved outputs associated with pinned
GitHub commit aff05ae34597f28309dc94561adbcd962901e2f4.
Sources: results/segmentation.csv, results/bigram.csv,
results/archivelens/estimands.csv, results/archivelens/followup/summary.json.

Do not replace this script with AI-generated image crops: those introduced wrong
XH-language names and incorrect equal-issue/equal-month values.
"""
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

OUT=Path(__file__).resolve().parent/'figures'
OUT.mkdir(parents=True, exist_ok=True)

EN='#365875'; XH='#ad653e'; INK='#293139'; GREY='#67737d'; PALE='#e5e9ed'; LITE='#c8d0d5'
mpl.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Liberation Sans','DejaVu Sans'],
    'font.size':8.0,'axes.titlesize':8.6,'axes.labelsize':7.8,
    'xtick.labelsize':7.1,'ytick.labelsize':7.5,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'figure.facecolor':'white','savefig.facecolor':'white',
    'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':GREY,
    'xtick.color':INK,'ytick.color':INK,
})

def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(.65)
    ax.tick_params(axis='y',length=0,pad=4)
    ax.tick_params(axis='x',length=2.5,width=.65,pad=3)

def interval(ax,x,lo,hi,y,color,open_marker=False,size=3.9):
    ax.plot([lo,hi],[y,y],lw=1.6,c=color,solid_capstyle='butt',zorder=3)
    ax.vlines([lo,hi],y-.13,y+.13,linewidth=1.0,color=color,zorder=3)
    ax.scatter([x],[y],s=size**2,marker='o',c=['white' if open_marker else color],
               edgecolors=color,linewidths=1.2,zorder=5)

# Figure 1: two synchronized full-range coefficient plots with exact numeric
# annotations. CI marks for XH are smaller in absolute units; values are shown
# exactly instead of using distorted/asymmetric axes.
items_a=[
 ('EN · BPE', -1.8775366962678142,-1.9715922955449294,-1.7818122365909044,'EN','-1.88','[-1.97, -1.78]'),
 ('EN · Unigram',-1.794836583356222,-1.908135912503281,-1.6706146811067157,'EN','-1.79','[-1.91, -1.67]'),
 ('XH · BPE',.452589326840824,.4068447015743001,.49737459942842027,'XH','+0.45','[+0.41, +0.50]'),
 ('XH · Unigram',.5549121537145503,.5127233062482931,.6004129314634647,'XH','+0.55','[+0.51, +0.60]'),
]
items_b=[
 ('EN · allocation',-.03721665481863443,-.04236444306237788,-.03211883749193364,'EN','-0.037','[-.042, -.032]'),
 ('EN · page choice', .019378108713970388,.01145893388190681,.027686052238209285,'EN','+0.019','[+.011, +.028]'),
 ('XH · allocation',-.002746012320800206,-.005233731543068796,-.0001411713905480318,'XH','-0.0027','[-.0052, -.0001]'),
 ('XH · page choice',.009836995568404606,.004841549194049428,.015236437871103249,'XH','+0.0098','[+.0048, +.0152]'),
]
fig,(a,b)=plt.subplots(1,2,figsize=(7.15,1.80),dpi=230)
fig.subplots_adjust(left=.145,right=.983,top=.77,bottom=.23,wspace=.80)

def draw_coefficients(ax,data,lim,ticks,axislabel,title):
    ax.set_xlim(*lim);ax.set_ylim(-.45,3.48)
    ax.set_yticks([3,2,1,0],[v[0] for v in data]);ax.set_xticks(ticks)
    ax.axvline(0,c=GREY,lw=.7,ls=(0,(2.0,2.0)),zorder=0)
    ax.axhline(1.5,c=PALE,lw=.6,zorder=0)
    clean_axis(ax);ax.set_xlabel(axislabel,labelpad=2.7)
    ax.set_title(title,loc='left',pad=6,fontweight='bold')
    for i,(label,v,lo,hi,view,vtext,ci) in enumerate(data):
        y=3-i;color=EN if view=='EN' else XH
        interval(ax,v,lo,hi,y,color,open_marker=(view=='XH'),size=3.7)
        # Place direct labels on the SAME row, immediately after each whisker.
        # This keeps every label well below the title and prevents row overlap.
        ax.annotate(vtext, (hi,y), xytext=(4.0,0), textcoords='offset points',
                    ha='left', va='center', fontsize=6.25, color=color,
                    fontweight='bold', annotation_clip=False)

draw_coefficients(a,items_a,(-2.37,1.08),[-2,-1,0,1],
                  'Difference in pieces / 100 chars','(a) Segmentation compactness')
draw_coefficients(b,items_b,(-.064,.047),[-.06,-.04,-.02,0,.02,.04],
                  'Difference in bigram loss (bits / char)','(b) Held-out predictive loss')
b.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: '0' if abs(x)<1e-9 else f'{x:+.2f}'))
for ext in ('pdf','png','svg'):
    fig.savefig(OUT/f'tokenizer_contrasts.{ext}',dpi=280)
plt.close(fig)

# Figure 2. Panel (a) has a discrete null reference interval, NOT a confidence
# interval for the observed count. Panel (b) shows ALL FOUR independently
# calculated estimands with their correct bootstrap intervals.
fig=plt.figure(figsize=(3.43,2.30),dpi=230)
a=fig.add_axes([.265,.645,.68,.18])
b=fig.add_axes([.305,.17,.64,.24])
fig.text(.04,.964,'(a) Reversal counts and permutation null',fontsize=8.05,
         fontweight='bold',ha='left',va='top')
fig.text(.04,.505,"(b) Imvo: 'ants' rate change",fontsize=8.05,
         fontweight='bold',ha='left',va='top')

# null central 95%, mean, observed, per source
nulls=[('Imvo', 23,29.01520304060812,19,40),
       ('Isigidimi',4,8.301060212042408,3,16)]
for i,(name,obs,mean,lo,hi) in enumerate(nulls):
    y=1-i
    a.add_patch(Rectangle((lo,y-.135),hi-lo,.27,facecolor=PALE,
                          edgecolor=LITE,lw=.7,zorder=1))
    a.vlines(mean,y-.18,y+.18,colors=GREY,lw=1.35,zorder=2)
    a.scatter([obs],[y],marker='D',s=25,color=INK,zorder=4)
    a.annotate(str(obs),(obs,y),xytext=(6,0),textcoords='offset points',
               fontsize=7.0,ha='left',va='center',color=INK,fontweight='bold')
a.set_xlim(0,51);a.set_ylim(-.48,1.45)
a.set_yticks([1,0],['Imvo','Isigidimi'])
a.set_xticks([0,10,20,30,40,50])
a.set_xlabel('Reversals (120 patterns)',labelpad=2)
clean_axis(a)
# The caption describes observed diamonds, shaded permutation range, and
# null-mean ticks. Omit the cramped in-plot legend to prevent collision.

ants=[
 ('Character mass',-.22773153,-2.89001124,2.42839931,'-0.23'),
 ('Equal page',4.79883694,2.08522937,7.51964975,'+4.80'),
 ('Equal issue',-.03090896,-2.63636400,2.61352611,'-0.03'),
 ('Equal month',.15104653,-2.46600021,2.69866814,'+0.15'),
]
b.set_xlim(-4.15,8.25);b.set_ylim(-.5,3.55)
b.set_yticks([3,2,1,0],[p[0] for p in ants]);b.set_xticks([-4,0,4,8])
b.axvline(0,c=GREY,lw=.7,ls=(0,(2,2)),zorder=0)
b.set_xlabel('Change / 10,000 eligible positions',labelpad=2)
clean_axis(b)
for i,(name,v,lo,hi,vtext) in enumerate(ants):
    y=3-i
    clr=EN if name=='Equal page' else INK
    interval(b,v,lo,hi,y,clr,open_marker=False,size=4.0)
    # Exact values give the near-zero estimates visible meaning. CI endpoints
    # are expressed by whiskers; see figure caption for interval definition.
    b.annotate(vtext,(v,y),xytext=(0,5.6),textcoords='offset points',
               fontsize=6.3,color=clr,ha='center',va='bottom',fontweight='bold')

for ext in ('pdf','png','svg'):
    fig.savefig(OUT/f'archivelens_audit.{ext}',dpi=290)
plt.close(fig)
print('Figures saved under',OUT)
