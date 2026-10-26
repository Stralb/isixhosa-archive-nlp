#!/usr/bin/env python3
"""Final research figures. Values independently checked against frozen outputs.
Reference commit: aff05ae34597f28309dc94561adbcd962901e2f4.
Do not interpret the plotted bootstrap intervals as Holm-adjusted inference.
"""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
P=Path(__file__).parent/"figures"
P.mkdir(exist_ok=True)
EN="#365875"; XH="#ad653e"; INK="#293139"; GREY="#67737d"; PALE="#e5e9ed"; LIGHT="#c8d0d5"
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Liberation Sans","DejaVu Sans"],"font.size":8,"axes.titlesize":8.6,"axes.labelsize":7.8,"xtick.labelsize":7.1,"ytick.labelsize":7.5,"pdf.fonttype":42,"svg.fonttype":"none"})

def clean(ax):
    for side in ["top","left","right"]: ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(.65)
    ax.tick_params(axis="y",length=0,pad=4)
    ax.tick_params(axis="x",length=2.5,width=.65,pad=3)
def mark(ax,v,lo,hi,y,col,outline=False):
    ax.hlines(y,lo,hi,lw=1.55,colors=col,zorder=3)
    ax.vlines([lo,hi],y-.13,y+.13,lw=1,color=col,zorder=3)
    ax.scatter([v],[y],s=16,marker="o",facecolor="white" if outline else col,edgecolor=col,lw=1.2,zorder=5)
seg=[
("EN - BPE",-1.8775366962678142,-1.9715922955449294,-1.7818122365909044,"EN","-1.88"),
("EN - Unigram",-1.794836583356222,-1.908135912503281,-1.6706146811067157,"EN","-1.79"),
("XH - BPE",.452589326840824,.4068447015743001,.49737459942842027,"XH","+0.45"),
("XH - Unigram",.5549121537145503,.5127233062482931,.6004129314634647,"XH","+0.55")]
bpc=[
("EN - allocation",-.03721665481863443,-.04236444306237788,-.03211883749193364,"EN","-0.037"),
("EN - page choice",.019378108713970388,.01145893388190681,.027686052238209285,"EN","+0.019"),
("XH - allocation",-.002746012320800206,-.005233731543068796,-.0001411713905480318,"XH","-0.0027"),
("XH - page choice",.009836995568404606,.004841549194049428,.015236437871103249,"XH","+0.0098")]
fig,(a,b)=plt.subplots(1,2,figsize=(7.15,1.80),dpi=220)
fig.subplots_adjust(left=.145,right=.983,top=.77,bottom=.23,wspace=.80)
for ax,vals,xlim,xticks,xlab,title in [
(a,seg,(-2.37,1.08),[-2,-1,0,1],"Difference in pieces / 100 chars","(a) Segmentation compactness"),
(b,bpc,(-.064,.047),[-.06,-.04,-.02,0,.02,.04],"Difference in bigram loss (bits / char)","(b) Held-out predictive loss")]:
    ax.set(xlim=xlim,ylim=(-.45,3.48))
    ax.set_yticks([3,2,1,0],[v[0] for v in vals]);ax.set_xticks(xticks)
    ax.axvline(0,color=GREY,lw=.7,ls=(0,(2,2)),zorder=0)
    ax.axhline(1.5,color=PALE,lw=.6,zorder=0);clean(ax)
    ax.set_xlabel(xlab,labelpad=2.7);ax.set_title(title,loc="left",pad=6,fontweight="bold")
    for i,(label,v,lo,hi,view,shown) in enumerate(vals):
        y=3-i;col=EN if view=="EN" else XH
        mark(ax,v,lo,hi,y,col,outline=view=="XH")
        ax.annotate(shown,(hi,y),xytext=(4,0),textcoords="offset points",ha="left",va="center",fontsize=6.25,color=col,fontweight="bold",annotation_clip=False)
b.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: "0" if abs(x)<1e-9 else f"{x:+.2f}"))
for ext in ("pdf","png","svg"):fig.savefig(P/f"tokenizer_contrasts.{ext}",dpi=280)
plt.close(fig)
fig=plt.figure(figsize=(3.43,2.30),dpi=220)
a=fig.add_axes([.265,.645,.68,.18]);b=fig.add_axes([.305,.17,.64,.24])
fig.text(.04,.964,"(a) Reversal counts and permutation null",fontsize=8.05,fontweight="bold",ha="left",va="top")
fig.text(.04,.505,"(b) Imvo: 'ants' rate change",fontsize=8.05,fontweight="bold",ha="left",va="top")
for y,(label,obs,avg,lo,hi) in zip([1,0],[("Imvo",23,29.01520304060812,19,40),("Isigidimi",4,8.301060212042408,3,16)]):
    a.add_patch(Rectangle((lo,y-.135),hi-lo,.27,facecolor=PALE,edgecolor=LIGHT,lw=.7,zorder=1))
    a.vlines(avg,y-.18,y+.18,color=GREY,lw=1.35,zorder=2)
    a.scatter([obs],[y],marker="D",s=25,color=INK,zorder=4)
    a.annotate(str(obs),(obs,y),xytext=(6,0),textcoords="offset points",fontsize=7,ha="left",va="center",color=INK,fontweight="bold")
a.set(xlim=(0,51),ylim=(-.48,1.45));a.set_yticks([1,0],["Imvo","Isigidimi"]);a.set_xticks([0,10,20,30,40,50]);a.set_xlabel("Reversals (120 patterns)",labelpad=2);clean(a)
ants=[
("Character mass",-.22773153,-2.89001124,2.42839931,"-0.23"),
("Equal page",4.79883694,2.08522937,7.51964975,"+4.80"),
("Equal issue",-.03090896,-2.63636400,2.61352611,"-0.03"),
("Equal month",.15104653,-2.46600021,2.69866814,"+0.15")]
b.set(xlim=(-4.15,8.25),ylim=(-.5,3.55))
b.set_yticks([3,2,1,0],[x[0] for x in ants]);b.set_xticks([-4,0,4,8])
b.axvline(0,color=GREY,lw=.7,ls=(0,(2,2)),zorder=0)
b.set_xlabel("Change / 10,000 eligible positions",labelpad=2);clean(b)
for i,(label,v,lo,hi,shown) in enumerate(ants):
    col=EN if label=="Equal page" else INK
    mark(b,v,lo,hi,3-i,col)
    b.annotate(shown,(v,3-i),xytext=(0,5.6),textcoords="offset points",ha="center",va="bottom",fontsize=6.3,color=col,fontweight="bold")
for ext in ("pdf","png","svg"):fig.savefig(P/f"archivelens_audit.{ext}",dpi=290)
plt.close(fig)
