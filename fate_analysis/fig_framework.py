import common  # sets the working folder to FATE_OUT
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8.5})
BLUE,ORANGE,INK,MUTE='#2a78d6','#eb6834','#222222','#555555'
fig=plt.figure(figsize=(180/25.4,108/25.4)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,180); ax.set_ylim(0,108); ax.axis('off')
def box(x,y,w,h,fc,ec,lw=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2.2',fc=fc,ec=ec,lw=lw))
def arrow(x0,y0,x1,y1):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=9,color=MUTE,lw=1))
# input strip
box(4,93,172,12,'#f3f6fb','#9db8dc')
ax.text(8,99,'Input',weight='bold',va='center',color=BLUE,fontsize=9)
ax.text(24,99,'Wearable recordings  +  a fatigue label attached to a labelled unit (session, task block or trial)\n'
        '+  participant identifiers  +  a candidate proxy for the label (time of day, task phase, preceding activity)',va='center',fontsize=8,color=INK,linespacing=1.35)
cols=[('C1','Partition ladder',
       ['L1  random samples','L2  random time blocks','L3  held-out sessions','L4  held-out participants','     (+ LOSO when n is small)','','Same model and features','at every level; overlap','checked in every fold']),
      ('C2','Unit-level metrics',
       ['Pooled OOF ROC-AUC,','  scores ranked within fold','Unit-level ROC-AUC','Participant-bootstrap 95% CI','Unit-label permutation p','Holm correction across models','Repeated fold assignments','Nested (grouped) tuning']),
      ('C3','Confound attribution',
       ['LII  leakage inflation index','   = AUC(L1) − AUC(L4)','CDI  confound dominance index','   = AUC(proxy) − AUC(label)','TS-AUC  mean unit-level AUC','   within proxy strata','Label effect adjusted for proxy','   (fixed effects, clustered SE)']),
      ('C4','Grading',
       ['Tier 0  leaky evaluation only','Tier 1  no evidence at L4','Tier 2  generalises, but','          attributable to proxy','Tier 3  fatigue-attributable','','Eight-item checklist','Open toolkit (fate/)'])]
x0,w,gap,y0,h=4,40,4,20,66
for i,(cid,title,lines) in enumerate(cols):
    x=x0+i*(w+gap); acc=ORANGE if cid=='C3' else BLUE
    box(x,y0,w,h,'white',acc,1.2)
    ax.add_patch(FancyBboxPatch((x,y0+h-12),w,12,boxstyle='round,pad=0,rounding_size=2.2',fc=acc,ec=acc))
    ax.text(x+3,y0+h-6,f'{cid}',color='white',weight='bold',va='center',fontsize=9.5)
    ax.text(x+11,y0+h-6,title,color='white',va='center',fontsize=9)
    for j,l in enumerate(lines): ax.text(x+3,y0+h-18-j*5.3,l,va='center',fontsize=7.8,color=INK)
    if i<3: arrow(x+w+0.3,y0+h/2,x+w+gap-0.3,y0+h/2)
    arrow(x+w/2,93,x+w/2,y0+h+0.3)
box(4,3,172,13,'#fdf3ee','#f0b394')
ax.text(8,9.5,'Output',weight='bold',va='center',color=ORANGE,fontsize=9)
ax.text(24,9.5,'For each model: LII, CDI, TS-AUC and a FATE tier. Tier 3 requires L4 significance after Holm correction,\n'
        'a significant TS-AUC and CDI ≤ 0; a published accuracy obtained at L1–L2 is reported as Tier 0 until re-evaluated.',va='center',fontsize=8,color=INK,linespacing=1.35)
for i in range(4): arrow(x0+i*(w+gap)+w/2,y0-0.3,x0+i*(w+gap)+w/2,16.3)
fig.savefig('figs/fig_framework.png',dpi=300); fig.savefig('figs/fig_framework.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
