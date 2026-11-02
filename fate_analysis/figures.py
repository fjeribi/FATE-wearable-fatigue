"""Figures 2-4 of the FATE paper (Figure 1: fig_framework.py). Run after fate_metrics.py and mefar_confound_all.py."""
import common  # sets the working folder to FATE_OUT
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'Liberation Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#555','axes.labelcolor':'#222','xtick.color':'#444','ytick.color':'#444'})
BLUE,ORANGE,GREY='#2a78d6','#eb6834','#b8b8b8'
PUBM=['Jampala RF','Jampala LR','Jampala KNN','Nooh GCA']
SHORT={'Logistic Regression':'LR','Gaussian Naive Bayes':'GNB','KNN':'KNN','Random Forest':'RF','XGBoost':'XGBoost','MLP':'MLP',
       'Jampala RF':'Jampala RF*','Jampala LR':'Jampala LR*','Jampala KNN':'Jampala KNN*','Nooh GCA':'Nooh GCA*'}
# ---------- Figure 2: ladder ----------
L=pd.concat([pd.read_csv('ladder_mefar.csv'),pd.read_csv('ladder_fs.csv')])
cfg={'mefar':(['random_sample','random_60s_chunk','held_out_session','held_out_participant'],
              ['L1 Random\nseconds','L2 Random\n60-s chunks','L3 Held-out\nsessions','L4 Held-out\nparticipants'],'(a) MEFAR (23 workers, 1-Hz)'),
     'fs':(['random_sample','held_out_block','held_out_session','held_out_participant'],
           ['L1 Random\nwindows','L2 Held-out\ntest blocks','L3 Held-out\nsessions','L4 Held-out\nparticipants'],'(b) FatigueSet (12 participants, 30-s windows)')}
fig,axs=plt.subplots(1,2,figsize=(180/25.4,3.1),sharey=True)
for ax,(ds,(rungs,labels,title)) in zip(axs,cfg.items()):
    t=L[L.dataset==ds].pivot(index='model',columns='rung',values='pooled_auc')[rungs]; x=np.arange(len(rungs))
    for m,row in t.iterrows():
        pub=m in PUBM
        ax.plot(x,row.values,color=ORANGE if pub else GREY,lw=1.2 if pub else 1,marker='s' if pub else 'o',ms=3.2,zorder=2 if pub else 1,alpha=0.95)
    med=t.median(); ax.plot(x,med.values,color=BLUE,lw=2,marker='o',ms=6,zorder=3,mec='white',mew=1.2)
    for xi,v in zip(x,med.values): ax.annotate(f'{v:.2f}',(xi,v),textcoords='offset points',xytext=(13,-12),ha='center',color=BLUE,fontsize=8,weight='bold',bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.9),zorder=5)
    ax.axhline(0.5,color='#777',lw=0.8,ls=(0,(4,3))); ax.text(-0.25,0.49,'chance',fontsize=7.5,color='#555',ha='left',va='top')
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=7.8); ax.set_ylim(0.38,1.03); ax.set_title(title,fontsize=9.5,loc='left')
    ax.grid(axis='y',color='#e6e6e6',lw=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel('Pooled out-of-fold ROC-AUC')
fig.legend([Line2D([],[],color=BLUE,lw=2,marker='o',ms=5),Line2D([],[],color=GREY,lw=1,marker='o',ms=3),Line2D([],[],color=ORANGE,lw=1.2,marker='s',ms=3)],
           ['Median of ten models','Reference classifier (LR, GNB, KNN, RF, XGBoost, MLP)','Re-implemented published pipeline'],loc='lower center',ncol=3,frameon=False,fontsize=7.8,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig('figs/fig_ladder.png',dpi=300); fig.savefig('figs/fig_ladder.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
# ---------- Figure 3: confound attribution ----------
C=pd.read_csv('mefar_confound_all.csv'); FI=pd.read_csv('fate_indices.csv')
fig=plt.figure(figsize=(180/25.4,4.6)); gs=fig.add_gridspec(1,3,width_ratios=[1.35,1,1],wspace=0.55)
ax=fig.add_subplot(gs[0]); C=C.sort_values('pooled'); y=np.arange(len(C))
ax.hlines(y,C[['morning_sess','evening_sess']].min(axis=1),C[['morning_sess','evening_sess']].max(axis=1),color='#dcdcdc',lw=3,zorder=1)
ax.scatter(C.morning_sess,y,color=ORANGE,s=26,zorder=3,edgecolor='white',lw=0.8,label='Morning sessions')
ax.scatter(C.evening_sess,y,color=BLUE,s=26,zorder=3,edgecolor='white',lw=0.8,marker='s',label='Evening sessions')
lab=[SHORT.get(m,m).replace(' (paper run)',' (earlier)') for m in C.model]
ax.set_yticks(y); ax.set_yticklabels(lab,fontsize=7.3); ax.axvline(0.5,color='#777',lw=0.8,ls=(0,(4,3)))
ax.set_xlabel('Session-level ROC-AUC, one time of day',fontsize=8); ax.set_xlim(0.25,0.9)
ax.set_title('(a) MEFAR: same time of day',fontsize=9,loc='left'); ax.legend(frameon=False,fontsize=7.3,loc='upper center',bbox_to_anchor=(0.45,-0.17),ncol=1,handletextpad=0.1)
ax.grid(axis='x',color='#eee',lw=0.6); ax.set_axisbelow(True)
for k,(ds,t,px) in enumerate([('MEFAR','(b) MEFAR','Time of day (proxy)'),('FatigueSet','(c) FatigueSet','Task phase (proxy)')]):
    ax=fig.add_subplot(gs[k+1]); f=FI[FI.dataset==ds].copy(); f['lab']=f.model.map(SHORT); f=f.sort_values('CDI'); y=np.arange(len(f))
    ax.hlines(y,f.auc_main,f.proxy_auc,color='#dcdcdc',lw=3,zorder=1)
    ax.scatter(f.auc_main,y,color=BLUE,s=26,zorder=4,edgecolor='white',lw=0.8,label='Fatigue (label)')
    ax.scatter(f.proxy_auc,y,color=ORANGE,s=26,zorder=3,edgecolor='white',lw=0.8,marker='D',label=px)
    ax.set_yticks(y); ax.set_yticklabels(f.lab,fontsize=7.3); ax.axvline(0.5,color='#777',lw=0.8,ls=(0,(4,3)))
    ax.set_xlim(0.4,0.9); ax.set_xlabel('ROC-AUC, held-out participants',fontsize=8)
    ax.set_title(t,fontsize=9,loc='left'); ax.legend(frameon=False,fontsize=7.3,loc='upper center',bbox_to_anchor=(0.45,-0.17),ncol=1,handletextpad=0.1)
    ax.grid(axis='x',color='#eee',lw=0.6); ax.set_axisbelow(True)
fig.text(0.985,0.965,'* re-implemented published pipeline',fontsize=7,color='#555',ha='right')
fig.subplots_adjust(left=0.15,right=0.975,top=0.9,bottom=0.23)
fig.savefig('figs/fig_confound.png',dpi=300); fig.savefig('figs/fig_confound.tif',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
# ---------- Figure 4: EEG ----------
cm=pd.read_csv('eeg_corr_mefar.csv',index_col=0); cf=pd.read_csv('eeg_corr_fatigueset.csv',index_col=0)
fig,axs=plt.subplots(1,2,figsize=(180/25.4,3.1))
from matplotlib.colors import LinearSegmentedColormap
cmap=LinearSegmentedColormap.from_list('b',['#f7f7f5','#cde2fb','#86b6ef','#2a78d6','#104281'])
for ax,m,t in [(axs[0],cm,'(a) MEFAR (NeuroSky, released features)'),(axs[1],cf,'(b) FatigueSet (Muse S, absolute band power)')]:
    im=ax.imshow(m.values,cmap=cmap,vmin=0,vmax=1)
    ax.set_xticks(range(len(m))); ax.set_yticks(range(len(m))); ax.set_xticklabels(m.columns,rotation=45,ha='right',fontsize=7.5); ax.set_yticklabels(m.index,fontsize=7.5)
    for i in range(len(m)):
        for j in range(len(m)):
            v=m.values[i,j]; ax.text(j,i,f'{(0.0 if abs(v)<0.005 else v):.2f}',ha='center',va='center',fontsize=6.5,color='white' if v>0.6 else '#222')
    ax.set_title(t,fontsize=9,loc='left'); ax.spines[:].set_visible(False)
cb=fig.colorbar(im,ax=axs,shrink=0.8,pad=0.02); cb.set_label("Spearman's ρ between band powers",fontsize=8); cb.ax.tick_params(labelsize=7)
fig.savefig('figs/fig_eeg.png',dpi=300,bbox_inches='tight'); fig.savefig('figs/fig_eeg.tif',dpi=300,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
print('ok')
