"""MF-AFEN (two-branch, element-wise sigmoid-gated fusion) and a matched single-branch MLP, in JAX, on FatigueSet.
Batch normalisation of the Keras original is replaced by layer normalisation (per-sample), otherwise the
layer sizes, dropout rates, optimiser (Adam 1e-3), batch 64, early stopping (patience 10, max 100 epochs) on a
participant-grouped inner validation split (20% of training participants) follow the paper."""
import os, sys, numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from common import *
jax.config.update('jax_platform_name','cpu')
d,F=load_fs(); PH=[f for f in F if not f.startswith('eeg_')]; NE=[f for f in F if f.startswith('eeg_')]
def dense(k,i,o): return {'w':jax.random.normal(k,(i,o))*jnp.sqrt(2/i),'b':jnp.zeros(o)}
def ln(x): return (x-x.mean(-1,keepdims=True))/(x.std(-1,keepdims=True)+1e-5)
def init(key,kind):
    ks=iter(jax.random.split(key,20))
    if kind=='mfafen':
        return {'p1':dense(next(ks),len(PH),64),'p2':dense(next(ks),64,32),'n1':dense(next(ks),len(NE),128),'n2':dense(next(ks),128,64),
                'n3':dense(next(ks),64,32),'gp':dense(next(ks),32,32),'gn':dense(next(ks),32,32),'f1':dense(next(ks),64,64),
                'f2':dense(next(ks),64,32),'o':dense(next(ks),32,1)}
    return {'a1':dense(next(ks),len(F),128),'a2':dense(next(ks),128,64),'a3':dense(next(ks),64,32),'o':dense(next(ks),32,1)}
def L(p,x): return x@p['w']+p['b']
def drop(x,r,key,train): 
    if not train or r==0: return x
    m=jax.random.bernoulli(key,1-r,x.shape); return x*m/(1-r)
def fwd(P,xp,xn,key,train,kind):
    ks=jax.random.split(key,8)
    if kind=='mfafen':
        hp=drop(jax.nn.relu(ln(L(P['p1'],xp))),.3,ks[0],train); hp=jax.nn.relu(L(P['p2'],hp))
        hn=drop(jax.nn.relu(ln(L(P['n1'],xn))),.3,ks[1],train); hn=drop(jax.nn.relu(L(P['n2'],hn)),.2,ks[2],train); hn=jax.nn.relu(L(P['n3'],hn))
        hp=hp*jax.nn.sigmoid(L(P['gp'],hp)); hn=hn*jax.nn.sigmoid(L(P['gn'],hn))
        h=jnp.concatenate([hp,hn],-1); h=drop(jax.nn.relu(ln(L(P['f1'],h))),.3,ks[3],train); h=drop(jax.nn.relu(L(P['f2'],h)),.2,ks[4],train)
    else:
        x=jnp.concatenate([xp,xn],-1)
        h=drop(jax.nn.relu(ln(L(P['a1'],x))),.3,ks[0],train); h=drop(jax.nn.relu(ln(L(P['a2'],h))),.25,ks[1],train); h=drop(jax.nn.relu(L(P['a3'],h)),.2,ks[2],train)
    return L(P['o'],h)[:,0]
def fit_predict(Xtr,ytr,gtr,Xte,seed,kind):
    imp=SimpleImputer(strategy='median').fit(Xtr); sc_=StandardScaler().fit(imp.transform(Xtr))
    T=lambda X: np.nan_to_num(sc_.transform(imp.transform(X)))
    Xtr_,Xte_=T(Xtr),T(Xte); ip=[F.index(f) for f in PH]; ineu=[F.index(f) for f in NE]
    tr,va=next(GroupShuffleSplit(1,test_size=0.2,random_state=seed).split(Xtr_,ytr,gtr))
    key=jax.random.PRNGKey(seed); P=init(key,kind); opt=optax.adam(1e-3); st=opt.init(P)
    def loss(P,xp,xn,y,k): return optax.sigmoid_binary_cross_entropy(fwd(P,xp,xn,k,True,kind),y).mean()
    @jax.jit
    def step(P,st,xp,xn,y,k):
        l,g=jax.value_and_grad(loss)(P,xp,xn,y,k); u,st=opt.update(g,st); return optax.apply_updates(P,u),st
    pred=jax.jit(lambda P,xp,xn: jax.nn.sigmoid(fwd(P,xp,xn,jax.random.PRNGKey(0),False,kind)))
    Xa,ya=Xtr_[tr],ytr[tr].astype(np.float32); Xv,yv=Xtr_[va],ytr[va].astype(np.float32)
    best,bestP,wait=1e9,P,0; rng=np.random.default_rng(seed)
    for ep in range(100):
        idx=rng.permutation(len(Xa))
        for i in range(0,len(idx),64):
            b=idx[i:i+64]; key,k=jax.random.split(key); P,st=step(P,st,Xa[b][:,ip],Xa[b][:,ineu],ya[b],k)
        pv=np.clip(np.asarray(pred(P,Xv[:,ip],Xv[:,ineu])),1e-6,1-1e-6); vl=-np.mean(yv*np.log(pv)+(1-yv)*np.log(1-pv))
        if vl<best-1e-4: best,bestP,wait=vl,P,0
        else:
            wait+=1
            if wait>=10: break
    return np.asarray(pred(bestP,Xte_[:,ip],Xte_[:,ineu]))
X=d[F].values; y=d.y.values; rows=[]
os.makedirs('cache/mfafen_fs',exist_ok=True)
for rung in ['random_sample','held_out_block','held_out_session','held_out_participant','loso']:
    if rung=='random_sample': folds=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y))
    elif rung=='loso': folds=list(LeaveOneGroupOut().split(X,y,d.participant))
    else:
        g={'held_out_block':'block_id','held_out_session':'session','held_out_participant':'participant'}[rung]
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,d[g]))
    for kind in ['mfafen','mlp']:
        for seed in [42,7,123]:
            f=f'cache/mfafen_fs/{rung}__{kind}__{seed}.csv.gz'
            if not os.path.exists(f):
                parts=[]
                for k,(tr,te) in enumerate(folds):
                    parts.append(pd.DataFrame({'idx':te,'fold':k,'score':fit_predict(X[tr],y[tr],d.participant.values[tr],X[te],seed+k,kind)}))
                pd.concat(parts).to_csv(f,index=False)
            o=pd.read_csv(f).sort_values('idx'); dd=d.iloc[o.idx.values].copy()
            auc,r=pooled_auc(dd.y.values,o.score.values,o.fold.values); dd['rank']=r
            rows.append({'rung':rung,'model':kind,'seed':seed,'pooled_auc':auc,'unit_auc':unit_auc(dd)}); print(rows[-1],flush=True)
r=pd.DataFrame(rows); r.to_csv('mfafen_fatigueset.csv',index=False)
print(r.groupby(['rung','model'])[['pooled_auc','unit_auc']].agg(['mean','std']).round(3))
