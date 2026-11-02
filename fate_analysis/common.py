import os, numpy as np, pandas as pd, warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')
SEED=42
# ---- paths (override with environment variables) ----
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(env, default): return os.path.abspath(os.environ.get(env, os.path.join(_ROOT, default)))
MEFAR_RESULTS = _p('MEFAR_RESULTS', 'results')            # output folder of scripts/01-11 (reconstructed table, deep OOF, Chalder scores)
MEFAR_CSV = os.path.join(MEFAR_RESULTS, 'MEFAR_subjectwise_1Hz_FINAL.csv')
MEFAR_DOWN = _p('MEFAR_DOWN', 'data/MEFAR_DOWN.csv')       # public preprocessed release
FATIGUESET_ROOT = _p('FATIGUESET_ROOT', 'data/fatigueset') # FatigueSet folder (01 ... 12, metadata.csv)
FATE_OUT = _p('FATE_OUT', 'results_fate')                  # all FATE outputs, caches and figures
os.makedirs(os.path.join(FATE_OUT, 'figs'), exist_ok=True); os.chdir(FATE_OUT)
FS_WINDOWS = os.path.join(FATE_OUT, 'fatigueset_windows.csv')
PHYS=["BVP","EDA","TEMP","AccX","AccY","AccZ","HR"]
NEURO=["Delta","Theta","Alpha1","Alpha2","Beta1","Beta2","Gamma1","Gamma2","Attention","Meditation"]

def sc(m): return Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',m)])
def imp(m): return Pipeline([('imp',SimpleImputer(strategy='median')),('m',m)])
def models(small=False):
    return {
     'Logistic Regression': sc(LogisticRegression(max_iter=3000,random_state=SEED)),
     'Gaussian Naive Bayes': sc(GaussianNB()),
     'KNN': sc(KNeighborsClassifier(n_neighbors=25 if small else 5)),
     'Random Forest': imp(RandomForestClassifier(n_estimators=300,min_samples_leaf=5,n_jobs=2,random_state=SEED)),
     'XGBoost': XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                              eval_metric='logloss',n_jobs=2,random_state=SEED,tree_method='hist'),
     'MLP': sc(MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,early_stopping=True,max_iter=300,random_state=SEED)),
    }

def load_mefar():
    d=pd.read_csv(MEFAR_CSV).replace([np.inf,-np.inf],np.nan).dropna(subset=PHYS+NEURO+['class']).reset_index(drop=True)
    d['y']=d['class'].astype(int); d['participant']=d.subject_id.astype(str); d['unit']=d.session_id
    d['chunk']=d.session_id+'_'+(d.second//60).astype(int).astype(str)
    return d, PHYS+NEURO

def load_fs(norm=False):
    d=pd.read_csv(FS_WINDOWS)
    feats=[c for c in d.columns if c not in ['participant','session','block','block_id','t_rel','mental_fatigue','physical_fatigue']]
    thr=d.groupby('block_id').mental_fatigue.first().median()
    d['y']=(d.mental_fatigue>=thr).astype(int); d['unit']=d.block_id; d['chunk']=d.block_id
    if norm:   # label-free within-participant z-scoring
        d[feats]=d.groupby('participant')[feats].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
    return d, feats

def pooled_auc(y,s,fold):
    r=pd.Series(s).groupby(np.asarray(fold)).rank(pct=True).values
    return roc_auc_score(y,r), r

def boot_ci(df,col='rank',n=1000,seed=SEED):
    rng=np.random.default_rng(seed); ps=df.participant.unique(); idx={p:np.flatnonzero(df.participant.values==p) for p in ps}
    y=df.y.values; s=df[col].values; out=[]
    for _ in range(n):
        b=np.concatenate([idx[p] for p in rng.choice(ps,len(ps))])
        if len(np.unique(y[b]))==2: out.append(roc_auc_score(y[b],s[b]))
    return np.percentile(out,[2.5,97.5])

def unit_auc(df,col='rank'):
    u=df.groupby('unit').agg(y=('y','first'),s=(col,'mean'))
    return roc_auc_score(u.y,u.s)

def unit_perm_p(df,col='rank',n=2000,seed=SEED):
    rng=np.random.default_rng(seed); u=df.groupby('unit').agg(y=('y','first'),s=(col,'mean'))
    obs=roc_auc_score(u.y,u.s); perm=[roc_auc_score(rng.permutation(u.y.values),u.s) for _ in range(n)]
    return (1+np.sum(np.array(perm)>=obs))/(1+n)
