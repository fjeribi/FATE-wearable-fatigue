"""Re-implementations of published MEFAR pipelines.

Jampala et al. (2025): Random Forest, Logistic Regression and K-Nearest Neighbors on the
downsampled MEFAR release (library defaults; stratified random 80:20 split).

Nooh et al. (2025), approximated: linear-scaling (min-max) normalisation, metaheuristic
wrapper feature selection (a binary genetic algorithm stands in for the binary olympiad
optimisation algorithm), and a graph convolutional autoencoder (GCA) classifier. The GCA is
implemented inductively: a k-nearest-neighbour graph (k = 10) is built over the training
samples, and each test sample is connected only to its k nearest training samples, so no
test-test edges exist. The encoder has two GraphSAGE-style graph-convolution layers
(128 and 64 units; separate weights for the node and the mean of its neighbours); a
decoder reconstructs the input features and a logistic head classifies the latent code;
loss = cross-entropy + 0.5 x reconstruction MSE. The multi-objective hippopotamus
hyperparameter optimisation is replaced by fixed settings (Adam, learning rate 0.01,
3000 full-batch epochs, dropout 0.2).
"""
from scipy.special import expit
import numpy as np, jax, jax.numpy as jnp, optax
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
import common  # noqa: F401  (paths and working folder)
jax.config.update('jax_platform_name', 'cpu')
SEED = 42

def jampala_models(scale=False):
    """Library defaults. The public MEFAR_DOWN release is already normalised, so on it the models are
    applied as published (scale=False). On raw-scale tables (the reconstructed MEFAR table, FatigueSet)
    a min-max scaler fitted on the training fold reproduces that normalisation (scale=True)."""
    imp = lambda m: Pipeline([('imp', SimpleImputer(strategy='median'))] + ([('sc', MinMaxScaler())] if scale else []) + [('m', m)])
    return {'Jampala RF': imp(RandomForestClassifier(random_state=SEED, n_jobs=2)),
            'Jampala LR': imp(LogisticRegression(max_iter=1000, random_state=SEED)),
            'Jampala KNN': imp(KNeighborsClassifier())}

def ga_select(X, y, rng, pop=12, gens=8, max_n=8000, fitness='knn'):
    """Binary GA wrapper; fitness = inner stratified 3-fold AUC of a kNN (default) or logistic-regression classifier on a subsample."""
    n, p = X.shape
    idx = rng.choice(n, min(n, max_n), replace=False); Xs, ys = X[idx], y[idx]
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xs, ys))
    def fit(mask):
        if mask.sum() == 0: return 0.0
        s = []
        for tr, te in folds:
            m = (KNeighborsClassifier(5) if fitness == 'knn' else LogisticRegression(max_iter=500)).fit(Xs[tr][:, mask], ys[tr])
            s.append(roc_auc_score(ys[te], m.predict_proba(Xs[te][:, mask])[:, 1]))
        return np.mean(s) - 0.002 * mask.sum()
    P = rng.random((pop, p)) < 0.5; P[0] = True
    F = np.array([fit(m) for m in P])
    for _ in range(gens):
        kids = []
        for _ in range(pop):
            a, b = [P[max(rng.choice(pop, 2), key=lambda i: F[i])] for _ in range(2)]
            cut = rng.integers(1, p); c = np.r_[a[:cut], b[cut:]]
            c ^= rng.random(p) < 1.0 / p; kids.append(c)
        K = np.array(kids); FK = np.array([fit(m) for m in K])
        allP = np.r_[P, K]; allF = np.r_[F, FK]; best = np.argsort(-allF)[:pop]
        P, F = allP[best], allF[best]
    return P[0]

class NoohGCA(BaseEstimator, ClassifierMixin):
    """Inductive graph convolutional autoencoder classifier.
    Each node is connected to its k nearest *training* nodes (self excluded for training nodes),
    so test samples never exchange information with one another. Layers are GraphSAGE-style
    (separate weights for the node itself and for the mean of its neighbours)."""
    def __init__(self, k=10, epochs=3000, lr=0.01, hidden=(128, 64), seed=SEED, select=True, max_train=None):
        self.k, self.epochs, self.lr, self.hidden, self.seed, self.select, self.max_train = k, epochs, lr, hidden, seed, select, max_train
    def _prep(self, X):
        return self.sc_.transform(self.imp_.transform(X))[:, self.mask_].astype(np.float32)
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        if self.max_train and len(X) > self.max_train:   # stratified random subset of training nodes
            keep = np.concatenate([rng.choice(np.flatnonzero(y == c), int(round(self.max_train * np.mean(y == c))), replace=False) for c in (0, 1)])
            X, y = X[keep], y[keep]
        self.imp_ = SimpleImputer(strategy='median').fit(X); Xi = self.imp_.transform(X)
        self.sc_ = MinMaxScaler().fit(Xi); Xs = self.sc_.transform(Xi)
        self.mask_ = ga_select(Xs, y, rng) if self.select else np.ones(Xs.shape[1], bool)
        X = Xs[:, self.mask_].astype(np.float32); self.Xtr_ = X; n, d = X.shape
        self.nn_ = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        nbr = self.nn_.kneighbors(X, return_distance=False)[:, 1:]        # drop self
        NB = jnp.array(nbr)
        key = jax.random.PRNGKey(self.seed); ks = jax.random.split(key, 8)
        g = lambda k, i, o: jax.random.normal(k, (i, o)) * jnp.sqrt(2 / i)
        a, b = self.hidden
        P = {'S1': g(ks[0], d, a), 'N1': g(ks[1], d, a), 'b1': jnp.zeros(a),
             'S2': g(ks[2], a, b), 'N2': g(ks[3], a, b), 'b2': jnp.zeros(b),
             'Wd': g(ks[4], b, d), 'bd': jnp.zeros(d), 'Wc': g(ks[5], b, 1), 'bc': jnp.zeros(1)}
        Xj = jnp.array(X); yj = jnp.array(y, dtype=jnp.float32)
        def enc(P, key, train):
            h = jax.nn.relu(Xj @ P['S1'] + Xj[NB].mean(1) @ P['N1'] + P['b1'])
            if train: h = h * jax.random.bernoulli(key, 0.8, h.shape) / 0.8
            return jax.nn.relu(h @ P['S2'] + h[NB].mean(1) @ P['N2'] + P['b2'])
        def loss(P, key):
            z = enc(P, key, True); rec = z @ P['Wd'] + P['bd']; logit = (z @ P['Wc'] + P['bc'])[:, 0]
            return optax.sigmoid_binary_cross_entropy(logit, yj).mean() + 0.5 * jnp.mean((rec - Xj) ** 2)
        opt = optax.adam(self.lr); st = opt.init(P)
        @jax.jit
        def step(P, st, key):
            l, gr = jax.value_and_grad(loss)(P, key); u, st = opt.update(gr, st); return optax.apply_updates(P, u), st
        for e in range(self.epochs):
            key, sub = jax.random.split(key); P, st = step(P, st, sub)
        self.P_ = {k: np.asarray(v) for k, v in P.items()}; self.classes_ = np.array([0, 1])
        Pn = self.P_; self.H1_ = np.maximum(X @ Pn['S1'] + X[nbr].mean(1) @ Pn['N1'] + Pn['b1'], 0)
        return self
    def predict_proba(self, X):
        X = self._prep(X); P = self.P_
        nbr = self.nn_.kneighbors(X, n_neighbors=self.k, return_distance=False)
        h1 = np.maximum(X @ P['S1'] + self.Xtr_[nbr].mean(1) @ P['N1'] + P['b1'], 0)
        z = np.maximum(h1 @ P['S2'] + self.H1_[nbr].mean(1) @ P['N2'] + P['b2'], 0)
        p = expit((z @ P['Wc'] + P['bc'])[:, 0])
        return np.c_[1 - p, p]
