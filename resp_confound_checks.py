#!/usr/bin/env python
"""
Response-token confound checks (CPU). Items (1), (2), (4) from the review queue.

(1) Answer-polarity / valence control on the P_S<->P_C response-token transfer:
    is the 0.68/0.79 jump genuine deception transfer or the emitted Yes/No token?
(2) DiffMean robustness on the response-token critical cells (does P_S->I'_S 0.84
    survive the difference-of-means estimator, which killed it at pre-answer?).
(4) Random-direction separability per test set — the "test-set easiness floor"
    that tells us how much every ->I' number is just an easy target.

Env: OUTPUT_DIR (default ~/output). Reuses cached response-token acts + Z feats +
probe_state.pkl + inference_P_bundle.pkl (answer polarity). No GPU.
"""
import os, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from probe_utils import (multiseed_best_layer, fit_final_probe, fit_diffmean,
                         cross_transfer_matrix, print_transfer_matrix,
                         random_direction_auc)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, MAX_ITER = 32, 10, 1.0, 2000

PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
for k, v in PS.items():
    globals()[k] = v          # labels_*, mask_*, groups_*, best_layer_Z*, ...
pb = pickle.load(open(op / 'inference_P_bundle.pkl', 'rb'))
df_sales, df_combined = pb['df_sales'], pb['df_combined']

def load(tag, pos='resp'):
    return {li: np.load(op / f'acts_{tag}_{pos}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}_{pos}_layer_{li:03d}.npy').exists()}

feats = {'P_S': load('P_S'), 'P_C': load('P_C'), 'I_S': load('I_S'), 'I_C': load('I_C')}
feats_Z = {li: np.concatenate([np.load(op / f'acts_Z_honest_layer_{li:03d}.npy'),
                               np.load(op / f'acts_Z_deceptive_layer_{li:03d}.npy')], 0)
           for li in range(N_LAYERS) if (op / f'acts_Z_honest_layer_{li:03d}.npy').exists()}
feats_Z_mean = {li: np.concatenate([np.load(op / f'acts_Z_mean_honest_layer_{li:03d}.npy'),
                                    np.load(op / f'acts_Z_mean_deceptive_layer_{li:03d}.npy')], 0)
                for li in range(N_LAYERS) if (op / f'acts_Z_mean_honest_layer_{li:03d}.npy').exists()}

LAB = {'P_S': labels_P_S_lied, 'P_C': labels_P_C_lied, 'I_S': labels_I_S, 'I_C': labels_I_C}
MSK = {'P_S': mask_P_S, 'P_C': mask_P_C, 'I_S': mask_I_S, 'I_C': mask_I_C}
GRP = {'P_S': groups_P_S, 'P_C': groups_P_C, 'I_S': groups_I_S, 'I_C': groups_I_C}

print('Selecting response-token best layers for P/I\'...')
BL = {}
for t in ['P_S', 'P_C', 'I_S', 'I_C']:
    BL[t], _ = multiseed_best_layer(feats[t], LAB[t], MSK[t], n_seeds=N_SEEDS, C=PROBE_C, groups=GRP[t])
results = {'best_layers': {k: int(v) for k, v in BL.items()}}

# ── (1) answer-polarity / valence control on P_S<->P_C ───────────────────────
print('\n' + '=' * 70 + '\n(1) ANSWER-POLARITY CONTROL on P_S<->P_C (response token)\n' + '=' * 70)
ANS = {'P_S': (df_sales['generated_answer'].to_numpy() == 'yes'),
       'P_C': (df_combined['generated_answer'].to_numpy() == 'yes')}

def masked(t):
    m = MSK[t]
    return feats[t], LAB[t][m], ANS[t][m], m

for t in ['P_S', 'P_C']:
    _, y, a, m = masked(t)
    phi = float(np.corrcoef(y, a.astype(int))[0, 1])
    # 2x2 lied x answer
    c = np.array([[int(((y == L) & (a == (A == 1))).sum()) for A in (1, 0)] for L in (1, 0)])
    print(f'  {t}: corr(actually_lied, answer=yes) phi = {phi:+.3f}   '
          f'contingency lied[yes,no]={c[0].tolist()} truth[yes,no]={c[1].tolist()}')
    results.setdefault('polarity', {})[t] = {'phi': phi, 'contingency': c.tolist()}

def fit_source(src):
    m = MSK[src]
    clf = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=0)
    clf.fit(feats[src][BL[src]][m], LAB[src][m])
    return clf

def auc_safe(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')

def stratified(src, tgt):
    clf = fit_source(src)
    m = MSK[tgt]
    X = feats[tgt][BL[src]][m]; y = LAB[tgt][m]; a = ANS[tgt][m]
    s = clf.predict_proba(X)[:, 1]
    pooled = auc_safe(y, s)
    yes = auc_safe(y[a], s[a]); no = auc_safe(y[~a], s[~a])
    # answer-balanced subsample: within each label class, equal yes/no
    rng = np.random.RandomState(0); keep = []
    for L in (0, 1):
        iy = np.where((y == L) & a)[0]; ino = np.where((y == L) & ~a)[0]
        k = min(len(iy), len(ino))
        if k:
            keep += rng.choice(iy, k, False).tolist() + rng.choice(ino, k, False).tolist()
    keep = np.array(sorted(keep))
    bal = auc_safe(y[keep], s[keep]) if len(keep) and len(np.unique(y[keep])) > 1 else float('nan')
    return dict(pooled=pooled, ans_yes=yes, ans_no=no, balanced=bal,
                n_yes=int(a.sum()), n_no=int((~a).sum()))

print(f'\n  {"transfer":>12s}   {"pooled":>7s}  {"ans=yes":>8s}  {"ans=no":>7s}  {"bal":>6s}')
for src, tgt in [('P_S', 'P_C'), ('P_C', 'P_S')]:
    r = stratified(src, tgt)
    results.setdefault('polarity_transfer', {})[f'{src}->{tgt}'] = r
    print(f'  {src+" -> "+tgt:>12s}   {r["pooled"]:7.3f}  {r["ans_yes"]:8.3f}  '
          f'{r["ans_no"]:7.3f}  {r["balanced"]:6.3f}   (n_yes={r["n_yes"]}, n_no={r["n_no"]})')
both = results['polarity_transfer']
ok = all(min(v['ans_yes'], v['ans_no']) >= 0.65 for v in both.values())
print(f'  => VERDICT: P_S<->P_C recovery {"SURVIVES" if ok else "DOES NOT survive"} the '
      f'within-answer control -> {"context-ROTATED (real)" if ok else "answer-token artifact"}.')

# ── (2) DiffMean on the response-token critical cells ────────────────────────
print('\n' + '=' * 70 + '\n(2) DIFFMEAN vs LR on the response-token critical cells\n' + '=' * 70)
std = ["I'_S", "I'_C", "P_S", "P_C", "Z", "Z_mean"]
sd = {
    "I'_S": (feats['I_S'], labels_I_S, mask_I_S, groups_I_S, BL['I_S'], PROBE_C, None),
    "I'_C": (feats['I_C'], labels_I_C, mask_I_C, groups_I_C, BL['I_C'], PROBE_C, None),
    "P_S":  (feats['P_S'], labels_P_S_lied, mask_P_S, groups_P_S, BL['P_S'], PROBE_C, None),
    "P_C":  (feats['P_C'], labels_P_C_lied, mask_P_C, groups_P_C, BL['P_C'], PROBE_C, None),
    "Z":      (feats_Z, labels_Z, mask_Z, None, best_layer_Z, PROBE_C, None),
    "Z_mean": (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_mean, PROBE_C, None),
}
lr_m, _ = cross_transfer_matrix(std, sd)
dm_m, _ = cross_transfer_matrix(std, sd, fit_fn=fit_diffmean)
lr, dm = np.nanmean(lr_m, 2), np.nanmean(dm_m, 2)
crit = [("P_S", "P_C"), ("P_C", "P_S"), ("P_S", "I'_S"), ("P_C", "I'_C"),
        ("I'_S", "P_S"), ("I'_C", "P_C"), ("Z", "P_S"), ("Z", "P_C")]
print(f'  {"train -> test":>14s}   {"LR":>6s}   {"DiffMean":>8s}   {"|Δ|":>5s}')
for tn, te in crit:
    i, j = std.index(tn), std.index(te)
    d = abs(lr[i, j] - dm[i, j]); flag = '  <-- moves' if d > 0.1 else ''
    print(f'  {tn+" -> "+te:>14s}   {lr[i,j]:6.3f}   {dm[i,j]:8.3f}   {d:5.3f}{flag}')
    results.setdefault('diffmean', {})[f'{tn}->{te}'] = {'lr': float(lr[i, j]), 'diffmean': float(dm[i, j])}

# ── (4) random-direction separability (test-set easiness floor) ──────────────
print('\n' + '=' * 70 + '\n(4) RANDOM-DIRECTION SEPARABILITY per test set (easiness floor)\n' + '=' * 70)
print('  (sign-agnostic max(auc,1-auc) of 1000 random unit directions at the set\'s best layer)')
TESTSETS = {"P_S": ('P_S', BL['P_S']), "P_C": ('P_C', BL['P_C']),
            "I'_S": ('I_S', BL['I_S']), "I'_C": ('I_C', BL['I_C']),
            "Z": (None, best_layer_Z), "Z_mean": (None, best_layer_Z_mean)}
print(f'  {"test set":>8s}   {"best L":>6s}   {"rand p95":>8s}   {"rand max":>8s}   {"frac>0.8":>8s}')
for name, (tag, layer) in TESTSETS.items():
    if tag:
        X = feats[tag][layer][MSK[tag]]; y = LAB[tag][MSK[tag]]
    else:
        fz = feats_Z if name == 'Z' else feats_Z_mean
        lab = labels_Z if name == 'Z' else labels_Z_mean
        msk = mask_Z if name == 'Z' else mask_Z_mean
        X = fz[layer][msk]; y = lab[msk]
    r = random_direction_auc(X, y, n_dirs=1000, seed=0)
    results.setdefault('rand_dir', {})[name] = {'layer': int(layer), **r}
    print(f'  {name:>8s}   {layer:6d}   {r["p95"]:8.3f}   {r["max"]:8.3f}   {r["frac_above_0.8"]:8.2f}')
print('  => Read: a high p95/max means the ->set transfer numbers reflect an EASY target,'
      '\n     not a meaningful shared direction. Compare I\' (saturated?) vs P_S/P_C.')

json.dump(results, open(op / 'resp_confound_checks_results.json', 'w'), indent=2)
print(f'\nSaved {op / "resp_confound_checks_results.json"}')
