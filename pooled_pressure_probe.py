#!/usr/bin/env python
"""
Pooled P_S+P_C pressure-deception probe.

Tests whether a single probe trained on POOLED P_S+P_C activations recovers
~diagonal performance on BOTH domains:
  - pool -> P_S  ~=  P_S -> P_S (diagonal)  AND  pool -> P_C ~= P_C -> P_C
    => a shared pressure-deception subspace exists; single-domain training just
       misses its orientation ("context-ROTATED", softens "context-bound").
  - pooling fails to recover one/both diagonals
    => pressure deception is represented domain-SPECIFICALLY ("context-BOUND").

The contrast is against the (low) cross cells P_S->P_C / P_C->P_S, which is what
promoted this experiment. CPU only — activations are cached.

Env: OUTPUT_DIR (default ~/output). Reads probe_state.pkl + acts_P_{S,C}_*.npy.
Runs every position present: pre-answer (acts_P_S_layer_*), response token
(acts_P_S_resp_layer_*), mean-response (acts_P_S_respmean_layer_*).
"""
import os, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, C, TEST_SIZE, MAX_ITER = 32, 10, 1.0, 0.2, 2000

PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
yS_full, mS, gS_full = PS['labels_P_S_lied'], PS['mask_P_S'], PS['groups_P_S']
yC_full, mC, gC_full = PS['labels_P_C_lied'], PS['mask_P_C'], PS['groups_P_C']


def load_feats(tag, pos):
    """pos in {'pre','resp','respmean'}; 'pre' uses the original acts_<tag>_layer_*."""
    def path(li):
        if pos == 'pre':
            return op / f'acts_{tag}_layer_{li:03d}.npy'
        return op / f'acts_{tag}_{pos}_layer_{li:03d}.npy'
    return {li: np.load(path(li)) for li in range(N_LAYERS) if path(li).exists()}


def _fit(X, y, seed):
    return LogisticRegression(max_iter=MAX_ITER, C=C, random_state=seed).fit(X, y)


def _auc(clf, X, y):
    return roc_auc_score(y, clf.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else 0.5


def grouped_holdout_best_layer(X, y, g):
    """Per-seed grouped split; return best layer by mean holdout AUC + that AUC."""
    layers = sorted(X)
    aucs = np.full((N_SEEDS, max(layers) + 1), np.nan)
    for seed in range(N_SEEDS):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed)
                      .split(np.arange(len(y)), y, g))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        for li in layers:
            aucs[seed, li] = _auc(_fit(X[li][tr], y[tr], seed), X[li][te], y[te])
    mean = np.nanmean(aucs, 0)
    best = int(np.nanargmax(mean))
    return best, float(mean[best]), float(np.nanstd(aucs[:, best]))


def cross(Xa, ya, la, Xb, yb):
    """Train on all of a at layer la, score all of b (off-diagonal cross-transfer)."""
    clf = _fit(Xa[la], ya, 0)
    return _auc(clf, Xb[la], yb)


def run_position(pos):
    fS, fC = load_feats('P_S', pos), load_feats('P_C', pos)
    if not fS or not fC:
        print(f'[{pos}] activations missing — skipping')
        return
    XS = {li: fS[li][mS] for li in fS}; yS = yS_full[mS]
    XC = {li: fC[li][mC] for li in fC}; yC = yC_full[mC]
    gS = np.array([f'S:{x}' for x in gS_full[mS]])
    gC = np.array([f'C:{x}' for x in gC_full[mC]])
    nS, nC = len(yS), len(yC)
    layers = sorted(set(XS) & set(XC))
    Xp = {li: np.concatenate([XS[li], XC[li]], 0) for li in layers}
    yp = np.concatenate([yS, yC]); gp = np.concatenate([gS, gC])
    dom = np.array([0] * nS + [1] * nC)             # 0=S, 1=C

    # ── single-domain diagonals + cross (baselines) ──────────────────────────
    bS, aucSS, sSS = grouped_holdout_best_layer(XS, yS, gS)
    bC, aucCC, sCC = grouped_holdout_best_layer(XC, yC, gC)
    aucSC = cross(XS, yS, bS, XC, yC)               # P_S -> P_C
    aucCS = cross(XC, yC, bC, XS, yS)               # P_C -> P_S

    # ── pooled: best layer by pooled holdout, then per-seed pool->S / pool->C ─
    bP, aucPP, sPP = grouped_holdout_best_layer(Xp, yp, gp)
    p2S, p2C = [], []
    for seed in range(N_SEEDS):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed)
                      .split(np.arange(len(yp)), yp, gp))
        clf = _fit(Xp[bP][tr], yp[tr], seed)
        teS, teC = te[dom[te] == 0], te[dom[te] == 1]
        if len(teS): p2S.append(_auc(clf, Xp[bP][teS], yp[teS]))
        if len(teC): p2C.append(_auc(clf, Xp[bP][teC], yp[teC]))
    p2S_m, p2C_m = float(np.mean(p2S)), float(np.mean(p2C))

    print(f'\n{"="*64}\nPOSITION: {pos}   (P_S n={nS}, P_C n={nC}, pooled n={nS+nC})\n{"="*64}')
    print(f'  P_S -> P_S  (diagonal)   AUC = {aucSS:.3f} ± {sSS:.3f}   [layer {bS}]')
    print(f'  P_C -> P_C  (diagonal)   AUC = {aucCC:.3f} ± {sCC:.3f}   [layer {bC}]')
    print(f'  P_S -> P_C  (cross)      AUC = {aucSC:.3f}')
    print(f'  P_C -> P_S  (cross)      AUC = {aucCS:.3f}')
    print(f'  pooled holdout           AUC = {aucPP:.3f} ± {sPP:.3f}   [layer {bP}]')
    print(f'  pool -> P_S              AUC = {p2S_m:.3f}   (vs diagonal {aucSS:.3f}, '
          f'cross {aucCS:.3f})')
    print(f'  pool -> P_C              AUC = {p2C_m:.3f}   (vs diagonal {aucCC:.3f}, '
          f'cross {aucSC:.3f})')
    # ── verdict ──────────────────────────────────────────────────────────────
    rec_S = p2S_m >= aucSS - 0.05
    rec_C = p2C_m >= aucCC - 0.05
    if rec_S and rec_C:
        print('  => VERDICT: pooled recovers BOTH diagonals -> SHARED pressure-deception '
              'subspace;\n     single-domain training just misses its orientation '
              '("context-ROTATED").')
    else:
        miss = ', '.join(d for d, r in [('P_S', rec_S), ('P_C', rec_C)] if not r)
        print(f'  => VERDICT: pooling FAILS to recover {miss} -> pressure deception is '
              f'represented\n     domain-SPECIFICALLY ("context-BOUND") — the stronger claim.')
    return dict(pos=pos, P_S_diag=aucSS, P_C_diag=aucCC, P_S_to_P_C=aucSC,
                P_C_to_P_S=aucCS, pool_holdout=aucPP, pool_to_S=p2S_m, pool_to_C=p2C_m)


if __name__ == '__main__':
    print(f'OUTPUT_DIR={OUTPUT_DIR}')
    results = {}
    for pos in ['pre', 'resp', 'respmean']:
        r = run_position(pos)
        if r:
            results[pos] = r
    import json
    out = op / 'pooled_pressure_probe_results.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved {out}')
