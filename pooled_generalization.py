#!/usr/bin/env python
"""
Pooled-probe generalization matrix (in-distribution) — HONEST HELD-OUT.

Trains a POOLED P_S+P_C probe and slots it into the NxN cross-transfer matrix as
an extra row/column, to read how the pooled pressure-deception direction
generalizes to every other in-distribution dataset (I', Z*).

CONTAMINATION FIX: P_S and P_C are SUBSETS of POOLED's training data, so the
plain cross_transfer_matrix cells POOLED->P_S / POOLED->P_C (and the reverse
column) score rows that were in training -> inflated. Those four cells are
recomputed HELD-OUT: per-seed GroupShuffleSplit on the pooled set, train on the
train fold, score only the disjoint rows (the held-out P_S/P_C subset for the
row; the pooled rows whose scenario is absent from the component's train fold for
the column). The disjoint cells (POOLED<->I'/Z) keep the standard matrix value.

Runs at BOTH positions so POOLED is native in each frame:
  resp = assertion/response token (P/I from acts_*_resp_*)
  pre  = pre-answer last token   (P/I from acts_*_pre_*)
Z* features are position-fixed (cached). Standalone, CPU-only.

Env: OUTPUT_DIR (default ~/output).
"""
import os, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from probe_utils import (multiseed_best_layer, fit_final_probe,
                         cross_transfer_matrix, bootstrap_ci_matrix,
                         print_transfer_matrix, _bootstrap_auc)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, TEST_SIZE, MAX_ITER = 32, 10, 1.0, 0.2, 2000

PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
g = globals()
for k, v in PS.items():
    g[k] = v

def load(tag, pos):
    return {li: np.load(op / f'acts_{tag}_{pos}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}_{pos}_layer_{li:03d}.npy').exists()}

# ── position-fixed Z* features ───────────────────────────────────────────────
feats_Z = {li: np.concatenate([np.load(op / f'acts_Z_honest_layer_{li:03d}.npy'),
                               np.load(op / f'acts_Z_deceptive_layer_{li:03d}.npy')], 0)
           for li in range(N_LAYERS) if (op / f'acts_Z_honest_layer_{li:03d}.npy').exists()}
feats_Z_mean = {li: np.concatenate([np.load(op / f'acts_Z_mean_honest_layer_{li:03d}.npy'),
                                    np.load(op / f'acts_Z_mean_deceptive_layer_{li:03d}.npy')], 0)
                for li in range(N_LAYERS) if (op / f'acts_Z_mean_honest_layer_{li:03d}.npy').exists()}
def load_tok(t): return {li: np.load(op / f'acts_Z_tok_{t}_layer_{li:03d}.npy').astype(np.float32)
                         for li in range(N_LAYERS) if (op / f'acts_Z_tok_{t}_layer_{li:03d}.npy').exists()}
tok_feats_h, tok_feats_d = load_tok('honest'), load_tok('deceptive')
counts_h = np.load(op / 'Z_tok_counts_honest.npy'); counts_d = np.load(op / 'Z_tok_counts_deceptive.npy')

def _fit(X, y, seed):
    return LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=seed).fit(X, y)
def _auc(clf, X, y):
    return roc_auc_score(y, clf.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else 0.5

def pooled_to_comp(Xp, yp, gp, dom, bl, code):
    """Held-out POOLED->component: per-seed pooled split, score held-out comp rows."""
    aucs, ys, ps = [], [], []
    idx = np.arange(len(yp))
    for seed in range(N_SEEDS):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed).split(idx, yp, gp))
        if len(np.unique(yp[tr])) < 2:
            continue
        clf = _fit(Xp[bl][tr], yp[tr], seed)
        sel = te[dom[te] == code]
        if len(sel) and len(np.unique(yp[sel])) > 1:
            aucs.append(_auc(clf, Xp[bl][sel], yp[sel]))
            ys.append(yp[sel]); ps.append(clf.predict_proba(Xp[bl][sel])[:, 1])
    yc, pc = np.concatenate(ys), np.concatenate(ps)
    return float(np.mean(aucs)), float(np.std(aucs)), _bootstrap_auc(yc, pc)

def comp_to_pooled(Xc, yc, gc_pref, blc, Xp, yp, gp):
    """Held-out component->POOLED: train comp on its train fold, score pooled rows
    whose scenario is ABSENT from the comp train fold (disjoint)."""
    aucs, ys, ps = [], [], []
    nc = len(yc); idx = np.arange(nc)
    for seed in range(N_SEEDS):
        tr, _ho = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed).split(idx, yc, gc_pref))
        if len(np.unique(yc[tr])) < 2:
            continue
        clf = _fit(Xc[blc][tr], yc[tr], seed)
        train_scen = set(gc_pref[tr].tolist())
        keep = np.array([gg not in train_scen for gg in gp])
        if keep.sum() and len(np.unique(yp[keep])) > 1:
            aucs.append(_auc(clf, Xp[blc][keep], yp[keep]))
            ys.append(yp[keep]); ps.append(clf.predict_proba(Xp[blc][keep])[:, 1])
    yc2, pc2 = np.concatenate(ys), np.concatenate(ps)
    return float(np.mean(aucs)), float(np.std(aucs)), _bootstrap_auc(yc2, pc2)

ALL = ["I'_S", "I'_C", "P_S", "P_C", "Z", "Z_mean", "Z_tok", "Z_multi", "POOLED"]
results = {}

for frame in ['resp', 'pre']:
    print(f'\n############### FRAME = {frame} ###############')
    fPS, fPC = load('P_S', frame), load('P_C', frame)
    fIS, fIC = load('I_S', frame), load('I_C', frame)
    if not fPS:
        print(f'[{frame}] P features missing — skipping frame'); continue

    # POOLED at this frame (pre-masked + concatenated; disjoint S:/C: group namespace)
    pl = sorted(set(fPS) & set(fPC))
    feats_POOLED = {li: np.concatenate([fPS[li][mask_P_S], fPC[li][mask_P_C]], 0) for li in pl}
    y_pool = np.concatenate([labels_P_S_lied[mask_P_S], labels_P_C_lied[mask_P_C]])
    g_pool = np.concatenate([np.array([f'S:{x}' for x in groups_P_S[mask_P_S]]),
                             np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])])
    dom = np.array([0] * int(mask_P_S.sum()) + [1] * int(mask_P_C.sum()))
    m_pool = np.ones(len(y_pool), dtype=bool)

    print('Selecting best layers...')
    bl = {}
    for tag, (f, y, m, gr) in {
            'P_S': (fPS, labels_P_S_lied, mask_P_S, groups_P_S),
            'P_C': (fPC, labels_P_C_lied, mask_P_C, groups_P_C),
            'I_S': (fIS, labels_I_S, mask_I_S, groups_I_S),
            'I_C': (fIC, labels_I_C, mask_I_C, groups_I_C),
            'POOLED': (feats_POOLED, y_pool, m_pool, g_pool)}.items():
        bl[tag], _a = multiseed_best_layer(f, y, m, n_seeds=N_SEEDS, C=PROBE_C, groups=gr)

    data = {
        "I'_S":   (fIS, labels_I_S,      mask_I_S, groups_I_S, bl['I_S'], PROBE_C, None),
        "I'_C":   (fIC, labels_I_C,      mask_I_C, groups_I_C, bl['I_C'], PROBE_C, None),
        "P_S":    (fPS, labels_P_S_lied, mask_P_S, groups_P_S, bl['P_S'], PROBE_C, None),
        "P_C":    (fPC, labels_P_C_lied, mask_P_C, groups_P_C, bl['P_C'], PROBE_C, None),
        "Z":      (feats_Z, labels_Z, mask_Z, None, best_layer_Z, PROBE_C, None),
        "Z_mean": (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_mean, PROBE_C, None),
        "Z_tok":  (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_tok, PROBE_C, None),
        "Z_multi":(feats_Z_mean, labels_Z_mean, mask_Z_mean, None, multi_layers, PROBE_C, None),
        "POOLED": (feats_POOLED, y_pool, m_pool, g_pool, bl['POOLED'], PROBE_C, None),
    }
    tok_td = {"Z_tok": (tok_feats_h, tok_feats_d, counts_h, counts_d)}
    multi_td = {"Z_multi": (tok_feats_h, tok_feats_d, counts_h, counts_d, multi_layers)}
    print(f'Computing {len(ALL)}x{len(ALL)} matrix...')
    full, preds = cross_transfer_matrix(ALL, data, tok_train_data=tok_td, multi_train_data=multi_td)
    ci = bootstrap_ci_matrix(preds, len(ALL), n_boot=1000)
    mean = np.nanmean(full, 2); std = np.nanstd(full, 2)

    # ── overwrite the 4 leaked cells with HONEST held-out ────────────────────
    iP, iPS, iPC = ALL.index('POOLED'), ALL.index('P_S'), ALL.index('P_C')
    blP = bl['POOLED']
    gPS_pref = np.array([f'S:{x}' for x in groups_P_S[mask_P_S]])
    gPC_pref = np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])
    fixes = {}
    # row: POOLED -> P_S / P_C  (held-out comp subset of pooled test fold)
    for code, jj, nm in [(0, iPS, 'POOLED->P_S'), (1, iPC, 'POOLED->P_C')]:
        m_, s_, (lo, hi) = pooled_to_comp(feats_POOLED, y_pool, g_pool, dom, blP, code)
        mean[iP, jj], std[iP, jj], ci[iP, jj, 0], ci[iP, jj, 1] = m_, s_, lo, hi
        fixes[nm] = {'auc': m_, 'std': s_, 'ci': [lo, hi]}
    # col: P_S / P_C -> POOLED  (pooled rows disjoint from comp train fold)
    XPS = {li: fPS[li][mask_P_S] for li in fPS}; yPS = labels_P_S_lied[mask_P_S]
    XPC = {li: fPC[li][mask_P_C] for li in fPC}; yPC = labels_P_C_lied[mask_P_C]
    for Xc, yc_, gpref, blc, ii, nm in [
            (XPS, yPS, gPS_pref, bl['P_S'], iPS, 'P_S->POOLED'),
            (XPC, yPC, gPC_pref, bl['P_C'], iPC, 'P_C->POOLED')]:
        m_, s_, (lo, hi) = comp_to_pooled(Xc, yc_, gpref, blc, feats_POOLED, y_pool, g_pool)
        mean[ii, iP], std[ii, iP], ci[ii, iP, 0], ci[ii, iP, 1] = m_, s_, lo, hi
        fixes[nm] = {'auc': m_, 'std': s_, 'ci': [lo, hi]}

    print_transfer_matrix(full, ALL, f'Cross-transfer AUC ({frame}) — RAW (POOLED P_S/P_C leaked)')
    print(f'\n=== HONEST held-out corrections ({frame}) ===')
    for nm, d in fixes.items():
        print(f'  {nm:<14s} AUC = {d["auc"]:.3f} ± {d["std"]:.3f}  [{d["ci"][0]:.3f}, {d["ci"][1]:.3f}]  [held-out]')
    print(f'\nPOOLED row (honest), frame={frame}:')
    for j, nj in enumerate(ALL):
        if nj == 'POOLED':
            print(f'  POOLED -> POOLED  {mean[iP,j]:.3f} ± {std[iP,j]:.3f}  (held-out diagonal)')
        else:
            tag = '  [held-out]' if nj in ('P_S', 'P_C') else ''
            print(f'  POOLED -> {nj:<7s} {mean[iP,j]:.3f}  [{ci[iP,j,0]:.3f}, {ci[iP,j,1]:.3f}]{tag}')

    results[frame] = {
        'probes': ALL, 'best_layers': {k: int(v) for k, v in bl.items()},
        'mean': mean.tolist(), 'std': std.tolist(), 'ci': ci.tolist(),
        'honest_fixes': fixes,
        'pooled_row': {nj: float(mean[iP, j]) for j, nj in enumerate(ALL)},
        'pooled_col': {ni: float(mean[i, iP]) for i, ni in enumerate(ALL)}}

json.dump(results, open(op / 'pooled_generalization_indist.json', 'w'), indent=2)
print(f'\nSaved {op / "pooled_generalization_indist.json"} (frames: {list(results)})')
