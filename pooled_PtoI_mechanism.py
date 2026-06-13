#!/usr/bin/env python
"""
Phase C + F: is P→I′ shared deception, or instruction-salience saturation?
Plus leakage-free (group-disjoint) + DiffMean rigor on P/POOLED→I′.

CPU-only. Reuses cached resp/pre acts + probe_state.pkl + inference_I_bundle.pkl.
Env: OUTPUT_DIR (default ~/output).
"""
import os, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from probe_utils import (multiseed_best_layer, fit_final_probe, fit_diffmean,
                         cross_transfer_matrix, group_disjoint_transfer,
                         all_layer_directions, _bootstrap_auc)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, TEST_SIZE, MAX_ITER = 32, 10, 1.0, 0.2, 2000
out = {}

PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
g = globals(); g.update(PS)

def load(tag, pos):
    return {li: np.load(op / f'acts_{tag}_{pos}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}_{pos}_layer_{li:03d}.npy').exists()}

def _fit(X, y, seed=0, C=PROBE_C):
    return LogisticRegression(max_iter=MAX_ITER, C=C, random_state=seed).fit(X, y)
def _auc(clf, X, y):
    return roc_auc_score(y, clf.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else 0.5

# resp-frame P/I features (the high-transfer frame)
fPS, fPC = load('P_S', 'resp'), load('P_C', 'resp')
fIS, fIC = load('I_S', 'resp'), load('I_C', 'resp')

# ── retrain resp best layers + build POOLED ──────────────────────────────────
bl = {}
for tag, (f, y, m, gr) in {
        'P_S': (fPS, labels_P_S_lied, mask_P_S, groups_P_S),
        'P_C': (fPC, labels_P_C_lied, mask_P_C, groups_P_C),
        'I_S': (fIS, labels_I_S, mask_I_S, groups_I_S),
        'I_C': (fIC, labels_I_C, mask_I_C, groups_I_C)}.items():
    bl[tag], _ = multiseed_best_layer(f, y, m, n_seeds=N_SEEDS, C=PROBE_C, groups=gr)
pl = sorted(set(fPS) & set(fPC))
fPOOL = {li: np.concatenate([fPS[li][mask_P_S], fPC[li][mask_P_C]], 0) for li in pl}
yPOOL = np.concatenate([labels_P_S_lied[mask_P_S], labels_P_C_lied[mask_P_C]])
gPOOL = np.concatenate([np.array([f'S:{x}' for x in groups_P_S[mask_P_S]]),
                        np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])])
mPOOL = np.ones(len(yPOOL), bool)
bl['POOLED'], _ = multiseed_best_layer(fPOOL, yPOOL, mPOOL, n_seeds=N_SEEDS, C=PROBE_C, groups=gPOOL)

probe_data = {
    "I'_S": (fIS, labels_I_S, mask_I_S, groups_I_S, bl['I_S'], PROBE_C, None),
    "I'_C": (fIC, labels_I_C, mask_I_C, groups_I_C, bl['I_C'], PROBE_C, None),
    "P_S":  (fPS, labels_P_S_lied, mask_P_S, groups_P_S, bl['P_S'], PROBE_C, None),
    "P_C":  (fPC, labels_P_C_lied, mask_P_C, groups_P_C, bl['P_C'], PROBE_C, None),
    "POOLED": (fPOOL, yPOOL, mPOOL, gPOOL, bl['POOLED'], PROBE_C, None),
}

# ════════ PHASE C: leakage-free (group-disjoint) + DiffMean on P/POOLED→I′ ════
print('\n===== PHASE C: P/POOLED → I′ rigor (resp frame) =====')
phaseC = {}
for ni, nj in [("P_S", "I'_S"), ("P_C", "I'_C")]:
    a, kept = group_disjoint_transfer(ni, nj, probe_data, n_seeds=N_SEEDS)
    phaseC[f'{ni}->{nj}'] = {'group_disjoint_auc': float(np.nanmean(a)),
                             'std': float(np.nanstd(a)), 'coverage': kept}
    print(f'  {ni}->{nj} group-disjoint AUC={np.nanmean(a):.3f}±{np.nanstd(a):.3f} (kept {kept:.2f})')

def pooled_to_I_disjoint(prefix, fI, yI, mI, gI, blP):
    """POOLED→I′ scoring only I′ rows whose scenario ∉ POOLED's per-seed S/C-train fold."""
    XI = fI[blP][mI]; yI = yI[mI]; gI = np.asarray(gI[mI])
    aucs = []
    idx = np.arange(len(yPOOL))
    for seed in range(N_SEEDS):
        tr, _ = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed).split(idx, yPOOL, gPOOL))
        if len(np.unique(yPOOL[tr])) < 2:
            continue
        clf = _fit(fPOOL[blP][tr], yPOOL[tr], seed)
        train_scen = {gg.split(':', 1)[1] for gg in gPOOL[tr] if gg.startswith(prefix + ':')}
        keep = np.array([str(gg) not in train_scen for gg in gI])
        if keep.sum() and len(np.unique(yI[keep])) > 1:
            aucs.append(_auc(clf, XI[keep], yI[keep]))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

for prefix, nj, fI, yI, mI, gI in [('S', "I'_S", fIS, labels_I_S, mask_I_S, groups_I_S),
                                    ('C', "I'_C", fIC, labels_I_C, mask_I_C, groups_I_C)]:
    m_, s_ = pooled_to_I_disjoint(prefix, fI, yI, mI, gI, bl['POOLED'])
    phaseC[f'POOLED->{nj}'] = {'group_disjoint_auc': m_, 'std': s_}
    print(f'  POOLED->{nj} group-disjoint AUC={m_:.3f}±{s_:.3f}')

# DiffMean twin of the same cells (full-set transfer; disjoint cells are clean)
dm, _ = cross_transfer_matrix(["I'_S", "I'_C", "P_S", "P_C", "POOLED"], probe_data, fit_fn=fit_diffmean)
names = ["I'_S", "I'_C", "P_S", "P_C", "POOLED"]
dmean = np.nanmean(dm, 2)
for ni, nj in [("P_S", "I'_S"), ("P_C", "I'_C"), ("POOLED", "I'_S"), ("POOLED", "I'_C")]:
    v = dmean[names.index(ni), names.index(nj)]
    phaseC.setdefault(f'{ni}->{nj}', {})['diffmean_auc'] = float(v)
    print(f'  {ni}->{nj} DiffMean AUC={v:.3f}')
out['phaseC'] = phaseC

# ════════ PHASE F1: compliance-contrast I′ (gating test) ═════════════════════
print('\n===== PHASE F1: re-run P→I′ against the FIXED (compliance-contrast) I′ =====')
phaseF1 = {}
ib = pickle.load(open(op / 'inference_I_bundle.pkl', 'rb'))
for comp_prefix, nj, fI, df_key, blsrc, fsrc, ysrc, msrc, grsrc in [
        ('S', "I'_S", fIS, 'df_I_sales_matched', 'P_S', fPS, labels_P_S_lied, mask_P_S, groups_P_S),
        ('C', "I'_C", fIC, 'df_I_combined_matched', 'P_C', fPC, labels_P_C_lied, mask_P_C, groups_P_C)]:
    df = ib[df_key]
    actually = df['actually_lied'].to_numpy().astype(int)
    clear = (~df['is_unclear'].to_numpy()) if 'is_unclear' in df.columns else np.ones(len(df), bool)
    comp_rate = float(actually.mean())
    # original instruction-polarity vs compliance labels agreement
    inst = labels_I_S if comp_prefix == 'S' else labels_I_C
    agree = float((actually == inst).mean())
    # train the P probe (resp) at its best layer; score I′ with BOTH labelings (clear rows)
    bP = blsrc
    blP = bl[blsrc]
    clf = _fit(fsrc[blP][msrc], ysrc[msrc])
    XI = fI[blP]
    auc_inst = _auc(clf, XI[clear & (np.ones(len(inst), bool))], inst[clear]) if len(np.unique(inst[clear])) > 1 else float('nan')
    auc_comp = _auc(clf, XI[clear], actually[clear]) if len(np.unique(actually[clear])) > 1 else float('nan')
    phaseF1[f'{blsrc}->{nj}'] = {'compliance_rate': comp_rate, 'label_agreement': agree,
                                 'auc_instruction_polarity': float(auc_inst),
                                 'auc_compliance_contrast': float(auc_comp), 'n_clear': int(clear.sum())}
    print(f'  {blsrc}->{nj}: compliance={comp_rate:.2f} agree={agree:.2f} | '
          f'AUC(instruction)={auc_inst:.3f}  AUC(compliance)={auc_comp:.3f}  (n_clear={int(clear.sum())})')
out['phaseF1'] = phaseF1

# ════════ PHASE F2: direction decomposition (residual after span(w_Z,w_I′)) ══
print('\n===== PHASE F2: residual of w_P_C after projecting out span(w_Z, w_I′_C) =====')
phaseF2 = {}
try:
    L = int(best_layer_P_C)
    dP, dI, dZ = dirs_P_C[L], dirs_I_C[L], dirs_Z.get(L) if isinstance(dirs_Z, dict) else None
    if dZ is None:
        # dirs_Z may be keyed differently; recompute Z direction at L
        feats_Z_L = np.concatenate([np.load(op / f'acts_Z_honest_layer_{L:03d}.npy'),
                                    np.load(op / f'acts_Z_deceptive_layer_{L:03d}.npy')], 0)
        dZ = (feats_Z_L[labels_Z == 1].mean(0) - feats_Z_L[labels_Z == 0].mean(0))
    B = np.stack([dI / np.linalg.norm(dI), dZ / np.linalg.norm(dZ)]).T   # 4096x2
    proj = B @ np.linalg.lstsq(B, dP, rcond=None)[0]
    resid = dP - proj
    # eval residual on I′_C (pre acts at L) and Z
    XIc = np.load(op / f'acts_I_C_pre_layer_{L:03d}.npy')[mask_I_C]
    sI = XIc @ resid
    aucI = roc_auc_score(labels_I_C[mask_I_C], sI)
    aucI = max(aucI, 1 - aucI)
    cos_resid = float(resid @ dP / (np.linalg.norm(resid) * np.linalg.norm(dP)))
    phaseF2 = {'layer': L, 'resid_frac_of_wP': float(np.linalg.norm(resid) / np.linalg.norm(dP)),
               'cos(resid,wP)': cos_resid, 'residual_AUC_on_I_C': float(aucI)}
    print(f'  L={L}: |resid|/|wP|={phaseF2["resid_frac_of_wP"]:.3f}, '
          f'residual→I′_C AUC={aucI:.3f} (deception signal beyond instruction-salience if >>0.5)')
except Exception as e:
    phaseF2 = {'error': f'{type(e).__name__}: {e}'}
    print('  F2 skipped:', phaseF2['error'])
out['phaseF2'] = phaseF2

# ════════ PHASE F3: random-direction separability of each test set ═══════════
print('\n===== PHASE F3: fraction of random directions scoring AUC>0.7 per test set =====')
phaseF3 = {}
rng = np.random.RandomState(0)
NR = 2000
def rand_frac(X, y, thr=0.7):
    X = X - X.mean(0)
    D = rng.randn(X.shape[1], NR); D /= np.linalg.norm(D, axis=0, keepdims=True)
    S = X @ D                                  # n x NR
    aucs = np.array([roc_auc_score(y, S[:, k]) for k in range(NR)])
    aucs = np.maximum(aucs, 1 - aucs)
    return float((aucs > thr).mean()), float(np.median(aucs))
for nm, fI, mI, yI, blk in [("I'_C", fIC, mask_I_C, labels_I_C, bl['I_C']),
                            ("P_C", fPC, mask_P_C, labels_P_C_lied, bl['P_C'])]:
    fr, med = rand_frac(fI[blk][mI], yI[mI])
    phaseF3[nm] = {'frac_rand_auc>0.7': fr, 'median_rand_auc': med, 'layer': int(blk)}
    print(f'  {nm}: frac(rand AUC>0.7)={fr:.3f}, median rand |AUC|={med:.3f}')
out['phaseF3'] = phaseF3

# ════════ PHASE F4: AUC-vs-C for P_S→I′_S + cosine rotation toward w_I′_S ═════
print('\n===== PHASE F4: P_S→I′_S AUC vs C, and cosine(P_S dir, w_I′_S) =====')
phaseF4 = {}
Lp = bl['P_S']
XPS = fPS[Lp][mask_P_S]; yPS = labels_P_S_lied[mask_P_S]
XIS = fIS[Lp][mask_I_S]; yIS = labels_I_S[mask_I_S]
wI = dirs_I_S[Lp] if Lp in dirs_I_S else (XIS[yIS == 1].mean(0) - XIS[yIS == 0].mean(0))
wI = wI / np.linalg.norm(wI)
for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    clf = _fit(XPS, yPS, C=C)
    auc = _auc(clf, XIS, yIS)
    w = clf.coef_[0]; cos = float(w @ wI / (np.linalg.norm(w) * np.linalg.norm(wI)))
    phaseF4[str(C)] = {'auc_P_S->I_S': float(auc), 'cos(P_S_dir, w_I_S)': cos}
    print(f'  C={C:<6}: P_S→I′_S AUC={auc:.3f}  cos(dir,w_I′_S)={cos:+.3f}')
out['phaseF4'] = phaseF4

json.dump(out, open(op / 'pooled_PtoI_mechanism.json', 'w'), indent=2)
print(f'\nSaved {op / "pooled_PtoI_mechanism.json"}')
