#!/usr/bin/env python
"""Confidence meter vs lying-specific direction — the four analyses (CPU).

#1 Partial out confidence: project the per-row confidence feature c out of the answer-token
   activations (projection fit on the train split only), retrain the probe on residuals,
   fact-disjoint AUROC. Controls: permuted-c (must stay ~raw) and R² of decoding c from acts.
#2 Confidence x label: p_chosen for lie vs truth, off-diagonal counts (is #1 powered?).
#3 Confidence baseline on P_S / P_C: AUROC(-p_chosen, lied).
#4 Layer-emergence curve (raw + residual + confidence line).

Outputs $OUTPUT_DIR/box_results/mcq_confidence_results.json + 3 PNGs.

Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
     OUTPUT_DIR=~/output python experiments/factual_recall/mcq_confidence_analysis.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
DIAG = Path(OUTPUT_DIR) / 'diag'
RES = Path(OUTPUT_DIR) / 'box_results'
N_SEEDS = 10
TEST = 0.3
C = 1.0


def load_mcq():
    meta = np.load(DIAG / 'meta_MCQ.npz', allow_pickle=True)
    y = meta['labels'].astype(int); groups = meta['groups']; picked = meta['picked_letter']
    z = np.load(DIAG / 'feats_MCQ_resp.npz')
    feats = {int(k[1:]): z[k] for k in z.files}
    # dedup to unique (fact, picked_letter)
    seen, keep = set(), []
    for i, (g, p) in enumerate(zip(groups.tolist(), picked.tolist())):
        if (g, p) not in seen:
            seen.add((g, p)); keep.append(i)
    keep = np.array(keep)
    feats = {L: X[keep] for L, X in feats.items()}
    y, groups, picked = y[keep], groups[keep], picked[keep]
    conf = json.load(open(RES / 'mcq_conf_MCQ.json'))
    pch = np.array([conf[g]['probs'][p] for g, p in zip(groups.tolist(), picked.tolist())])
    logp = np.log(np.clip(pch, 1e-6, 1.0))
    logit_vec = np.array([[conf[g]['logits'][L] for L in ['A', 'B', 'C', 'D']]
                          for g in groups.tolist()])
    return feats, y, groups, pch, logp, logit_vec


def residual(X, Cmat, tr):
    """Project Cmat (n,k incl. intercept) out of X using a train-fit OLS; return residuals."""
    B, *_ = np.linalg.lstsq(Cmat[tr], X[tr], rcond=None)
    return X - Cmat @ B


def auroc_split(feats, y, groups, layer, cfeat, seeds=N_SEEDS, permute=False):
    """Mean fact-disjoint AUROC at one layer, optionally residualizing on cfeat (n,k0)."""
    X = feats[layer]; y = np.asarray(y); aucs = []
    rng = np.random.RandomState(0)
    for sd in range(seeds):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=sd).split(X, y, groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xr = X
        if cfeat is not None:
            cf = cfeat.copy()
            if permute:
                cf = cf[rng.permutation(len(cf))]
            Cmat = np.column_stack([np.ones(len(cf)), cf])
            Xr = residual(X, Cmat, tr)
        clf = LogisticRegression(max_iter=2000, C=C, random_state=sd).fit(Xr[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xr[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else None


def decode_c_r2(feats, logp, groups, layer, seeds=5):
    """R^2 of linearly decoding the confidence scalar from the activations (fact-disjoint)."""
    X = feats[layer]; r2 = []
    for sd in range(seeds):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=sd).split(X, logp, groups))
        m = Ridge(alpha=10.0).fit(X[tr], logp[tr])
        r2.append(r2_score(logp[te], m.predict(X[te])))
    return round(float(np.mean(r2)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layers', nargs='+', type=int, default=[14, 22, 30])
    ap.add_argument('--out', default=str(RES / 'mcq_confidence_results.json'))
    args = ap.parse_args()

    feats, y, groups, pch, logp, logit_vec = load_mcq()
    n = len(y); nlie = int(y.sum())
    print(f'MCQ dedup rows={n}  lie={nlie} truth={n-nlie}  facts={len(np.unique(groups))}', flush=True)

    c_scalar = logp.reshape(-1, 1)
    c_full = np.column_stack([logp, logit_vec])           # log p_chosen + 4 letter logits

    out = {'n': n, 'n_lie': nlie, 'n_truth': n - nlie}

    # ── #1 partial-out, at selected layers ──────────────────────────────────
    out['conf_alone_auc'] = round(float(roc_auc_score(y, -pch)), 3)
    print(f'#1 confidence-alone AUROC (-p_chosen) = {out["conf_alone_auc"]}', flush=True)
    out['partial_out'] = {}
    for L in args.layers:
        row = {
            'raw': round(auroc_split(feats, y, groups, L, None), 3),
            'resid_scalar': round(auroc_split(feats, y, groups, L, c_scalar), 3),
            'resid_full': round(auroc_split(feats, y, groups, L, c_full), 3),
            'resid_permuted_scalar': round(auroc_split(feats, y, groups, L, c_scalar, permute=True), 3),
            'decode_c_R2': decode_c_r2(feats, logp, groups, L),
        }
        out['partial_out'][L] = row
        print(f'  L{L}: raw={row["raw"]}  resid_scalar={row["resid_scalar"]}  '
              f'resid_full={row["resid_full"]}  perm_ctrl={row["resid_permuted_scalar"]}  '
              f'decode_c_R2={row["decode_c_R2"]}', flush=True)

    # ── #2 confidence x label distribution ──────────────────────────────────
    lie_p = pch[y == 1]; tru_p = pch[y == 0]
    tmed = float(np.median(tru_p))
    conf_lies = int((lie_p > 0.9).sum()); unconf_truths = int((tru_p < 0.5).sum())
    out['distribution'] = {
        'lie_pchosen_mean': round(float(lie_p.mean()), 3),
        'truth_pchosen_mean': round(float(tru_p.mean()), 3),
        'truth_median': round(tmed, 3),
        'confident_lies_gt0.9': conf_lies, 'frac_lies_gt0.9': round(conf_lies / len(lie_p), 3),
        'lies_above_truth_median': int((lie_p > tmed).sum()),
        'frac_lies_above_truth_median': round(float((lie_p > tmed).mean()), 3),
        'unconfident_truths_lt0.5': unconf_truths,
        'powered': bool(conf_lies >= 20 and (lie_p > tmed).mean() > 0.1),
    }
    print(f'#2 confident lies(>0.9)={conf_lies}/{len(lie_p)}  '
          f'frac lies>truth-median={out["distribution"]["frac_lies_above_truth_median"]}  '
          f'powered={out["distribution"]["powered"]}', flush=True)
    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 31)
    plt.hist(tru_p, bins=bins, alpha=0.6, label=f'truth (n={len(tru_p)})', color='tab:green', density=True)
    plt.hist(lie_p, bins=bins, alpha=0.6, label=f'lie (n={len(lie_p)})', color='tab:red', density=True)
    plt.axvline(tmed, ls='--', c='k', lw=1, label='truth median')
    plt.xlabel('p_chosen (model confidence in emitted letter)'); plt.ylabel('density')
    plt.title('MCQ: answer-confidence by label'); plt.legend()
    plt.tight_layout(); plt.savefig(RES / 'mcq_conf_dist.png', dpi=120); plt.close()

    # ── #3 P_S / P_C confidence baseline ────────────────────────────────────
    out['pressure_conf_baseline'] = {}
    for ds, fn in [('P_S', 'mcq_conf_PS.json'), ('P_C', 'mcq_conf_PC.json')]:
        p = RES / fn
        if not p.exists():
            out['pressure_conf_baseline'][ds] = {'note': 'missing'}; continue
        d = json.load(open(p))
        m = np.array(d['mask'], bool); pc = np.array(d['pchosen']); li = np.array(d['lied'])
        auc = float(roc_auc_score(li[m], -pc[m])) if len(np.unique(li[m])) > 1 else None
        out['pressure_conf_baseline'][ds] = {
            'neg_pchosen_auc': round(auc, 3) if auc is not None else None,
            'n': int(m.sum()), 'lied': int(li[m].sum()),
            'lie_pchosen_mean': round(float(pc[m & (li == 1)].mean()), 3),
            'truth_pchosen_mean': round(float(pc[m & (li == 0)].mean()), 3)}
        print(f'#3 {ds} confidence baseline AUROC(-p_chosen)='
              f'{out["pressure_conf_baseline"][ds]["neg_pchosen_auc"]} '
              f'(lie conf {out["pressure_conf_baseline"][ds]["lie_pchosen_mean"]} vs '
              f'truth {out["pressure_conf_baseline"][ds]["truth_pchosen_mean"]})', flush=True)
        if ds == 'P_S':
            plt.figure(figsize=(7, 4)); b = np.linspace(0.5, 1, 26)
            plt.hist(pc[m & (li == 0)], bins=b, alpha=0.6, label='truth', color='tab:green', density=True)
            plt.hist(pc[m & (li == 1)], bins=b, alpha=0.6, label='lie', color='tab:red', density=True)
            plt.xlabel('p_chosen (Yes/No)'); plt.ylabel('density')
            plt.title('P_S (salesperson): answer-confidence by label'); plt.legend()
            plt.tight_layout(); plt.savefig(RES / 'mcq_conf_PS_dist.png', dpi=120); plt.close()

    # ── #4 layer-emergence curve ────────────────────────────────────────────
    pr = json.load(open(RES / 'mcq_probe_results.json'))
    raw_curve = {int(k): v for k, v in pr['resp']['by_layer_crossfact_LR'].items()}
    layers = sorted(raw_curve)
    resid_curve = {L: round(auroc_split(feats, y, groups, L, c_scalar), 3) for L in layers}
    out['layer_curve'] = {'raw': raw_curve, 'resid_scalar': resid_curve}
    plt.figure(figsize=(8, 4.5))
    plt.plot(layers, [raw_curve[L] for L in layers], '-o', ms=3, label='raw probe (cross-fact LR)')
    plt.plot(layers, [resid_curve[L] for L in layers], '-s', ms=3,
             label='probe after partialling out confidence')
    plt.axhline(out['conf_alone_auc'], ls='--', c='gray', label=f'confidence alone ({out["conf_alone_auc"]})')
    plt.axhline(0.5, ls=':', c='k', lw=0.8)
    plt.xlabel('layer'); plt.ylabel('cross-fact AUROC'); plt.ylim(0.4, 1.0)
    plt.title('MCQ lie/truth: layer emergence (raw vs confidence-residualised)')
    plt.legend(); plt.tight_layout(); plt.savefig(RES / 'mcq_layer_curve.png', dpi=120); plt.close()

    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out} + 3 PNGs', flush=True)


if __name__ == '__main__':
    main()
