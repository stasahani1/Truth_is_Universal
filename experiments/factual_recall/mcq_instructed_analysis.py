#!/usr/bin/env python
"""Instructed-incorrect MCQ analysis — the powered confidence-independence test (CPU).

Uses the instructed-incorrect activations (mcq_instructed_extract.py). The contrast is
  lie   = DEC-instructed rows the model actually got wrong (it complied)
  truth = HON-instructed rows it got right
both on facts it provably knows, so the lie label is clean AND lies can be confident.

#A  Are instructed-lies confident? (off-diagonal that pressure-MCQ lacked). fig + counts.
#B  Does the existing pressure-trained MCQ probe fire on instructed lies? (direction transfer)
#C  In-domain probe on the instructed set, then PARTIAL OUT confidence and refit — now
    powered. Residual AUROC well above chance + permuted control => lie-direction, not a meter.
#D  Confidence baseline (-p_chosen) on the instructed set.

Run: OMP_NUM_THREADS=1 ... OUTPUT_DIR=~/output \
     python experiments/factual_recall/mcq_instructed_analysis.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
DIAG = Path(OUTPUT_DIR) / 'diag'
RES = Path(OUTPUT_DIR) / 'box_results'
N_SEEDS, TEST, C = 10, 0.3, 1.0


def load_feats(tag, pos='resp'):
    z = np.load(DIAG / f'feats_{tag}_{pos}.npz')
    return {int(k[1:]): z[k] for k in z.files}


def residual(X, Cmat, tr):
    B, *_ = np.linalg.lstsq(Cmat[tr], X[tr], rcond=None)
    return X - Cmat @ B


def auroc_split(feats, y, groups, layer, cfeat=None, seeds=N_SEEDS, permute=False):
    X = feats[layer]; y = np.asarray(y); aucs = []; rng = np.random.RandomState(0)
    for sd in range(seeds):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST, random_state=sd).split(X, y, groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        Xr = X
        if cfeat is not None:
            cf = cfeat.copy()
            if permute:
                cf = cf[rng.permutation(len(cf))]
            Xr = residual(X, np.column_stack([np.ones(len(cf)), cf]), tr)
        clf = LogisticRegression(max_iter=2000, C=C, random_state=sd).fit(Xr[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xr[te])[:, 1]))
    return round(float(np.mean(aucs)), 3) if aucs else None


def best_layer(feats, y, groups, cfeat=None):
    best, bl = -1, None
    for L in sorted(feats):
        a = auroc_split(feats, y, groups, L, cfeat)
        if a is not None and a > best:
            best, bl = a, L
    return bl, round(best, 3)


def mcq_pressure_dirs():
    """LR direction per layer from the pressure MCQ probe (deduped). None if cache absent."""
    p = DIAG / 'meta_MCQ.npz'
    if not p.exists() or not (DIAG / 'feats_MCQ_resp.npz').exists():
        return None
    m = np.load(p, allow_pickle=True); y = m['labels'].astype(int)
    g = m['groups']; pk = m['picked_letter']
    seen, keep = set(), []
    for i, (a, b) in enumerate(zip(g.tolist(), pk.tolist())):
        if (a, b) not in seen:
            seen.add((a, b)); keep.append(i)
    keep = np.array(keep); feats = load_feats('MCQ'); y = y[keep]
    return {L: LogisticRegression(max_iter=2000, C=C).fit(X[keep], y).coef_[0] for L, X in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='MCQI')
    ap.add_argument('--contrast', choices=['dec_hon', 'dec_internal'], default='dec_hon',
                    help='dec_hon: DEC-lie vs HON-truth (prompt differs). '
                         'dec_internal: within DEC instruction, complied-lie vs resisted-truth (prompt frame matched).')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.out is None:
        sfx = ('' if args.tag == 'MCQI' else '_' + args.tag) + ('' if args.contrast == 'dec_hon' else '_internal')
        args.out = str(RES / f'mcq_instructed_results{sfx}.json')

    m = np.load(DIAG / f'meta_{args.tag}.npz', allow_pickle=True)
    cond = m['condition'].astype(int); picked = m['picked_letter']; correct = m['correct_letter']
    pchosen = m['pchosen']; groups = m['groups']
    feats = load_feats(args.tag)

    if args.contrast == 'dec_internal':
        # within the DEC instruction only: lie = complied (gave wrong), truth = resisted (gave correct)
        lie_rows = (cond == 1) & (picked != correct)
        tru_rows = (cond == 1) & (picked == correct)
    else:
        lie_rows = (cond == 1) & (picked != correct)      # DEC-lie
        tru_rows = (cond == 0) & (picked == correct)      # HON-truth
    mask = lie_rows | tru_rows
    y = lie_rows[mask].astype(int)                         # lie=1
    g = groups[mask]; pc = pchosen[mask]
    fm = {L: X[mask] for L, X in feats.items()}
    n_lie, n_tru = int(y.sum()), int((1 - y).sum())
    out = {'n_facts': int(len(np.unique(groups))), 'contrast': args.contrast,
           'dec_compliance': round(float(((cond == 1) & (picked != correct)).sum() / (cond == 1).sum()), 3),
           'n_lie': n_lie, 'n_truth': n_tru}

    # ── #A confidence of instructed lies ────────────────────────────────────
    lie_p, tru_p = pc[y == 1], pc[y == 0]
    out['A_confidence'] = {
        'lie_pchosen_mean': round(float(lie_p.mean()), 3),
        'truth_pchosen_mean': round(float(tru_p.mean()), 3),
        'confident_lies_gt0.9': int((lie_p > 0.9).sum()),
        'frac_lies_gt0.9': round(float((lie_p > 0.9).mean()), 3),
        'frac_lies_gt0.5': round(float((lie_p > 0.5).mean()), 3)}
    print(f'#A instructed-lie confidence: mean {out["A_confidence"]["lie_pchosen_mean"]}, '
          f'confident(>0.9) {out["A_confidence"]["confident_lies_gt0.9"]}/{n_lie} '
          f'({out["A_confidence"]["frac_lies_gt0.9"]})  [pressure-MCQ had 1/949]', flush=True)
    plt.figure(figsize=(7, 4)); b = np.linspace(0, 1, 31)
    plt.hist(tru_p, bins=b, alpha=0.6, label=f'truth (n={n_tru})', color='tab:green', density=True)
    plt.hist(lie_p, bins=b, alpha=0.6, label=f'lie (n={n_lie})', color='tab:red', density=True)
    plt.xlabel('p_chosen'); plt.ylabel('density'); plt.legend()
    plt.title('Instructed-incorrect MCQ: confidence by label (lies are now committed)')
    plt.tight_layout(); plt.savefig(RES / 'mcq_instructed_conf_dist.png', dpi=120); plt.close()

    # ── #D confidence baseline ──────────────────────────────────────────────
    out['D_confidence_baseline_auc'] = round(float(roc_auc_score(y, -pc)), 3) if n_lie and n_tru else None

    # ── #C in-domain probe + partial-out (powered test) ─────────────────────
    bl, raw = best_layer(fm, y, g)
    c_scalar = np.log(np.clip(pc, 1e-6, 1)).reshape(-1, 1)
    out['C_indomain'] = {
        'best_layer': int(bl), 'raw_auc': raw,
        'resid_auc': auroc_split(fm, y, g, bl, c_scalar),
        'resid_permuted_ctrl': auroc_split(fm, y, g, bl, c_scalar, permute=True)}
    print(f'#C instructed in-domain: best L{bl} raw={raw}  resid={out["C_indomain"]["resid_auc"]}  '
          f'perm_ctrl={out["C_indomain"]["resid_permuted_ctrl"]}  conf_baseline={out["D_confidence_baseline_auc"]}',
          flush=True)

    # ── #B existing pressure-MCQ probe fires on instructed lies? ────────────
    dirs = mcq_pressure_dirs()
    if dirs is not None:
        by = {L: round(float(roc_auc_score(y, fm[L] @ dirs[L])), 3) for L in fm if L in dirs}
        blx = max(by, key=by.get)
        out['B_pressure_probe_transfer'] = {'best_layer': int(blx), 'best_auc': by[blx],
                                            'at_resid_layer_L30': by.get(30)}
        print(f'#B pressure-MCQ direction on instructed lies: best L{blx} = {by[blx]}', flush=True)
    else:
        out['B_pressure_probe_transfer'] = {'note': 'feats_MCQ pressure cache absent'}

    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out} + mcq_instructed_conf_dist.png', flush=True)


if __name__ == '__main__':
    main()
