#!/usr/bin/env python
"""Score the MCQ-trained lie/truth probe on the pressure/instructed datasets (reverse of T4).

Fits the MCQ probe (LR direction + DiffMean direction) at every layer on the deduped MCQ
answer-token activations, then for each target {P_S, P_C, I_S, I_C} reports the transfer
AUROC (raw) at the MCQ best layer and at the best-matching layer, alongside the target's own
in-domain oracle (LR, fact/scenario-grouped held-out) as a ceiling reference.

Run (CPU, single-thread BLAS):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  OUTPUT_DIR=~/output python experiments/factual_recall/mcq_xfer_score.py --mcq-tag MCQ
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

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
DIAG = Path(OUTPUT_DIR) / 'diag'
C = 1.0


def load_feats(tag, pos):
    z = np.load(DIAG / f'feats_{tag}_{pos}.npz')
    return {int(k[1:]): z[k] for k in z.files}


def mcq_directions(tag):
    """Fit LR + DiffMean directions at every layer on deduped MCQ resp activations."""
    meta = np.load(DIAG / f'meta_{tag}.npz', allow_pickle=True)
    y = meta['labels'].astype(int); groups = meta['groups']; picked = meta['picked_letter']
    feats = load_feats(tag, 'resp')
    seen, keep = set(), []
    for i, (g, p) in enumerate(zip(groups.tolist(), picked.tolist())):
        if (g, p) not in seen:
            seen.add((g, p)); keep.append(i)
    keep = np.array(keep)
    y = y[keep]
    w_lr, w_dm = {}, {}
    for L, X in feats.items():
        Xk = X[keep]
        w_lr[L] = LogisticRegression(max_iter=2000, C=C).fit(Xk, y).coef_[0]
        w_dm[L] = Xk[y == 1].mean(0) - Xk[y == 0].mean(0)
    return w_lr, w_dm, len(y)


def oracle(feats, y, groups, mask, seeds=10):
    """In-domain ceiling: best-layer grouped held-out LR AUROC."""
    y = np.asarray(y); groups = np.asarray(groups)
    best = 0.0
    for L, X in feats.items():
        aucs = []
        for sd in range(seeds):
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=sd)
                          .split(X[mask], y[mask], groups[mask]))
            Xm, ym = X[mask], y[mask]
            if len(np.unique(ym[tr])) < 2 or len(np.unique(ym[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=C, random_state=sd).fit(Xm[tr], ym[tr])
            aucs.append(roc_auc_score(ym[te], clf.predict_proba(Xm[te])[:, 1]))
        if aucs:
            best = max(best, float(np.mean(aucs)))
    return round(best, 3)


def transfer(w_dict, feats, y, mask):
    """AUROC at every layer applying the MCQ direction; return {best_layer, best_auc, by_layer}."""
    y = np.asarray(y)[mask]
    by = {}
    for L, X in feats.items():
        if L not in w_dict:
            continue
        by[L] = round(float(roc_auc_score(y, X[mask] @ w_dict[L])), 3)
    bl = max(by, key=by.get)
    return {'best_layer': int(bl), 'best_auc': by[bl], 'by_layer': by}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcq-tag', default='MCQ')
    ap.add_argument('--targets', nargs='+', default=['P_S', 'P_C', 'I_S', 'I_C'])
    ap.add_argument('--mcq-layer', type=int, default=30, help='MCQ best resp layer (for the matched cell)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    out = args.out or str(Path(OUTPUT_DIR) / 'box_results' / f'mcq_xfer_{args.mcq_tag}.json')

    print(f'Fitting MCQ ({args.mcq_tag}) directions at all layers...', flush=True)
    w_lr, w_dm, n_mcq = mcq_directions(args.mcq_tag)

    results = {'mcq_tag': args.mcq_tag, 'n_mcq_rows': n_mcq, 'mcq_layer': args.mcq_layer, 'targets': {}}
    print(f'\n{"target":6}  {"n/lie":>9}  {"oracle":>6}  {"LR@L"+str(args.mcq_layer):>8}  '
          f'{"LR best":>14}  {"DM best":>14}', flush=True)
    for ds in args.targets:
        try:
            meta = np.load(DIAG / f'meta_{ds}.npz', allow_pickle=True)
        except FileNotFoundError:
            print(f'{ds}: meta not found — skip'); continue
        y = meta['labels'].astype(int); mask = meta['mask'].astype(bool); groups = meta['groups']
        feats = load_feats(ds, 'resp')
        tr_lr = transfer(w_lr, feats, y, mask)
        tr_dm = transfer(w_dm, feats, y, mask)
        lr_matched = round(float(roc_auc_score(np.asarray(y)[mask],
                          feats[args.mcq_layer][mask] @ w_lr[args.mcq_layer])), 3) \
                     if args.mcq_layer in feats else None
        orc = oracle(feats, y, groups, mask)
        n, nlie = int(mask.sum()), int(np.asarray(y)[mask].sum())
        results['targets'][ds] = {
            'n': n, 'n_lie': nlie, 'oracle_auc': orc,
            'LR_matched_layer': lr_matched,
            'LR_transfer': {'best_layer': tr_lr['best_layer'], 'best_auc': tr_lr['best_auc']},
            'DM_transfer': {'best_layer': tr_dm['best_layer'], 'best_auc': tr_dm['best_auc']},
            'LR_by_layer': tr_lr['by_layer'],
        }
        print(f'{ds:6}  {str(n)+"/"+str(nlie):>9}  {orc:>6}  {str(lr_matched):>8}  '
              f'{str(tr_lr["best_auc"])+" (L"+str(tr_lr["best_layer"])+")":>14}  '
              f'{str(tr_dm["best_auc"])+" (L"+str(tr_dm["best_layer"])+")":>14}', flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nWrote {out}', flush=True)


if __name__ == '__main__':
    main()
