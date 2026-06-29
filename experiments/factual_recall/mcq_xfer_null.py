#!/usr/bin/env python
"""Best-of-33-layers NULL for the T7 reverse-transfer cells (CPU).

The T7 numbers (MCQ direction -> P_S/P_C/I'_S/I'_C) are each max-over-33-layers, which
inflates peaks. This computes the null distribution of that same best-of-33 procedure for
directions with NO real relationship to the target, so the real peaks can be judged:

  nullA  one random unit vector applied across all 33 layers, take best layer.
  nullB  33 independent random unit vectors (one per layer), take best layer  (matches the
         real procedure's per-layer refitting -> the honest "33 chances to get lucky" null).
  nullC  MCQ LR directions fit on SHUFFLED MCQ labels, per layer, transferred best-of-33
         (procedure-identical to the real transfer, just with the MCQ signal destroyed).

Real best-of-33 AUROCs are read from mcq_xfer_{MCQ,MCQ_hot}.json. Output:
  $OUTPUT_DIR/box_results/mcq_xfer_null.json  + prints a verdict table.

Run: OMP_NUM_THREADS=1 ... OUTPUT_DIR=~/output \
     python experiments/factual_recall/mcq_xfer_null.py --n-dirs 2000 --n-shuf 30
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
DIAG = Path(OUTPUT_DIR) / 'diag'
RES = Path(OUTPUT_DIR) / 'box_results'
TARGETS = ['P_S', 'P_C', 'I_S', 'I_C']


def load_feats(tag, pos='resp'):
    z = np.load(DIAG / f'feats_{tag}_{pos}.npz')
    return {int(k[1:]): z[k] for k in z.files}


def fast_auc_multi(S, y):
    """Vectorised AUROC for a score matrix S (n, m) vs binary y -> (m,)."""
    y = np.asarray(y); npos = int(y.sum()); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.full(S.shape[1], np.nan)
    order = np.argsort(S, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(S.shape[0])[:, None]
    np.put_along_axis(ranks, order, rows + 1.0, axis=0)   # ranks 1..n (no ties: continuous scores)
    sumrk = ranks[y == 1].sum(0)
    return (sumrk - npos * (npos + 1) / 2.0) / (npos * nneg)


def target_layer_aucs(feats, y, mask, V):
    """For a fixed direction-set V (n_dirs,d) applied at EACH layer -> (n_layers, n_dirs) AUROC."""
    layers = sorted(feats); ym = np.asarray(y)[mask]
    out = np.zeros((len(layers), V.shape[0]))
    for li, L in enumerate(layers):
        S = feats[L][mask] @ V.T            # (n_mask, n_dirs)
        out[li] = fast_auc_multi(S, ym)
    return out


def rand_unit(n, d, rng):
    V = rng.standard_normal((n, d)); V /= np.linalg.norm(V, axis=1, keepdims=True); return V


def summ(a):
    a = np.asarray(a)
    return {'mean': round(float(a.mean()), 3), 'p95': round(float(np.percentile(a, 95)), 3),
            'p99': round(float(np.percentile(a, 99)), 3), 'max': round(float(a.max()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-dirs', type=int, default=2000)
    ap.add_argument('--n-shuf', type=int, default=30)
    ap.add_argument('--out', default=str(RES / 'mcq_xfer_null.json'))
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    real = {ds: {'MCQ': json.load(open(RES / 'mcq_xfer_MCQ.json'))['targets'][ds]['LR_transfer']['best_auc'],
                 'MCQ_hot': json.load(open(RES / 'mcq_xfer_MCQ_hot.json'))['targets'][ds]['LR_transfer']['best_auc']}
            for ds in TARGETS}

    tgt = {}
    for ds in TARGETS:
        m = np.load(DIAG / f'meta_{ds}.npz', allow_pickle=True)
        tgt[ds] = {'feats': load_feats(ds), 'y': m['labels'].astype(int), 'mask': m['mask'].astype(bool)}
        d = tgt[ds]['feats'][0].shape[1]

    out = {}
    # ── nullA: one random vector across all layers ; nullB: independent per layer ──
    for ds in TARGETS:
        f, y, mask = tgt[ds]['feats'], tgt[ds]['y'], tgt[ds]['mask']
        d = f[0].shape[1]; layers = sorted(f)
        VA = rand_unit(args.n_dirs, d, rng)
        aucsA = target_layer_aucs(f, y, mask, VA)        # (L, n_dirs)
        bestA = aucsA.max(0)
        # nullB: resample a fresh V per layer
        bAll = np.zeros((len(layers), args.n_dirs))
        for li, L in enumerate(layers):
            VL = rand_unit(args.n_dirs, d, rng)
            S = f[L][mask] @ VL.T
            bAll[li] = fast_auc_multi(S, np.asarray(y)[mask])
        bestB = bAll.max(0)
        out[ds] = {'n': int(mask.sum()), 'lied': int(np.asarray(y)[mask].sum()),
                   'real_MCQ': real[ds]['MCQ'], 'real_MCQ_hot': real[ds]['MCQ_hot'],
                   'nullA_one_vec': summ(bestA), 'nullB_per_layer': summ(bestB)}
        print(f'[{ds}] real MCQ={real[ds]["MCQ"]} hot={real[ds]["MCQ_hot"]}  '
              f'nullB p95={out[ds]["nullB_per_layer"]["p95"]} max={out[ds]["nullB_per_layer"]["max"]}',
              flush=True)

    # ── nullC: shuffled-label MCQ directions, best-of-33 (procedure-matched) ──
    print(f'\nnullC: {args.n_shuf} shuffled-label MCQ refits x 33 layers ...', flush=True)
    mm = np.load(DIAG / 'meta_MCQ.npz', allow_pickle=True)
    ym = mm['labels'].astype(int); gm = mm['groups']; pk = mm['picked_letter']
    seen, keep = set(), []
    for i, (g, p) in enumerate(zip(gm.tolist(), pk.tolist())):
        if (g, p) not in seen:
            seen.add((g, p)); keep.append(i)
    keep = np.array(keep)
    mfe = load_feats('MCQ'); mfe = {L: X[keep] for L, X in mfe.items()}
    ym = ym[keep]
    nullC = {ds: [] for ds in TARGETS}
    perm_rng = np.random.RandomState(1)
    for s in range(args.n_shuf):
        yp = perm_rng.permutation(ym)
        w = {}
        for L in sorted(mfe):
            w[L] = LogisticRegression(max_iter=1000, C=1.0).fit(mfe[L], yp).coef_[0]
        for ds in TARGETS:
            f, y, mask = tgt[ds]['feats'], tgt[ds]['y'], tgt[ds]['mask']
            ymk = np.asarray(y)[mask]
            best = max(float(fast_auc_multi((f[L][mask] @ w[L])[:, None], ymk)[0]) for L in sorted(f))
            nullC[ds].append(best)
        if (s + 1) % 10 == 0:
            print(f'  shuffle {s+1}/{args.n_shuf}', flush=True)
    for ds in TARGETS:
        out[ds]['nullC_shuffled_label'] = summ(nullC[ds])
        # verdict: real vs nullB p95 (the honest random null) and nullC p95
        rb = out[ds]['nullB_per_layer']['p95']; rc = out[ds]['nullC_shuffled_label']['p95']
        out[ds]['real_minus_nullB_p95'] = round(real[ds]['MCQ'] - rb, 3)
        out[ds]['real_minus_nullC_p95'] = round(real[ds]['MCQ'] - rc, 3)

    json.dump(out, open(args.out, 'w'), indent=2)
    print('\n' + '=' * 78)
    print(f'{"target":6}{"real MCQ":>10}{"nullA p95":>11}{"nullB p95":>11}{"nullC p95":>11}'
          f'{"real-nullC":>12}')
    for ds in TARGETS:
        o = out[ds]
        print(f'{ds:6}{o["real_MCQ"]:>10}{o["nullA_one_vec"]["p95"]:>11}'
              f'{o["nullB_per_layer"]["p95"]:>11}{o["nullC_shuffled_label"]["p95"]:>11}'
              f'{o["real_minus_nullC_p95"]:>12}')
    print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
