#!/usr/bin/env python
"""Step 4 (CPU): train/eval the MCQ lie/truth probe + confound diagnostics.

Central claim builder. From the cached answer-token (resp) and pre-answer (pre)
activations, with rollouts grouped by fact_id:

  * cross-fact held-out AUROC  (GroupShuffleSplit on fact_id) — the load-bearing number;
    "does the enactment lie-direction generalise to facts never seen?"
  * in-distribution AUROC      (random rollout split) — "is there ANY linear lie/truth
    separation when every external variable is pinned within an item?"
  * DiffMean twin + shuffled-label floor (estimator-robustness + ~0.5 control).
  * letter diagnostics: a 4-way letter decoder (how much letter identity is encoded) and
    a letter-disjoint transfer (train on correct_letter in {A,B}, test on {C,D} and the
    reverse) — a pure letter-reader cannot transfer, so a surviving AUROC proves lie-state.
  * cross-score the existing trained probes (probe_state.pkl: P_S, P_C, POOLED, Z-family)
    on this clean MCQ set.

Raw AUROC throughout (no de-inversion). Both positions (resp, pre), both estimators
(LR, DiffMean).

Run (single-thread BLAS!):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  OUTPUT_DIR=~/output python experiments/factual_recall/mcq_probe_eval.py
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, json, pickle, argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parents[2]))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
sys.path.insert(0, REPO_DIR)
import probe_utils as pu

N_SEEDS = 10
TEST_SIZE = 0.3
C = 1.0


def load_feats(diag, ds, pos):
    z = np.load(Path(diag) / f'feats_{ds}_{pos}.npz')
    return {int(k[1:]): z[k] for k in z.files}


def multiseed_at_layer(feats, y, groups, layer, fit_fn, grouped=True, seeds=N_SEEDS):
    """Mean/std held-out AUROC at one layer over seeds, raw AUROC."""
    X = feats[layer]
    y = np.asarray(y)
    aucs = []
    for sd in range(seeds):
        if grouped:
            sp = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=sd)
            tr, te = next(sp.split(X, y, groups))
        else:
            sp = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=sd)
            tr, te = next(sp.split(X, y))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = fit_fn(X[tr], y[tr], C=C, seed=sd)
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])))
    return {'auc': round(float(np.mean(aucs)), 3), 'std': round(float(np.std(aucs)), 3),
            'n_seeds': len(aucs)}


def eval_position(feats, y, groups, pos_name):
    mask = np.ones(len(y), dtype=bool)
    # layer selection by cross-fact held-out (grouped)
    best_layer, _ = pu.multiseed_best_layer(feats, y, mask, groups=groups,
                                             n_seeds=N_SEEDS, C=C, test_size=TEST_SIZE)
    res = {'best_layer': int(best_layer)}
    res['crossfact_LR'] = multiseed_at_layer(feats, y, groups, best_layer, pu._default_lr_fit, grouped=True)
    res['crossfact_DiffMean'] = multiseed_at_layer(feats, y, groups, best_layer, pu.fit_diffmean, grouped=True)
    res['indist_LR'] = multiseed_at_layer(feats, y, groups, best_layer, pu._default_lr_fit, grouped=False)
    res['indist_DiffMean'] = multiseed_at_layer(feats, y, groups, best_layer, pu.fit_diffmean, grouped=False)
    res['shuffled_floor'] = multiseed_at_layer(feats, y, groups, best_layer, pu.fit_lr_shuffled, grouped=True)
    # by-layer cross-fact curve (LR)
    res['by_layer_crossfact_LR'] = {
        li: multiseed_at_layer(feats, y, groups, li, pu._default_lr_fit, grouped=True)['auc']
        for li in sorted(feats)}
    print(f'[{pos_name}] best L{best_layer}  crossfact LR={res["crossfact_LR"]} '
          f'DM={res["crossfact_DiffMean"]}  indist LR={res["indist_LR"]}  '
          f'shuffled={res["shuffled_floor"]}', flush=True)
    return res


def letter_decoder(feats, picked, groups, best_layer, seeds=N_SEEDS):
    """4-way: predict picked_letter from the answer-token activation (how much letter ID)."""
    X = feats[best_layer]
    y = np.asarray(picked)
    accs, maucs = [], []
    for sd in range(seeds):
        tr, te = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=sd).split(X, y, groups))
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=C, random_state=sd).fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
        try:
            maucs.append(float(roc_auc_score(y[te], clf.predict_proba(X[te]),
                                             multi_class='ovr', labels=clf.classes_)))
        except ValueError:
            pass
    return {'acc': round(float(np.mean(accs)), 3) if accs else None,
            'macro_auc': round(float(np.mean(maucs)), 3) if maucs else None,
            'chance_acc': round(1.0 / len(np.unique(y)), 3)}


def letter_disjoint(feats, y, correct_letter, best_layer, fit_fn=pu._default_lr_fit):
    """Train on facts whose correct_letter in setA, test on setB (and reverse).

    Truth-letter distribution differs train<->test, so a pure letter-reader cannot
    transfer; a surviving lie/truth AUROC proves lie-state, not letter identity.
    """
    X = feats[best_layer]
    y = np.asarray(y)
    cl = np.asarray(correct_letter)
    out = {}
    for nameA, A, B in [('AB->CD', {'A', 'B'}, {'C', 'D'}),
                        ('CD->AB', {'C', 'D'}, {'A', 'B'})]:
        trm = np.isin(cl, list(A))
        tem = np.isin(cl, list(B))
        if len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            out[nameA] = None
            continue
        clf = fit_fn(X[trm], y[trm], C=C, seed=0)
        out[nameA] = round(float(roc_auc_score(y[tem], clf.predict_proba(X[tem])[:, 1])), 3)
    return out


# ── cross-score existing probes (best-effort against probe_state.pkl schema) ──────
CROSS_PROBES = ['P_S', 'P_C', 'POOLED', 'Z', 'Z_mean', 'Z_tok']


def _find_direction(state, name):
    """Return (w, best_layer, scaler) for a probe name, or (None, None, None)."""
    bl = state.get(f'best_layer_{name}')
    for wk in (f'w_{name}', f'dir_{name}'):
        if wk in state:
            return np.asarray(state[wk]), bl, state.get(f'scaler_{name}')
    clf = state.get(f'clf_{name}')
    if clf is not None and hasattr(clf, 'coef_'):
        return clf.coef_[0], bl, state.get(f'scaler_{name}')
    # dirs_<name> = {layer: vec}
    dirs = state.get(f'dirs_{name}')
    if isinstance(dirs, dict) and bl in dirs:
        return np.asarray(dirs[bl]), bl, state.get(f'scaler_{name}')
    return None, None, None


def cross_score(feats_resp, feats_pre, y):
    path = Path(OUTPUT_DIR) / 'probe_state.pkl'
    if not path.exists():
        print(f'  probe_state.pkl not found at {path}; skipping cross-score.', flush=True)
        return {'_missing': True}
    state = pickle.load(open(path, 'rb'))
    out = {'_available_keys': sorted([k for k in state if k.startswith(('w_', 'best_layer_', 'clf_', 'dirs_'))])}
    for name in CROSS_PROBES:
        w, bl, scaler = _find_direction(state, name)
        if w is None or bl is None:
            out[name] = {'note': 'direction/best_layer not found'}
            continue
        row = {'best_layer': int(bl)}
        for pos, feats in (('resp', feats_resp), ('pre', feats_pre)):
            if bl not in feats:
                row[pos] = None
                continue
            X = feats[bl]
            if scaler is not None and hasattr(scaler, 'transform'):
                try:
                    X = scaler.transform(X)
                except Exception:
                    pass
            score = X @ w
            row[pos] = round(float(roc_auc_score(y, score)), 3)
        out[name] = row
        print(f'  cross-score {name} (L{bl}): {out[name]}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--diag', default=str(Path(OUTPUT_DIR) / 'diag'))
    ap.add_argument('--tag', default='MCQ', help='cache tag to load (feats_<tag>_*, meta_<tag>)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--rollout-weighted', action='store_true',
                    help='keep every rollout (sampling-weighted, with exact-duplicate acts); '
                         'default dedups to unique (fact, picked_letter) representations')
    args = ap.parse_args()
    if args.out is None:
        suffix = '' if (args.tag == 'MCQ' and not args.rollout_weighted) else \
                 f'_{args.tag}' + ('_rolloutweighted' if args.rollout_weighted else '')
        args.out = str(Path(OUTPUT_DIR) / 'box_results' / f'mcq_probe_results{suffix}.json')

    meta = np.load(Path(args.diag) / f'meta_{args.tag}.npz', allow_pickle=True)
    y = meta['labels'].astype(int)             # lie=1, truth=0
    groups = meta['groups']
    correct_letter = meta['correct_letter']
    picked = meta['picked_letter']
    feats_resp = load_feats(args.diag, args.tag, 'resp')
    feats_pre = load_feats(args.diag, args.tag, 'pre')
    n_rollouts_raw = len(y)

    if not args.rollout_weighted:
        # answer-token activation is deterministic in (fact, letter); dedup the exact
        # duplicates so each distinct lie/truth representation is counted once.
        seen, keep = set(), []
        for i, (fid, pl) in enumerate(zip(groups.tolist(), picked.tolist())):
            k = (fid, pl)
            if k not in seen:
                seen.add(k); keep.append(i)
        keep = np.array(keep)
        feats_resp = {li: feats_resp[li][keep] for li in feats_resp}
        feats_pre = {li: feats_pre[li][keep] for li in feats_pre}
        y, groups = y[keep], groups[keep]
        correct_letter, picked = correct_letter[keep], picked[keep]

    n_facts = len(np.unique(groups))
    print(f'{n_rollouts_raw} rollouts -> {len(y)} unique (fact,letter) rows  '
          f'({int(y.sum())} lie / {int((1-y).sum())} truth)  over {n_facts} facts '
          f'[mode={"rollout-weighted" if args.rollout_weighted else "dedup"}]', flush=True)

    results = {
        'mode': 'rollout-weighted' if args.rollout_weighted else 'dedup',
        'n_rollouts_raw': int(n_rollouts_raw),
        'n_rows': int(len(y)), 'n_lie': int(y.sum()), 'n_truth': int((1 - y).sum()),
        'n_facts': int(n_facts),
        'resp': eval_position(feats_resp, y, groups, 'resp'),
        'pre': eval_position(feats_pre, y, groups, 'pre'),
    }
    bl_resp = results['resp']['best_layer']
    results['letter_decoder_resp'] = letter_decoder(feats_resp, picked, groups, bl_resp)
    results['letter_disjoint_resp_LR'] = letter_disjoint(feats_resp, y, correct_letter, bl_resp, pu._default_lr_fit)
    results['letter_disjoint_resp_DiffMean'] = letter_disjoint(feats_resp, y, correct_letter, bl_resp, pu.fit_diffmean)
    print(f'letter-decoder (resp L{bl_resp}): {results["letter_decoder_resp"]}', flush=True)
    print(f'letter-disjoint LR (resp): {results["letter_disjoint_resp_LR"]}', flush=True)
    results['cross_score'] = cross_score(feats_resp, feats_pre, y)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nWrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
