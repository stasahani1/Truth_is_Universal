"""
Probe & analysis utilities for mechanism comparison pipeline.

Extracted from mechanism_comparison_colab.ipynb — pure numpy/sklearn
functions that don't depend on model/tokenizer.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ── Module-level defaults ─────────────────────────────────────────────────────
# NOTE: These are fallback defaults for standalone use of this module. They are
# DELIBERATELY excluded from __all__ below so that `from probe_utils import *`
# in the notebook cannot clobber the notebook's own config constants (e.g.
# ZOU_PROBE_C). The notebook always passes C explicitly, so these are inert there.
MAX_ITER = 2000
N_SEEDS = 10
TEST_SIZE = 0.2
PROBE_C = 1.0
ZOU_PROBE_C = 0.1
N_RANDOM_DIRS = 1000

# Only export functions via `import *` — never the config constants above, and
# never the private helpers (leading underscore). This keeps the notebook's
# configuration cell as the single source of truth for hyperparameters.
__all__ = [
    'fit_probe', 'eval_probe', 'probe_direction', 'multiseed_best_layer',
    'cosine_sim', 'random_cosine_null', 'cosine_zscore', 'random_direction_auc',
    'fit_final_probe',
    'all_layer_directions', 'multiseed_layer_cosines', 'cross_transfer_matrix',
    'cross_transfer_matrix_oracle_layer', 'group_disjoint_transfer',
    'bootstrap_ci_matrix', 'print_transfer_matrix', 'fit_tok_probe',
    'multiseed_best_layer_tok', 'all_layer_directions_tok', 'multi_layer_range',
    'fit_multi_tok_probe', 'multi_probe_directions', 'multiseed_multi_tok_auc',
    'fit_diffmean', 'fit_lr_shuffled',
]


# ── Core probe functions (from Cell 7) ────────────────────────────────────────

def fit_probe(X, y, C=PROBE_C, seed=42, class_weight=None, max_iter=MAX_ITER):
    """Fit a logistic-regression probe."""
    return LogisticRegression(
        max_iter=max_iter, C=C, random_state=seed, class_weight=class_weight,
    ).fit(X, y)


def eval_probe(clf, X, y):
    """Evaluate a fitted probe. Returns {auc, acc, probs}."""
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
    acc = float(accuracy_score(y, preds))
    return {'auc': auc, 'acc': acc, 'probs': probs}


def probe_direction(clf, scaler=None):
    """Return the unit-normalised weight vector of a fitted probe.

    If *scaler* (a fitted StandardScaler) is provided the weight vector is
    mapped back to the original (un-scaled) feature space first.  This is
    needed for token-level probes that use StandardScaler.
    """
    w = clf.coef_[0].copy()
    if scaler is not None:
        # w_original = w / scale  (bias absorbed into intercept)
        # Guard against zero-variance features (scale_ == 0) from float16 quantization
        safe_scale = scaler.scale_.copy()
        safe_scale[safe_scale == 0] = 1.0
        w = w / safe_scale
    n = np.linalg.norm(w)
    if n > 0:
        w /= n
    return w


def multiseed_best_layer(feats_dict, labels, mask, n_seeds=N_SEEDS, C=PROBE_C,
                         groups=None, class_weight=None, test_size=TEST_SIZE,
                         max_iter=MAX_ITER):
    """Sweep all layers x seeds; return best_layer, per_layer_aucs.

    Args:
        feats_dict: {layer_idx: np.ndarray(n, d)}
        labels: np.ndarray(n,) int
        mask: np.ndarray(n,) bool — which entries to use
        groups: optional np.ndarray(n,) for grouped splits
        class_weight: passed to LogisticRegression

    Returns:
        best_layer (int), layer_aucs (n_seeds x n_layers array)
    """
    n_layers = len(feats_dict)
    valid_labels = labels[mask]
    valid_feats = {li: feats_dict[li][mask] for li in range(n_layers)}
    n = int(mask.sum())

    per_seed_aucs = []
    for seed in range(n_seeds):
        if groups is not None:
            valid_groups = groups[mask]
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                   random_state=seed)
            tr_idx, ho_idx = next(gss.split(np.arange(n), valid_labels,
                                            valid_groups))
        else:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            split = int(n * (1 - test_size))
            tr_idx, ho_idx = perm[:split], perm[split:]

        if (len(np.unique(valid_labels[tr_idx])) < 2 or
                len(np.unique(valid_labels[ho_idx])) < 2):
            continue

        seed_aucs = []
        for li in range(n_layers):
            clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed,
                                     class_weight=class_weight)
            clf.fit(valid_feats[li][tr_idx], valid_labels[tr_idx])
            probs = clf.predict_proba(valid_feats[li][ho_idx])[:, 1]
            seed_aucs.append(float(roc_auc_score(valid_labels[ho_idx], probs)))
        per_seed_aucs.append(seed_aucs)

    arr = np.array(per_seed_aucs)  # (n_valid_seeds, n_layers)
    mean_ho = arr.mean(axis=0)
    best = int(np.argmax(mean_ho))
    print(f'  Best layer: {best}  (mean holdout AUC = {mean_ho[best]:.3f}, '
          f'std = {arr[:, best].std():.3f}, {arr.shape[0]} seeds)')
    return best, arr


def cosine_sim(v1, v2):
    """Cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm (degenerate probe direction,
    e.g. from constant features or strong regularisation).
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def random_cosine_null(dim, n_samples=N_RANDOM_DIRS):
    """Sample random unit vectors and compute pairwise cosine distribution."""
    vecs = np.random.randn(n_samples, dim)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    cosines = []
    for i in range(n_samples):
        for j in range(i + 1, min(i + 50, n_samples)):
            cosines.append(float(np.dot(vecs[i], vecs[j])))
    cosines = np.array(cosines)
    return {'mean': float(cosines.mean()), 'std': float(cosines.std()),
            'p5': float(np.percentile(cosines, 5)),
            'p95': float(np.percentile(cosines, 95))}


def cosine_zscore(observed, null_mean, null_std):
    """Z-score of an observed cosine against a null distribution."""
    return (observed - null_mean) / null_std if null_std > 0 else 0.0


def random_direction_auc(X, y, n_dirs=N_RANDOM_DIRS, seed=0):
    """Test-set 'easiness' floor: AUC achievable by RANDOM directions.

    Samples *n_dirs* random unit vectors, projects X onto each, and computes the
    sign-agnostic AUC max(auc, 1-auc) vs labels y (direction sign is arbitrary).
    A high p95/max means the labels are separable along *many* directions — i.e.
    the test set is trivially easy — so a high cross-transfer AUC into this set
    reflects test-set easiness, not a meaningful shared direction.

    Returns {mean, std, p95, max, frac_above_*} over the n_dirs AUCs.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {'mean': 0.5, 'std': 0.0, 'p95': 0.5, 'max': 0.5,
                'frac_above_0.7': 0.0, 'frac_above_0.8': 0.0}
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    aucs = np.empty(n_dirs)
    for i in range(n_dirs):
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        a = roc_auc_score(y, X @ v)
        aucs[i] = max(a, 1.0 - a)          # sign of the random direction is arbitrary
    return {'mean': float(aucs.mean()), 'std': float(aucs.std()),
            'p95': float(np.percentile(aucs, 95)), 'max': float(aucs.max()),
            'frac_above_0.7': float((aucs > 0.7).mean()),
            'frac_above_0.8': float((aucs > 0.8).mean())}


# ── Probe training helpers (from Cell 25) ─────────────────────────────────────

def fit_final_probe(feats, labels, mask, best_layer, C=PROBE_C, cw=None,
                    max_iter=MAX_ITER):
    """Fit a final probe at the best layer on all masked data."""
    X = feats[best_layer][mask]
    y = labels[mask]
    clf = fit_probe(X, y, C=C, seed=0, class_weight=cw, max_iter=max_iter)
    res = eval_probe(clf, X, y)
    w = probe_direction(clf)
    print(f'  Final probe: AUC={res["auc"]:.3f}, acc={res["acc"]:.3f}, n={len(y)}')
    return clf, w, res


def all_layer_directions(feats, labels, mask, C=PROBE_C, cw=None, n_layers=None,
                         max_iter=MAX_ITER):
    """Fit a probe at each layer and return unit direction vectors."""
    if n_layers is None:
        n_layers = len(feats)
    dirs = {}
    for li in range(n_layers):
        if li not in feats:
            continue
        X = feats[li][mask]
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        clf = fit_probe(X, y, C=C, seed=0, class_weight=cw, max_iter=max_iter)
        dirs[li] = probe_direction(clf)
    return dirs


# ── Cosine analysis (from Cell 30) ────────────────────────────────────────────

def _seed_train_idx(n, labels, groups, seed, test_size=TEST_SIZE):
    """Per-seed train-split indices for a resample.

    GroupShuffleSplit on *groups* when present (so scenarios don't straddle the
    split), else a plain RandomState(seed) permutation split. Falls back to all
    rows if the resulting train split is single-class (degenerate). This is what
    gives a per-seed FIT genuine sampling variance — without it, refitting on the
    full data every seed with only random_state varying is a deterministic
    point estimate under lbfgs (the fake-zero-std bug).
    """
    idx_all = np.arange(n)
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                random_state=seed)
        tr_idx, _ = next(gss.split(idx_all, labels, groups))
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        cut = int(n * (1 - test_size))
        tr_idx = perm[:cut]
    if len(np.unique(labels[tr_idx])) < 2:
        return idx_all
    return tr_idx


def multiseed_layer_cosines(feats_a, labels_a, mask_a, groups_a, C_a,
                             feats_b, labels_b, mask_b, groups_b, C_b,
                             n_seeds=N_SEEDS, cw_a=None, cw_b=None,
                             precomputed_dirs_a=None, precomputed_dirs_b=None,
                             max_iter=MAX_ITER):
    """Compute cosine(w_a, w_b) at each layer across multiple seeds.

    Each seed fits each (non-precomputed) side on a different per-seed subsample
    of its data (group-aware when groups are present), so the cosine's per-seed
    std reflects genuine sampling variance of the fitted directions. Fitting on
    the full data every seed with only random_state varying would be a
    deterministic point estimate under lbfgs (std == 0, a fake spread).

    If *precomputed_dirs_a* or *precomputed_dirs_b* is provided (a dict
    {layer: unit_vector}), those directions are used instead of fitting — they
    are a single fixed direction, so they contribute no spread. A pair with BOTH
    sides precomputed (e.g. Z_tok vs Z_multi) therefore has an inherent std of 0;
    callers should label such pairs as single-direction rather than presenting
    ±0.000 next to genuine spreads.
    """
    n_layers = min(len(feats_a), len(feats_b))
    cosines = np.zeros((n_seeds, n_layers))

    for seed in range(n_seeds):
        # Per-seed resample indices for each side that is actually fit
        # (skipped for precomputed sides). Row identity is consistent across
        # layers, so the indices are computed once per side per seed.
        tr_a = None
        if precomputed_dirs_a is None:
            ya_full = labels_a[mask_a]
            ga_full = groups_a[mask_a] if groups_a is not None else None
            if len(np.unique(ya_full)) >= 2:
                tr_a = _seed_train_idx(len(ya_full), ya_full, ga_full, seed)
        tr_b = None
        if precomputed_dirs_b is None:
            yb_full = labels_b[mask_b]
            gb_full = groups_b[mask_b] if groups_b is not None else None
            if len(np.unique(yb_full)) >= 2:
                tr_b = _seed_train_idx(len(yb_full), yb_full, gb_full, seed)

        for li in range(n_layers):
            if li not in feats_a or li not in feats_b:
                continue

            # Direction A
            if precomputed_dirs_a is not None and li in precomputed_dirs_a:
                wa = precomputed_dirs_a[li]
            elif precomputed_dirs_a is not None:
                # Precomputed dirs provided but this layer is missing —
                # skip rather than fitting a spurious single-layer probe
                continue
            else:
                if tr_a is None:
                    continue
                Xa = feats_a[li][mask_a][tr_a]; ya = labels_a[mask_a][tr_a]
                clf_a = LogisticRegression(max_iter=max_iter, C=C_a,
                                           random_state=seed,
                                           class_weight=cw_a).fit(Xa, ya)
                wa = probe_direction(clf_a)

            # Direction B
            if precomputed_dirs_b is not None and li in precomputed_dirs_b:
                wb = precomputed_dirs_b[li]
            elif precomputed_dirs_b is not None:
                # Precomputed dirs provided but this layer is missing — skip
                continue
            else:
                if tr_b is None:
                    continue
                Xb = feats_b[li][mask_b][tr_b]; yb = labels_b[mask_b][tr_b]
                clf_b = LogisticRegression(max_iter=max_iter, C=C_b,
                                           random_state=seed,
                                           class_weight=cw_b).fit(Xb, yb)
                wb = probe_direction(clf_b)

            cosines[seed, li] = cosine_sim(wa, wb)

    return cosines  # (n_seeds, n_layers)


# ── Cross-transfer (from Cell 32) ─────────────────────────────────────────────

def _bootstrap_auc(y_true, y_score, n_boot=1000, seed=42):
    """Bootstrap 95% CI for AUC. Returns (lower, upper)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _subsample_tok_for_train(tfh, tfd, ch, cd, layers, seed):
    """Prompt-level subsample of token features for one seed.

    Each seed trains on a different random subset of *prompts* (Z prompts are
    independent fact fragments with no group key, so a plain RandomState(seed)
    prompt split is used). This gives the token-level rows of the cross-transfer
    matrix genuine seed variance — with the default lbfgs solver, random_state
    alone is inert, so fitting on all tokens every seed would be a deterministic
    point estimate.

    Returns (tfh_sub, tfd_sub, ch_sub, cd_sub, diag_ho_idx, n_te_h):
      - tfh_sub/tfd_sub: {layer: tokens} dicts containing only train-prompt
        tokens, for the requested layers (feed straight to fit_tok_probe /
        fit_multi_tok_probe — no change to their internals).
      - ch_sub/cd_sub: per-prompt token counts for the train prompts.
      - diag_ho_idx: held-out prompt rows in the mean-pooled feats_Z_mean layout
        (honest prompt p -> row p, deceptive prompt q -> row len(ch) + q), used
        to score the diagonal without train-set leakage.
      - n_te_h: number of held-out HONEST prompts, i.e. how many leading entries
        of diag_ho_idx are label-0. Callers assert the diagonal label layout
        (first n_te_h rows honest, rest deceptive) to catch a changed Z_mean
        row ordering / masking.
    """
    (tr_tok_h, tr_tok_d, _te_tok_h, _te_tok_d, _tech, _tecd,
     tr_prompts_h, tr_prompts_d, te_prompts_h, te_prompts_d) = _tok_prompt_split(
        ch, cd, seed)
    tfh_sub = {L: tfh[L][tr_tok_h] for L in layers}
    tfd_sub = {L: tfd[L][tr_tok_d] for L in layers}
    ch_sub = ch[tr_prompts_h]
    cd_sub = cd[tr_prompts_d]
    diag_ho_idx = np.concatenate([te_prompts_h, len(ch) + te_prompts_d])
    return tfh_sub, tfd_sub, ch_sub, cd_sub, diag_ho_idx, len(te_prompts_h)


# ── Pluggable estimators for the cross-transfer standard branch ───────────────
# The standard branch of cross_transfer_matrix fits its probe through `fit_fn`.
# The DEFAULT (_default_lr_fit) reproduces the original inline LogisticRegression
# call exactly, so passing fit_fn=None leaves every existing number unchanged.
# Swapping in fit_diffmean / fit_lr_shuffled reuses the SAME per-seed subsample,
# direction, and score-all-of-j machinery — only the estimator changes.

def _stable_sigmoid(s):
    """Overflow-safe, strictly monotone sigmoid (AUC-preserving)."""
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    pos = s >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-s[pos]))
    e = np.exp(s[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class _DiffMeanClf:
    """Difference-of-means probe with an LR-compatible interface.

    Direction w = mean(X[y==1]) - mean(X[y==0]); score = (X - mu) @ w, where mu
    is the training mean (centering shifts the threshold, not the ranking).
    `predict_proba` returns a stable sigmoid of the score so the object plugs
    straight into the same `clf.predict_proba(...)[:, 1]` scoring loop as
    LogisticRegression. AUC is invariant to the monotone sigmoid, so calibration
    is irrelevant — this is purely a direction estimator.
    """

    def __init__(self, w, mu):
        self.coef_ = w.reshape(1, -1)
        self._w = w
        self._mu = mu

    def decision_function(self, X):
        return (np.asarray(X) - self._mu) @ self._w

    def predict_proba(self, X):
        p = _stable_sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])


def _default_lr_fit(X, y, C=PROBE_C, seed=0, cw=None, max_iter=MAX_ITER):
    """Exact reproduction of the original inline LR fit (the fit_fn default)."""
    return LogisticRegression(max_iter=max_iter, C=C, random_state=seed,
                              class_weight=cw).fit(X, y)


def fit_diffmean(X, y, C=None, seed=0, cw=None, max_iter=None):
    """Difference-of-class-means direction (estimator-robustness twin of LR)."""
    X = np.asarray(X)
    y = np.asarray(y)
    mu = X.mean(axis=0)
    w = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    return _DiffMeanClf(w, mu)


def fit_lr_shuffled(X, y, C=PROBE_C, seed=0, cw=None, max_iter=MAX_ITER):
    """Random-label control: LR fit on per-seed shuffled labels (≈0.5 floor)."""
    rng = np.random.RandomState(seed)
    y_shuf = rng.permutation(np.asarray(y))
    return LogisticRegression(max_iter=max_iter, C=C, random_state=seed,
                              class_weight=cw).fit(X, y_shuf)


def cross_transfer_matrix(probe_names, probe_data, n_seeds=N_SEEDS,
                          max_iter=MAX_ITER, tok_train_data=None,
                          multi_train_data=None, fit_fn=None):
    """Compute n x n cross-transfer AUC matrix.

    Cell (i, j) = train on i, test on j at i's best layer.

    *tok_train_data*: optional dict mapping probe name to
    (tok_feats_h, tok_feats_d, counts_h, counts_d) for token-level probes.
    When a row's probe name is in tok_train_data, the training probe is
    fit via the token-level procedure (StandardScaler + LR with
    fit_intercept=False), and test data is evaluated through the scaler.

    *multi_train_data*: optional dict mapping probe name to
    (tok_feats_h, tok_feats_d, counts_h, counts_d, layers) for multi-layer
    probes.  When a row's probe name is in multi_train_data, the probe is
    trained via fit_multi_tok_probe and test data is concatenated across
    the same layers before evaluation.

    For the standard (non-token, non-multi) branch, each seed trains on a
    different group-aware subsample of probe i (GroupShuffleSplit on
    scenario_id when groups are present, else a random split), so the seed
    spread reflects genuine sampling variance rather than the inert
    random_state of the lbfgs solver. The diagonal (i, i) is scored on the
    held-out split only, making it true self-transfer instead of train-set AUC.
    Off-diagonal cells score all of j (a distinct dataset, no leakage).

    *fit_fn*: optional estimator for the STANDARD branch only, signature
    fit_fn(X, y, C=, seed=, cw=, max_iter=) -> object with predict_proba. The
    default (None -> _default_lr_fit) reproduces the original inline LR fit
    byte-for-byte, so existing results are unchanged. Pass fit_diffmean for the
    difference-of-means robustness twin or fit_lr_shuffled for the random-label
    control — the per-seed subsample / diagonal / scoring machinery is identical;
    only the estimator changes. (tok/multi rows ignore fit_fn — they keep their
    own token-level fits.)

    Returns:
        (auc_matrix, preds_dict)
        auc_matrix: np.ndarray (n, n, n_seeds)
        preds_dict: dict mapping (i, j) -> (y_true, y_score) arrays,
            stored from the last seed. Use with _bootstrap_auc() for CIs.
            (Cross-seed spread is captured by auc_matrix's seed axis.)
    """
    n = len(probe_names)
    auc_matrix = np.zeros((n, n, n_seeds))
    preds_dict = {}
    if tok_train_data is None:
        tok_train_data = {}
    if multi_train_data is None:
        multi_train_data = {}
    _fit = fit_fn if fit_fn is not None else _default_lr_fit

    for i, name_i in enumerate(probe_names):
        fi, li, mi, gi, bl_i, Ci, cwi = probe_data[name_i]

        is_tok_train = name_i in tok_train_data
        is_multi_train = name_i in multi_train_data

        for seed in range(n_seeds):
            # diag_ho_idx: indices into probe i's masked rows that are held out
            # of training, used to score the diagonal (i==i) without train-set
            # leakage. n_te_h (tok/multi only) is the count of held-out HONEST
            # prompts, used by the diagonal layout assert below.
            diag_ho_idx = None
            n_te_h = None
            if is_multi_train:
                # Multi-layer training on a per-seed prompt subsample, so the
                # row carries real seed variance and the diagonal is held out.
                tfh, tfd, ch, cd, layers_i = multi_train_data[name_i]
                tfh_s, tfd_s, ch_s, cd_s, diag_ho_idx, n_te_h = _subsample_tok_for_train(
                    tfh, tfd, ch, cd, layers_i, seed)
                clf, scaler, _ = fit_multi_tok_probe(
                    tfh_s, tfd_s, ch_s, cd_s, layers=layers_i, C=Ci, seed=seed,
                    max_iter=max_iter)
            elif is_tok_train:
                # Token-level training on a per-seed prompt subsample (same
                # rationale as multi); diagonal scored on the held-out prompts.
                tfh, tfd, ch, cd = tok_train_data[name_i]
                tfh_s, tfd_s, ch_s, cd_s, diag_ho_idx, n_te_h = _subsample_tok_for_train(
                    tfh, tfd, ch, cd, [bl_i], seed)
                clf, scaler = fit_tok_probe(tfh_s, tfd_s, ch_s, cd_s,
                                            layer=bl_i, C=Ci, seed=seed,
                                            max_iter=max_iter)
            else:
                # Standard training. Each seed sees a different subsample of the
                # training data so the seed spread reflects genuine sampling
                # variance (with the default lbfgs solver, random_state alone is
                # inert). Split is group-aware on scenario_id when groups exist,
                # mirroring subset_layer_aucs. The held-out rows score the
                # diagonal so (i,i) is real self-transfer, not train-set AUC.
                Xi = fi[bl_i][mi]; yi = li[mi]
                n_i = len(yi)
                gi_masked = gi[mi] if gi is not None else None
                if gi_masked is not None:
                    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                            random_state=seed)
                    tr_idx, diag_ho_idx = next(
                        gss.split(np.arange(n_i), yi, gi_masked))
                else:
                    rng = np.random.RandomState(seed)
                    perm = rng.permutation(n_i)
                    cut = int(n_i * (1 - TEST_SIZE))
                    tr_idx, diag_ho_idx = perm[:cut], perm[cut:]
                # Guard against a degenerate split (single-class train/holdout).
                if (len(np.unique(yi[tr_idx])) < 2 or
                        len(np.unique(yi[diag_ho_idx])) < 2):
                    auc_matrix[i, :, seed] = np.nan
                    continue
                clf = _fit(Xi[tr_idx], yi[tr_idx], C=Ci, seed=seed, cw=cwi,
                           max_iter=max_iter)
                scaler = None

            for j, name_j in enumerate(probe_names):
                fj, lj, mj, gj, bl_j, Cj, cwj = probe_data[name_j]

                if is_multi_train:
                    # Multi-layer: concatenate test features across layers
                    layers_i = multi_train_data[name_i][4]
                    # Check all required layers exist in test features
                    if any(li_k not in fj for li_k in layers_i):
                        auc_matrix[i, j, seed] = np.nan
                        continue
                    Xj_concat = np.concatenate(
                        [fj[li_k][mj] for li_k in layers_i], axis=1)
                    yj = np.asarray(lj[mj])
                    # Diagonal: score only the held-out prompts (self-transfer,
                    # not train-set AUC). Off-diagonal scores all of j.
                    if i == j and diag_ho_idx is not None:
                        # diag_ho_idx indexes the mean-pooled feats_Z_mean layout
                        # [honest prompts…, deceptive prompts…]; valid only when
                        # mj is all-True and that block ordering holds. Assert it
                        # so a future masked/interleaved assembly fails loudly
                        # instead of silently scoring the wrong rows.
                        assert mj.all(), \
                            "diag_ho_idx layout assumes unmasked Z_mean test features"
                        _yh = np.asarray(lj[mj])[diag_ho_idx]
                        assert (_yh[:n_te_h] == 0).all() and (_yh[n_te_h:] == 1).all(), \
                            ("diag_ho_idx label composition changed — Z_mean row "
                             "layout no longer [honest…, deceptive…]")
                        Xj_concat = Xj_concat[diag_ho_idx]
                        yj = yj[diag_ho_idx]
                    if len(np.unique(yj)) < 2:
                        auc_matrix[i, j, seed] = np.nan
                        continue
                    Xj_in = scaler.transform(Xj_concat)
                else:
                    if bl_i not in fj:
                        auc_matrix[i, j, seed] = np.nan
                        continue
                    Xj = fj[bl_i][mj]; yj = np.asarray(lj[mj])
                    # Diagonal: score only the held-out split so (i,i) measures
                    # self-transfer, not the train-set AUC. Off-diagonal cells
                    # are a different dataset, so score all of j.
                    # NOTE: bl_i was selected by a layer sweep that saw all of i's
                    # data, so this held-out diagonal is still mildly optimistic
                    # (layer selection peeked at the holdout). Nested per-seed
                    # layer selection would close that gap; the diagonal isn't a
                    # load-bearing number, so we leave it.
                    if i == j and diag_ho_idx is not None:
                        # For a tok/multi row scored through the single-layer
                        # test path, diag_ho_idx indexes the mean-pooled
                        # feats_Z_mean layout — assert the same blocked
                        # [honest…, deceptive…] / unmasked invariant.
                        if is_tok_train or is_multi_train:
                            assert mj.all(), \
                                "diag_ho_idx layout assumes unmasked Z_mean test features"
                            _yh = np.asarray(lj[mj])[diag_ho_idx]
                            assert (_yh[:n_te_h] == 0).all() and (_yh[n_te_h:] == 1).all(), \
                                ("diag_ho_idx label composition changed — Z_mean row "
                                 "layout no longer [honest…, deceptive…]")
                        Xj = Xj[diag_ho_idx]; yj = yj[diag_ho_idx]
                    if len(np.unique(yj)) < 2:
                        auc_matrix[i, j, seed] = np.nan
                        continue

                    if scaler is not None:
                        Xj_in = scaler.transform(Xj)
                    else:
                        Xj_in = Xj

                probs = clf.predict_proba(Xj_in)[:, 1]
                auc_matrix[i, j, seed] = roc_auc_score(yj, probs)
                preds_dict[(i, j)] = (np.asarray(yj), np.asarray(probs))

    return auc_matrix, preds_dict


def cross_transfer_matrix_oracle_layer(probe_names, probe_data, n_seeds=N_SEEDS,
                                       max_iter=MAX_ITER, tok_train_data=None,
                                       multi_train_data=None):
    """Oracle-ceiling cross-transfer: per (i, j) pick the test-OPTIMAL layer of j.

    This is NOT a transfer estimate — it peeks at the test labels to choose the
    best-scoring layer, so it is an UPPER BOUND on what probe i's direction
    could achieve on j if the right layer were known a priori. Report it only as
    a ceiling alongside the principled (bl_i) cross_transfer_matrix.

    Its value as a diagnostic: if a critical cell (e.g. Z -> P) fails even at the
    oracle layer, the failure cannot be blamed on a train/test layer mismatch —
    it closes off that alternative explanation.

    Training mirrors cross_transfer_matrix: each seed trains on a group-aware
    subsample (standard rows) or a prompt subsample (token/multi rows), so the
    rows carry real seed variance, and the diagonal (i, i) is scored on the
    held-out split — even though the diagonal is itself an oracle-layer ceiling.

    For multi-layer probes: no layer sweep (uses the same multi-layer concat).

    Returns (auc_matrix, best_layer_matrix).
    """
    from scipy import stats as _stats

    n = len(probe_names)
    auc_matrix = np.zeros((n, n, n_seeds))
    # Track per-seed best layers; final matrix stores mode across seeds
    best_layer_per_seed = np.zeros((n, n, n_seeds), dtype=int)
    if tok_train_data is None:
        tok_train_data = {}
    if multi_train_data is None:
        multi_train_data = {}

    for i, name_i in enumerate(probe_names):
        fi, li, mi, gi, bl_i, Ci, cwi = probe_data[name_i]

        is_tok_train = name_i in tok_train_data
        is_multi_train = name_i in multi_train_data

        for seed in range(n_seeds):
            diag_ho_idx = None
            n_te_h = None
            if is_multi_train:
                tfh, tfd, ch, cd, layers_i = multi_train_data[name_i]
                tfh_s, tfd_s, ch_s, cd_s, diag_ho_idx, n_te_h = _subsample_tok_for_train(
                    tfh, tfd, ch, cd, layers_i, seed)
                clf, scaler, _ = fit_multi_tok_probe(
                    tfh_s, tfd_s, ch_s, cd_s, layers=layers_i, C=Ci, seed=seed,
                    max_iter=max_iter)
            elif is_tok_train:
                tfh, tfd, ch, cd = tok_train_data[name_i]
                tfh_s, tfd_s, ch_s, cd_s, diag_ho_idx, n_te_h = _subsample_tok_for_train(
                    tfh, tfd, ch, cd, [bl_i], seed)
                clf, scaler = fit_tok_probe(tfh_s, tfd_s, ch_s, cd_s,
                                            layer=bl_i, C=Ci, seed=seed,
                                            max_iter=max_iter)
            else:
                # Standard branch: per-seed group-aware subsample (see
                # cross_transfer_matrix for the rationale).
                Xi = fi[bl_i][mi]; yi = li[mi]
                n_i = len(yi)
                gi_masked = gi[mi] if gi is not None else None
                if gi_masked is not None:
                    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                            random_state=seed)
                    tr_idx, diag_ho_idx = next(
                        gss.split(np.arange(n_i), yi, gi_masked))
                else:
                    rng = np.random.RandomState(seed)
                    perm = rng.permutation(n_i)
                    cut = int(n_i * (1 - TEST_SIZE))
                    tr_idx, diag_ho_idx = perm[:cut], perm[cut:]
                if (len(np.unique(yi[tr_idx])) < 2 or
                        len(np.unique(yi[diag_ho_idx])) < 2):
                    auc_matrix[i, :, seed] = np.nan
                    best_layer_per_seed[i, :, seed] = -1
                    continue
                clf = LogisticRegression(max_iter=max_iter, C=Ci,
                                         random_state=seed,
                                         class_weight=cwi)
                clf.fit(Xi[tr_idx], yi[tr_idx])
                scaler = None

            for j, name_j in enumerate(probe_names):
                fj, lj, mj, gj, bl_j, Cj, cwj = probe_data[name_j]
                # On the diagonal, restrict the eval to i's held-out rows.
                ho = diag_ho_idx if (i == j and diag_ho_idx is not None) else None

                # For a tok/multi diagonal, ho indexes the mean-pooled
                # feats_Z_mean layout [honest…, deceptive…]; assert that blocked,
                # unmasked invariant so a changed assembly fails loudly.
                if ho is not None and (is_tok_train or is_multi_train):
                    assert mj.all(), \
                        "diag_ho_idx layout assumes unmasked Z_mean test features"
                    _yh = np.asarray(lj[mj])[ho]
                    assert (_yh[:n_te_h] == 0).all() and (_yh[n_te_h:] == 1).all(), \
                        ("diag_ho_idx label composition changed — Z_mean row "
                         "layout no longer [honest…, deceptive…]")

                if is_multi_train:
                    # Multi-layer: no sweep — use same layer concat
                    layers_i = multi_train_data[name_i][4]
                    if any(li_k not in fj for li_k in layers_i):
                        auc_matrix[i, j, seed] = np.nan
                        best_layer_per_seed[i, j, seed] = -1
                        continue
                    Xj_concat = np.concatenate(
                        [fj[li_k][mj] for li_k in layers_i], axis=1)
                    yj = np.asarray(lj[mj])
                    if ho is not None:
                        Xj_concat = Xj_concat[ho]; yj = yj[ho]
                    if len(np.unique(yj)) < 2:
                        auc_matrix[i, j, seed] = np.nan
                        best_layer_per_seed[i, j, seed] = -1
                        continue
                    Xj_in = scaler.transform(Xj_concat)
                    probs = clf.predict_proba(Xj_in)[:, 1]
                    auc_matrix[i, j, seed] = roc_auc_score(yj, probs)
                    best_layer_per_seed[i, j, seed] = layers_i[0]
                    continue

                # Sweep all layers in fj
                yj = np.asarray(lj[mj])
                if ho is not None:
                    yj = yj[ho]
                if len(np.unique(yj)) < 2:
                    auc_matrix[i, j, seed] = np.nan
                    best_layer_per_seed[i, j, seed] = -1
                    continue

                best_auc = -1.0
                best_l = bl_i  # fallback
                for test_layer in sorted(fj.keys()):
                    Xj = fj[test_layer][mj]
                    if ho is not None:
                        Xj = Xj[ho]
                    if scaler is not None:
                        # Check dimension compatibility
                        if Xj.shape[1] != scaler.n_features_in_:
                            continue
                        Xj_in = scaler.transform(Xj)
                    else:
                        # Check dimension compatibility with clf
                        if Xj.shape[1] != clf.coef_.shape[1]:
                            continue
                        Xj_in = Xj

                    probs = clf.predict_proba(Xj_in)[:, 1]
                    auc_val = roc_auc_score(yj, probs)
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_l = test_layer

                auc_matrix[i, j, seed] = best_auc
                best_layer_per_seed[i, j, seed] = best_l

    # Compute mode of best layer across seeds
    best_layer_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            layers_ij = best_layer_per_seed[i, j, :]
            valid = layers_ij[layers_ij >= 0]
            if len(valid) > 0:
                best_layer_matrix[i, j] = int(_stats.mode(valid, keepdims=False).mode)
            else:
                best_layer_matrix[i, j] = -1

    return auc_matrix, best_layer_matrix


def group_disjoint_transfer(name_i, name_j, probe_data, n_seeds=N_SEEDS,
                            max_iter=MAX_ITER):
    """Leakage-free cross-transfer AUROC for an ordered pair i -> j.

    For pairs whose datasets share a scenario_id namespace (e.g. I'_S and P_S,
    both built from df_sales), the plain off-diagonal cell of
    cross_transfer_matrix scores ALL of j — including scenarios that were in i's
    train split. Surface scenario features can then ride along and inflate the
    transfer estimate.

    This restricts each seed's evaluation to the rows of j whose scenario_id is
    ABSENT from i's train split. Use it to check whether a high transfer cell
    (e.g. P -> I') is real or partly scenario memorization: if it survives
    group-disjoint eval, the transfer is real.

    Only meaningful when i and j share a scenario_id namespace; both probes must
    carry group arrays. Evaluation uses i's best layer (bl_i), matching the
    principled cross_transfer_matrix.

    Returns:
        aucs: np.ndarray (n_seeds,) of AUROCs (np.nan for a degenerate seed)
        mean_kept: float, mean fraction of j evaluated per seed (coverage)
    """
    fi, li, mi, gi, bl_i, Ci, cwi = probe_data[name_i]
    fj, lj, mj, gj, bl_j, Cj, cwj = probe_data[name_j]
    if gi is None or gj is None:
        raise ValueError(
            f"group_disjoint_transfer needs group arrays for both "
            f"{name_i} and {name_j} (one is None)")

    Xi = fi[bl_i][mi]; yi = li[mi]
    gi_masked = gi[mi]
    n_i = len(yi)

    Xj = fj[bl_i][mj]; yj = np.asarray(lj[mj])
    gj_masked = gj[mj]

    aucs = []
    kept_fracs = []
    for seed in range(n_seeds):
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=seed)
        tr_idx, _ho = next(gss.split(np.arange(n_i), yi, gi_masked))
        if len(np.unique(yi[tr_idx])) < 2:
            aucs.append(np.nan)
            continue
        train_scenarios = set(np.asarray(gi_masked)[tr_idx].tolist())

        clf = LogisticRegression(max_iter=max_iter, C=Ci, random_state=seed,
                                 class_weight=cwi)
        clf.fit(Xi[tr_idx], yi[tr_idx])

        # Keep only j rows whose scenario is absent from i's train split.
        keep = np.array([g not in train_scenarios for g in gj_masked])
        kept_fracs.append(float(keep.mean()))
        if keep.sum() == 0 or len(np.unique(yj[keep])) < 2:
            aucs.append(np.nan)
            continue
        probs = clf.predict_proba(Xj[keep])[:, 1]
        aucs.append(float(roc_auc_score(yj[keep], probs)))

    mean_kept = float(np.mean(kept_fracs)) if kept_fracs else 0.0
    return np.array(aucs), mean_kept


def bootstrap_ci_matrix(preds_dict, n, n_boot=1000, seed=42):
    """Compute bootstrap 95% CI for each cell from a preds_dict.

    Args:
        preds_dict: dict (i, j) -> (y_true, y_score) from cross_transfer_matrix
        n: number of probes (matrix is n x n)
        n_boot: number of bootstrap resamples
        seed: random seed for reproducibility

    Returns:
        ci_matrix: np.ndarray (n, n, 2) where [:,:,0] = lower, [:,:,1] = upper
    """
    ci_matrix = np.full((n, n, 2), np.nan)
    for (i, j), (y_true, y_score) in preds_dict.items():
        if len(np.unique(y_true)) < 2:
            continue
        lo, hi = _bootstrap_auc(y_true, y_score, n_boot=n_boot, seed=seed)
        ci_matrix[i, j, 0] = lo
        ci_matrix[i, j, 1] = hi
    return ci_matrix


def print_transfer_matrix(transfer_array, probe_names, title='Cross-transfer AUC',
                           ci_matrix=None):
    """Pretty-print a cross-transfer AUC matrix (n x n x n_seeds).

    If *ci_matrix* (n x n x 2) is provided, display bootstrap 95% CIs
    instead of ±std.
    """
    mean = np.nanmean(transfer_array, axis=2)
    std = np.nanstd(transfer_array, axis=2)
    n = len(probe_names)

    if ci_matrix is not None:
        print(f'\n{title} [95% bootstrap CI]:')
        header = '           ' + '  '.join(f'{name:>16s}' for name in probe_names)
        print(header)
        for i, name_i in enumerate(probe_names):
            row = f'{name_i:>8s}   '
            for j in range(n):
                lo = ci_matrix[i, j, 0]
                hi = ci_matrix[i, j, 1]
                if np.isnan(lo):
                    row += f'{"nan":>16s}  '
                else:
                    row += f'{mean[i,j]:.3f} [{lo:.2f},{hi:.2f}]  '
            print(row)
    else:
        print(f'\n{title} (mean +/- std):')
        header = '           ' + '  '.join(f'{name:>8s}' for name in probe_names)
        print(header)
        for i, name_i in enumerate(probe_names):
            row = f'{name_i:>8s}   '
            for j in range(n):
                row += f'{mean[i,j]:.3f}±{std[i,j]:.2f}  '
            print(row)
    return mean, std


# ── Token-level probe functions (Z_tok, Apollo methodology) ───────────────────

def _tok_prompt_split(counts_h, counts_d, seed, test_size=TEST_SIZE):
    """Split prompts into train/test, return token-level indices.

    Returns:
        tr_tok_h, tr_tok_d: token indices for training honest/deceptive
        te_tok_h, te_tok_d: token indices for test honest/deceptive
        te_counts_h, te_counts_d: token counts per test prompt
        tr_prompts_h, tr_prompts_d: prompt indices for training honest/deceptive
        te_prompts_h, te_prompts_d: prompt indices for test honest/deceptive

    The prompt-index arrays index into counts_h/counts_d (and, equivalently,
    into the per-prompt rows of the mean-pooled feats_Z_mean array, where honest
    prompt p -> row p and deceptive prompt q -> row len(counts_h) + q).
    """
    n_h = len(counts_h)
    n_d = len(counts_d)

    rng = np.random.RandomState(seed)

    # Split honest prompts
    perm_h = rng.permutation(n_h)
    split_h = int(n_h * (1 - test_size))
    tr_prompts_h, te_prompts_h = perm_h[:split_h], perm_h[split_h:]

    # Split deceptive prompts
    perm_d = rng.permutation(n_d)
    split_d = int(n_d * (1 - test_size))
    tr_prompts_d, te_prompts_d = perm_d[:split_d], perm_d[split_d:]

    # Convert prompt indices to token indices via cumsum
    cs_h = np.concatenate([[0], np.cumsum(counts_h)])
    cs_d = np.concatenate([[0], np.cumsum(counts_d)])

    def prompts_to_tok_idx(prompts, cs):
        idx = []
        for p in prompts:
            idx.extend(range(cs[p], cs[p + 1]))
        return np.array(idx, dtype=np.int64)

    tr_tok_h = prompts_to_tok_idx(tr_prompts_h, cs_h)
    tr_tok_d = prompts_to_tok_idx(tr_prompts_d, cs_d)
    te_tok_h = prompts_to_tok_idx(te_prompts_h, cs_h)
    te_tok_d = prompts_to_tok_idx(te_prompts_d, cs_d)

    te_counts_h = counts_h[te_prompts_h]
    te_counts_d = counts_d[te_prompts_d]

    return (tr_tok_h, tr_tok_d, te_tok_h, te_tok_d, te_counts_h, te_counts_d,
            tr_prompts_h, tr_prompts_d, te_prompts_h, te_prompts_d)


def fit_tok_probe(tok_feats_h, tok_feats_d, counts_h, counts_d,
                  layer, C=ZOU_PROBE_C, seed=0, max_iter=MAX_ITER):
    """Fit a token-level probe at one layer on all data.

    Following Apollo methodology:
    - StandardScaler on features
    - LogisticRegression with fit_intercept=False

    Returns (clf, scaler).
    """
    Xh = tok_feats_h[layer]
    Xd = tok_feats_d[layer]
    X = np.concatenate([Xh, Xd], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(Xh.shape[0]), np.ones(Xd.shape[0])])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed,
                              fit_intercept=False)
    clf.fit(X_scaled, y)
    return clf, scaler


def _tok_eval_prompt_level(clf, scaler, X_h_tok, X_d_tok,
                           counts_h, counts_d):
    """Evaluate a token-level probe at prompt level.

    For each test prompt, compute mean(decision_function(tokens)) -> one score.
    Then AUROC on prompt-level scores vs prompt-level labels.
    """
    # Score all tokens
    all_tok = np.concatenate([X_h_tok, X_d_tok], axis=0).astype(np.float32)
    all_scores = clf.decision_function(scaler.transform(all_tok))

    # Aggregate to prompt level
    prompt_scores = []
    prompt_labels = []

    # Honest prompts
    cs = np.concatenate([[0], np.cumsum(counts_h)])
    for i in range(len(counts_h)):
        start, end = cs[i], cs[i + 1]
        if end > start:
            prompt_scores.append(all_scores[start:end].mean())
            prompt_labels.append(0)

    # Deceptive prompts (offset by total honest tokens)
    offset = X_h_tok.shape[0]
    cs_d = np.concatenate([[0], np.cumsum(counts_d)])
    for i in range(len(counts_d)):
        start, end = offset + cs_d[i], offset + cs_d[i + 1]
        if end > start:
            prompt_scores.append(all_scores[start:end].mean())
            prompt_labels.append(1)

    prompt_scores = np.array(prompt_scores)
    prompt_labels = np.array(prompt_labels)

    if len(np.unique(prompt_labels)) < 2:
        return 0.5
    return float(roc_auc_score(prompt_labels, prompt_scores))


def multiseed_best_layer_tok(tok_feats_h, tok_feats_d, counts_h, counts_d,
                              n_seeds=N_SEEDS, C=ZOU_PROBE_C,
                              test_size=TEST_SIZE, max_iter=MAX_ITER):
    """Token-level probe layer sweep.

    Splits at PROMPT level (no data leakage). Trains on individual tokens,
    evaluates by mean-pooling per-token scores back to prompt-level, then
    computes AUROC on prompt-level scores.
    """
    n_layers = min(len(tok_feats_h), len(tok_feats_d))

    per_seed_aucs = []
    for seed in range(n_seeds):
        (tr_tok_h, tr_tok_d, te_tok_h, te_tok_d,
         te_counts_h, te_counts_d, *_) = _tok_prompt_split(
            counts_h, counts_d, seed, test_size)

        seed_aucs = []
        for li in range(n_layers):
            if li not in tok_feats_h or li not in tok_feats_d:
                seed_aucs.append(0.5)
                continue

            # Training data: all tokens from train prompts
            Xh_tr = tok_feats_h[li][tr_tok_h].astype(np.float32)
            Xd_tr = tok_feats_d[li][tr_tok_d].astype(np.float32)
            X_tr = np.concatenate([Xh_tr, Xd_tr], axis=0)
            y_tr = np.concatenate([np.zeros(Xh_tr.shape[0]),
                                   np.ones(Xd_tr.shape[0])])

            if len(np.unique(y_tr)) < 2:
                seed_aucs.append(0.5)
                continue

            # Fit with StandardScaler + no-intercept (Apollo methodology)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            clf = LogisticRegression(max_iter=max_iter, C=C,
                                      random_state=seed,
                                      fit_intercept=False)
            clf.fit(X_tr_scaled, y_tr)

            # Evaluate at prompt level
            Xh_te = tok_feats_h[li][te_tok_h]
            Xd_te = tok_feats_d[li][te_tok_d]
            auc = _tok_eval_prompt_level(clf, scaler, Xh_te, Xd_te,
                                         te_counts_h, te_counts_d)
            seed_aucs.append(auc)

        per_seed_aucs.append(seed_aucs)

    arr = np.array(per_seed_aucs)  # (n_seeds, n_layers)
    mean_ho = arr.mean(axis=0)
    best = int(np.argmax(mean_ho))
    print(f'  Best layer: {best}  (mean holdout AUC = {mean_ho[best]:.3f}, '
          f'std = {arr[:, best].std():.3f}, {arr.shape[0]} seeds)')
    return best, arr


def all_layer_directions_tok(tok_feats_h, tok_feats_d, counts_h, counts_d,
                              C=ZOU_PROBE_C, n_layers=None,
                              max_iter=MAX_ITER):
    """Get direction vector at each layer via token-level fitting."""
    if n_layers is None:
        n_layers = min(len(tok_feats_h), len(tok_feats_d))
    dirs = {}
    for li in range(n_layers):
        if li not in tok_feats_h or li not in tok_feats_d:
            continue
        clf, scaler = fit_tok_probe(tok_feats_h, tok_feats_d, counts_h, counts_d,
                                     layer=li, C=C, seed=0, max_iter=max_iter)
        dirs[li] = probe_direction(clf, scaler=scaler)
    return dirs


# ── Multi-layer probe functions (Z_multi, Apollo full methodology) ────────────

def multi_layer_range(n_layers):
    """Return layer indices for the middle 50% of the model (Apollo convention).

    For a model with 32 layers this returns [8, 9, ..., 23] (16 layers).
    """
    start = n_layers // 4
    end = start + n_layers // 2
    return list(range(start, end))


def _concat_multi_layer_tok(tok_feats, layers):
    """Concatenate token features across multiple layers horizontally.

    Args:
        tok_feats: dict {layer_idx: np.ndarray(n_tokens, hidden_dim)}
        layers: list of layer indices to concatenate

    Returns:
        np.ndarray(n_tokens, len(layers) * hidden_dim)
    """
    return np.concatenate([tok_feats[li] for li in layers], axis=1)


def fit_multi_tok_probe(tok_feats_h, tok_feats_d, counts_h, counts_d,
                         layers, C=ZOU_PROBE_C, seed=0, max_iter=MAX_ITER):
    """Fit a multi-layer token-level probe (Apollo full methodology).

    Concatenates features across *layers*, then fits StandardScaler +
    LogisticRegression(fit_intercept=False).

    Returns (clf, scaler, layers).
    """
    Xh = _concat_multi_layer_tok(tok_feats_h, layers)
    Xd = _concat_multi_layer_tok(tok_feats_d, layers)
    X = np.concatenate([Xh, Xd], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(Xh.shape[0]), np.ones(Xd.shape[0])])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=seed,
                              fit_intercept=False)
    clf.fit(X_scaled, y)
    return clf, scaler, layers


def multi_probe_directions(clf, scaler, layers, hidden_dim):
    """Extract per-layer direction vectors from a multi-layer probe.

    1. Denormalize the full coefficient vector through the scaler.
    2. Reshape to (n_layers, hidden_dim).
    3. Unit-normalize each layer's sub-vector independently.

    Returns {layer_idx: unit_direction} — directly comparable with
    P/I/Z_tok per-layer directions.
    """
    w = clf.coef_[0].copy()
    safe_scale = scaler.scale_.copy()
    safe_scale[safe_scale == 0] = 1.0
    w = w / safe_scale

    w_per_layer = w.reshape(len(layers), hidden_dim)
    dirs = {}
    for idx, li in enumerate(layers):
        v = w_per_layer[idx].copy()
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        dirs[li] = v
    return dirs


def multiseed_multi_tok_auc(tok_feats_h, tok_feats_d, counts_h, counts_d,
                             layers, n_seeds=N_SEEDS, C=ZOU_PROBE_C,
                             test_size=TEST_SIZE, max_iter=MAX_ITER):
    """Multi-layer token probe holdout evaluation across seeds.

    Splits at PROMPT level via _tok_prompt_split (no data leakage).
    Concatenates features across *layers*, trains StandardScaler + LR,
    evaluates via _tok_eval_prompt_level (already dimensionality-agnostic).

    Returns np.ndarray of shape (n_seeds,) with prompt-level AUCs.
    """
    aucs = []
    for seed in range(n_seeds):
        (tr_tok_h, tr_tok_d, te_tok_h, te_tok_d,
         te_counts_h, te_counts_d, *_) = _tok_prompt_split(
            counts_h, counts_d, seed, test_size)

        # Training data: concatenate across layers for train tokens
        Xh_tr = _concat_multi_layer_tok(
            {li: tok_feats_h[li][tr_tok_h] for li in layers}, layers
        ).astype(np.float32)
        Xd_tr = _concat_multi_layer_tok(
            {li: tok_feats_d[li][tr_tok_d] for li in layers}, layers
        ).astype(np.float32)
        X_tr = np.concatenate([Xh_tr, Xd_tr], axis=0)
        y_tr = np.concatenate([np.zeros(Xh_tr.shape[0]),
                                np.ones(Xd_tr.shape[0])])

        if len(np.unique(y_tr)) < 2:
            aucs.append(0.5)
            continue

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        clf = LogisticRegression(max_iter=max_iter, C=C,
                                  random_state=seed, fit_intercept=False)
        clf.fit(X_tr_scaled, y_tr)

        # Test data: concatenate across layers for test tokens
        Xh_te = _concat_multi_layer_tok(
            {li: tok_feats_h[li][te_tok_h] for li in layers}, layers
        )
        Xd_te = _concat_multi_layer_tok(
            {li: tok_feats_d[li][te_tok_d] for li in layers}, layers
        )
        auc = _tok_eval_prompt_level(clf, scaler, Xh_te, Xd_te,
                                      te_counts_h, te_counts_d)
        aucs.append(auc)

    aucs = np.array(aucs)
    print(f'  Z_multi holdout AUC = {aucs.mean():.3f} ± {aucs.std():.3f} '
          f'({len(aucs)} seeds, layers {layers[0]}-{layers[-1]})')
    return aucs
