#!/usr/bin/env python
"""
Phase B + B5: robustness battery for the Liars'-Bench cells, esp. the star
P_C -> HP-KR = 0.857. CPU-only; uses cached LB acts + lb_meta_*.json (text/model)
+ probe_state.pkl + cached P/I resp acts + inference_P_bundle.pkl.

Checks per key cell: LR (seeds+bootstrap CI) | DiffMean twin | TF-IDF surface
(within + cross) | Llama-3.3-70B-only generator subset | balAcc@1%FPR.
Plus B5: pressure-level ablation + cross-position (pre- vs resp-trained P_C).

CAVEAT (off-policy): 8B reading 70B-class transcripts; HP-KR ground truth = the
generator's private belief. HP-KR source = WMDP (bio/cyber/chem) -> watch TF-IDF.
Env: OUTPUT_DIR (default ~/output).
"""
import os, json, pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from probe_utils import multiseed_best_layer, fit_diffmean, _bootstrap_auc
import ood_eval_utils as ood

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, TEST_SIZE, MAX_ITER = 32, 10, 1.0, 0.2, 2000
PS = pickle.load(open(op / 'probe_state.pkl', 'rb')); g = globals(); g.update(PS)
out = {'caveat': 'off-policy (8B reads 70B-class transcripts); HP-KR=WMDP; balAcc@1% is LB native metric'}

def load(tag, pos=None):
    suf = f'_{pos}' if pos else ''
    return {li: np.load(op / f'acts_{tag}{suf}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}{suf}_layer_{li:03d}.npy').exists()}

# ── LB cached acts (resp + pre) + labels + meta (text/model) ─────────────────
lb_feats = {k: load(f'OOD_LB_{k}') for k in ['ID', 'HP_KR']}
lb_pre   = {k: load(f'OOD_LB_{k}', 'pre') for k in ['ID', 'HP_KR']}
lb_lab   = {k: np.load(op / f'labels_OOD_LB_{k}.npy') for k in ['ID', 'HP_KR']}
lb_meta  = {k: json.load(open(op / f'lb_meta_{k}.json')) for k in ['ID', 'HP_KR']}
lb_model = {k: np.array([m['model'] for m in lb_meta[k]]) for k in lb_meta}
lb_text  = {k: [m['text'] for m in lb_meta[k]] for k in lb_meta}
for k in lb_lab:
    assert len(lb_lab[k]) == len(lb_meta[k]), f'{k}: acts/meta length mismatch'

# ── training-side resp probes (retrain P/I at response token) + POOLED ───────
fr = {'P_S': load('P_S', 'resp'), 'P_C': load('P_C', 'resp'),
      'I_S': load('I_S', 'resp'), 'I_C': load('I_C', 'resp')}
Y = {'P_S': labels_P_S_lied, 'P_C': labels_P_C_lied, 'I_S': labels_I_S, 'I_C': labels_I_C}
M = {'P_S': mask_P_S, 'P_C': mask_P_C, 'I_S': mask_I_S, 'I_C': mask_I_C}
G = {'P_S': groups_P_S, 'P_C': groups_P_C, 'I_S': groups_I_S, 'I_C': groups_I_C}
BL = {}
for t in fr:
    BL[t], _ = multiseed_best_layer(fr[t], Y[t], M[t], n_seeds=N_SEEDS, C=PROBE_C, groups=G[t])
pl = sorted(set(fr['P_S']) & set(fr['P_C']))
fr['POOLED'] = {li: np.concatenate([fr['P_S'][li][mask_P_S], fr['P_C'][li][mask_P_C]], 0) for li in pl}
Y['POOLED'] = np.concatenate([labels_P_S_lied[mask_P_S], labels_P_C_lied[mask_P_C]])
G['POOLED'] = np.concatenate([np.array([f'S:{x}' for x in groups_P_S[mask_P_S]]),
                              np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])])
M['POOLED'] = np.ones(len(Y['POOLED']), bool)
BL['POOLED'], _ = multiseed_best_layer(fr['POOLED'], Y['POOLED'], M['POOLED'], n_seeds=N_SEEDS, C=PROBE_C, groups=G['POOLED'])

def Xy(t):
    return {li: fr[t][li][M[t]] for li in fr[t]}, Y[t][M[t]], np.asarray(G[t])[M[t]] if t != 'POOLED' else G['POOLED']

def lr_seeds(t, test_feats, test_y):
    """Retrain probe t across seeds (group split), score all of test set (disjoint)."""
    X, y, gr = Xy(t); bl = BL[t]; aucs = []; last = None
    idx = np.arange(len(y))
    for seed in range(N_SEEDS):
        tr, _ = next(GroupShuffleSplit(1, test_size=TEST_SIZE, random_state=seed).split(idx, y, gr))
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=seed).fit(X[bl][tr], y[tr])
        p = clf.predict_proba(test_feats[bl])[:, 1]
        aucs.append(roc_auc_score(test_y, p))
        last = p
    lo, hi = _bootstrap_auc(np.asarray(test_y), np.asarray(last))
    return float(np.mean(aucs)), float(np.std(aucs)), (lo, hi), last

def diffmean_auc(t, test_feats, test_y):
    from sklearn.metrics import roc_auc_score
    X, y, gr = Xy(t); bl = BL[t]
    clf = fit_diffmean(X[bl], y)
    p = clf.predict_proba(test_feats[bl])[:, 1]
    a = roc_auc_score(test_y, p)
    return float(a)

# ── TF-IDF helpers (from notebook surface-leakage cell) ──────────────────────
def _pipe():
    return make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                         LogisticRegression(max_iter=MAX_ITER))
def tfidf_indomain(texts, labels, n_seeds=5):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels); aucs = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed); n = len(labels); perm = rng.permutation(n)
        cut = int(n * (1 - TEST_SIZE)); tr, te = perm[:cut], perm[cut:]
        if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[te])) < 2:
            continue
        pipe = _pipe().fit([texts[k] for k in tr], labels[tr])
        p = pipe.predict_proba([texts[k] for k in te])[:, 1]
        aucs.append(roc_auc_score(labels[te], p))
    return float(np.mean(aucs)) if aucs else float('nan')
def tfidf_cross(tr_t, tr_l, te_t, te_l):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(te_l)) < 2:
        return float('nan')
    pipe = _pipe().fit(tr_t, np.asarray(tr_l))
    return float(roc_auc_score(te_l, pipe.predict_proba(te_t)[:, 1]))

from sklearn.metrics import roc_auc_score
# P_C / P_S text from inference bundle for cross-TFIDF
pb = pickle.load(open(op / 'inference_P_bundle.pkl', 'rb'))
def ptext(df, m):
    t = (df['role_context'].astype(str) + ' ' + df['user_question'].astype(str) + ' '
         + df['raw_response'].astype(str)).to_numpy()
    return list(t[m])
PC_text = ptext(pb['df_combined'], mask_P_C); PC_lab = labels_P_C_lied[mask_P_C]
PS_text = ptext(pb['df_sales'], mask_P_S);   PS_lab = labels_P_S_lied[mask_P_S]

# ════════ key cells ══════════════════════════════════════════════════════════
KEY = [('P_C', 'HP_KR'), ('POOLED', 'HP_KR'), ('I_C', 'HP_KR'),
       ('P_S', 'ID'), ('I_S', 'ID')]
res = {}
print('\n=== Key LB cells: LR(seed mean±std)[CI] | DiffMean | balAcc@1% (response convention) ===')
for t, sub in KEY:
    tf, ty = lb_feats[sub], lb_lab[sub]
    m, s, (lo, hi), p = lr_seeds(t, tf, ty)
    dm = diffmean_auc(t, tf, ty)
    ba = ood.balanced_accuracy_at_fpr(ty, p, target_fpr=0.01)
    res[f'{t}->{sub}'] = {'lr_auc': m, 'lr_std': s, 'lr_ci': [lo, hi], 'diffmean_auc': dm, 'balacc_1pct': float(ba)}
    print(f'  {t:>6}->{sub:<5} LR={m:.3f}±{s:.3f} [{lo:.3f},{hi:.3f}]  DiffMean={dm:.3f}  balAcc@1%={ba:.3f}')
out['key_cells'] = res

# ── TF-IDF surface controls ──────────────────────────────────────────────────
print('\n=== TF-IDF surface controls (text-only) ===')
tfidf = {}
tfidf['within_HP_KR'] = tfidf_indomain(lb_text['HP_KR'], lb_lab['HP_KR'])
tfidf['within_ID'] = tfidf_indomain(lb_text['ID'], lb_lab['ID'])
tfidf['P_C_text->HP_KR_text'] = tfidf_cross(PC_text, PC_lab, lb_text['HP_KR'], lb_lab['HP_KR'])
tfidf['P_S_text->ID_text'] = tfidf_cross(PS_text, PS_lab, lb_text['ID'], lb_lab['ID'])
for k, v in tfidf.items():
    print(f'  {k}: {v:.3f}')
out['tfidf'] = tfidf

# ── Generator subset: Llama-3.3-70B-only ─────────────────────────────────────
print('\n=== Llama-3.3-70B-only generator subset (re-score key cells) ===')
LLAMA = 'llama-v3.3-70b-instruct'
gen = {}
for t, sub in KEY:
    sel = lb_model[sub] == LLAMA
    if sel.sum() < 20 or len(np.unique(lb_lab[sub][sel])) < 2:
        continue
    tf = {li: lb_feats[sub][li][sel] for li in lb_feats[sub]}; ty = lb_lab[sub][sel]
    m, s, ci, p = lr_seeds(t, tf, ty)
    gen[f'{t}->{sub}'] = {'n': int(sel.sum()), 'lr_auc': m, 'diffmean': diffmean_auc(t, tf, ty)}
    print(f'  {t:>6}->{sub:<5} (n={int(sel.sum())} llama-70b)  LR={m:.3f}  DiffMean={gen[f"{t}->{sub}"]["diffmean"]:.3f}')
out['llama70b_subset'] = gen

# ════════ B5: P_C -> HP-KR deep dive ═════════════════════════════════════════
print('\n=== B5: P_C->HP-KR deep validation ===')
b5 = {}
# cross-position: pre-trained clf vs resp-trained clf, scored on HP-KR resp & pre
fPC_pre = load('P_C', 'pre')
clf_pre = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=0).fit(
    fPC_pre[int(best_layer_P_C)][mask_P_C], labels_P_C_lied[mask_P_C])
clf_resp = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=0).fit(
    fr['P_C'][BL['P_C']][mask_P_C], labels_P_C_lied[mask_P_C])
b5['preTrained->HPKR_resp'] = float(roc_auc_score(lb_lab['HP_KR'], clf_pre.predict_proba(lb_feats['HP_KR'][int(best_layer_P_C)])[:, 1]))
b5['preTrained->HPKR_pre'] = float(roc_auc_score(lb_lab['HP_KR'], clf_pre.predict_proba(lb_pre['HP_KR'][int(best_layer_P_C)])[:, 1]))
b5['respTrained->HPKR_resp'] = float(roc_auc_score(lb_lab['HP_KR'], clf_resp.predict_proba(lb_feats['HP_KR'][BL['P_C']])[:, 1]))
print(f"  pre-trained P_C  -> HP-KR(resp)={b5['preTrained->HPKR_resp']:.3f}  HP-KR(pre)={b5['preTrained->HPKR_pre']:.3f}")
print(f"  resp-trained P_C -> HP-KR(resp)={b5['respTrained->HPKR_resp']:.3f}   (cross-position resolution)")
# pressure-level ablation (df_combined pressure_level among masked P_C)
plvl = pb['df_combined']['pressure_level'].to_numpy()[mask_P_C]
for lvl in ['level_2', 'level_3']:
    sub_m = plvl == lvl
    if sub_m.sum() < 20 or len(np.unique(labels_P_C_lied[mask_P_C][sub_m])) < 2:
        continue
    Xc = fr['P_C'][BL['P_C']][mask_P_C][sub_m]; yc = labels_P_C_lied[mask_P_C][sub_m]
    clf = LogisticRegression(max_iter=MAX_ITER, C=PROBE_C, random_state=0).fit(Xc, yc)
    a = roc_auc_score(lb_lab['HP_KR'], clf.predict_proba(lb_feats['HP_KR'][BL['P_C']])[:, 1])
    b5[f'trainOn_{lvl}->HPKR_resp'] = {'n_train': int(sub_m.sum()), 'auc': float(a)}
    print(f"  P_C[{lvl}] (n={int(sub_m.sum())}) -> HP-KR(resp) AUC={a:.3f}")
out['B5_P_C_HPKR'] = b5

json.dump(out, open(op / 'lb_robustness.json', 'w'), indent=2)
print(f'\nSaved {op / "lb_robustness.json"}')
