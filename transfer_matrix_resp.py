#!/usr/bin/env python
"""
AUROC cross-transfer (generalization) matrix with P/I' at the RESPONSE token.

Standalone, CPU-only. Reuses the response-token activations already extracted by
reextract_response_position.py (acts_*_resp_*.npy in OUTPUT_DIR) plus the cached
Z* features/probes from probe_state.pkl. Retrains P/I' on the response token,
then computes the 8x8 cross_transfer_matrix and compares it to the original
(pre-answer) matrix in results.json. Skips the (slow) cosine recompute.

Env: OUTPUT_DIR (default ~/output).
"""
import os, json, pickle
import numpy as np
from pathlib import Path
from probe_utils import (multiseed_best_layer, fit_final_probe,
                         cross_transfer_matrix, bootstrap_ci_matrix,
                         print_transfer_matrix)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C = 32, 10, 1.0

# ── cached state ─────────────────────────────────────────────────────────────
PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
g = globals()
for k, v in PS.items():
    g[k] = v   # labels_*, mask_*, groups_*, best_layer_Z*, dirs_*, multi_layers, ...

def load(tag, pos=None):
    suf = f'_{pos}' if pos else ''
    return {li: np.load(op / f'acts_{tag}{suf}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}{suf}_layer_{li:03d}.npy').exists()}

# ── response-token P/I' features ─────────────────────────────────────────────
feats_P_S_r = load('P_S', 'resp'); feats_P_C_r = load('P_C', 'resp')
feats_I_S_r = load('I_S', 'resp'); feats_I_C_r = load('I_C', 'resp')

# ── cached Z* features ───────────────────────────────────────────────────────
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

# ── retrain P/I' on the response token ───────────────────────────────────────
print('Retraining P/I\' on the response token...')
TRAIN = {'P_S': (feats_P_S_r, labels_P_S_lied, mask_P_S, groups_P_S),
         'P_C': (feats_P_C_r, labels_P_C_lied, mask_P_C, groups_P_C),
         'I_S': (feats_I_S_r, labels_I_S, mask_I_S, groups_I_S),
         'I_C': (feats_I_C_r, labels_I_C, mask_I_C, groups_I_C)}
bl = {}
for tag, (f, y, m, gr) in TRAIN.items():
    print(f'=== {tag} ===')
    bl[tag], _ = multiseed_best_layer(f, y, m, n_seeds=N_SEEDS, C=PROBE_C, groups=gr)

# ── matrix ───────────────────────────────────────────────────────────────────
all_probe_names = ["I'_S", "I'_C", "P_S", "P_C", "Z", "Z_mean", "Z_tok", "Z_multi"]
all_probe_data = {
    "I'_S":   (feats_I_S_r, labels_I_S,      mask_I_S, groups_I_S, bl['I_S'], PROBE_C, None),
    "I'_C":   (feats_I_C_r, labels_I_C,      mask_I_C, groups_I_C, bl['I_C'], PROBE_C, None),
    "P_S":    (feats_P_S_r, labels_P_S_lied, mask_P_S, groups_P_S, bl['P_S'], PROBE_C, None),
    "P_C":    (feats_P_C_r, labels_P_C_lied, mask_P_C, groups_P_C, bl['P_C'], PROBE_C, None),
    "Z":      (feats_Z,      labels_Z,      mask_Z,      None, best_layer_Z,      PROBE_C, None),
    "Z_mean": (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_mean, PROBE_C, None),
    "Z_tok":  (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_tok,  PROBE_C, None),
    "Z_multi":(feats_Z_mean, labels_Z_mean, mask_Z_mean, None, multi_layers,      PROBE_C, None),
}
tok_train_data   = {"Z_tok":  (tok_feats_h, tok_feats_d, counts_h, counts_d)}
multi_train_data = {"Z_multi":(tok_feats_h, tok_feats_d, counts_h, counts_d, multi_layers)}

print(f'\nComputing {len(all_probe_names)}x{len(all_probe_names)} cross-transfer matrix '
      '(P/I at response token)...')
full, preds = cross_transfer_matrix(all_probe_names, all_probe_data,
                                    tok_train_data=tok_train_data,
                                    multi_train_data=multi_train_data)
ci = bootstrap_ci_matrix(preds, len(all_probe_names), n_boot=1000)
mean_full, std_full = print_transfer_matrix(full, all_probe_names,
                                            'Cross-transfer AUC (P/I at response token)',
                                            ci_matrix=ci)

# ── compare to the original pre-answer matrix ────────────────────────────────
orig = json.load(open(op / 'results.json')) if (op / 'results.json').exists() else {}
om_d = orig.get('cross_transfer_NxN', {})
om = np.array(om_d['mean']) if om_d.get('mean') and om_d.get('probes') == all_probe_names else None
if om is not None:
    zi = [all_probe_names.index(n) for n in ['Z', 'Z_mean', 'Z_tok', 'Z_multi']]
    zblk = np.abs(mean_full[np.ix_(zi, zi)] - om[np.ix_(zi, zi)])
    print(f'\nZ-only block reproduced vs original: max|Δ| = {zblk.max():.3f} '
          f'({"OK" if zblk.max() < 0.05 else "CHECK"})')
    print('\n--- Critical cells: pre-answer -> response ---')
    crit = [("I'_S","P_S"),("P_S","I'_S"),("I'_C","P_C"),("P_C","I'_C"),
            ("Z","P_S"),("Z","P_C"),("Z","I'_S"),("Z","I'_C"),
            ("P_S","P_C"),("P_C","P_S"),("I'_S","I'_C"),("I'_C","I'_S")]
    print(f'{"train -> test":>16s}   {"pre":>6s}   {"resp":>6s}   {"Δ":>6s}')
    for tn, te in crit:
        i, j = all_probe_names.index(tn), all_probe_names.index(te)
        print(f'{tn+" -> "+te:>16s}   {om[i,j]:6.3f}   {mean_full[i,j]:6.3f}   {mean_full[i,j]-om[i,j]:+6.3f}')

json.dump({'probes': all_probe_names, 'best_layers_resp': {k: int(v) for k, v in bl.items()},
           'mean': mean_full.tolist(), 'std': std_full.tolist(), 'ci': ci.tolist()},
          open(op / 'transfer_matrix_resp.json', 'w'), indent=2)
print(f'\nSaved {op / "transfer_matrix_resp.json"}')
