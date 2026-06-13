#!/usr/bin/env python
"""
Pooled-probe generalization onto Liars' Bench (HP-KR + ID) — GPU extraction + scoring.

Headless port of the notebook's Liars' Bench cells (load → extract pre/response/token
activations → score every probe at both conventions), with a POOLED P_S+P_C probe
added to the probe set. Answers: how does the pooled pressure-deception probe
generalize to HP-KR (harm-pressure-knowledge-report, the P analogue) and ID
(instructed-deception, the I' analogue)?

Needs a GPU + a gated HF token (Llama-3.1-8B) with the Cadenza-Labs/liars-bench gate
accepted. Activations are cached as acts_OOD_LB_<KEY>_*.npy so re-runs are CPU-only.

Env: HF_TOKEN (required), OUTPUT_DIR (default ~/output), REPO_DIR (default script dir).
"""
import os, sys, gc, json, pickle
from pathlib import Path
import numpy as np
import torch

REPO_DIR = os.environ.get('REPO_DIR', os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)

MODEL_KEY = 'llama'
MODEL_MAP = {'llama': 'meta-llama/Llama-3.1-8B-Instruct',
             'qwen':  'Qwen/Qwen2.5-14B-Instruct'}
LOAD_IN_4BIT = True
N_LAYERS, N_SEEDS, PROBE_C = 32, 10, 1.0
LB_CAP = 1000

import probe_utils as pu
from probe_utils import multiseed_best_layer, fit_final_probe
import ood_eval_utils as ood

# ── cached probe state (labels/masks/groups, Z* probes, best layers, scalers) ─
PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
g = globals()
for k, v in PS.items():
    g[k] = v

hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
if not hf_token:
    raise SystemExit('ERROR: export HF_TOKEN=hf_... (Llama-3.1 + Liars\' Bench are gated).')


# ── activation extraction harness (from reextract_response_position.py) ───────
def extract_activations_batch(input_ids_list, batch_size=8, tag=''):
    """Last-real-token activations per sequence -> {layer: (n, d) float32}."""
    tokenizer.padding_side = 'right'
    all_vecs = {}
    from tqdm.auto import tqdm
    with torch.no_grad():
        for start in tqdm(range(0, len(input_ids_list), batch_size), desc=f'acts {tag}'):
            batch = input_ids_list[start:start + batch_size]
            last_pos = [s.shape[0] - 1 for s in batch]
            max_len = max(s.shape[0] for s in batch)
            ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(batch):
                ids[i, :s.shape[0]] = s; mask[i, :s.shape[0]] = 1
            ids, mask = ids.to(model.device), mask.to(model.device)
            pos_t = torch.tensor(last_pos, dtype=torch.long, device=model.device)
            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, use_cache=False)
            for li, lh in enumerate(out.hidden_states):
                bi = torch.arange(lh.shape[0], device=lh.device)
                all_vecs.setdefault(li, []).extend(
                    list(lh[bi, pos_t, :].detach().float().cpu().numpy()))
    return {li: np.stack(v) for li, v in all_vecs.items()}


def extract_activations_all_response_tokens(input_ids_list, response_starts,
                                            batch_size=4, tag=''):
    """All response-token activations -> {layer: list[(n_tok_i, d)]}."""
    tokenizer.padding_side = 'right'
    all_vecs = {}
    from tqdm.auto import tqdm
    with torch.no_grad():
        for start in tqdm(range(0, len(input_ids_list), batch_size), desc=f'tok {tag}'):
            batch = input_ids_list[start:start + batch_size]
            starts = response_starts[start:start + batch_size]
            max_len = max(s.shape[0] for s in batch)
            ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(batch):
                ids[i, :s.shape[0]] = s; mask[i, :s.shape[0]] = 1
            ids, mask = ids.to(model.device), mask.to(model.device)
            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, use_cache=False)
            for li, lh in enumerate(out.hidden_states):
                for i in range(len(batch)):
                    rs, sl = starts[i], batch[i].shape[0]
                    all_vecs.setdefault(li, []).append(
                        lh[i, rs:sl, :].detach().float().cpu().numpy())
    return all_vecs


# ── Liars' Bench load (notebook cell: load ID + HP-KR) ───────────────────────
lb_subsets = {'ID': 'instructed_deception', 'HP_KR': 'harm_pressure_kr'}

def _balanced_cap(rolls, labs, cap, seed=42):
    labs = np.asarray(labs)
    if cap is None or len(labs) <= cap:
        return rolls, labs
    rng = np.random.RandomState(seed)
    per = cap // 2
    idx = []
    for cls in (0, 1):
        ci = np.where(labs == cls)[0]
        if len(ci) > per:
            ci = rng.choice(ci, per, replace=False)
        idx.extend(ci.tolist())
    idx = sorted(idx)
    return [rolls[i] for i in idx], labs[idx]

lb_data = {}
for key, sub in lb_subsets.items():
    try:
        rolls, labs = ood.load_liars_bench(sub, token=hf_token)
        if not (len(labs) and len(np.unique(labs)) >= 2):
            print(f"Liars' Bench [{key}]: single-class/empty — skipping."); continue
        n0 = len(labs)
        rolls, labs = _balanced_cap(rolls, labs, LB_CAP)
        if len(labs) < n0:
            print(f"  [{key}] capped {n0} -> {len(labs)} "
                  f"({int((labs==0).sum())} honest / {int((labs==1).sum())} deceptive)")
        lb_data[key] = (rolls, labs)
    except Exception as e:
        print(f"Liars' Bench [{key}] load failed: {type(e).__name__}: {e}")
has_LB = len(lb_data) > 0
print(f"\nLiars' Bench loaded subsets: {list(lb_data.keys())}")
if not has_LB:
    raise SystemExit('No Liars\' Bench subsets loaded (check HF token + gate acceptance).')


# ── model (only if any subset uncached) ──────────────────────────────────────
def _tag(key): return f'OOD_LB_{key}'
def _cached(key):
    t = _tag(key)
    return ((op / f'acts_{t}_layer_000.npy').exists()
            and (op / f'acts_{t}_pre_layer_000.npy').exists()
            and (op / f'tok_acts_{t}_layer_000.npz').exists())

need_model = any(not _cached(k) for k in lb_data)
model = None
if need_model:
    print('Loading Llama-3.1-8B (4-bit) for extraction...')
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    model_id = MODEL_MAP[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    mk = {'token': hf_token}
    if LOAD_IN_4BIT:
        mk['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        mk['device_map'] = 'auto'
    else:
        mk['device_map'] = 'auto'
        mk['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **mk)
    model.eval()
    print('Model loaded.')

    _eot_ids = set()
    for _t in ['<|eot_id|>', '<|end_of_text|>']:
        try:
            _tid = tokenizer.convert_tokens_to_ids(_t)
            if isinstance(_tid, int) and _tid >= 0 and _tid != tokenizer.unk_token_id:
                _eot_ids.add(_tid)
        except Exception:
            pass
    if tokenizer.eos_token_id is not None:
        _eot_ids.add(tokenizer.eos_token_id)

    def _ct_ids(messages, add_gen):
        out = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=add_gen, return_tensors='pt')
        # transformers 5.x may return a BatchEncoding/dict instead of a tensor
        if hasattr(out, 'input_ids'):
            out = out.input_ids
        elif isinstance(out, dict):
            out = out['input_ids']
        if hasattr(out, 'tolist'):
            out = out.tolist()
        while isinstance(out, list) and len(out) and isinstance(out[0], list):
            out = out[0]
        return [int(t) for t in out]

    def _resp_ids_and_start(transcript):
        if len(transcript) and transcript[-1]['role'] == 'assistant':
            ids = _ct_ids(transcript, add_gen=False)
            while len(ids) > 1 and ids[-1] in _eot_ids:
                ids = ids[:-1]
            prefix = transcript[:-1]
            pre_ids = _ct_ids(prefix, add_gen=True) if prefix else []
            rs = min(len(pre_ids), len(ids) - 1)
        else:
            ids = _ct_ids(transcript, add_gen=True)
            rs = max(0, len(ids) - 1)
        return ids, min(rs, len(ids) - 1)

    def _pre_ids(transcript):
        msgs = list(transcript)
        if msgs and msgs[-1]['role'] == 'assistant':
            msgs = msgs[:-1]
        if not msgs:
            msgs = list(transcript)
        return _ct_ids(msgs, add_gen=True)

# ── extract / load each subset ───────────────────────────────────────────────
lb_feats, lb_feats_pre, lb_tok_feats, lb_labels = {}, {}, {}, {}
for key, (rolls, labs) in lb_data.items():
    t = _tag(key)
    if _cached(key):
        print(f"[{key}] loading cached activations")
        lb_feats[key] = {li: np.load(op / f'acts_{t}_layer_{li:03d}.npy')
                         for li in range(N_LAYERS) if (op / f'acts_{t}_layer_{li:03d}.npy').exists()}
        lb_feats_pre[key] = {li: np.load(op / f'acts_{t}_pre_layer_{li:03d}.npy')
                             for li in range(N_LAYERS) if (op / f'acts_{t}_pre_layer_{li:03d}.npy').exists()}
        lb_labels[key] = np.load(op / f'labels_{t}.npy')
        lb_tok_feats[key] = {}
        for li in range(N_LAYERS):
            tp = op / f'tok_acts_{t}_layer_{li:03d}.npz'
            if tp.exists():
                npz = np.load(tp)
                lb_tok_feats[key][li] = [npz[k] for k in sorted(
                    npz.files, key=lambda x: int(x.replace('arr_', '')))]
        continue
    print(f"[{key}] extracting pre + response + token for {len(rolls)} rollouts...")
    resp_ids, resp_starts, pre_ids_l = [], [], []
    for r in rolls:
        rid, rs = _resp_ids_and_start(r['transcript'])
        resp_ids.append(torch.tensor(rid, dtype=torch.long)); resp_starts.append(rs)
        pre_ids_l.append(torch.tensor(_pre_ids(r['transcript']), dtype=torch.long))
    feats = extract_activations_batch(resp_ids, batch_size=2, tag=f'{t}_resp')
    for li in feats:
        np.save(op / f'acts_{t}_layer_{li:03d}.npy', feats[li])
    lb_feats[key] = feats
    fpre = extract_activations_batch(pre_ids_l, batch_size=2, tag=f'{t}_pre')
    for li in fpre:
        np.save(op / f'acts_{t}_pre_layer_{li:03d}.npy', fpre[li])
    lb_feats_pre[key] = fpre
    np.save(op / f'labels_{t}.npy', labs); lb_labels[key] = labs
    tok = extract_activations_all_response_tokens(resp_ids, resp_starts, batch_size=4, tag=f'{t}_tok')
    for li in tok:
        np.savez(op / f'tok_acts_{t}_layer_{li:03d}.npz', *tok[li])
    lb_tok_feats[key] = tok
    print(f"  [{key}] saved pre + response + token, {len(rolls)} examples")

if need_model and model is not None:
    del model; torch.cuda.empty_cache(); gc.collect(); print('Model freed.')

# ── build the POOLED probe (response token, P_S + P_C) — same as Phase 1 ─────
def load(tag, pos=None):
    suf = f'_{pos}' if pos else ''
    return {li: np.load(op / f'acts_{tag}{suf}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'acts_{tag}{suf}_layer_{li:03d}.npy').exists()}
feats_P_S_r, feats_P_C_r = load('P_S', 'resp'), load('P_C', 'resp')
_pl = sorted(set(feats_P_S_r) & set(feats_P_C_r))
feats_POOLED = {li: np.concatenate([feats_P_S_r[li][mask_P_S], feats_P_C_r[li][mask_P_C]], 0) for li in _pl}
labels_POOLED = np.concatenate([labels_P_S_lied[mask_P_S], labels_P_C_lied[mask_P_C]])
groups_POOLED = np.concatenate([np.array([f'S:{x}' for x in groups_P_S[mask_P_S]]),
                                np.array([f'C:{x}' for x in groups_P_C[mask_P_C]])])
mask_POOLED = np.ones(len(labels_POOLED), dtype=bool)
print('Fitting POOLED probe (response token)...')
best_layer_POOLED, _ = multiseed_best_layer(feats_POOLED, labels_POOLED, mask_POOLED,
                                            n_seeds=N_SEEDS, C=PROBE_C, groups=groups_POOLED)
clf_POOLED, _, _ = fit_final_probe(feats_POOLED, labels_POOLED, mask_POOLED, best_layer_POOLED)
print(f'  POOLED best layer {best_layer_POOLED}')

# ── score every probe on Liars' Bench at BOTH conventions (notebook cell 82) ──
ood_results = {}
standard_probes = [
    ('P_S', clf_P_S, best_layer_P_S), ('P_C', clf_P_C, best_layer_P_C),
    ("I'_S", clf_I_S, best_layer_I_S), ("I'_C", clf_I_C, best_layer_I_C),
    ('Z', clf_Z, best_layer_Z), ('Z_mean', clf_Z_mean, best_layer_Z_mean),
    ('POOLED', clf_POOLED, best_layer_POOLED),
]
# POOLED is response-trained -> native at the response ('resp') convention, like Z_mean.
_LASTTOK_NATIVE = {'P_S', 'P_C', "I'_S", "I'_C", 'Z'}
_RESP_NATIVE = {'POOLED'}

def _respmean(tok_feats):
    return {li: np.stack([a.mean(axis=0) for a in tok_feats[li]]) for li in tok_feats}

def _store(name, rk, auc, acc, scores, labels, mode):
    ood_results.setdefault(name, {})
    ood_results[name][rk] = {
        'auc': auc, 'acc': acc,
        'tpr_at_1pct_fpr': ood.tpr_at_fpr(labels, scores, target_fpr=0.01),
        'bal_acc_1pct': ood.balanced_accuracy_at_fpr(labels, scores, target_fpr=0.01),
        'mode': mode}
    tag = f' [{mode}]' if mode else ''
    print(f'{name:>8s} {rk}: AUC={auc:.3f}  balAcc@1%={ood_results[name][rk]["bal_acc_1pct"]:.3f}{tag}')

def _eval_lb(feats_lt, tok_feats, labels, result_key, faithful):
    respmean = _respmean(tok_feats) if (faithful and tok_feats) else None
    for name, clf, layer in standard_probes:
        if name == 'Z_mean' and respmean is not None and layer in respmean:
            res = ood.eval_probe_ood(clf, respmean, labels, layer)
            _store(name, result_key, res['auc'], res['acc'], res['probs'], labels, 'response-mean')
        else:
            res = ood.eval_probe_ood(clf, feats_lt, labels, layer)
            # mode tag: native at pre for pre-trained P/I'/Z; native at resp for POOLED
            if faithful:  # response convention
                mode = None if name in _RESP_NATIVE else (
                    'last-token-approx' if name not in _LASTTOK_NATIVE else None)
            else:         # pre-answer convention
                mode = None if name in _LASTTOK_NATIVE else 'pre-approx'
            _store(name, result_key, res['auc'], res['acc'], res['probs'], labels, mode)
    if faithful and tok_feats and best_layer_Z_tok in tok_feats:
        res = ood.eval_tok_probe_ood(clf_Z_tok, scaler_Z_tok, tok_feats, labels, best_layer_Z_tok)
        _store('Z_tok', result_key, res['auc'], res['acc'], res['scores'], labels, 'token-level')
    else:
        res = ood.eval_tok_probe_ood(clf_Z_tok, scaler_Z_tok, feats_lt, labels, best_layer_Z_tok)
        _store('Z_tok', result_key, res['auc'], res['acc'], res['scores'], labels, 'last-token-approx')
    if faithful and respmean is not None and all(li in respmean for li in multi_layers):
        res = ood.eval_multi_probe_ood(clf_Z_multi, scaler_Z_multi, respmean, labels, multi_layers)
        _store('Z_multi', result_key, res['auc'], res['acc'], res['probs'], labels, 'response-mean')
    elif all(li in feats_lt for li in multi_layers):
        res = ood.eval_multi_probe_ood(clf_Z_multi, scaler_Z_multi, feats_lt, labels, multi_layers)
        _store('Z_multi', result_key, res['auc'], res['acc'], res['probs'], labels, 'last-token-approx')

for key in lb_data:
    print(f'\n=== Liars\' Bench [{key}] — OUR convention (pre-answer) ===')
    _eval_lb(lb_feats_pre[key], None, lb_labels[key], f'LB_{key}_pre', faithful=False)
    print(f'--- Liars\' Bench [{key}] — THEIR convention (response) ---')
    _eval_lb(lb_feats[key], lb_tok_feats.get(key), lb_labels[key], f'LB_{key}_resp', faithful=True)

# ── table + save ─────────────────────────────────────────────────────────────
probe_order = ["P_S", "P_C", "I'_S", "I'_C", "Z", "Z_mean", "Z_tok", "Z_multi", "POOLED"]
print('\n' + '=' * 72)
print("LIARS' BENCH — pooled-probe generalization   cells = AUC (balAcc@1%)")
print("  ID = I' analogue, HP_KR = P analogue.   POOLED native = response column.")
print('=' * 72)
for suffix, title in [('resp', 'response convention'), ('pre', 'pre-answer convention')]:
    print(f'\n[{title}]')
    print(f'{"Probe":>8s}  ' + '  '.join(f'{k:>16s}' for k in lb_data))
    for name in probe_order:
        cells = []
        for s in lb_data:
            r = ood_results.get(name, {}).get(f'LB_{s}_{suffix}')
            cells.append(f'{r["auc"]:.3f} ({r["bal_acc_1pct"]:.2f})' if r else '—')
        print(f'{name:>8s}  ' + '  '.join(f'{c:>16s}' for c in cells))

out = {'subsets': {k: {'n': int(len(lb_labels[k])),
                       'honest': int((lb_labels[k] == 0).sum()),
                       'deceptive': int((lb_labels[k] == 1).sum())} for k in lb_data},
       'best_layer_POOLED': int(best_layer_POOLED),
       'metrics': ood_results}
json.dump(out, open(op / 'pooled_ood_liars_bench.json', 'w'), indent=2)
print(f'\nSaved {op / "pooled_ood_liars_bench.json"}')
