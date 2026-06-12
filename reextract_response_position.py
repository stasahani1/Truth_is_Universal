#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reextract_response_position.py
Run on a remote headless GPU (e.g. Prime Intellect). Standalone port of
reextract_response_position.ipynb.

Required env vars:
  HF_TOKEN     : HuggingFace token (Llama-3.1-8B is gated).
  OUTPUT_DIR   : dir holding the main pipeline caches (probe_state.pkl, acts_*,
                 inference_*_bundle.pkl, Z*, OOD/LB). Default /mnt/gdrive/mechanism_comparison_output
  REPO_DIR     : repo root containing probe_utils.py / ood_eval_utils.py
                 (defaults to this script's directory).
Run:
  export HF_TOKEN=hf_...
  export OUTPUT_DIR=/mnt/gdrive            # or a local copy of the cache dir
  python reextract_response_position.py
"""
import matplotlib
matplotlib.use('Agg')                      # headless: no display
import os, sys

REPO_DIR = os.environ.get('REPO_DIR', os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '/mnt/gdrive/mechanism_comparison_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'REPO_DIR={REPO_DIR}\nOUTPUT_DIR={OUTPUT_DIR}')
if not os.path.exists(os.path.join(OUTPUT_DIR, 'probe_state.pkl')):
    print('WARNING: probe_state.pkl not found in OUTPUT_DIR — caches must be present '
          '(mount/copy the main pipeline output here first).')


# ======================================================================
# # Re-extract P & I' at the response token — retrain & recompute only what changes
# 
# Probes **P_S/P_C/I'_S/I'_C** are trained at the **pre-answer** prompt-end in the main pipeline.
# This notebook re-extracts them at three positions in ONE forward pass — **pre-answer**,
# **last response (assertion) token**, **mean-over-response** — retrains on the assertion token,
# and recomputes only the cosine pairs / transfer-matrix cells / OOD-LB scores that involve P or I'.
# Z* probes, their features, and the Z↔Z cells are reused from cache. No re-inference (the model's
# answers are cached in the inference bundles).
# ======================================================================

# ---- cell 2 ----
# ── Configuration (must match the main notebook) ─────────────────────────────
MODEL_KEY = 'llama'
MODEL_MAP = {'llama': 'meta-llama/Llama-3.1-8B-Instruct',
             'qwen':  'Qwen/Qwen2.5-14B-Instruct'}
LOAD_IN_4BIT = True
MAX_ITER = 2000
N_SEEDS = 10
TEST_SIZE = 0.2
N_LAYERS = 32
PROBE_C = 1.0
ZOU_PROBE_C = 0.1
N_RANDOM_DIRS = 1000
FORWARD_BATCH_SIZE = 8

import gc, json, pickle
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from probe_utils import *
import ood_eval_utils as ood
from importlib import reload; reload(ood)
print('probe_utils + ood_eval_utils loaded')

# ---- cell 3 ----
# ── Inference utilities ───────────────────────────────────────────────────────────

def _to_1d(t):
    """Normalise apply_chat_template output to a 1-D int64 tensor."""
    if isinstance(t, torch.Tensor):
        return t.squeeze()
    if isinstance(t, (list, tuple)):
        inner = t[0] if isinstance(t[0], list) else t
        return torch.tensor(inner, dtype=torch.long)
    ids = t['input_ids']
    ids = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
    return ids.squeeze()


def build_prompt_ids(system_prompt: str, user_prompt: str,
                     add_gen_prompt: bool = True) -> torch.Tensor:
    """Tokenise [system + user] with optional generation prompt."""
    msgs = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_prompt},
    ]
    return _to_1d(tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=add_gen_prompt,
        return_tensors='pt',
    ))


def build_zou_prompt_ids(system_prompt: str, user_prompt: str,
                         assistant_prefix: str) -> torch.Tensor:
    """Build token IDs for Zou-style probes: chat template + assistant prefix.

    Uses add_generation_prompt=True to get the standard assistant header,
    then appends the truncated-fact prefix as raw token IDs.
    """
    msgs = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user',   'content': user_prompt},
    ]
    base_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
    )
    if isinstance(base_ids[0], list):
        base_ids = base_ids[0]
    prefix_ids = tokenizer(assistant_prefix, add_special_tokens=False).input_ids
    full_ids = base_ids + prefix_ids
    return torch.tensor(full_ids, dtype=torch.long)


def get_yn_token_ids() -> list:
    """Return token IDs for all Yes/No surface forms."""
    ids = set()
    for surface in ('Yes', 'No', 'yes', 'no', ' Yes', ' No', ' yes', ' no'):
        ids.update(tokenizer(surface, add_special_tokens=False).input_ids)
    return list(ids)


def normalize_answer(text: str) -> str:
    """Return 'yes', 'no', or 'unclear'."""
    t = text.strip().lower().lstrip(' .,;:!?\"\'\'\n')
    if t.startswith('yes'):
        return 'yes'
    if t.startswith('no'):
        return 'no'
    return 'unclear'


def run_inference_batch(prompt_pairs: list) -> tuple:
    """Constrained Yes/No decoding on a list of (system, user) prompt pairs.

    Returns (raw_responses, normalized_answers).
    """
    yn_ids = get_yn_token_ids()

    def _yn_prefix_fn(batch_id, input_ids):
        return yn_ids

    tokenizer.padding_side = 'left'
    seqs = [build_prompt_ids(sys, usr) for sys, usr in prompt_pairs]
    max_len = max(s.shape[0] for s in seqs)
    pad_id  = tokenizer.pad_token_id
    ids  = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        offset = max_len - s.shape[0]
        ids[i, offset:]  = s
        mask[i, offset:] = 1

    ids, mask = ids.to(model.device), mask.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=1, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=_yn_prefix_fn,
        )
    raw_list, norm_list = [], []
    for i in range(len(seqs)):
        new_ids = out[i, max_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        raw_list.append(raw)
        norm_list.append(normalize_answer(raw))
    return raw_list, norm_list

# ---- cell 4 ----
# ── Activation extraction ─────────────────────────────────────────────────────────

def extract_activations_batch(input_ids_list: list,
                              batch_size: int = FORWARD_BATCH_SIZE,
                              tag: str = '') -> dict:
    """Forward-pass activation extraction at the last real token per sequence.

    Args:
        input_ids_list: list of 1-D int64 tensors (variable length).
        batch_size: GPU batch size.
        tag: label for progress bar.

    Returns:
        {layer_idx: np.ndarray of shape (n, hidden_dim)} in float32.
    """
    tokenizer.padding_side = 'right'
    all_vecs = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(input_ids_list), batch_size),
                          desc=f'Activations {tag}'):
            batch_seqs = input_ids_list[start:start + batch_size]
            last_pos = [s.shape[0] - 1 for s in batch_seqs]
            max_len  = max(s.shape[0] for s in batch_seqs)
            pad_id   = tokenizer.pad_token_id

            ids  = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(batch_seqs):
                ln = s.shape[0]
                ids[i, :ln]  = s
                mask[i, :ln] = 1

            ids  = ids.to(model.device)
            mask = mask.to(model.device)
            pos_t = torch.tensor(last_pos, dtype=torch.long, device=model.device)

            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, use_cache=False)

            for li, lh in enumerate(out.hidden_states):
                bi = torch.arange(lh.shape[0], device=lh.device)
                v  = lh[bi, pos_t, :].detach().float().cpu().numpy()
                all_vecs.setdefault(li, []).extend(list(v))

    feats = {}
    for li, vlist in all_vecs.items():
        feats[li] = np.stack(vlist)
    print(f'  Extracted {len(feats)} layers, {len(input_ids_list)} examples.')
    return feats

def extract_activations_all_response_tokens(
        input_ids_list: list,
        response_start_positions: list,
        batch_size: int = FORWARD_BATCH_SIZE,
        tag: str = '') -> dict:
    """Extract activations at ALL response tokens (not just the last one).

    For each sequence, extracts at every token position from response_start
    to end-of-sequence. Returns one activation per response token, allowing
    mean-aggregation during probe training/evaluation.

    Args:
        input_ids_list: list of 1-D int64 tensors (variable length).
        response_start_positions: list of ints — index of first response token
            in each sequence.
        batch_size: GPU batch size.
        tag: label for progress bar.

    Returns:
        {layer_idx: list of np.ndarray}, where each inner array has shape
        (n_response_tokens_i, hidden_dim) for the i-th example.
    """
    tokenizer.padding_side = 'right'
    all_vecs = {}

    with torch.no_grad():
        for start in tqdm(range(0, len(input_ids_list), batch_size),
                          desc=f'All-token activations {tag}'):
            batch_seqs = input_ids_list[start:start + batch_size]
            batch_starts = response_start_positions[start:start + batch_size]
            max_len = max(s.shape[0] for s in batch_seqs)
            pad_id  = tokenizer.pad_token_id

            ids  = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, s in enumerate(batch_seqs):
                ln = s.shape[0]
                ids[i, :ln]  = s
                mask[i, :ln] = 1

            ids  = ids.to(model.device)
            mask = mask.to(model.device)

            out = model(input_ids=ids, attention_mask=mask,
                        output_hidden_states=True, use_cache=False)

            for li, lh in enumerate(out.hidden_states):
                for i in range(len(batch_seqs)):
                    resp_start = batch_starts[i]
                    seq_len = batch_seqs[i].shape[0]
                    # Extract all tokens from response_start to end
                    vecs = lh[i, resp_start:seq_len, :].detach().float().cpu().numpy()
                    all_vecs.setdefault(li, []).append(vecs)

    print(f'  Extracted {len(all_vecs)} layers, {len(input_ids_list)} examples '
          f'(all response tokens).')
    return all_vecs


# ---- cell 5 ----
# ── Load cached state from the main pipeline ─────────────────────────────────
op = Path(OUTPUT_DIR)
with open(op / 'probe_state.pkl', 'rb') as f:
    PS = pickle.load(f)
print(f'probe_state.pkl: {len(PS)} objects')
# unpack everything (clf_*, w_*, dirs_*, best_layer_*, aucs_*, labels_*, mask_*, groups_*, Z*)
for k, v in PS.items():
    globals()[k] = v

# inference bundles -> dataframes carrying the model's generated answers (raw_response)
with open(op / 'inference_P_bundle.pkl', 'rb') as f:
    _pb = pickle.load(f)
df_sales, df_combined = _pb['df_sales'], _pb['df_combined']
with open(op / 'inference_I_bundle.pkl', 'rb') as f:
    _ib = pickle.load(f)
df_I_sales_matched = _ib['df_I_sales_matched']
df_I_combined_matched = _ib['df_I_combined_matched']

# game scenarios (P-analogue OOD) — optional
_gb = op / 'inference_G_bundle.pkl'
if _gb.exists():
    with open(_gb, 'rb') as f:
        df_game = pickle.load(f)['df_game']
    print(f'G bundle loaded: {len(df_game)} game rows')
else:
    df_game = None
    print('No inference_G_bundle.pkl — Game OOD will be skipped.')

for nm, df in [('df_sales', df_sales), ('df_combined', df_combined),
               ("df_I_sales_matched", df_I_sales_matched),
               ("df_I_combined_matched", df_I_combined_matched)]:
    assert 'raw_response' in df.columns, f'{nm} missing raw_response'
print('bundles loaded; raw_response present in all dataframes')
print(f'P_S rows={len(df_sales)}, P_C rows={len(df_combined)}, '
      f"I'_S matched={len(df_I_sales_matched)}, I'_C matched={len(df_I_combined_matched)}")

# ======================================================================
# ## 1. Re-extract P & I' at three positions (one forward pass)
# ======================================================================

# ---- cell 7 ----
# ── Load model for re-extraction ─────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_id = MODEL_MAP[MODEL_KEY]
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
_mk = {'token': hf_token}
if LOAD_IN_4BIT:
    if not torch.cuda.is_available():
        raise RuntimeError('LOAD_IN_4BIT=True requires a GPU.')
    _mk['quantization_config'] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    _mk['device_map'] = 'auto'
elif torch.cuda.is_available():
    _mk['device_map'] = 'auto'
    _mk['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    _mk['torch_dtype'] = torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, **_mk)
model.eval()
print('Model loaded.')

# ---- cell 8 ----
# ── Three-position extraction in one forward pass ────────────────────────────
# For each row: full = [system/role_context, user] + assistant:<generated answer>.
# response_start = len(prompt_ids)-1, so the per-example array is
# [pre_answer_token, response_tok_0, response_tok_1, ...]. From it we read:
#   pre        = row 0          (sanity: ~ existing acts_<tag> cache)
#   last-resp  = row[-1]        (assertion token; the train/headline position)
#   mean-resp  = row[1:].mean(0)
LAYERS = list(range(N_LAYERS))

def reextract_three_positions(df, sys_col, usr_col, ans_col, tag, batch_size=2):
    full_ids, resp_starts = [], []
    for _, r in df.iterrows():
        prompt = build_prompt_ids(str(r[sys_col]), str(r[usr_col]))   # 1-D tensor, gen-prompt
        ans = str(r[ans_col]).strip()
        ans_ids = tokenizer(ans, add_special_tokens=False).input_ids if ans and ans.lower() != 'unclear' else []
        full = torch.cat([prompt, torch.tensor(ans_ids, dtype=torch.long)]) if ans_ids else prompt
        full_ids.append(full)
        resp_starts.append(prompt.shape[0] - 1)              # include pre-answer token as row 0
    raw = extract_activations_all_response_tokens(full_ids, resp_starts,
                                                  batch_size=batch_size, tag=tag)
    pre, resp, respmean = {}, {}, {}
    for li in LAYERS:
        arrs = raw[li]
        pre[li]     = np.stack([a[0] for a in arrs])
        resp[li]    = np.stack([a[-1] for a in arrs])
        respmean[li]= np.stack([(a[1:].mean(0) if a.shape[0] > 1 else a[0]) for a in arrs])
    for pos, d in [('pre', pre), ('resp', resp), ('respmean', respmean)]:
        for li in LAYERS:
            np.save(op / f'acts_{tag}_{pos}_layer_{li:03d}.npy', d[li])
    print(f'  {tag}: saved pre/resp/respmean for {len(full_ids)} rows, {len(LAYERS)} layers')
    return pre, resp, respmean

EXTRACT = [
    ('P_S',  df_sales,              'role_context',  'user_question', 'raw_response'),
    ('P_C',  df_combined,           'role_context',  'user_question', 'raw_response'),
    ("I_S",  df_I_sales_matched,    'system_prompt', 'user_question', 'raw_response'),
    ("I_C",  df_I_combined_matched, 'system_prompt', 'user_question', 'raw_response'),
]
if df_game is not None:
    EXTRACT.append(('G', df_game, 'role_context', 'user_question', 'raw_response'))
acts3 = {}   # tag -> {'pre':..., 'resp':..., 'respmean':...}
for tag, df, sc, uc, ac in EXTRACT:
    print(f'[{tag}] extracting...')
    pre, resp, rm = reextract_three_positions(df, sc, uc, ac, tag)
    acts3[tag] = {'pre': pre, 'resp': resp, 'respmean': rm}

del model; torch.cuda.empty_cache(); gc.collect()
print('Re-extraction complete; model freed.')

# ---- cell 9 ----
# ── Sanity: the new "pre" position should match the existing pre-answer cache ──
def _load_cached(tag):
    return {li: np.load(op / f'acts_{tag}_layer_{li:03d}.npy') for li in LAYERS
            if (op / f'acts_{tag}_layer_{li:03d}.npy').exists()}
for tag in ['P_S', 'P_C', 'I_S', 'I_C']:
    cached = _load_cached(tag)
    if cached and 0 in cached and acts3[tag]['pre'][0].shape == cached[0].shape:
        d = np.abs(acts3[tag]['pre'][16] - cached[16]).max()
        print(f'  {tag}: max|new_pre - cached| at layer 16 = {d:.4g} '
              f'({"OK ~0" if d < 1e-2 else "DIFF — check tokenization"})')
    else:
        print(f'  {tag}: no comparable cached pre-answer activations (shape mismatch or missing)')

# ======================================================================
# ## 2. Retrain P & I' on the assertion (last-response) token
# ======================================================================

# ---- cell 11 ----
# ── Retrain on the response-token features (labels/masks/groups reused from cache) ──
# Only the FEATURES change position; labels_*/mask_*/groups_* are position-independent
# and come straight from probe_state.pkl (loaded above).
TRAIN = {
    'P_S': ('resp', labels_P_S_lied, mask_P_S, groups_P_S),
    'P_C': ('resp', labels_P_C_lied, mask_P_C, groups_P_C),
    'I_S': ('resp', labels_I_S,      mask_I_S, groups_I_S),
    'I_C': ('resp', labels_I_C,      mask_I_C, groups_I_C),
}
new_probes = {}   # tag -> dict(clf, w, dirs, best_layer, aucs, feats)
for tag, (pos, labels, mask, groups) in TRAIN.items():
    feats = acts3[tag][pos]
    n = feats[0].shape[0]
    assert n == len(labels) == len(mask), f'{tag}: feats {n} vs labels {len(labels)} vs mask {len(mask)}'
    print(f'=== {tag} (response token) ===')
    bl, aucs = multiseed_best_layer(feats, labels, mask, n_seeds=N_SEEDS, C=PROBE_C, groups=groups)
    clf, w, _ = fit_final_probe(feats, labels, mask, bl)
    dirs = all_layer_directions(feats, labels, mask)
    new_probes[tag] = dict(clf=clf, w=w, dirs=dirs, best_layer=bl, aucs=aucs, feats=feats)

# friendly handles for the response-position probes
clf_P_S_r, best_layer_P_S_r, dirs_P_S_r, aucs_P_S_r, feats_P_S_r = (new_probes['P_S'][k] for k in ('clf','best_layer','dirs','aucs','feats'))
clf_P_C_r, best_layer_P_C_r, dirs_P_C_r, aucs_P_C_r, feats_P_C_r = (new_probes['P_C'][k] for k in ('clf','best_layer','dirs','aucs','feats'))
clf_I_S_r, best_layer_I_S_r, dirs_I_S_r, aucs_I_S_r, feats_I_S_r = (new_probes['I_S'][k] for k in ('clf','best_layer','dirs','aucs','feats'))
clf_I_C_r, best_layer_I_C_r, dirs_I_C_r, aucs_I_C_r, feats_I_C_r = (new_probes['I_C'][k] for k in ('clf','best_layer','dirs','aucs','feats'))

print('\nWithin-distribution AUC (response token) vs cached pre-answer:')
for tag, blc, aucsc in [('P_S', best_layer_P_S, aucs_P_S), ('P_C', best_layer_P_C, aucs_P_C),
                        ('I_S', best_layer_I_S, aucs_I_S), ('I_C', best_layer_I_C, aucs_I_C)]:
    nb_ = new_probes[tag]
    print(f"  {tag}: resp L{nb_['best_layer']} AUC={nb_['aucs'][:,nb_['best_layer']].mean():.3f}  |  "
          f"pre L{blc} AUC={aucsc[:,blc].mean():.3f}")

# ---- cell 12 ----
# ── Reload cached Z* features (unchanged) for the cosine / matrix recompute ─────
def load_feats(tag, n_layers=N_LAYERS):
    return {li: np.load(op / f'acts_{tag}_layer_{li:03d}.npy') for li in range(n_layers)
            if (op / f'acts_{tag}_layer_{li:03d}.npy').exists()}

feats_Z = {li: np.concatenate([np.load(op / f'acts_Z_honest_layer_{li:03d}.npy'),
                               np.load(op / f'acts_Z_deceptive_layer_{li:03d}.npy')], 0)
           for li in range(N_LAYERS) if (op / f'acts_Z_honest_layer_{li:03d}.npy').exists()}
feats_Z_mean = {li: np.concatenate([np.load(op / f'acts_Z_mean_honest_layer_{li:03d}.npy'),
                                    np.load(op / f'acts_Z_mean_deceptive_layer_{li:03d}.npy')], 0)
                for li in range(N_LAYERS) if (op / f'acts_Z_mean_honest_layer_{li:03d}.npy').exists()}

def load_tok(tag):
    return {li: np.load(op / f'acts_Z_tok_{tag}_layer_{li:03d}.npy').astype(np.float32)
            for li in range(N_LAYERS) if (op / f'acts_Z_tok_{tag}_layer_{li:03d}.npy').exists()}
tok_feats_h = load_tok('honest'); tok_feats_d = load_tok('deceptive')
counts_h = np.load(op / 'Z_tok_counts_honest.npy'); counts_d = np.load(op / 'Z_tok_counts_deceptive.npy')
print(f'Z feats: last-token {len(feats_Z)} layers, mean {len(feats_Z_mean)} layers, '
      f'tok {len(tok_feats_h)} layers')
print('Z* probes/dirs/best-layers reused from probe_state.pkl (unchanged).')

# ======================================================================
# ## 3. Recompute cosines & transfer matrix (P/I' = response token; Z* = cached)
# Only the 22 pairs / rows+cols touching P or I' change. The 6 Z↔Z pairs and the Z-only matrix
# block are reproduced from the cached Z features (deterministic) and asserted against the
# original `results.json`.
# ======================================================================

# ---- cell 14 ----
# ── Unified probe configs: P/I' -> response-token feats; Z* -> cached ─────────
all_probe_keys  = ['I_S', 'I_C', 'P_S', 'P_C', 'Z', 'Z_mean', 'Z_tok', 'Z_multi']
all_probe_names = ["I'_S", "I'_C", "P_S", "P_C", "Z", "Z_mean", "Z_tok", "Z_multi"]

all_probe_configs = {
    'I_S':    (feats_I_S_r, labels_I_S,      mask_I_S, groups_I_S, PROBE_C, None),
    'I_C':    (feats_I_C_r, labels_I_C,      mask_I_C, groups_I_C, PROBE_C, None),
    'P_S':    (feats_P_S_r, labels_P_S_lied, mask_P_S, groups_P_S, PROBE_C, None),
    'P_C':    (feats_P_C_r, labels_P_C_lied, mask_P_C, groups_P_C, PROBE_C, None),
    'Z':      (feats_Z,      labels_Z,      mask_Z,      None, PROBE_C, None),
    'Z_mean': (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, PROBE_C, None),
    'Z_tok':  (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, PROBE_C, None),
    'Z_multi':(feats_Z_mean, labels_Z_mean, mask_Z_mean, None, PROBE_C, None),
}
precomputed_dirs = {'Z_tok': dirs_Z_tok_C1, 'Z_multi': dirs_Z_multi_C1}
Z_ONLY = {'Z', 'Z_mean', 'Z_tok', 'Z_multi'}

pair_list = []
for i in range(len(all_probe_keys)):
    for j in range(i + 1, len(all_probe_keys)):
        ka, kb = all_probe_keys[i], all_probe_keys[j]
        pair_list.append((f'{all_probe_names[i]} vs {all_probe_names[j]}', ka, kb))

full_cosines = {}
for pn, ka, kb in pair_list:
    fa, la, ma, ga, Ca, cwa = all_probe_configs[ka]
    fb, lb, mb, gb, Cb, cwb = all_probe_configs[kb]
    full_cosines[pn] = multiseed_layer_cosines(
        fa, la, ma, ga, Ca, fb, lb, mb, gb, Cb,
        cw_a=cwa, cw_b=cwb,
        precomputed_dirs_a=precomputed_dirs.get(ka),
        precomputed_dirs_b=precomputed_dirs.get(kb))
    involves_PI = not (ka in Z_ONLY and kb in Z_ONLY)
    mean_c = full_cosines[pn].mean(axis=0)
    peak = int(np.argmax(np.abs(mean_c[3:])) + 3)   # skip embedding layers 0-2
    tagp = 'P/I changed' if involves_PI else 'Z-only (cached)'
    print(f'{pn:>18s}: peak|cos| L{peak} = {mean_c[peak]:+.3f}   [{tagp}]')

n_pi = sum(1 for _, ka, kb in pair_list if not (ka in Z_ONLY and kb in Z_ONLY))
print(f'\nRecomputed {len(pair_list)} pairs ({n_pi} involve P/I, {len(pair_list)-n_pi} Z-only).')

# ---- cell 15 ----
# ── Verify the Z-only pairs are unchanged vs the original results.json ────────
_orig = json.load(open(op / 'results.json')) if (op / 'results.json').exists() else {}
_orig_pc = _orig.get('peak_cosines', {})
print('Z-only pair check (new peak cosine vs original results.json):')
for pn, ka, kb in pair_list:
    if ka in Z_ONLY and kb in Z_ONLY:
        mean_c = full_cosines[pn].mean(axis=0)
        peak = int(np.argmax(np.abs(mean_c[3:])) + 3)
        new = mean_c[peak]
        old = _orig_pc.get(pn, {}).get('cosine')
        msg = f'old={old:+.3f}' if old is not None else 'old=n/a'
        flag = '' if (old is None or abs(new - old) < 0.02) else '  <-- DIFF'
        print(f'  {pn:>16s}: new={new:+.3f}  {msg}{flag}')

# ---- cell 16 ----
# ── z-scores vs random-direction null (same null as the main notebook) ───────
hidden_dim = feats_P_S_r[0].shape[1]
null_stats = {li: random_cosine_null(hidden_dim, N_RANDOM_DIRS) for li in range(N_LAYERS)}
full_zscores = {}
for pn in full_cosines:
    cos_arr = full_cosines[pn]
    z = np.zeros_like(cos_arr)
    for li in range(cos_arr.shape[1]):
        z[:, li] = [cosine_zscore(c, null_stats[li]['mean'], null_stats[li]['std']) for c in cos_arr[:, li]]
    full_zscores[pn] = z
print('z-scores computed for all pairs.')

# ---- cell 17 ----
# ── Transfer matrix: P/I' -> response feats + new best layers; Z* -> cached ──
all_probe_data = {
    "I'_S":   (feats_I_S_r, labels_I_S,      mask_I_S, groups_I_S, best_layer_I_S_r, PROBE_C, None),
    "I'_C":   (feats_I_C_r, labels_I_C,      mask_I_C, groups_I_C, best_layer_I_C_r, PROBE_C, None),
    "P_S":    (feats_P_S_r, labels_P_S_lied, mask_P_S, groups_P_S, best_layer_P_S_r, PROBE_C, None),
    "P_C":    (feats_P_C_r, labels_P_C_lied, mask_P_C, groups_P_C, best_layer_P_C_r, PROBE_C, None),
    "Z":      (feats_Z,      labels_Z,      mask_Z,      None, best_layer_Z,      PROBE_C, None),
    "Z_mean": (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_mean, PROBE_C, None),
    "Z_tok":  (feats_Z_mean, labels_Z_mean, mask_Z_mean, None, best_layer_Z_tok,  PROBE_C, None),
    "Z_multi":(feats_Z_mean, labels_Z_mean, mask_Z_mean, None, multi_layers,      PROBE_C, None),
}
tok_train_data   = {"Z_tok":  (tok_feats_h, tok_feats_d, counts_h, counts_d)}
multi_train_data = {"Z_multi":(tok_feats_h, tok_feats_d, counts_h, counts_d, multi_layers)}

print(f'Computing {len(all_probe_names)}x{len(all_probe_names)} cross-transfer (response position for P/I)...')
full_transfer, preds = cross_transfer_matrix(all_probe_names, all_probe_data,
                                             tok_train_data=tok_train_data,
                                             multi_train_data=multi_train_data)
ci_full = bootstrap_ci_matrix(preds, len(all_probe_names), n_boot=1000)
mean_full, std_full = print_transfer_matrix(full_transfer, all_probe_names,
                                            'Cross-transfer AUC (P/I at response token)',
                                            ci_matrix=ci_full)

# ---- cell 18 ----
# ── Verify Z-only matrix block unchanged vs original results.json ────────────
_orig_m = _orig.get('cross_transfer_NxN', {})
if _orig_m.get('mean') and _orig_m.get('probes') == all_probe_names:
    om = np.array(_orig_m['mean'])
    zidx = [all_probe_names.index(n) for n in ['Z', 'Z_mean', 'Z_tok', 'Z_multi']]
    diff = np.abs(mean_full[np.ix_(zidx, zidx)] - om[np.ix_(zidx, zidx)])
    print(f'Z-only block max|new-old| = {diff.max():.4f} '
          f'({"OK reproduced" if diff.max() < 0.03 else "DIFF — investigate"})')
    # show the P/I rows/cols that moved
    print('\nCells that changed vs original (|delta|>0.03):')
    for i, ni in enumerate(all_probe_names):
        for j, nj in enumerate(all_probe_names):
            d = mean_full[i, j] - om[i, j]
            if abs(d) > 0.03:
                print(f'  {ni:>6s} -> {nj:<6s}: {om[i,j]:.3f} -> {mean_full[i,j]:.3f} ({d:+.3f})')
else:
    print('No comparable original matrix in results.json (skipping diff).')

# ---- cell 19 ----
# ── Headline comparison: I'/P dissociation & Z->P, pre-answer vs response ────
crit = [("I'_S","P_S"),("P_S","I'_S"),("I'_C","P_C"),("P_C","I'_C"),
        ("Z","P_S"),("Z","P_C"),("Z","I'_S"),("Z","I'_C"),("P_S","P_C")]
om = np.array(_orig_m['mean']) if _orig_m.get('mean') else None
print(f'{"train -> test":>16s}   {"pre-answer":>10s}   {"response":>9s}')
for tn, te in crit:
    i, j = all_probe_names.index(tn), all_probe_names.index(te)
    pre = f'{om[i,j]:.3f}' if om is not None else '  n/a'
    print(f'{tn+" -> "+te:>16s}   {pre:>10s}   {mean_full[i,j]:9.3f}')

# ======================================================================
# ## 4. Refresh OOD / Liars'-Bench with the response-position P/I' probes
# IT and Liars'-Bench are response-native → now position-consistent. SB and Game are pre-answer
# features → scored too but **flagged** as a position mismatch (re-extract those at the response
# token separately if you want them consistent).
# ======================================================================

# ---- cell 21 ----
# ── Re-score OOD/LB with the new response-position probes (CPU; cached feats) ──
def _load_layers(prefix):
    return {li: np.load(op / f'{prefix}_layer_{li:03d}.npy') for li in range(N_LAYERS)
            if (op / f'{prefix}_layer_{li:03d}.npy').exists()}

# (name, acts-prefix, labels-file, response-native?)
OOD_SETS = [
    ('IT',     'acts_OOD_IT',       'labels_OOD_IT.npy',     True),
    ('SB',     'acts_OOD_SB',       'labels_OOD_SB.npy',     False),
    ('LB_ID',  'acts_OOD_LB_ID',    'labels_OOD_LB_ID.npy',  True),
    ('LB_HPKR','acts_OOD_LB_HP_KR', 'labels_OOD_LB_HP_KR.npy', True),
]
NEW = [('P_S', clf_P_S_r, best_layer_P_S_r), ('P_C', clf_P_C_r, best_layer_P_C_r),
       ("I'_S", clf_I_S_r, best_layer_I_S_r), ("I'_C", clf_I_C_r, best_layer_I_C_r)]

ood_resp = {}
for ds, prefix, lblf, native in OOD_SETS:
    if not (op / lblf).exists() or not (op / f'{prefix}_layer_000.npy').exists():
        print(f'[{ds}] cached feats/labels not found — skip (run the OOD/LB cells first).')
        continue
    feats = _load_layers(prefix); labels = np.load(op / lblf)
    tagp = '' if native else '  [pre-answer feats — position MISMATCH for response probe]'
    print(f'\n=== {ds} ({len(labels)} ex){tagp} ===')
    for name, clf, layer in NEW:
        if layer not in feats:
            print(f'  {name}: layer {layer} missing'); continue
        res = ood.eval_probe_ood(clf, feats, labels, layer)
        ba = ood.balanced_accuracy_at_fpr(labels, res['probs'], target_fpr=0.01)
        ood_resp.setdefault(name, {})[ds] = {'auc': res['auc'], 'acc': res['acc'],
                                             'bal_acc_1pct': ba, 'position_match': native}
        print(f'  {name:>6s}: AUC={res["auc"]:.3f}  balAcc@1%={ba:.3f}')
# ── Game (re-extracted at response token; P-analogue) ───────────────────────
if df_game is not None and 'G' in acts3:
    g_clear = (~df_game['is_unclear']).to_numpy()
    g_feats = {li: acts3['G']['resp'][li][g_clear] for li in range(N_LAYERS)}
    g_y = df_game['actually_lied'].to_numpy()[g_clear]
    print(f'=== Game ({int(g_clear.sum())} clear; re-extracted at response token) ===')
    for name, clf, layer in NEW:
        res = ood.eval_probe_ood(clf, g_feats, g_y, layer)
        ba = ood.balanced_accuracy_at_fpr(g_y, res['probs'], target_fpr=0.01)
        ood_resp.setdefault(name, {})['G'] = {'auc': res['auc'], 'acc': res['acc'],
                                              'bal_acc_1pct': ba, 'position_match': True}
        print(f'  {name:>6s}: AUC={res["auc"]:.3f}  balAcc@1%={ba:.3f}')

print('\nOOD/LB refresh complete (response-position P/I probes).')


# ======================================================================
# ## 5. Save outputs (originals untouched)
# ======================================================================

# ---- cell 23 ----
# ── Save new probes + results (separate files; pre-answer artifacts untouched) ──
probe_state_resp = {}
for tag in ['P_S', 'P_C', 'I_S', 'I_C']:
    nbp = new_probes[tag]
    probe_state_resp[f'clf_{tag}_resp']        = nbp['clf']
    probe_state_resp[f'w_{tag}_resp']          = nbp['w']
    probe_state_resp[f'dirs_{tag}_resp']       = nbp['dirs']
    probe_state_resp[f'best_layer_{tag}_resp'] = nbp['best_layer']
    probe_state_resp[f'aucs_{tag}_resp']       = nbp['aucs']
with open(op / 'probe_state_resp.pkl', 'wb') as f:
    pickle.dump(probe_state_resp, f)

results_resp = {
    'position': 'last_response_token',
    'best_layers_resp': {t: int(new_probes[t]['best_layer']) for t in ['P_S','P_C','I_S','I_C']},
    'within_dist_aucs_resp': {t: float(new_probes[t]['aucs'][:, new_probes[t]['best_layer']].mean())
                              for t in ['P_S','P_C','I_S','I_C']},
    'cross_transfer_NxN': {'probes': all_probe_names,
                           'mean': mean_full.tolist(), 'std': std_full.tolist()},
    'peak_cosines': {pn: {'cosine': float(full_cosines[pn].mean(0)[int(np.argmax(np.abs(full_cosines[pn].mean(0)[3:]))+3)]),
                          'zscore': float(full_zscores[pn].mean(0)[int(np.argmax(np.abs(full_cosines[pn].mean(0)[3:]))+3)])}
                     for pn, _, _ in pair_list},
    'ood_resp': ood_resp,
}
with open(op / 'results_resp.json', 'w') as f:
    json.dump(results_resp, f, indent=2)
print('Saved: probe_state_resp.pkl, results_resp.json')
print('Three-position activation caches: acts_<P_S|P_C|I_S|I_C>_<pre|resp|respmean>_layer_*.npy')

# ---- cell 24 ----
# ── Figure: cross-transfer heatmap (response position) ───────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mean_full, cmap='RdYlGn', vmin=0.3, vmax=1.0)
for i in range(len(all_probe_names)):
    for j in range(len(all_probe_names)):
        ax.text(j, i, f'{mean_full[i,j]:.2f}', ha='center', va='center', fontsize=7,
                fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(len(all_probe_names))); ax.set_yticks(range(len(all_probe_names)))
ax.set_xticklabels(all_probe_names, fontsize=8, rotation=45, ha='right')
ax.set_yticklabels(all_probe_names, fontsize=8)
ax.set_xlabel('Test on'); ax.set_ylabel('Train on')
ax.set_title("Cross-transfer AUC — P/I' at response token")
plt.colorbar(im, ax=ax, shrink=0.8); plt.tight_layout()
plt.savefig(op / 'experiment_A_resp.png', bbox_inches='tight'); plt.show()
print('Saved experiment_A_resp.png')
