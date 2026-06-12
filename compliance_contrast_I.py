#!/usr/bin/env python
"""
compliance_contrast_I.py  (item 3, GPU)

Compliance-contrast instructed-deception probe: among DECEPTIVE-instructed prompts,
contrast those that actually LIED (complied) vs those that stayed TRUTHFUL despite
the instruction. Label = actually_lied within the deceptive-instructed subset, clear
rows only. This isolates the ACT of lying under instruction from the instruction TEXT
(the original I' probe is an instruction detector), making the P<->I' transfer cells
interpretable for the first time.

Reuses the cached answers in inference_I_bundle.pkl (no re-inference) but needs a NEW
forward pass (the deceptive-but-truthful rows were never extracted). Extracts at the
response token, retrains, recomputes P<->I'comp transfer vs the instruction-detector I'.

Env: OUTPUT_DIR (default ~/output), HF_TOKEN (gated model).
"""
import matplotlib; matplotlib.use('Agg')
import os, json, pickle, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from probe_utils import (multiseed_best_layer, fit_final_probe,
                         cross_transfer_matrix, print_transfer_matrix, bootstrap_ci_matrix)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
op = Path(OUTPUT_DIR)
N_LAYERS, N_SEEDS, PROBE_C, FORWARD_BATCH_SIZE = 32, 10, 1.0, 8
LAYERS = list(range(N_LAYERS))
MODEL_KEY = 'llama'
MODEL_MAP = {'llama': 'meta-llama/Llama-3.1-8B-Instruct', 'qwen': 'Qwen/Qwen2.5-14B-Instruct'}
LOAD_IN_4BIT = True
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')

# ── model + tokenizer ─────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_id = MODEL_MAP[MODEL_KEY]
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
_mk = {'token': hf_token}
if LOAD_IN_4BIT:
    if not torch.cuda.is_available():
        raise RuntimeError('LOAD_IN_4BIT=True requires a GPU.')
    _mk['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    _mk['device_map'] = 'auto'
elif torch.cuda.is_available():
    _mk['device_map'] = 'auto'
    _mk['torch_dtype'] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    _mk['torch_dtype'] = torch.float32
model = AutoModelForCausalLM.from_pretrained(model_id, **_mk); model.eval()
print('Model loaded.')

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


# ── EOT ids (Apollo last-content-token convention) ───────────────────────────
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

def reextract_resp(df, sys_col, usr_col, ans_col, tag, batch_size=2):
    full_ids, resp_starts = [], []
    for _, r in df.iterrows():
        prompt = build_prompt_ids(str(r[sys_col]), str(r[usr_col]))
        ans = str(r[ans_col]).strip()
        aid = tokenizer(ans, add_special_tokens=False).input_ids if ans and ans.lower() != 'unclear' else []
        full = torch.cat([prompt, torch.tensor(aid, dtype=torch.long)]) if aid else prompt
        full_ids.append(full); resp_starts.append(prompt.shape[0] - 1)
    raw = extract_activations_all_response_tokens(full_ids, resp_starts, batch_size=batch_size, tag=tag)
    resp = {li: np.stack([a[-1] for a in raw[li]]) for li in LAYERS}
    for li in LAYERS:
        np.save(op / f'acts_{tag}_resp_layer_{li:03d}.npy', resp[li])
    return resp

# ── build deceptive-instructed clear subsets (label = actually_lied) ─────────
ib = pickle.load(open(op / 'inference_I_bundle.pkl', 'rb'))
def dec_subset(df):
    d = df[(df['instruction_type'] == 'deceptive') & (~df['is_unclear'])].copy()
    return d
dec = {'Icomp_S': dec_subset(ib['df_I_sales']), 'Icomp_C': dec_subset(ib['df_I_combined'])}

print('\n=== compliance-contrast I\' class balance (deceptive-instructed, clear) ===')
for tag, d in dec.items():
    nl = int((d['actually_lied'] == 1).sum()); nt = int((d['actually_lied'] == 0).sum())
    print(f'  {tag}: lied(complied)={nl}  truthful-despite-instruction={nt}  total={len(d)}')
    if min(nl, nt) < 15:
        print(f'    WARNING: minority class = {min(nl,nt)} — probe UNDERPOWERED, interpret with care.')

# ── extract + retrain ────────────────────────────────────────────────────────
feats_cc, labels_cc, groups_cc = {}, {}, {}
for tag, d in dec.items():
    print(f'[{tag}] extracting response-token activations ({len(d)} rows)...')
    feats_cc[tag] = reextract_resp(d, 'system_prompt', 'user_question', 'raw_response', tag)
    labels_cc[tag] = d['actually_lied'].to_numpy().astype(int)
    groups_cc[tag] = d['scenario_id'].to_numpy()

del model; torch.cuda.empty_cache(); gc.collect(); print('Model freed.')

bl_cc = {}
for tag in dec:
    m = np.ones(len(labels_cc[tag]), bool)
    print(f'=== retrain {tag} ===')
    bl_cc[tag], _ = multiseed_best_layer(feats_cc[tag], labels_cc[tag], m,
                                         n_seeds=N_SEEDS, C=PROBE_C, groups=groups_cc[tag])

# ── transfer: I'comp <-> P, alongside the instruction-detector I' ────────────
PS = pickle.load(open(op / 'probe_state.pkl', 'rb'))
labels_P_S_lied, mask_P_S, groups_P_S = PS['labels_P_S_lied'], PS['mask_P_S'], PS['groups_P_S']
labels_P_C_lied, mask_P_C, groups_P_C = PS['labels_P_C_lied'], PS['mask_P_C'], PS['groups_P_C']
labels_I_S, mask_I_S, groups_I_S = PS['labels_I_S'], PS['mask_I_S'], PS['groups_I_S']
labels_I_C, mask_I_C, groups_I_C = PS['labels_I_C'], PS['mask_I_C'], PS['groups_I_C']

def load(tag): return {li: np.load(op / f'acts_{tag}_resp_layer_{li:03d}.npy') for li in LAYERS
                       if (op / f'acts_{tag}_resp_layer_{li:03d}.npy').exists()}
feats_P_S, feats_P_C = load('P_S'), load('P_C')
feats_I_S, feats_I_C = load('I_S'), load('I_C')

m_cc_S = np.ones(len(labels_cc['Icomp_S']), bool)
m_cc_C = np.ones(len(labels_cc['Icomp_C']), bool)
bl_PS, _ = multiseed_best_layer(feats_P_S, labels_P_S_lied, mask_P_S, n_seeds=N_SEEDS, C=PROBE_C, groups=groups_P_S)
bl_PC, _ = multiseed_best_layer(feats_P_C, labels_P_C_lied, mask_P_C, n_seeds=N_SEEDS, C=PROBE_C, groups=groups_P_C)
bl_IS, _ = multiseed_best_layer(feats_I_S, labels_I_S, mask_I_S, n_seeds=N_SEEDS, C=PROBE_C, groups=groups_I_S)
bl_IC, _ = multiseed_best_layer(feats_I_C, labels_I_C, mask_I_C, n_seeds=N_SEEDS, C=PROBE_C, groups=groups_I_C)

names = ["I'comp_S", "I'comp_C", "I'_S", "I'_C", "P_S", "P_C"]
data = {
    "I'comp_S": (feats_cc['Icomp_S'], labels_cc['Icomp_S'], m_cc_S, groups_cc['Icomp_S'], bl_cc['Icomp_S'], PROBE_C, None),
    "I'comp_C": (feats_cc['Icomp_C'], labels_cc['Icomp_C'], m_cc_C, groups_cc['Icomp_C'], bl_cc['Icomp_C'], PROBE_C, None),
    "I'_S": (feats_I_S, labels_I_S, mask_I_S, groups_I_S, bl_IS, PROBE_C, None),
    "I'_C": (feats_I_C, labels_I_C, mask_I_C, groups_I_C, bl_IC, PROBE_C, None),
    "P_S":  (feats_P_S, labels_P_S_lied, mask_P_S, groups_P_S, bl_PS, PROBE_C, None),
    "P_C":  (feats_P_C, labels_P_C_lied, mask_P_C, groups_P_C, bl_PC, PROBE_C, None),
}
full, _ = cross_transfer_matrix(names, data)
mean = np.nanmean(full, 2)
print_transfer_matrix(full, names, "I'comp / I' / P cross-transfer (response token)")

print('\n--- Compliance-contrast vs instruction-detector: does P->I survive? ---')
for tn in ['P_S', 'P_C']:
    for dom in ['S', 'C']:
        i = names.index(tn); j_c = names.index(f"I'comp_{dom}"); j_i = names.index(f"I'_{dom}")
        print(f'  {tn} -> I\'comp_{dom} = {mean[i,j_c]:.3f}   (vs instruction-detector '
              f'{tn} -> I\'_{dom} = {mean[i,j_i]:.3f})')
for dom in ['S', 'C']:
    for te in ['P_S', 'P_C']:
        i_c = names.index(f"I'comp_{dom}"); i_i = names.index(f"I'_{dom}"); j = names.index(te)
        print(f'  I\'comp_{dom} -> {te} = {mean[i_c,j]:.3f}   (vs I\'_{dom} -> {te} = {mean[i_i,j]:.3f})')

json.dump({'names': names, 'best_layers': {**{k: int(v) for k, v in bl_cc.items()},
           'P_S': int(bl_PS), 'P_C': int(bl_PC), "I'_S": int(bl_IS), "I'_C": int(bl_IC)},
           'mean': mean.tolist(),
           'class_balance': {t: {'lied': int((dec[t]['actually_lied']==1).sum()),
                                 'truthful': int((dec[t]['actually_lied']==0).sum())} for t in dec}},
          open(op / 'compliance_contrast_I_results.json', 'w'), indent=2)
print(f'\nSaved {op / "compliance_contrast_I_results.json"}')
