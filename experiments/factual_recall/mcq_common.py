#!/usr/bin/env python
"""Shared GPU helpers for the MCQ factual-recall experiment (steps 2-3, 5-logit).

Mirrors the conventions in diag_extract.py: Llama-3.1-8B 4-bit, constrained
single-token decode, and hidden-state extraction at a given position over all
33 layers. Kept separate from diag_extract.py because importing that module
loads the model at top level; here loading is explicit via load_model().
"""
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm.auto import tqdm

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'
N_LAYERS = 33  # 32 hidden + embedding
LETTERS = ['A', 'B', 'C', 'D']
HF_TOKEN = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')


def load_model():
    print(f'Loading {MODEL_ID} (4-bit)...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map='auto', token=HF_TOKEN)
    model.eval()
    print('Model loaded.', flush=True)
    return tokenizer, model


def letter_variants(tokenizer):
    """{letter: sorted token-id list} over the 'A' / ' A' surface variants.

    Used both to constrain decoding to the four letters and to aggregate
    per-letter logits for the output-confidence baseline (logsumexp over variants).
    """
    out = {}
    for L in LETTERS:
        ids = set()
        for s in (L, ' ' + L):
            ids.update(tokenizer(s, add_special_tokens=False).input_ids)
        out[L] = sorted(ids)
    return out


def canonical_letter_token(tokenizer, L):
    """The single token id used to materialise a chosen letter in the transcript."""
    ids = tokenizer(L, add_special_tokens=False).input_ids
    return ids[-1]


def build_pre_ids(tokenizer, system, user):
    """Prompt + generation prompt -> 1-D long tensor (assistant turn about to start)."""
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': system})
    msgs.append({'role': 'user', 'content': user})
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if isinstance(ids[0], list):
        ids = ids[0]
    return torch.tensor(ids, dtype=torch.long)


def normalize_letter(text):
    t = text.strip().lstrip(' .,;:!?"\'\n').upper()
    return t[0] if (t and t[0] in LETTERS) else 'unclear'


def sample_letters(model, tokenizer, pre_ids_list, allowed_ids, n_samples,
                   temperature, bs=16, tag=''):
    """Constrained single-letter sampling.

    Returns a list (len == len(pre_ids_list)) of lists of n_samples letters
    (each 'A'/'B'/'C'/'D' or 'unclear').
    """
    allowed = list(allowed_ids)

    def prefix_fn(_bid, _ids):
        return allowed

    tokenizer.padding_side = 'left'
    do_sample = temperature is not None and temperature > 0
    results = [[] for _ in pre_ids_list]
    for s in tqdm(range(0, len(pre_ids_list), bs), desc=f'sample {tag}'):
        batch = pre_ids_list[s:s + bs]
        ml = max(t.shape[0] for t in batch)
        ids = torch.full((len(batch), ml), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, t in enumerate(batch):
            off = ml - t.shape[0]
            ids[i, off:] = t
            mask[i, off:] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        gen_kwargs = dict(input_ids=ids, attention_mask=mask, max_new_tokens=1,
                          pad_token_id=tokenizer.pad_token_id,
                          prefix_allowed_tokens_fn=prefix_fn,
                          num_return_sequences=n_samples)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=float(temperature))
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        out = out.view(len(batch), n_samples, -1)
        for i in range(len(batch)):
            for k in range(n_samples):
                tokid = out[i, k, ml:][0].item()
                results[s + i].append(normalize_letter(tokenizer.decode([tokid])))
    return results


def extract_last(model, tokenizer, input_ids_list, positions, batch_size=8, tag=''):
    """Hidden state at positions[i] for each seq -> {layer: (n, d) float32}.

    Direct port of diag_extract.extract_last (right padding).
    """
    tokenizer.padding_side = 'right'
    out_v = {}
    with torch.no_grad():
        for s in tqdm(range(0, len(input_ids_list), batch_size), desc=f'last {tag}'):
            batch = input_ids_list[s:s + batch_size]
            pos = positions[s:s + batch_size]
            ml = max(t.shape[0] for t in batch)
            ids = torch.full((len(batch), ml), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, t in enumerate(batch):
                ids[i, :t.shape[0]] = t
                mask[i, :t.shape[0]] = 1
            ids, mask = ids.to(model.device), mask.to(model.device)
            pt = torch.tensor(pos, dtype=torch.long, device=model.device)
            o = model(input_ids=ids, attention_mask=mask,
                      output_hidden_states=True, use_cache=False)
            bi = torch.arange(len(batch), device=model.device)
            for li, lh in enumerate(o.hidden_states):
                out_v.setdefault(li, []).extend(
                    list(lh[bi, pt, :].detach().float().cpu().numpy()))
    return {li: np.stack(v) for li, v in out_v.items()}


def save_feats(diag_dir, ds, pos, feats):
    Path(diag_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(diag_dir) / f'feats_{ds}_{pos}.npz',
             **{f'L{li:02d}': feats[li] for li in feats})
    print(f'  saved feats_{ds}_{pos}.npz  ({len(feats)} layers, '
          f'n={feats[0].shape[0]})', flush=True)
