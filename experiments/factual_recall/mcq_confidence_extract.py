#!/usr/bin/env python
"""Extract answer-confidence features for the confidence-vs-lying analysis (GPU).

MCQ : per fact, the 4-way letter logits + softmax probs under the FIXED pressure prompt
      (deterministic). -> mcq_conf_MCQ.json  {fact_id: {logits:{A..D}, probs:{A..D}}}
P_S/P_C: per row, the Yes/No 2-way logits + p_chosen of the emitted answer + lied label +
      pressure mask (same row order as meta_<ds>.npz from mcq_xfer_extract).
      -> mcq_conf_PS.json / mcq_conf_PC.json  {pchosen, lied, mask}

Run (box): HF_TOKEN=... OUTPUT_DIR=~/output \
  python experiments/factual_recall/mcq_confidence_extract.py --datasets MCQ P_S P_C
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parents[2]))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcq_common as mc
from mcq_pressure_rollouts import PRESSURE_VARIANTS

PRESS_PATH = {'P_S': 'my_datasets/scenarios_pressure.json',
              'P_C': 'my_datasets/combined_dataset.json'}


def last_logits(tok, model, pre_ids, bs=12):
    """Logits at the final (pre-answer) position for each prompt -> (n, vocab) on CPU float."""
    tok.padding_side = 'left'
    out = []
    for s in range(0, len(pre_ids), bs):
        batch = pre_ids[s:s + bs]
        ml = max(t.shape[0] for t in batch)
        ids = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, t in enumerate(batch):
            off = ml - t.shape[0]; ids[i, off:] = t; mask[i, off:] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float().cpu()
        out.append(lg)
    return torch.cat(out, 0)


def do_mcq(tok, model, variants, system, res):
    facts = json.load(open(Path(REPO_DIR) / 'my_datasets' / 'mcq_verified.json'))
    pre_ids = [mc.build_pre_ids(tok, system, f['user_prompt']) for f in facts]
    lg = last_logits(tok, model, pre_ids)
    out = {}
    for j, f in enumerate(facts):
        per = {L: float(torch.logsumexp(lg[j, variants[L]], -1)) for L in mc.LETTERS}
        pr = torch.softmax(torch.tensor([per[L] for L in mc.LETTERS]), -1)
        out[f['fact_id']] = {'logits': per,
                             'probs': {L: float(pr[k]) for k, L in enumerate(mc.LETTERS)}}
    json.dump(out, open(res / 'mcq_conf_MCQ.json', 'w'))
    print(f'[MCQ] saved confidence for {len(out)} facts', flush=True)


def do_pressure(tok, model, ds, YES, NO, res):
    df = pd.DataFrame(json.load(open(Path(REPO_DIR) / PRESS_PATH[ds])))
    df['ground_truth'] = df['ground_truth'].astype(str).str.strip().str.lower()
    is_dec = (df['pressure_level'] != 'level_1').to_numpy()
    pre_ids = [mc.build_pre_ids(tok, str(s), str(u))
               for s, u in zip(df['role_context'], df['user_question'])]
    lg = last_logits(tok, model, pre_ids)
    ly = torch.logsumexp(lg[:, YES], -1)
    ln = torch.logsumexp(lg[:, NO], -1)
    pr = torch.softmax(torch.stack([ly, ln], -1), -1)  # (n,2) [p_yes, p_no]
    pyes = pr[:, 0].numpy()
    gen = np.where(pyes >= 0.5, 'yes', 'no')
    pchosen = np.where(pyes >= 0.5, pyes, 1 - pyes)
    lied = (gen != df['ground_truth'].to_numpy()).astype(int)
    mask = is_dec  # pressure rows
    out = {'pchosen': pchosen.tolist(), 'lied': lied.tolist(), 'mask': mask.tolist(),
           'p_yes': pyes.tolist()}
    json.dump(out, open(res / f'mcq_conf_{ds.replace("_","")}.json', 'w'))
    print(f'[{ds}] n={len(lied)} pressure={int(mask.sum())} '
          f'lied(pressure)={int(lied[mask].sum())}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['MCQ', 'P_S', 'P_C'])
    ap.add_argument('--variant', default='strong')
    args = ap.parse_args()
    res = Path(OUTPUT_DIR) / 'box_results'; res.mkdir(parents=True, exist_ok=True)
    tok, model = mc.load_model()
    variants = mc.letter_variants(tok)
    system = PRESSURE_VARIANTS[args.variant]
    YES = sorted({i for s in ('Yes', ' Yes', 'yes', ' yes') for i in tok(s, add_special_tokens=False).input_ids})
    NO = sorted({i for s in ('No', ' No', 'no', ' no') for i in tok(s, add_special_tokens=False).input_ids})
    for ds in args.datasets:
        print(f'\n===== confidence: {ds} =====', flush=True)
        if ds == 'MCQ':
            do_mcq(tok, model, variants, system, res)
        else:
            do_pressure(tok, model, ds, YES, NO, res)
    print('\nCONFIDENCE EXTRACTION DONE', flush=True)


if __name__ == '__main__':
    main()
