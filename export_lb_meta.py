#!/usr/bin/env python
"""Export Liars'-Bench per-row metadata (transcript text, label, model, dataset)
in the SAME deterministic order as the cached activations (load_liars_bench +
_balanced_cap seed=42), so TF-IDF + generator-subset checks can run on CPU/local.
Env: HF_TOKEN, OUTPUT_DIR (default ~/output)."""
import os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ood_eval_utils as ood
from datasets import load_dataset

op = Path(os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output')))
hf = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
SUB = {'ID': 'instructed_deception', 'HP_KR': 'harm_pressure_kr'}

def cap_idx(labs, c=1000, seed=42):
    labs = np.asarray(labs)
    if len(labs) <= c:
        return list(range(len(labs)))
    rng = np.random.RandomState(seed); per = c // 2; idx = []
    for cls in (0, 1):
        ci = np.where(labs == cls)[0]
        if len(ci) > per:
            ci = rng.choice(ci, per, replace=False)
        idx.extend(ci.tolist())
    return sorted(idx)

for key, sub in SUB.items():
    cfg = ood._LB_SUBSETS[sub]
    ds = load_dataset("Cadenza-Labs/liars-bench",
                      data_files={'test': f"{cfg}/test-*.parquet"}, split='test', token=hf)
    rows = []
    for r in ds:
        lab = ood._lb_extract_label(r); tr = ood._lb_extract_transcript(r)
        rows.append({'text': ' '.join(str(m.get('content', '')) for m in tr),
                     'label': int(lab), 'model': r.get('model'), 'dataset': r.get('dataset')})
    labs = np.array([x['label'] for x in rows])
    meta = [rows[i] for i in cap_idx(labs)]
    json.dump(meta, open(op / f'lb_meta_{key}.json', 'w'))
    from collections import Counter
    print(f'[{key}] n={len(meta)}  models={dict(Counter(m["model"] for m in meta))}')
    print(f'        datasets={dict(Counter(m["dataset"] for m in meta))}')
print('done')
