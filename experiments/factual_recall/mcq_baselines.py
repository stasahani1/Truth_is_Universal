#!/usr/bin/env python
"""Step 5: baselines beside the MCQ probe.

(1) TF-IDF on the prompt — CPU, always. Lie and truth rollouts of a fact share identical
    prompt text, so grouped TF-IDF is **chance by construction**. We run it anyway and
    assert AUROC ~= 0.50: that equality is the headline contrast (text provably cannot
    carry the label, yet the activations do).

(2) Output-confidence — needs the model (GPU). Under the same fixed pressure prompt,
    compute the 4-way letter distribution per fact (deterministic), take p_chosen for each
    rollout's picked letter, and score raw AUROC of -p_chosen vs the lie label (the
    matrix_baseline.py convention, generalised 2-way -> 4-way).

Reads : $OUTPUT_DIR/box_results/mcq_rollouts.json, my_datasets/mcq_verified.json
Writes: $OUTPUT_DIR/box_results/mcq_baselines.json

Run (CPU, TF-IDF only):
  python experiments/factual_recall/mcq_baselines.py
Run (box, with output-confidence):
  HF_TOKEN=... OUTPUT_DIR=~/output python experiments/factual_recall/mcq_baselines.py --with-logits
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parents[2]))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def tfidf_grouped(texts, y, groups, seeds=5):
    """Mirror matrix_baseline.tfidf_grouped: grouped CV, LR, raw AUROC."""
    y = np.asarray(y)
    aucs = []
    for sd in range(seeds):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=sd)
                      .split(np.arange(len(y)), y, groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=2000))
        p.fit([texts[i] for i in tr], y[tr])
        aucs.append(roc_auc_score(y[te], p.predict_proba([texts[i] for i in te])[:, 1]))
    return round(float(np.mean(aucs)), 3) if aucs else None


def output_confidence(rollouts, prompt_by_fact, variant):
    """4-way letter p_chosen under the fixed pressure prompt; raw AUROC of -p_chosen."""
    import torch
    import mcq_common as mc
    from mcq_pressure_rollouts import PRESSURE_VARIANTS
    tokenizer, model = mc.load_model()
    variants = mc.letter_variants(tokenizer)
    system = PRESSURE_VARIANTS[variant]

    facts = sorted({r['fact_id'] for r in rollouts})
    pre_ids = [mc.build_pre_ids(tokenizer, system, prompt_by_fact[f]) for f in facts]
    p_by_fact = {}
    bs = 12
    tokenizer.padding_side = 'left'
    for s in range(0, len(pre_ids), bs):
        batch = pre_ids[s:s + bs]
        ml = max(t.shape[0] for t in batch)
        ids = torch.full((len(batch), ml), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, t in enumerate(batch):
            off = ml - t.shape[0]
            ids[i, off:] = t
            mask[i, off:] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
        per_letter = torch.stack([torch.logsumexp(lg[:, variants[L]], -1) for L in mc.LETTERS], -1)
        pr = torch.softmax(per_letter, -1)  # (b, 4)
        for i in range(len(batch)):
            p_by_fact[facts[s + i]] = {L: float(pr[i, j]) for j, L in enumerate(mc.LETTERS)}

    y = np.array([r['label'] for r in rollouts])
    pchosen = np.array([p_by_fact[r['fact_id']][r['picked_letter']] for r in rollouts])
    auc = float(roc_auc_score(y, -pchosen)) if len(np.unique(y)) > 1 else None
    return {'neg_pchosen_auc': round(auc, 3) if auc is not None else None,
            'n': int(len(y)), 'variant': variant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rollouts', default=str(Path(OUTPUT_DIR) / 'box_results' / 'mcq_rollouts.json'))
    ap.add_argument('--verified', default=str(Path(REPO_DIR) / 'my_datasets' / 'mcq_verified.json'))
    ap.add_argument('--with-logits', action='store_true', help='also run the GPU output-confidence baseline')
    ap.add_argument('--out', default=str(Path(OUTPUT_DIR) / 'box_results' / 'mcq_baselines.json'))
    args = ap.parse_args()

    summary = json.load(open(args.rollouts))
    rollouts = summary['rollouts']
    prompt_by_fact = {f['fact_id']: f['user_prompt'] for f in json.load(open(args.verified))}

    y = [r['label'] for r in rollouts]
    groups = [r['fact_id'] for r in rollouts]
    texts = [prompt_by_fact[r['fact_id']] for r in rollouts]   # identical within a fact

    tfidf_auc = tfidf_grouped(texts, y, groups)
    out = {'prompt_tfidf_auc': tfidf_auc,
           'tfidf_note': 'chance by construction (lie/truth share prompt text); expect ~0.50',
           'n_rollouts': len(y)}
    print(f'prompt TF-IDF AUROC = {tfidf_auc}  (expected ~0.50)', flush=True)
    if tfidf_auc is not None and abs(tfidf_auc - 0.5) > 0.1:
        print('  WARNING: TF-IDF is not near chance — prompt text may be leaking the label.',
              flush=True)

    if args.with_logits:
        out['output_confidence'] = output_confidence(rollouts, prompt_by_fact,
                                                     summary.get('variant', 'strong'))
        print(f'output-confidence: {out["output_confidence"]}', flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2)
    print(f'Wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
