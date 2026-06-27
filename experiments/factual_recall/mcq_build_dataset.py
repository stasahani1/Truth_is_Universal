#!/usr/bin/env python
"""Step 1 (CPU): build the factual-recall MCQ pool with a balanced correct-letter.

Pulls 4-option multiple-choice items from a standard benchmark (MMLU and/or ARC),
freezes ONE scrambled option arrangement per item, records the correct letter, and
**balances the correct letter across A/B/C/D over the whole set** (safeguard (a) of
the design — so letter-identity and the lie/truth label cannot correlate globally).

Output: my_datasets/mcq_facts.json — one row per item:
  {fact_id, source, subject, question,
   options: {A,B,C,D}, correct_letter,
   user_prompt}     # the fixed MCQ user message (question + lettered options + instruction)

The same `user_prompt` is reused verbatim by mcq_verify_known.py (no pressure) and
mcq_pressure_rollouts.py (pressure prepended as a system message), so lie and truth
rollouts share identical prompt text by construction.

Run:
  python experiments/factual_recall/mcq_build_dataset.py \
      --source mmlu --max-items 1500 --seed 0
"""
import os, sys, json, argparse, random
from pathlib import Path
from collections import Counter

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parents[2]))
DATASETS_DIR = Path(REPO_DIR) / 'my_datasets'

LETTERS = ['A', 'B', 'C', 'D']
INSTRUCTION = "Answer with a single letter (A, B, C, or D) and nothing else."


def _load_mmlu(max_rows):
    """MMLU: question, choices (4), answer (int 0-3), subject. Use the test split."""
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'all', split='test')
    rows = []
    for r in ds:
        ch = r['choices']
        if len(ch) != 4 or any(not str(c).strip() for c in ch):
            continue
        rows.append({'source': 'mmlu', 'subject': r.get('subject', ''),
                     'question': r['question'].strip(),
                     'choices': [str(c).strip() for c in ch],
                     'answer_idx': int(r['answer'])})
    return rows


def _load_arc(max_rows):
    """ARC-Challenge: keep only 4-option items (some have 3 or 5)."""
    from datasets import load_dataset
    ds = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test')
    rows = []
    for r in ds:
        texts = [str(t).strip() for t in r['choices']['text']]
        labels = list(r['choices']['label'])
        key = r['answerKey']
        if len(texts) != 4 or key not in labels or any(not t for t in texts):
            continue
        rows.append({'source': 'arc', 'subject': 'arc_challenge',
                     'question': r['question'].strip(),
                     'choices': texts,
                     'answer_idx': labels.index(key)})
    return rows


def build(rows, seed):
    """Freeze a scrambled arrangement per item with a globally balanced correct letter.

    For each item we place the correct option on a *target* letter (chosen round-robin
    to equalise the A/B/C/D marginal exactly) and shuffle the three distractors into the
    remaining slots. This gives an exactly balanced correct-letter distribution while
    still randomising option order per item.
    """
    rng = random.Random(seed)
    rng.shuffle(rows)
    # round-robin target letters → exact balance (±1 across the four letters)
    targets = [LETTERS[i % 4] for i in range(len(rows))]
    rng.shuffle(targets)

    out = []
    for i, (r, tgt) in enumerate(zip(rows, targets)):
        correct = r['choices'][r['answer_idx']]
        distractors = [c for j, c in enumerate(r['choices']) if j != r['answer_idx']]
        rng.shuffle(distractors)
        slots = {tgt: correct}
        rem = [L for L in LETTERS if L != tgt]
        for L, d in zip(rem, distractors):
            slots[L] = d
        options = {L: slots[L] for L in LETTERS}
        opt_block = "\n".join(f"{L}. {options[L]}" for L in LETTERS)
        user_prompt = f"{r['question']}\n\n{opt_block}\n\n{INSTRUCTION}"
        out.append({
            'fact_id': f"{r['source']}_{i:05d}",
            'source': r['source'],
            'subject': r['subject'],
            'question': r['question'],
            'options': options,
            'correct_letter': tgt,
            'user_prompt': user_prompt,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['mmlu', 'arc', 'both'], default='mmlu')
    ap.add_argument('--max-items', type=int, default=1500,
                    help='cap on items pulled (0 = no cap)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(DATASETS_DIR / 'mcq_facts.json'))
    args = ap.parse_args()

    rows = []
    if args.source in ('mmlu', 'both'):
        rows += _load_mmlu(args.max_items)
    if args.source in ('arc', 'both'):
        rows += _load_arc(args.max_items)
    # shuffle the FULL pool before capping so the sample spans all subjects
    random.Random(args.seed).shuffle(rows)
    if args.max_items and len(rows) > args.max_items:
        rows = rows[:args.max_items]
    print(f'Loaded {len(rows)} raw 4-option items from {args.source} '
          f'(sampled across full pool).', flush=True)

    items = build(rows, args.seed)

    marg = Counter(it['correct_letter'] for it in items)
    print('correct-letter marginal:', dict(sorted(marg.items())), flush=True)
    # exact-balance sanity: counts differ by at most 1
    assert max(marg.values()) - min(marg.values()) <= 1, f'unbalanced: {marg}'

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(items, open(args.out, 'w'), indent=2)
    print(f'Wrote {len(items)} items -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
