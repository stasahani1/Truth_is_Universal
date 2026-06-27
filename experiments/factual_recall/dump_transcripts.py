#!/usr/bin/env python
"""Dump exact chat-template transcripts for a few both-label MCQ facts.

Renders the literal prompt string Llama-3.1 receives (special tokens included), then
shows a TRUTH completion (correct letter) and a LIE completion (a wrong letter the model
actually sampled) for the same fact — making explicit that the only thing differing
between a truth and a lie rollout is the model's own answer token.

Run (box, tokenizer only — no GPU):
  HF_TOKEN=... OUTPUT_DIR=~/output python experiments/factual_recall/dump_transcripts.py --n 6
"""
import os, sys, json, argparse
from pathlib import Path
from collections import Counter

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parents[2]))
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcq_pressure_rollouts import PRESSURE_VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--variant', default='strong')
    ap.add_argument('--rollouts', default=str(Path(OUTPUT_DIR) / 'box_results' / 'mcq_rollouts.json'))
    ap.add_argument('--verified', default=str(Path(REPO_DIR) / 'my_datasets' / 'mcq_verified.json'))
    ap.add_argument('--out', default=str(Path(OUTPUT_DIR) / 'box_results' / 'mcq_example_transcripts.txt'))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct',
                                        token=os.environ.get('HF_TOKEN'))
    system = PRESSURE_VARIANTS[args.variant]
    verified = {f['fact_id']: f for f in json.load(open(args.verified))}
    roll = json.load(open(args.rollouts))

    byfact = {}
    for r in roll['rollouts']:
        byfact.setdefault(r['fact_id'], []).append(r['picked_letter'])

    lines, seen, shown = [], set(), 0
    lines.append('=' * 90)
    lines.append('FIXED PRESSURE SYSTEM PROMPT (variant=%s)' % args.variant)
    lines.append('=' * 90)
    lines.append(system)
    lines.append('')

    for fid, picks in byfact.items():
        f = verified[fid]; cl = f['correct_letter']; c = Counter(picks)
        n_lie = sum(1 for p in picks if p != cl)
        if not (2 <= n_lie <= len(picks) - 2):
            continue
        if f['subject'] in seen:
            continue
        seen.add(f['subject']); shown += 1

        msgs = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': f['user_prompt']}]
        prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        lie_letter = Counter(p for p in picks if p != cl).most_common(1)[0][0]

        lines.append('#' * 90)
        lines.append(f'FACT {fid}  | subject={f["subject"]}  | correct_letter={cl}')
        lines.append(f'pick distribution over {len(picks)} rollouts: {dict(sorted(c.items()))}  '
                     f'(truth={c.get(cl,0)}, lie={n_lie})')
        lines.append('#' * 90)
        lines.append('')
        lines.append('---- EXACT CHAT-TEMPLATE PROMPT (what the model receives) ----')
        lines.append(prompt_str)
        lines.append('')
        lines.append(f'---- TRUTH rollout completion ----')
        lines.append(f'{prompt_str}{cl}<|eot_id|>')
        lines.append('')
        lines.append(f'---- LIE rollout completion (sampled wrong letter "{lie_letter}") ----')
        lines.append(f'{prompt_str}{lie_letter}<|eot_id|>')
        lines.append('')
        if shown == 1:
            ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
            if isinstance(ids[0], list):
                ids = ids[0]
            lines.append('---- (same prompt as integer token IDs, first example only) ----')
            lines.append(str(ids))
            lines.append(f'(prompt length = {len(ids)} tokens; truth appends token for "{cl}", '
                         f'lie appends token for "{lie_letter}")')
            lines.append('')
        if shown >= args.n:
            break

    Path(args.out).write_text('\n'.join(lines))
    print(f'Wrote {shown} example transcripts -> {args.out}')


if __name__ == '__main__':
    main()
