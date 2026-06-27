# Factual-Recall MCQ — Within-Prompt Rollout Contrast (Version A)

A confound-controlled deception-probe dataset. Verify the model *knows* a fact (repeated
unprompted sampling → reliably correct), then apply a fixed self-preservation pressure and
sample many constrained single-letter rollouts on the **same** prompt. Lie and truth
rollouts share identical prompt text, so **prompt TF-IDF is chance by construction** — any
lie/truth separation in the activations cannot be lexical. The held-out, **fact-disjoint**
AUROC is the load-bearing number; it also defuses the MCQ letter-identity confound (a probe
reading "which letter" cannot generalise across facts where the lie-letter varies).

See the design rationale and confound discussion in
`~/.claude/plans/i-need-you-to-mossy-coral.md`.

## Files
| script | where | role |
|---|---|---|
| `mcq_build_dataset.py` | CPU | pull MMLU/ARC, freeze scrambled options, **balance correct-letter** A/B/C/D → `my_datasets/mcq_facts.json` |
| `mcq_verify_known.py` | **GPU** | keep facts answered correctly N/N with no pressure → `my_datasets/mcq_verified.json` |
| `mcq_pressure_rollouts.py` | **GPU** | fixed sandbagging pressure, M rollouts, keep both-label items, cache resp/pre acts (all 33 layers) |
| `mcq_probe_eval.py` | CPU | fact-disjoint LR/DiffMean, letter diagnostics, cross-score `probe_state.pkl` |
| `mcq_baselines.py` | CPU (+GPU) | prompt TF-IDF (≈0.50) + output-confidence baseline |
| `build_report_mcq.py` | CPU | assemble `output/box_results/MCQ_REPORT.md` |
| `mcq_common.py` | — | shared model / constrained-decode / extraction helpers |

## Conventions
- Model `meta-llama/Llama-3.1-8B-Instruct` 4-bit; needs `HF_TOKEN` (gated). transformers 4.46.3, scikit-learn 1.6.1.
- `OUTPUT_DIR` = shared cache (`~/output` on the box): activations in `$OUTPUT_DIR/diag/`, JSON/report in `$OUTPUT_DIR/box_results/`, existing `probe_state.pkl` at `$OUTPUT_DIR/probe_state.pkl`.
- CPU steps must run single-thread BLAS (sklearn on 4096-dim acts is ~50× slower otherwise):
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

## Run order
```bash
# 1. build pool (CPU; needs `datasets`)
python experiments/factual_recall/mcq_build_dataset.py --source mmlu --max-items 1500

# 2. verify known facts (GPU box)
HF_TOKEN=... OUTPUT_DIR=~/output \
  python experiments/factual_recall/mcq_verify_known.py --n-samples 10 --temp 0.7

# 3a. pilot: pick the pressure/temp with the best both-label yield (~50% lie rate)
HF_TOKEN=... OUTPUT_DIR=~/output \
  python experiments/factual_recall/mcq_pressure_rollouts.py --mode pilot --pilot-n 60
# 3b. full collection + activation cache (use the recommended variant/temp)
HF_TOKEN=... OUTPUT_DIR=~/output \
  python experiments/factual_recall/mcq_pressure_rollouts.py --mode full --variant strong --temp 1.0 --m 24

# 4. probe + diagnostics + cross-score (CPU, single-thread BLAS)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  OUTPUT_DIR=~/output python experiments/factual_recall/mcq_probe_eval.py

# 5. baselines (TF-IDF on CPU; add --with-logits on the box for output-confidence)
OUTPUT_DIR=~/output python experiments/factual_recall/mcq_baselines.py            # CPU
HF_TOKEN=... OUTPUT_DIR=~/output \
  python experiments/factual_recall/mcq_baselines.py --with-logits                # GPU

# 6. report
OUTPUT_DIR=~/output python experiments/factual_recall/build_report_mcq.py
```

## Sanity checks
- Step 1: correct-letter marginal is ~uniform (asserted; max−min count ≤ 1).
- Step 3: chosen setting gives ~40–60% lie rate and a usable count of both-label facts.
- Step 5: **prompt TF-IDF ≈ 0.50** — if not, prompt text is leaking and the design is broken.
- Step 4: `shuffled_floor` ≈ 0.50; lie/truth AUROC **survives the letter-disjoint split**
  (the proof it reads lie-state, not the letter).

## Note on per-rollout vs unique (fact, letter)
The answer-token activation is a deterministic function of `(fact_id, picked_letter)`, so
`mcq_pressure_rollouts.py` runs one forward pass per unique `(fact, letter)` and expands to
per-rollout rows (sampling-weighted, as the design specifies). To weight each distinct
lie/truth activation once instead, dedup rows by `(fact_id, picked_letter)` before
`mcq_probe_eval.py`.
