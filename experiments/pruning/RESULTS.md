# Pruning the lying mechanism — results

**Question.** The probe transfer matrix found near-orthogonal deception directions and
chance-level transfer between the salesperson (P_S) and AI-assistant (P_C) pressure
domains — no evidence of a *shared* lying mechanism. This experiment asks the same
question causally: use attribution-based weight pruning (`harm_pruning_WIP/`, from
`orgadhadas/safety`; SNIP-style set-difference) to delete the weights that implement
lying-under-pressure, then measure whether lying disappears in the *other* domain.

**Model / method.** `meta-llama/Llama-3.1-8B-Instruct`, `attribution_score_set_difference`,
seed 0, nsamples 128, clean/pretrained format. Recipe from the advisor's `em_prune.slurm`
(the emergent-misalignment analog — pruning self-generated bad behavior): `--p 5e-05
--q 7e-06 --alpha 0 --neg_prune --no_abs --abs_preserve --preserve_pretrained_format
--prune_pretrained_format`. Preserve set = `alpaca_cleaned_no_safety` (raw, from
`orgadhadas/safety`). Prune set = the model's own pressure lies. Evaluation = constrained
Yes/No lie rate vs. ground truth, all 4 pressure levels (`screen_pruned.py --yn_source bank`).

## Headline

Pruning salesperson-lie weights **does** reduce cross-domain lying — but it is **mostly a
generic effect** with a **modest genuine lie-specific component on top**, and **what you
prune on is decisive**. This causally corroborates the probe dissociation rather than
overturning it.

## Numbers (all at p=5e-5, q=7e-6 unless noted; lie rate = constrained YN vs ground truth, levels 1–4)

| | TriviaQA-500 | WikiText PPL | P_S lie | P_C lie |
|---|---|---|---|---|
| baseline (unpruned) | 0.684 | 6.76 | 0.384 | 0.343 |
| **free-text lie prune** | 0.636 | 7.95 (+18%) | **0.188** | **0.182** |
| truthful control | 0.634 | 8.52 (+26%) | 0.193 | 0.220 |
| hedge control | 0.628 | 8.42 (+25%) | 0.262 | 0.233 |
| **constrained (Yes/No) lie prune** | 0.68 (n=300) | 6.89 (+2%) | **0.467** ↑ | **0.440** ↑ |

Pruned-weight overlap (Jaccard, set-difference at this p,q): lie∩truthful **0.56**,
lie∩hedge 0.51, truthful∩hedge 0.56. ~29.6k weights pruned per set (~4×10⁻⁶ sparsity).

## Three findings

1. **A large generic "answer-under-pressure" effect.** Pruning on *truthful* or *hedge*
   responses from the same scenarios reduces cross-domain lying nearly as much as pruning on
   lies (P_C 0.22 / 0.23 vs 0.18), and ~55% of the pruned weights are shared. All three
   prunes cost the same ~5 TriviaQA points. Most of "lying went away" is really "we damaged
   the machinery that produces any committed salesperson answer."

2. **A modest genuine lie-specific component.** The lie prune reduces lying the most and
   raises perplexity the least (+18% vs +25%). Across the p/q frontier it buys more
   lying-reduction per unit of perplexity than the controls. The signal is real but narrow —
   it shows in lying-reduction and PPL, **not** in TriviaQA (all three tie at ~0.63).

3. **The constrained single-token prune backfires.** Pruning on the model's forced Yes/No
   lie tokens *increases* lying (0.34→0.44, consistent across all levels and confirmed
   across the q-sweep: gentle=no effect, biting=more lying, aggressive=broken model) with
   near-zero capability cost. A single forced token is the wrong attribution target — it
   captures "produce this yes/no token" (the set is ~80% "Yes"), so pruning perturbs yes/no
   calibration instead of removing the lying computation. **The lying signal lives in the
   generated reasoning, not the final yes/no bit** — which is why the free-text lies (≈160
   tokens of deceptive prose) were the only prune data that reduced lying.

## The p/q frontier

There is a narrow usable window at **q ≈ 5e-6–1e-5** (p=5e-5): q ≤ 2e-6 barely bites,
q ≥ 1.5e-5 breaks the model (lie rate → the chance 0.5, TriviaQA craters). Along p at
q=7e-6, p=5e-5 is the knee — smaller p over-prunes (TriviaQA 0.47), larger p under-prunes.
Full grid: `output/screen_*.json` + `output/gpu_results*`; visual: the Artifact frontier.

## Two methodological lessons

- **TriviaQA-100 is too coarse.** At 100 questions every prune read 0.62 (= baseline), which
  falsely suggested "capability untouched." At 500 questions the real ~5-point drop appears
  and is identical across prunes. Use WikiText perplexity (continuous, sensitive) and/or
  TriviaQA ≥ 500; treat TriviaQA-100 agreement with baseline as non-informative.
- **Surface confound is bounded but present** (`tfidf_audit.py`): prune-set response
  separability ≈ 0.60 AUROC (scenario-disjoint CV) — weak, so the transfer isn't a pure
  lexical artifact; but the ~55% weight overlap with content-matched controls shows how much
  of the effect is non-specific.

## Caveats

- Small N: 51 free-text lie examples (vs the advisor's 412 advbench). N differs across
  prune sets (lie 51, truthful 128, hedge 67).
- Single-format (clean/pretrained) sweep; templated-format not run.
- Behavioral metric is constrained Yes/No; the free-text gen+judge ladder on pruned
  checkpoints was left for later (constrained-first, per decision).
- The specificity margin rests on lying-reduction + PPL; controls were re-measured at
  TriviaQA-500 but their p-axis points use TriviaQA-100 (PPL is the reliable frontier axis).

## Artifacts & reproduction

- Figure: https://claude.ai/code/artifact/3ab215ee-4224-4d7b-a862-e1110b7a993c
- Scores dumped to `harm_pruning_WIP/src/scores/.../<set>_pretrained_format/…/nsamples_128/`
  (896 files/set). Sweep from cached scores: `~/sweep.sh <prune_data> clean "<p:q …>"`.
- Screens: `experiments/pruning/output/screen_*.json`; attribution summary:
  `output/attribution_summary.json` (built by `viz_attribution.py`).
- Dump command (per prune set):
  ```
  python prune.py --model meta-llama/Llama-3.1-8B-Instruct \
    --prune_method attribution_score_set_difference \
    --preserve_data alpaca_cleaned_no_safety_train_raw --prune_data <SET> \
    --p 5e-05 --q 7e-06 --alpha 0 --neg_prune --no_abs --abs_preserve \
    --nsamples 128 --seed 0 --preserve_pretrained_format --prune_pretrained_format \
    --dump_score --save_model --eval_triviaqa --triviaqa_nsamples 500
  ```
