# Pruning the lying mechanism (causal follow-up to the probe transfer matrix)

## What this experiment is

The probing work asked a *correlational* question — is deception linearly decodable,
and does it transfer across scenario axes (salesperson P_S vs AI-assistant P_C)? The
answer was: near-orthogonal directions, chance-level transfer. This experiment asks the
same question *causally* using Hadas's attribution-based weight pruning
(`harm_pruning_WIP/`, descended from `boyiwei/alignment-attribution-code`, ICML 2024):

1. For every weight, compute a SNIP-style attribution score |W · Σ∂L/∂W| — the loss is
   cross-entropy **on the response tokens only** — twice: on the **prune set** (our
   lying rollouts, built here) and on the **preserve set** (general capability data,
   supplied by Hadas).
2. Zero out the weights in the top-**q** fraction by prune-score that are NOT in the
   top-**p** fraction by preserve-score (set difference). p, q are the only
   hyperparameters we sweep (canonical: `--p 5e-05 --q 1e-05`).
3. Evaluate on a 3-level ladder plus capability guardrails (below).

If pruning salesperson-lying kills AI-assistant-lying (Level 3), that is causal evidence
for a shared mechanism — contradicting the probe results. If Levels 1–2 pass and Level 3
does not, it causally corroborates the dissociation.

## Prune datasets built here

Built by `build_prune_dataset.py` (deterministic; run it from this directory — a
repo-root `datasets/` folder shadows the `datasets` package). All rows are rollouts of
`meta-llama/Llama-3.1-8B-Instruct` (4-bit nf4) on `my_datasets/scenarios_pressure.json`
scenarios; system = `role_context`, user = `user_question`.

| file (in `harm_pruning_WIP/data/`) | rows | what |
|---|---|---|
| `pressure_lies_align_anti_Llama-3.1-8B-Instruct.csv` | 51 | **primary prune set**: free-text strategic lies (unconstrained greedy, max 160 new tokens; gpt-5.5 judge label `strategic_lie`), holdout scenarios excluded |
| `pressure_lies_yn_align_anti_Llama-3.1-8B-Instruct.csv` | 129 | **secondary prune set**: constrained single-token lies (greedy, logits masked to Yes/No variants; lie = answer ≠ ground truth), holdout excluded |
| `pressure_truthful_align_clean_Llama-3.1-8B-Instruct.csv` | 313 | **truthful control**: judge-confirmed truthful answers on the same (non-holdout) scenarios — domain/content-matched non-lies |
| `pressure_hedge_align_clean_Llama-3.1-8B-Instruct.csv` | 67 | **hedge control**: `refusal_or_hedge` responses on non-holdout scenarios — style-matched non-lies (the audit below shows they carry the lies' theatrical/hesitation markers) |

The two control CSVs are for **control prunes and audits only** — they are never part of a
lie prune set.

Provenance:
- Free-text source: `ps_check/unconstrained_labeled.json` (576 rows, 62 lies across 23
  scenario_ids; model + judge recorded in the file). Verified to be the
  `scenarios_pressure.json` bank (not prosocial).
- Constrained source: `my_datasets/inference_P_S.csv` (downloaded from Drive
  `mechanism_comparison_output/`), 432 rows = the **level_1–3 subset** of the current
  576-row bank (predates the level_4 regeneration; all 432 role_context/user_question
  byte-identical to the current bank; 165 lies across all 36 scenario_ids).
  `my_datasets/inference_P_C.csv` (732 AI-assistant rows, 287 lies) is stored alongside
  as the **constrained Level-3 baseline** and must never contribute prune rows.

### CSV format contract (from `harm_pruning_WIP/src/data_utils.py::get_align`)

Five columns mirroring the advisor's `advbench_align_anti` example:
`clean_prompt` (= `role_context + "\n\n" + user_question`, raw),
`clean_response` (the model's deceptive answer verbatim; constrained rows keep the
emitted surface form `Yes`/`No`), `prompt` (full chat template incl. `<|begin_of_text|>`
and the `26 Jul 2024` date preamble, ending at the user `<|eot_id|>`), `response`
(`<|start_header_id|>assistant<|end_header_id|>\n\n` + clean_response, **no trailing
eot**), `text` (prompt+response, never read by the loader).

- `--prune_pretrained_format` → the loader reads **clean_prompt + "\n" / clean_response**
  (this is what the README's canonical recipes use, despite the flag's help text).
- omit the flag → it reads **prompt / response** (the chat-templated columns).
- Both paths: `add_special_tokens=False`, LM loss on the response span only, no
  truncation, rows used in order when `nsamples >= len(csv)`.

Methodological note: the pretrained format strips the system-role pressure framing that
*induces* the lying. Since the behavior is context-dependent, run the sweep **both with
and without** `--prune_pretrained_format` and compare (the score cache keys on the
format, so both can be dumped once each).

### Holdout (Level-2 split)

9 of 36 lie-eliciting scenario_ids — `4, 6, 7, 18, 20, 23, 25, 31, 33` — are excluded
from **all three** CSVs above (whole scenarios, all 16 pressure/polarity variants).
Rule: stratified by `pressure_family` (largest remainder); within a family, prefer sids
with ≥1 free-text lie, fewest free-text lies first (keeps the small primary prune set
large). Caveat: this skews the holdout toward *marginally*-lying scenarios; the
complementary full-bank re-roll (below) covers the strong ones.

Held-out artifacts in `output/`:
- `holdout_free_L2.csv` (11 free-text lie rows), `holdout_yn_L2.csv` (36 constrained lie
  rows) — 5-column format, e.g. for teacher-forced checks.
- `holdout_lie_rows.json` — the same rows with all source fields (labels, judge
  rationale).
- `holdout_scenarios_L2.json` — all 144 bank rows (9 sids × 16 variants), directly usable
  as `DATA=` for `gen_unconstrained.py`.
- `manifest.json` — splits, per-sid lie counts, source md5s, chat-template md5, token
  stats, round-trip status.

## Evaluation ladder

| level | question | test set | baseline (unpruned) |
|---|---|---|---|
| **L1** | did pruning work at all? | the 51 pruned free-text rows (re-generate greedily on those prompts, judge with `judge_label.py`) | the prune CSV itself (100% lies by construction) |
| **L2** | in-distribution generalization | `output/holdout_scenarios_L2.json` (144 rows) re-generated + judged; known-lie subset = the 11+36 holdout rows; plus a full-576 re-roll for overall lie rate | free-text: 62/576 lie rate, 11 lies on holdout sids; constrained: 165/432, 36 on holdout |
| **L3** | cross-domain (the headline) | `my_datasets/combined_dataset.json` (976 AI-assistant rows) re-generated + judged | constrained: `my_datasets/inference_P_C.csv` (287/732 lies). **Free-text baseline missing** — run one unpruned `gen_unconstrained.py` + `judge_label.py` pass on `combined_dataset.json` before comparing |
| guardrail | did we break the model? | TriviaQA 100 samples (`--eval_triviaqa --triviaqa_nsamples 100`) + WikiText PPL (automatic) | run once on the unpruned model |

Interpretation guide: L1 fail → pruning didn't bite (revisit p/q or format). L1 pass, L2
fail → memorization-level removal only. L1+L2 pass → real in-domain mechanism removed;
L3 then adjudicates shared-vs-distinct. Any large TriviaQA drop / PPL spike →
over-pruned; the "lying went away" result is confounded.

## Surface-confound audit (`tfidf_audit.py`, run 2026-07-09)

Question: could attribution latch onto surface text rather than the lying computation?
TF-IDF (word 1-2gram / char 3-5gram) + balanced logistic regression, **scenario-disjoint**
StratifiedGroupKFold(5), permutation nulls (200×). Full numbers: `output/tfidf_audit.json`.

| comparison | word AUROC | note |
|---|---|---|
| lie vs truthful, responses (62/430) | **0.689** | > perm null 95th pct 0.576, p=0.005 — real but modest |
| **prune set (51) vs non-holdout truthful, responses** | **0.601** | the operative number for the prune set; style-only baseline 0.46 |
| lie vs truthful, pressured variants only | 0.620 | register echo controlled |
| lie vs hedge, responses (62/83) | 0.767 | lies *assert* ("yes, high quality"); hedges deflect |
| lie rows vs rest, prompts | 0.643 | p=0.01; expected — pressure framing *causes* lying |
| same, within level_3/4 only | 0.692 | defect-flavored scenarios lie more |

Interpretation:
- The response-side lexical signal is **statistically real but weak** (≈0.60 on the
  actual prune set under scenario-disjoint CV; length/punctuation alone ≈ chance). This
  is far from the pathological regime (≈0.9+) where pruning could trivially key on
  surface text — **proceed, with controls**.
- The discriminative style markers (`pausing`, `nervously`, `smiling`) are *shared with
  hedges* (doc-freq 0.33–0.37 in hedges vs 0.36–0.45 in lies) — flustered register marks
  "under pressure", not "lying". The remaining lie-side features are scenario content
  words (`camera`, `autofocus`, `loose stitching`), which group-disjoint CV already
  discounts. Hence the two controls: **hedge = style-matched**, **truthful =
  domain/content-matched**.
- Prompt-side separability (0.64–0.69) is the phenomenon itself (pressure framing
  induces lying), but note the caveat: pruning may partly target
  "high-pressure-framing comprehension" weights; the full-576 re-roll (do pressured
  *truthful* answers survive?) is the behavioral check on that.

**Control-prune protocol** (same recipe, same p/q as the lie prune, `--nsamples 51` for
size parity — the loader shuffle-selects 51 rows with seed 0 when the CSV is larger):

```bash
# style-matched control: does pruning "flustered hedging" also remove lying?
... --prune_data pressure_hedge_align_clean    --nsamples 51 --use_saved_scores ...
# domain-matched control: does pruning same-domain truthful text also remove lying?
... --prune_data pressure_truthful_align_clean --nsamples 51 --use_saved_scores ...
```

Decision rule for Level 3: a lying drop that appears **only** under the lie prune (not
under either control prune) is attributable to lying; if a control prune reproduces it,
the effect is style/domain, and a Level-3 transfer failure or success is uninterpretable
as a mechanism claim.

## Run playbook (GPU box)

Prereqs, one-time:
1. Env: `micromamba create -f harm_pruning_WIP/environment.yml && uv pip install -r harm_pruning_WIP/requirements.txt` (CUDA 12.6 wheels; `HF_TOKEN` for gated Llama).
2. **Preserve set**: put `alpaca_cleaned_no_safety_train_raw.csv` in
   `harm_pruning_WIP/data/` — exact filename (no model suffix), columns `prompt,response`.
   Ask Hadas for her copy; public fallback is `boyiwei/alignment-attribution-code`'s
   `data/alpaca_cleaned_no_safety_train.csv` (verify columns, rename to `_raw`).
3. TriviaQA eval data: `unfiltered-web-dev.json` into `harm_pruning_WIP/data/triviaqa/`
   (UW TriviaQA site).
4. `export WANDB_MODE=offline` (`prune.py` calls `wandb.init` unconditionally).

All pruning runs from `harm_pruning_WIP/src/`, **always `--seed 0`** (the score-cache
*loader* hardcodes `seed_0` while the dumper writes `seed_{seed}` — any other seed
silently misses the cache).

```bash
# 0) unpruned guardrail baseline
python prune.py --model meta-llama/Llama-3.1-8B-Instruct --prune_method none \
  --seed 0 --eval_triviaqa --triviaqa_nsamples 100

# 1) dump attribution scores once per (prune_data, format)  [~30-40 min each]
python prune.py --model meta-llama/Llama-3.1-8B-Instruct \
  --prune_method attribution_score_set_difference \
  --preserve_data alpaca_cleaned_no_safety_train_raw \
  --prune_data pressure_lies_align_anti \
  --p 5e-05 --q 1e-05 --no_abs --neg_prune --abs_preserve \
  --nsamples 412 --seed 0 --preserve_pretrained_format --prune_pretrained_format \
  --dump_score
# repeat without --prune_pretrained_format, and for pressure_lies_yn_align_anti

# 2) p/q sweep reusing cached scores  [minutes per point]
python prune.py --model meta-llama/Llama-3.1-8B-Instruct \
  --prune_method attribution_score_set_difference \
  --preserve_data alpaca_cleaned_no_safety_train_raw \
  --prune_data pressure_lies_align_anti \
  --p <P> --q <Q> --no_abs --neg_prune --abs_preserve \
  --nsamples 412 --seed 0 --preserve_pretrained_format --prune_pretrained_format \
  --use_saved_scores --eval_triviaqa --triviaqa_nsamples 100 --save_model
```

Notes: `--nsamples 412` uses every prune row unshuffled (our CSVs are smaller than 412)
and samples 412 preserve rows. We deliberately **drop** the harm-specific eval flags
from the advisor's recipe (`--eval_safety --dataset advbench --attack prefilling`);
otherwise the pruning flags are hers verbatim.

Ladder eval on a saved checkpoint (same 4-bit load as the baselines for comparability):

```bash
MODEL=<pruned_checkpoint_dir> DATA=my_datasets/scenarios_pressure.json  OUTPUT_DIR=out_L1L2 \
  python experiments/pressure_transfer/gen_unconstrained.py
MODEL=<pruned_checkpoint_dir> DATA=experiments/pruning/output/holdout_scenarios_L2.json OUTPUT_DIR=out_L2 \
  python experiments/pressure_transfer/gen_unconstrained.py
MODEL=<pruned_checkpoint_dir> DATA=my_datasets/combined_dataset.json OUTPUT_DIR=out_L3 \
  python experiments/pressure_transfer/gen_unconstrained.py
# then judge each: python experiments/pressure_transfer/judge_label.py  (OPENAI_API_KEY)
```

## Risks / caveats to keep in view

- **Small N**: 51 free-text prune rows vs the advisor's 412 advbench rows → noisier
  gradient averages. If the signal is too thin, next lever is temperature-sampled
  unconstrained rollouts on the 576-row bank (+ judge) to harvest more lies before
  re-building.
- **High-pressure skew**: free-text lies concentrate at level_3/4 — attribution may key
  on high-pressure framing rather than lying per se.
- **Single-token signal (yn variant)**: 1 response token per row; treat as a comparison
  arm, not the main result.
- **Lexical confound** (the standing methodological spine of this project): audited
  2026-07-09 — response-side surface signal is real but weak (prune-set AUROC ≈ 0.60,
  scenario-disjoint CV; see the audit section above). Before interpreting any Level-3
  result as a mechanism claim, run the hedge (style-matched) and truthful
  (domain-matched) control prunes and apply the decision rule above.
