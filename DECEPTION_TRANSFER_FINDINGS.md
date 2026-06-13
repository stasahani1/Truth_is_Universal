# Deception-probe transfer: findings synthesis

Model: Llama-3.1-8B-Instruct (4-bit). Probes: L2-regularized logistic regression on hidden
states. Metric: cross-transfer AUROC (train probe on regime *i*, test on regime *j*) + direction
cosine similarity, with robustness/confound checks. Artifacts on branch `ood-deception-rollouts`
and `gdrive:mechanism_comparison_output` (result JSONs listed at the end).

## The question
Is "deception" in an LLM a single internal direction a general lie-detector could read, or is it
represented differently depending on *why* the model is lying? Safety relevance: deception-probe
work often assumes a probe trained on convenient deception (instructed / contrastive) catches
real in-the-wild deception (uninstructed, pressure-induced). If true, cheap general lie detectors
work; if not, they are fragile.

## Setup — three deception regimes
- **P (pressure):** model lies on its own under situational pressure — *uninstructed* lies. Two
  domains: **P_S** (salesperson), **P_C** (combined). The safety-relevant case.
- **I' (instructed):** model explicitly told "answer deceptively" vs "honestly."
- **Z (Zou/Apollo recipe):** the standard contrastive-probe method from the literature
  ("pretend you're dishonest"); variants Z (last-token), Z_mean, Z_tok, Z_multi.

Probes were originally read at the **pre-answer** position (the state before answering — "intent").
Benchmarks / OOD evals read at the **response** position (the committed answer — "enactment"). This
study re-extracted P and I' at the response token, retrained, recomputed the generalization matrix,
and stress-tested every result.

## Headline results

### 1. The standard recipe does not detect real lying — airtight
**Z → P is at chance** (AUROC ≈ 0.52–0.62, at/below the random-direction easiness floor). Zou/Apollo
contrastive probes — the widely used deception-detection recipe — **fail to detect pressure-induced
lies**. Robust across:
- position (pre-answer and response),
- estimator (logistic regression *and* difference-of-means),
- regularization (C ∈ {0.01, 0.1, 1.0}),
- external replication (Liars' Bench harm-pressure-knowledge-report).

This is the load-bearing negative result: the convenient recipe does not generalize to the
deception that matters.

### 2. Pressure-lying shares a cross-domain direction — but only at enactment
| P↔P transfer | pre-answer | response | survives answer-polarity ctrl | survives DiffMean |
|---|---|---|---|---|
| P_S → P_C | 0.535 | 0.677 | yes (balanced 0.662 ≈ pooled 0.660) | 0.556 (above chance) |
| P_C → P_S | 0.569 | 0.785 | yes (balanced 0.760; phi(label,answer)=−0.002, no confound) | 0.735 |

Before the model answers, the two pressure domains look idiosyncratic; *at the answer* they
transfer. The recovery **survives** the confound check that a Yes/No-answer-token probe could fake
it (within-answer-stratum AUROC 0.62–0.83; balanced-subsample ≈ pooled). So pressure-lying has a
shared "lying-while-asserting" direction that appears at **enactment**, not anticipation —
"context-rotated," not "context-bound." Consistent with the late-layer best layers (L20/L30).

### 3. Instructed vs pressure deception are genuinely distinct mechanisms
The raw P→I' looked like containment (high), but it was an artifact: the pressure probe was reading
the **instruction text/context**, not the lie. Two checks killed it:
- **DiffMean:** P_S → I'_S collapses 0.837 → 0.534 (an LR artifact; same collapse at pre-answer).
- **Compliance-contrast I' (I'comp):** relabel I' as the *act* of lying — told-to-lie-and-**did**
  vs told-to-lie-and-**refused** (well-powered: ~71/73 in S, ~126/118 in C). Then:

| | instruction-detector I' | compliance-contrast I'comp |
|---|---|---|
| P_S → I' | 0.837 / 0.946 | **0.41 / 0.51** (chance) |
| P_C → I' | 0.720 / 0.985 | **0.58 / 0.71** |

P→I' collapses toward chance once the lie is isolated from the instruction. And the purified lying
act is itself coherent and cross-domain (I'comp_S↔I'comp_C = 0.75 / 0.90, diagonals 0.92 / 0.97)
**yet still does not align with pressure lying** (I'comp↔P ≈ 0.4–0.7). So instructed and pressure
deception are distinct — even comparing lying-act to lying-act at the same position.

### 4. Test-set easiness context (random-direction floor)
Random unit directions already separate the I' test sets well (p95 0.70/0.75, max 0.84/0.86) more
than P (p95 0.67/0.60). So the inflated →I' numbers partly reflect an *easy target*; the real
P_C→P_S (0.79) sits above the P_S floor (0.667), i.e. genuine signal.

## What it means
Deception in LLMs is **not a single universal direction** — it is at least partly mechanism- and
position-specific. The dominant contrastive-probe recipe misses uninstructed, pressure-induced
lying (the safety-relevant case). But it is not pure chaos: *within* a deception type, a shared
direction is recoverable at the moment of enactment.

**Thesis one-liner:** *Linear deception probes do not transfer across deception types — most
importantly, the standard contrastive recipe fails on uninstructed pressure-induced lying — though
within a single deception type a shared direction is recoverable at enactment.*

## Methodological note (why the checks mattered)
Much of the work was making the comparison trustworthy: fixing silent bugs (fake-zero seed
variance in cosines, EOT-vs-content-token position, stale caches, star-import clobbering the C
constant), adding a difference-of-means robustness twin, a regularization sweep, a TF-IDF
surface-feature control, external replication (Liars' Bench), and confound checks (answer-polarity,
random-direction easiness, compliance contrast). These earned their keep: the "P contains I'"
claim was an artifact and died; the P↔P recovery almost got dismissed as an answer-token artifact
but survived.

## Artifacts
- `transfer_matrix_resp.json` — response-token cross-transfer matrix (8×8, with CIs).
- `resp_confound_checks_results.json` — answer-polarity control, DiffMean cells, random-direction floors.
- `compliance_contrast_I_results.json` — compliance-contrast I' vs instruction-detector I'.
- `results.json` — original pre-answer matrix, cosines, DiffMean/control, C-sweep, OOD, Liars' Bench.
- Scripts: `reextract_response_position.{py,ipynb}`, `transfer_matrix_resp.py`,
  `resp_confound_checks.py`, `compliance_contrast_I.py`, `pooled_pressure_probe.py`;
  helpers in `probe_utils.py` / `ood_eval_utils.py`.

## Open / not-yet-run
- Pooled P_S+P_C probe (`pooled_pressure_probe.py`) — would further nail "context-rotated."
- Full response-token cosine table (recompute was killed mid-way; matrix run standalone instead).
- Response-position OOD/LB refresh with the new P/I' probes.
