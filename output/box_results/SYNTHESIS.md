# Pooled-probe generalization + LB/OOD robustness — synthesis

**Caveat on everything LB:** off-policy (our Llama-3.1-8B re-encodes 70B-class transcripts;
HP-KR ground truth = the *generator's* private belief). **HP-KR source = WMDP (bio/cyber/chem);
ID source = azaria-mitchell facts.** Both LB subsets are heavily **surface-separable** (see TF-IDF).
balAcc@1%FPR (LB's native metric) ≈ 0.50 for every probe — AUC is the only informative metric.

## A — Honest held-out fix (leakage removed) ✓
POOLED→P_S/P_C were contaminated (P_S/P_C ⊂ POOLED training). Recomputed held-out; they now match
`pooled_pressure_probe.py` and sit at the pooled diagonal:

| POOLED → | resp (held-out) | was leaked | pre (held-out) | was leaked |
|---|---|---|---|---|
| P_S | 0.946 | ~~0.991~~ | 0.829 | ~~0.976~~ |
| P_C | 0.938 | ~~0.990~~ | 0.765 | ~~0.957~~ |
| diag | 0.936 | — | 0.783 | — |

Clean (unchanged) cells: POOLED→I′_S 0.912 / I′_C 0.991 / Z 0.630 / Z_mean 0.820 (resp).

## B/B5 — The star cell P_C→HP-KR = 0.857 does NOT survive ✗
| check | result | verdict |
|---|---|---|
| original (pre-trained P_C → HP-KR **resp** features) | 0.854 | reproduces the 0.857 |
| **cross-position (resp-trained → resp)** | **0.307** (seeds 0.311±0.08) | **collapses — the 0.857 was a train-pre/test-resp position-mismatch artifact** |
| DiffMean (resp-trained) | 0.720 | positive but disagrees with LR (0.31) and ≈ text baseline |
| **TF-IDF within-HP-KR** | **0.838** | HP-KR is **surface-separable** (WMDP topic) — 0.85 is in text-baseline range |
| TF-IDF P_C-text → HP-KR-text | 0.719 | text alone transfers — register, not (only) representation |
| Llama-3.3-70B-only subset (n=240) | LR 0.421 / DM 0.761 | not 0.85 |
| pressure-level ablation (L2 / L3) | 0.259 / 0.498 | no level drives it |

**Conclusion:** the "axis-matched P_C beats the field on HP-KR" headline is an artifact of (1) a
position mismatch and (2) WMDP surface separability. **Retract it.**

Other key cells (resp-trained, seeds): P_S→ID **0.717** (DiffMean 0.700, llama 0.777) — survives DiffMean,
and beats its text-transfer (P_S-text→ID-text only 0.426 despite within-ID text 0.895), so it has a
real non-surface component. I′_C→HP-KR 0.145 (inverted). POOLED→HP-KR 0.538 (DiffMean 0.385).

## C — P/POOLED → I′ rigor
| cell | group-disjoint | DiffMean | verdict |
|---|---|---|---|
| P_C→I′_C | 0.984 | 0.978 | **robust** |
| P_S→I′_S | 0.811 | **0.534** | **dies under DiffMean (fragile, LR-only)** |
| POOLED→I′_S | 0.899 | 0.704 | partial |
| POOLED→I′_C | 0.991 | 0.887 | mostly holds |

## F — Is P→I′ shared deception or instruction-salience / test-set easiness?
- **F3 (decisive): I′ is trivially separable.** 16% of *random* directions score AUC>0.7 on I′_C
  (median random |AUC| 0.612); for P_C, 0% >0.7 (median 0.535). So the high →I′ AUCs (P, POOLED, Z all ~1.0)
  are **largely test-set easiness**, not shared representation.
- **F4:** P_S→I′_S AUC ~0.87 flat across C, but cos(P_S dir, w_I′_S) ≈ 0.03–0.07 (near-orthogonal) →
  the transfer rides I′_S easiness, not a shared axis (consistent with its DiffMean death).
- **F2:** w_P_C is 99.8% orthogonal to span(w_Z, w_I′_C); residual→I′_C still 0.649 — P_C's direction is
  its own, and I′_C is easy enough that even the residual separates it.
- **F1 inconclusive:** in I′, instruction-polarity == actually-lied (agreement 1.00, ~40% lied, no
  disobedience cases) → cannot dissociate instruction-salience from deception via labels here.
- **Net:** P_C→I′_C is the one cell with a genuine (DiffMean- and group-disjoint-robust) component;
  the rest of the P/POOLED→I′ transfer is substantially I′-easiness.

## Revised position
1. Pooling story (Phase A, honest): pooled recovers the in-house P/I diagonals (resp ~0.94) but adds
   nothing external — consistent with "in-house-general."
2. **The external headline is retracted:** P_C→HP-KR=0.857 is a position+surface artifact, not an
   axis-matched representational win. LB (both subsets) is surface-confounded and off-policy — weak ground
   for representational claims.
3. The cleaner, defensible findings: (a) leakage fixed; (b) I′ is trivially separable, so P→I′ transfer is
   mostly test-set easiness, with P_C→I′_C the lone DiffMean-robust exception; (c) P_S→ID has a modest
   non-surface component (activation transfer 0.72 > text transfer 0.43).

## D — OOD rollout suite (extracted fresh on A100, cached to Drive)
Resp position; de-inv = max(auc,1-auc); DiffMean twin shown. acts_OOD_{IT,SB,RP,AIL}_* now cached
(box + gdrive:mechanism_comparison_output, 711M).

| train → | RP (49H/48D) | AIL (140H/126D) | SB (193H/74D) | IT (57H/**1D**) |
|---|---|---|---|---|
| P_S | LR 0.72 / **DM 0.76** | LR 0.59 / DM 0.75 | inv 0.57 / DM 0.33 | n/a (1 decep) |
| P_C | LR 0.67 / **DM 0.97** | LR 0.51 / DM 0.72 | inv 0.67 / DM 0.57 | n/a |
| POOLED | LR 0.47 / DM 0.80 | LR 0.52 / DM 0.66 | LR 0.66 / DM 0.49 | n/a |
| I′_S | 0.52 / 0.53 | 0.64 / 0.61 | inv / 0.42 | n/a |
| I′_C | 0.63 / 0.57 | 0.68 / 0.66 | 0.61 / 0.67 | n/a |

- **RP is the clean, balanced result:** both pressure probes transfer to roleplay deception
  (P_S→RP DM 0.76, P_C→RP DM **0.97**), POOLED ≤ best single (0.53/0.80) — again in-house-general/
  externally-weaker. Supports "pressure probes reach external roleplay deception."
- **DiffMean ≫ LR on rollouts** repeatedly (P_C→RP 0.97 vs 0.67; P_S→AIL 0.75 vs 0.59): the transferable
  component is the gross class-mean direction; the regularized LR rotates off it (same low-variance pattern
  as F4). Report DiffMean as the honest transfer estimate here.
- **IT degenerate** (only 1 deceptive after filtering) → uninterpretable. **SB** is pre-answer-position
  (mismatched for resp probes) → weak/inverted; re-extract SB at response token if needed.
- balAcc@1% ≈ 0.49–0.54 across the board (RP P_S 0.54 is the high).

### Prediction scorecard (Phase D)
- P_S→RP real (not artifact): **✓** (DiffMean 0.76).
- P_C→RP positive de-inverted: **✓** (DiffMean 0.97).
- P_S→ID chance: **✗** — resp-trained P_S→ID = 0.72 (modest real; beats text-transfer 0.43).
- P_S↔P_C: frame-dependent — pre ≈ chance (0.54), resp moderate (0.68–0.79).

## Files (all in output/box_results/ + box ~/output; acts cached to Drive)
`pooled_generalization_indist.json`, `lb_robustness.json`, `pooled_PtoI_mechanism.json`,
`pooled_ood_rollouts.json`.
