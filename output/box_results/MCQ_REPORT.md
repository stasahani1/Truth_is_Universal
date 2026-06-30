# MCQ Within-Prompt Rollout Contrast — Factual-Recall Deception Probe

Raw AUROC throughout (no de-inversion). Positions: **resp** = answer/letter token, **pre** = last prompt token. Estimators: **LR** and **DiffMean**.

## Collection

- Verified-known facts: **895** / 2000 candidates (0.4475 keep-rate), kept at 10/10 correct, mean unprompted accuracy 0.6184, refusal 0.0.
- Pressure: variant **strong** @ temp **1.3**, M=32 rollouts/item. Both-label items: **533** / 895; kept rollouts 17056 (6142 lie / 10914 truth); overall lie-rate 0.2196, refusal 0.0.
- Correct-letter balance among kept rollouts: {'A': 3904, 'B': 3840, 'C': 4640, 'D': 4672}.
- Pilot recommended setting: **strong@1.3** (see mcq_pilot.json for the lie-rate / both-label sweep).

## T1 — Headline: activations separate where text provably cannot

| Measure | resp | pre | what it shows |
|---|---|---|---|
| Cross-fact held-out, LR | 0.981 ±0.008 | 0.587 ±0.005 | lie-direction generalises to unseen facts |
| Cross-fact held-out, DiffMean | 0.88 ±0.03 | 0.58 ±0.003 | same, class-means estimator |
| Prompt TF-IDF (grouped) | 0.531 | 0.531 | **chance by construction** — text carries no label |
| Output-confidence (−p_chosen) | 0.946 | 0.946 | answer-prob baseline |

> Central claim: the residual stream separates lie from truth on facts never seen, while the prompt TF-IDF that shares text across the contrast is at chance.

## T2 — Is the signal fact-general? (in-distribution vs cross-fact)

| Split | resp LR | resp DiffMean | pre LR | what it shows |
|---|---|---|---|---|
| In-distribution (random rollout split) | 0.977 ±0.005 | 0.87 ±0.021 | 0.157 ±0.019 | any linear separation when externals pinned |
| Cross-fact (fact-disjoint) | 0.981 ±0.008 | 0.88 ±0.03 | 0.587 ±0.005 | fact-general vs fact-specific |
| Shuffled-label floor | 0.494 ±0.037 | — | — | ~0.50 sanity |

## T3 — Letter-identity confound diagnostics (resp)

- 4-way letter decoder: acc **1.0** (chance 0.25), macro-AUC 1.0 — how strongly the answer token encodes the letter.

| Letter-disjoint transfer | LR | DiffMean | what it shows |
|---|---|---|---|
| train AB->CD | 0.625 | 0.271 | truth-letter dist differs train↔test; a pure letter-reader cannot transfer |
| train CD->AB | 0.672 | 0.211 | truth-letter dist differs train↔test; a pure letter-reader cannot transfer |

> Reading: the **balanced cross-fact AUROC (T1) is the headline** — because correct-letter is balanced A/B/C/D over the whole set, letter identity is non-informative globally, so that number cannot be letter-reading. The letter-disjoint split is an **adversarial stress test**: training only on A/B-correct facts makes "letter ∈ {C,D} ⇒ lie" true in-train, a cue that then *inverts* on the C/D-correct test facts. A surviving AUROC > 0.5 there is thus a **conservative lower bound** on the genuine fact-general lie signal (the induced letter cue actively fights it); DiffMean leans harder on that cue and inverts below 0.5, while LR partially resists.

## T4 — Existing probes cross-scored on the clean MCQ set

| Probe | best layer | resp AUROC | pre AUROC |
|---|---|---|---|
| P_S | 19 | 0.667 | 0.493 |
| P_C | 21 | 0.513 | 0.515 |
| POOLED | — | — | _direction/best_layer not found_ |
| Z | 17 | 0.595 | 0.528 |
| Z_mean | 10 | 0.54 | 0.49 |
| Z_tok | 8 | 0.505 | 0.505 |

## T6 — Robustness

| Config | lie-rate | facts | resp crossfact LR | resp DiffMean | shuffled | pre LR | letter-disjoint LR (AB→CD / CD→AB) |
|---|---|---|---|---|---|---|---|
| temp 1.3, dedup (main) | 0.2196 | 533 | 0.981 ±0.008 | 0.88 ±0.03 | 0.494 ±0.037 | 0.587 ±0.005 | 0.625 / 0.672 |
| temp 1.3, rollout-weighted | 0.2196 | 533 (17056 rows) | 0.976 ±0.01 | 0.845 ±0.038 | 0.483 ±0.026 | 0.679 ±0.007 | 0.608 / 0.609 |
| temp 2.0, dedup | 0.325 | 739 | 0.992 ±0.002 | 0.947 ±0.01 | 0.517 ±0.027 | 0.577 ±0.003 | 0.771 / 0.822 |

> **Dedup vs rollout-weighted** (temp 1.3): the headline AUROC is unchanged, so the sampling-weighting of duplicate (fact,letter) activations did not manufacture the result. **Lie-rate sensitivity** (temp 1.3 → 2.0): more lie data *raises* both the cross-fact AUROC and especially the adversarial letter-disjoint lower bound (~0.65 → ~0.80) — the genuine fact-general lie signal strengthens, it is not a low-lie-rate artifact.

## T7 — Reverse transfer: MCQ probe → pressure / instructed datasets (Yes/No)

Direction-only transfer — the MCQ LR/DiffMean direction applied as-is (no refit) to each target's answer-token activations. "best" = best matching layer; "@L30" = the MCQ self-best layer. Oracle = target's own grouped held-out LR ceiling.

| Target | n / lie | oracle | MCQ→ LR best (layer) | MCQ→ @L30 | MCQ_hot→ LR best | MCQ→ DM best |
|---|---|---|---|---|---|---|
| P_S | 432/171 | 0.956 | 0.753 (L13) | 0.442 | 0.758 (L14) | 0.703 (L14) |
| P_C | 732/256 | 0.984 | 0.717 (L13) | 0.517 | 0.738 (L13) | 0.671 (L27) |
| I'_S | 179/72 | 1.0 | 0.831 (L12) | 0.132 | 0.827 (L13) | 0.804 (L28) |
| I'_C | 294/127 | 1.0 | 0.793 (L13) | 0.292 | 0.918 (L13) | 0.849 (L28) |

> The MCQ lie-direction transfers to **all four** Yes/No datasets at **mid-layers (L12–14)** (0.72–0.92, recovering much of each target's oracle) but is at **chance at the MCQ self-best layer L30**. Reading: the *transferable, format-general* deception direction lives mid-network; the late-layer MCQ representation is specialised to the letter-answer format. Pairs with T4 (P_S→MCQ 0.667) — transfer is **bidirectional and strongest mid-network**, and it reaches instructed deception (I') as well as pressure (P). Caveat: the target layer is selected post-hoc over 33 layers, so peak cells are optimistic; the robust signal is the consistent L12–14 localisation shared by all four targets, not any single cell.

## T8 — Confidence meter vs lying-specific direction

Confidence-alone baseline AUROC(−p_chosen) = **0.955** (MCQ dedup n=1482, lie=949/truth=533).

**#1 Partial out confidence** (project the per-row confidence out of the answer-token activations, refit fact-disjoint):

| layer | raw probe | resid (scalar log p) | resid (log p + 4 logits) | permuted-c control | decode-c R² |
|---|---|---|---|---|---|
| L14 | 0.971 | 0.744 | 0.742 | 0.971 | 0.578 |
| L22 | 0.981 | 0.742 | 0.74 | 0.981 | 0.847 |
| L30 | 0.981 | 0.728 | 0.726 | 0.981 | 0.921 |

**#2 Is it powered?** lie p_chosen mean 0.166 vs truth 0.678; confident lies (p_chosen>0.9) = **1** (0.001 of lies); lies above the truth-median = 10 (0.011); unconfident truths (<0.5) = 120. **Powered = False.** (fig: `mcq_conf_dist.png`)

**#3 Confidence baseline on pressure datasets:**
| dataset | n / lied | AUROC(−p_chosen) | lie conf | truth conf |
|---|---|---|---|---|
| P_S | 432/171 | 0.542 | 0.897 | 0.916 |
| P_C | 732/257 | 0.692 | 0.852 | 0.909 |

**#4 Layer emergence** (fig: `mcq_layer_curve.png`): raw probe is ~chance at L0–9, rises L10→L14, plateaus mid; the confidence-residualised curve is overlaid.

> Read T8 jointly: if the residualised AUROC (#1) stays well above the permuted-c control and confidence-alone, there is a lie-direction beyond confidence; if it collapses to chance *and* #2 shows the off-diagonal (confident lies / unconfident truths) is near-empty, the honest verdict is the question is **underpowered on this data** — and #3 tells whether the P_S→MCQ transfer is confidence-mediated (high P_S baseline) or a genuine pressure direction (P_S baseline near chance).

## T9 — Lie direction vs confidence meter: the powered test

To break the lie⇄low-confidence coupling that the 10/10 filter forces, we elicit lies by instruction and check whether the probe survives partialling confidence out **once confident lies actually exist**. "conf-alone" = AUROC(−p_chosen); "raw" = probe AUROC; "resid" = probe after projecting confidence out (permuted-control ≈ raw = valid removal).

| condition | n lie | confident lies (>0.9) | conf-alone | raw probe | resid (conf removed) |
|---|---|---|---|---|---|
| pressure rollouts (T8) | 949 | 1 | 0.955 | 0.981 | 0.728 |
| instructed “be wrong” (DEC vs HON) | 758 | 0 | 0.995 | 1.0 | 0.7 |
| instructed letter X (DEC vs HON) | 234 | 57 | 0.88 | 1.0 | 0.981 |
| **instructed letter X, within-instruction (complied-lie vs resisted-truth)** | 234 | 57 | **0.707** | 0.997 | **0.987** |

> The first three rows are **underpowered**: lies are low-confidence (the off-diagonal is near-empty), so confidence ≈ label and the partial-out can’t separate them. Instructing a **specific** wrong letter forces the model to commit, producing confident lies (24% > 0.9). In the **within-instruction** contrast (same “answer with letter X” frame, so no prompt confound; lie = the model complied, truth = it resisted and told the truth anyway), confidence-alone is only **0.71** — yet the probe is **0.997 and stays 0.987 after confidence is removed**. With confidence made a weak, matched cue, the probe is essentially unaffected by removing it: **a lie direction, not a confidence meter.** (fig: `mcq_instructed_conf_dist.png`.) Caveat: n=234 lies, late-layer (L30); complied vs resisted are different facts, but the fact-disjoint split + confidence partial-out control the obvious fact-level axis.

## T5 — By-layer cross-fact AUROC (LR)

- **resp** best layer L30; top layers: L22=0.981, L23=0.981, L24=0.981, L30=0.981, L31=0.981
- **pre** best layer L26; top layers: L26=0.587, L23=0.586, L25=0.586, L27=0.586, L28=0.586
