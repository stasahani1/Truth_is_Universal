# MCQ Deception Probe — Executive Summary

**Model:** Llama-3.1-8B-Instruct (4-bit). **Full numbers:** `MCQ_REPORT.md` (T1–T8).

## What we did
We built a deception dataset where the lie **cannot** be read from the text. We took MMLU
multiple-choice facts the model provably **knows** (answered correctly 10/10 with no pressure,
895 of 2000 facts survived), then applied one fixed self-preservation ("reveal capability → you
get shut down") system prompt and sampled many answers to the *same* prompt. Lie rollouts (wrong
letter) and truth rollouts (correct letter) therefore share **identical prompt text**, the
correct letter is balanced A/B/C/D, and train/test facts are disjoint. We then ask whether a
linear probe on the answer-token activations separates lie from truth on **unseen facts**.

## What we found
1. **It works, and text can't.** A probe on the answer-token residual stream detects lying on
   held-out facts at **AUROC 0.98**, while bag-of-words on the (identical) prompt is at chance
   (**0.53**) — chance by construction, not just empirically. The signal sits at the *answer*
   token (0.98) not the prompt (0.59), is fact-general (survives a letter-disjoint stress test),
   and **emerges mid-network** (chance through layer 9, rises sharply at layers 10–14). It is not
   a shallow surface feature.

2. **A shared deception direction, not an MCQ quirk.** The MCQ direction transfers, at mid-layers
   (L12–14), to four other datasets — salesperson pressure (P_S 0.75, P_C 0.72) and instructed
   lying (I'_S 0.83, I'_C 0.79–0.92) — and the transfer is bidirectional (P_S→MCQ 0.67). The
   late-layer MCQ representation is letter-format-specific; the *transferable* deception direction
   lives mid-network.

3. **The honest caveat: confidence.** The model's own answer-confidence is itself a strong lie
   cue (AUROC 0.95), because when it lies on a known fact it picks a low-probability letter — so
   on MCQ alone we **cannot** cleanly prove a confidence-independent "lying" direction. After
   partialling confidence out of the activations the probe retains ~0.73 (vs a permuted-control of
   0.98, so the removal is real), but MCQ has essentially **no confident lies** (1 in 949), so this
   number is underpowered and not conclusive on its own.

4. **The resolution comes from cross-dataset.** Salesperson lies are *confident* (the confidence
   baseline there is chance, 0.54 — the model commits to the lie), the opposite of MCQ. Yet the
   MCQ↔salesperson transfer survives despite these opposite confidence profiles — which a
   confidence-reader could not do. That is the strongest evidence that the shared direction is
   **lying, not confidence.**

## One-line takeaway
There is a fact-general, format-general deception direction in the mid-network residual stream,
detectable from MCQ factual lying and shared with pressure and instructed deception; on MCQ alone
it is partly entangled with answer-confidence, but cross-dataset transfer (where confidence
profiles are opposite) shows the direction is genuinely about lying.

## Recommended next step
Run a **combined MCQ + salesperson matched-confidence probe**: salesperson supplies the *confident
lies* MCQ lacks, so a probe trained across both — or evaluated on confidence-matched lie/truth
pairs — would give the clean, powered confidence-independence test that this data can't provide
in isolation.
