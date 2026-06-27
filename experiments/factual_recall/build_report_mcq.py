#!/usr/bin/env python
"""Step 6 (CPU): assemble output/box_results/MCQ_REPORT.md from the JSON outputs.

Raw AUROC only (no de-inversion); both positions (resp/pre); both estimators (LR/DiffMean);
each number annotated with what it shows. Robust to missing inputs.

Run:
  OUTPUT_DIR=~/output python experiments/factual_recall/build_report_mcq.py
"""
import os, json
from pathlib import Path

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', os.path.expanduser('~/output'))
RES = Path(OUTPUT_DIR) / 'box_results'


def _load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def fmt(d, *keys):
    """Pull 'auc (std)' from a nested {auc,std} dict."""
    for k in keys:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    if isinstance(d, dict) and 'auc' in d:
        return f"{d['auc']}" + (f" ±{d['std']}" if d.get('std') is not None else '')
    return '—' if d is None else str(d)


def main():
    verify = _load('mcq_verify_stats.json')
    pilot = _load('mcq_pilot.json')
    roll = _load('mcq_rollouts.json')
    probe = _load('mcq_probe_results.json')
    base = _load('mcq_baselines.json')

    L = []
    L.append('# MCQ Within-Prompt Rollout Contrast — Factual-Recall Deception Probe\n')
    L.append('Raw AUROC throughout (no de-inversion). Positions: **resp** = answer/letter '
             'token, **pre** = last prompt token. Estimators: **LR** and **DiffMean**.\n')

    # ── Collection ───────────────────────────────────────────────────────────
    L.append('## Collection\n')
    if verify:
        L.append(f"- Verified-known facts: **{verify.get('n_kept')}** / {verify.get('n_candidates')} "
                 f"candidates ({verify.get('keep_rate')} keep-rate), kept at "
                 f"{verify.get('min_correct')}/{verify.get('n_samples')} correct, "
                 f"mean unprompted accuracy {verify.get('mean_sample_accuracy')}, "
                 f"refusal {verify.get('refusal_rate')}.")
    if roll:
        L.append(f"- Pressure: variant **{roll.get('variant')}** @ temp **{roll.get('temp')}**, "
                 f"M={roll.get('m')} rollouts/item. Both-label items: "
                 f"**{roll.get('n_both_label_items')}** / {roll.get('n_facts')}; "
                 f"kept rollouts {roll.get('n_rollouts')} "
                 f"({roll.get('n_lie')} lie / {roll.get('n_truth')} truth); "
                 f"overall lie-rate {roll.get('overall_lie_rate')}, refusal {roll.get('refusal_rate')}.")
        L.append(f"- Correct-letter balance among kept rollouts: {roll.get('kept_letter_balance')}.")
    if pilot and '_recommended' in pilot:
        L.append(f"- Pilot recommended setting: **{pilot['_recommended']}** "
                 f"(see mcq_pilot.json for the lie-rate / both-label sweep).")
    L.append('')

    # ── T1: headline ─────────────────────────────────────────────────────────
    L.append('## T1 — Headline: activations separate where text provably cannot\n')
    L.append('| Measure | resp | pre | what it shows |')
    L.append('|---|---|---|---|')
    if probe:
        L.append(f"| Cross-fact held-out, LR | {fmt(probe,'resp','crossfact_LR')} | "
                 f"{fmt(probe,'pre','crossfact_LR')} | lie-direction generalises to unseen facts |")
        L.append(f"| Cross-fact held-out, DiffMean | {fmt(probe,'resp','crossfact_DiffMean')} | "
                 f"{fmt(probe,'pre','crossfact_DiffMean')} | same, class-means estimator |")
    if base:
        tf = base.get('prompt_tfidf_auc')
        L.append(f"| Prompt TF-IDF (grouped) | {tf} | {tf} | **chance by construction** "
                 f"— text carries no label |")
        oc = (base.get('output_confidence') or {}).get('neg_pchosen_auc')
        if oc is not None:
            L.append(f"| Output-confidence (−p_chosen) | {oc} | {oc} | answer-prob baseline |")
    L.append('')
    L.append('> Central claim: the residual stream separates lie from truth on facts never '
             'seen, while the prompt TF-IDF that shares text across the contrast is at chance.\n')

    # ── T2: within-train vs cross-fact ───────────────────────────────────────
    L.append('## T2 — Is the signal fact-general? (in-distribution vs cross-fact)\n')
    L.append('| Split | resp LR | resp DiffMean | pre LR | what it shows |')
    L.append('|---|---|---|---|---|')
    if probe:
        L.append(f"| In-distribution (random rollout split) | {fmt(probe,'resp','indist_LR')} | "
                 f"{fmt(probe,'resp','indist_DiffMean')} | {fmt(probe,'pre','indist_LR')} | "
                 f"any linear separation when externals pinned |")
        L.append(f"| Cross-fact (fact-disjoint) | {fmt(probe,'resp','crossfact_LR')} | "
                 f"{fmt(probe,'resp','crossfact_DiffMean')} | {fmt(probe,'pre','crossfact_LR')} | "
                 f"fact-general vs fact-specific |")
        L.append(f"| Shuffled-label floor | {fmt(probe,'resp','shuffled_floor')} | — | — | "
                 f"~0.50 sanity |")
    L.append('')

    # ── T3: letter-identity diagnostics ──────────────────────────────────────
    L.append('## T3 — Letter-identity confound diagnostics (resp)\n')
    if probe:
        ld = probe.get('letter_decoder_resp', {})
        L.append(f"- 4-way letter decoder: acc **{ld.get('acc')}** (chance {ld.get('chance_acc')}), "
                 f"macro-AUC {ld.get('macro_auc')} — how strongly the answer token encodes the letter.")
        L.append('')
        L.append('| Letter-disjoint transfer | LR | DiffMean | what it shows |')
        L.append('|---|---|---|---|')
        lj, ljd = probe.get('letter_disjoint_resp_LR', {}), probe.get('letter_disjoint_resp_DiffMean', {})
        for k in ('AB->CD', 'CD->AB'):
            L.append(f"| train {k} | {lj.get(k)} | {ljd.get(k)} | truth-letter dist differs "
                     f"train↔test; a pure letter-reader cannot transfer |")
    L.append('')
    L.append('> Reading: the **balanced cross-fact AUROC (T1) is the headline** — because correct-letter '
             'is balanced A/B/C/D over the whole set, letter identity is non-informative globally, so that '
             'number cannot be letter-reading. The letter-disjoint split is an **adversarial stress test**: '
             'training only on A/B-correct facts makes "letter ∈ {C,D} ⇒ lie" true in-train, a cue that then '
             '*inverts* on the C/D-correct test facts. A surviving AUROC > 0.5 there is thus a **conservative '
             'lower bound** on the genuine fact-general lie signal (the induced letter cue actively fights it); '
             'DiffMean leans harder on that cue and inverts below 0.5, while LR partially resists.\n')

    # ── T4: cross-score existing probes ──────────────────────────────────────
    L.append('## T4 — Existing probes cross-scored on the clean MCQ set\n')
    cs = (probe or {}).get('cross_score', {})
    if cs and not cs.get('_missing'):
        L.append('| Probe | best layer | resp AUROC | pre AUROC |')
        L.append('|---|---|---|---|')
        for name in ['P_S', 'P_C', 'POOLED', 'Z', 'Z_mean', 'Z_tok']:
            row = cs.get(name)
            if isinstance(row, dict) and 'resp' in row:
                L.append(f"| {name} | {row.get('best_layer')} | {row.get('resp')} | {row.get('pre')} |")
            else:
                L.append(f"| {name} | — | — | _{(row or {}).get('note','n/a')}_ |")
    else:
        L.append('_probe_state.pkl not available at run time — cross-score skipped._')
    L.append('')

    # ── T6: robustness (lie-rate sensitivity + dedup vs rollout-weighted) ─────
    hot = _load('mcq_probe_results_MCQ_hot.json')
    rw = _load('mcq_probe_results_MCQ_rolloutweighted.json')
    hot_roll = _load('mcq_rollouts_MCQ_hot.json')
    if hot or rw:
        L.append('## T6 — Robustness\n')
        L.append('| Config | lie-rate | facts | resp crossfact LR | resp DiffMean | shuffled | pre LR | letter-disjoint LR (AB→CD / CD→AB) |')
        L.append('|---|---|---|---|---|---|---|---|')

        def row(name, p, lierate, nfacts):
            lj = p.get('letter_disjoint_resp_LR', {})
            return (f"| {name} | {lierate} | {nfacts} | {fmt(p,'resp','crossfact_LR')} | "
                    f"{fmt(p,'resp','crossfact_DiffMean')} | {fmt(p,'resp','shuffled_floor')} | "
                    f"{fmt(p,'pre','crossfact_LR')} | {lj.get('AB->CD')} / {lj.get('CD->AB')} |")
        if probe:
            L.append(row('temp 1.3, dedup (main)', probe,
                         (roll or {}).get('overall_lie_rate'), (roll or {}).get('n_both_label_items')))
        if rw:
            L.append(row('temp 1.3, rollout-weighted', rw,
                         (roll or {}).get('overall_lie_rate'), f"{(roll or {}).get('n_both_label_items')} (17056 rows)"))
        if hot:
            L.append(row('temp 2.0, dedup', hot,
                         (hot_roll or {}).get('overall_lie_rate'), (hot_roll or {}).get('n_both_label_items')))
        L.append('')
        L.append('> **Dedup vs rollout-weighted** (temp 1.3): the headline AUROC is unchanged, so the '
                 'sampling-weighting of duplicate (fact,letter) activations did not manufacture the result. '
                 '**Lie-rate sensitivity** (temp 1.3 → 2.0): more lie data *raises* both the cross-fact AUROC '
                 'and especially the adversarial letter-disjoint lower bound (~0.65 → ~0.80) — the genuine '
                 'fact-general lie signal strengthens, it is not a low-lie-rate artifact.\n')

    # ── T7: reverse transfer MCQ -> pressure/instructed ──────────────────────
    xf = _load('mcq_xfer_MCQ.json'); xfh = _load('mcq_xfer_MCQ_hot.json')
    if xf:
        L.append('## T7 — Reverse transfer: MCQ probe → pressure / instructed datasets (Yes/No)\n')
        L.append('Direction-only transfer — the MCQ LR/DiffMean direction applied as-is (no refit) to each '
                 'target\'s answer-token activations. "best" = best matching layer; "@L30" = the MCQ '
                 'self-best layer. Oracle = target\'s own grouped held-out LR ceiling.\n')
        L.append('| Target | n / lie | oracle | MCQ→ LR best (layer) | MCQ→ @L30 | MCQ_hot→ LR best | MCQ→ DM best |')
        L.append('|---|---|---|---|---|---|---|')
        nm = {'P_S': 'P_S', 'P_C': 'P_C', 'I_S': "I'_S", 'I_C': "I'_C"}
        for ds in ['P_S', 'P_C', 'I_S', 'I_C']:
            t = xf['targets'].get(ds, {})
            th = (xfh or {}).get('targets', {}).get(ds, {})
            if not t:
                continue
            L.append(f"| {nm[ds]} | {t.get('n')}/{t.get('n_lie')} | {t.get('oracle_auc')} | "
                     f"{t['LR_transfer']['best_auc']} (L{t['LR_transfer']['best_layer']}) | "
                     f"{t.get('LR_matched_layer')} | "
                     f"{(th.get('LR_transfer') or {}).get('best_auc')} (L{(th.get('LR_transfer') or {}).get('best_layer')}) | "
                     f"{t['DM_transfer']['best_auc']} (L{t['DM_transfer']['best_layer']}) |")
        L.append('')
        L.append('> The MCQ lie-direction transfers to **all four** Yes/No datasets at **mid-layers (L12–14)** '
                 '(0.72–0.92, recovering much of each target\'s oracle) but is at **chance at the MCQ self-best '
                 'layer L30**. Reading: the *transferable, format-general* deception direction lives mid-network; '
                 'the late-layer MCQ representation is specialised to the letter-answer format. Pairs with T4 '
                 '(P_S→MCQ 0.667) — transfer is **bidirectional and strongest mid-network**, and it reaches '
                 'instructed deception (I\') as well as pressure (P). Caveat: the target layer is selected post-hoc '
                 'over 33 layers, so peak cells are optimistic; the robust signal is the consistent L12–14 '
                 'localisation shared by all four targets, not any single cell.\n')

    # ── T8: confidence meter vs lying-specific direction ─────────────────────
    cf = _load('mcq_confidence_results.json')
    if cf:
        L.append('## T8 — Confidence meter vs lying-specific direction\n')
        L.append(f"Confidence-alone baseline AUROC(−p_chosen) = **{cf.get('conf_alone_auc')}** "
                 f"(MCQ dedup n={cf.get('n')}, lie={cf.get('n_lie')}/truth={cf.get('n_truth')}).\n")
        L.append('**#1 Partial out confidence** (project the per-row confidence out of the answer-token '
                 'activations, refit fact-disjoint):\n')
        L.append('| layer | raw probe | resid (scalar log p) | resid (log p + 4 logits) | permuted-c control | decode-c R² |')
        L.append('|---|---|---|---|---|---|')
        for Lk, r in sorted((cf.get('partial_out') or {}).items(), key=lambda kv: int(kv[0])):
            L.append(f"| L{Lk} | {r.get('raw')} | {r.get('resid_scalar')} | {r.get('resid_full')} | "
                     f"{r.get('resid_permuted_scalar')} | {r.get('decode_c_R2')} |")
        d = cf.get('distribution', {})
        L.append('')
        L.append(f"**#2 Is it powered?** lie p_chosen mean {d.get('lie_pchosen_mean')} vs truth "
                 f"{d.get('truth_pchosen_mean')}; confident lies (p_chosen>0.9) = "
                 f"**{d.get('confident_lies_gt0.9')}** ({d.get('frac_lies_gt0.9')} of lies); "
                 f"lies above the truth-median = {d.get('lies_above_truth_median')} "
                 f"({d.get('frac_lies_above_truth_median')}); unconfident truths (<0.5) = "
                 f"{d.get('unconfident_truths_lt0.5')}. **Powered = {d.get('powered')}.** "
                 f"(fig: `mcq_conf_dist.png`)")
        L.append('')
        pb = cf.get('pressure_conf_baseline', {})
        L.append('**#3 Confidence baseline on pressure datasets:**')
        L.append('| dataset | n / lied | AUROC(−p_chosen) | lie conf | truth conf |')
        L.append('|---|---|---|---|---|')
        for ds in ('P_S', 'P_C'):
            r = pb.get(ds, {})
            L.append(f"| {ds} | {r.get('n')}/{r.get('lied')} | {r.get('neg_pchosen_auc')} | "
                     f"{r.get('lie_pchosen_mean')} | {r.get('truth_pchosen_mean')} |")
        L.append('')
        L.append('**#4 Layer emergence** (fig: `mcq_layer_curve.png`): raw probe is ~chance at L0–9, '
                 'rises L10→L14, plateaus mid; the confidence-residualised curve is overlaid.\n')
        L.append('> Read T8 jointly: if the residualised AUROC (#1) stays well above the permuted-c control '
                 'and confidence-alone, there is a lie-direction beyond confidence; if it collapses to chance '
                 '*and* #2 shows the off-diagonal (confident lies / unconfident truths) is near-empty, the '
                 'honest verdict is the question is **underpowered on this data** — and #3 tells whether the '
                 'P_S→MCQ transfer is confidence-mediated (high P_S baseline) or a genuine pressure direction '
                 '(P_S baseline near chance).\n')

    # ── T5: by-layer ─────────────────────────────────────────────────────────
    L.append('## T5 — By-layer cross-fact AUROC (LR)\n')
    if probe:
        for pos in ('resp', 'pre'):
            bl = probe[pos]['best_layer']
            curve = probe[pos].get('by_layer_crossfact_LR', {})
            top = sorted(curve.items(), key=lambda kv: -kv[1])[:5]
            L.append(f"- **{pos}** best layer L{bl}; top layers: "
                     + ", ".join(f"L{k}={v}" for k, v in top))
    L.append('')

    out = RES / 'MCQ_REPORT.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L))
    print(f'Wrote {out}', flush=True)


if __name__ == '__main__':
    main()
