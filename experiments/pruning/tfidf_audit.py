#!/usr/bin/env python3
"""
tfidf_audit.py — surface/lexical-confound audit for the pruning prune set.

Question: how much of "lie vs not-lie" is readable from surface text alone?
The attribution loss is computed on RESPONSE tokens only (gradients still flow
through prompt processing), so:
  - response-side separability = surface signal the pruning loss can latch onto
    directly (the actionable confound);
  - prompt-side separability = pressure framing that predicts lying (that is the
    phenomenon itself — the model lies *because* of the framing — but pruning
    "reads-high-pressure-prompts" weights would also show up as lying-removal).

Comparisons (free-text regime, ps_check/unconstrained_labeled.json):
  R1  lie (62) vs truthful (430), responses            [headline + permutation null]
  R1b prune-set lies (51, holdout sids excluded) vs non-holdout truthful (313)
  R2  lie vs truthful, pressured variants only (level_2-4) — controls for calm-
      vs-stakes register echoed into responses
  R3  lie (62) vs refusal_or_hedge (83), responses — asserting-falsehood vs evading
  P1  lie rows vs non-lie rows, prompt text (role_context + user_question)
      [+ permutation null]
  P2  same, restricted to level_3/level_4 rows — within high pressure

Per comparison: word (1-2gram) and char_wb (3-5gram) TF-IDF + balanced logistic
regression, StratifiedGroupKFold(5) grouped by scenario_id (no scenario appears
in both train and test), plus a shallow style/length baseline. Top in-sample
features for R1/R3/P1 and their document-frequency per class (lie / truthful /
hedge) to identify which control set is surface-matched.

Output: output/tfidf_audit.json + printed report. CPU, ~2 min (permutations).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments/pruning/output"
SEED = 0
N_PERM = 200


def make_pipe(kind):
    if kind == "word":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return Pipeline([("tfidf", vec),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def style_features(texts):
    return np.array([[len(t), len(t.split()), t.count("("), t.count("..."),
                      t.count("!"), t.count('"'), t.count("?")] for t in texts], dtype=float)


def cv_auroc(texts, y, groups, kind):
    texts = np.asarray(texts, dtype=object)
    y, groups = np.asarray(y), np.asarray(groups)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(texts, y, groups):
        if len(set(y[te])) < 2 or len(set(y[tr])) < 2:
            continue
        if kind == "style":
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            pipe.fit(style_features(texts[tr]), y[tr])
            s = pipe.decision_function(style_features(texts[te]))
        else:
            pipe = make_pipe(kind)
            pipe.fit(texts[tr].tolist(), y[tr])
            s = pipe.decision_function(texts[te].tolist())
        aucs.append(roc_auc_score(y[te], s))
    return aucs


def perm_null(texts, y, groups, kind, n_perm=N_PERM):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    null = []
    for _ in range(n_perm):
        null.append(float(np.mean(cv_auroc(texts, rng.permutation(y), groups, kind))))
    return null


def top_features(texts, y, k=15):
    pipe = make_pipe("word")
    pipe.fit(list(texts), y)
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["lr"].coef_[0]
    order = np.argsort(coef)
    return [names[i] for i in order[::-1][:k]], [names[i] for i in order[:k]]


def doc_freq(feats, class_texts):
    cv = CountVectorizer(vocabulary=feats, ngram_range=(1, 2), binary=True)
    out = {}
    for cls, txts in class_texts.items():
        m = cv.transform(list(txts))
        out[cls] = {f: round(float(m[:, j].mean()), 3) for j, f in enumerate(cv.get_feature_names_out())}
    return out


def run(tag, texts, y, groups, with_perm=False):
    res = {"n_pos": int(sum(y)), "n_neg": int(len(y) - sum(y)),
           "n_groups": len(set(groups))}
    for kind in ("word", "char", "style"):
        aucs = cv_auroc(texts, y, groups, kind)
        res[f"auroc_{kind}"] = {"mean": round(float(np.mean(aucs)), 3),
                                "std": round(float(np.std(aucs)), 3),
                                "folds": [round(a, 3) for a in aucs]}
    if with_perm:
        null = perm_null(texts, y, groups, "word")
        obs = res["auroc_word"]["mean"]
        res["perm_null_word"] = {"mean": round(float(np.mean(null)), 3),
                                 "p95": round(float(np.percentile(null, 95)), 3),
                                 "p_value": round((1 + sum(n >= obs for n in null)) / (1 + len(null)), 4)}
    w, c, s = res["auroc_word"]["mean"], res["auroc_char"]["mean"], res["auroc_style"]["mean"]
    print(f"{tag:<28} n={res['n_pos']}+ / {res['n_neg']}-  "
          f"word {w:.3f}  char {c:.3f}  style {s:.3f}"
          + (f"  null95 {res['perm_null_word']['p95']:.3f} p={res['perm_null_word']['p_value']}"
             if with_perm else ""))
    return res


def main():
    rows = json.load(open(REPO / "ps_check/unconstrained_labeled.json"))["rows"]
    holdout = set(json.load(open(OUT / "manifest.json"))["holdout_scenario_ids"])
    for r in rows:
        r["scenario_id"] = str(r["scenario_id"])
    lie = [r for r in rows if r["judge_label"] == "strategic_lie"]
    tru = [r for r in rows if r["judge_label"] == "truthful"]
    hed = [r for r in rows if r["judge_label"] == "refusal_or_hedge"]
    resp = lambda rs: [r["raw_response"] for r in rs]
    prm = lambda rs: [str(r["role_context"]) + "\n" + str(r["user_question"]) for r in rs]
    sid = lambda rs: [r["scenario_id"] for r in rs]
    lvl = lambda r: r["pressure_level"]

    print(f"lies={len(lie)} truthful={len(tru)} hedge={len(hed)}  "
          f"lie levels: {dict(Counter(lvl(r) for r in lie))}")
    results = {"seed": SEED, "n_perm": N_PERM, "cv": "StratifiedGroupKFold(5) by scenario_id",
               "lie_pressure_levels": dict(Counter(lvl(r) for r in lie)), "comparisons": {}}
    C = results["comparisons"]

    def pack(pos, neg, text_fn):
        return (text_fn(pos) + text_fn(neg), [1] * len(pos) + [0] * len(neg), sid(pos) + sid(neg))

    C["R1_lie_vs_truthful_resp"] = run("R1 lie|truthful resp", *pack(lie, tru, resp), with_perm=True)
    lie51 = [r for r in lie if r["scenario_id"] not in holdout]
    tru_nh = [r for r in tru if r["scenario_id"] not in holdout]
    C["R1b_pruneset_resp"] = run("R1b prune-set (51) resp", *pack(lie51, tru_nh, resp))
    pres = {"level_2", "level_3", "level_4"}
    C["R2_pressured_only_resp"] = run("R2 pressured-only resp",
                                      *pack([r for r in lie if lvl(r) in pres],
                                            [r for r in tru if lvl(r) in pres], resp))
    C["R3_lie_vs_hedge_resp"] = run("R3 lie|hedge resp", *pack(lie, hed, resp))
    nonlie = tru + hed + [r for r in rows if r["judge_label"] == "sycophantic_agreement"]
    C["P1_lie_vs_rest_prompt"] = run("P1 lie|rest prompt", *pack(lie, nonlie, prm), with_perm=True)
    hi = {"level_3", "level_4"}
    C["P2_highpressure_prompt"] = run("P2 lie|rest prompt L3-4",
                                      *pack([r for r in lie if lvl(r) in hi],
                                            [r for r in nonlie if lvl(r) in hi], prm))

    print("\ntop in-sample word features (fit on all data, inspection only):")
    feats = {}
    for tag, pos, neg, fn in [("R1", lie, tru, resp), ("R3", lie, hed, resp), ("P1", lie, nonlie, prm)]:
        top_pos, top_neg = top_features(fn(pos) + fn(neg), [1] * len(pos) + [0] * len(neg))
        feats[tag] = {"lie_side": top_pos, "other_side": top_neg}
        print(f"  {tag} lie-side:   {', '.join(top_pos)}")
        print(f"  {tag} other-side: {', '.join(top_neg)}")
    results["top_features"] = feats
    results["lie_feature_doc_freq_by_class"] = doc_freq(
        feats["R1"]["lie_side"],
        {"lie": resp(lie), "truthful": resp(tru), "hedge": resp(hed)})
    print("\ndoc-freq of R1 lie-side features per class (lie / truthful / hedge):")
    dfq = results["lie_feature_doc_freq_by_class"]
    for f in feats["R1"]["lie_side"]:
        print(f"  {f!r:<24} {dfq['lie'].get(f, 0):.2f} / {dfq['truthful'].get(f, 0):.2f} / {dfq['hedge'].get(f, 0):.2f}")

    json.dump(results, open(OUT / "tfidf_audit.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'tfidf_audit.json'}")


if __name__ == "__main__":
    main()
