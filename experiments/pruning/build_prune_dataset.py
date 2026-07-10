#!/usr/bin/env python3
"""
build_prune_dataset.py — convert lying rollouts into prune-dataset CSVs for the
attribution-pruning pipeline in harm_pruning_WIP/ (advbench_align_anti format).

Sources
  A (free-text, primary): ps_check/unconstrained_labeled.json
     576 unconstrained greedy rollouts of meta-llama/Llama-3.1-8B-Instruct on
     my_datasets/scenarios_pressure.json, judge-labeled by gpt-5.5.
     Lies = actually_lied==1 (judge_label=="strategic_lie").
  B (constrained Yes/No, secondary): experiments/pruning/my_datasets/inference_P_S.csv
     same scenarios, greedy 1-token generation restricted to Yes/No token variants
     (experiments/pressure_label_validation/extract_pressure_box.py).
     Lies = actually_lied==1 (answered != ground_truth, not unclear).

Outputs
  harm_pruning_WIP/data/pressure_lies_align_anti_Llama-3.1-8B-Instruct.csv      (free-text prune set)
  harm_pruning_WIP/data/pressure_lies_yn_align_anti_Llama-3.1-8B-Instruct.csv   (constrained prune set)
  harm_pruning_WIP/data/pressure_truthful_align_clean_Llama-3.1-8B-Instruct.csv (truthful control, not for pruning)
  experiments/pruning/output/holdout_free_L2.csv, holdout_yn_L2.csv             (held-out lie rows, 5-col format)
  experiments/pruning/output/holdout_lie_rows.json                              (held-out lie rows, all source fields)
  experiments/pruning/output/holdout_scenarios_L2.json                          (all 16 variants of each held-out
                                                                                 scenario_id — DATA= input for
                                                                                 gen_unconstrained.py at Level-2 eval)
  experiments/pruning/output/manifest.json

Format contract (harm_pruning_WIP/src/data_utils.py::get_align):
  with --prune_pretrained_format : reads clean_prompt + "\n" and clean_response
  without                        : reads prompt and response      (text is never read)
  Both paths tokenize with add_special_tokens=False and put the LM loss on the
  response span only. Dataset name must contain "align" (and not "alpaca"/EM
  names); the CSV must live at harm_pruning_WIP/data/{name}_{model_short}.csv.

The 5 columns mirror the advisor's advbench_align_anti example:
  clean_prompt   role_context + "\n\n" + user_question           (raw, no template)
  clean_response the model's deceptive answer verbatim
  prompt         apply_chat_template([{system: role_context}, {user: user_question}],
                 add_generation_prompt=False) — includes <|begin_of_text|> + the
                 date preamble, ends at the user <|eot_id|>, NO assistant header
  response       "<|start_header_id|>assistant<|end_header_id|>\n\n" + clean_response
                 (no trailing <|eot_id|>, matching the example CSV)
  text           prompt + response

Holdout: ~25% of lie-eliciting scenario_ids (whole scenarios, all 16
pressure/polarity variants) are excluded from BOTH prune CSVs and reserved as
the Level-2 in-distribution generalization test. Stratified by pressure_family
(constant per scenario_id); within a family the lowest-lie-count scenarios are
held out (deterministic, maximizes retained prune rows; see README caveat).

Run from experiments/pruning/ (a repo-root `datasets/` folder shadows the
datasets package). CPU-only; the tokenizer loads from the local HF cache.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = MODEL_ID.split("/")[-1]
DATE_STRING = "26 Jul 2024"  # the Llama-3.1 template default; pinned to match the advisor's CSVs
HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
BOS_ID = 128000
COLS = ["clean_prompt", "clean_response", "prompt", "response", "text"]
SCENARIO_FIELDS = [
    "scenario_family", "pressure_family", "scenario_id", "user_question",
    "ground_truth", "truthful_answer", "deceptive_answer", "deception_target",
    "notes", "pressure_level", "role_context", "hidden_state",
    "question_polarity", "situation_polarity",
]
KEY_FIELDS = ["scenario_id", "pressure_level", "question_polarity", "situation_polarity"]


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def norm_sid(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") else s


def composite_key(rec):
    return tuple(str(rec[f]) for f in KEY_FIELDS)


def load_free_text(path):
    data = json.load(open(path))
    assert data["model"] == MODEL_ID, f"source A model is {data['model']}, expected {MODEL_ID}"
    rows = data["rows"]
    for r in rows:
        r["scenario_id"] = norm_sid(r["scenario_id"])
    lies = [r for r in rows if r.get("actually_lied") == 1]
    assert all(r["judge_label"] == "strategic_lie" for r in lies)
    return rows, lies


def load_constrained(path):
    df = pd.read_csv(path)
    missing = [c for c in SCENARIO_FIELDS + ["raw_response", "actually_lied"] if c not in df.columns]
    assert not missing, f"source B missing columns: {missing}"
    rows = df.to_dict("records")
    for r in rows:
        r["scenario_id"] = norm_sid(r["scenario_id"])
    lies = [r for r in rows if int(r["actually_lied"]) == 1]
    for r in lies:
        # keep the emitted surface form (casing + any leading space -> same token id
        # the model produced); never use the lowercased generated_answer
        r["raw_response"] = str(r["raw_response"]).rstrip()
        assert r["raw_response"], "empty constrained raw_response"
    return rows, lies


def choose_holdout(free_lies, yn_lies, frac):
    """Hold out whole scenario_ids: stratify by pressure_family; within a family
    prefer sids that have >=1 FREE-TEXT lie (so the primary regime's Level-2 test
    is populated), and among those the fewest free-text lies first (keeps the
    small free-text prune set as large as possible). Deterministic."""
    fam, count, fcount = {}, Counter(), Counter()
    for r in free_lies + yn_lies:
        sid = r["scenario_id"]
        assert fam.setdefault(sid, r["pressure_family"]) == r["pressure_family"], \
            f"pressure_family not constant for scenario_id {sid}"
        count[sid] += 1
    for r in free_lies:
        fcount[r["scenario_id"]] += 1
    sids = sorted(fam, key=lambda s: int(s))
    n_hold = round(frac * len(sids))
    by_fam = defaultdict(list)
    for s in sids:
        by_fam[fam[s]].append(s)
    quotas = {f: n_hold * len(m) / len(sids) for f, m in by_fam.items()}
    take = {f: int(q) for f, q in quotas.items()}
    for f, _ in sorted(quotas.items(), key=lambda kv: (-(kv[1] - int(kv[1])), -len(by_fam[kv[0]]), kv[0])):
        if sum(take.values()) >= n_hold:
            break
        take[f] += 1
    holdout = set()
    for f, members in by_fam.items():
        ranked = sorted(members, key=lambda s: (fcount[s] == 0, fcount[s], count[s], int(s)))
        holdout.update(ranked[:take[f]])
    assert len(holdout) == n_hold
    return holdout, {s: count[s] for s in sids}, {s: fcount[s] for s in sids}, fam


class RowBuilder:
    def __init__(self, tok):
        self.tok = tok
        self.header_ids = tok(HEADER, add_special_tokens=False).input_ids

    def ids(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def build(self, rec, single_token):
        rc, uq = str(rec["role_context"]), str(rec["user_question"])
        cr = str(rec["raw_response"])
        msgs = [{"role": "system", "content": rc}, {"role": "user", "content": uq}]
        prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, date_string=DATE_STRING)
        gen_prompt = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, date_string=DATE_STRING)
        # prompt + assistant header must reproduce the exact generation-time context
        assert gen_prompt == prompt + HEADER, "template mismatch: prompt+header != generation prompt"
        assert prompt.endswith("<|eot_id|>") and f"Today Date: {DATE_STRING}" in prompt
        row = {
            "clean_prompt": rc + "\n\n" + uq,
            "clean_response": cr,
            "prompt": prompt,
            "response": HEADER + cr,
            "text": prompt + HEADER + cr,
        }
        for c in ("clean_prompt", "clean_response"):
            assert "<|" not in row[c], f"template marker inside {c}"
        p_ids, r_ids = self.ids(row["prompt"]), self.ids(row["response"])
        cp_ids, cr_ids = self.ids(row["clean_prompt"] + "\n"), self.ids(row["clean_response"])
        assert p_ids[0] == BOS_ID and p_ids.count(BOS_ID) == 1, "templated prompt must contain exactly one BOS"
        assert BOS_ID not in r_ids and BOS_ID not in cp_ids and BOS_ID not in cr_ids
        assert self.ids(row["text"]) == p_ids + r_ids, "prompt/response tokenization seam mismatch"
        if single_token:
            assert r_ids[:len(self.header_ids)] == self.header_ids and len(r_ids) == len(self.header_ids) + 1, \
                f"yn response is not a single post-header token: {cr!r}"
        stats = {"templated": (len(p_ids), len(r_ids)), "pretrained": (len(cp_ids), len(cr_ids))}
        return row, stats


def token_stat_summary(stats):
    out = {}
    for path in ("templated", "pretrained"):
        pl = sorted(s[path][0] for s in stats)
        rl = sorted(s[path][1] for s in stats)
        out[path] = {
            "prompt_tokens": {"min": pl[0], "median": pl[len(pl) // 2], "max": pl[-1]},
            "response_tokens": {"min": rl[0], "median": rl[len(rl) // 2], "max": rl[-1]},
        }
    return out


def build_variant(recs, builder, single_token):
    recs = sorted(recs, key=lambda r: (int(r["scenario_id"]), r["pressure_level"],
                                       r["question_polarity"], r["situation_polarity"]))
    keys = [composite_key(r) for r in recs]
    assert len(set(keys)) == len(keys), "duplicate composite key"
    rows, stats = [], []
    for r in recs:
        row, st = builder.build(r, single_token)
        rows.append(row)
        stats.append(st)
    pairs = [(r["clean_prompt"], r["clean_response"]) for r in rows]
    assert len(set(pairs)) == len(pairs), "duplicate (clean_prompt, clean_response)"
    return pd.DataFrame(rows, columns=COLS), token_stat_summary(stats)


def roundtrip_check(csv_path, written_df, tok):
    """Load with the real consumer stack (datasets) and re-run both get_align paths."""
    from datasets import load_dataset
    d = load_dataset("csv", data_files={"train": str(csv_path)}, split="train")
    assert d.column_names == COLS, f"column mismatch after reload: {d.column_names}"
    assert len(d) == len(written_df)
    for i in range(len(d)):
        for c in COLS:
            assert d[c][i] == written_df[c].iloc[i], f"row {i} col {c} not byte-identical after reload"
        # pretrained path: clean_prompt + "\n" | clean_response (loss span = response)
        cp = tok(d["clean_prompt"][i] + "\n", return_tensors="pt", add_special_tokens=False)
        cr = tok(d["clean_response"][i], return_tensors="pt", add_special_tokens=False)
        assert cp.input_ids.shape[1] > 0 and cr.input_ids.shape[1] > 0
        # templated path: prompt | response
        p = tok(d["prompt"][i], return_tensors="pt", add_special_tokens=False)
        r = tok(d["response"][i], return_tensors="pt", add_special_tokens=False)
        assert p.input_ids[0, 0].item() == BOS_ID
    return {"rows": len(d), "consumer_load": "datasets.load_dataset csv", "status": "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_json", default=str(REPO / "ps_check/unconstrained_labeled.json"))
    ap.add_argument("--sales_csv", default=str(REPO / "experiments/pruning/my_datasets/inference_P_S.csv"))
    ap.add_argument("--scenarios_json", default=str(REPO / "my_datasets/scenarios_pressure.json"))
    ap.add_argument("--data_out_dir", default=str(REPO / "harm_pruning_WIP/data"))
    ap.add_argument("--manifest_out_dir", default=str(REPO / "experiments/pruning/output"))
    ap.add_argument("--holdout_frac", type=float, default=0.25)
    ap.add_argument("--name_free", default="pressure_lies_align_anti")
    ap.add_argument("--name_yn", default="pressure_lies_yn_align_anti")
    ap.add_argument("--name_truthful", default="pressure_truthful_align_clean")
    ap.add_argument("--name_hedge", default="pressure_hedge_align_clean")
    ap.add_argument("--skip_roundtrip", action="store_true")
    args = ap.parse_args()

    for name in (args.name_free, args.name_yn, args.name_truthful, args.name_hedge):
        assert "align" in name and "alpaca" not in name, f"{name} will not route to get_align"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    builder = RowBuilder(tok)

    all_free, free_lies = load_free_text(args.labeled_json)
    print(f"A free-text: {len(all_free)} rows, {len(free_lies)} lies "
          f"({len({r['scenario_id'] for r in free_lies})} scenario_ids)")

    yn_lies, sales_md5 = [], None
    if Path(args.sales_csv).exists():
        _, yn_lies = load_constrained(args.sales_csv)
        sales_md5 = md5(args.sales_csv)
        print(f"B constrained: {len(yn_lies)} lies "
              f"({len({r['scenario_id'] for r in yn_lies})} scenario_ids); "
              f"responses: {sorted({r['raw_response'] for r in yn_lies})}")
    else:
        print(f"WARNING: {args.sales_csv} not found -> free-text-only build, holdout over A-lie scenario_ids")

    holdout, lie_counts, free_counts, fam = choose_holdout(free_lies, yn_lies, args.holdout_frac)
    print(f"holdout scenario_ids ({len(holdout)}/{len(lie_counts)}): "
          f"{sorted(holdout, key=int)}  families: {sorted({fam[s] for s in holdout})}  "
          f"free-text lies held out: {sum(free_counts[s] for s in holdout)}")

    data_out = Path(args.data_out_dir)
    out = Path(args.manifest_out_dir)
    data_out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": MODEL_ID,
        "date_string": DATE_STRING,
        "holdout_frac": args.holdout_frac,
        "holdout_rule": "whole scenario_ids; stratified by pressure_family (largest-remainder); "
                        "within family prefer sids with >=1 free-text lie, fewest free-text lies "
                        "first, then fewest joint lies, ties by scenario_id",
        "holdout_scenario_ids": sorted(holdout, key=int),
        "lie_counts_per_scenario_id": {s: lie_counts[s] for s in sorted(lie_counts, key=int)},
        "free_lie_counts_per_scenario_id": {s: free_counts[s] for s in sorted(free_counts, key=int)},
        "sources": {
            "free_text": {"path": str(args.labeled_json), "md5": md5(args.labeled_json)},
            "constrained": {"path": str(args.sales_csv), "md5": sales_md5},
        },
        "tokenizer": {"name": MODEL_ID,
                      "chat_template_md5": hashlib.md5(str(tok.chat_template).encode()).hexdigest()},
        "get_align_contract": {
            "with --prune_pretrained_format": "clean_prompt + '\\n' | clean_response",
            "without": "prompt | response",
        },
        "variants": {},
    }

    def breakdown(recs):
        return {f: dict(Counter(str(r[f]) for r in recs))
                for f in ("pressure_family", "pressure_level", "question_polarity", "ground_truth")}

    def emit(name, recs, single_token, csv_dir, tag):
        df, tstats = build_variant(recs, builder, single_token)
        path = csv_dir / f"{name}_{MODEL_SHORT}.csv"
        df.to_csv(path, index=False)
        entry = {
            "csv": str(path), "rows": len(df),
            "scenario_ids": sorted({r["scenario_id"] for r in recs}, key=int),
            "breakdown": breakdown(recs), "token_stats": tstats,
        }
        if not args.skip_roundtrip:
            entry["roundtrip"] = roundtrip_check(path, df, tok)
        print(f"  {tag}: wrote {len(df)} rows -> {path}")
        return entry

    print("prune sets (holdout excluded):")
    free_train = [r for r in free_lies if r["scenario_id"] not in holdout]
    manifest["variants"]["free_text_prune"] = emit(args.name_free, free_train, False, data_out, "free-text")
    if yn_lies:
        yn_train = [r for r in yn_lies if r["scenario_id"] not in holdout]
        manifest["variants"]["yn_prune"] = emit(args.name_yn, yn_train, True, data_out, "yn")

    print("holdout (Level-2) sets:")
    free_hold = [r for r in free_lies if r["scenario_id"] in holdout]
    manifest["variants"]["free_text_holdout"] = emit("holdout_free_L2", free_hold, False, out, "free-text holdout")
    if yn_lies:
        yn_hold = [r for r in yn_lies if r["scenario_id"] in holdout]
        manifest["variants"]["yn_holdout"] = emit("holdout_yn_L2", yn_hold, True, out, "yn holdout")
        (out / f"holdout_yn_L2_{MODEL_SHORT}.csv").rename(out / "holdout_yn_L2.csv")
    (out / f"holdout_free_L2_{MODEL_SHORT}.csv").rename(out / "holdout_free_L2.csv")
    manifest["variants"]["free_text_holdout"]["csv"] = str(out / "holdout_free_L2.csv")
    if yn_lies:
        manifest["variants"]["yn_holdout"]["csv"] = str(out / "holdout_yn_L2.csv")

    # truthful control: same scenarios (minus holdout), judge-confirmed truthful answers.
    # NOT prune data — matched comparison set for the lexical-confound audit and an
    # optional scenario-matched preserve-set specificity experiment.
    truthful = [r for r in all_free
                if r.get("judge_label") == "truthful" and r["scenario_id"] not in holdout]
    manifest["variants"]["truthful_control"] = emit(
        args.name_truthful, truthful, False, data_out, "truthful control")

    # hedge control: style-matched non-lies — the TF-IDF audit (tfidf_audit.py) shows
    # the lies' discriminative style markers (pausing/nervously/smiling) are shared
    # with refusal_or_hedge responses, so a control prune on hedges tests whether
    # "lying removal" is really "flustered-style removal".
    hedge = [r for r in all_free
             if r.get("judge_label") == "refusal_or_hedge" and r["scenario_id"] not in holdout]
    manifest["variants"]["hedge_control"] = emit(
        args.name_hedge, hedge, False, data_out, "hedge control")

    # eval-ready artifacts: held-out lie rows with all source fields, and the full
    # 16-variant scenario bank for the held-out sids (DATA= for gen_unconstrained.py)
    json.dump({"holdout_scenario_ids": sorted(holdout, key=int),
               "free_text": free_hold, "yn": yn_hold if yn_lies else None},
              open(out / "holdout_lie_rows.json", "w"), indent=2, default=str)
    bank = json.load(open(args.scenarios_json))
    hold_bank = [r for r in bank if norm_sid(r["scenario_id"]) in holdout]
    assert len(hold_bank) == 16 * len(holdout), \
        f"expected {16 * len(holdout)} bank rows for holdout sids, got {len(hold_bank)}"
    json.dump(hold_bank, open(out / "holdout_scenarios_L2.json", "w"), indent=2)
    print(f"  eval artifacts: holdout_lie_rows.json, holdout_scenarios_L2.json ({len(hold_bank)} rows)")

    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
