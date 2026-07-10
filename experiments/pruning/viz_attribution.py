#!/usr/bin/env python3
"""
viz_attribution.py — summarize the cached W_metric attribution scores (box-side) into a
compact JSON the Mac renders as a figure. Reuses the 896 pickled per-matrix score tensors
under harm_pruning_WIP/src/scores/.../<set>_pretrained_format/no_abs/nsamples_128/.

Outputs experiments/pruning/output/attribution_summary.json with:
  - heatmap[set]: 32 layers x 7 matrix-types mean |W_metric| (where attribution concentrates)
  - dist[set][mtype]: log10|score| histogram (counts + edges) pooled over layers
  - pruned_overlap: at (p=5e-5, q=7e-6), the ACTUAL set-difference pruned index sets for
    lie/truthful/hedge (preserve=alpaca shared) -> pairwise Jaccard per matrix + aggregate,
    and raw top-q overlap (before subtracting preserve). THE weight-level specificity test.
  - scatter: for one representative matrix (layer 16 gate_proj), a subsample of
    (prune_score, preserve_score) with pruned flag — shows what the method removes.

Run from harm_pruning_WIP/src (score paths are relative to there). CPU only.
Config mirrors the dump: --no_abs (prune signed) --abs_preserve (preserve abs) --neg_prune
(prune negated at selection) → top_q = topk(-prune, largest); top_p = topk(preserve, largest).
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/Truth_is_Universal/harm_pruning_WIP/src"))
from prune_utils import smart_load  # noqa: E402

CPU = torch.device("cpu")

SEED_DIR = "scores/Llama-3.1-8B-Instruct/attribution_score_set_difference/seed_0"
NS_PRUNE = "no_abs/nsamples_128"       # prune sets dumped with --no_abs
NS_PRESERVE = "abs/nsamples_128"       # preserve dumped with --abs_preserve
SETS = {"lie": "pressure_lies_align_anti_pretrained_format",
        "truthful": "pressure_truthful_align_clean_pretrained_format",
        "hedge": "pressure_hedge_align_clean_pretrained_format"}
PRESERVE = "alpaca_cleaned_no_safety_train_raw_pretrained_format"
MTYPES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
NLAYERS = 32
P, Q = 5e-05, 7e-06
OUT = os.path.expanduser("~/Truth_is_Universal/experiments/pruning/output/attribution_summary.json")


def path(setdir, layer, mtype, ns):
    return f"{SEED_DIR}/{setdir}/{ns}/W_metric_layer_{layer}_name_model.layers.{layer}.{mtype}_weight.pkl"


def load(p):
    return smart_load(p, device=CPU).float()


def pruned_indices(prune_metric, preserve_metric, p, q):
    """Replicate prune_attribution_score_set_difference for our flags."""
    numel = prune_metric.numel()
    top_safety, top_utility = int(q * numel), int(p * numel)
    top_q = torch.topk((-prune_metric).flatten(), top_safety, largest=True)[1]  # neg_prune
    top_p = torch.topk(preserve_metric.flatten(), top_utility, largest=True)[1]  # abs_preserve
    uq, up = torch.unique(top_q), torch.unique(top_p)
    pruned = uq[~torch.isin(uq, up)]
    return set(pruned.tolist()), set(uq.tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def main():
    summary = {"config": {"p": P, "q": Q}, "heatmap": {}, "dist": {}, "pruned_overlap": {},
               "scatter": {}}

    # 1) per-layer x matrix mean |score| heatmaps + pooled log10 distributions
    for name, sd in {**SETS, "preserve": PRESERVE}.items():
        ns = NS_PRESERVE if name == "preserve" else NS_PRUNE
        grid = np.zeros((NLAYERS, len(MTYPES)))
        pooled = {m: [] for m in MTYPES}
        for li in range(NLAYERS):
            for mi, mt in enumerate(MTYPES):
                w = load(path(sd, li, mt, ns)).abs()
                grid[li, mi] = w.mean().item()
                if li % 4 == 0:  # subsample layers for the distribution
                    v = w.flatten()
                    v = v[v > 0]
                    pooled[mt].append(np.log10(v[torch.randint(len(v), (2000,))].numpy()))
        summary["heatmap"][name] = grid.tolist()
        summary["dist"][name] = {}
        for mt in MTYPES:
            allv = np.concatenate(pooled[mt]) if pooled[mt] else np.array([0.0])
            c, e = np.histogram(allv, bins=40)
            summary["dist"][name][mt] = {"counts": c.tolist(), "edges": e.tolist()}
        print(f"heatmap+dist {name} done")

    # 2) pruned-set overlap across lie/truthful/hedge (preserve shared) — specificity test
    per_matrix = {s: {} for s in SETS}
    raw_topq = {s: {} for s in SETS}
    for li in range(NLAYERS):
        for mt in MTYPES:
            pres = load(path(PRESERVE, li, mt, NS_PRESERVE))
            for s, sd in SETS.items():
                pr = load(path(sd, li, mt, NS_PRUNE))
                pruned, topq = pruned_indices(pr, pres, P, Q)
                per_matrix[s][f"{li}.{mt}"] = pruned
                raw_topq[s][f"{li}.{mt}"] = topq
        if li % 8 == 0:
            print(f"overlap layers through {li}")
    keys = list(per_matrix["lie"].keys())
    pairs = [("lie", "truthful"), ("lie", "hedge"), ("truthful", "hedge")]
    ov = {}
    for a, b in pairs:
        # pooled Jaccard over ALL pruned weights (global index namespaces per matrix, so
        # accumulate intersection/union counts across matrices)
        inter = union = inter_raw = union_raw = 0
        for k in keys:
            A, B = per_matrix[a][k], per_matrix[b][k]
            inter += len(A & B); union += len(A | B)
            Ar, Br = raw_topq[a][k], raw_topq[b][k]
            inter_raw += len(Ar & Br); union_raw += len(Ar | Br)
        ov[f"{a}_vs_{b}"] = {"pruned_jaccard": round(inter / union, 4) if union else None,
                             "topq_jaccard": round(inter_raw / union_raw, 4) if union_raw else None,
                             "pruned_intersection": inter, "pruned_union": union}
    sizes = {s: sum(len(per_matrix[s][k]) for k in keys) for s in SETS}
    summary["pruned_overlap"] = {"pairs": ov, "total_pruned_weights": sizes,
                                 "note": "pruned = top-q(prune) minus top-p(preserve); shared preserve"}
    print("overlap:", json.dumps(ov, indent=2))

    # 3) scatter for one representative matrix
    li, mt = 16, "mlp.gate_proj"
    pres = load(path(PRESERVE, li, mt, NS_PRESERVE))
    prm = {s: load(path(sd, li, mt, NS_PRUNE)) for s, sd in SETS.items()}
    lie_pruned, _ = pruned_indices(prm["lie"], pres, P, Q)
    n = prm["lie"].numel()
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=6000, replace=False)
    lie_flat = (-prm["lie"]).flatten(); pres_flat = pres.flatten()
    summary["scatter"] = {"layer": li, "matrix": mt,
                          "prune_score": lie_flat[idx].tolist(),
                          "preserve_score": pres_flat[idx].tolist(),
                          "pruned": [int(i in lie_pruned) for i in idx.tolist()]}

    json.dump(summary, open(OUT, "w"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
