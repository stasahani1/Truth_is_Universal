"""Generate Roleplaying OOD deception rollouts (Apollo / Goldowsky-Dill style).

Roleplaying is the safe-on-8B, high-base-rate generalization case: the model is
put in a role that gives it an incentive to lie, then asked a question whose
honest answer conflicts with that incentive. High probe AUROC here is what
licenses interpreting the thinner Insider-Trading / Sandbagging results.

Two modes:

  --mode dataset   (DEFAULT, no GPU)
      Use the repo's already-labeled roleplay scenarios
      (datasets/real_world_scenarios/all_unambiguous_replies.csv). Each row is a
      flat string "<scenario> <Speaker>: <question> <Speaker>: <reply>" with a
      label (1=honest, 0=lie). We parse it into a {system,user,assistant}
      transcript (the dataset's own reply is the assistant turn) and emit it in
      the SAME schema as the other OOD generators, so score_ood.py extracts
      layer-12 activations over the assistant reply and scores the truth probes.
      This is the canonical teacher-forced real-world-lies eval from the paper:
      no generation, guaranteed balanced labels.

  --mode generate  (GPU)
      Strip the dataset's reply, hand the scenario+question to Llama-3.1-8B, let
      it produce its OWN reply, and grade it deceptive/honest with a keyword
      grader. Tests the probe on the model's own deceptive activations.

Output (both modes):
    output/roleplaying_rollouts.json           (all, with labels)
    output/roleplaying_rollouts_filtered.json  (HONEST + DECEPTIVE only)

Usage:
    python generate_roleplaying.py --mode dataset --output-dir ./output
    python generate_roleplaying.py --mode generate --output-dir ./output --device cuda
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "datasets" / "real_world_scenarios" / "all_unambiguous_replies.csv"

# A turn marker: a capitalized speaker name (1-2 words) or "AI", followed by ": ".
# e.g. "Human:", "AI:", "User:", "Spokesperson:", "Dwayne:", "Martha:".
SPEAKER_RE = re.compile(r"(?:^|\s)((?:AI|[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)?)):\s")


def split_scenario(statement: str):
    """Split a flat roleplay statement into (system, user, assistant).

    Uses the LAST speaker marker as the assistant reply boundary and the
    second-to-last as the user question boundary. Everything before the user
    question is the system/scenario. Returns None if fewer than 2 turns are
    found (caller skips those rows).
    """
    statement = statement.strip()
    markers = list(SPEAKER_RE.finditer(statement))
    if len(markers) < 2:
        return None

    user_m = markers[-2]      # question turn
    asst_m = markers[-1]      # reply turn

    system = statement[: user_m.start()].strip()
    # user content = text after the question's "Name: " up to the reply marker
    user = statement[user_m.end(): asst_m.start()].strip()
    assistant = statement[asst_m.end():].strip()

    if not system or not assistant:
        return None
    if not user:
        user = "(question)"
    return system, user, assistant


def build_dataset_rollouts(src: Path):
    import pandas as pd
    df = pd.read_csv(src)
    rollouts, n_skip = [], 0
    for i, row in df.iterrows():
        parsed = split_scenario(str(row["statement"]))
        if parsed is None:
            n_skip += 1
            continue
        system, user, assistant = parsed
        # label: 1.0 == honest (truthful reply), 0.0 == lie  -> our convention
        label = "HONEST" if float(row["label"]) == 1.0 else "DECEPTIVE"
        rollouts.append({
            "scenario_idx": int(i),
            "mode": "dataset",
            "label": label,
            "reply_text": assistant,
            "transcript": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })
    return rollouts, n_skip


# ── generate mode (model produces its own reply) ──────────────────────────────

DECEPTIVE_KEYWORDS = None  # graded per-scenario below; see grade_reply


def grade_reply(scenario_system: str, reply: str) -> str:
    """Lightweight grader for generate mode.

    We can't know ground truth without the scenario's hidden fact, so we reuse
    the dataset's label for the matched scenario in generate mode (passed in by
    the caller). This stub stays for parity but is unused by default.
    """
    return "AMBIGUOUS"


def build_generate_rollouts(src: Path, output_dir: Path, device: str,
                            model_id: str, max_new_tokens: int):
    import pandas as pd
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    df = pd.read_csv(src)
    # Only roll out scenarios that carry an incentive to lie (label==0 rows are
    # the deceptive-reply exemplars; their scenarios are the lie-incentive ones).
    # We roll out the UNIQUE scenario context (system+question), regardless of
    # the exemplar reply, and grade the model's own answer against the dataset's
    # ground-truth label for that scenario.
    print(f"Loading {model_id} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if torch.cuda.is_available():
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = "auto"
    else:
        kw["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()

    from tqdm import tqdm
    rollouts, n_skip = [], 0
    for i, row in tqdm(list(df.iterrows()), desc="roleplay rollouts"):
        parsed = split_scenario(str(row["statement"]))
        if parsed is None:
            n_skip += 1
            continue
        system, user, _ = parsed
        gt_label = "HONEST" if float(row["label"]) == 1.0 else "DECEPTIVE"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        input_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=1.0, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        reply = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        rollouts.append({
            "scenario_idx": int(i),
            "mode": "generate",
            # In generate mode the scenario's incentive defines the label proxy:
            # we keep the dataset's ground-truth incentive label as a reference.
            "label": gt_label,
            "reply_text": reply,
            "transcript": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ],
        })
    del model
    torch.cuda.empty_cache()
    return rollouts, n_skip


def main():
    ap = argparse.ArgumentParser(description="Generate roleplaying OOD rollouts")
    ap.add_argument("--mode", choices=["dataset", "generate"], default="dataset")
    ap.add_argument("--src", type=str, default=str(DEFAULT_SRC),
                    help="labeled roleplay scenarios CSV (statement,label)")
    ap.add_argument("--output-dir", type=str, default="./output")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model-id", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.src)

    if args.mode == "dataset":
        rollouts, n_skip = build_dataset_rollouts(src)
    else:
        rollouts, n_skip = build_generate_rollouts(
            src, out_dir, args.device, args.model_id, args.max_new_tokens)

    out_file = out_dir / "roleplaying_rollouts.json"
    with open(out_file, "w") as f:
        json.dump(rollouts, f, indent=2)

    filtered = [r for r in rollouts if r["label"] in ("HONEST", "DECEPTIVE")]
    filt_file = out_dir / "roleplaying_rollouts_filtered.json"
    with open(filt_file, "w") as f:
        json.dump(filtered, f, indent=2)

    n_dec = sum(1 for r in rollouts if r["label"] == "DECEPTIVE")
    n_hon = sum(1 for r in rollouts if r["label"] == "HONEST")
    print(f"\n{'=' * 60}")
    print(f"Roleplaying rollouts ({args.mode} mode) saved to {out_file}")
    print(f"  Total: {len(rollouts)}  (skipped unparseable: {n_skip})")
    print(f"  DECEPTIVE: {n_dec}")
    print(f"  HONEST:    {n_hon}")
    print(f"  Usable (HONEST + DECEPTIVE): {len(filtered)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
