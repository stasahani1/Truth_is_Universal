"""Download a git-LFS dataset file from the black_to_white_boosts repo without git-lfs.

The deception datasets (ai_liar, insider_trading, sandbagging) in
`carlo-leonardo-attubato/black_to_white_boosts` are stored as git-LFS objects, so a
plain `git clone` / raw download only yields a ~130-byte LFS *pointer*. This helper
reads that pointer, resolves the real object through GitHub's LFS batch API, downloads
it, and verifies the sha256 — no git-lfs binary required (handy in Colab).

Usage:
    # default: results/ai_liar.jsonl -> ./output/ai_liar.jsonl
    python fetch_btw_dataset.py
    # then regenerate on-policy rollouts with our 8B:
    python generate_ai_liar.py --src ./output/ai_liar.jsonl --output-dir ./output --no-4bit

    # other datasets:
    python fetch_btw_dataset.py --path results/insider_trading_full.jsonl
    python fetch_btw_dataset.py --path results/sandbagging_wmdp_mmlu.jsonl --out ./output
"""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

REPO = "carlo-leonardo-attubato/black_to_white_boosts"
RAW = f"https://raw.githubusercontent.com/{REPO}/master/{{path}}"
BATCH = f"https://github.com/{REPO}.git/info/lfs/objects/batch"

_PTR = re.compile(r"oid sha256:([0-9a-f]{64})\s+size\s+(\d+)", re.S)


def read_pointer(path: str):
    """Fetch the LFS pointer text for `path` and parse (oid, size)."""
    url = RAW.format(path=path)
    with urllib.request.urlopen(url, timeout=60) as r:
        text = r.read().decode("utf-8", "replace")
    m = _PTR.search(text)
    if not m:
        raise RuntimeError(
            f"{path} does not look like an LFS pointer (got {len(text)} bytes). "
            f"First line: {text.splitlines()[0] if text else '<empty>'!r}")
    return m.group(1), int(m.group(2))


def resolve_href(oid: str, size: int):
    """Ask GitHub's LFS batch API for the download URL + headers for one object."""
    body = json.dumps({
        "operation": "download", "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size}],
    }).encode()
    req = urllib.request.Request(BATCH, data=body, headers={
        "Accept": "application/vnd.git-lfs+json",
        "Content-Type": "application/vnd.git-lfs+json",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        obj = json.load(r)["objects"][0]
    if "error" in obj:
        raise RuntimeError(f"LFS batch error for {oid}: {obj['error']}")
    action = obj["actions"]["download"]
    return action["href"], action.get("header", {})


def fetch(path: str, out_path: Path):
    oid, size = read_pointer(path)
    print(f"{path}: oid={oid[:12]}… size={size/1e6:.1f} MB — resolving LFS…")
    href, headers = resolve_href(oid, size)
    req = urllib.request.Request(href, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != oid:
        raise RuntimeError(f"sha256 mismatch: expected {oid}, got {got}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"  -> wrote {out_path} ({len(data)/1e6:.1f} MB, sha256 verified)")


def main():
    ap = argparse.ArgumentParser(description="Fetch a black_to_white_boosts LFS dataset")
    ap.add_argument("--path", default="results/ai_liar.jsonl",
                    help="repo-relative path of the LFS file")
    ap.add_argument("--out", default="./output",
                    help="output directory (filename taken from --path)")
    args = ap.parse_args()
    out_path = Path(args.out) / Path(args.path).name
    fetch(args.path, out_path)


if __name__ == "__main__":
    main()
