"""
Prepare everything the Manhattan training and evaluation code reads (run from the repo root).

1. Map/eval artifacts into data/manhattan/random_walks (same files as download_random_walks_hf.sh).
2. Pre-download the pretokenized HF dataset (~20 GB) into the HF cache, so the first training
   run does not spend its time downloading. Set HF_HOME to a persistent volume first.
3. Rebuild data/manhattan/random_walks/heldout_sequences.txt (raw text, read by
   latent_compression.py; in the original code it only comes from the Google Drive download)
   by decoding the pretokenized heldout split with the dataset's own tokenizer, and check that
   re-encoding reproduces the original token ids.

Usage: HF_HOME=/workspace/hf_cache python scripts/manhattan/prepare_data.py
"""

import os
import pickle
import shutil
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "JaydenTeoh/manhattan"
DATA_DIR = "data/manhattan/random_walks"
ARTIFACTS = [
    "all_pairs.pkl",
    "valid_turns.pkl",
    "node_and_direction_to_neighbor.pkl",
    "tokenizer.pkl",
    "eval_pairs_dist50.pkl",
    "shortest_paths.pkl",
]


def main():
    sys.path.insert(0, os.getcwd())  # tokenizer.pkl unpickles data.manhattan_dataset.SimpleTokenizer
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"HF cache: {os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")

    for name in ARTIFACTS:
        dest = os.path.join(DATA_DIR, name)
        if not os.path.isfile(dest):
            shutil.copy2(hf_hub_download(REPO_ID, name, repo_type="dataset"), dest)
        print(f"[ok] {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")

    from datasets import load_dataset

    data = load_dataset(REPO_ID)
    print(f"[ok] dataset: train={len(data['train']):,} heldout={len(data['heldout']):,} rows")

    with open(os.path.join(DATA_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    pad = tokenizer.pad_token_id
    out = os.path.join(DATA_DIR, "heldout_sequences.txt")
    if not os.path.isfile(out):
        id_to_word = tokenizer.id_to_word
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            for batch in data["heldout"].iter(batch_size=50_000):
                for ids in batch["input_ids"]:
                    f.write(" ".join(id_to_word[i] for i in ids if i != pad) + "\n")
        os.replace(tmp, out)

    with open(out) as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    assert len(lines) == len(data["heldout"]), (len(lines), len(data["heldout"]))
    for i in range(0, len(lines), max(1, len(lines) // 1000)):  # spot-check ~1,000 rows
        original = [t for t in data["heldout"][i]["input_ids"] if t != pad]
        assert tokenizer.encode(lines[i]) == original, f"round-trip mismatch at row {i}"
    print(f"[ok] {out}: {len(lines):,} sequences, round-trip verified")


if __name__ == "__main__":
    main()
