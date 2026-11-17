#!/bin/bash
# Post-hoc Path-Star evaluation of a saved checkpoint (solve rate, per-token accuracy,
# normalized latent error per rollout step, gate statistics). No W&B logging, no checkpoint writes.
#
# Usage (from repo root):
#   bash scripts/attend/eval_stargraph.sh <ckpt.pt> [overrides...]
# Examples:
#   bash scripts/attend/eval_stargraph.sh output/attend/5_5/learned/seed1234/Attend-learned-seed1234/ckpt_iter_20000.pt
#   # test-time ablation: evaluate a learned-gate model with uniform head weights
#   bash scripts/attend/eval_stargraph.sh <ckpt> model.head_uniform_gate=true
#
# Results: <experiment_dir>/eval_<ckpt-name>/<experiment_name>/eval_results.json
# On a machine without a GPU, use: LAUNCH=python bash scripts/attend/eval_stargraph.sh ...
set -euo pipefail

CKPT=${1:?usage: eval_stargraph.sh <ckpt.pt> [overrides...]}
shift
EXP_DIR=$(dirname "$CKPT")
CONFIG="$EXP_DIR/materialized_config.yaml"
[ -f "$CONFIG" ] || { echo "Missing $CONFIG next to the checkpoint"; exit 1; }
TAG=$(basename "$CKPT" .pt)
[ $# -gt 0 ] && TAG="${TAG}_$(echo "$*" | tr ' =/[],' '_-____')"
OUT_DIR="$EXP_DIR/eval_$TAG"

LAUNCH=${LAUNCH:-fabric run --devices 1 --precision bf16-mixed}
$LAUNCH train.py --config "$CONFIG" --no_pbar --checkpoint_path "$CKPT" \
    trainer.val_only=true trainer.log_to_wandb=false trainer.compile=false \
    trainer.out_dir="$OUT_DIR" "$@"

echo "Results: $(ls "$OUT_DIR"/*/eval_results.json)"
