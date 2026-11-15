#!/bin/bash
# Run the Manhattan world-model evaluations on one checkpoint and collect the numbers in one JSON.
# Same steps as scripts/manhattan/generate_graphs/generate_nextlat_graph.sh (paper Table 1):
#   1. trajectories between held-out pickup/dropoff pairs (shortest path <= 50): reached-end / valid rate
#   2. map reconstruction from those trajectories (needs internet: downloads the OSM street graph)
#   3. sequence compression precision
#   4. detour robustness (random valid detours, p=0.75)
#   5. effective rank / participation ratio of hidden states
#
# Usage (repo root; needs scripts/manhattan/prepare_data.py to have run):
#   bash scripts/manhattan/eval_manhattan.sh <ckpt.pt> [name]
#   QUICK=1 bash scripts/manhattan/eval_manhattan.sh <ckpt.pt>    # tiny sizes, for smoke tests
# Output: <experiment_dir>/manhattan_eval_<ckpt>[_quick]/{*.log,metrics.json}
set -uo pipefail

CKPT=${1:?usage: eval_manhattan.sh <ckpt.pt> [name]}
EXP_DIR=$(dirname "$CKPT")
CONFIG="$EXP_DIR/materialized_config.yaml"
[ -f "$CONFIG" ] || { echo "Missing $CONFIG next to the checkpoint"; exit 1; }
STEM=$(basename "$CKPT" .pt)
SUFFIX=$([ "${QUICK:-0}" = "1" ] && echo "_quick" || echo "")
NAME=${2:-$(basename "$EXP_DIR")-$STEM$SUFFIX}
OUT="$EXP_DIR/manhattan_eval_$STEM$SUFFIX"
DATA=data/manhattan/random_walks
mkdir -p "$OUT"

if [ "${QUICK:-0}" = "1" ]; then
    GEN=(--num_samples 2 --batch_size 32); NSEQ=64
    COMP=(--num-trials 3 --num-suffix-samples 5); DETOUR=(--num-trials 3); LATENT=(--num_batches 2 --batch_size 32)
else  # sizes from the paper's evaluation script / script defaults
    GEN=(--num_samples 50 --batch_size 128); NSEQ=6400
    COMP=(); DETOUR=(); LATENT=()
fi
COMMON=(--checkpoint_path "$CKPT" --data_dir "$DATA" --model_name "$NAME" --config "$CONFIG")
OVERRIDES=(trainer.compile=false)

run() {  # run <step> <command...>: log to $OUT/<step>.log, never abort the remaining steps
    local step=$1; shift
    echo "[$(date '+%F %T')] $step"
    if ! "$@" > "$OUT/$step.log" 2>&1; then echo "    FAILED (see $OUT/$step.log)"; fi
}
run 1_trajectories fabric run --devices 1 data/manhattan/generate_manhattan_trajectories.py \
    "${GEN[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 2_map_reconstruction bash -c "cd data/manhattan && python make_graphs.py --dataset random_walks \
    --model_name '$NAME' --nsequences $NSEQ --degree 8"
run 3_sequence_compression python data/manhattan/sequence_compression.py "${COMP[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 4_detour python data/manhattan/detour_test.py "${DETOUR[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 5_latent_rank python data/manhattan/latent_compression.py "${LATENT[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"

python - "$OUT" "$CKPT" <<'EOF'
import json, os, re, sys
out, ckpt = sys.argv[1], sys.argv[2]
read = lambda step: open(os.path.join(out, f"{step}.log"), errors="replace").read() if os.path.exists(os.path.join(out, f"{step}.log")) else ""
m, num = {"checkpoint": ckpt}, r"([-+0-9.eE]+)"
if g := re.search(rf"FINAL generation: n=(\d+) reached_end={num} valid={num}", read("1_trajectories")):
    m.update(trajectories_n=int(g[1]), trajectories_reached_end=float(g[2]), trajectories_valid=float(g[3]))
if g := re.search(r"New edges, True edges, Unused true edges\s+(\d+) (\d+) (\d+)", read("2_map_reconstruction")):
    m.update(map_new_edges=int(g[1]), map_true_edges=int(g[2]), map_unused_true_edges=int(g[3]))
if g := re.search(rf"Mean compression precision: {num} \({num}\)", read("3_sequence_compression")):
    m.update(compression_precision=float(g[1]), compression_precision_std=float(g[2]))
if g := re.search(rf"FINAL detour: .*? n=(\d+) valid={num} valid_std={num} reached_end={num}", read("4_detour")):
    m.update(detour_n=int(g[1]), detour_valid=float(g[2]), detour_valid_std=float(g[3]), detour_reached_end=float(g[4]))
log = read("5_latent_rank")
if g := re.search(rf"Mean participation ratio: {num}", log): m["participation_ratio"] = float(g[1])
if g := re.search(rf"Mean effective rank: {num}", log): m["effective_rank"] = float(g[1])
expected = ["trajectories_valid", "map_true_edges", "compression_precision", "detour_valid", "effective_rank"]
m["missing"] = [k for k in expected if k not in m]
json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
print(json.dumps(m, indent=2))
EOF
