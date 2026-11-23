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
#   PARSE_ONLY=1 bash scripts/manhattan/eval_manhattan.sh <ckpt.pt>  # rebuild metrics.json from existing logs
# Other env options:
#   STEPS="4 3"         run only these steps (numbers above; default all)
#   DETOUR_TRIALS=1000  detour trials (default 100)
#   PAIRED=1            detour scenarios seeded per trial, identical across models (detour_test.py --paired)
#   TAG=_paired1000     suffix for the output folder, keeping such runs apart from the standard eval
# Output: <experiment_dir>/manhattan_eval_<ckpt>[_quick][TAG]/{*.log,metrics.json}
set -uo pipefail

CKPT=${1:?usage: eval_manhattan.sh <ckpt.pt> [name]}
EXP_DIR=$(dirname "$CKPT")
CONFIG="$EXP_DIR/materialized_config.yaml"
[ -f "$CONFIG" ] || { echo "Missing $CONFIG next to the checkpoint"; exit 1; }
STEM=$(basename "$CKPT" .pt)
SUFFIX=$([ "${QUICK:-0}" = "1" ] && echo "_quick" || echo "")${TAG:-}
NAME=${2:-$(basename "$EXP_DIR")-$STEM$SUFFIX}
OUT="$EXP_DIR/manhattan_eval_$STEM$SUFFIX"
STEPS=${STEPS:-1 2 3 4 5}
DATA=data/manhattan/random_walks
mkdir -p "$OUT"
touch "$OUT/RUNNING"  # lets finish_manhattan.py wait for an eval already in progress instead of redoing it
trap 'rm -f "$OUT/RUNNING"' EXIT
# The eval scripts create a default lightning Fabric(), which uses every visible GPU and then
# asserts world_size == 1: pin them to one GPU (the first of CUDA_VISIBLE_DEVICES, default 0).
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}

if [ "${QUICK:-0}" = "1" ]; then
    GEN=(--num_samples 2 --batch_size 32); NSEQ=64
    COMP=(--num-trials 3 --num-suffix-samples 5); DETOUR=(--num-trials 3); LATENT=(--num_batches 2 --batch_size 32)
else  # sizes from the paper's evaluation script / script defaults
    GEN=(--num_samples 50 --batch_size 128); NSEQ=6400
    COMP=(); DETOUR=(); LATENT=()
fi
[ -n "${DETOUR_TRIALS:-}" ] && DETOUR+=(--num-trials "$DETOUR_TRIALS")
[ "${PAIRED:-0}" = "1" ] && DETOUR+=(--paired)
COMMON=(--checkpoint_path "$CKPT" --data_dir "$DATA" --model_name "$NAME" --config "$CONFIG")
OVERRIDES=(trainer.compile=false)

run() {  # run <step> <command...>: log to $OUT/<step>.log, never abort the remaining steps
    local step=$1; shift
    [[ " $STEPS " == *" ${step%%_*} "* ]] || return 0
    echo "[$(date '+%F %T')] $step"
    if ! "$@" > "$OUT/$step.log" 2>&1; then echo "    FAILED (see $OUT/$step.log)"; fi
}
# fabric run's default port (29400) clashes with concurrent training or a second eval: use a free one
PORT=$(python -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
if [ "${PARSE_ONLY:-0}" != "1" ]; then  # PARSE_ONLY=1: only rebuild metrics.json from existing logs
run 1_trajectories fabric run --devices 1 --main-port "$PORT" data/manhattan/generate_manhattan_trajectories.py \
    "${GEN[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 2_map_reconstruction bash -c "cd data/manhattan && python make_graphs.py --dataset random_walks \
    --model_name '$NAME' --nsequences $NSEQ --degree 8"
run 3_sequence_compression python data/manhattan/sequence_compression.py "${COMP[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 4_detour python data/manhattan/detour_test.py "${DETOUR[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
run 5_latent_rank python data/manhattan/latent_compression.py "${LATENT[@]}" "${COMMON[@]}" "${OVERRIDES[@]}"
fi

python - "$OUT" "$CKPT" "$STEPS" <<'EOF'
import json, os, re, sys
out, ckpt, steps = sys.argv[1], sys.argv[2], sys.argv[3].split()
read = lambda step: open(os.path.join(out, f"{step}.log"), errors="replace").read() if os.path.exists(os.path.join(out, f"{step}.log")) else ""
# Last match: progress bars print running values (starting at 0.000) before the final result line
last = lambda pattern, text: (list(re.finditer(pattern, text)) or [None])[-1]
m, num = {"checkpoint": ckpt}, r"([-+0-9.eE]+)"
if g := last(rf"FINAL generation: n=(\d+) reached_end={num} valid={num}", read("1_trajectories")):
    m.update(trajectories_n=int(g[1]), trajectories_reached_end=float(g[2]), trajectories_valid=float(g[3]))
if g := last(r"New edges, True edges, Unused true edges\s+(\d+) (\d+) (\d+)", read("2_map_reconstruction")):
    m.update(map_new_edges=int(g[1]), map_true_edges=int(g[2]), map_unused_true_edges=int(g[3]))
if g := last(rf"Mean compression precision: {num} \({num}\)", read("3_sequence_compression")):
    m.update(compression_precision=float(g[1]), compression_precision_std=float(g[2]))
if g := last(rf"FINAL detour: .*? n=(\d+) valid={num} valid_std={num} reached_end={num}", read("4_detour")):
    m.update(detour_n=int(g[1]), detour_valid=float(g[2]), detour_valid_std=float(g[3]), detour_reached_end=float(g[4]))
    # Per-trial outcomes (in trial order), for paired comparisons between models on identical scenarios
    m["detour_trials_valid"] = [int(v == "True") for v in re.findall(r"Trial \d+: Reached end: \w+, is valid: (\w+)", read("4_detour"))]
log = read("5_latent_rank")
if g := last(rf"Mean participation ratio: {num}", log): m["participation_ratio"] = float(g[1])
if g := last(rf"Mean effective rank: {num}", log): m["effective_rank"] = float(g[1])
expected = {"1": "trajectories_valid", "2": "map_true_edges", "3": "compression_precision", "4": "detour_valid", "5": "effective_rank"}
m["missing"] = [k for s, k in expected.items() if s in steps and k not in m]
json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=2)
print(json.dumps(m, indent=2))
EOF
