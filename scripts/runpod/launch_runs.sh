#!/bin/bash
# Launch an Attend-NextLat comparison: variants x seeds, one single-GPU job each.
#
# Usage (from repo root, inside tmux):
#   bash scripts/runpod/launch_runs.sh <task> [JOBS_PER_GPU]
#   <task> is a Path-Star graph (5_5, 7_7, ...) or manhattan
#   e.g. bash scripts/runpod/launch_runs.sh 7_7 1   (one job per GPU is fastest on an H100)
#
# Env overrides: SEEDS="1234 1235 1236"  VARIANTS="nextlat learned uniform"  NUM_GPUS=<n>
#                OUT_ROOT=output/attend   EXTRA_ARGS="trainer.foo=bar ..."
# Defaults: Path-Star runs all 3 variants x 3 seeds; manhattan runs nextlat + learned, seed 1234.
#
# Each job writes to $OUT_ROOT/<task>/<variant>/seed<S>/ (own checkpoint pointers, so
# jobs never clobber each other). Re-running this script skips finished jobs (DONE marker)
# and resumes unfinished ones from their latest checkpoint.
set -uo pipefail

GRAPH=${1:?usage: launch_runs.sh <task> [JOBS_PER_GPU]}
JOBS_PER_GPU=${2:-1}
if [ "$GRAPH" = "manhattan" ]; then
    SEEDS=(${SEEDS:-1234})
    VARIANTS=(${VARIANTS:-nextlat learned})
    # Same cache location as setup.sh, so training reads the pre-downloaded dataset
    export HF_HOME=${HF_HOME:-/workspace/hf_cache}
else
    SEEDS=(${SEEDS:-1234 1235 1236})
    VARIANTS=(${VARIANTS:-nextlat learned uniform})
fi
config_for() {
    if [ "$GRAPH" = "manhattan" ]; then
        echo "config/manhattan/attend/${1}_manhattan.yaml"
    else
        echo "config/stargraph/$GRAPH/attend/${1}_stargraph_${GRAPH}.yaml"
    fi
}
OUT_ROOT=${OUT_ROOT:-output/attend}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NUM_GPUS" -ge 1 ] || { echo "No GPUs found (set NUM_GPUS to override)"; exit 1; }

# Build the job list (seed-major: one run of every variant per seed comes first), skipping finished runs
jobs=()
for variant in "${VARIANTS[@]}"; do
    cfg=$(config_for "$variant")
    [ -f "$cfg" ] || { echo "Missing config $cfg"; exit 1; }
done
for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        cfg=$(config_for "$variant")
        run_dir="$OUT_ROOT/$GRAPH/$variant/seed$seed"
        if [ -f "$run_dir/DONE" ]; then
            echo "skip (done): $run_dir"
        else
            jobs+=("$cfg|$seed|$run_dir")
        fi
    done
done

n_slots=$((NUM_GPUS * JOBS_PER_GPU))
echo "Launching ${#jobs[@]} jobs on $NUM_GPUS GPU(s) x $JOBS_PER_GPU job(s)/GPU = $n_slots slot(s)"
echo "Logs: $OUT_ROOT/$GRAPH/<variant>/seed<S>/train.log"

run_slot() {
    local slot=$1 gpu=$(($1 % NUM_GPUS)) port=$((29500 + 10 * $1))
    local i
    # Static round-robin assignment of jobs to slots (all jobs have equal length)
    for ((i = slot; i < ${#jobs[@]}; i += n_slots)); do
        IFS='|' read -r cfg seed run_dir <<< "${jobs[$i]}"
        mkdir -p "$run_dir"
        echo "[slot $slot | gpu $gpu] start $run_dir"
        CUDA_VISIBLE_DEVICES=$gpu fabric run --devices 1 --precision bf16-mixed --main-port "$port" \
            train.py --config "$cfg" --no_pbar \
            seed="$seed" trainer.out_dir="$run_dir" trainer.init_from=resume ${EXTRA_ARGS:-} \
            >> "$run_dir/train.log" 2>&1
        status=$?
        if [ $status -eq 0 ]; then
            date > "$run_dir/DONE"
            echo "[slot $slot | gpu $gpu] done  $run_dir"
        else
            echo "[slot $slot | gpu $gpu] FAILED ($status) $run_dir -- see $run_dir/train.log"
        fi
    done
}

for ((s = 0; s < n_slots; s++)); do
    run_slot "$s" &
done
wait
echo "All slots finished. Re-run this script to retry any FAILED jobs (they resume from checkpoints)."
[ "$GRAPH" = "manhattan" ] || echo "Summary: python scripts/attend/summarize_stargraph.py $OUT_ROOT/$GRAPH"
