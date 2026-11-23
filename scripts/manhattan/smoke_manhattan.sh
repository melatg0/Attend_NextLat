#!/bin/bash
# Manhattan smoke + speed test (run from repo root after: SKIP_GIT=1 bash scripts/runpod/setup.sh manhattan).
# For each variant: train 300 steps, resume to 400 (checks resume), then run every post-hoc eval at
# tiny sizes. Reports measured steps/s, GPU utilization and peak memory, and projected run times.
#
# Usage: bash scripts/manhattan/smoke_manhattan.sh [variant ...]    (default: nextlat learned)
set -euo pipefail

VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(nextlat learned)
SMOKE_DIR=${SMOKE_DIR:-output/smoke/manhattan}
WANDB_SMOKE_PROJECT=${WANDB_SMOKE_PROJECT:-attend-nextlat-smoke}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}

rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
for variant in "${VARIANTS[@]}"; do
    cfg="config/manhattan/attend/${variant}_manhattan.yaml"
    run_dir="$SMOKE_DIR/$variant"
    common=(--config "$cfg" --no_pbar seed=1234 trainer.out_dir="$run_dir" trainer.init_from=resume
        trainer.val_interval=100 trainer.test_interval=100 trainer.val_batches=5
        data.generalization_num_samples=2 trainer.save_recovery_checkpoint=100
        trainer.wandb_project="$WANDB_SMOKE_PROJECT")

    echo "==> [$variant] train to step 300"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i 0 -l 5 \
        > "$SMOKE_DIR/$variant.gpu.csv" &
    sampler=$!
    fabric run --devices 1 --precision bf16-mixed train.py "${common[@]}" trainer.train_batches=300 \
        > "$SMOKE_DIR/$variant.part1.log" 2>&1 || { kill $sampler; tail -30 "$SMOKE_DIR/$variant.part1.log"; exit 1; }
    kill $sampler
    grep -E "Parameter counts|Vocab size|Manhattan .*Accuracy" "$SMOKE_DIR/$variant.part1.log" | tail -4 || true

    echo "==> [$variant] resume to step 400"
    fabric run --devices 1 --precision bf16-mixed train.py "${common[@]}" trainer.train_batches=400 \
        > "$SMOKE_DIR/$variant.part2.log" 2>&1 || { tail -30 "$SMOKE_DIR/$variant.part2.log"; exit 1; }
    grep -q "Starting training from step=[1-9]" "$SMOKE_DIR/$variant.part2.log" \
        || { echo "FAIL: [$variant] did not resume from a checkpoint"; exit 1; }

    echo "==> [$variant] post-hoc evals (QUICK sizes)"
    ckpt=$(ls "$run_dir"/*/ckpt_iter_401.pt)
    QUICK=1 bash scripts/manhattan/eval_manhattan.sh "$ckpt" | grep -E "FAILED|missing" -A3 || true

    python - "$run_dir" "$SMOKE_DIR/$variant.gpu.csv" <<'EOF'
import csv, glob, sys
run_dir, gpu_csv = sys.argv[1], sys.argv[2]
sps = []
for f in glob.glob(f"{run_dir}/*/version_0/metrics.csv"):
    sps += [float(r["steps_per_sec"]) for r in csv.DictReader(open(f)) if r.get("steps_per_sec")]
sps = sps[len(sps) // 2:]  # skip compile warm-up
rate = sum(sps) / max(len(sps), 1)
gpu = [line.split(",") for line in open(gpu_csv) if line.strip()]
util = [float(u) for u, _ in gpu[2:]] or [0.0]
mem = max((float(m) for _, m in gpu), default=0.0)
print(f"    steps/s ~ {rate:.2f} | GPU util ~ {sum(util) / len(util):.0f}% | peak memory {mem / 1024:.1f} GB")
for steps in (50_000, 100_000, 255_000, 400_000):
    print(f"    {steps:>7,} steps -> ~{steps / max(rate, 1e-9) / 3600:.1f} h (train only, + ~10-20% evals)")
EOF
done

echo
echo "==> Manhattan smoke test passed. Check W&B project '$WANDB_SMOKE_PROJECT' for"
echo "    generalization/test_accuracy, val/valid_accuracy, val/nsl1_step_*, gate/*, params/*."
