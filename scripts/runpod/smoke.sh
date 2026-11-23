#!/bin/bash
# Short end-to-end check of all three variants before launching the full sweep.
# Exercises: forward/backward, metric logging to W&B, checkpoint saving, Path-Star eval,
# W&B model artifact upload, resuming from a checkpoint, and post-hoc val_only eval.
#
# Usage: bash scripts/runpod/smoke.sh [graph]   (default 5_5; run from repo root)
set -euo pipefail

GRAPH=${1:-5_5}
SMOKE_DIR=${SMOKE_DIR:-output/smoke/$GRAPH}
WANDB_SMOKE_PROJECT=${WANDB_SMOKE_PROJECT:-attend-nextlat-smoke}
STEPS=60

rm -rf "$SMOKE_DIR"
for variant in nextlat learned uniform; do
    cfg="config/stargraph/$GRAPH/attend/${variant}_stargraph_${GRAPH}.yaml"
    run_dir="$SMOKE_DIR/$variant"
    common=(--config "$cfg" --no_pbar seed=1234
        trainer.out_dir="$run_dir" trainer.init_from=resume
        trainer.val_interval=30 trainer.test_interval=30 trainer.val_batches=4 trainer.test_batches=4
        trainer.save_recovery_checkpoint=20 "trainer.keep_checkpoint_steps=[30]"
        trainer.wandb_project="$WANDB_SMOKE_PROJECT")

    echo "==> [$variant] train to step $((STEPS / 2))"
    fabric run --devices 1 --precision bf16-mixed train.py "${common[@]}" \
        trainer.train_batches=$((STEPS / 2)) 2>&1 | tee "$SMOKE_DIR/$variant.part1.log" | \
        grep -E "Parameter counts|StarGraph Test Accuracy|Saving checkpoint|Error|Traceback" || true

    echo "==> [$variant] resume to step $STEPS"
    fabric run --devices 1 --precision bf16-mixed train.py "${common[@]}" \
        trainer.train_batches=$STEPS 2>&1 | tee "$SMOKE_DIR/$variant.part2.log" | \
        grep -E "Resuming|Starting training|StarGraph Test Accuracy|Uploading|Error|Traceback" || true
    grep -q "Starting training from step=[1-9]" "$SMOKE_DIR/$variant.part2.log" \
        || { echo "FAIL: [$variant] did not resume from a checkpoint"; exit 1; }

    echo "==> [$variant] post-hoc eval of step-30 checkpoint"
    ckpt=$(ls "$run_dir"/*/ckpt_iter_30.pt)
    bash scripts/attend/eval_stargraph.sh "$ckpt" 2>&1 | tail -n 3

    python - "$run_dir" <<'EOF'
import csv, glob, sys
vals, dyn = [], None
for f in glob.glob(f"{sys.argv[1]}/*/version_*/metrics.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("steps_per_sec"):
            vals.append(float(r["steps_per_sec"]))
        if r.get("params/dynamics_model"):
            dyn = r["params/dynamics_model"]
vals = vals[len(vals) // 2:]  # skip warm-up
sps = sum(vals) / max(len(vals), 1)
print(f"    steps/sec ~ {sps:.2f} -> ~{20000 / max(sps, 1e-9) / 3600:.2f} h per 20k-step run "
      f"(train only); dynamics params = {dyn}")
EOF
done

echo
echo "==> Smoke test passed. Check W&B project '$WANDB_SMOKE_PROJECT' for nsl1_step_*, gate/*, params/*, val_(*)/test_accuracy."
echo "    If GPU utilization (nvidia-smi) is low, raise JOBS_PER_GPU for launch_runs.sh."
