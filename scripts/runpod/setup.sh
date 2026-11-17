#!/bin/bash
# One-time (idempotent) setup of a RunPod pod for Attend-NextLat experiments (Path-Star, Manhattan).
#
# Usage (on the pod):
#   export WANDB_API_KEY=...          # set as a RunPod secret/env var; never commit it
#   export WANDB_ENTITY=...           # optional: team or username for W&B runs
#   export REPO_URL=...               # optional: defaults to the GitHub repo over HTTPS
#   bash setup.sh [task ...]          # Path-Star graphs (e.g. 7_7) and/or manhattan; default: 5_5
#
# Everything lives on the network volume (/workspace) so it survives pod restarts.
# For manhattan, the ~20 GB Hugging Face dataset is cached in $HF_HOME (default /workspace/hf_cache);
# scripts/runpod/launch_runs.sh uses the same default, so training reads the cached copy.
set -euo pipefail

WORKDIR=${WORKDIR:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/melatg0/Attend_NextLat.git}
BRANCH=${BRANCH:-feature/head-specific-dynamics}
REPO_DIR=${REPO_DIR:-$WORKDIR/Attend_NextLat}
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(5_5)
REQUIREMENTS=scripts/runpod/requirements-stargraph.txt
[[ " ${TASKS[*]} " == *" manhattan "* ]] && REQUIREMENTS=scripts/runpod/requirements-manhattan.txt

echo "==> Repository: $REPO_DIR ($BRANCH)"
if [ "${SKIP_GIT:-0}" = "1" ]; then
    # Code was copied onto the pod directly (e.g. rsync over SSH); don't touch git remotes.
    [ -f "$REPO_DIR/train.py" ] || { echo "SKIP_GIT=1 but $REPO_DIR/train.py not found"; exit 1; }
elif [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi
cd "$REPO_DIR"
echo "    commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)$(git status --porcelain 2>/dev/null | grep -q . && echo " (+ uncommitted changes)")"

echo "==> Python dependencies"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$REQUIREMENTS"
python - <<'EOF'
import torch
major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
assert (major, minor) >= (2, 6), f"torch {torch.__version__} < 2.6: pip install -U torch"
print(f"    torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, "
      f"GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"      [{i}] {torch.cuda.get_device_name(i)}")
EOF

echo "==> W&B login"
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login --relogin "$WANDB_API_KEY" >/dev/null && echo "    logged in"
else
    echo "    WARNING: WANDB_API_KEY not set; runs will fail to log to W&B (or set trainer.log_to_wandb=false)"
fi

echo "==> Data"
for g in "${TASKS[@]}"; do
    if [ "$g" = "manhattan" ]; then
        export HF_HOME=${HF_HOME:-$WORKDIR/hf_cache}
        echo "    Manhattan: artifacts, ~20 GB dataset (cache: $HF_HOME), heldout text"
        python scripts/manhattan/prepare_data.py 2>&1 | grep -E "^\[ok\]|HF cache|Error|error" | sed 's/^/      /'
        continue
    fi
    # Path-Star (deterministic; identical on every machine)
    paths=${g%_*}; length=${g#*_}
    train_file="data/stargraph/graph_${paths}_${length}_sample_200000.txt"
    test_file="data/stargraph/graph_${paths}_${length}_test_20000.txt"
    if [ -f "$train_file" ] && [ -f "$test_file" ]; then
        echo "    G($paths,$length): already present"
    else
        echo "    G($paths,$length): generating"
        python data/stargraph/prepare.py --num_samples 200000 --num_paths "$paths" \
            --path_length "$length" --max_nodes 100 --generate_test_data > /dev/null
    fi
    wc -l "$train_file" "$test_file" | sed 's/^/      /'
done

echo "==> Unit tests"
python -m unittest tests.test_head_specific_dynamics

echo "==> Setup complete. Next: smoke test (scripts/runpod/smoke.sh <graph> or scripts/manhattan/smoke_manhattan.sh),"
echo "    then in tmux: bash scripts/runpod/launch_runs.sh <task> 1"
