#!/bin/bash
# Pod-side autopilot for a batch of Manhattan runs (start it in tmux; needs no local machine):
#   setup -> smoke test -> train VARIANTS to 255k steps (one GPU each) -> checkpoint sweep
#   (sweep_manhattan.py) -> final evals + table (finish_manhattan.py).
# Each stage sends a W&B alert (emailed if "Scriptable run alerts" email is on in W&B settings).
#
# Usage (repo root on the pod, with the earlier runs to compare against copied into
# output/attend/manhattan/<variant>/seed1234/ — eval folders and training CSVs are enough):
#   tmux new -d -s autopilot "bash scripts/manhattan/autopilot_manhattan.sh"
# Env (defaults = the head/combiner allocation sweep at seed 1234):
#   VARIANTS="learned_share010 learned_share050"   runs to train; the i-th goes to GPU i
#   COMPARE="nextlat learned"                       earlier runs included in the tables
#   REFERENCE="learned nextlat"                     paired detour tests against these
#   NAME=headshare  STEPS=255000  SKIP_SETUP=0  SKIP_SMOKE=0  SKIP_ARMED=0
cd "$(dirname "$0")/../.." || exit 1
REPO=$(pwd -P)  # before sourcing rp_environment, which exports PWD=/
source /etc/rp_environment 2>/dev/null
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
VARIANTS=(${VARIANTS:-learned_share010 learned_share050})
COMPARE=(${COMPARE:-nextlat learned})
REFERENCE=(${REFERENCE:-learned nextlat})
NAME=${NAME:-headshare}
STEPS=${STEPS:-255000}
ROOT=output/attend/manhattan
LOG=output/autopilot_$NAME.log
mkdir -p output
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
alert() {  # alert <title> <text>
    if python - "$1" "$2" <<'EOF' >> "$LOG" 2>&1
import sys, wandb
run = wandb.init(project="attend-nextlat-manhattan", name="autopilot-notify", job_type="notify")
run.alert(title=sys.argv[1], text=sys.argv[2], level=wandb.AlertLevel.INFO, wait_duration=0)
run.finish()
EOF
    then log "alert sent: $1"; else log "alert FAILED (see above): $1"; fi
}

missing=$(for v in "${COMPARE[@]}"; do [ -f "$ROOT/$v/seed1234/DONE" ] || echo -n "$v "; done)
[ -n "$missing" ] && log "WARNING: comparison runs not found under $ROOT (copy them first): $missing"

# 1. setup (deps incl. wandb, dataset, eval artifacts); alerts work only after this
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    SKIP_GIT=1 bash scripts/runpod/setup.sh manhattan > output/setup_manhattan.log 2>&1 \
        || { alert "Manhattan setup FAILED ($NAME)" "$(tail -15 output/setup_manhattan.log)"; exit 1; }
    log "setup ok"
fi
[ "${SKIP_ARMED:-0}" = "1" ] || alert "Manhattan autopilot armed ($NAME)" "Setup done. Runs: ${VARIANTS[*]} to $STEPS steps, compared with ${COMPARE[*]}. Next: smoke test -> training -> checkpoint sweep -> final evals; you get an email at each stage."

# 2. smoke test: train/resume/eval each variant briefly; measures speed and peak memory
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    if ! bash scripts/manhattan/smoke_manhattan.sh "${VARIANTS[@]}" > output/smoke_$NAME.log 2>&1; then
        alert "Manhattan smoke test FAILED ($NAME, training not started)" "$(tail -20 output/smoke_$NAME.log)"; exit 1
    fi
    SUMMARY=$(STEPS=$STEPS python - "output/smoke_$NAME.log" <<'EOF'
import os, re, sys
log = open(sys.argv[1]).read()
steps = int(os.environ["STEPS"])
variants = re.findall(r"==> \[(\w+)\] post-hoc evals", log)
stats = re.findall(r"steps/s ~ ([\d.]+) \| GPU util ~ (\d+)% \| peak memory ([\d.]+) GB", log)
lines = []
for v, (sps, util, mem) in zip(variants, stats):
    hours = steps / float(sps) / 3600 * 1.12  # + ~12% measured for in-training evals
    lines.append(f"{v}: {float(sps):.2f} steps/s, peak {mem} GB -> {steps // 1000}k steps ~{hours:.0f} h")
print("; ".join(lines) + (". Some post-hoc evals FAILED (see the smoke log)." if "FAILED" in log else ". Evals ok."))
EOF
)
    log "smoke ok: $SUMMARY"
fi

# 3. training, one GPU per variant
tmux new -d -s runs "source /etc/rp_environment 2>/dev/null; cd $REPO; VARIANTS='${VARIANTS[*]}' EXTRA_ARGS='trainer.train_batches=$STEPS' bash scripts/runpod/launch_runs.sh manhattan 1 2>&1 | tee output/launch_$NAME.log"

# 4. checkpoint sweep (waits for each run's checkpoints; slots on a GPU open when its run is done)
SLOTS=(); i=0
for v in "${VARIANTS[@]}"; do SLOTS+=(--slot "$i:$v" --slot "$i:$v" --slot "$i:$v"); i=$((i + 1)); done
tmux new -d -s sweep "source /etc/rp_environment 2>/dev/null; cd $REPO; python scripts/manhattan/sweep_manhattan.py \
    --variants ${VARIANTS[*]} --compare ${COMPARE[*]} --reference ${REFERENCE[*]} --final-full ${VARIANTS[0]} \
    ${SLOTS[*]} --wait-for $ROOT/summary/results-manhattan-$NAME.json --name sweep-$NAME > output/sweep_$NAME.log 2>&1"

sleep 600
STATUS=$(for v in "${VARIANTS[@]}"; do echo -n "$v: $(tail -n1 $ROOT/$v/seed1234/train.log 2>/dev/null | cut -c1-90) | "; done)
alert "Manhattan training started ($NAME)" "${SUMMARY:-} After 10 min: $STATUS Next emails: final results, then the checkpoint sweep."

# 5. final evals + table once every run is done (finish_manhattan.py sends the results email;
#    the sweep sends its own when it has summarized)
until grep -q "All slots finished" output/launch_$NAME.log 2>/dev/null; do sleep 600; done
if grep -qE "\] FAILED \(" output/launch_$NAME.log; then  # per-run failure lines only, not the closing hint
    alert "Manhattan training FAILED ($NAME)" "$(grep -E '\] FAILED \(' output/launch_$NAME.log)"
fi
log "training finished"
python scripts/manhattan/finish_manhattan.py --root $ROOT --variants ${VARIANTS[*]} ${COMPARE[*]} \
    --name results-manhattan-$NAME > output/finish_$NAME.log 2>&1 \
    || alert "Manhattan final evals FAILED ($NAME)" "$(tail -15 output/finish_$NAME.log)"
log "autopilot done"
