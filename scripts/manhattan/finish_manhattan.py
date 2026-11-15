"""
Finish a Manhattan batch on the pod, so nothing depends on a local machine:
  1. run scripts/manhattan/eval_manhattan.sh on each finished run's final checkpoint
  2. comparison table: post-hoc world-model metrics + last logged training-time metrics
  3. upload the table, per-run metrics and configs to W&B as an artifact
  4. W&B alert (emailed if "Scriptable run alerts" email is enabled in W&B user settings)

Usage (repo root, on the pod):
    python scripts/manhattan/finish_manhattan.py --root output/attend/manhattan --name results-manhattan
Options: --no-wandb, --no-alert, --quick (tiny eval sizes, for smoke tests), --variants nextlat learned
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attend"))
from summarize_stargraph import last_value, load_run  # noqa: E402

# Training-time metrics (W&B / CSV keys) reported at the last logged step
CURVE_KEYS = [
    "generalization/test_accuracy",   # generated trips reaching the destination (random pairs)
    "generalization/valid_accuracy",  # generated trips with only legal turns (random pairs)
    "val/valid_accuracy",             # same, on 1,000 heldout trips' pairs
    "val/next_token_loss",
    "val/nsl1",
    "val/nsl1_step_1",
    "val/nsl1_step_8",
    "val/gate/entropy",
    "params/dynamics_model",
]
EVAL_KEYS = [
    "trajectories_valid", "trajectories_reached_end", "map_true_edges", "map_new_edges",
    "compression_precision", "detour_valid", "detour_reached_end", "effective_rank",
    "participation_ratio",
]


def final_checkpoint(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "ckpt_iter_*.pt"))
    step = lambda p: int(re.search(r"ckpt_iter_(\d+)\.pt$", p).group(1))
    return max(ckpts, key=step) if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/attend/manhattan")
    ap.add_argument("--project", default="attend-nextlat-manhattan")
    ap.add_argument("--name", default="results-manhattan")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-alert", action="store_true")
    args = ap.parse_args()

    rows, files, skipped = [], [], []
    for run_dir in sorted(glob.glob(os.path.join(args.root, "*", "seed*"))):
        variant, seed = run_dir.split(os.sep)[-2], run_dir.split(os.sep)[-1]
        if args.variants and variant not in args.variants:
            continue
        if not os.path.isfile(os.path.join(run_dir, "DONE")):
            print(f"[skip] {run_dir}: not finished")
            skipped.append(f"{variant}/{seed}")
            continue
        exp_dir = os.path.dirname(glob.glob(os.path.join(run_dir, "*", "materialized_config.yaml"))[0])
        ckpt = final_checkpoint(exp_dir)
        suffix = "_quick" if args.quick else ""
        metrics_path = os.path.join(exp_dir, f"manhattan_eval_{os.path.basename(ckpt)[:-3]}{suffix}", "metrics.json")
        if not os.path.isfile(metrics_path):
            print(f"[eval] {ckpt}", flush=True)
            env = dict(os.environ, QUICK="1" if args.quick else "0")
            subprocess.run(["bash", "scripts/manhattan/eval_manhattan.sh", ckpt], check=True, env=env)
        ev = json.load(open(metrics_path))
        curve = load_run(run_dir)
        row = {"variant": variant, "seed": seed, "step": int(re.search(r"(\d+)\.pt$", ckpt).group(1))}
        row.update({k: ev.get(k) for k in EVAL_KEYS})
        for k in CURVE_KEYS:
            v = last_value(curve, k)
            row[k.replace("/", "_")] = v[1] if v else None
        if ev.get("missing"):
            row["eval_missing"] = ",".join(ev["missing"])
        rows.append(row)
        files += [(os.path.join(exp_dir, "materialized_config.yaml"), f"{variant}_{seed}_config.yaml"),
                  (metrics_path, f"{variant}_{seed}_eval.json")]

    if not rows:
        print("No finished runs found.")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: "n/a" if v is None else (
        str(int(v)) if isinstance(v, float) and v.is_integer() and v > 1 else
        f"{v:.4f}" if isinstance(v, float) else str(v))
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    md = "\n".join(md)
    out_dir = os.path.join(args.root, "summary")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, f"{args.name}.md"), "w").write(md + "\n")
    json.dump(rows, open(os.path.join(out_dir, f"{args.name}.json"), "w"), indent=2)
    print(md)

    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    headline = "; ".join(
        f"{r['variant']}: valid trips {pct(r['trajectories_valid'])}, detour valid {pct(r['detour_valid'])}, "
        f"compression {fmt(r['compression_precision'])}, eff. rank {fmt(r['effective_rank'])}"
        for r in rows)
    if skipped:
        headline += f". NOT FINISHED (check train.log): {', '.join(skipped)}"
    if any(r.get("eval_missing") for r in rows):
        headline += ". Some evals failed: see eval_missing in the table."
    print("\n" + headline)

    if args.no_wandb:
        return
    import wandb
    run = wandb.init(project=args.project, name=args.name, job_type="summary", config={"root": args.root})
    run.log({"results": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    for r in rows:
        for c in cols:
            if isinstance(r.get(c), (int, float)):
                run.summary[f"{r['variant']}_{r['seed']}/{c}"] = r[c]
    art = wandb.Artifact(args.name, type="results", description=headline)
    art.add_file(os.path.join(out_dir, f"{args.name}.md"))
    art.add_file(os.path.join(out_dir, f"{args.name}.json"))
    for src, dst in files:
        art.add_file(src, name=dst)
    run.log_artifact(art)
    if not args.no_alert:
        run.alert(title=f"Attend-NextLat {args.name}: results ready",
                  text=headline + f"\nFull table: W&B project {args.project}, run {args.name}.",
                  level=wandb.AlertLevel.INFO)
    run.finish()
    print("Uploaded to W&B" + ("" if args.no_alert else " and sent alert"))


if __name__ == "__main__":
    main()
