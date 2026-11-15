"""
Finish an Attend-NextLat Path-Star batch on the pod, so nothing depends on a local machine:
  1. full-test-set evaluation of each finished run's final checkpoint (eval_results.json)
  2. comparison table (full-test numbers + curve stats from the training CSV logs)
  3. upload table, eval results and each run's materialized config to W&B as an artifact
  4. W&B alert (emailed if "Scriptable run alerts" email is enabled in W&B user settings)

Usage (repo root, on the pod):
    python scripts/runpod/finish_and_notify.py --root output/attend_compile/7_7 \
        --project attend-nextlat-stargraph --name results-7_7-seed1234
Options: --no-wandb (local only), --no-alert (upload but don't alert), --variants nextlat learned
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attend"))
from summarize_stargraph import first_step_reaching, load_run  # noqa: E402


def final_checkpoint(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "ckpt_iter_*.pt"))
    step = lambda p: int(re.search(r"ckpt_iter_(\d+)\.pt$", p).group(1))
    return max(ckpts, key=step) if ckpts else None


def full_eval(exp_dir, ckpt):
    """Evaluate ckpt on the full test set (skipped if already done); return the results dict."""
    tag = os.path.basename(ckpt)[:-3]
    pattern = os.path.join(exp_dir, f"eval_{tag}_full", "*", "eval_results.json")
    if not glob.glob(pattern):
        print(f"[eval] {ckpt}", flush=True)
        # Call train.py directly (as eval_stargraph.sh does) to control the output dir name
        cmd = ["fabric", "run", "--devices", "1", "--precision", "bf16-mixed", "train.py",
               "--config", os.path.join(exp_dir, "materialized_config.yaml"), "--no_pbar",
               "--checkpoint_path", ckpt, "trainer.val_only=true", "trainer.log_to_wandb=false",
               "trainer.compile=false", "trainer.test_batches=200", "trainer.val_batches=200",
               f"trainer.out_dir={os.path.join(exp_dir, f'eval_{tag}_full')}"]
        with open(os.path.join(exp_dir, f"eval_{tag}_full.log"), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return json.load(open(glob.glob(pattern)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--project", default="attend-nextlat-stargraph")
    ap.add_argument("--name", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-alert", action="store_true")
    args = ap.parse_args()
    name = args.name or "results-" + os.path.basename(os.path.normpath(args.root))

    rows, files, skipped = [], [], []
    for run_dir in sorted(glob.glob(os.path.join(args.root, "*", "seed*"))):
        variant, seed = run_dir.split(os.sep)[-2], run_dir.split(os.sep)[-1]
        if args.variants and variant not in args.variants:
            continue
        if not os.path.isfile(os.path.join(run_dir, "DONE")):
            print(f"[skip] {run_dir}: not finished")
            skipped.append(f"{variant}/{seed}")
            continue
        exp_dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(run_dir, "*", "materialized_config.yaml"))]
        exp_dir = exp_dirs[0]
        res = full_eval(exp_dir, final_checkpoint(exp_dir))
        acc_key = next(k for k in res if k.endswith("/test_accuracy"))
        prefix = acc_key.rsplit("/", 1)[0]
        curve = load_run(run_dir)
        steps = sorted({int(k.split("_")[-1]) for k in res if k.startswith("val/nsl1_step_")})
        dyn = next((m["params/dynamics_model"] for m in curve.values() if "params/dynamics_model" in m), None)
        rows.append({
            "variant": variant, "seed": seed, "eval_step": res["step"],
            "solve_rate_full": res[acc_key], "first_step_acc_full": res[f"{prefix}/token_2"],
            "steps_to_90": first_step_reaching(curve, acc_key, 0.90),
            "steps_to_95": first_step_reaching(curve, acc_key, 0.95),
            "val_nsl1": res.get("val/nsl1"),
            "val_nsl1_step_1": res.get("val/nsl1_step_1"),
            f"val_nsl1_step_{steps[-1]}": res.get(f"val/nsl1_step_{steps[-1]}") if steps else None,
            "gate_entropy": res.get("val/gate/entropy"),
            "dynamics_params": dyn,
        })
        files += [(os.path.join(exp_dir, "materialized_config.yaml"), f"{variant}_{seed}_config.yaml"),
                  (glob.glob(os.path.join(exp_dir, "eval_*_full", "*", "eval_results.json"))[0],
                   f"{variant}_{seed}_eval_full.json")]

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
    open(os.path.join(out_dir, f"{name}.md"), "w").write(md + "\n")
    json.dump(rows, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=2)
    print(md)

    headline = "; ".join(
        f"{r['variant']}: solve {100 * r['solve_rate_full']:.1f}%, first-step {100 * r['first_step_acc_full']:.1f}%, "
        f"90% at step {r['steps_to_90'] if r['steps_to_90'] is not None else 'never'}"
        for r in rows)
    if skipped:
        headline += f". NOT FINISHED (check train.log): {', '.join(skipped)}"
    print("\n" + headline)

    if args.no_wandb:
        return
    import wandb
    run = wandb.init(project=args.project, name=name, job_type="summary", config={"root": args.root})
    run.log({"results": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    for r in rows:
        for c in cols:
            if isinstance(r.get(c), (int, float)):
                run.summary[f"{r['variant']}_{r['seed']}/{c}"] = r[c]
    art = wandb.Artifact(name, type="results", description=headline)
    art.add_file(os.path.join(out_dir, f"{name}.md"))
    art.add_file(os.path.join(out_dir, f"{name}.json"))
    for src, dst in files:
        art.add_file(src, name=dst)
    run.log_artifact(art)
    if not args.no_alert:
        run.alert(title=f"Attend-NextLat {name}: results ready, pod can be deleted",
                  text=headline + f"\nFull table: W&B project {args.project}, run {name}.",
                  level=wandb.AlertLevel.INFO)
    run.finish()
    print("Uploaded to W&B" + ("" if args.no_alert else " and sent alert"))


if __name__ == "__main__":
    main()
