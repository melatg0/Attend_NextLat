"""
Checkpoint sweep for the Manhattan comparison: does latent prediction error move with sequence
compression and detour robustness over training, or apart from them?

For every run under --root (one per variant/seed) and every step in --steps ("final" = the run's
last checkpoint, once its DONE marker exists), this queues:
  - a paired detour eval (--detour-trials, identical trips and detour decisions for every model):
    folder manhattan_eval_<ckpt>_paired<N>
  - at --compression-steps: sequence compression + effective rank at standard sizes, in the standard
    folder manhattan_eval_<ckpt> (skipped where that folder already has them)
  - for --final-full variants: the standard full eval of the final checkpoint, where
    finish_manhattan.py looks for it, so its final stage reuses it instead of redoing it

Jobs run in a pool of GPU slots: --slot GPU[:VARIANT] (repeatable) opens a slot on GPU once VARIANT's
run is DONE (or immediately without a variant). Several slots per GPU are fine: these evals run at
batch size 1 and leave the GPU mostly idle. Resumable: jobs whose metrics.json exists are skipped.

Then a table per (step, variant): validation latent error (training CSV, last value at or before the
step), compression, effective rank, paired detour validity, and for every variant against each
--reference the paired detour difference (points) with exact McNemar p-values, raw and Holm-adjusted
over the comparisons at that step. --compare adds already-evaluated runs to the table only (their
checkpoints may be gone: eval folders and training CSVs are enough). Written to
<root>/summary/<name>.{md,json}, uploaded to W&B with an alert.

Usage (repo root, on the pod):
    python scripts/manhattan/sweep_manhattan.py --steps 100000 150000 200000 final \
        --compression-steps 150000 200000 --final-full nextlat \
        --slot 0:nextlat --slot 0:nextlat --slot 0:nextlat --slot 1:learned --slot 1:learned \
        --wait-for output/attend/manhattan/summary/results-manhattan-255k.json
Head-share sweep (new runs vs the earlier nextlat and share-0.25 runs, copied back into --root):
    python scripts/manhattan/sweep_manhattan.py --variants learned_share010 learned_share050 \
        --compare nextlat learned --reference learned nextlat --final-full learned_share010 \
        --slot 0:learned_share010 --slot 0:learned_share010 --slot 0:learned_share010 \
        --slot 1:learned_share050 --slot 1:learned_share050 --slot 1:learned_share050 \
        --wait-for output/attend/manhattan/summary/results-manhattan-headshare.json --name sweep-headshare
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attend"))
from summarize_stargraph import last_value, load_run  # noqa: E402

LATENT_KEYS = ["val/nsl1", "val/nsl1_step_1", "val/nsl1_step_8", "val/next_token_loss"]


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def exp_dir_of(run_dir):
    cfgs = glob.glob(os.path.join(run_dir, "*", "materialized_config.yaml"))
    return os.path.dirname(cfgs[0]) if cfgs else None


def ckpt_step(path):
    return int(re.search(r"ckpt_iter_(\d+)\.pt$", path).group(1))


def resolve_ckpt(run_dir, step):
    """Checkpoint path for a step, or None if not available yet."""
    exp_dir = exp_dir_of(run_dir)
    if exp_dir is None:
        return None
    if step == "final":  # same choice as finish_manhattan.py
        if not os.path.isfile(os.path.join(run_dir, "DONE")):
            return None
        ckpts = glob.glob(os.path.join(exp_dir, "ckpt_iter_*.pt"))
        return max(ckpts, key=ckpt_step) if ckpts else None
    path = os.path.join(exp_dir, f"ckpt_iter_{step}.pt")
    # a checkpoint is complete once a later one (or the recovery pointer past it) exists
    return path if os.path.isfile(path) and time.time() - os.path.getmtime(path) > 120 else None


def eval_dir(ckpt, tag=""):
    return os.path.join(os.path.dirname(ckpt), f"manhattan_eval_{os.path.basename(ckpt)[:-3]}{tag}")


def read_metrics(ckpt, tag=""):
    path = os.path.join(eval_dir(ckpt, tag), "metrics.json")
    return json.load(open(path)) if os.path.isfile(path) else {}


class Job:
    def __init__(self, kind, variant, run_dir, step, env):
        self.kind, self.variant, self.run_dir, self.step, self.env = kind, variant, run_dir, step, env
        self.state = "pending"  # pending -> running -> done

    def __repr__(self):
        return f"{self.kind} {self.variant} {self.step}"

    def needed(self, ckpt, args):
        m = read_metrics(ckpt, self.env.get("TAG", ""))
        if self.kind == "detour":
            return "detour_valid" not in m
        if self.kind == "compression":
            return "compression_precision" not in m or "effective_rank" not in m
        return not m  # full


def build_jobs(args, runs):
    jobs = []
    tag = f"_paired{args.detour_trials}"
    for variant, run_dir in runs:  # long full evals first, so they finish before finish_manhattan.py runs
        if variant in args.final_full:
            jobs.append(Job("full", variant, run_dir, "final", {}))
    for step in args.steps:  # matched steps together, so each comparison completes as early as possible
        for variant, run_dir in runs:
            jobs.append(Job("detour", variant, run_dir, step,
                            {"STEPS": "4", "PAIRED": "1", "DETOUR_TRIALS": str(args.detour_trials), "TAG": tag}))
    for step in args.compression_steps:
        for variant, run_dir in runs:
            jobs.append(Job("compression", variant, run_dir, step, {"STEPS": "3 5"}))
    return jobs


def run_pool(args, runs, jobs):
    lock = threading.Lock()

    def take(gpu):
        with lock:
            pending = [j for j in jobs if j.state == "pending"]
            if not pending:
                return None, None
            for job in pending:
                ckpt = resolve_ckpt(job.run_dir, job.step)
                if ckpt is None:
                    if os.path.isfile(os.path.join(job.run_dir, "DONE")):  # run finished without it
                        job.state = "done"
                        log(f"skip (no checkpoint for this step): {job}")
                    continue
                if not job.needed(ckpt, args):
                    job.state = "done"
                    log(f"skip (already evaluated): {job} -> {ckpt}")
                    continue
                job.state = "running"
                return job, ckpt
            return "wait", None

    def slot(gpu, gate):
        gate_dir = dict(runs).get(gate) if gate else None
        while gate_dir and not os.path.isfile(os.path.join(gate_dir, "DONE")):
            time.sleep(60)
        log(f"slot open on GPU {gpu}" + (f" ({gate} is done)" if gate else ""))
        while True:
            job, ckpt = take(gpu)
            if job is None:
                return
            if job == "wait":  # remaining jobs wait for checkpoints that are not written yet
                time.sleep(120)
                continue
            log(f"GPU {gpu}: start {job} ({ckpt})")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), **job.env)
            out = subprocess.run(["bash", "scripts/manhattan/eval_manhattan.sh", ckpt],
                                 env=env, capture_output=True, text=True)
            failed = [l.strip() for l in out.stdout.splitlines() if "FAILED" in l]
            log(f"GPU {gpu}: end {job}" + (f": {'; '.join(failed)}" if failed else ""))
            with lock:
                job.state = "done"

    threads = []
    for spec in args.slot:
        gpu, _, gate = spec.partition(":")
        t = threading.Thread(target=slot, args=(gpu, gate or None), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def mcnemar_p(a, b):
    """Exact two-sided McNemar test on paired 0/1 outcomes."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    tail = sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n
    return min(1.0, 2 * tail), n10, n01


def summary_ckpt(run_dir, step):
    """Like resolve_ckpt, but also accepts a checkpoint whose file is gone if its eval folders exist
    (e.g. earlier runs copied back without checkpoints, to compare new runs against)."""
    exp_dir = exp_dir_of(run_dir)
    if exp_dir is None:
        return None
    steps = {ckpt_step(p) for p in glob.glob(os.path.join(exp_dir, "ckpt_iter_*.pt"))}
    steps |= {int(m.group(1)) for d in glob.glob(os.path.join(exp_dir, "manhattan_eval_ckpt_iter_*"))
              if (m := re.search(r"manhattan_eval_ckpt_iter_(\d+)", d))}
    if step == "final":
        if not os.path.isfile(os.path.join(run_dir, "DONE")) or not steps:
            return None
        s = max(steps)
    else:
        s = int(step)
        if s not in steps:
            return None
    return os.path.join(exp_dir, f"ckpt_iter_{s}.pt")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted


def summarize(args, runs):
    rows, files = [], []
    tag = f"_paired{args.detour_trials}"
    refs = args.reference or [runs[0][0]]
    for step in args.steps:
        trials, step_rows = {}, []
        for variant, run_dir in runs:
            ckpt = summary_ckpt(run_dir, step)
            if ckpt is None:
                continue
            std, det = read_metrics(ckpt), read_metrics(ckpt, tag)
            s = ckpt_step(ckpt)
            curve = load_run(run_dir)
            row = {"step": s, "variant": variant}
            for k in LATENT_KEYS:
                v = last_value(curve, k, max_step=s)
                row[k.replace("/", "_")] = v[1] if v else None
            row.update(compression=std.get("compression_precision"), compression_std=std.get("compression_precision_std"),
                       effective_rank=std.get("effective_rank"),
                       detour_valid=det.get("detour_valid"), detour_valid_std=det.get("detour_valid_std"),
                       detour_n=det.get("detour_n"), detour_reached_end=det.get("detour_reached_end"))
            step_rows.append(row)
            trials[variant] = det.get("detour_trials_valid") or []
            for t in ("", tag):
                path = os.path.join(eval_dir(ckpt, t), "metrics.json")
                if os.path.isfile(path):
                    files.append((path, f"{variant}_{s}{t}.json"))
        # Paired detour comparisons of every variant against each reference (same trials, in order).
        # A reference is compared only with references listed before it, so no pair is tested twice.
        # Holm adjustment within each reference's family of comparisons at this step.
        for ref in refs:
            tests = []
            for r in step_rows:
                v = r["variant"]
                if v == ref or (v in refs and refs.index(v) < refs.index(ref)):
                    continue
                a, b = trials.get(v), trials.get(ref)
                if a and b and len(a) == len(b):
                    r[f"detour_diff_vs_{ref}"] = round(100 * (sum(a) - sum(b)) / len(a), 2)  # points
                    tests.append((r, mcnemar_p(a, b)[0]))
            for (r, p), p_holm in zip(tests, holm([p for _, p in tests])):
                r[f"detour_p_vs_{ref}"], r[f"detour_p_holm_vs_{ref}"] = round(p, 4), round(p_holm, 4)
        rows += step_rows
    if not rows:
        print("Nothing to summarize.")
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: "n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    md = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                   + ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows])
    out_dir = os.path.join(args.root, "summary")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, f"{args.name}.md"), "w").write(md + "\n")
    json.dump(rows, open(os.path.join(out_dir, f"{args.name}.json"), "w"), indent=2)
    print(md)

    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
    parts = {}
    for r in rows:
        vs = "".join(f" [{r[f'detour_diff_vs_{ref}']:+.1f} pts vs {ref}, Holm p={r[f'detour_p_holm_vs_{ref}']}]"
                     for ref in refs if f"detour_diff_vs_{ref}" in r)
        parts.setdefault(r["step"], []).append(
            f"{r['variant']} detour {pct(r['detour_valid'])}{vs}, compression {fmt(r['compression'])}, "
            f"rank {fmt(r['effective_rank'])}, nsl1 {fmt(r['val_nsl1'])}")
    headline = " | ".join(f"{s}: " + "; ".join(v) for s, v in parts.items())
    print("\n" + headline)
    if args.no_wandb:
        return
    import wandb
    run = wandb.init(project=args.project, name=args.name, job_type="summary", config=vars(args))
    run.log({"sweep": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    art = wandb.Artifact(args.name, type="results", description=headline[:1000])
    art.add_file(os.path.join(out_dir, f"{args.name}.md"))
    art.add_file(os.path.join(out_dir, f"{args.name}.json"))
    for src, dst in files:
        art.add_file(src, name=dst)
    run.log_artifact(art)
    if not args.no_alert:
        run.alert(title=f"Attend-NextLat {args.name}: checkpoint sweep ready",
                  text=headline + f"\nFull table: W&B project {args.project}, run {args.name}.",
                  level=wandb.AlertLevel.INFO)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/attend/manhattan")
    ap.add_argument("--seed", default="seed1234")
    ap.add_argument("--variants", nargs="*", default=["nextlat", "learned"], help="runs to evaluate")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="extra runs for the table only (already evaluated; checkpoints may be gone)")
    ap.add_argument("--reference", nargs="*", default=None,
                    help="variants every other variant is paired-tested against (default: the first)")
    ap.add_argument("--steps", nargs="+", default=["100000", "150000", "200000", "final"])
    ap.add_argument("--compression-steps", nargs="*", default=["150000", "200000"])
    ap.add_argument("--final-full", nargs="*", default=[])
    ap.add_argument("--detour-trials", type=int, default=1000)
    ap.add_argument("--slot", action="append", default=None, help="GPU[:VARIANT], repeatable")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--wait-for", default=None, help="file to wait for (up to 8 h) before summarizing")
    ap.add_argument("--project", default="attend-nextlat-manhattan")
    ap.add_argument("--name", default="sweep-manhattan")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-alert", action="store_true")
    args = ap.parse_args()
    args.slot = args.slot or ["0"]

    runs = [(v, os.path.join(args.root, v, args.seed)) for v in args.variants]
    if not args.summary_only:
        jobs = build_jobs(args, runs)
        log(f"{len(jobs)} jobs: {jobs}")
        run_pool(args, runs, jobs)
        log("all jobs finished")
    runs = [(v, os.path.join(args.root, v, args.seed)) for v in args.compare] + runs
    if args.wait_for:  # e.g. finish_manhattan.py's table: the final standard evals (compression, rank)
        deadline = time.time() + 8 * 3600
        while not os.path.isfile(args.wait_for) and time.time() < deadline:
            time.sleep(120)
        log(f"{args.wait_for} {'found' if os.path.isfile(args.wait_for) else 'still missing after 8 h'}")
    summarize(args, runs)


if __name__ == "__main__":
    main()
