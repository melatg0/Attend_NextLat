"""
Summarize Attend-NextLat Path-Star runs: mean +- std over seeds for each variant.

Reads the CSV logs written during training, so it works on the pod or locally after
copying the output tree (no W&B API needed).

Usage:
    python scripts/attend/summarize_stargraph.py output/attend/5_5 [--threshold 0.95] [--csv out.csv]

Expected layout: <root>/<variant>/seed<S>/<experiment_name>/version_*/metrics.csv
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict


def load_run(run_dir):
    """Merge all metrics.csv rows of a run (resumes create new version_* dirs) keyed by step."""
    by_step = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "version_*", "metrics.csv"))):
        with open(path) as f:
            for row in csv.DictReader(f):
                if not row.get("step"):
                    continue
                step = int(float(row["step"]))
                for k, v in row.items():
                    if v not in ("", None) and k != "step":
                        try:
                            by_step[step][k] = float(v)
                        except ValueError:
                            pass
    return dict(sorted(by_step.items()))


def last_value(rows, key, max_step=None):
    """Value of `key` at the largest step <= max_step where it was logged."""
    best = None
    for step, metrics in rows.items():
        if max_step is not None and step > max_step:
            break
        if key in metrics:
            best = (step, metrics[key])
    return best


def first_step_reaching(rows, key, threshold):
    for step, metrics in rows.items():
        if metrics.get(key, -math.inf) >= threshold:
            return step
    return None


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return m, s, len(values)


def fmt(m, s, n, pct=False, integer=False):
    if m is None:
        return "n/a"
    if pct:
        return f"{100 * m:.1f} ± {100 * s:.1f}"
    if integer:
        return f"{m:.0f} ± {s:.0f}"
    return f"{m:.4f} ± {s:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="e.g. output/attend/5_5")
    parser.add_argument("--threshold", type=float, default=0.95, help="solve-rate threshold for steps-to-threshold")
    parser.add_argument("--csv", default=None, help="optional path for a per-run CSV dump")
    args = parser.parse_args()

    per_variant = defaultdict(list)
    per_run_rows = []
    for variant_dir in sorted(glob.glob(os.path.join(args.root, "*"))):
        if not os.path.isdir(variant_dir):
            continue
        variant = os.path.basename(variant_dir)
        for run_dir in sorted(glob.glob(os.path.join(variant_dir, "seed*"))):
            rows = load_run(run_dir)
            if not rows:
                continue
            acc_key = next((k for m in rows.values() for k in m if re.match(r"val_\(.*\)/test_accuracy$", k)), None)
            if acc_key is None:
                continue
            prefix = acc_key.rsplit("/", 1)[0]
            final = last_value(rows, acc_key)
            final_step = final[0]
            run = {
                "variant": variant,
                "seed": os.path.basename(run_dir),
                "done": os.path.isfile(os.path.join(run_dir, "DONE")),
                "eval_step": final_step,
                "solve_rate": final[1],
                "first_step_acc": (last_value(rows, f"{prefix}/token_2") or (None, None))[1],
                "steps_to_threshold": first_step_reaching(rows, acc_key, args.threshold),
                "best_solve_rate": max(m[acc_key] for m in rows.values() if acc_key in m),
            }
            # Latent metrics from the same validation step as the final solve rate
            for key in sorted({k for m in rows.values() for k in m if k.startswith("val/nsl1") or k.startswith("val/gate/entropy") or k in ("params/dynamics_model",)}):
                v = last_value(rows, key, max_step=final_step if key.startswith("val/") else None)
                run[key] = v[1] if v else None
            per_variant[variant].append(run)
            per_run_rows.append(run)

    if not per_run_rows:
        print(f"No runs with Path-Star accuracy found under {args.root}")
        return

    columns = [
        ("solve_rate", "solve rate (%)", dict(pct=True)),
        ("first_step_acc", "first-step acc (%)", dict(pct=True)),
        ("best_solve_rate", "best solve rate (%)", dict(pct=True)),
        ("steps_to_threshold", f"steps to {args.threshold:.0%}", dict(integer=True)),
        ("val/nsl1", "val nsl1", {}),
    ]
    extra = sorted({k for r in per_run_rows for k in r if k.startswith("val/nsl1_step") or k == "val/gate/entropy"})
    columns += [(k, k.replace("val/", "val "), {}) for k in extra]

    print(f"Path-Star summary for {args.root} (mean ± std over seeds; solve rate at last logged eval)\n")
    header = ["variant", "n"] + [c[1] for c in columns] + ["dyn params"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for variant, runs in per_variant.items():
        cells = [variant, str(len(runs))]
        for key, _, kw in columns:
            vals = [r.get(key) for r in runs]
            m, s, n = mean_std(vals)
            cell = fmt(m, s, n, **kw)
            missing = sum(v is None for v in vals)
            if key == "steps_to_threshold" and missing:
                cell += f" ({missing} never)"
            cells.append(cell)
        params = {r.get("params/dynamics_model") for r in runs} - {None}
        cells.append(",".join(f"{p:.0f}" for p in sorted(params)) or "n/a")
        print("| " + " | ".join(cells) + " |")

    unfinished = [f"{r['variant']}/{r['seed']} (eval step {r['eval_step']})" for r in per_run_rows if not r["done"]]
    if unfinished:
        print(f"\nNot finished: {', '.join(unfinished)}")

    if args.csv:
        keys = sorted({k for r in per_run_rows for k in r})
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_run_rows)
        print(f"\nPer-run CSV written to {args.csv}")


if __name__ == "__main__":
    main()
