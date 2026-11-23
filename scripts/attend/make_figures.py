"""
Paper-style figures (NextLat, Teoh et al. 2025) from our finished runs, plus the checkpoint-trend
figure behind the Manhattan analysis. Reads only local result folders (training CSVs, eval
metrics.json, sweep/summary JSONs); no checkpoints or GPU needed.

  fig1_manhattan_world_model   paper Table 1 as bars (+ table1_manhattan.md with the paper's rows)
  fig2_pathstar_solve_rate     paper Fig. 5 for G(7,7): final solve rate + solve rate over training
  fig3_latent_losses           paper Fig. 13b/16 style: val latent error and val KL over training
  fig4_latent_by_horizon       latent error at each rollout step, final eval
  fig5_manhattan_checkpoints   latent error vs compression vs detours vs effective rank over training
  fig6_gate_entropy            learned-gate entropy over training
  table5_cost.md               paper Table 5 style: parameters and training speed

Usage (repo root):
    python scripts/attend/make_figures.py
    python scripts/attend/make_figures.py --manhattan-root <folder> [<folder> ...] --sweep <sweep.json>
Manhattan variants are gathered from all --manhattan-root folders; the sweep JSON with the most variants is used.
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_stargraph import load_run  # noqa: E402

# Fixed color per variant (validated categorical order; never re-assigned when variants change)
ORDER = ["nextlat", "learned_share010", "learned", "learned_share050", "uniform"]  # Manhattan shares ascending
COLORS = {"nextlat": "#2a78d6", "learned": "#eb6834", "uniform": "#1baf7a",
          "learned_share010": "#eda100", "learned_share050": "#e87ba4"}
MARKERS = {"nextlat": "o", "learned": "s", "uniform": "^", "learned_share010": "D", "learned_share050": "v"}
LABELS = {"nextlat": "NextLat (baseline)", "learned": "Attend, learned gate",
          "uniform": "Attend, uniform gate", "learned_share010": "Attend, learned gate, share 0.10",
          "learned_share050": "Attend, learned gate, share 0.50"}
SHORT = {"nextlat": "NextLat\n(baseline)", "learned": "Attend\nlearned gate", "uniform": "Attend\nuniform gate",
         "learned_share010": "Attend\nshare 0.10", "learned_share050": "Attend\nshare 0.50"}
REF_GREY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"

# NextLat paper, Table 1 (Manhattan; 6 epochs, d=8)
PAPER_TABLE1 = {
    "GPT": {"trajectories_valid": 0.970, "compression": 0.65, "effective_rank": 160.1, "detour_valid": 0.85},
    "MTP": {"trajectories_valid": 0.981, "compression": 0.64, "effective_rank": 57.7, "detour_valid": 0.95},
    "JTP": {"trajectories_valid": 0.971, "compression": 0.32, "effective_rank": 215.8, "detour_valid": 0.87},
    "NextLat": {"trajectories_valid": 0.987, "compression": 0.71, "effective_rank": 52.7, "detour_valid": 0.95},
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.5,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def ordered(variants):
    return sorted(variants, key=lambda v: ORDER.index(v) if v in ORDER else len(ORDER))


def short(v, task):
    if task == "manhattan" and v == "learned":
        return "Attend\nshare 0.25"
    return SHORT.get(v, v)


def label(v, task):
    if task == "manhattan" and v == "learned":
        return LABELS[v] + ", share 0.25"
    return LABELS.get(v, v)


def find_runs(root):
    """{variant: run_dir} for runs with a training CSV (seed1234)."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "seed1234"))):
        if glob.glob(os.path.join(d, "*", "version_*", "metrics.csv")):
            runs[d.split(os.sep)[-2]] = d
    return {v: runs[v] for v in ordered(runs)}


def series(rows, key):
    pts = [(s, m[key]) for s, m in rows.items() if key in m]
    return [p[0] for p in pts], [p[1] for p in pts]


def end_labels(ax, items, min_gap=10):
    """Direct labels at the lines' right ends (ink text, not the series color), pushed apart
    vertically so they never overlap. items: [(x, y, text)]."""
    if not items:
        return
    ax.figure.canvas.draw()
    to_disp = ax.transData.transform
    pts = sorted(((to_disp((x, y))[1], x, y, t) for x, y, t in items))
    placed = []
    for dy, x, y, t in pts:
        target = dy if not placed or dy - placed[-1] >= min_gap else placed[-1] + min_gap
        placed.append(target)
        ax.annotate(t, (x, y), xytext=(4, target - dy), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK2, annotation_clip=False)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".{png,pdf}"))


def kfmt(ax):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x / 1000:g}k" if x else "0"))


# ---------------------------------------------------------------- data loading
def manhattan_final(runs, sweep):
    """Final-checkpoint world-model metrics per variant. Detours: paired 1,000-trial result from the
    sweep when present (same trips for every model), else the standard 100-trial eval."""
    out = {}
    for v, d in runs.items():
        evals = glob.glob(os.path.join(d, "*", "manhattan_eval_ckpt_iter_*", "metrics.json"))
        std = [p for p in evals if re.search(r"ckpt_iter_\d+/metrics\.json$", p)]
        if not std:
            continue
        p = max(std, key=lambda p: int(re.search(r"ckpt_iter_(\d+)/", p).group(1)))
        ev = json.load(open(p))
        step = int(re.search(r"ckpt_iter_(\d+)/", p).group(1))
        row = {"step": step, "trajectories_valid": ev.get("trajectories_valid"),
               "trajectories_reached_end": ev.get("trajectories_reached_end"),
               "compression": ev.get("compression_precision"),
               "compression_std": ev.get("compression_precision_std"),
               "effective_rank": ev.get("effective_rank"), "map_new_edges": ev.get("map_new_edges"),
               "detour_valid": ev.get("detour_valid"), "detour_valid_std": ev.get("detour_valid_std"),
               "detour_n": ev.get("detour_n")}
        paired = [r for r in sweep if r["variant"] == v and r["step"] == step and r.get("detour_valid") is not None]
        if paired:
            row.update(detour_valid=paired[0]["detour_valid"], detour_valid_std=paired[0]["detour_valid_std"],
                       detour_n=paired[0]["detour_n"])
        out[v] = row
    return out


def load_sweep(path, roots):
    if path is None:
        cands = [c for r in roots for c in glob.glob(os.path.join(r, "summary", "sweep-*.json"))]
        if not cands:
            return []
        path = max(cands, key=lambda p: len({r["variant"] for r in json.load(open(p))}))
    print("sweep:", path)
    return json.load(open(path))


# ---------------------------------------------------------------- figures
def fig1_world_model(final, out):
    metrics = [("trajectories_valid", "Valid trajectories", True, None),
               ("compression", "Sequence compression", False, "compression_std"),
               ("effective_rank", "Effective latent rank (lower = more compact)", False, None),
               ("detour_valid", "Detour robustness (valid)", True, "detour_valid_std")]
    # Bars start at zero (a truncated bar would exaggerate the gaps); exact values are labeled.
    vs = list(final)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
    for ax, (key, title, pct, err) in zip(axes, metrics):
        vals = [final[v][key] * (100 if pct else 1) for v in vs]
        errs = [(final[v].get(err) or 0) * (100 if pct else 1) for v in vs] if err else [0] * len(vs)
        bars = ax.bar(range(len(vs)), vals, width=0.65, color=[COLORS.get(v, REF_GREY) for v in vs],
                      edgecolor="white", linewidth=1.5, zorder=2,
                      yerr=errs if any(errs) else None, error_kw=dict(ecolor=INK2, lw=1, capsize=3, zorder=3))
        for b, val, e in zip(bars, vals, errs):
            txt = f"{val:.1f}%" if pct else (f"{val:.2f}" if val < 10 else f"{val:.0f}")
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, val + e), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        top = max(v + e for v, e in zip(vals, errs))
        ax.set_ylim(0, 110 if pct else top * 1.15)
        if pct:
            ax.set_yticks(range(0, 101, 20))
        ax.set_xlim(-0.6, len(vs) - 0.4)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels([short(v, "manhattan") for v in vs], fontsize=7)
        ax.grid(axis="x", visible=False)
    n = {final[v].get("detour_n") for v in vs}
    fig.suptitle("Manhattan world-model metrics at the final checkpoint (seed 1234)", fontsize=10)
    fig.text(0.5, -0.06, f"Detours: {'/'.join(str(x) for x in n)} paired trials (same trips and detours for every "
             "model); error bars = 1 s.e. Compression: 100 trials.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig1_manhattan_world_model")


def table1_md(final, out):
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = ["| model | step | valid trajectories | sequence compression | effective rank (↓) | detour robustness |",
             "|---|---|---|---|---|---|"]
    for v, r in final.items():
        lines.append(f"| ours: {label(v, 'manhattan')} | {r['step']} | {pct(r['trajectories_valid'])} | "
                     f"{r['compression']:.2f} | {r['effective_rank']:.1f} | {pct(r['detour_valid'])} (n={r['detour_n']}) |")
    for m, r in PAPER_TABLE1.items():
        lines.append(f"| paper: {m} | 6 epochs | {pct(r['trajectories_valid'])} | {r['compression']:.2f} | "
                     f"{r['effective_rank']:.1f} | {pct(r['detour_valid'])} |")
    path = os.path.join(out, "table1_manhattan.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def fig2_pathstar(ps_runs, ps_summary, out):
    graph_key = None
    rows = {v: load_run(d) for v, d in ps_runs.items()}
    for r in rows.values():
        for m in r.values():
            graph_key = next((k for k in m if re.fullmatch(r"val_\(\d+, \d+\)/test_accuracy", k)), graph_key)
    g = re.search(r"\((\d+), (\d+)\)", graph_key).groups()
    summ = {r["variant"]: r for r in ps_summary}
    vs = list(ps_runs)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 2]})
    vals = [100 * summ[v]["solve_rate_full"] for v in vs]
    bars = a.bar(range(len(vs)), vals, width=0.7, color=[COLORS[v] for v in vs], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, vals):
        a.annotate(f"{val:.1f}%", (bar.get_x() + bar.get_width() / 2, val), xytext=(0, 2),
                   textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    a.set_xticks(range(len(vs)))
    a.set_xticklabels([short(v, "pathstar") for v in vs], fontsize=7.5)
    a.set_ylim(0, 105)
    a.set_ylabel("Solve rate (%)")
    a.set_title(f"(a) Final solve rate, full test set ({summ[vs[0]]['eval_step']} steps)", fontsize=9)
    a.grid(axis="x", visible=False)
    for v in vs:
        x, y = series(rows[v], graph_key)
        y = [100 * t for t in y]
        b.plot(x, y, color=COLORS[v], marker=MARKERS[v], ms=3, markevery=8, label=label(v, "pathstar"))
    b.axhline(90, color=REF_GREY, ls=":", lw=1)
    b.annotate("90%", (0, 90), xytext=(2, -9), textcoords="offset points", fontsize=7, color=REF_GREY)
    note = "; ".join(f"{label(v, 'pathstar').split(', ')[-1]}: {summ[v]['steps_to_90'] or 'never'}"
                     for v in vs)
    b.set_title(f"(b) Solve rate during training (5,120-graph eval slice)", fontsize=9)
    b.set_xlabel("Training step")
    b.set_ylabel("Solve rate (%)")
    b.set_ylim(0, 105)
    kfmt(b)
    b.legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"Path-Star G({g[0]},{g[1]}), seed 1234", fontsize=10)
    fig.text(0.5, -0.04, f"Steps to 90% solve rate: {note}.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig2_pathstar_solve_rate")


def fig3_latent_losses(tasks, out):
    fig, axes = plt.subplots(len(tasks), 2, figsize=(10, 3.0 * len(tasks)), squeeze=False)
    for row, (task, runs, title) in enumerate(tasks):
        for col, (key, ylab) in enumerate([("val/nsl1", "Val latent error (normalized smooth L1)"),
                                           ("val/kl_loss", "Val KL(p(·|h) ‖ p(·|ĥ))")]):
            ax = axes[row][col]
            ends = []
            for v, d in runs.items():
                x, y = series(load_run(d), key)
                if not x:
                    continue
                ax.plot(x, y, color=COLORS.get(v, REF_GREY), marker=MARKERS.get(v, "o"), ms=3,
                        markevery=max(1, len(x) // 10), label=label(v, task))
                ends.append((x[-1], y[-1], f"{y[-1]:.3g}"))
            ax.set_yscale("log")
            end_labels(ax, ends)
            ax.set_ylabel(ylab)
            ax.set_xlabel("Training step")
            ax.set_title(f"{title}: {ylab.split(' (')[0].replace('Val ', 'validation ')}", fontsize=9)
            kfmt(ax)
            if col == 0:
                ax.legend(fontsize=7.5)
    fig.suptitle("Next-latent prediction losses over training", fontsize=10)
    fig.text(0.5, -0.02, "Normalized smooth L1 = smooth L1 / mean per-dimension variance of the true latents, "
             "averaged over rollout steps (~0.5 = no better than predicting the mean). Log scale.",
             ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig3_latent_losses")


def fig4_horizon(tasks, out):
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.2), squeeze=False)
    for ax, (task, runs, title) in zip(axes[0], tasks):
        for v, d in runs.items():
            rows = load_run(d)
            ks, ys, step = [], [], None
            for k in range(1, 17):
                x, y = series(rows, f"val/nsl1_step_{k}")
                if x:
                    ks.append(k), ys.append(y[-1])
                    step = x[-1]
            if ks:
                ax.plot(ks, ys, color=COLORS.get(v, REF_GREY), marker=MARKERS.get(v, "o"), ms=5,
                        label=label(v, task))
        ax.set_xlabel("Rollout step k (predicting h_{t+k})")
        ax.set_ylabel("Val latent error (normalized smooth L1)")
        ax.set_title(f"{title} (step {step:,})", fontsize=9)
        ax.set_xticks(ks)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5)
    fig.suptitle("Latent error by rollout step at the end of training", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig4_latent_by_horizon")


def fig5_checkpoints(sweep, out):
    vs = ordered({r["variant"] for r in sweep})
    panels = [("val_nsl1", "Val latent error", None, 1, "lower"),
              ("compression", "Sequence compression", "compression_std", 1, "higher"),
              ("detour_valid", "Detour validity (%)", "detour_valid_std", 100, "higher"),
              ("effective_rank", "Effective latent rank", None, 1, "lower")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (key, title, err, scale, better) in zip(axes, panels):
        for v in vs:
            pts = sorted((r["step"], r[key], r.get(err) if err else None) for r in sweep
                         if r["variant"] == v and r.get(key) is not None)
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] * scale for p in pts]
            e = [(p[2] or 0) * scale for p in pts] if err else None
            ax.errorbar(x, y, yerr=e, color=COLORS.get(v, REF_GREY), marker=MARKERS.get(v, "o"), ms=5,
                        capsize=3, elinewidth=1, label=label(v, "manhattan"))
        if key == "detour_valid":
            # one p per step only makes sense for a single comparison (2 models); with more, see the sweep table
            pcols = [k for k in ("detour_p_holm_vs_nextlat", "detour_mcnemar_p") if k in sweep[0]]
            pcol = pcols[0] if pcols and len(vs) == 2 else None
            if pcol:  # p-values in a row along the top, clear of the markers
                trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
                for s in sorted({r["step"] for r in sweep}):
                    ps = [r[pcol] for r in sweep if r["step"] == s and r.get(pcol) is not None]
                    if ps:
                        ax.text(s, 0.97, f"p={min(ps):.2g}", transform=trans, ha="center", va="top",
                                fontsize=6.5, color=INK2)
        ax.set_title(f"{title} ({better} = better)", fontsize=9)
        ax.set_xlabel("Training step")
        kfmt(ax)
    axes[0].legend(fontsize=7.5)
    lo, hi = axes[2].get_ylim()
    axes[2].set_ylim(lo, hi + (hi - lo) * 0.2)
    axes[2].margins(x=0.12)
    fig.suptitle("Manhattan, seed 1234: does lower latent error go with a better world model?", fontsize=10)
    fig.text(0.5, -0.05, "Detours: 1,000 paired trials per point (same trips and decisions for every model), "
             "error bars = 1 s.e" + ("., p = exact McNemar test vs the baseline" if len(vs) == 2 else "") + ". Compression: 100 trials. "
             "One seed: a trend, not an established correlation.", ha="center", fontsize=7, color=INK2)
    fig.tight_layout()
    save(fig, out, "fig5_manhattan_checkpoints")


def fig6_gate(tasks, out):
    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 3.0), squeeze=False)
    for ax, (task, runs, title) in zip(axes[0], tasks):
        ends = []
        for v, d in runs.items():
            x, y = series(load_run(d), "val/gate/entropy")
            if x:
                ax.plot(x, y, color=COLORS.get(v, REF_GREY), marker=MARKERS.get(v, "o"), ms=3,
                        markevery=max(1, len(x) // 10), label=label(v, task))
                ends.append((x[-1], y[-1], f"{y[-1]:.2f}"))
        ax.axhline(1.0, color=REF_GREY, ls="--", lw=1)
        ax.annotate("uniform weights", (0, 1.0), xytext=(2, -10), textcoords="offset points", fontsize=7,
                    color=REF_GREY)
        ax.set_ylim(0, 1.05)
        end_labels(ax, ends)
        ax.set_ylabel("Gate entropy / log(#heads)")
        ax.set_xlabel("Training step")
        ax.set_title(title, fontsize=9)
        kfmt(ax)
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Learned gate entropy over training (validation, averaged over rollout steps)", fontsize=10)
    fig.tight_layout()
    save(fig, out, "fig6_gate_entropy")


def table5_md(tasks, out):
    lines = ["| task | model | predictor params | total params | median steps/s | relative speed |",
             "|---|---|---|---|---|---|"]
    for task, runs, title in tasks:
        base = None
        for v, d in runs.items():
            rows = load_run(d)
            _, sps = series(rows, "steps_per_sec")
            sps = sorted(sps)[len(sps) // 2] if sps else None
            base = base or sps
            dyn = next((m["params/dynamics_model"] for m in rows.values() if "params/dynamics_model" in m), None)
            tot = next((m["params/total"] for m in rows.values() if "params/total" in m), None)
            lines.append(f"| {title} | {label(v, task)} | {dyn:,.0f} | {tot / 1e6:.1f}M | {sps:.2f} | "
                         f"{sps / base:.2f}x |")
    lines.append("")
    lines.append("Steps/s = median logged training throughput (one H100 NVL per run, compiled, bf16); "
                 "relative to the NextLat baseline on the same task. Manhattan wall-clock to 255k steps incl. "
                 "in-training evals: 25.8 h (baseline), 29.3 h (learned gate).")
    path = os.path.join(out, "table5_cost.md")
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manhattan-root", nargs="+", default=["output/results/manhattan_headshare_seed1234",
                    "output/results/manhattan_seed1234"], help="result folders; the first holding a variant wins")
    ap.add_argument("--pathstar-root", default="output/results/pathstar_7_7_seed1234")
    ap.add_argument("--sweep", default=None, help="sweep JSON (default: the one in <manhattan-root>/summary "
                    "with the most variants)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mh_runs = {}
    for root in args.manhattan_root:
        for v, d in find_runs(root).items():
            mh_runs.setdefault(v, d)
    mh_runs = {v: mh_runs[v] for v in ordered(mh_runs)}
    ps_runs = find_runs(args.pathstar_root)
    sweep = load_sweep(args.sweep, args.manhattan_root)
    ps_summary = json.load(open(glob.glob(os.path.join(args.pathstar_root, "summary", "results-*.json"))[0]))
    print("manhattan:", list(mh_runs), "| path-star:", list(ps_runs))

    tasks = [("pathstar", ps_runs, "Path-Star G(7,7)"), ("manhattan", mh_runs, "Manhattan")]
    final = manhattan_final(mh_runs, sweep)
    fig1_world_model(final, args.out)
    table1_md(final, args.out)
    fig2_pathstar(ps_runs, ps_summary, args.out)
    fig3_latent_losses(tasks, args.out)
    fig4_horizon(tasks, args.out)
    if sweep:
        fig5_checkpoints(sweep, args.out)
    fig6_gate([(t, {v: d for v, d in r.items() if v != "uniform"}, n) for t, r, n in tasks], args.out)
    table5_md(tasks, args.out)


if __name__ == "__main__":
    main()
