"""
Figures 1-3 for the two-column paper, one column wide each, panels stacked vertically.

Follows the usual two-column standard (e.g. IEEE: one-column figures ~3.5in wide, text 8pt Times,
never below 6pt; resolution and font size judged at final printed size). Our template's column is
3.375in (letter, 0.75in margins, 0.25in column gap), so each figure is drawn at exactly that width
and text is 8-9pt at printed size. Include with width=\\columnwidth: no scaling, so the text prints at
the stated size. Serif (STIX = Times), TrueType fonts embedded, no titles/footnotes inside (captions).

  paper_fig1_manhattan.pdf       Manhattan world-model metrics, final checkpoint (4 stacked bar panels)
  paper_fig2_pathstar.pdf        Path-Star G(7,7): (a) final solve rate, (b) solve rate over training
  paper_fig3_latent_losses.pdf   validation latent error and KL over training, both tasks (4 stacked)
  page_preview.png               mock letter page at true scale, to check fit and legibility

Usage (repo root): python scripts/attend/make_paper_figures.py [--out figures/paper]
"""

import argparse
import glob
import json
import os
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402  (data loading, colors, markers)
import matplotlib  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.375  # template column width (in); IEEE's is 3.5
SHORT = {"nextlat": "NextLat", "learned_share010": "Share 0.10", "learned": "Share 0.25",
         "learned_share050": "Share 0.50"}
PS_NAMES = {"nextlat": "NextLat", "learned": "Attend, learned gate", "uniform": "Attend, uniform gate"}
MH_NAMES = {"nextlat": "NextLat", "learned_share010": "Attend, share 0.10", "learned": "Attend, share 0.25",
            "learned_share050": "Attend, share 0.50"}

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "axes.titlepad": 3, "grid.linewidth": 0.4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
MIN_PT = 8  # smallest text size used anywhere (checked in check_fonts)


def kfmt(ax, nbins=5):
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0"))


def check_fonts(fig, name):
    fig.canvas.draw()
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    assert min(sizes) >= MIN_PT, f"{name}: text below {MIN_PT}pt ({min(sizes)})"


def save(fig, out, name):
    check_fonts(fig, name)
    for ext in ("pdf", "png"):  # no bbox="tight": keep the exact printed size
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    w, h = fig.get_size_inches()
    print(f"wrote {name}.pdf/.png  ({w:.3f} x {h:.2f} in)")
    plt.close(fig)


def fig1(final, out):
    """Four stacked panels, one per metric; horizontal bars with the model name on each bar row."""
    vs = list(final)
    panels = [("trajectories_valid", "(a) Valid trajectories (%) ↑", True, None),
              ("compression", "(b) Sequence compression ↑", False, "compression_std"),
              ("effective_rank", "(c) Effective rank (lower = more compact) ↓", False, None),
              ("detour_valid", "(d) Detour validity (%) ↑", True, "detour_valid_std")]
    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 5.0))
    y = list(range(len(vs)))[::-1]  # first model on top
    for ax, (key, title, pct, err) in zip(axes, panels):
        vals = [final[v][key] * (100 if pct else 1) for v in vs]
        errs = [(final[v].get(err) or 0) * (100 if pct else 1) for v in vs] if err else [0] * len(vs)
        ax.barh(y, vals, height=0.7, color=[mf.COLORS[v] for v in vs], edgecolor="white", linewidth=0.8,
                xerr=errs if any(errs) else None, error_kw=dict(ecolor=mf.INK2, lw=0.7, capsize=2))
        for yy, val, e in zip(y, vals, errs):
            ax.annotate(f"{val:.1f}" if pct else (f"{val:.2f}" if val < 10 else f"{val:.0f}"),
                        (val + e, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT[v] for v in vs])
        top = max(v + e for v, e in zip(vals, errs))
        ax.set_xlim(0, 100 * 1.12 if pct else top * 1.15)
        if pct:
            ax.set_xticks(range(0, 101, 25))
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color=mf.GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3, h_pad=0.7)
    save(fig, out, "paper_fig1_manhattan")


def fig2(ps_runs, summary, out):
    rows = {v: mf.load_run(d) for v, d in ps_runs.items()}
    key = next(k for r in rows.values() for m in r.values() for k in m
               if re.fullmatch(r"val_\(\d+, \d+\)/test_accuracy", k))
    summ = {r["variant"]: r for r in summary}
    vs = list(ps_runs)
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.5), gridspec_kw={"height_ratios": [1, 1.8]})
    y = list(range(len(vs)))[::-1]
    vals = [100 * summ[v]["solve_rate_full"] for v in vs]
    a.barh(y, vals, height=0.65, color=[mf.COLORS[v] for v in vs], edgecolor="white", linewidth=0.8)
    for yy, val in zip(y, vals):
        a.annotate(f"{val:.1f}%", (val, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    a.set_yticks(y)
    a.set_yticklabels([PS_NAMES[v] for v in vs])
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, 112)
    a.set_xticks(range(0, 101, 25))
    a.set_title("(a) Final solve rate (%), full test set", loc="left")
    a.grid(axis="x", color=mf.GRID)
    for v in vs:
        x, yv = mf.series(rows[v], key)
        b.plot(x, [100 * t for t in yv], color=mf.COLORS[v], marker=mf.MARKERS[v], markevery=10, label=PS_NAMES[v])
    b.axhline(90, color=mf.REF_GREY, ls=":", lw=0.8)
    b.annotate("90%", (300, 90), xytext=(0, 1.5), textcoords="offset points", va="bottom", fontsize=8,
               color=mf.REF_GREY)
    b.set_ylim(0, 103)
    b.set_xlim(0, 20800)
    b.set_xlabel("Training step")
    b.set_ylabel("Solve rate (%)")
    b.set_title("(b) Solve rate during training", loc="left")
    kfmt(b)
    b.grid(color=mf.GRID)
    b.legend(loc="lower right")
    for ax in (a, b):
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3, h_pad=0.8)
    save(fig, out, "paper_fig2_pathstar")


def fig3(tasks, out):
    """Four stacked panels: Path-Star latent error, Path-Star KL, Manhattan latent error, Manhattan KL."""
    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 6.4))
    specs = [("val/nsl1", "latent error"), ("val/kl_loss", "KL divergence")]
    i = 0
    for task, runs, tname in tasks:
        names = PS_NAMES if task == "pathstar" else MH_NAMES
        for key, metric in specs:
            ax = axes[i]
            for v, d in runs.items():
                x, y = mf.series(mf.load_run(d), key)
                if x:
                    ax.plot(x, y, color=mf.COLORS[v], marker=mf.MARKERS[v], markevery=max(1, len(x) // 8),
                            label=names[v])
            ax.set_yscale("log")
            ax.set_title(f"({'abcd'[i]}) {tname}: {metric}", loc="left")
            ax.set_ylabel("Smooth L1 (norm.)" if key == "val/nsl1" else "KL")
            kfmt(ax)
            ax.grid(color=mf.GRID, which="major")
            ax.set_axisbelow(True)
            # legends where each panel is empty: Path-Star KL lower right, Manhattan latent upper right
            if (task, key) == ("pathstar", "val/kl_loss"):
                ax.legend(loc="lower right")
            elif (task, key) == ("manhattan", "val/nsl1"):
                ax.legend(loc="upper right")
            i += 1
    axes[1].set_xlabel("Training step")
    axes[3].set_xlabel("Training step")
    fig.tight_layout(pad=0.3, h_pad=0.6)
    save(fig, out, "paper_fig3_latent_losses")


def preview(out):
    """Mock letter page at true scale (0.75in margins, two 3.375in columns), with 11pt body text."""
    page = plt.figure(figsize=(8.5, 11))

    def place(name, x, y_top):
        img = mpimg.imread(os.path.join(out, f"{name}.png"))
        h = COL_W * img.shape[0] / img.shape[1]
        ax = page.add_axes([x / 8.5, (11 - y_top - h) / 11, COL_W / 8.5, h / 11])
        ax.imshow(img), ax.axis("off")
        return y_top + h

    L, R = 0.75, 0.75 + COL_W + 0.25
    for x in (L, R):  # column outlines (margins of the template)
        page.patches.append(plt.Rectangle((x / 8.5, 0.75 / 11), COL_W / 8.5, 9.5 / 11, fill=False, lw=0.4,
                                          ec="#cccccc", transform=page.transFigure, figure=page))
    y = place("paper_fig1_manhattan", L, 0.75)
    page.text(L / 8.5, 1 - (y + 0.2) / 11, "Figure 1: caption at 10pt.", fontsize=10, family="STIXGeneral")
    y = place("paper_fig2_pathstar", L, y + 0.4)
    page.text(L / 8.5, 1 - (y + 0.2) / 11, "Figure 2: caption at 10pt.", fontsize=10, family="STIXGeneral")
    y = place("paper_fig3_latent_losses", R, 0.75)
    page.text(R / 8.5, 1 - (y + 0.2) / 11, "Figure 3: caption at 10pt.", fontsize=10, family="STIXGeneral")
    body = "Body text at 11pt, to compare with the figure text. The quick brown fox jumps over the lazy dog. "
    for i, line in enumerate(textwrap.wrap(body * 8, 50)):
        if y + 0.55 + i * 0.19 > 10.2:
            break
        page.text(R / 8.5, 1 - (y + 0.55 + i * 0.19) / 11, line, fontsize=11, family="STIXGeneral", va="top")
    page.savefig(os.path.join(out, "page_preview.png"), dpi=150)
    plt.close(page)
    print("wrote page_preview.png (mock letter page at true scale)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/paper")
    ap.add_argument("--manhattan-root", nargs="+", default=["output/results/manhattan_headshare_seed1234",
                                                             "output/results/manhattan_seed1234"])
    ap.add_argument("--pathstar-root", default="output/results/pathstar_7_7_seed1234")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mh = {}
    for root in args.manhattan_root:
        for v, d in mf.find_runs(root).items():
            mh.setdefault(v, d)
    mh = {v: mh[v] for v in mf.ordered(mh)}
    ps = mf.find_runs(args.pathstar_root)
    sweep = mf.load_sweep(None, args.manhattan_root)
    summary = json.load(open(glob.glob(os.path.join(args.pathstar_root, "summary", "results-*.json"))[0]))
    fig1(mf.manhattan_final(mh, sweep), args.out)
    fig2(ps, summary, args.out)
    fig3([("pathstar", ps, "Path-Star G(7,7)"), ("manhattan", mh, "Manhattan")], args.out)
    preview(args.out)


if __name__ == "__main__":
    main()
