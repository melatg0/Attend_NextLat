"""
Reconstructed Manhattan street maps, one panel per model: true street edges the model's trips used
in grey, edges that don't exist in red (curved toward the claimed direction, as in
data/manhattan/make_maps.py). Reads the graph pickles written by data/manhattan/make_graphs.py
(nodes carry 'lat_long'; edges carry 'edge_type' in {true, new, true_unused}), so it needs no GPU,
OSM download or browser.

Usage (repo root):
    python scripts/attend/plot_maps.py --graphs output/results/maps --out figures
"""

import argparse
import glob
import math
import os
import pickle
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

# variant -> (file-name pattern, panel title); order = panel order
MODELS = [
    ("nextlat", r"^NextLat-baseline-seed1234-ckpt_iter_255001", "NextLat (baseline)"),
    ("learned_share010", r"^Attend-learned-share010-seed1234-ckpt_iter_255001", "Attend, learned gate, share 0.10"),
    ("learned", r"^Attend-learned-seed1234-ckpt_iter_255001", "Attend, learned gate, share 0.25"),
    ("learned_share050", r"^Attend-learned-share050-seed1234-ckpt_iter_255001", "Attend, learned gate, share 0.50"),
]
LAT, LON = (40.702, 40.800), (-74.022, -73.933)  # make_maps.py's full-island view
TRUE_C, NEW_C, INK, INK2 = "#52514e", "#e34948", "#0b0b0b", "#52514e"


def curve(lat1, lon1, lat2, lon2, direction, n=20):
    """Quadratic Bezier bent toward the edge's direction label (same rule as make_maps.py)."""
    dlat = 0.002 if "N" in direction else -0.002 if "S" in direction else 0
    dlon = 0.002 if "E" in direction else -0.002 if "W" in direction else 0
    clat, clon = lat1 + dlat, lon1 + dlon
    t = np.linspace(0, 1, n)
    lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * clat + t ** 2 * lat2
    lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * clon + t ** 2 * lon2
    return np.column_stack([lon, lat])


def draw(ax, g, title):
    pos = {n: d["lat_long"] for n, d in g.nodes(data=True)}
    true, new, counts = [], [], {"true": 0, "new": 0, "true_unused": 0}
    for u, v, _, d in g.edges(keys=True, data=True):
        counts[d["edge_type"]] += 1
        (la1, lo1), (la2, lo2) = pos[u], pos[v]
        if d["edge_type"] == "true":
            true.append([(lo1, la1), (lo2, la2)])
        elif d["edge_type"] == "new":
            new.append(curve(la1, lo1, la2, lo2, d["direction"]))
    ax.add_collection(LineCollection(true, colors=TRUE_C, linewidths=0.35, alpha=0.45, zorder=1))
    ax.add_collection(LineCollection(new, colors=NEW_C, linewidths=0.7, alpha=0.85, zorder=2))
    ax.set_xlim(*LON)
    ax.set_ylim(*LAT)
    ax.set_aspect(1 / math.cos(math.radians(sum(LAT) / 2)))  # equal ground distance on both axes
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title.replace(", share", "\nshare"), fontsize=9.5, color=INK)
    ax.text(0.97, 0.03, f"{counts['new']:,}\nnon-existent\nedges", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color=INK, linespacing=1.1)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="output/results/maps")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    files = [os.path.basename(p) for p in glob.glob(os.path.join(args.graphs, "*.pkl"))]
    panels = []
    for variant, pat, title in MODELS:
        hit = [f for f in files if re.match(pat, f) and "quick" not in f]
        if hit:
            panels.append((title, os.path.join(args.graphs, sorted(hit)[0])))
        else:
            print(f"[skip] no graph for {variant}")
    cols, rows = len(panels), 1  # the island is tall and narrow: one row reads best
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 5.6), squeeze=False)
    for ax, (title, path) in zip(axes.flat, panels):
        c = draw(ax, pickle.load(open(path, "rb")), title)
        print(f"{title}: {c}")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Street maps reconstructed from 6,400 generated trips (Manhattan, seed 1234, 255k steps)",
                 fontsize=10.5)
    fig.text(0.5, 0.01, "Grey: true streets the trips used. Red: edges the trips imply that don't exist "
             "(curved toward the claimed direction). Same trip endpoints for every model.\n"
             "The reconstruction processes trips in a random order, so counts vary between rebuilds of the "
             "same trips (baseline 767 vs 839, share 0.25 750 vs 907).",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    os.makedirs(args.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig7_reconstructed_maps.{ext}"), dpi=220, bbox_inches="tight")
    print("wrote", os.path.join(args.out, "fig7_reconstructed_maps.{png,pdf}"))


if __name__ == "__main__":
    main()
