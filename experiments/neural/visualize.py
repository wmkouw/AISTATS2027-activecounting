"""Figures for the neural rate-field study.

Reads ``results/``, writes ``figures/neural.pdf``. Run after ``run.py``.

Two panels, neither of which needs the CRCNS archive. (a) The structural check: the Fano
factor the parameterisation produces across stimulus directions, against the one value
Taouali et al. quote exactly. It is bimodal because the inverse dispersion is tuned more
sharply than the mean, which is their Fig. 8C. (b) The obstacle: what the manuscript's
posterior does to counts generated with the rate redrawn every trial, which is the process
Taouali et al. actually describe.

Run: python experiments/neural/visualize.py
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import taouali as T                                                   # noqa: E402
import viz                                                            # noqa: E402

viz.apply_style()


def load(name):
    with open(os.path.join(HERE, "results", name), newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def panel_fano(ax):
    rows = load("fano_profile.csv")
    d = np.array([r["degrees_from_pd"] for r in rows])
    ff = np.array([r["fano_factor"] for r in rows])
    ax.plot(d, ff, color=viz.CAT[0], zorder=5)
    anchor = 1.0 + T.PD_MEAN / T.PD_PHI
    ax.plot([0.0], [anchor], marker="o", markersize=5, color=viz.CAT[1],
            markeredgecolor="white", markeredgewidth=0.7, linestyle="none", zorder=7)
    ax.annotate("quoted: $1+7.14/21.47$", xy=(0.0, anchor), xytext=(8, -12),
                textcoords="offset points", fontsize=7, color=viz.CAT[1])
    ax.axhline(1.0, color=viz.INK["secondary"], linewidth=0.9, zorder=4)
    ax.annotate("Poisson", xy=(0.97, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7,
                color=viz.INK["muted"], ha="right")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("direction relative to preferred ($^\\circ$)")
    ax.set_ylabel("Fano factor")
    ax.set_title("(a) tuned dispersion makes $F$ bimodal")


def panel_calibration(ax):
    rows = load("concentration.csv")
    style = {"fixed rate": (viz.CAT[0], "-", "rate fixed (manuscript)"),
             "redrawn per trial": (viz.CAT[1], "-", "rate redrawn (Taouali et al.)")}
    for reading, (colour, ls, label) in style.items():
        sel = sorted([r for r in rows if r["reading"] == reading],
                     key=lambda r: r["trials"])
        x = np.array([r["trials"] for r in sel])
        y = np.array([r["var_z"] for r in sel])
        ax.plot(x, y, color=colour, linestyle=ls, marker="o", markersize=3.5, zorder=5)
        ax.annotate(label, xy=(x[-1], y[-1]), xytext=(-4, 9 if colour == viz.CAT[1] else -12),
                    textcoords="offset points", color=colour, fontsize=7.5,
                    fontweight="bold", ha="right", annotation_clip=False)
    ff = float(np.mean([r["mean_fano"] for r in rows]))
    ax.axhline(ff, color=viz.INK["muted"], linewidth=0.9, linestyle=(0, (3, 2)), zorder=3)
    ax.annotate("mean Fano factor", xy=(0.03, ff), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7,
                color=viz.INK["muted"])
    ax.axhline(1.0, color=viz.INK["secondary"], linewidth=0.9, zorder=4)
    ax.annotate("calibrated", xy=(0.03, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7,
                color=viz.INK["secondary"])
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.8)
    ax.set_xlabel("trials at the context")
    ax.set_ylabel("$\\mathbb{V}[z]$, posterior overconfidence")
    ax.set_title("(b) the redraw is not a small-sample effect")


def main():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.0))
    panel_fano(axes[0])
    panel_calibration(axes[1])
    viz.savefig(fig, os.path.join(HERE, "figures", "neural.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
