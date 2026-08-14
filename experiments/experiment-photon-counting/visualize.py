"""Figures for the photon-counting study.

Reads ``results/``, writes ``figures/photon.pdf``. Run after ``fit.py`` and ``run.py``.

Three panels for the three claims. (a) The mixing law the real flux field needs, fitted
rather than assumed. (b) What the acquisition is worth once the budget is scarce enough that
most targets cannot be observed. (c) Where the telescope time went, against how uncertain
each target's class prior was, which is the mechanism behind (b).

Run: python experiments/experiment-photon-counting/visualize.py
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "experiment-gigpoisson"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402
from fetch import fetch                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()

STYLE = {
    "eig": dict(color=viz.CAT[0], linewidth=2.4, zorder=6),
    "eig@gamma-poisson": dict(color=viz.CAT[1], linewidth=2.0, zorder=5),
    "eig@lognormal-poisson": dict(color=viz.CAT[2], linewidth=2.0, zorder=5),
    "uniform": dict(color=viz.INK["primary"], linewidth=1.4, linestyle=(0, (5, 2))),
    "d-optimality": dict(color=viz.INK["secondary"], linewidth=1.1,
                         linestyle=(0, (6, 2, 1, 2))),
    "neyman": dict(color=viz.INK["secondary"], linewidth=1.1, linestyle=(0, (1, 2))),
    "maxent": dict(color=viz.INK["muted"], linewidth=1.1),
    "thompson": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (2, 2, 1, 2))),
    "random": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (2, 3))),
    "epig": dict(color=viz.INK["muted"], linewidth=1.1, linestyle=(0, (4, 2))),
    "predictive-variance": dict(color=viz.INK["muted"], linewidth=0.9,
                                linestyle=(0, (1, 3))),
    "epistemic": dict(color=viz.INK["muted"], linewidth=0.9, linestyle=(0, (3, 3))),
}
LABEL = {"eig": "EIG, GIG-Poisson (ours)", "eig@gamma-poisson": "EIG, gamma-Poisson",
         "eig@lognormal-poisson": "EIG, lognormal-Poisson", "uniform": "systematic",
         "d-optimality": "D-optimality", "neyman": "Neyman alloc.",
         "maxent": "max-entropy", "thompson": "Thompson", "random": "random",
         "epig": "EPIG", "predictive-variance": "total var.",
         "epistemic": "epistemic var."}


def load(name):
    return np.atleast_1d(np.genfromtxt(os.path.join(HERE, "results", name),
                                       delimiter=",", names=True, dtype=None,
                                       encoding="utf-8"))


def panel_fit(ax):
    """(a) The measured flux field, against the best fit of each mixing law."""
    flux = fetch() * 1e10
    lo, hi = np.percentile(flux, 5), np.percentile(flux, 95)
    band = flux[(flux >= lo) & (flux <= hi)]
    edges = np.geomspace(band.min(), band.max(), 45)
    ax.hist(band, bins=edges, density=True, color=viz.INK["grid"],
            edgecolor=viz.INK["axis"], linewidth=0.4, label="4FGL sources")
    x = np.geomspace(band.min(), band.max(), 400)
    a, _, sc = stats.gamma.fit(band, floc=0)
    ax.plot(x, stats.gamma.pdf(x, a, 0, sc), color=viz.CAT[1], label="gamma")
    s, _, sc = stats.lognorm.fit(band, floc=0)
    ax.plot(x, stats.lognorm.pdf(x, s, 0, sc), color=viz.CAT[2], label="lognormal")
    p, b, _, sc = stats.geninvgauss.fit(band, floc=0)
    ax.plot(x, stats.geninvgauss.pdf(x, p, b, 0, sc), color=viz.CAT[0], linewidth=2.2,
            label="GIG (ours)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel(r"photon flux ($10^{-10}$ cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("density")
    ax.set_title("(a) the mixing law the sky needs", loc="left")
    ax.legend(loc="lower left", fontsize=7.5)


def panel_policy(ax, seq):
    for name, st in STYLE.items():
        m = seq["policy"] == name
        if not np.any(m):
            continue
        spent = np.unique(seq["spent"][m])
        mean = np.array([np.mean(seq["nlpd"][m & (seq["spent"] == s)]) for s in spent])
        sem = np.array([np.std(seq["nlpd"][m & (seq["spent"] == s)], ddof=1)
                        / np.sqrt(np.sum(m & (seq["spent"] == s))) for s in spent])
        ax.plot(spent, mean, **st)
        if name in ("eig", "eig@gamma-poisson"):
            ax.fill_between(spent, mean - sem, mean + sem, color=st["color"],
                            alpha=0.18, linewidth=0)
    ax.set_xlabel("telescope time spent (Ms)")
    ax.set_ylabel("held-out NLPD (nats)")
    ax.set_title("(b) 48 targets, 120 Ms", loc="left")
    ax.legend(handles=[plt.Line2D([], [], **STYLE[k], label=LABEL[k])
                       for k in ("eig", "eig@gamma-poisson", "eig@lognormal-poisson",
                                 "uniform", "maxent")],
              loc="upper right", fontsize=6.5, handlelength=2.2, labelspacing=0.3)


def panel_allocation(ax, alloc, seq):
    """(c) Time given to a target against how uncertain its class prior was."""
    for name in ("eig", "uniform", "maxent"):
        m = alloc["policy"] == name
        g, v = alloc["flux_true"][m], alloc["time_allocated"][m]
        order = np.argsort(g)
        edges = np.quantile(g, np.linspace(0, 1, 9))
        idx = np.clip(np.digitize(g[order], edges[1:-1]), 0, 7)
        ax.plot([np.median(g[order][idx == i]) for i in range(8)],
                [np.mean(v[order][idx == i]) for i in range(8)],
                marker="o", markersize=3.2, markeredgewidth=0.0, **STYLE[name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true flux ($10^{-10}$ cm$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("telescope time given (Ms)")
    ax.set_title("(c) where the time went", loc="left")


def main():
    seq = load("sequential.csv")
    alloc = load("allocation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    panel_fit(axes[0])
    panel_policy(axes[1], seq)
    panel_allocation(axes[2], alloc, seq)
    viz.savefig(fig, os.path.join(HERE, "figures", "photon.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
