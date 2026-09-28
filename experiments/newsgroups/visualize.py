"""Figure for the linguistic field.

Reads the field itself, writes ``figures/newsgroups.pdf``. Run after ``run.py``.

One panel, matching the panels of ``rate-fields`` exactly -- same palette, same
axes, same annotation -- so that the two figures can be read against each other even though
they sit in different parts of the paper. The fitted order is the point: it comes out
positive here, inside the range a gamma mixing law can reach, where both physical fields come
out negative and outside it.

Run: python experiments/newsgroups/visualize.py
"""

import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rate-fields"))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import matplotlib.pyplot as plt                                       # noqa: E402

import viz                                                            # noqa: E402
from fields import LINGUISTIC_FIELDS                                  # noqa: E402

viz.apply_style()


def main():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    name, load, _what = LINGUISTIC_FIELDS[0]
    x = load()
    # Half the width of the two-panel physical figure, so a panel is the same size on the
    # page whichever of the two figures it appears in.
    fig, ax = plt.subplots(1, 1, figsize=(4.9, 1.65))
    edges = np.geomspace(np.percentile(x, 0.5), x.max(), 40)
    ax.hist(x, bins=edges, density=True, color=viz.INK["grid"],
            edgecolor=viz.INK["axis"], linewidth=0.4)
    grid = np.geomspace(edges[0], edges[-1], 400)

    # On this field the gamma and the generalised inverse Gaussian coincide, which is the
    # result and not a drawing error. Drawn at the same widths as the physical panels the
    # gamma vanishes underneath and the panel reads as though a curve is missing, so it is
    # widened here to show as a halo around the fit that sits on top of it.
    a, _, sc = stats.gamma.fit(x, floc=0.0)
    ax.plot(grid, stats.gamma.pdf(grid, a, 0.0, sc), color=viz.CAT[1],
            linewidth=4.0, alpha=0.85, solid_capstyle="round", label="gamma")
    s, _, sc = stats.lognorm.fit(x, floc=0.0)
    ax.plot(grid, stats.lognorm.pdf(grid, s, 0.0, sc), color=viz.CAT[2],
            label="lognormal")
    p, b, _, sc = stats.geninvgauss.fit(x, floc=0.0)
    ax.plot(grid, stats.geninvgauss.pdf(grid, p, b, 0.0, sc), color=viz.CAT[0],
            linewidth=1.6, label="GIG (ours)")
    ax.annotate("gamma and GIG coincide", xy=(0.97, 0.06), xycoords="axes fraction",
                fontsize=7, color=viz.INK["muted"], ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(max(1e-7, ax.get_ylim()[0]), None)
    ax.set_xlabel("rate")
    ax.set_ylabel("density")
    ax.set_title("{}\n$n={}$, fitted order ${:+.2f}$".format(name, x.size, p),
                 loc="left", fontsize=8.5)
    ax.legend(loc="lower left", fontsize=7.5)
    viz.savefig(fig, os.path.join(HERE, "figures", "newsgroups.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
