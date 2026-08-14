"""Figure for the three-field model comparison.

Reads ``results/rate_field_fits.csv`` and the fields themselves, writes
``figures/rate_fields.pdf``. Run after ``compare.py``.

One panel per field, each showing the measured rates against the best fit of every candidate
mixing law. The fitted order is printed on each panel because it is the diagnostic: the gamma
occupies the edge of the family at a positive order, so a field whose fitted order is
negative is a field the gamma cannot reach however its two parameters are set.

Run: python experiments/experiment-rate-fields/visualize.py
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
from fields import FIELDS                                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
viz.apply_style()


def main():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    for ax, (name, load, _what) in zip(axes, FIELDS):
        x = load()
        edges = np.geomspace(np.percentile(x, 0.5), x.max(), 40)
        ax.hist(x, bins=edges, density=True, color=viz.INK["grid"],
                edgecolor=viz.INK["axis"], linewidth=0.4)
        grid = np.geomspace(edges[0], edges[-1], 400)

        a, _, sc = stats.gamma.fit(x, floc=0.0)
        ax.plot(grid, stats.gamma.pdf(grid, a, 0.0, sc), color=viz.CAT[1],
                label="gamma")
        s, _, sc = stats.lognorm.fit(x, floc=0.0)
        ax.plot(grid, stats.lognorm.pdf(grid, s, 0.0, sc), color=viz.CAT[2],
                label="lognormal")
        p, b, _, sc = stats.geninvgauss.fit(x, floc=0.0)
        ax.plot(grid, stats.geninvgauss.pdf(grid, p, b, 0.0, sc), color=viz.CAT[0],
                linewidth=2.2, label="GIG (ours)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(max(1e-7, ax.get_ylim()[0]), None)
        ax.set_xlabel("rate")
        ax.set_title("{}\n$n={}$, fitted order ${:+.2f}$".format(name, x.size, p),
                     loc="left", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("density")
            ax.legend(loc="lower left", fontsize=7.5)
    viz.savefig(fig, os.path.join(HERE, "figures", "rate_fields.pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
