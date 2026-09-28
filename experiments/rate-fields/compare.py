"""Which mixing law do the measured fields of rates need?

Every other experiment in this paper either generates its rates from the law of Section 3.2,
which makes the model correct by construction, or argues from one field. This one asks the
question of both physical fields at once and lets them disagree.

The linguistic field is fitted by ``experiments/newsgroups`` instead, with the
same laws and the same code: it is a control on the claim rather than evidence for it, and
the fitting loop here is exported so the two cannot drift apart.

Each mixing law is fitted to each field by maximum likelihood with the location held at
zero, so that every family has exactly the parameters its name implies, and scored by AIC so
that the third parameter pays its penalty. Nothing about the acquisition, the budget or the
policies enters: this is a statement about the model alone, and it is the only claim in the
paper that survives independently of every design choice made elsewhere.

Writes ``results/rate_field_fits.csv``. Figures come from ``visualize.py``.

Run: python experiments/rate-fields/compare.py
"""

import csv
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fields import PHYSICAL_FIELDS                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

LAWS = (
    ("gamma", 2, lambda x: stats.gamma.fit(x, floc=0.0),
     lambda x, p: stats.gamma.logpdf(x, p[0], 0.0, p[2]).sum(),
     lambda p: "shape={:.3g}".format(p[0])),
    ("lognormal", 2, lambda x: stats.lognorm.fit(x, floc=0.0),
     lambda x, p: stats.lognorm.logpdf(x, p[0], 0.0, p[2]).sum(),
     lambda p: "sigma={:.3g}".format(p[0])),
    ("GIG", 3, lambda x: stats.geninvgauss.fit(x, floc=0.0),
     lambda x, p: stats.geninvgauss.logpdf(x, p[0], p[1], 0.0, p[3]).sum(),
     lambda p: "order={:+.3g}, omega={:.3g}".format(p[0], p[1])),
)


def fit_fields(fields):
    """Fit every law to every field, printing as it goes. Returns the rows to write."""
    rows = []
    for field_name, load, what in fields:
        x = load()
        fits = {}
        for law, k, fit, ll, describe in LAWS:
            params = fit(x)
            fits[law] = (k, float(ll(x, params)), describe(params))
        best = min(2 * k - 2 * l for k, l, _ in fits.values())
        print("\n{}: {} observations, each a {}".format(field_name, x.size, what))
        print("  variance-to-mean {:.2f}, max/median {:.1f}".format(
            x.var() / x.mean(), x.max() / np.median(x)))
        print("  {:<12} {:>2} {:>12} {:>11} {:>9}   {}".format(
            "law", "k", "log-lik", "AIC", "dAIC", "fitted"))
        for law, (k, l, desc) in fits.items():
            aic = 2 * k - 2 * l
            print("  {:<12} {:>2} {:>12.1f} {:>11.1f} {:>9.1f}   {}".format(
                law, k, l, aic, aic - best, desc))
            rows.append((field_name, x.size, law, k, l, aic, aic - best,
                         float(x.var() / x.mean()), desc))
        # The per-observation figure is the comparable one: a field with more observations
        # will show a larger total advantage for the same strength of evidence.
        per = (2 * 2 - 2 * fits["gamma"][1] - best) / x.size
        print("  advantage of the third parameter over the gamma: "
              "{:.0f} AIC total, {:.2f} per observation".format(
                  2 * 2 - 2 * fits["gamma"][1] - best, per))

    return rows


HEADER = ["field", "n", "law", "n_params", "loglik", "aic", "delta_aic",
          "variance_to_mean", "fitted"]


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    print("\n  wrote {} ({} rows)".format(os.path.basename(path), len(rows)))


def main():
    write_rows(os.path.join(HERE, "results", "rate_field_fits.csv"),
               fit_fields(PHYSICAL_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
