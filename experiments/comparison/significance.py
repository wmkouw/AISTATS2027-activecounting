"""Significance tests between the models on held-out NLPD, per setting.

Every agent in an episode faces the same ground truth, the same photon or stone streams and
the same held-out counts, so the per-episode NLPD values are paired by episode. Per setting:

    Friedman      omnibus test that the methods' NLPD distributions differ, over episodes.
    pairwise      every pair of methods, by a paired t-test on the per-episode differences
                  and by a Wilcoxon signed-rank test, which does not assume they are normal.
    Holm          both families of pairwise p-values corrected within the setting, so a
                  setting's table controls the family-wise error rate at its level.

Reported per pair: mean difference (row minus column, so negative means the row is better),
its 95 per cent confidence interval, the number of episodes in which the row is better, and
the Holm-adjusted p-values. Scored at the final checkpoint of each setting. Holm is applied
over every pair in a setting; the printout shows the pairs against the proposed method, the
CSV all of them.

Writes ``results/significance.csv`` and prints one block per setting.

Run: python experiments/comparison/significance.py
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SETTINGS = [
    ("bulk", "bulk sampling (240 m^3)"),
    ("gamma", "gamma-ray, 5-95% band, 48 sources"),
    ("band1-99_gamma", "gamma-ray, 1-99% band, 48 sources"),
    ("band1-99_n192_eval25_gamma", "gamma-ray survey, 192 sources, scored at 25 Ms"),
    ("hardxray", "hard X-ray follow-up, 192 sources"),
]
METHODS = ["eig", "eig@gamma-poisson", "eig@gammamix-poisson", "eig@logskewnormal-quad",
           "eig@logskewnormal-poisson", "eig@logskewnormal-poisson-m8k", "systematic", "random",
           "d-optimality", "thompson", "lucb", "lucb@gamma-poisson", "dad",
           "bo-ei@gp-poisson", "bo-ei@gp-poisson-cat"]
SHORT = {"eig": "GIG", "eig@gamma-poisson": "gamma", "eig@gammamix-poisson": "gamma-mix",
         "eig@logskewnormal-quad": "LSN-quad", "eig@logskewnormal-poisson": "LSN-MCMC",
         "eig@logskewnormal-poisson-m8k": "LSN-MCMC-8k", "systematic": "systematic",
         "random": "random", "d-optimality": "D-opt", "thompson": "Thompson",
         "lucb": "LUCB-GIG", "lucb@gamma-poisson": "LUCB-gamma", "dad": "DAD",
         "bo-ei@gp-poisson": "BO-RBF", "bo-ei@gp-poisson-cat": "BO-cat"}
#: The reference every method is printed against; the CSV holds every pair.
REFERENCE = "eig"


def holm(p):
    """Holm step-down adjustment of a vector of p-values."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (p.size - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def setting_table(name):
    d = pd.read_csv(os.path.join(RESULTS, name + ".csv"))
    fin = d[np.isclose(d.spent, d.spent.max())]
    wide = fin.pivot(index="episode", columns="agent", values="nlpd")
    methods = [m for m in METHODS if m in wide.columns]
    wide = wide[methods].dropna()
    fried = stats.friedmanchisquare(*[wide[m].values for m in methods])
    rows = []
    for a, b in itertools.combinations(methods, 2):
        x = wide[a].values - wide[b].values
        n = x.size
        se = x.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        t = stats.ttest_rel(wide[a].values, wide[b].values)
        w = stats.wilcoxon(wide[a].values, wide[b].values, zero_method="pratt")
        rows.append(dict(setting=name, a=a, b=b, n=n, mean_diff=x.mean(),
                         ci_lo=x.mean() - half, ci_hi=x.mean() + half,
                         a_better=int((x < 0).sum()), t=t.statistic, p_t=t.pvalue,
                         p_wilcoxon=w.pvalue))
    tab = pd.DataFrame(rows)
    tab["p_t_holm"] = holm(tab.p_t)
    tab["p_wilcoxon_holm"] = holm(tab.p_wilcoxon)
    means = wide.mean()
    return tab, fried, means, wide.shape[0]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    out = []
    for name, label in SETTINGS:
        path = os.path.join(RESULTS, name + ".csv")
        if not os.path.exists(path):
            continue
        tab, fried, means, n = setting_table(name)
        out.append(tab)
        print("\n== {} ({} episodes)".format(label, n))
        print("   mean NLPD: " + ", ".join("{} {:.4f}".format(SHORT[m], v)
                                          for m, v in means.items()))
        print("   Friedman chi2 = {:.1f}, p = {:.1e}".format(fried.statistic, fried.pvalue))
        print("   {:<15} {:>9} {:>21} {:>7} {:>7} {:>10} {:>10}".format(
            "A vs B", "A - B", "95% CI", "A wins", "t", "p_t Holm", "p_W Holm"))
        for r in tab[(tab.a == REFERENCE) | (tab.b == REFERENCE)].itertuples():
            print("   {:<15} {:>+9.4f} [{:>+8.4f}, {:>+8.4f}] {:>4d}/{:<2d} {:>+7.1f} "
                  "{:>7.1e} {:<4} {:>7.1e} {:<4}".format(
                      SHORT[r.a] + " vs " + SHORT[r.b], r.mean_diff, r.ci_lo, r.ci_hi,
                      r.a_better, r.n, r.t, r.p_t_holm, stars(r.p_t_holm),
                      r.p_wilcoxon_holm, stars(r.p_wilcoxon_holm)))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(RESULTS, "significance.csv"), index=False)
    print("\nwrote {}".format(os.path.relpath(os.path.join(RESULTS, "significance.csv"))))


if __name__ == "__main__":
    main()
