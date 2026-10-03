"""Figures for the six-agent comparison, shared by the bulk-sampling and gamma-ray studies.

Reads ``experiments/comparison/results/<prefix><setting>.csv`` (one row per agent, episode
and checkpoint, written by ``experiments/comparison/run.py``) and draws three figures:

    nlpd_box      NLPD at the full budget, one box per method over the Monte Carlo runs,
                  with every run drawn as a point.
    regret_topm   mean simple top-m regret against budget spent, one panel per m = 1..5,
                  shaded one standard error.
    nlpd_time     mean NLPD against mean wall-clock ms per round (design + update), log x.

**Encoding.** Six methods, but the palette clears colour-vision separation across all pairs
for three slots only, and the scatter and small multiples put every pair side by side. So
colour goes to the three EIG models the paper is about (slot order fixed), and the other
three are neutral inks told apart by marker and dash. Every method is also named on the
figure itself -- tick labels on the boxplot, direct labels on the lines and the points --
so no identity is carried by colour alone.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import viz

__all__ = ["METHODS", "load", "nlpd_box", "regret_topm", "nlpd_time", "make_all"]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "experiments", "comparison", "results")

#: Plot order, short label, and style. Order is fixed so a method keeps its place and its
#: look in every figure of both studies.
def _ink(colour, marker, dash, width=1.6, z=3):
    return dict(color=colour, marker=marker, linestyle=dash, linewidth=width, zorder=z)


METHODS = [
    ("eig", "GIG-Poisson (ours)",
     dict(color=viz.CAT[0], marker="o", linestyle="-", linewidth=2.4, zorder=6)),
    ("eig@gamma-poisson", "gamma-Poisson",
     dict(color=viz.CAT[1], marker="s", linestyle="-", linewidth=2.0, zorder=5)),
    ("eig@logskewnormal-poisson", "log-skew-normal\n(MCMC + NMC)",
     dict(color=viz.CAT[2], marker="D", linestyle="-", linewidth=2.0, zorder=5)),
    ("eig@gammamix-poisson", "gamma mixture",
     _ink(viz.INK["primary"], "X", (0, (5, 2)), 1.8, 4)),
    ("eig@logskewnormal-quad", "log-skew-normal\n(exact quadrature)",
     _ink(viz.INK["secondary"], "d", (0, (3, 1.5)), 1.8, 4)),
    ("eig@logskewnormal-poisson-m8k", "log-skew-normal\n(MCMC + NMC, M=8192)",
     _ink(viz.INK["secondary"], "h", (0, (1, 1)), 1.8, 4)),
    ("eig@lognormal-likelihood", "log-normal lik.",
     _ink(viz.INK["secondary"], "v", (0, (5, 2)))),
    ("systematic", "systematic", _ink(viz.INK["primary"], "|", (0, (8, 2)))),
    ("random", "random", _ink(viz.INK["muted"], "x", (0, (2, 2)))),
    ("d-optimality", "D-optimality", _ink(viz.INK["secondary"], "<", (0, (6, 2, 1, 2)))),
    ("thompson", "Thompson", _ink(viz.INK["muted"], ">", (0, (1, 2)))),
    ("lucb", "Bayes-LUCB (GIG)", _ink(viz.INK["primary"], "*", (0, (4, 1, 1, 1)))),
    ("lucb@gamma-poisson", "Bayes-LUCB (gamma)", _ink(viz.INK["secondary"], "8", (0, (4, 1, 1, 1)))),
    ("dad", "amortised policy (DAD)", _ink(viz.INK["primary"], "p", (0, (3, 1, 1, 1, 1, 1)))),
    ("bo-ei@gp-poisson", "BO-EI (RBF)",
     _ink(viz.INK["muted"], "^", (0, (1, 1.5)))),
    ("bo-ei@gp-poisson-cat", "BO-EI (categorical)",
     _ink(viz.INK["muted"], "P", (0, (6, 2, 1, 2)))),
]

#: Two views of one study, so that no figure carries more methods than it can label: the
#: model axis (every model under expected information gain) and the acquisition axis (every
#: allocation rule, against ours).
GROUPS = {
    "models": ["eig", "eig@gamma-poisson", "eig@gammamix-poisson"],
    # The log-skew-normal has the GIG's flexibility and reaches its performance, more slowly;
    # it is reported in an appendix rather than alongside the main models.
    "appendix_lsn": ["eig", "eig@logskewnormal-quad", "eig@logskewnormal-poisson",
                     "eig@logskewnormal-poisson-m8k"],
    "acquisitions": ["eig", "systematic", "random", "d-optimality", "thompson", "lucb",
                     "lucb@gamma-poisson", "dad", "bo-ei@gp-poisson", "bo-ei@gp-poisson-cat"],
}
LABEL = {n: l for n, l, _ in METHODS}

#: Direct-label placement on the scatter, in points from the marker. Fixed per method so a
#: label sits in the same place in both studies; the two BO variants land on nearly the same
#: point, so one is labelled above and one below.
LABEL_AT = {
    "eig": dict(xytext=(0, -14), ha="center", va="top"),
    "bo-ei@gp-poisson": dict(xytext=(9, 6), ha="left", va="bottom"),
    "bo-ei@gp-poisson-cat": dict(xytext=(9, -6), ha="left", va="top"),
    "lucb@gamma-poisson": dict(xytext=(9, -6), ha="left", va="top"),
    "eig@logskewnormal-poisson-m8k": dict(xytext=(9, -6), ha="left", va="top"),
    "eig@logskewnormal-quad": dict(xytext=(6, -18), ha="left", va="top"),
    "eig@logskewnormal-poisson": dict(xytext=(6, 9), ha="left", va="bottom"),
    "eig@gammamix-poisson": dict(xytext=(-8, 6), ha="right", va="bottom"),
}
_LABEL_DEFAULT = dict(xytext=(8, 5), ha="left", va="bottom")
STYLE = {n: s for n, _, s in METHODS}


def load(setting, prefix=""):
    d = pd.read_csv(os.path.join(RESULTS, "{}{}.csv".format(prefix, setting)))
    d["ms_round"] = 1e3 * (d.act_seconds + d.observe_seconds) / d.rounds.clip(lower=1)
    return d


def _present(d):
    have = set(d.agent)
    return [n for n, _, _ in METHODS if n in have]


def _final(d):
    return d[np.isclose(d.spent, d.spent.max())]


# --------------------------------------------------------------------------

def nlpd_box(d, title, unit_budget):
    """Figure 1: NLPD per Monte Carlo run at the full budget."""
    fin = _final(d)
    names = _present(d)
    data = [fin[fin.agent == n].nlpd.values for n in names]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.05 * len(names)), 3.0))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=viz.INK["primary"], linewidth=1.4),
                    whiskerprops=dict(color=viz.INK["axis"], linewidth=1.0),
                    capprops=dict(color=viz.INK["axis"], linewidth=1.0))
    rng = np.random.default_rng(0)
    for i, (n, box) in enumerate(zip(names, bp["boxes"])):
        c = STYLE[n]["color"]
        box.set(facecolor=c, alpha=0.18, edgecolor=c, linewidth=1.2)
        x = i + 1 + rng.uniform(-0.14, 0.14, size=data[i].size)
        ax.scatter(x, data[i], s=9, color=c, alpha=0.75, linewidths=0,
                   marker=STYLE[n]["marker"], zorder=4)
    ax.set_xticks(range(1, len(names) + 1))
    many = len(names) > 6
    ax.set_xticklabels([LABEL[n].replace("\n", " ") if many else LABEL[n] for n in names],
                       fontsize=7.5, rotation=25 if many else 0, ha="right" if many else "center",
                       rotation_mode="anchor")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("NLPD (nats), lower is better")
    ax.set_title("{}: held-out NLPD at the full budget ({}), {} runs"
                 .format(title, unit_budget, fin.episode.nunique()), loc="left")
    return fig


def regret_topm(d, title, budget_label, ms=(1, 2, 3, 4, 5)):
    """Figure 2: mean top-m regret against budget, one panel per m."""
    names = _present(d)
    ms = [m for m in ms if "regret_top{}".format(m) in d.columns]
    fig, axes = plt.subplots(1, len(ms), figsize=(2.05 * len(ms) + 0.6, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    top = 0.0
    for ax, m in zip(axes, ms):
        col = "regret_top{}".format(m)
        for n in names:
            g = d[d.agent == n].groupby("spent")[col]
            mu, se = g.mean(), g.std(ddof=1) / np.sqrt(g.count())
            top = max(top, float((mu + se).max()))
            st = STYLE[n]
            ax.fill_between(mu.index, (mu - se).clip(lower=0), mu + se, color=st["color"],
                            alpha=0.12, linewidth=0, zorder=st["zorder"] - 1)
            ax.plot(mu.index, mu.values, color=st["color"], linestyle=st["linestyle"],
                    linewidth=st["linewidth"], marker=st["marker"], markersize=4,
                    zorder=st["zorder"], label=LABEL[n].replace("\n", " "))
        ax.set_title("top-{}".format(m), loc="left")
        ax.set_xlabel(budget_label)
    # Set once, after every panel is drawn: the axes share y, and a limit set per panel
    # would freeze the top at whatever the earlier panels happened to reach.
    axes[0].set_ylim(0.0, 1.05 * top)
    axes[0].set_ylabel("mean simple regret")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(len(names), 5),
               fontsize=7.5)
    fig.suptitle("{}: top-m regret (shaded: $\\pm$1 s.e. over {} runs)"
                 .format(title, d.episode.nunique()), x=0.01, ha="left",
                 fontsize=10, color=viz.INK["primary"])
    return fig


def nlpd_time(d, title):
    """Figure 3: mean NLPD against mean ms per round, log x, with one-s.e. bars."""
    fin = _final(d)
    names = _present(d)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for n in names:
        g = fin[fin.agent == n]
        x, y = g.ms_round.mean(), g.nlpd.mean()
        xe = g.ms_round.std(ddof=1) / np.sqrt(len(g))
        ye = g.nlpd.std(ddof=1) / np.sqrt(len(g))
        st = STYLE[n]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=st["marker"], color=st["color"],
                    markersize=8, markeredgecolor=viz.INK["surface"], markeredgewidth=1.2,
                    elinewidth=1.0, capsize=0, zorder=st["zorder"])
        ax.annotate(LABEL[n].replace("\n", " "), (x, y), textcoords="offset points",
                    fontsize=7.5, color=viz.INK["secondary"],
                    **LABEL_AT.get(n, _LABEL_DEFAULT))
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock per round, design + update (ms, log scale)")
    ax.set_ylabel("mean NLPD (nats), lower is better")
    ax.set_title("{}: accuracy against cost, {} runs".format(title, fin.episode.nunique()),
                 loc="left")
    ax.margins(x=0.12, y=0.15)
    return fig


def make_all(setting, prefix, outdir, title, budget_unit):
    """Write the three figures for one setting into ``outdir``, once per group of methods.

    Files are ``comparison_<prefix><group>_{nlpd_box,regret_topm,nlpd_time}.pdf``; a group
    with fewer than two methods in the results is skipped.
    """
    viz.apply_style()
    full = load(setting, prefix)
    os.makedirs(outdir, exist_ok=True)
    final = full.spent.max()
    for group, members in GROUPS.items():
        d = full[full.agent.isin(members)]
        if d.agent.nunique() < 2:
            continue
        stem = os.path.join(outdir, "comparison_{}{}_".format(prefix, group))
        sub = "{} ({})".format(title, group)
        viz.savefig(nlpd_box(d, sub, "{:g} {}".format(final, budget_unit)),
                    stem + "nlpd_box.pdf")
        viz.savefig(regret_topm(d, sub, "budget spent ({})".format(budget_unit)),
                    stem + "regret_topm.pdf")
        viz.savefig(nlpd_time(d, sub), stem + "nlpd_time.pdf")
