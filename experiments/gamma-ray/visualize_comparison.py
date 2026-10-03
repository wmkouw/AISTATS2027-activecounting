"""Figures for the agent comparison on gamma-ray photometry.

Reads ``experiments/comparison/results/gamma.csv`` (the default 5-95 percentile catalogue
band) and, where they exist, ``band1-99_gamma.csv`` (the gentler cut) and
``band1-99_n192_eval25_gamma.csv`` (the survey regime: 192 targets on the same 120 Ms,
NLPD scored on 25 Ms integrations); writes into ``figures/``, each set under its prefix:

    comparison_nlpd_box.pdf      NLPD per Monte Carlo run at the full budget, per method
    comparison_regret_topm.pdf   mean top-m regret against budget, m = 1..5
    comparison_nlpd_time.pdf     mean NLPD against ms per round (log scale)

Run after ``python experiments/comparison/run.py gamma [--band 1 99]``:
    python experiments/gamma-ray/visualize_comparison.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import comparison_figs                                                # noqa: E402

if __name__ == "__main__":
    for prefix, title in (("", "Gamma-ray, 5-95% flux band"),
                          ("band1-99_", "Gamma-ray, 1-99% flux band"),
                          ("band1-99_n192_eval25_",
                           "Gamma-ray survey: 192 targets, scored at 25 Ms")):
        if os.path.exists(os.path.join(comparison_figs.RESULTS, prefix + "gamma.csv")):
            comparison_figs.make_all("gamma", prefix, os.path.join(HERE, "figures"),
                                     title, "Ms")
