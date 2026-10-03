"""Figures for the agent comparison on bulk sampling.

Reads ``experiments/comparison/results/bulk.csv``; writes into ``figures/``:

    comparison_nlpd_box.pdf      NLPD per Monte Carlo run at the full budget, per method
    comparison_regret_topm.pdf   mean top-m regret against budget, m = 1..5
    comparison_nlpd_time.pdf     mean NLPD against ms per round (log scale)

Run after ``python experiments/comparison/run.py bulk``:
    python experiments/bulk-sampling/visualize_comparison.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import comparison_figs                                                # noqa: E402

if __name__ == "__main__":
    comparison_figs.make_all("bulk", "", os.path.join(HERE, "figures"),
                             "Bulk sampling", "m$^3$")
