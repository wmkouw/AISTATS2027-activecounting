"""Figures for the five-agent comparison on hard X-ray follow-up of Swift-BAT sources.

Reads ``experiments/comparison/results/hardxray.csv``; writes into ``figures/``:

    comparison_nlpd_box.pdf      NLPD per Monte Carlo run at the full budget, per method
    comparison_regret_topm.pdf   mean top-m regret against budget, m = 1..5
    comparison_nlpd_time.pdf     mean NLPD against ms per round (log scale)

Run after ``python experiments/comparison/run.py hardxray``:
    python experiments/hard-xray/visualize_comparison.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import comparison_figs                                                # noqa: E402

if __name__ == "__main__":
    comparison_figs.make_all("hardxray", "", os.path.join(HERE, "figures"),
                             "Hard X-ray follow-up of Swift-BAT sources", "ks")
