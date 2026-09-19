"""Allocating a radiation survey over measured Safecast cells.

Two comparisons on one field. Holding the acquisition fixed and varying the mixing law asks
whether the third parameter reaches the decision, which is the question the paper is about
and the reason this study exists: the Safecast field is fitted at order -2.07, the furthest
of the three measured fields from anything a gamma can reach. Holding the model fixed and
varying the acquisition asks whether the criterion matters, and is run for completeness.

The budget is swept rather than fixed. An abundant budget leaves every cell well measured
whatever the schedule does, and nothing separates there; the sweep shows where the transition
is instead of asserting a regime.

Every agent on a survey sees the same cells, the same priors and the same decay streams, so a
difference between two agents is a difference in decisions rather than in luck.

Run: python experiments/experiment-safecast/run.py
"""

import csv
import os
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import agents                                                          # noqa: E402
from baselines import BudgetedSystematic                               # noqa: E402
from environment import SafecastSurvey                                 # noqa: E402
from methods.countmodels import MODELS                                 # noqa: E402

SEED = 0
N_ENV = 32                  # surveys
N_HELDOUT = 24
TOP_M = 8                   # cells a remediation plan would commit to
#: Detector-time budgets, seconds. 480 s lets every cell be touched at the shortest dwell;
#: 60 s lets one in eight. The transition is somewhere between.
BUDGETS = (480.0, 240.0, 120.0, 60.0)
HEADLINE = 120.0


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


# -- correctness gate ------------------------------------------------------

def _nested_mc_eig(model, belief, f, rng, n=4000):
    lam = model.sample_rate(belief, n, rng)
    y = rng.poisson(f * lam).astype(float)
    ll_out = -f * lam + y * np.log(f * lam) - gammaln(y + 1.0)
    lam_in = model.sample_rate(belief, n, rng)
    mu = f * lam_in
    ll_in = -mu[None, :] + y[:, None] * np.log(mu)[None, :] - gammaln(y + 1.0)[:, None]
    return float(np.mean(ll_out - (logsumexp(ll_in, axis=1) - np.log(n))))


def gate(env, verbose=True):
    """Each model's acquisition against nested Monte Carlo, on this field's own priors."""
    rng = np.random.default_rng(SEED)
    rows, bad = [], 0
    for cls in MODELS:
        model = cls()
        priors = env.initial_beliefs(model)
        for k in (0, env.n_cells // 2):
            for t in (env.dwells[0], env.dwells[-1]):
                f = env.exposure((k, t))
                exact = model.eig(priors[k], f)
                lo = [_nested_mc_eig(model, priors[k], f, rng, 1500) for _ in range(4)]
                hi = [_nested_mc_eig(model, priors[k], f, rng, 6000) for _ in range(4)]
                corrected = (4.0 * np.mean(hi) - np.mean(lo)) / 3.0
                sem = float(np.sqrt((16.0 * np.var(hi, ddof=1) / 4
                                     + np.var(lo, ddof=1) / 4) / 9.0))
                z = (exact - corrected) / max(sem, 1e-12)
                rows.append((model.name, k, float(t), f, exact, corrected, sem, z))
                if abs(z) > 4.0 or exact < -1e-12:
                    bad += 1
    if verbose:
        print("Correctness gate: each model's EIG against nested Monte Carlo")
        for name, k, t, f, e, c, s, z in rows:
            print("  %-22s cell %3d  dwell %5.0fs  exact %9.6f  mc %9.6f  z %+6.2f"
                  % (name, k, t, e, c, z))
        print("  ->  %s (%d of %d outside 4 sigma)"
              % ("PASS" if bad == 0 else "FAIL", bad, len(rows)))
    return bad == 0, rows


# -- one survey, one agent -------------------------------------------------

def evaluate(env, agent, truths, heldout):
    """``(nlpd, dex, topm_regret)``: held-out score, log-rate error, remediation regret."""
    f_eval = env.eval_dwell / 60.0
    nlpd, dex = [], []
    for k in range(env.n_cells):
        nlpd.append(-float(np.mean(agent.logpmf(k, f_eval,
                                                np.asarray(heldout[k], dtype=float)))))
        dex.append(abs(np.log10(max(agent.rate_mean(k), 1e-12)) - np.log10(truths[k])))
    means = np.array([agent.rate_mean(k) for k in range(env.n_cells)])
    chosen = np.argsort(-means)[:TOP_M]
    best = np.sort(truths)[::-1][:TOP_M]
    regret = float(best.sum() - truths[chosen].sum()) / float(best.sum())
    return float(np.mean(nlpd)), float(np.mean(dex)), regret


def run_agent(env, agent, moments, truths, streams, heldout, rng, budget):
    agent.reset(env, moments, np.ones(env.n_cells))
    consumed = np.zeros(env.n_cells)
    alloc = np.zeros(env.n_cells)
    spent, rounds, decide = 0.0, 0, 0.0
    while spent + env.dwells.min() <= budget + 1e-9:
        t0 = time.perf_counter()
        k, t = agent.act(env, rng)
        decide += time.perf_counter() - t0
        if spent + t > budget + 1e-9:
            t = float(env.dwells.min())
        f = env.exposure((k, t))
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(streams[k], hi) - np.searchsorted(streams[k], lo))
        consumed[k] = hi
        agent.observe(k, y, f)
        alloc[k] += t
        spent += t
        rounds += 1
    return (rounds, decide) + evaluate(env, agent, truths, heldout), alloc


# -- the study -------------------------------------------------------------

def main():
    rd = os.path.join(HERE, "results")
    env = SafecastSurvey()

    ok, gate_rows = gate(env)
    save_csv(os.path.join(rd, "gate_eig.csv"),
             ["model", "cell", "dwell", "exposure", "eig_exact", "eig_corrected", "sem", "z"],
             gate_rows)
    if not ok:
        print("\nABORT: an acquisition failed its correctness gate.")
        return 1

    built = agents.build_all(cross="models")
    print("\nSurvey study: %d surveys, %d cells, budgets %s s, %d agents"
          % (N_ENV, env.n_cells, "/".join("%.0f" % b for b in BUDGETS), len(built)))

    rows, alloc_rows = [], []
    for e in range(N_ENV):
        rng = np.random.default_rng(SEED + e)
        moments, truths, _ = env.blocks(rng)
        streams = env.stone_fields(truths, None, rng)
        heldout = env.held_out(truths, None, rng, n=N_HELDOUT)
        for budget in BUDGETS:
            for agent in built + [BudgetedSystematic(budget=budget)]:
                (rounds, decide, nlpd, dex, regret), alloc = run_agent(
                    env, agent, moments, truths, streams, heldout,
                    np.random.default_rng(9000 + e), budget)
                rows.append((agent.name, e, budget, rounds, decide, nlpd, dex, regret))
                if budget == HEADLINE:
                    for k in range(env.n_cells):
                        alloc_rows.append((agent.name, e, k, env._strata[k],
                                           truths[k], alloc[k]))
        print("  survey %2d/%d done" % (e + 1, N_ENV))

    save_csv(os.path.join(rd, "survey.csv"),
             ["policy", "env", "budget", "rounds", "decide_seconds", "nlpd", "dex",
              "topm_regret"], rows)
    save_csv(os.path.join(rd, "allocation.csv"),
             ["policy", "env", "cell", "stratum", "truth_cpm", "dwell_s"], alloc_rows)
    report(rows)
    return 0


def report(rows):
    from scipy import stats
    import collections
    by = collections.defaultdict(dict)
    for name, e, budget, rounds, decide, nlpd, dex, regret in rows:
        by[budget].setdefault(name, []).append((e, nlpd, dex, regret, decide, rounds))

    for budget in BUDGETS:
        tab = by[budget]
        base = np.array([r[1] for r in sorted(tab["eig"])])
        print("\nBudget %.0f s  (%.1f s per cell, at most %d cells touched)"
              % (budget, budget / 96.0, budget / 5.0))
        print("  %-24s %8s %7s %8s %7s %8s %7s"
              % ("agent", "NLPD", "t", "dex", "t", "regret", "t"))
        for name in sorted(tab, key=lambda n: np.mean([r[1] for r in tab[n]])):
            v = sorted(tab[name])
            a = np.array([r[1] for r in v]); d = np.array([r[2] for r in v])
            g = np.array([r[3] for r in v])
            bd = np.array([r[2] for r in sorted(tab["eig"])])
            bg = np.array([r[3] for r in sorted(tab["eig"])])
            ta = 0.0 if name == "eig" else -stats.ttest_rel(a, base).statistic
            td = 0.0 if name == "eig" else -stats.ttest_rel(d, bd).statistic
            tg = 0.0 if name == "eig" else -stats.ttest_rel(g, bg).statistic
            print("  %-24s %8.4f %7.1f %8.4f %7.1f %8.4f %7.1f"
                  % (name, a.mean(), ta, d.mean(), td, g.mean(), tg))


if __name__ == "__main__":
    raise SystemExit(main())
