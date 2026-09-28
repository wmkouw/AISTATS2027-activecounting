"""Bayesian optimisation on a discrete-output Gaussian process, against the criteria.

The comparison the manuscript is missing on the model axis. Every mixing law in
``countmodels`` treats the contexts as independent; the surrogate here is a Gaussian process
over log-rates with a Poisson observation model, so the output is discrete and the kernel
lets a measurement at one context move the belief at another. It is driven by expected
improvement and by an upper confidence bound, the two standard Bayesian optimisation
acquisitions, rather than by an information criterion.

Run on both studies that give their contexts a class structure for the kernel to use:

    shape    the synthetic two-class design of ``synthetic/shape-scarcity``, where the
             classes differ in the shape of the prior and the budget is scarce.
    photon   the gamma-ray campaign, where the classes are real source classes and every
             agent's prior is fitted to the population of the source's own class.

**Nothing here writes to another study's results.** This module imports the two environments
and keeps its own ``results/``; the existing studies and the tables the manuscript inputs are
left exactly as they are.

Run: python experiments/bo-baseline/run.py [shape|photon|both]
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "synthetic", "shape-scarcity"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "gamma-ray"))

import agents                                                         # noqa: E402

SEED = 0
TOP_M = 4
N_HELDOUT = 24


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, HERE), len(rows)))


def paired_t(ours, theirs):
    d = np.asarray(ours, float) - np.asarray(theirs, float)
    if d.size < 2 or d.std(ddof=1) == 0.0:
        return 0.0
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


# --------------------------------------------------------------------------
# One episode, written against the shared environment protocol
# --------------------------------------------------------------------------

def evaluate(agent, n_ctx, f_eval, truths, heldout):
    nlpd, sq = [], []
    for k in range(n_ctx):
        nlpd.append(-float(np.mean(agent.logpmf(k, f_eval,
                                                np.asarray(heldout[k], dtype=float)))))
        est = max(agent.rate_mean(k), 1e-12)
        sq.append((np.log10(est) - np.log10(truths[k])) ** 2)
    means = np.array([agent.rate_mean(k) for k in range(n_ctx)])
    chosen = np.argsort(-means)[:TOP_M]
    best = np.sort(truths)[::-1][:TOP_M]
    regret = float(best.sum() - truths[chosen].sum()) / float(best.sum())
    return float(np.mean(nlpd)), float(np.sqrt(np.mean(sq))), regret


def run_agent(env, agent, moments, truths, recovery, streams, heldout, budget,
              f_eval, rng):
    agent.reset(env, moments, recovery)
    n_ctx = env.n_contexts
    consumed = np.zeros(n_ctx)
    allocated = np.zeros(n_ctx)
    spent, rounds, decide = 0.0, 0, 0.0
    smallest = float(min(env.action_values))

    while spent < budget - 1e-9:
        t0 = time.perf_counter()
        k, v = agent.act(env, rng)
        decide += time.perf_counter() - t0
        if spent + env.cost((k, v)) > budget + 1e-9:
            v = smallest
            if spent + env.cost((k, v)) > budget + 1e-9:
                break
        f = env.exposure((k, v), recovery)
        lo, hi = consumed[k], consumed[k] + f
        y = int(np.searchsorted(streams[k], hi) - np.searchsorted(streams[k], lo))
        consumed[k] = hi
        agent.observe(k, y, f)
        allocated[k] += env.cost((k, v))
        spent += env.cost((k, v))
        rounds += 1

    nlpd, dex, regret = evaluate(agent, n_ctx, f_eval, truths, heldout)
    return (nlpd, dex, regret, float((allocated > 0).mean()),
            1000.0 * decide / max(rounds, 1))


# --------------------------------------------------------------------------

def make_shape(seed):
    from domain import ShapeScarcity
    env = ShapeScarcity(n_contexts=12, regime="low", budget=8.0, seed=0)

    def draw(rng):
        moments, truths, rec = env.blocks(rng)
        return (moments, truths, rec, env.streams(truths, rng),
                env.held_out(truths, rng, n=N_HELDOUT), env.budget, env.eval_exposure)
    return env, draw


def make_photon(seed):
    from environment import PhotonCounting
    env = PhotonCounting()

    def draw(rng):
        moments, truths, rec = env.blocks(rng)
        f_eval = env.eval_time * env.exposure((0, 1.0))
        return (moments, truths, rec, env.stone_fields(truths, rec, rng),
                env.held_out(truths, rec, rng, n=N_HELDOUT), env.budget, f_eval)
    return env, draw


SETTINGS = {"shape": (make_shape, 60), "photon": (make_photon, 12)}


def study(which):
    make, n_env = SETTINGS[which]
    env, draw = make(SEED)
    built = agents.build_all(cross="models", include_bo=True)
    print("\n=== {} ===  {} contexts, {} episodes, {} agents"
          .format(which, env.n_contexts, n_env, len(built)))

    rows, per = [], {}
    t0 = time.time()
    for e in range(n_env):
        rng_env = np.random.default_rng(1000 + e)
        moments, truths, rec, streams, heldout, budget, f_eval = draw(rng_env)
        for agent in agents.build_all(cross="models", include_bo=True):
            out = run_agent(env, agent, moments, truths, rec, streams, heldout,
                            budget, f_eval, np.random.default_rng(5000 + e))
            per.setdefault(agent.name, []).append(out)
            rows.append((which, e, agent.name) + out)
        if (e + 1) % max(1, n_env // 4) == 0:
            print("    {}/{} episodes [{:.0f}s]".format(e + 1, n_env, time.time() - t0))

    ref = np.asarray(per["eig"], float)
    summary = []
    print("  {:<22} {:>8} {:>6} {:>8} {:>6} {:>8} {:>6} {:>7} {:>8}"
          .format("agent", "NLPD", "t", "dex", "t", "regret", "t", "seen%", "ms"))
    for name in sorted(per):
        a = np.asarray(per[name], float)
        ts = [paired_t(ref[:, c], a[:, c]) for c in (0, 1, 2)]
        summary.append((which, name, a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(),
                        a[:, 3].mean(), a[:, 4].mean(), ts[0], ts[1], ts[2],
                        a[:, 0].std(ddof=1) / np.sqrt(len(a)),
                        a[:, 1].std(ddof=1) / np.sqrt(len(a)),
                        a[:, 2].std(ddof=1) / np.sqrt(len(a))))
        print("  {:<22} {:>8.4f} {:>6.1f} {:>8.4f} {:>6.1f} {:>8.4f} {:>6.1f} "
              "{:>6.0f}% {:>8.2f}".format(name, a[:, 0].mean(), ts[0], a[:, 1].mean(),
                                          ts[1], a[:, 2].mean(), ts[2],
                                          100 * a[:, 3].mean(), a[:, 4].mean()))
    return rows, summary


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    names = ("shape", "photon") if which == "both" else (which,)
    rd = os.path.join(HERE, "results")
    os.makedirs(rd, exist_ok=True)
    all_rows, all_sum = [], []
    for nm in names:
        r, s = study(nm)
        all_rows += r
        all_sum += s
    save_csv(os.path.join(rd, "episodes.csv"),
             ["setting", "episode", "agent", "nlpd", "dex", "regret", "touched", "ms"],
             all_rows)
    save_csv(os.path.join(rd, "summary.csv"),
             ["setting", "agent", "nlpd", "dex", "regret", "touched", "ms",
              "t_nlpd", "t_dex", "t_regret", "sem_nlpd", "sem_dex", "sem_regret"],
             all_sum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
