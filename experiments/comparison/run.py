"""Head-to-head on the two overdispersed settings: six agents, one harness, one clock.

    eig                          EIG, GIG-Poisson (ours): conjugate, two quadratures
    eig@gamma-poisson            EIG, gamma-Poisson: conjugate, negative binomial
    eig@logskewnormal-poisson    EIG by nested Monte Carlo, log-skew-normal prior, MCMC
    bo-ei@gp-poisson             BO, expected improvement, GP-Poisson with an RBF kernel
    bo-ei@gp-poisson-cat         BO, expected improvement, GP-Poisson, categorical kernel

Every agent runs through ``experiments/common/harness.py``: the same episodes (ground truth,
photon/stone streams and held-out counts from seed ``1000 + e``), the same budget
discipline, and wall-clock timing of design selection plus belief update and nothing else.

Episodes run in parallel, one process per episode, each process single-threaded so that
BLAS threads do not compete with each other and every agent within an episode is timed
under the same load.

Run:
    python experiments/comparison/run.py both --smoke        # 2 episodes, to results/smoke_*
    python experiments/comparison/run.py both --reps 50      # the study

Resumable: every (episode, agent) run is written to ``results/partial/<prefix><setting>/``
the moment it finishes, and rerunning the same command skips every run already there.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                       # noqa: E402
import csv                                                            # noqa: E402
import sys                                                            # noqa: E402
import time                                                           # noqa: E402

import numpy as np                                                    # noqa: E402
from scipy.special import gammaln, logsumexp                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "common"))

import agents                                                         # noqa: E402
from harness import TOP_M, run_episode                                # noqa: E402

SEED = 0
REFERENCE = "eig"
HEADER = (["setting", "agent", "episode", "spent", "rounds", "act_seconds",
           "observe_seconds", "nlpd"] + ["regret_top{}".format(m) for m in TOP_M]
          + ["rate_rmse", "rate_dex", "touched"])


#: Overrides of the gamma-ray environment: ``band`` (flux percentile band), ``n_sources``
#: and ``eval_time`` (the held-out integration NLPD is scored at). Empty keeps the defaults.
ENV_KW = {}


def make_env(setting):
    """Each study's own environment, imported by path so the two do not collide."""
    import importlib.util
    if setting == "bulk":
        path, cls = os.path.join(ROOT, "experiments", "bulk-sampling", "domain.py"), "BulkSampling"
        cps = (30.0, 60.0, 120.0, 180.0, 240.0)
    elif setting == "hardxray":
        sys.path.insert(0, os.path.join(ROOT, "experiments", "hard-xray"))
        path, cls = (os.path.join(ROOT, "experiments", "hard-xray", "environment.py"),
                     "HardXRayFollowup")
        cps = (15.0, 30.0, 60.0, 90.0, 120.0)
    else:
        sys.path.insert(0, os.path.join(ROOT, "experiments", "gamma-ray"))
        path, cls = os.path.join(ROOT, "experiments", "gamma-ray", "environment.py"), "PhotonCounting"
        cps = (15.0, 30.0, 60.0, 90.0, 120.0)
    spec = importlib.util.spec_from_file_location("env_" + setting, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kw = dict(ENV_KW) if setting == "gamma" else {}
    return getattr(mod, cls)(**kw), cps


def save_csv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  wrote {} ({} rows)".format(os.path.relpath(path, ROOT), len(rows)))


# --------------------------------------------------------------------------
# Correctness gate: every model's EIG against an estimate it does not share code with
# --------------------------------------------------------------------------

def _nmc(lam, lam_in, draw_y, loglik):
    y = draw_y(lam)
    ll_out = loglik(y, lam)
    ll_in = loglik(y[:, None], lam_in[None, :])
    return float(np.mean(ll_out - (logsumexp(ll_in, axis=1) - np.log(lam_in.size))))


def _poisson(f, rng):
    return (lambda lam: rng.poisson(f * lam).astype(float),
            lambda y, lam: -f * lam + y * np.log(f * lam) - gammaln(y + 1.0))


def _binned_lognormal(f, sigma, rng):
    from methods.countmodels import _discrete_lognormal_logpmf
    draw = (lambda lam: np.floor(np.exp(rng.normal(np.log(f * lam) - 0.5 * sigma ** 2,
                                                   sigma)) + 0.5))
    return draw, lambda y, lam: _discrete_lognormal_logpmf(y, f * lam, sigma)


def _grid_eig_logskewnormal(belief, f):
    """Exact EIG under a log-skew-normal posterior, by quadrature on a fine grid."""
    from scipy.stats import skewnorm
    xi, om, al = belief["prior"]
    s = np.linspace(xi - 12 * om, xi + 12 * om, 20001)
    lw = skewnorm.logpdf(s, al, xi, om) + belief["T"] * s - belief["F"] * np.exp(s)
    lw -= logsumexp(lw)
    keep = lw > lw.max() - 40.0
    s, lw = s[keep], lw[keep]
    mu = f * np.exp(s)
    m1 = np.exp(logsumexp(lw + np.log(mu)))
    m2 = np.exp(logsumexp(lw + 2.0 * np.log(mu)))
    y = np.arange(int(m1 + 20.0 * np.sqrt(m2) + 50.0) + 1, dtype=float)
    ll = -mu[None] + y[:, None] * np.log(mu)[None] - gammaln(y + 1.0)[:, None]
    lp = logsumexp(ll + lw[None], axis=1)
    return float(-np.sum(np.exp(lp) * lp) + np.sum(np.exp(lw)[None] * np.exp(ll) * ll))


def _grid_mean_logskewnormal(belief):
    from scipy.stats import skewnorm
    xi, om, al = belief["prior"]
    s = np.linspace(xi - 12 * om, xi + 12 * om, 20001)
    lw = skewnorm.logpdf(s, al, xi, om) + belief["T"] * s - belief["F"] * np.exp(s)
    return float(np.exp(logsumexp(lw + s) - logsumexp(lw)))


def gate(setting, env):
    """Closed forms against bias-corrected nested MC; the sampler's NMC against quadrature."""
    from methods.countmodels import GammaPoisson, GIGPoisson
    from methods.gammamixture import GammaMixturePoisson
    from methods.logskewnormal import LogSkewNormalPoisson, LogSkewNormalQuad
    rng = np.random.default_rng(SEED)
    moments = env.blocks(np.random.default_rng(1000))[0]
    rec = np.ones(env.n_contexts) if setting != "bulk" else env.blocks(
        np.random.default_rng(1000))[2]
    ctxs = (0, env.n_contexts // 2)
    vals = (float(env.action_values[0]), float(env.action_values[-1]))
    rows, bad = [], 0

    for model in (GIGPoisson(), GammaPoisson(), GammaMixturePoisson(), LogSkewNormalPoisson(),
                  LogSkewNormalQuad()):
        if hasattr(model, "set_rng"):
            model.set_rng(np.random.default_rng(SEED + 1))
        if hasattr(env, "initial_beliefs"):
            beliefs = env.initial_beliefs(model)
        else:
            beliefs = [model.prior_from_moments(*mv) for mv in moments]
        for k in ctxs:
            belief = beliefs[k]
            for v in vals:
                f = env.exposure((k, v), rec)
                exact_or_est = model.eig(belief, f)
                if model.name == "logskewnormal-quad":
                    ref = _grid_eig_logskewnormal(belief, f)
                    z = (exact_or_est - ref) / 1e-3
                    fail = abs(exact_or_est - ref) > 0.01
                    rows.append((setting, model.name, k, v, f, ref, exact_or_est, 1e-3, z,
                                 "quad-vs-grid"))
                elif model.name == "logskewnormal-poisson":
                    reps = [model.eig(belief, f) for _ in range(16)]
                    est, sem = float(np.mean(reps)), float(np.std(reps, ddof=1) / 4.0)
                    ref = _grid_eig_logskewnormal(belief, f)
                    z = (est - ref) / max(sem, 1e-12)
                    # Recorded, never failed: nested MC is biased up at O(1/M), and that
                    # bias is a property of the baseline, not a defect to be gated out.
                    rows.append((setting, model.name, k, v, f, ref, est, sem, z, "nmc-vs-grid"))
                    # What can fail is the sampler: after one observation at this exposure,
                    # the MCMC posterior mean against quadrature.
                    y = float(np.round(f * model.rate_mean(belief)))
                    post = model.update(belief, y, f)
                    lam = np.exp(post["s"])
                    ref_m = _grid_mean_logskewnormal(post)
                    sem_m = float(lam.std() / np.sqrt(lam.size))
                    z_m = (lam.mean() - ref_m) / max(sem_m, 1e-12)
                    fail = abs(z_m) > 6.0
                    rows.append((setting, model.name, k, v, f, ref_m, float(lam.mean()), sem_m,
                                 z_m, "mcmc-mean-vs-grid"))
                else:
                    if model.name == "lognormal-likelihood":
                        draw, ll = _binned_lognormal(f, belief["sigma"], rng)
                    else:
                        draw, ll = _poisson(f, rng)

                    def one(n):
                        return _nmc(model.sample_rate(belief, n, rng),
                                    model.sample_rate(belief, n, rng), draw, ll)
                    lo = [one(1500) for _ in range(4)]
                    hi = [one(6000) for _ in range(4)]
                    corr = (4.0 * np.mean(hi) - np.mean(lo)) / 3.0
                    sem = float(np.sqrt((16.0 * np.var(hi, ddof=1) / 4
                                         + np.var(lo, ddof=1) / 4) / 9.0))
                    z = (exact_or_est - corr) / max(sem, 1e-12)
                    # Significant *and* material: the 1/M extrapolation of the reference
                    # is itself off where the bias is not yet in its asymptotic regime.
                    fail = (abs(z) > 4.0 and abs(exact_or_est - corr) > 0.02) \
                        or exact_or_est < -1e-12
                    rows.append((setting, model.name, k, v, f, exact_or_est, corr, sem, z,
                                 "closed-vs-nmc"))
                bad += int(fail)
    print("Gate [{}]: {} of {} checks failed".format(setting, bad, len(rows)))
    for r in rows:
        print("  {:<24} ctx {:2d} u {:6.3g}  ref {:9.5f}  est {:9.5f}  sem {:8.1e}  z {:+6.2f}  {}"
              .format(r[1], r[2], r[3], r[5], r[6], r[7], r[8], r[9]))
    return bad == 0, rows


# --------------------------------------------------------------------------

def one_run(setting, e, name, env_kw, path):
    """One Monte Carlo run: one agent on one episode. Written to disk as soon as it ends.

    The file is written to a temporary name and renamed, so an interruption leaves either
    a complete result or none, never a truncated one; a restart skips every run whose file
    exists.
    """
    global ENV_KW
    ENV_KW = dict(env_kw)
    env, cps = make_env(setting)
    agent = [a for a in agents.build_comparison() if a.name == name][0]
    rows = [(setting, name, e) + tuple(r) for r in run_episode(env, agent, e, cps)]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    os.replace(tmp, path)
    return rows


def _load(path):
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r)
        return [(a, b, int(c)) + tuple(float(x) for x in rest) for a, b, c, *rest in r]


#: Scheduled first, so the longest runs do not start last and leave the pool idle.
SLOWEST_FIRST = ("eig", "eig@logskewnormal-poisson-m8k", "eig@logskewnormal-quad",
                 "eig@logskewnormal-poisson", "eig@gammamix-poisson")


def paired_t(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size < 2 or d.std(ddof=1) == 0.0:
        return 0.0
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def summarise(setting, rows, final):
    """Mean, standard error and paired t against the reference, at the full budget."""
    names = [a.name for a in agents.build_comparison()]
    col = {h: i for i, h in enumerate(HEADER)}
    fin = [r for r in rows if abs(r[col["spent"]] - final) < 1e-9]
    by = {n: sorted([r for r in fin if r[1] == n], key=lambda r: r[2]) for n in names}
    metrics = ["nlpd"] + ["regret_top{}".format(m) for m in TOP_M] + ["rate_dex"]
    out = []
    print("\n[{}] at the full budget, {} episodes; t is paired against {} "
          "(positive = better than {})".format(setting, len(by[REFERENCE]), REFERENCE,
                                                REFERENCE))
    print("  {:<28}".format("agent") + "".join("{:>18}".format(m) for m in metrics)
          + "{:>10}{:>10}{:>10}".format("ms/round", "ms/act", "ms/obs"))
    ref = by[REFERENCE]
    for n in names:
        rs = by[n]
        if not rs:
            continue
        rec = [setting, n, len(rs)]
        line = "  {:<28}".format(n)
        for m in metrics:
            x = np.array([r[col[m]] for r in rs])
            sem = x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0
            t = 0.0 if n == REFERENCE else paired_t([r[col[m]] for r in rs],
                                                    [r[col[m]] for r in ref])
            # Lower is better for every metric here, so flip the sign: positive t favours n.
            rec += [x.mean(), sem, -t]
            line += "{:>10.4f} {:>+6.1f} ".format(x.mean(), -t)
        rounds = np.array([max(r[col["rounds"]], 1) for r in rs], float)
        act = 1e3 * np.array([r[col["act_seconds"]] for r in rs]) / rounds
        obs = 1e3 * np.array([r[col["observe_seconds"]] for r in rs]) / rounds
        rec += [(act + obs).mean(), act.mean(), obs.mean(), rounds.mean()]
        line += "{:>10.2f}{:>10.2f}{:>10.2f}".format((act + obs).mean(), act.mean(), obs.mean())
        print(line)
        out.append(rec)
    header = ["setting", "agent", "episodes"]
    for m in metrics:
        header += [m, "sem_" + m, "t_" + m]
    header += ["ms_per_round", "ms_act", "ms_observe", "rounds"]
    return header, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", nargs="?", default="both", choices=("bulk", "gamma", "hardxray", "both"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="2 episodes, smoke_* outputs")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="gamma-ray flux percentile band (default: the environment's 5 95)")
    ap.add_argument("--sources", type=int, default=None,
                    help="gamma-ray: number of targets sharing the budget (default 48)")
    ap.add_argument("--eval-time", type=float, default=None,
                    help="gamma-ray: held-out integration NLPD is scored at, Ms (default 5)")
    ap.add_argument("--pop-size", type=int, default=None,
                    help="gamma-ray: fit each class prior to only this many known sources")
    ap.add_argument("--agents", nargs="+", default=None,
                    help="run only these agents (names as in agents.build_comparison)")
    ap.add_argument("--regate", action="store_true",
                    help="run the gate even if this setting has results on disk")
    args = ap.parse_args()
    global ENV_KW
    ENV_KW = {}
    if args.band is not None:
        ENV_KW["band"] = tuple(args.band)
    if args.sources is not None:
        ENV_KW["n_sources"] = args.sources
    if args.eval_time is not None:
        ENV_KW["eval_time"] = args.eval_time
    if args.pop_size is not None:
        ENV_KW["pop_size"] = args.pop_size
    reps = 2 if args.smoke else args.reps
    prefix = "smoke_" if args.smoke else ""
    if "band" in ENV_KW:
        prefix += "band{:g}-{:g}_".format(*ENV_KW["band"])
    if "n_sources" in ENV_KW:
        prefix += "n{}_".format(ENV_KW["n_sources"])
    if "eval_time" in ENV_KW:
        prefix += "eval{:g}_".format(ENV_KW["eval_time"])
    if "pop_size" in ENV_KW:
        prefix += "pop{}_".format(ENV_KW["pop_size"])
    rd = os.path.join(HERE, "results")
    os.makedirs(rd, exist_ok=True)

    from joblib import Parallel, delayed
    settings = ("bulk", "gamma") if args.setting == "both" else (args.setting,)
    for setting in settings:
        env, cps = make_env(setting)
        gate_path = os.path.join(rd, prefix + "gate_{}.csv".format(setting))
        part = os.path.join(rd, "partial", prefix + setting)
        os.makedirs(part, exist_ok=True)
        if os.path.exists(gate_path) and os.listdir(part) and not args.regate:
            print("Gate [{}]: passed in an earlier invocation, resuming".format(setting))
        else:
            ok, grows = gate(setting, env)
            save_csv(gate_path, ["setting", "model", "context", "action", "exposure",
                                 "reference", "estimate", "sem", "z", "check"], grows)
            if not ok:
                print("ABORT: gate failed for {}".format(setting))
                return 1

        names = [a.name for a in agents.build_comparison()]
        if args.agents:
            unknown = set(args.agents) - set(names)
            if unknown:
                print("unknown agents: {}".format(", ".join(sorted(unknown))))
                return 1
            names = [n for n in names if n in args.agents]
        order = [n for n in SLOWEST_FIRST if n in names] + [n for n in names
                                                            if n not in SLOWEST_FIRST]
        tasks = [(e, n, os.path.join(part, "e{:03d}_{}.csv".format(e, n.replace("@", "_"))))
                 for n in order for e in range(reps)]
        todo = [t for t in tasks if not os.path.exists(t[2])]
        print("\n[{}] {} episodes x {} agents: {} runs on disk, {} to do, {} workers".format(
            setting, reps, len(names), len(tasks) - len(todo), len(todo), args.jobs))
        t0 = time.time()
        if todo:
            Parallel(n_jobs=min(args.jobs, len(todo)), verbose=5)(
                delayed(one_run)(setting, e, n, ENV_KW, path) for e, n, path in todo)
        print("  [{:.0f} s]".format(time.time() - t0))
        rows = [r for _, _, path in tasks for r in _load(path)]
        save_csv(os.path.join(rd, prefix + "{}.csv".format(setting)), HEADER, rows)
        header, summ = summarise(setting, rows, cps[-1])
        save_csv(os.path.join(rd, prefix + "summary_{}.csv".format(setting)), header, summ)
    return 0


if __name__ == "__main__":
    sys.exit(main())
