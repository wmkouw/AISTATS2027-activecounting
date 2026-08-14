---
name: project-lineage-poissongamma-efe
description: heavycount-efe descends from the poissongamma-efe paper; that repo holds the framework, code and the review history this project builds on.
metadata:
  type: project
---

This project (`heavycount-efe`, target AISTATS 2027) is a successor to
`/home/synd/Wouter/Onderzoek/Projecten/tue/poissongamma-efe`, the paper *A class of fast
active sensors from controllable exponential families* (Elsevier format, drafted through
2026-08-07). Started 2026-08-08 from the bare AISTATS 2026 starter pack; `main.tex` here is
still the unmodified template.

What the predecessor supplies:

- **The framework.** A conjugacy-preserving sensing action decomposes into a $\chi$-component
  (direction of the sufficient-statistic increment) and a $\nu$-component (size of the
  pseudo-count increment). Theorem 1: the predictive depends on the action through one scalar
  $d(u)$, so scoring a candidate set costs one univariate evaluation per candidate plus a
  family-level precomputation $c(\mathcal{F})$. Poisson-with-exposure is the worked
  $\nu$-action; negative binomial is the unbounded-discrete member.
- **Code to reuse.** `methods/cpv/` (one module per member, common interface in `base.py`),
  `methods/baselines/` (nested Monte Carlo, criteria), `experiments/exp*/` each with
  `run.py` + `visualize.py`, shared machinery in `experiments/common.py`. Every study runs a
  correctness gate ($\text{EIG} = H - A$ to machine precision, plus a bias-corrected nested-MC
  check) and aborts on failure. Fixed seeds; policies within a study share environments,
  noise stream and held-out set.
- **Its memories**, at `poissongamma-efe/.claude/memory/` — five of them, on the novelty
  boundary, the narrowed cost claim, the active-sensing vocabulary, and the title preference.
  They were written for that paper but the framing lessons carry over. Read them before
  pitching this one.
- **Two hostile reviews** in `poissongamma-efe/reviews/`.

`notes/sichel-gig-poisson.md` here is a trimmed copy of the note of the same name there: the
catalogue placements and paper-internal framing were stripped, the distribution facts, the
log-space Bessel hazard and the literature sweep kept. In the predecessor the Sichel member
was catalogued but never implemented or exercised numerically, which is the gap this project
starts from. See [[sichel-is-the-starting-point]].

**Why:** this repo has no git history and almost no content, so nothing in it reveals where
the ideas, the code or the settled framing decisions live.

**How to apply:** before deriving or implementing anything, check whether the predecessor
already has it. Do not restate its settled decisions as open questions.
