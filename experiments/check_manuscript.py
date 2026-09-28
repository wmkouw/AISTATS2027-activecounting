"""Things that must still be true of main.tex after any compression pass."""
import io
import re

p = r"C:\Syndr\Wouter\Onderzoek\Projecten\tue\activecounting\AISTATS2027-activecounting\main.tex"
_raw = io.open(p, encoding="utf-8").read()
# Strip LaTeX comments before scanning: a reference inside a commented-out
# paragraph is not a dangling reference, it is a deleted one.
s = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in _raw.split("\n"))

LABELS = ["fig:prediction", "fig:bulk", "fig:scarcity", "fig:photon", "fig:rate_fields",
          "tab:prediction_slopes", "tab:bulk", "tab:photon",
          "tab:rate_fields", "tab:grid",
          "eq:likelihood", "eq:gig", "eq:gig_mean", "eq:predictive_mean",
          "eq:predictive_variance", "eq:fano", "eq:posterior", "eq:posterior_history",
          "eq:consistency", "eq:consistency_rate", "eq:sichel", "eq:eig", "eq:aleatoric",
          "eq:policy", "eq:ratio_expansion", "eq:variance_rate",
          "prop:update", "prop:consistency", "prop:predictive",
          "cor:sufficient", "rem:conditioning", "rem:dopt", "eq:dopt_limit",
          "as:exposure", "lem:bessel_bracket",
          "lem:lln", "appx:consistency", "appx:consistency_constant", "appx:grid",
          "sec:model", "sec:inference", "sec:inference_policy", "sec:experiments",
          "sec:experiments_verification", "sec:experiments_bulk",
          "sec:experiments_photon", "sec:experiments_shape", "sec:experiments_fields",
          "tab:shape_scarcity", "fig:shape_scarcity", "sec:discussion",
          "sec:conclusion", "sec:model_prior", "sec:model_dispersion"]

# Paths as they must appear in main.tex AFTER the experiments/ regrouping. Until the
# manuscript's \includegraphics are updated these will fail, which is the intended signal.
GRAPHICS = ["experiments/synthetic/verification/figures/prediction",
            "experiments/synthetic/shape-scarcity/figures/shape_scarcity",
            "experiments/bulk-sampling/figures/bulk",
            "experiments/bulk-sampling/figures/scarcity",
            "experiments/gamma-ray/figures/photon",
            "experiments/rate-fields/figures/rate_fields"]

INPUTS = ["experiments/tables/bulk", "experiments/tables/photon",
          "experiments/tables/rate_fields", "experiments/tables/grid",
          "experiments/tables/shape_scarcity", "experiments/tables/slopes"]

bad = []
for lab in LABELS:
    if ("\\label{%s}" % lab) not in s:
        bad.append("label %s" % lab)
for g in GRAPHICS:
    if g not in s:
        bad.append("graphic %s" % g)
for t in INPUTS:
    if ("\\input{%s}" % t) not in s:
        bad.append("input %s" % t)

# every \ref / \eqref must resolve to a \label that exists
defined = set(re.findall(r"\\label\{([^}]*)\}", s))
for m in re.finditer(r"\\(?:eq)?ref\{([^}]*)\}", s):
    if m.group(1) not in defined:
        bad.append("dangling ref %s" % m.group(1))

# balanced float environments
for env in ("figure", "figure*", "table", "table*"):
    o = len(re.findall(r"\\begin\{%s\}" % re.escape(env), s))
    c = len(re.findall(r"\\end\{%s\}" % re.escape(env), s))
    if o != c:
        bad.append("unbalanced %s: %d begin, %d end" % (env, o, c))

if bad:
    print("FAILED:")
    for b in sorted(set(bad)):
        print("   " + b)
else:
    print("all invariants hold (%d labels, %d graphics, %d table inputs)"
          % (len(LABELS), len(GRAPHICS), len(INPUTS)))
