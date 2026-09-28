# Neural spike counts as a rate field

From Taouali, Benvenuti, Wallisch, Chavane and Perrinet, *Testing the odds of inherent vs.
observed overdispersion in neural spike counts*, J Neurophysiol 115:434–444, 2016.

## Why this setting is a good fit

**The exposure is literally Assumption 1.** Their Eq. 1: a spike count in a window of length
`dt` is Poisson about `f = lambda dt`. Their `f` is our `f(u)`, and Corollary 1's additivity
makes trials and window length interchangeable — which is how a physiologist actually spends
the budget.

**The field's incumbent model is the gamma-Poisson.** They adopt the negative binomial *as a
Gamma-Poisson mixture* (Eq. 2), with `FF = 1 + mu/phi` (Eq. 4). The manuscript's pitch — the
conjugate model for counts is the gamma-Poisson pair, and what it lacks is a third parameter
— lands directly on current best practice here, which it does not in diamonds (where the
Sichel is already the domain's own model) or gamma-ray astronomy (where neither is used).

**Dispersion is *tuned*.** `phi` varies with stimulus direction, fitted by a von Mises centred
on the preferred direction (their Fig. 6C/D, Fig. 8B). Contexts therefore differ in the
*shape* of their count distribution at comparable dispersion — which is exactly the regime
`synthetic/shape-scarcity` had to construct by hand, and here it is a documented empirical
phenomenon. BIC picks the tuned model for 37 of their 40 MT cells.

**Scarcity is acute.** "mostly less than 20 trials per condition", and they close by asking
for more data with more trials. Recording time is a hard budget and preparations degrade.

**There is an explicit opening.** Their test rejects the negative binomial for **52% of LGN
and 41% of V1** cell/condition pairs. Those are cells a two-parameter mixture cannot fit.

## The obstacle, measured rather than argued

Their overdispersion is **doubly stochastic**: the rate is redrawn on every trial. Ours is
fixed per context, and Proposition 2 says the posterior concentrates on it. `run.py` study 3
generates counts both ways from the same tuning curves and runs the manuscript's estimator on
each.

A first version asked whether `F_t Var[lambda | D_t]` converges to the target — the constant
of Proposition 2. **It does, under both readings**, so it separates nothing: the posterior
variance is driven by the accumulated count, which grows at the same rate whether the rate was
redrawn or not. What the redraw changes is not the size of the posterior variance but whether
it is honest.

The diagnostic that works is the standardised error `z = (E[lambda|D] - target)/sd`, which has
variance 1 when calibrated:

| trials | rate fixed (ours) | rate redrawn (theirs) |
|---|---|---|
| 4 | 1.16 (95% cover) | 2.08 (87%) |
| 16 | 1.26 (90%) | 2.07 (85%) |
| 64 | 1.29 (90%) | 1.89 (88%) |
| 256 | 1.25 (92%) | 1.88 (85%) |

Mean Fano factor across the contexts used: **1.79**.

Two things to read off this. The overconfidence under the redraw tracks the Fano factor —
the posterior variance is too small by roughly the factor by which the counts are
overdispersed. And it **does not improve with trials**: 1.88 at 256 trials against 2.08 at 4.
It is a structural misspecification, not a small-sample artefact, and a 95% interval covers
85%.

That is the question the manuscript's Discussion flags as unresolved and load-bearing. It is
now quantified: adopting this setting means either modelling the trial-to-trial redraw, which
costs the conjugacy, or stating plainly that the posterior is overconfident by a factor of
the Fano factor.

## What is measured and what is generated

Kept strictly apart, because a rate field that silently fell back to simulated numbers is the
one mistake Section 4.5 cannot survive.

- `fields.lgn_spike_rates()` reads the **measured** LGN field, one observation per (cell,
  direction) pair. CRCNS requires a free registration so the archive is not in the repository
  and cannot be fetched automatically; the loader raises with instructions rather than
  substituting anything. Study 1 reports that it did not run.
- `taouali.py` is a **generator**, parameterised from their published figures, for study 3 and
  for any future allocation study. Fitting a mixing law to rates it produced would measure the
  parameterisation, not neurons, so study 1 never touches it.

To get the measured field: register at <https://crcns.org/register>, download `lgn-1`
(Scholl et al. 2013), unpack to `experiments/rate-fields/data/crcns-lgn-1/`, and adjust
`_lgn_tuning_from_raw` to the archive's layout — the field names there are a guess until the
files are in hand. V1 and MT are not shared, and MT is the data set with the strongest
overdispersion and the tuned-dispersion result.

## The generator, and how far it can be trusted

Anchors quoted exactly in the paper: at the preferred direction, mean `7.14` and inverse
dispersion `21.47`, giving `FF = 1.33` (their Fig. 8 caption). MT is 40 cells at 16
directions. Ranges come from Fig. 6 (mean 0–35, inverse dispersion 0–20) and Fig. 8 (both
0–25, FF up to ~2.8).

Not in the paper, and therefore ours: the per-cell tuning parameters, which are fitted but not
tabulated, and the spread of parameters across cells, shown only as box plots.

The parameterisation is falsifiable rather than asserted. Because both the mean and `phi` are
bell-shaped on direction and `FF = 1 + mu/phi`, the Fano profile must come out **bimodal** —
lowest at the preferred direction, peaking off it (their Fig. 8C). `fano_profile()` reproduces
this and `figures/neural.pdf` panel (a) plots it: `FF = 1.333` at the preferred direction
against their quoted `1.333`, peaking at `2.20` around ±70°, falling to `1.10` at 180°. Their
Fig. 8C peaks nearer ±90° at about 2.7, so the shape is right and the amplitude is
approximately right.

This check caught a real error: a first parameterisation gave both tunings the same
concentration, which makes `mu/phi` nearly constant and the Fano profile flat. The bimodality
exists only because dispersion is tuned *more sharply* than the mean.

## One structural limitation of the model in this setting

Eq. 6 of the manuscript gives `F(u) = 1 + f(u) V/E > 1` always, so the GIG-Poisson family
cannot represent **underdispersion**. This field cares: Taouali et al. cite sub-Poisson
variability (Kara et al. 2000) and note that the Fano factor falls when a stimulus is applied.
In LGN and V1, where their test finds significant overdispersion for only 19% and 26% of
pairs, a real share of the data may sit at or below `FF = 1`. `references.bib` already carries
an uncited refractory-Poisson paper on exactly this.

## Running

```bash
cd experiments/neural
python run.py        # study 1 skips without the archive; 2 and 3 run in ~3 s
python visualize.py
```

Reproducible from seed 0.
