# Which mixing law do real rate fields need?

Written 2026-08-11. `python compare.py` then `python visualize.py`. Data is cached, so it
reruns offline once fetched.

## What this is

The one experiment in the project that depends on no design choice: no acquisition, no
budget, no policy, no simulator. Three fields of rates measured in the world, three candidate
mixing laws, maximum likelihood, AIC.

Every other experiment either generates its rates from the GIG-Poisson law itself, which
makes the model correct by construction, or argues from a single field.

## The three fields

| field | n | one observation is | source |
|---|---|---|---|
| Fermi-LAT | 4557 | photon flux of a gamma-ray source | 4FGL catalogue via VizieR |
| Safecast | 1107 | count rate of a 500 m cell near Fukushima Daiichi | Safecast API, CC0 |
| 20 Newsgroups | 22460 | relative rate of a term in a corpus | sklearn |

The first two are domains where counting is the measurement process. The third is included
deliberately as an outsider, and it disagrees, which is the point.

## Result

| field | gamma | lognormal | GIG | fitted order | AIC/obs over gamma |
|---|---|---|---|---|---|
| Fermi-LAT | +1773 | +555 | **0** | **-1.43** | 0.39 |
| Safecast | +568 | +136 | **0** | **-2.07** | 0.51 |
| 20 Newsgroups | +7 | +1362 | **0** | **+1.73** | 0.00 |

**The fitted order is the diagnostic.** The gamma is the boundary of the family at
`omega -> 0` with a positive order. A field whose order fits negative is one no gamma can
reach however its two parameters are set; a field whose order fits positive sits inside the
gamma's range and the third parameter has nothing to do. Both physical fields are the first
kind; the linguistic one is the second.

That gives a cheap check to run **before** designing any sensing: fit the three laws to
whatever rates are already in hand and read off the order.

Two details worth keeping. The ordering of the two-parameter laws is unstable, with the
lognormal much the better on the physical fields and much the worse on the linguistic one, so
a practitioner committed to one two-parameter alternative has picked wrong somewhere. And the
per-observation advantage is nearly the same on the two physical fields (0.39, 0.51) despite
one being a satellite catalogue and the other Geiger readings from moving cars.

## Caveats, stated rather than hidden

**Safecast is crowd-sourced from vehicles.** Cells are sampled where roads run and when
drivers passed, over years during which the contamination decayed. A cell's rate is the mean
of its passes, not a controlled measurement; cells with fewer than ten passes are dropped for
that reason. This is weaker ground truth than a catalogue.

**Fermi is flux-limited** and its fluxes are estimates, so the field is the detected
population rather than the intrinsic one.

**Newsgroups excludes terms absent from any group**, because an observed zero there stands in
for an unknown positive rate and fitting it as zero would measure the truncation. That
restriction removes the rare terms, which are exactly where burstiness is strongest, so this
is a statement about common terms only.

Neither of the first two objections bears on a comparison between mixing laws on the same
numbers, which is all that is done here. All three would bear on claims about the underlying
populations.

## Files

| file | what it is |
|---|---|
| `fields.py` | the three loaders, with caching |
| `compare.py` | fits and AIC; writes `results/rate_field_fits.csv` |
| `visualize.py` | three panels into `figures/rate_fields.pdf`, included by the paper |
| `data/safecast_raw.npy` | 115k raw Safecast measurements, 60 km around Daiichi |
