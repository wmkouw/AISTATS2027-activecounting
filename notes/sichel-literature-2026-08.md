# Sichel / GIG-Poisson literature sweep

Date: 2026-08-08. Supersedes the single-query sweep recorded in `sichel-gig-poisson.md`
(2026-08-07), which that note flagged as too weak to cite.

Eight queries, deliberately in different vocabularies, to avoid the failure mode that
killed the predecessor project's earlier direction: a mechanism owned under a vocabulary
nobody searched for.

## What the distribution is, and who owns it

**Origin.** Sichel introduced the generalised inverse Gaussian-Poisson (GIGP) mixture in
the early 1970s for the valuation of diamondiferous deposits, where diamonds occur in
clusters and no existing discrete law reproduced the observed stone-count frequencies. He
then applied the same model across an unusually wide range of data, from sentence lengths
and word frequencies to ore bodies.

**Canonical references.** Verified details in bold, the rest to be checked before use.

| Reference | Role |
|---|---|
| **Sichel (1975), *On a Distribution Law for Word Frequencies*, JASA 70(351a), 542-547** | the GIGP as a word-frequency law; twenty published distributions refitted |
| Sichel (early 1970s), statistical valuation of diamondiferous deposits, J. S. Afr. Inst. Min. Metall. | origin of the mixture; exact year and title unverified |
| Sichel (1982), repeat-buying / bibliometrics | consumer purchase behaviour and informetrics; unverified |
| Sichel (1997), species-abundance frequencies and species-individual functions via GIGP, S. Afr. Statist. J. | ecology; partially verified |
| **Stein, Zucchini & Juritz (1987), *Parameter Estimation for the Sichel Distribution and Its Multivariate Extension*, JASA 82(399), 938-944** | reparameterisation plus an ML algorithm; the standard estimation citation |
| **Rigby, Stasinopoulos & Akantziliotou (2008), *A framework for modelling overdispersed count data, including the Poisson-shifted generalized inverse Gaussian distribution*, CSDA 53(2), 381-393** | the modern regression framing; mean as an explicit parameter |
| **Rigby, Stasinopoulos, Heller & De Bastiani (2019), *Distributions for Modeling Location, Scale and Shape: Using GAMLSS in R*, CRC** | textbook treatment; `gamlss.dist` implements `SICHEL` and `SI` |
| **Willmot (1987), *The Poisson-inverse Gaussian distribution as an alternative to the negative binomial*, Scand. Actuarial J. 1987(3-4)** | the two-parameter sibling; the actuarial precedent for leaving the negative binomial |
| **Low, Ong et al. (2017), *Generalized Sichel Distribution and Associated Inference*, JSTA 16(3)** | four-parameter extension via the extended GIG |
| arXiv:2303.08139 (2023), *Limit Shape of the Generalized Inverse Gaussian-Poisson Distribution* | recent distributional theory |

**Application areas, all attested:** insurance claim counts and ratemaking, word
frequencies and linguistics, bibliometrics and informetrics, species abundance in ecology
and palaeontology, mining and ore valuation, protein abundance, consumer purchase
behaviour, market research, industrial psychology.

**How the literature justifies the third parameter.** Consistently as *skew and tail*, not
as dispersion: the Sichel is described as a long-tailed law suited to highly skewed,
overdispersed and zero-inflated counts, and is benchmarked against the negative binomial,
Poisson-inverse Gaussian, Delaporte and Poisson-Tweedie as fellow Poisson mixtures. This
matters because it is the correct motivation and the one the note of 2026-08-07 got wrong
(see below).

Worth knowing for the experiments: in an actuarial claim-frequency comparison the best fit
was negative binomial type II, with Sichel and Delaporte close behind. The Sichel is not
automatically the winner on real count data, so a claim that it is must be earned on the
data we choose rather than assumed.

## What is absent

**No hit connects the Sichel or the GIG-Poisson to active sensing, sequential or adaptive
experimental design, expected information gain, or Bayesian optimal experimental design.**
Queries were run in the count-modelling vocabulary, the actuarial vocabulary, the ecology
vocabulary and the design vocabulary, including one query pairing the distribution names
directly with sequential design and information gain. That query returned the two
literatures side by side with no intersection: Sichel papers on one side, generic EIG and
adaptive-design papers on the other.

A search for exposure allocation and adaptive design under mixed Poisson or actuarial
credibility, the place an exposure-allocation treatment was most likely to be hiding,
returned sequential sampling and treatment allocation only, nothing information-based for
a mixed Poisson.

**Nearest neighbour in the design literature.** Overstall (2022), *Properties of using
Fisher information gain for Bayesian design of experiments*, JSPI, arXiv:2003.07315:
an exponential-family design objective collapsing to a reduced-dimension space. It surfaced
again here, independently of the predecessor project, where it is already recorded as the
most important missing citation. Read it in full before any novelty claim is written.

## Strength of evidence

Better than the 2026-08-07 sweep, still not conclusive.

- Eight queries in four vocabularies, rather than one query.
- Still `WebSearch`, which is US-only and puts a summarising model between the query and
  the papers. A hit is strong evidence; a miss remains weaker evidence.
- Google Scholar and the actuarial journals were not swept directly, and citation chasing
  from Rigby et al. (2008) and Willmot (1987) has not been done.
- Several bibliographic details above are unverified, marked as such.

Enough to say *we found no prior treatment* with a stated method. Not enough for an
unqualified *there is none*.

## Correction this sweep forced

`sichel-gig-poisson.md` says a gamma-Poisson pair "fixes the relation between predictive
mean and variance", and that the GIG supplies free dispersion inside the class. **That is
wrong.** A negative binomial has two free parameters, so its mean and variance are both
free, and a reviewer will say so.

The correct statement, which is also the one the literature makes, is that the third
parameter buys the *tail* at matched mean and variance. Measured on our own implementation
at matched predictive mean and variance, the tail mass beyond `y = 50` differs from the
moment-matched negative binomial by a factor of 216 at `omega = 3`, falling through 1 and
down to 0.7 by `omega = 0.03`, and the mass at zero differs by up to 1.6.

`main.tex` §3.3, `experiments/experiment-gigpoisson/dgp.py` and its README now state it
this way. The numbers need their own experiment folder before they enter the paper text.

## Next

1. Add the verified references to Zotero, then cite them in `main.tex` §3.3, where two
   `% TODO` comments mark the places.
2. Verify the unverified rows, ideally from the papers rather than from search summaries.
3. Read Overstall (2022) in full.
4. Decide whether the coverage claim is worth making at all, given that it rests on an
   absence this sweep can support but not prove.
