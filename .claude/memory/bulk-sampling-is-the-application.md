---
name: bulk-sampling-is-the-application
description: The applied experiment is alluvial diamond bulk sampling, chosen because Sichel invented the GIGP for that problem and the exposure action and budget are physically real there.
metadata:
  type: project
---

Chosen 2026-08-08, replacing the placeholder subsections "Word frequencies" and "Photon
counting" that Wouter judged too generic.

**The setting.** A property is divided into blocks; block `k` has an unknown stone density
`lam_k` in stones per m^3. An action is a pair (block, sample volume), the exposure is
`f(u) = v * r_k` with `r_k` the plant's known recovery factor for that material, and the
count is the number of stones recovered. Implemented in
`experiments/experiment-bulk-sampling/`.

**Why this domain and not another.** Three things line up that do not line up elsewhere:

1. Sichel introduced the GIGP in the early 1970s specifically for diamondiferous deposits,
   because stones cluster and no standard discrete law reproduces observed stone-count
   frequencies. The model choice is the domain's, not ours, so a reviewer cannot ask "why
   this distribution".
2. The exposure action is physical and unambiguous: cubic metres of gravel put through the
   plant. Word frequencies and photon counting both have an exposure, but neither has a
   decision-maker who allocates it under a budget.
3. **The budget is naturally in cubic metres, not in number of samples.** This forces every
   criterion to be scored per unit of gravel. Counting a budget in rounds would hand the win
   to whichever policy asks for the biggest sample, which is a measurement artefact rather
   than a result.

**The narrative the domain buys.** Sichel built the law to *value* a deposit once samples
are in hand. The literature never used it to decide *which* samples to take. That is the
coverage claim of [[sichel-is-the-starting-point]] made concrete on a real problem rather
than asserted in the abstract.

**Not real data.** The study is a simulator whose structure is taken from the domain, not a
public dataset. Say so in the paper; do not let it read as a field trial.

**The result, 48 properties, 2026-08-08.** Our policy leads on the held-out predictive score
and on grade RMSE against every baseline. Maximum entropy sampling is the *worst* policy in
the study, behind random: it touches 2.8 of 12 blocks because count entropy grows with the
mean, so it locks onto the richest block. That is the aleatoric term behaving exactly as the
model section says.

**But on the top-4 ranking decision it is a tie**, paired t of -0.4 against D-optimality and
+0.3 against the systematic programme. A coarse ranking saturates: any policy that visits
every block recovers the order of the best four. Do not claim a decision-level win. The case
rests on prediction and grade estimation.

**How to apply:** keep the per-cost normalisation and the volume budget. They are what make
the comparison fair, and dropping either would flatter our policy. Keep the regret tie in
the paper; a reviewer who reruns this will find it.
