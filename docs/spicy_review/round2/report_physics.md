# Referee report

**Manuscript:** "Predicting how a repetition penalty changes switching into and out of repetitive text, from unpenalized samples"
**Referee expertise:** statistical physics (nonequilibrium response theory, path ensembles and large deviations, Markov jump processes)

---

## 1. Summary in my own words

**Question.** Autoregressive sampling normalizes the distribution at every step. Can the effect of a logit-level repetition penalty on a coarse-grained "switching" statistic be predicted from unpenalized samples alone? The statistic is K, the ratio of the rate of entering a labeled "repetitive" state to the rate of leaving it.

**What was done.** The author writes down the discrete-time likelihood-ratio (Girsanov/Malliavin-weight) score of a penalized path, ∂ log P_h/∂h = (N − Q)/T. Here N counts chosen candidates and Q is its compensator, the sum of the per-step candidate probabilities q_t. From this score the author obtains ∂ ln K/∂h as a combination of covariances of four counts with N − Q. The derivative is used to choose a penalty h* that should lower a "crossover temperature" T_× (where K = 1) by 0.10. Then the author tests, under a preregistered rule, whether the realized change in ln K at h* matches the first-order prediction h*·g. Further analyses:

- they compare the step-wise response with the response under exponential tilting of whole paths by exp(hN/T), which is a Doob-transformed ensemble;
- they decompose the response by q_t and by entry versus exit;
- they apply the same approach to the multiplicative (CTRL / Hugging Face) penalty;
- they compare against an "equilibrium two-state" shortcut.

**Main result.** In one preregistered replication (120 new BBC articles, seven models of 0.4–1.7B parameters), the median over models of realized/predicted change in ln K was 1.04 (0.98–1.15), inside the registered band of 0.8–1.25. Two 7B models gave a mean of 0.96 (0.84–1.11). Because g came from the unpenalized arm of the same test, this is in substance a test of whether the response is still linear at h*. The step-wise response of ln K is 0.7–1.5% of the tilted-ensemble response (s_pen). The author reads this as roughly one over the relaxation time τ ≈ 90–146 steps. The multiplicative penalty is predicted only at divisors below those used in practice, and the intended shift of T_× was not achieved model by model.

## 2. Most interesting finding, and whether the abstract leads with it

Two findings stand out from a statistical-physics point of view.

1. **Linear response holds where path reweighting fails.** An intensive switching statistic responds linearly up to |h|/T ≈ 1–2, and approximately up to 2h* in the first test. At those same strengths, importance reweighting of whole paths is hopeless: Var(log w) ≈ 3–25.
2. **The step-wise response is about 100 times smaller than the tilted response.** This is the expected signature of a local (non-Doob) perturbation compared with a global tilt in a process with a slow mode.

Neither is in the first two sentences of the abstract. Those sentences ("A repetition penalty acts on a language model at every sampling step… We ask whether its effect… can be computed from samples generated without it.") state the setting and the question. The headline number (1.04) arrives in the sixth sentence, after a definition of the score. The 1/τ finding comes near the end. I would recommend opening with the finding itself, but see concern M1: the way the "strength" of the penalty is framed needs to change first.

## 3. Major concerns

### M1. "Reweighting had already failed" is not evidence that the penalty is strong in the sense that matters

Abstract: "The penalties were strong enough to shift that ratio, and at them reweighting the unpenalized texts by their exact probability ratios had already failed." Discussion: "Whole-text reweighting therefore cannot work at these penalties."

The variance of the log-weight, (h/T)² Σ_t q_t(1 − q_t), is extensive in path length. In the language of large deviations, importance sampling of whole trajectories fails at any fixed h once L is large enough. The validity of linear response for an intensive observable such as ln K is a different condition. It is controlled by h times a susceptibility per relaxation time, not by h²L. The manuscript half-recognizes this ("We attribute the contrast to this difference of scale, without having tested it further"). Yet the abstract uses the failure of reweighting as a calibration of how hard the test was. That is misleading.

A further point: ln K is a log-ratio of rates. In any Arrhenius-like or two-state description it is affine in a field coupled to a state-dependent quantity. Linearity of ln K over a wide range of h is therefore the generic expectation, not a surprise. The one bounded and directly observable quantity the author also tested, the fraction of repeats in the last 64 steps, gave 1.23 (1.11–1.38) with its upper end outside the band (F4). That is a hint that linearity is partly a property of the chosen observable.

**Requests.**
- (a) Remove the reweighting argument from the abstract, or reframe it as the extensive/intensive contrast it is.
- (b) Estimate the second-order term from unpenalized samples. The second derivative of log P_h is available in closed form: −Σ_t q_t(1−q_t)/T², plus the square of the score. This would show where linearity is expected to break, model by model.
- (c) Report Δ(2h*)/2Δ(h*) per model for the first test in the main text.

None of these changes the confirmatory decision. (b) would replace an indirect argument with a direct one.

### M2. The confirmatory test is a test of linearity, not of out-of-sample prediction, and the practically relevant version cannot tell the model's own derivative from a shared constant

The paper is candid: "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test." The score identity is exact, so a ratio near one with in-sample g says that (i) the code estimates the derivative correctly and (ii) the response is close to linear at h*. The title and abstract ("Predicting how a repetition penalty changes…", "computed from samples generated without it") suggest something stronger.

The design-only analysis (F4), which is what a user would actually do, is exploratory. In it:
- "A ḡ averaged over the other six models also gave a median inside the range" (1.09, 0.99–1.21);
- "the accuracy per model of the two could not be distinguished" (difference 0.04, −0.10 to 0.11);
- ḡ varies only from 1.86 to 2.81 across the smaller models.

As run, the experiment therefore does not show that the model-specific information in the score matters. A universal coefficient of about 2.4 in ln K per unit h does about as well. The direct finite-difference slope from h = ±0.3 did better still: |ln ρ| of 0.029 against 0.057–0.062 (F4).

**Request.** State this plainly in the abstract and discussion. This changes how the conclusion should be worded, not the numbers.

### M3. "Agreement holds for the median" conflates imprecision with disagreement; heterogeneity is not tested

"two of the seven per-model intervals of the replication … lay inside the range". Failing to lie inside a 0.45-wide band mostly reflects per-model imprecision. Only SmolLM2-1.7B (1.02–1.42) excludes 1. Separately, the bootstrap resamples articles but not models, so the interval of the median says nothing about a new model.

**Requests.**
- (a) A heterogeneity test across models on ln ρ, for example a Cochran-type Q using the bootstrap variances.
- (b) Wording that separates "imprecise" from "inconsistent".

Neither is likely to change the conclusion; both would make it accurate.

### M4. The s_pen ≈ 1/τ "reading" is tested with a criterion that has no power

Abstract: "What survives is roughly one part in the hundred or so steps over which the switching relaxes." Sec. IV C: "a coefficient of variation of s_pen τ below 0.3 across the smaller models would count as agreement with this reading. It was 0.14."

From Table II, the smaller models have τ = 90–122 steps (CV ≈ 0.10) and s_pen = 0.81–1.46% (CV ≈ 0.17). If s_pen were simply a constant near 1.2% with no relation to τ, the CV of s_pen τ would be about √(0.17² + 0.10²) ≈ 0.2. That is also below 0.3. The criterion could not have failed. F8 confirms that the ordering is absent: "The rank correlation between s_pen and 1/τ over the same models was only 0.11, so the reading fixes the size of s_pen, not its order across models." The only support for the 1/τ dependence comes from the two 7B models (0.68%, τ = 134–146).

There are further problems:
- τ = 1/(k_01 + k_10) is the relaxation time of a stationary two-state Markov chain. Appendix F2 shows that the labeled states are not such a chain: the exit-hazard ratio is 0.28 (0.23–0.34), and ln K drifts by 1.3–2.1 between halves.
- With L/τ ≈ 5, the texts are far from stationary.
- τ is partly set by the 32-token window and the hysteresis.

**Requests.**
- (a) Derive the two-state Doob-transform prediction explicitly. For a discrete-time two-state chain tilted by a per-step field ε on state occupation, the leading eigenvector gives δ ln K ≈ 2ετ to first order. Say what O(1) constant the reading actually predicts for s_pen τ.
- (b) Measure the lag-resolved kernels Cov(o_s, c_t) and Cov(o_s, c_t − q_t) as functions of s − t, where o_s is the per-step switching indicator. This would show directly that the tilted response accumulates over |s − t| ≲ τ while the step-wise response is local. It is the natural physics diagnostic and needs only the released per-step data.
- (c) Use an integrated autocorrelation time of c_t or of the state label in place of the Markov formula.
- (d) Compute s_pen and τ under the eight alternative labelings of F1. The paper says this was not done: "We did not estimate … s_pen under other labelings". Because τ depends on the window and s_pen need not, this is the cleanest test of whether s_pen τ ≈ 1 is a labeling artefact.

(b) through (d) could change the conclusion of Sec. IV C. Until they are done, the abstract sentence should be weakened to an order-of-magnitude statement.

### M5. "Explains why the effect is small": small relative to what?

The step-wise response is small only relative to exponential tilting by exp(hN/T), which no sampler implements and nobody would expect a logit penalty to produce. The natural naive expectation, the equilibrium two-state shortcut, is off by a factor of 0.69 and not 0.01 (Appendix H). In practical terms the penalty is not small: it moves T_× by 0.1 or more. The identity Cov(O,Q) = Cov(O, Σ_t ν_t) in Sec. III B is correct; I checked the Doob-martingale decomposition. But it is a rewriting, not an explanation.

**Request.** Rephrase. For example: "step-wise normalization removes the part of a path-level tilt that acts through later choices; for a slow switching mode this is ~99% of it." This is a wording change.

### M6. Internal contradictions and unexplained numbers

1. **Who checked the derivations.** Sec. V: "the author has since gone through the derivations of Sec. III (Appendix E)." Appendix E: "The derivations of Sec. III and Appendices G and H were likewise written by an AI assistant and checked by other AI instances, not by the author." These cannot both be true as written. Appendix E should be updated. The author should also confirm responsibility for code and derivations in line with journal policy.
2. **Linearity up to |h| ≈ 1.6.** F3 states "the change in ln K stayed close to linear in h up to |h| ≈ 1.6". The only model that reaches |h| ≈ 1.6 is GPT-2-XL (2h* = −1.58). For that model the 2h*−h* difference excluded zero: +0.15 (0.005–0.47), with ratio 1.17 → 1.32. The claim needs a qualifier.
3. **Agreement on the design runs.** Appendix C cites "medians of 1.00 and 0.96". Table F10 / Fig. S2b give 0.98 over 49 pairs. F4 gives 1.02 (0.95–1.09) for the measured slope. The reader cannot map these to definitions.
4. **The synthetic labeling bias.** Appendix C: "a synthetic check in which the labeling raised the ratio by about 10%". The score identity is exact for any h-independent statistic, including non-causal labels. So what mechanism raises the ratio: ratio-estimator bias of ln of count ratios, or finite-h curvature? If a ~10% upward bias is known, a band centered on 1 is miscentered. This matters for reading 1.04, the first test's 1.17, the BCa interval 1.04–1.47 at h*, and the pooled 1.09 (0.98–1.15). Please explain. This could change the interpretation, though not the decision.
5. **Overshoot of T_×.** Sec. IV B: "we found that the spread was already visible in the design data". F6 shows the design data implied 1.64 for GPT-2-XL against 2.41 realized, and 0.97 for BLOOM against 0.51 realized. The design data explain the direction for these two models but less than half of the magnitude. Please weaken the claim.
6. **Ratio range in the discussion.** Sec. V: "The first-order prediction held at penalties with |h*|/T* between 0.67 and 2.1." It held for the median, not per model (M3). Also, BLOOM's 0.37/0.56 = 0.66 from the tabulated values; a minor rounding point.
7. **"Several" models.** The abstract says the temperature target "missed its target in several models". By the interpolated analysis (Table F8), no model's interval lay inside the band: three were outside it and six undecided. "Was not confirmed for any model" is the accurate summary.

### M7. The headline number after an inconclusive first test

The paper commits honestly to "a single replication", and I commend this. Still, the total evidence at h* is the pooled 160-article estimate, 1.09 (0.98–1.15). Taken with the first test's BCa interval (1.04–1.47), it suggests the realized response may run 5–10% above first order. The abstract should report the pooled number next to 1.04, or at least mention it. The decision does not change; the point estimate the reader takes away does.

### M8. The equilibrium comparison is built on an arbitrary energy

Appendix H shows that the shortcut depends on the per-step energy:
- with −log p at T = 1, ρ_eq = 0.69;
- with −z, the sign is wrong for two models and ρ_eq = 2.4–3.3 for the others;
- with the third choice, 0.21–0.26.

The paper concludes, correctly, that "whether the two-state model overstates the effect … is a property of the chosen energy". Having said that, the comparison (Sec. V paragraph 3, F7, F9, Fig. S2, Table H1, κ, s_temp, g_C^tilt) supports little beyond one sentence. The detailed numbers suggest a quantitative conclusion that the paper itself says does not exist. Most of this should be cut; see Section 4 of this report.

### M9. Estimator design: an easy variance reduction is left out

g is estimated as Cov(e_i, σ) with σ summed over all 512 steps. For a count e = Σ_s o_s where o_s depends on x_≤s+15, every term with t > s + 15 has exactly zero covariance, because c_t − q_t is a martingale increment. Dropping those terms (the "reward-to-go" or causal truncation of Malliavin-weight estimators) gives an unbiased estimator with lower variance. Further truncation at lags beyond a few τ adds a bias of order e^(−W/τ). The paper reports that the score-based ḡ has a standard deviation of 0.21–0.32 per unit h against 0.11–0.13 for the finite-difference slope (F4). Causal truncation could close much of that gap and might make per-model decisions possible. This would not change the conclusions; it would strengthen the practical case.

## 4. What could be removed or moved, and which caveats the reader needs

**Could be removed or moved to an appendix without changing the conclusions:**
- **Appendix H and its satellites:** Sec. III C, Discussion paragraph 3, Table H1, κ, s_temp, b_tilt, g_C^tilt, the r_C records and Table F9, Fig. S2, and F9 (other genres, whose decision questions concern only the equilibrium model). Keep one sentence in the main text: the shortcut needs an energy per step that the sampler does not define, and with the most natural choice it overstates the response by about 45%.
- **"Penalized ln K at the design temperature"** (end of F6): a three-way decomposition with no decision. Remove.
- **F10, third part:** the "usable temperatures" analysis that was "undecided because no alternative rule could be tested". Remove.
- **Fig. 3b / the entry–exit split:** weak, and only partly predicted. Move to the appendix.
- **Robustness detail in Sec. IV B** (seeds, BCa, number of resamples): keep only the sentence that decisions were unchanged.
- **Instruction-tuned models:** one undecided test and one not run. A footnote suffices.
- **Most of the "why the shift missed" analysis:** keep one sentence in the main text.

**Caveats the reader really needs, in the main text:**
1. g came from the same runs, so the registered test is a test of linearity at h* (M2).
2. Agreement holds for a median over a fixed set of models, with intervals over articles only (M3).
3. The design-only prediction cannot be told apart from a shared coefficient (M2).
4. The multiplicative penalty is confirmed only below practical strengths.
5. The T_× target was not met per model.
6. One replication followed an undecided first test; pooled 1.09 (M7).
7. The states are a labeling of a non-Markov, drifting process, so K and τ are summaries.

Model revisions, the unrecorded EOS probability, and the exact grid rules belong in a limitations appendix.

## 5. Undefined terms, jargon, and passages that read as notes

- **"score"** appears in the abstract before it is defined; ML and physics readers attach different meanings to it.
- **"compensator"** is defined only implicitly. Say "the predictable part (compensator) of N in its Doob–Meyer decomposition" once.
- **"entropic and frenetic parts"** (Sec. V): undefined. Define them or drop the sentence.
- **"the identity"** is used as a noun for the first-order prediction ("the identity's share", "identity" in Fig. S2b, Tables F10 and F12). Use "first-order prediction".
- **"record", "rule fixed", "design runs" / "design articles" / "first test"**: idiosyncratic status words. "record" is used in F3 before Table C1 defines it. The first test used the design articles but new runs, which confuses the reader.
- **"the reading"** (Sec. IV C, F8): vague. State the hypothesis.
- **Δc and ΔE** appear in the Fig. S2 caption and in F7 before Appendix H defines them.
- **Symbol proliferation**: s_pen, s_temp, κ, r_C, ρ, ρ_eq, ρ_out, g_N, g_C, g_C^tilt, b_tilt, U_comp. Several disappear if M8 is followed.
- **"slightly broader set"** (Sec. II): candidates, defined by a 2-token context, are a much broader set than 4-gram repeats. In the ordinary state a candidate is chosen at about 20% of steps.
- **Passages that read as lists of results rather than an argument:** Appendix F (especially F4, F6, F10), the last paragraph of Sec. V, and Sec. IV B. The abstract is also a string of short declaratives.

## 6. Checklist

- **Large promise, small experiment.** Yes, in two places:
  - The title and abstract promise prediction of a penalty's effect. The confirmatory evidence is one replication of one statistic (ln K summed over temperatures), judged by a median over seven models, with g in-sample.
  - "The same score explains why the effect is small … one part in the hundred or so steps" rests on a criterion that could not fail (M4), over seven models whose τ spans only 90–122.
- **Single run, model or dataset.** Not a single model: nine models plus two instruction-tuned. But:
  - every prospective test uses one genre (BBC news, April–June 2025);
  - one text length (512);
  - four samples per article;
  - one labeling family for the registered decision;
  - base models only, no truncation;
  - one replication.
  The additive, multiplicative and 7B tests share the same 120 articles.
- **Parameter choices without a stated reason or time of choice:**
  - window of 32 tokens and thresholds 1/2 and 1/4 (when fixed?);
  - the 4-gram repeat definition against the 2-token candidate context (a reason is given; the time is not);
  - target shift of 0.10;
  - 64-token prompts, 4 samples per article;
  - grid spacing 0.05 and ±0.15;
  - the ten-switch usability rule;
  - q bins at 0.1 and 0.9;
  - hazard bins 49–96 and ≥97;
  - CV thresholds 0.3 and 0.6;
  - the 0.5–2% "about 1%" band;
  - the ±0.45 drift rule.
  The 512-token length ("fixed in the first experiments") and the 0.8–1.25 band have stated reasons.
- **Computational cost.** Reported in part (Appendix E): about 22 L4 GPU-hours for the replication and 6 A100 GPU-hours for the second stage of the 7B test. Missing: the design runs, the first test, the multiplicative and stronger-divisor runs, genres, instruction-tuned models, and a total. The claim that locating T_× is "comparable in cost to trying a few penalty strengths" should be backed by numbers.
- **Broad conclusion from a narrow case.**
  - "Whole-text reweighting therefore cannot work at these penalties" was tested only with four samples per article; the authors did not try pooled self-normalized weights.
  - The applicability to watermarking biases and min-p/top-p (Appendix G) is checked only in a five-token toy model. Also, at the finite |h|/T ≈ 1–2 used here, truncation sets would often change, so the fixed-set argument holds only infinitesimally.

## 7. Could the author defend each design choice? Where would they struggle?

The author can defend these well:
- summing realized and predicted changes before dividing (stated: noisy single temperatures);
- the band symmetric on a log scale;
- masking end-of-sequence tokens for fixed length;
- shared random numbers;
- using different events for the penalty and for the labels.

Where I expect difficulty:

1. **Why ln K?** Why not occupation or the repeat fraction, given that log-rate ratios are generically linear in fields (M1)? The bounded observable showed signs of nonlinearity.
2. **Why the median for seven models and the mean for two?** The two statistics have different robustness properties. Why is the band identical for both?
3. **Why centre the band on 1** when a synthetic check showed a ~10% upward bias, and what produces that bias given that the identity is exact (M6.4)?
4. **Why is exponential tilting by exp(hN/T) the right baseline** for "how small" the effect is (M5)? What constant does the two-state Doob argument predict for s_pen τ (M4)?
5. **Why is τ = 1/(k_01 + k_10) meaningful** after Appendix F2 shows the states are not Markov?
6. **Why 32 / 1/2 / 1/4**, and were they chosen before or after looking at the switching statistics?
7. **Derivations and code.** Appendix E says the author did not check the code line by line and, in one place, did not check the derivations. In a referee's or editor's conversation, the author would need to reproduce the Sec. III B decomposition and the Appendix H Clausius–Clapeyron step unaided. Both are correct as written: I checked ∂ log p/∂h = (c − q)/T, N − E[N] = (N − Q) + Σν, the −|z| logit derivative of the multiplicative penalty, b = mΔE, g_C = (b/ΔE)Δc/T, and dT_×/dh = T_× Δc/ΔE.

## 8. Minor issues

- **Abstract:** "For an additive penalty h on repeat candidates, the tokens that have previously followed the current last two tokens," — the appositive reads as if h were the tokens. Rephrase.
- **Sec. III B:** "For h > 0, exponential tilting also rewards…" — the first-order statement does not depend on the sign of h.
- **Sec. III A:** please state that the identity is the discrete-time analogue of the Girsanov formula for jump processes, and cite the standard Malliavin-weight literature at that point, not only in Sec. V.
- **Citations** mix arXiv identifiers and author–year. Please unify them.
- **Fig. 1c:** the caption does not say which grid each curve uses; the curves span different 1/T ranges. Also say that T_× sits at the top edge of each unpenalized grid, which is relevant to F6.
- **Fig. 3a:** specify that it shows the unpenalized arm.
- **F10:** "0.01 (−0.446 to 0.161)" has inconsistent precision. "all four in the same direction" — say which direction.
- **Table I:** GPT-2-XL is usually quoted as 1.5B parameters; 1.6 is acceptable as rounding of 1.56, but say so.
- **Table II caption:** τ is a median over temperatures while s_pen is a sum over temperatures. Explain the mixed aggregation.
- **F1:** shares of 75% + 25% plus one all-repetitive continuation for Pythia-1.4B; fine with rounding, but say so.
- **Sec. IV D:** the decomposition by q_t bins is a legitimate causal attribution, since applying the penalty only at steps whose predictable q_t falls in a bin gives exactly that partial covariance. The paper undersells this; say it.
- **Sec. II:** "temperatures between 0.19 and 0.90" — state that 0.19 is the bottom of Mistral's test grid and 0.20–0.90 the initial larger-model grid.
- **Appendix G:** "If no token lies on the boundary of S_t at h = 0" — say "almost surely".
- **Style:** "Prior to division by T" and "before division by T" are both used. Pick one.

## 9. Recommendation: major revision

**Strengths.**
- The theoretical core is correct and well chosen: the discrete-time Girsanov score with its compensator, and the Doob-martingale decomposition linking step-wise and tilted responses.
- The prospective protocol (dated preregistration, hashed predictions, one committed replication, an independent reimplementation that matched to four decimals) is exemplary.
- The reporting of negative and undecided outcomes is unusually honest: the T_× target was missed per model, the multiplicative penalty fails at practical strengths, and the states are not bimodal.
- The finding that an intensive switching statistic responds linearly at strengths where path reweighting fails is useful. So is the observation that step-wise normalization removes about 99% of the tilted response.

**Why not minor revision.**
- The abstract and discussion claim more than the evidence supports: "shows", "explains why the effect is small", "one part in the hundred or so steps", and reweighting failure used as a measure of strength (M1, M4, M5).
- The practically relevant prediction (design-only) is exploratory and cannot be told apart from a shared constant (M2).
- The 1/τ interpretation rests on a criterion without power and on a Markov timescale the paper itself rejects (M4).
- There are internal contradictions, including who checked the derivations, and an unexplained ~10% labeling bias (M6).

**Additions requested, and whether each would change the conclusions:**

| Request | Would it change the conclusions? |
|---|---|
| Reframe the abstract; report pooled 1.09; state the in-sample nature and the shared-constant result (M1, M2, M7) | Wording only. |
| Heterogeneity test across models (M3) | Unlikely; qualifies the claim. |
| Second-order term from unpenalized samples; per-model Δ(2h*)/2Δ(h*) (M1) | Unlikely; sharpens the claimed range of validity. |
| Two-state Doob derivation with an explicit constant, lag-resolved covariance kernels, integrated autocorrelation time, and s_pen and τ under the eight labelings (M4) | **Could change** the Sec. IV C conclusion and the abstract's 1/τ sentence. |
| Mechanism of the synthetic ~10% labeling bias (M6.4) | Could change the interpretation of ρ ≈ 1.04–1.09, not the decision. |
| Causally truncated score estimator (M9) | No; would tighten intervals and might allow per-model decisions. |
| Fix the Sec. V / Appendix E contradiction and the other inconsistencies (M6) | No, but required. |
| Condense Appendix H and the equilibrium comparisons (M8, Section 4) | No; improves focus. |