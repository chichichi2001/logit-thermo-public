# Referee report

**Manuscript:** "Predicting how repetition penalties act in language models from unpenalized samples"
**Journal:** PRX Intelligence
**Referee expertise:** statistics and research methods (study design, bootstrap, multiple comparisons, preregistration, replication)

---

## 1. Summary in my own words

**Question.** When a repetition penalty is applied to an autoregressive sampler, can its effect on a text-level behavioural statistic be predicted from samples drawn without the penalty? Why is that effect much smaller than a "whole-text" (Boltzmann-like) view would suggest?

**What was done.** The author defines a two-state labeling of generated text (repetitive or ordinary, from the fraction of repeated 4-grams in a 32-token window with hysteresis). The summary statistic is ln K, the log ratio of the entry rate to the exit rate. For an additive penalty h on "repeat candidates" (tokens that previously followed the current last two tokens), the score of a text at h = 0 is (N − Q)/T. A likelihood-ratio (score-function) identity then gives the derivative g(T) = ∂ ln K/∂h from unpenalized samples. The author uses this derivative to choose a penalty h* that should lower the crossover temperature T_× (where K = 1) by 0.10. The ratio of the realized change in ln K to the first-order prediction h*·g(T) is then tested against a preregistered equivalence range of 0.8–1.25. The first test, on the 40 design articles, was undecided. A preregistered replication on 120 new articles in seven models of 0.4–1.7B parameters gave a median ratio of 1.04 (0.98–1.15). A test on two 7B models gave a mean of 0.96 (0.84–1.11), and a test of the multiplicative (CTRL/Hugging Face) penalty at the predicted divisor gave 1.01. Several exploratory analyses accompany these tests. The main ones are: (i) the step-wise response is about 1% of the response to exponential tilting of whole texts by exp(hN/T) (s_pen); (ii) the predicted response is concentrated at steps with 0.1 ≤ q_t ≤ 0.9; (iii) an "equilibrium two-state" shortcut overstates the effect by about 45% with one choice of energy.

**Main result.** For this labeling, this statistic, English news prompts and nine base models of up to 7.6B parameters, the first-order score-function prediction of Δ ln K stays within about ±25% for the median model at penalties strong enough to move T_× by about 0.1. The multiplicative penalty follows it only for divisors up to about 1.1.

I checked the derivations in Sec. III A–C and Appendix B (the score (c_t − q_t)/T, the martingale decomposition N − E[N] = (N − Q) + Σν_t, the temperature score, b_tilt = −T g_U and the log-weight in Appendix D), and they are correct. This matters because Appendix E states that the derivations were "checked by other AI instances, not by the author". I also recomputed every median, mean and derived ratio I could from the tables (Tables I, F4, F5, F6, F7, F8, F9, F10, F11 and H1). They agree with the text to rounding, with one small exception noted in Section 8. The preregistration record, the disclosure of the undecided first test, the independent reimplementation that reproduced the ratios to four decimals, and Table C1 are of a standard rarely seen in this literature. The author deserves credit for them.

---

## 2. The most interesting finding, and whether the abstract leads with it

In my view the most interesting result is a contrast that the abstract does not mention. Reweighting the four unpenalized texts of an article by their exact likelihood ratios collapses: the effective sample size is 1.3–2.1 of 4 (Appendix D, Discussion). Yet a first-order linear-response estimate of an *aggregate* switching statistic remains accurate at the same penalty (|h*|/T* = 0.7–2.1), and even, for the median model, at 2h* (Table F4: 1.10). In other words, the path measures already differ substantially while ln K is still nearly linear in h. That is a useful and somewhat surprising statement about when score-function prediction can be trusted. A close second is the s_pen ≈ 1% result. It is cleanly derived, but it is exploratory, has no decision rule, and depends on text length (Section 3.6 below).

The first two sentences of the abstract are:

> "A repetition penalty in a language model is applied, and the distribution renormalized, at every sampling step. We predict its effect on how generated text switches between repetitive and ordinary stretches, using only samples generated without the penalty, and show that this first-order prediction holds at penalties strong enough to change the switching."

The first sentence is background and states no finding. The second states the confirmatory result but not its size or its conditions. The ESS contrast appears only in the Discussion. The 99% cancellation appears as the seventh sentence. I recommend that the abstract open with the finding (the prediction holds within the registered ±25% for the median model while likelihood-ratio reweighting is already degenerate), give at most two or three numbers, and move the definitions of N, Q, T and of the candidate set out of the abstract.

---

## 3. Major concerns

### 3.1 The confirmatory test is in-sample for the derivative, and the abstract under-signals this

Sec. IV B states:

> "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test."

and the Discussion states:

> "Because the tests estimated the derivative from their own unpenalized runs, they test whether the first-order prediction holds at h*."

This is correct and honestly written, but it changes what was confirmed. g(T) is estimated from the same articles, the same random-number streams (common random numbers, Appendix A) and the same unpenalized arm that forms the baseline of Δ(T). The test therefore checks that (a) the score-function identity is unbiased, which is a mathematical fact, and (b) ln K is close to linear in h up to h*. It does not test what a practitioner would do: estimate the response on one set of prompts and apply the penalty to another. That is the analysis of Appendix F4 (ρ_out). It is exploratory, and its intervals "omit [the] uncertainty [of ḡ] over the design articles". The abstract's "With the derivative estimated from the unpenalized runs of the same test" is accurate. However, "We predict its effect ... using only samples generated without the penalty" will be read by most readers as an out-of-sample prediction.

**Request.** Say in the abstract and the Introduction, in plain words, that the registered tests evaluate linearity at h* with an in-sample derivative. Give F4 more weight, and recompute its intervals with ḡ resampled (bootstrap the design articles and the test articles jointly, or at least propagate the uncertainty of ḡ). *Would this change the conclusions?* It would not change the registered conclusion. It could change the exploratory claim in the Discussion that "a prediction from the design articles alone carried over to new articles". Once ḡ is resampled, the ρ_out intervals will widen considerably. ρ_out/ρ already ranges from 0.82 to 1.35 across models, so g itself moves by up to 35% between article sets (Qwen2.5-7B).

### 3.2 "Holds" refers to the median model; per-model agreement is much weaker, and the comparison with the equilibrium model is asymmetric

Of the seven per-model replication intervals in Table I, only three lie inside 0.8–1.25 (Pythia-1.4B, GPT-Neo-1.3B, Qwen2.5-1.5B). Four reach above 1.25, and SmolLM2-1.7B (1.02–1.42) excludes 1. Neither 7B model's interval lies inside the range (0.73–1.10 and 0.86–1.30). So by the paper's own rule applied per model, the identity is "confirmed" for 3 of 9 models.

Sec. IV B then judges the competitor per model:

> "The equilibrium two-state model with −log p_t as the energy, evaluated after the tests, would have fallen outside the range for six of the nine models (Appendix H)."

By Table F9, the equilibrium model's ρ_eq interval lies *inside* the range for two models (GPT-2-XL 0.91–1.06, Mistral 0.84–1.09). Judged per model, therefore, the equilibrium model is confirmed for 2 of 9 and the identity for 3 of 9. The correct contrast is at the level of the summary statistic: identity median 1.04 (0.98–1.15) against equilibrium median 0.69 (0.64–0.72). That contrast is decisive and should be the one stated. The two models should be judged on the same footing. The identity's intervals are wider mainly because g is a noisy score-function estimate (compare the narrow ρ_out intervals, in which ḡ is fixed). The paper should say so instead of letting interval width read as better calibration of the competitor.

There is also real between-model heterogeneity that the median hides. SmolLM2's interval excludes 1, and in F4 the ρ_out intervals of GPT-2-XL (1.22–1.37), SmolLM2 (1.07–1.19), BLOOM (0.78–0.94) and Qwen2.5-7B (1.10–1.29) all exclude 1. The Discussion concedes that "the bootstrap intervals resample articles, not models". The population about which "holds" is claimed is therefore these seven fixed models.

**Requests.** (i) Report per-model decisions under the registered rule next to the median decision, in Table I. (ii) Report a heterogeneity summary, for example a test of equal per-model log ratios using the bootstrap covariance, or a random-effects estimate with between-model variance. (iii) Rewrite the six-of-nine sentence so that both predictors are compared at the same level. *Would this change the conclusions?* It would not change the registered conclusion. It would change how "holds" must be worded: "for the median of seven models, to within 25%", not as a property of each model.

### 3.3 The ratio estimator and the percentile bootstrap, near a decision boundary

The test statistic is a ratio, ρ = Σ_u Δ(T_u)/Σ_u P(T_u), whose denominator is itself a noisy estimate (the score-function covariance from 4 samples × 40 or 120 articles). Two consequences follow.

* **Ratio bias.** E[1/P̂] > 1/E[P̂], so ρ is biased upward by roughly CV(P̂)². With 120 articles this is probably under 1%. With 40 articles, where per-model intervals span 0.62–1.98, it may be several percent. The per-model intervals are visibly right-skewed. In Table F4, five of seven upper ends exceed 1.25 at h* but only two lower ends fall below 0.8. In Table I, four upper ends exceed 1.25 and no lower end is below 0.8 (Qwen2.5-1.5B sits exactly at 0.80). The first test's "inconclusive" outcome (median 1.17, upper end 1.298) is therefore partly a property of the estimator. That matters, because the outcome triggered the replication.
* **Percentile intervals.** For a ratio with a noisy denominator, percentile bootstrap intervals can under-cover. A Fieller-type interval, a BCa interval, or a test built on the difference Δ − P scaled by its standard error would be better behaved.

**Requests.** Report the bootstrap bias (mean of the bootstrap ρ minus the point estimate) for each test, and report a BCa or Fieller interval as a sensitivity analysis. *Would this change the conclusions?* Probably not for the replication. It could change the first test from "undecided" to "confirmed", which would alter the narrative of an inconclusive test followed by a replication. The paper should say so whichever way it goes.

### 3.4 Monte Carlo error of 1,000 resamples, at decisions settled by the third decimal

Several decisions are settled by margins smaller than the Monte Carlo error of the bootstrap endpoints:

> "The upper ends, 1.298 and 1.299, lay just above 1.25." (F3)
> "The upper end in the first half, 0.8003, just failed the rule fixed beforehand" (F7)
> "The median over the smaller models was 0.01 (−0.446 to 0.161), just inside ±0.45" (F10)

Table F2's first row shows that a different seed moves the replication interval from 0.98–1.15 to 0.99–1.14. The Monte Carlo error is therefore about 0.01, larger than the margins above. With B = 1,000, a 2.5% percentile endpoint is itself uncertain by roughly ±0.5 percentage points of coverage.

**Request.** Rerun the bootstrap with B ≥ 10,000 for every decision, and report the Monte Carlo standard error of each endpoint. Any decision that depends on the seed should be labeled "on the boundary", not decided. *Would this change the conclusions?* Not for the replication (margins of 0.18 and 0.10). It could change the F7 and F10 decisions, and Sec. IV B relies on the F10 decision (see 3.5).

### 3.5 Overstated claim: "the ratio did not drift systematically"

Sec. IV B states:

> "and across temperatures the ratio did not drift systematically (Appendix F10)."

The supporting interval in F10 is −0.446 to 0.161 for a change measured across the usable temperatures. Its width, about 0.6, exceeds the whole decision range (0.45 on the ratio scale), and its lower end is 0.004 from the margin. For the larger models it is −1.41 to 0.29. This analysis has essentially no power. What can be said is that drift as large as the range width was not detected, and drift of nearly that size is not excluded. The same appendix reports that "the half-width of the interval of the difference between realized and predicted change had a median of 39% of the prediction" at single temperatures. Per-temperature agreement is therefore simply not measured well.

**Request.** Replace the sentence with an accurate statement of power. *Would this change the conclusions?* It removes a supporting claim but not the registered result.

### 3.6 The headline "about 99%" is exploratory, has no decision rule, and depends on length

The abstract states as a finding:

> "In our runs, step-wise normalization therefore cancels about 99% of the first-order response the same penalty would produce on whole texts."

Table C1 classifies Table II as "Exploratory, record" (no decision rule). F8 shows that s_pen depends on the text length (1.16% at 512 tokens, 1.38% at 384, 1.79% at 256) and on the covariance convention (0.90% when fully pooled). The rule that labels all of these "about 1%" accepts any interval inside 0.5–2%, a factor of four. Such a rule cannot fail at the scale of interest and is not informative. The author also notes that "length and position are confounded".

There is a substantive question as well. The denominator g_N is the response to tilting by exp(hN/T), a whole-text reweighting that no sampler implements. Because N is dominated by occupancy of the repetitive state (candidate rates of 92–94% against 16–21%, F1), Var(N) across texts is very large. Most of s_pen ≈ 1% may simply restate Var(N − Q) ≪ Var(N), that is, that N measures the state. The paper should explain why this denominator is the right yardstick. The Introduction says it "explains why the effect is small", but small relative to what? Appendix H shows that the naive equilibrium model, which uses the same kind of step-wise responses, is off by only 45%, not by a factor of 100. The 99% cancellation therefore does not translate into a practical error of any approach a reader might use.

**Requests.** Label the 99% figure as exploratory and length-specific wherever it appears. Give intervals per model in Table II. State the practical consequence, or drop the claim from the abstract. *Would this change the conclusions?* It would change the abstract and Sec. IV C framing, not the registered results.

### 3.7 Contradiction about the preregistration status of the design runs

Sec. IV B states:

> "These documents did not cover Fig. S1, whose runs were made earlier."

Appendix C (Design choices) states that the decision range was set "from the agreement already seen on the design runs at h = ±0.3, with medians of 1.00 and 0.96".

Yet Table C1 lists "Shift of T_× on the design runs (Fig. S2a) | F7 | Preregistered", and F7 says "One comparison was preregistered" for these runs. The numerator of R in Fig. S2a is the slope dT_×/dh, which is exactly what Fig. S1 plots, from the same runs. Table C1 defines preregistered as "committed in a dated document before the data were generated or opened". Fig. S2a cannot be preregistered under that definition if Fig. S1 had already been seen, unless the equilibrium comparison was registered at an earlier stage than the tests.

**Request.** State the date order of (a) the design runs, (b) the registration of the R comparison, (c) the inspection of Fig. S1 and of the identity's agreement on those runs, and (d) the setting of the 0.8–1.25 range. If the order is ambiguous, reclassify Fig. S2a as exploratory. *Would this change the conclusions?* It would change only the status of the evidence against the equilibrium model, which is secondary.

### 3.8 Evidence that the registrations were not altered later

The registrations were "committed ... to a version-controlled repository", with hashes recorded "before penalized generation". Git commit dates can be set arbitrarily and history can be rewritten. Hashes recorded inside the same repository prove only internal consistency, not timing.

**Request.** Provide third-party timestamps for each registration (for example, push events on a public host, an OSF or AsPredicted registration, or hashes posted publicly before the runs). Archiving as version 1.0 on Zenodo after the fact does not provide this. *Would this change the conclusions?* No, if the evidence exists. It would considerably strengthen the claim of preregistration.

### 3.9 Sequential testing, and the missing pooled estimate

The registered procedure was: test once, and if undecided, replicate once. Equivalence tests are about falsely declaring agreement, and two chances at a decision inflate that error (to at most about twice the per-test 2.5% per side). The author fixed "no further replication" in advance, which limits the damage. However, the paper reports the replication as the result and the first test as "inconclusive". The first test's point estimate (1.17) lies well above the replication's (1.04).

**Requests.** (i) State the operating characteristic of the two-stage procedure. (ii) Report a pooled or meta-analytic estimate of the first test and the replication as a sensitivity analysis, with the caveat that the 40 articles also determined h*. (iii) Answer this question: in the first test, were the unpenalized-arm continuations at temperatures shared with the design grid bit-identical to the design runs that produced ḡ and h*? Seeds "depended on the temperature, the sample and the batch, but not on the penalty", so I expect they were. If so, the first test was not independent of the design, and its regression toward ḡ is a selection effect. *Would this change the conclusions?* A pooled estimate near 1.07 would still lie inside the range. It would give readers the honest summary of all the confirmatory evidence.

### 3.10 The quantity the design targeted (T_×) was not controlled, model by model

The abstract states:

> "The penalties were chosen to shift the temperature at which the entry and exit rates are equal, and that temperature did not always move as intended, although the change in the log ratio was predicted."

"Did not always" understates Table F8. With the interpolated crossing (the most favourable of the three estimators, chosen after the tests), *none* of the nine per-model intervals lies inside 0.8–1.25. Three lie entirely outside: SmolLM2 1.43–1.81, GPT-2-XL 2.19–2.79, BLOOM 0.39–0.63. The per-model ratios range from 0.51 to 2.41. The straight-line estimator, which the design itself used, gives a median of 1.32 (undecided) and a mean of 2.17 for the larger models ("larger than intended"). The three estimators give medians of 1.32, 0.95 and 0.87 for the same data, a spread equal to the whole decision range. This shows that T_× is fragile. The main text's

> "The median then became 0.87, slightly below the target, so the overshoot came from the curvature (Appendix F6)."

infers a cause from a change of estimator. That is a post hoc explanation, not a demonstration.

Fig. S1 and Table F10, which existed before the tests, may already have signaled the GPT-2-XL problem. Reading the slopes from Table F10 (GPT-2-XL, about 0.21 per unit h) together with h* = −0.79 implies a shift of about 0.17, that is, an overshoot of about 1.65× before any test. On the same design runs the identity also underpredicted GPT-2-XL's response of ln K by a median factor of 1.60 (Table F10, "Identity" column). Yet in the replication GPT-2-XL's ρ was 1.08. These GPT-2-XL numbers do not agree with one another, and the paper says only that the departure is "for reasons we have not identified".

**Requests.** Rewrite the abstract sentence to give the per-model picture. Present the interpolated analysis as one of three estimators, not as an explanation. Discuss the GPT-2-XL inconsistency between the design runs, F4 and the replication. *Would this change the conclusions?* It changes the practical claim. The method predicts Δ ln K but cannot be used to place T_× for an individual model within ±25%. Sec. IV B also says T_× is what "a user would target". That claim is itself doubtful and should be justified or dropped.

### 3.11 The equivalence margin is justified by convenience, not by practical relevance

> "The range 0.8–1.25 is symmetric on a logarithmic scale. It was set before the first test, from the agreement already seen on the design runs at h = ±0.3 ... and from a synthetic check in which the labeling raised the ratio by about 10%."

An equivalence margin should express what error would matter. Here it expresses what the author expected to achieve. Please state what a 25% error in Δ ln K means for the quantity a user cares about. From the design, it corresponds to roughly ±0.025 in T_×. Is that small? The sample-size reasoning ("We expected 120 articles to narrow it by about 1/√3, to about 0.2") is fine as a precision target. But with a half-width of 0.1, confirmation required a true ratio in roughly 0.9–1.15. Given the first test's 1.17, the a priori probability of confirmation was not high, and the paper should say so. *Would this change the conclusions?* No. It would let readers judge how strong "confirmed" is.

### 3.12 EOS masking may drive the phenomenon studied

> "End-of-sequence tokens, the <|endoftext|> token and unused padding entries of the vocabulary were excluded, so that every continuation had the same length."

A base model forced to continue a news article for 512 tokens past its natural end may loop for that reason. The drift of ln K along the text (F2: ln K rises by 1.3–2.1 from the first half to the second) is consistent with this. The identity remains valid for the masked sampler, so the confirmatory result is unaffected. However, the relevance to real sampling, where EOS is allowed, is not established.

**Request.** Report the mean EOS probability that was masked, by position. Repeat at least the replication's unpenalized arm with EOS allowed and a bounded stopping time (the paper notes that the identity covers this case). *Would this change the conclusions?* Possibly the descriptive ones (the switching statistics and T_×). Not the identity.

---

## 4. What could be removed or moved, and which caveats readers need

The paper is long and diffuse for its central result. Much of the length is caveat and exploratory detail that does not bear on the conclusions.

**Could be cut or moved to the supplement without changing the conclusions:**

* **The equilibrium two-state model** (Sec. III D, the six-of-nine sentence in IV B, the ρ_eq material, r_C, Fig. S2, Table F9, Table H1, most of Appendix H). It is a foil that the paper itself shows to be arbitrary ("Whether the two-state model overstates the effect of a penalty is therefore a property of the chosen energy, not of the sampler"). Its comparison was not registered with a decision rule. One paragraph and one appendix would suffice.
* "For comparison, the tilted response of Sec. IV C would give a ratio of about 0.01." This compares against a predictor nobody would use; delete it.
* The remark on bounded stopping times in Sec. III A ("The identity also holds with a bounded stopping time ... but we did not use one"), unless the EOS analysis of 3.12 is added.
* Appendix G (truncated sampling). It is a five-token toy, and "Nothing in this appendix has been tested with language models." One sentence in the Discussion is enough.
* F6's three-part decomposition of the penalized ln K at T*, the quadratic-fit "record", and the F10 analysis of usable temperatures, which by construction could not decide anything ("no alternative rule could be tested").
* The entry-ratio column of Table F3, the long inline list of per-model candidate rates in F1, and the Middle/side column of Table F1.
* Sec. IV D's paragraph on the entry and exit split (Fig. 3b) could go to an appendix. It is exploratory and does not affect any conclusion.
* "They answer three questions raised in review" (F10) refers to an earlier review process and should be removed.

**Caveats a reader really needs, in the main text:**

1. The derivative in the registered tests comes from the same runs (3.1).
2. "Holds" refers to the median of seven fixed models; per-model deviations of 10–30% occur (3.2).
3. The additive penalty "is not an option in common libraries", and the multiplicative penalty follows the prediction only for θ ≤ about 1.1, below the 1.1–1.3 used in practice.
4. T_× is not controlled per model (3.10).
5. The states are a labeling, K averages over a strong positional drift, and EOS was masked (3.12).
6. English news, models up to 7.6B, one length, no truncation.

Most of the other caveats (revision hashes, the import statement fixed after the freeze, bit-identical GPU repeats) belong in the supplement.

---

## 5. Undefined terms, jargon, and passages that read as notes

* **Undefined or under-defined for this readership:** "compensator" (defined only by example), "score", "static susceptibility of the path ensemble", "tilted ensemble", "Doob transform", "escape-rate term in response theory", "van 't Hoff plot", "Clausius–Clapeyron", "Boltzmann form", "label bias", and the symbols κ, r_C, ρ_eq, ρ_out, s_pen, s_temp, g_N, g_U, b_tilt. Most readers of PRX Intelligence will not know the physics terms. Each should be defined in one plain sentence or removed.
* **Project-internal terms:** "usable temperatures", "design temperature", "design articles", "record" and "rule fixed" (Table C1), and "the identity" used as a noun for a predictor. These are workable but should be defined once and used consistently. "Record" in particular is opaque.
* The Introduction's "which explains why the effect is small" names no comparison. Small relative to what?
* **Passages that read as notes or lists:** the limitations paragraph of the Discussion ("Because the tests estimated ... The confirmation rests on ... Decisions concerned ... All prospective tests used ... The widely used presence and frequency penalties ... The two states are ... The change in ln K was predicted ...") is a sequence of disconnected sentences. Appendix F10 ("First ... Second ... Third ...") and Appendix F1 also read as lab notes. Table C1 is useful but reads as a ledger; a short paragraph explaining how to read it would help.
* Sec. III C is very compressed. The weighting used to define s_temp ("we sum numerator and denominator with the weights of the equilibrium prediction g_C of Sec. III D") makes s_temp depend on the equilibrium model, which is introduced only afterwards.
* Appendix B says the choice of candidates "keeps the events that define the states distinct from the events the penalty acts on". Appendix C, however, says "Every repeat is also a chosen candidate". The events are nested, not distinct. The real point is that the state definition does not depend on h.

---

## 6. Common weaknesses: present or not

* **Large promise, small experiment.** Partly. The title "Predicting how repetition penalties act in language models" promises more than was tested. Only one statistic (ln K under one family of labelings) was tested, the main penalty is not used in practice, and the practical penalty works only below practical strengths.
* **Single run, model or dataset.** Each test was run once, with one dataset (BBC News, April–June 2025, 120 articles) for all confirmatory tests. The additive, multiplicative and 7B tests share the same 120 articles, and the multiplicative tests share the replication's unpenalized arm, so they are not independent confirmations. The Discussion acknowledges the shared articles. The "larger models" claim rests on the mean of two models, neither of whose intervals lies inside the range. The other genres (F9) tested only the equilibrium questions, not the identity's ratio with a decision rule, so the Discussion's pointer to "runs on novels, scientific papers and code" should not be read as a generalization of the main result. For code, T_× was extrapolated to 1.2–1.7, far outside the sampled range, so the claim that "T_× rose beyond it in 11 of the 12 model–genre pairs" partly rests on extrapolation.
* **Parameter choices without a stated reason or time of choice.** Reasons are given for the candidate context of two tokens, the 512-token length ("fixed in the first experiments"), the median versus the mean, the range, and the hazard bins. No reason or time of choice is given for: 4-gram repeats; the 32-token window and the 1/2 and 1/4 thresholds (a sensitivity analysis exists, but not when these were chosen); 64-token prompts; **four samples per article** (the most consequential choice, since it governs within-article covariances and the ESS diagnostic); the target shift of 0.10; seven temperatures spaced by 0.05 with the unpenalized crossing placed near the grid's upper edge; B = 1,000 resamples; the cap θ ≤ 3; the window 0.3–0.8 for prediction; and 35 code files against 40 for the other genres.
* **Compute.** Partly reported. The replication (about 22 L4 GPU-hours) and the second stage of the 7B test (about 6 A100 GPU-hours) are given. The design runs, the first test, the multiplicative tests at four strengths, the genres, the instruction-tuned models and the total are not. Please give a total, and give the cost of the unpenalized runs that the prediction needs as a fraction of a direct search over penalties. The Discussion says only that it is "comparable".
* **Broad conclusion from a narrow case.** Yes, in the abstract: "The derivative also shows that the response arises mainly at the minority of steps where the model is uncertain whether to choose a repeat candidate." This rests on a decomposition of the *predicted* response, for one statistic, in exploratory analysis. Likewise, the Discussion's statement that the derivative "applies to any small shift added to the logits of a set of tokens, such as logit biases and watermarking biases" is true of the identity but untested for prediction at finite strength. The paper states this fairly ("We tested only the two repetition penalties").

---

## 7. Could the author defend each design choice? Where would they struggle?

The author could defend the following well: the score-function identity, the use of common random numbers, resampling articles rather than samples, the median over models ("so that no single model could decide it"), and the choice to fix usable temperatures at the point estimate.

The author would most likely struggle with these questions:

1. **"Why test with g from the same runs, rather than with the design-time prediction a user would actually make?"** The honest answer, that it tests linearity, is in the Discussion, but the abstract does not reflect it.
2. **"Why a ratio of two noisy sums, with a percentile bootstrap, when the decisions fall near the boundary?"** The rationale in Appendix C (single temperatures are noisy, so take a ratio of sums) addresses noise in the numerator, not bias from the denominator.
3. **"Why ±25%? What error in Δ ln K matters to anyone?"** The stated basis is the agreement already seen, not practical significance.
4. **"Why replicate rather than pool, and what is the combined evidence?"**
5. **"Why is exponential tilting the right denominator for s_pen, and what follows practically from 99%?"**
6. **"Why make T_× the design target, given that it depends on labeling, length and estimator, and that you yourself compare it 'only within a fixed labeling and length'?"**
7. **"Why four samples per article?"**
8. **"Which T_× estimate did you use where?"** At least four are used: the design fit, the test-grid straight line, interpolation and the quadratic fit. Table F1 picks temperatures "closest to T_× estimated from the unpenalized runs of the same test". For Pythia-1.4B this gives 0.62, although Table I's T_× is 0.57 and the interpolated crossing is about 0.577. This implies a third, unreported value of at least 0.595.

Because Appendix E states that the author "did not check [the code] line by line" and that the derivations were not checked by the author, I would also ask the author to be ready to explain the identities and the estimator without assistance. They are correct, but an editor will expect the author to own them.

---

## 8. Minor issues

* Sec. IV B: "With ln θ set to twice its predicted value the ratio was still 1.00, but at eight times it fell below the range." This skips k = 4 (0.86, 0.77–0.99, undecided), which Table F7 shows to be the onset of the failure. Please mention it.
* F3: "Doubling the penalty barely changed the ratio, so the change in ln K stayed close to linear in h up to |h| ≈ 1.6." The value |h| ≈ 1.6 is reached only by GPT-2-XL, which is the model whose ratio grew the most (1.17 to 1.32, 1.00–2.00). Please report the paired difference between the ratios at h* and 2h*, with its interval, rather than "barely changed".
* Sec. IV B: "the mean of the two larger models still departed from the target". Its interpolated interval (1.14–1.53) includes 1.25, so the rule would call it undecided; "departed" is too strong.
* Appendix H: "differs from ρ_eq by up to 0.07, for GPT-2-XL". From the table values, 1.08 × 0.98 − 0.98 = 0.078, so "about 0.08" would be more accurate.
* Table F2 gives ρ as a median over seven models and ρ_eq as a median over nine. Please use the same set of models within one table, or say so in the header rather than only in the caption.
* F10: "4 had an interval excluding zero, about as many as chance would give". The expected number under the null is 2.45 (49 × 0.05), and P(X ≥ 4) is about 0.23, so the statement is fair. Note that pairs within a model are not independent.
* Fig. 3b plots the interval of the measured share alone against a point. The caption admits this, but the figure then visually suggests six disagreements where the paired analysis finds two. Plot the paired difference instead.
* Fig. 3a: state whether the shares are summed over temperatures or are medians over temperatures, and why some predicted-response shares in the q < 0.1 bin are negative.
* Sec. III B: "For h > 0, exponential tilting also rewards each choice ..." The statement holds for either sign of h (for h < 0 it penalizes).
* Table I: GPT-2-XL is usually given as 1.5B parameters (1.56B). "1.6" is acceptable after rounding but differs from common usage.
* Sec. II: "Each test used seven temperatures per model, centered on a design temperature". The design runs were not centered this way (0.40–0.70). Please say "each prospective test".
* Appendix D: the ESS from four samples is very noisy, as the author notes. A pooled, self-normalized ESS over articles would make the Discussion's central contrast more convincing. *This would not change the conclusions, but it would strengthen the most interesting finding.*
* Data availability: "Generated continuations are available on request". Depositing them (they are model outputs, not BBC text) would allow full reanalysis of the bootstrap. Please consider it.
* Abstract: "The penalty we study adds h to the logits of repeat candidates" does not give the sign convention (h < 0 discourages repetition), which the reader needs in order to read the h* values.
* Grammar and style: the prose is generally clean. Some sentences are overloaded with parentheticals (for example, the F6 opening paragraph and the s_temp definition in Sec. III C), and splitting them would help.

---

## 9. Recommendation: **Major revision**

**Strengths.** The core identity is correct and simple. The prospective design, the explicit decision rules, the disclosure of an undecided first test, the status table for every analysis, the independent reimplementation and the extensive sensitivity analyses are exemplary for this field. The registered replication result (median 1.04, 0.98–1.15) is well supported for what it tests.

**Why not minor revision.** The paper's framing outruns its confirmatory evidence in ways a statistical reader will notice:

* the in-sample derivative (3.1);
* the median hiding per-model disagreement, together with an asymmetric comparison with the competitor (3.2);
* the failure to control T_× in any individual model, which the abstract describes as "did not always" (3.10);
* an underpowered "no drift" claim (3.5);
* an exploratory, length-dependent 99% figure presented as a headline (3.6).

Several statistical points also need attention: the ratio estimator and percentile intervals near the boundary (3.3), the Monte Carlo error at decisions settled by the third decimal (3.4), the two-stage testing and the absence of a pooled estimate (3.9), the contradiction about the preregistration status of the design runs (3.7), and external evidence of registration timing (3.8).

**Effect of the requested additions on the conclusions.** I expect most of them to leave the registered conclusion (median ratio within 0.8–1.25) unchanged:

* Unchanged: BCa or Fieller intervals, B ≥ 10,000, a pooled estimate, the heterogeneity summary, external timestamps and per-model decisions.
* Weakened: the resampled ρ_out intervals could weaken the exploratory claim that design-time predictions carry over.
* Rewording only: the corrected descriptions of T_× control and drift change how the claims must be worded, not the data.
* Possibly changed: an EOS-allowed run could change the descriptive switching results, but not the identity.

The paper also needs to be substantially shorter and more focused. Moving the equilibrium foil and the minor exploratory analyses to the supplement would let the actual contribution stand out: a prospective test showing that a first-order score-function prediction of an aggregate statistic holds at penalty strengths where likelihood-ratio reweighting has already failed.