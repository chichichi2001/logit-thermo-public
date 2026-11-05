# Referee report

**Manuscript:** "Predicting how repetition penalties act in language models from unpenalized samples"
**Venue:** PRX Intelligence
**Referee background:** nonequilibrium statistical physics (large deviations, tilted path ensembles, response theory, Markov processes)

---

## 0. Checks I carried out

The paper is long and contains many numbers, so I state first what I checked myself and what I took on trust.

- **Derivations.** I re-derived Sec. III A (the score (N − Q)/T and the covariance response), the Doob-martingale decomposition N − E[N] = (N − Q) + Σ_t ν_t and its consequence Q − E[Q] = Σ_t ν_t (Sec. III B), the first-order form of the finite-horizon Doob transform, the temperature score −(U − U_comp) and its invariance under per-step constants (Sec. III C), the response of ln K built from ratios of expected counts (Appendix B), the multiplicative-penalty derivative −|z| (Appendix B/F5), the log-weights of Appendix D, and the two-state formulas g_C = (b/ΔE)Δc_T/T and dT_×/dh = T_× Δc/ΔE (Appendix H). All are correct. The factorization g/g_C = (s_pen/s_temp)κ is an algebraic identity, as the authors say.
- **Internal arithmetic.** I recomputed the medians and means in Tables I, F4, F5, F6/F7 (including the k = 1 row of F7) and F8. I checked ρ_out/ρ in Table F5, g/g_C in Table H1 against s_pen, s_temp and κ, and ρ × g/g_C against ρ_eq. I checked |h*|/T* (0.66–2.08, quoted as "0.7 and 2.1"), the divisors θ_k = θ*^k in Table F7, the "1.8 to 14 times" statement in F5, and that the per-bin fractions in Table F1 sum to one. I also checked the label geometry (480 pairs, steps 16–496, the nine-step minimum exit time). All agree to rounding. The largest rounding gap is GPT-2-XL in ρ × g/g_C versus ρ_eq: 1.08 × 0.98 = 1.06 against 0.98, and the text reports "up to 0.07".
- **Figures.** Figs. 1–3 and S1–S2 agree with the tables where I could read them. For example, the slopes of Fig. S1 reproduce the "Slope" column of Table F10, and the bin shares in Fig. 3a match the text.

So I found no arithmetic errors. My concerns are about what the tests test, how the headline numbers should be read, and how the results are framed.

---

## 1. What the paper asks, does and finds (in my words)

**Question.** Can the effect of a logit penalty on repeat candidates be predicted, before the penalty is applied, from unpenalized samples alone? Here the effect is measured on a switching statistic of generated text: ln K, the log ratio of the rates of entering and leaving labeled repetitive stretches. A second, related question: how does a penalty that is renormalized at each step compare with the same penalty applied as a reweighting of whole texts (an exponentially tilted path ensemble)?

**What was done.** The author uses the likelihood-ratio (score-function) identity. For an additive penalty h on the candidate set, the score of a text at h = 0 is (N − Q)/T, so ∂E[O]/∂h = Cov(O, N − Q)/T for any statistic O. Applying this to the four counts behind K gives g(T) = ∂ ln K/∂h. The procedure:

1. From unpenalized runs on 40 design articles, fit ln K = a + b/T and locate the crossover temperature T_×.
2. Choose h* so that T_× should drop by 0.10.
3. Generate penalized text, and compare the realized change in ln K with h* g(T). Here g(T) is re-estimated from the unpenalized arm of the same test, and both sides are summed over seven temperatures.

The decision rule was fixed in advance: the bootstrap interval of the median ratio over models must lie inside 0.8–1.25. The first test, on the design articles, was undecided. A preregistered replication on 120 new articles with the same seven models gave 1.04 (0.98–1.15). A test on two 7B models gave a mean of 0.96 (0.84–1.11). The HF multiplicative penalty agreed at the predicted divisor (1.01) and at twice the predicted ln θ. It was undecided at four times and below the range at eight times.

Two further analyses were exploratory:

- s_pen = g/g_N, the step-wise response divided by the response to tilting by exp(hN/T), is 0.7–1.5%.
- Splitting the predicted response by q_t places about 80% of it at steps with 0.1 ≤ q_t ≤ 0.9, roughly in proportion to Σ q(1 − q).

**Main result.** On English news continuations of 512 tokens from base models of 0.4–7.6B parameters, a first-order prediction from unpenalized samples matches the realized change in ln K to within about ±10% for the median over models and ±20% per model. This holds at |h|/T up to about 2 at the design temperature, and in the first test up to about 4. The prediction of the shift of T_×, which is what a user would actually target, is much less reliable model by model (Sec. 3, M4).

---

## 2. The most interesting finding, and whether the abstract leads with it

In my view the most interesting finding is this. **A linear-response estimate built from a few hundred unpenalized texts stays accurate for ln K at per-step perturbations |h|/T ≈ 1–2, and in the first test even at 2h*.** At the same strengths, whole-text importance weights have already collapsed. This contrast between degenerate trajectory reweighting and robust linear response of an intensive statistic is well known in the large-deviation setting, but it is shown here cleanly and with preregistration in a new system. The second candidate is s_pen ≈ 1%. I argue in M2 that its size is probably set by the slowness of the switching, and that it compares the penalty with a reference no practitioner uses.

The first two sentences of the abstract are:

> "A repetition penalty in a language model is applied, and the distribution renormalized, at every sampling step. We predict its effect on how generated text switches between repetitive and ordinary stretches, using only samples generated without the penalty, and show that this first-order prediction holds at penalties strong enough to change the switching."

- The first sentence is background and carries no finding.
- The second states the prediction result but gives no measure of strength (|h|/T) and no measure of accuracy.
- The 99% cancellation, which the Introduction and the title of Sec. IV C treat as a central result, does not appear until the seventh sentence.

The author needs to decide which single message the paper carries. I suggest the prediction result, stated with its accuracy and the range of |h|/T. The second sentence should then say what is surprising, for example that linear response holds where reweighting fails. The abstract has about fifteen sentences and reads as a list of findings (see Sec. 5).

---

## 3. Major concerns

### M1. The preregistered test is a test of linearity and implementation, not of out-of-sample prediction

> "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test." (Sec. IV B)

> "Because the tests estimated the derivative from their own unpenalized runs, they test whether the first-order prediction holds at h*." (Sec. V)

The identity is exact at h = 0. Once g is estimated from the unpenalized arm of the same test, on the same articles and temperatures, the ratio Δ/(h* g) can depart from 1 for only three reasons:

1. curvature of ln K(h) between 0 and h*;
2. estimation noise;
3. an implementation error.

The confirmatory tests therefore show that ln K is close to linear in h up to h*, and that the code is correct. Both results are worth having. Two consequences follow, though:

- **"New articles" adds generality but not predictive content.** The ratio would mean the same thing on the design articles. Novelty of the articles matters only for the design-time prediction h ḡ, which is the exploratory Appendix F4 (ρ_out = 1.05 (0.99–1.11) for the smaller models and 1.07 (1.01–1.14) for the larger ones). That is the result the title promises, and it is currently in an appendix and not preregistered. I recommend:
  - presenting F4 in the main text, beside Table I, clearly labeled as exploratory;
  - saying plainly that ḡ was not resampled, so its intervals are too narrow;
  - if the author wants a confirmatory version, preregistering a small test whose predicted change is fixed in full before any data are collected.

  *(Moving F4 would not change the conclusions. A new preregistered test of h ḡ could strengthen them, or could narrow them if per-model failures like GPT-2-XL's 1.29 (1.22–1.37) repeat.)*
- **The replication uses the same seven models as the design.** The only new models are the two 7B models, decided on their mean, with an interval over articles only. With two models, the claim about "models" is a statement about these two. The Discussion says the bootstrap resamples articles, not models. The abstract's "median over models of 1.04" should carry the same qualification.

### M2. The 99% cancellation: what sets its size, and who needs it?

> "Each term of Q is fixed by the text before its step, yet Q covaries with the switching statistics almost as strongly as N does." (Abstract)

> "With the same formula we measure how much of a whole-text, Boltzmann-like response survives step-wise normalization, which explains why the effect is small" (Sec. I)

**(a) The "yet" suggests a surprise that is not there.** q_t is predictable, but in this system it is close to a state indicator: candidate-choice rates are 92–94% in the repetitive state and 16–21% in the ordinary state (Appendix F1). Q is therefore essentially an occupation time of the repetitive state, and so is N. Both covary strongly with any statistic of the switching. Being predictable does not stop a variable from covarying with later outcomes.

**(b) The size of s_pen probably follows from the slowness of the switching.** In a two-timescale picture, take a coarse two-state chain with per-step switching probabilities k_01 and k_10, both much less than 1.

- A step-wise field h changes each switching log-probability by O(h/T). The change in ln K is therefore O(h/T) times the number of decisive steps per switch.
- Tilting by exp(hN/T) changes the Doob eigenvector ratio r_1/r_0 by about (h/T)Δc·τ, where τ ~ 1/(k_01 + k_10) is the relaxation time in steps, or the remaining horizon, whichever is shorter.

So s_pen ~ (decisive steps per switch)/τ. With τ of order 100 steps, s_pen ~ 1% follows. This fits the persistence data in F2 (dwell bins of 49–96 and ≥97 steps both well populated). It also fits the observation that s_pen rises when the horizon is shortened (1.79% at 256 tokens, Table F11). The paper itself says

> "κ ≈ 1 is what the two-state form predicts, but it would also follow if one slow mode dominated both responses."

The same slow mode is the natural explanation of s_pen.

*Request:* report the entry and exit rates, or mean dwell times, at each usable temperature; they are not given anywhere. Then compare s_pen with τ⁻¹ across models, temperatures and the lengths of F8. *(This would not change the measured numbers. It would change how "cancels about 99%" is read: from a property of step-wise normalization to a restatement of the switching timescale. I consider that essential for a physics readership.)*

**(c) The reference response is one that nobody uses.** Exponential tilting by exp(hN/T) is a sequence-level reweighting that no sampler implements. Appendix H shows that the natural shortcut, a two-state Boltzmann model calibrated by the temperature response, is off by 45%, not by a factor of 100, because the temperature response is cancelled in the same way:

> "The factors show why the error is not a factor of 100"

So the sentence in the Introduction, "which explains why the effect is small", has no referent. Small compared with what? The penalty shifts T_× by about 0.1 at |h| ≈ 0.4–0.8, which is not a small effect. Please do one of the following:

- name the reader for whom the tilted response is the right baseline (for example, controlled generation with lookahead, where a Doob-type future factor is estimated, as in the cited works of Yang & Klein and Zhao et al.); or
- move s_pen out of the abstract.

*(Would not change the conclusions; changes the framing.)*

### M3. Why linear response survives at |h|/T ≈ 2: the importance-weight argument is weak and the explanation is untested

> "There, reweighting the four unpenalized continuations of an article by their exact probability ratios put most of the weight on one or two of them, with an effective sample size of 1.3 to 2.1 (Appendix D). We attribute the contrast to scale, without having tested it."

- **An ESS from four samples tells us almost nothing.** It is bounded by 4, very noisy, and biased upward when the weights are heavy-tailed. The relevant quantity can be computed analytically from the recorded q_t: to second order, Var(log w | x) ≈ (h/T)² Σ_t q_t(1 − q_t), with further terms from Var(N − Q). Report its distribution at h*. I expect values of order 10 or more, which would make the degeneracy of whole-text reweighting obvious without any appeal to four samples.
- **The proposed explanation is correct in spirit and easy to test.** The likelihood ratio is time-extensive, while ln K is a ratio of time-extensive counts. Its expansion in h is controlled by the scaled cumulants per step, not by L. The second-order coefficient ∂² ln K/∂h² can be estimated from the same unpenalized samples, from the second derivative of log P_h, which is −Σ q_t(1 − q_t)/T², together with Cov(e_i, σ²) terms. It can also be checked against the h* and 2h* arms of the first test. There is also a simple mechanism worth checking. Switching events are rare choices: entry is choosing a candidate when q_t is small, and exit is declining one when q_t is near 1. In both limits the log-probability of the choice is exactly linear in h/T, since log q' ≈ log q + h/T and log(1 − q') ≈ log(1 − q) − h/T. This would make ln(rate) linear in h far beyond the per-step regime, in the manner of an Arrhenius law.

*(Neither addition would change the conclusions. Each would replace a sentence that is currently speculation with a result, and the second-order estimate would show where the prediction should fail, which the multiplicative k = 8 test seems to probe.)*

### M4. The quantity a user targets, the shift of T_×, fails in about half the models; the main text understates this

> "The median then became 0.87, slightly below the target, so the overshoot came from the curvature (Appendix F6). Some smaller models, GPT-2-XL most of all, and the mean of the two larger models still departed from the target, for reasons we have not identified." (Sec. IV B)

From Table F8, the interpolated shift ratios per model are:

| Model | Ratio (interval) | Intended 1.0 |
|---|---|---|
| GPT-2-XL | 2.41 (2.19–2.79) | missed (overshoot) |
| SmolLM2 | 1.60 (1.43–1.81) | missed (overshoot) |
| Qwen2.5-7B | 1.53 (1.23–1.86) | missed (overshoot) |
| BLOOM | 0.51 (0.39–0.63) | missed (undershoot) |
| Pythia-1.4B | 1.16 (1.02–1.28) | borderline |

Four of nine intervals exclude 0.8–1.25. The spread across models runs from 0.51 to 2.41. "Some smaller models ... still departed" and "did not always move as intended" (abstract) understate this. A median of 0.87 over models that miss by factors between 0.5 and 2.4 should not be summarized as "the overshoot came from the curvature". Curvature explains the shift of the median from 1.32 to 0.87, not the spread across models.

**"For reasons we have not identified" is partly contradicted by the paper's own appendices, at least for GPT-2-XL.**

- On the design runs, the identity's per-temperature median ratio for GPT-2-XL was already 1.60 (Table F10), against 0.94–1.15 for the other models.
- The directly measured slope dT_×/dh = 0.21 (Table F10, Fig. S1) implies |h| ≈ 0.48 for a shift of 0.10. The identity-based h* was −0.79, so the measured slope predicts a shift of about 1.66 times the intended one.
- In F4, ρ_out/ρ = 1.20: the design-time ḡ was 20% smaller than the g on the replication articles.

So for GPT-2-XL the design-time ḡ was low, and the design data showed it. The same cross-check with the Fig. S1 slope also flags SmolLM2 (implied 1.27). Please discuss this, and say whether the design procedure should have compared the identity's implied dT_×/dh with the measured slope before h* was fixed.

**A design problem.** The test grid T* ± 0.15 is centered on the penalized target, so the unpenalized crossing at T* + 0.10 sits near the edge of the grid by construction. That is where a straight-line fit is least accurate. The paper diagnoses this (F6) but does not say it was a design choice that could have been avoided, for example by using a grid that spans both T* and T_×.

*(Stating the per-model failure rate in the abstract and Sec. IV B would change the practical conclusion. The procedure predicts Δ ln K reliably, but it does not reliably deliver a chosen T_× for a given model.)*

### M5. Multiplicative penalty: "up to the weakest strengths used in practice" holds for about one model, and the failure at k = 8 may be partly a selection effect

> "The prediction therefore held for divisors up to about 1.1, the weakest values used in practice, and not reliably beyond." (Sec. IV B)

- **Only one model was tested at practical strengths.** At k = 2 the divisors were 1.04–1.11 (Table F7). Only GPT-2-XL (θ*² = 1.111) reached the "1.1–1.3 common in practice"; the other five were at 1.04–1.08. The supported statement is that the prediction held for divisors up to about 1.04–1.08 in five models and 1.11 in one. The abstract's "followed the prediction only up to the weakest strengths used in practice" should say this.
- **The k = 8 result may be partly an artifact of temperature selection.** At k = 8, Pythia-410M kept one usable temperature, and three other models kept two or three. The ratio is computed over the temperatures at which enough entries survived the penalty. Those are the temperatures where the penalty suppressed repetition *least*, which biases the ratio downward. Please recompute the k = 2, 4, 8 ratios on a fixed set of temperatures usable at all k, or show that the selection does not matter. *(This could change the conclusion "below the range at eight times", or at least its size.)*
- **GPT-Neo-1.3B is effectively untested.** θ* = 1.51 gave 1.32 (0.87–2.71), and k = 2 (θ = 2.27) gave 1.64 (1.11–3.59). With ḡ_λ 13–22 times smaller than in the other models, this model's prediction is barely constrained. Say so where Fig. 2 shows it.

### M6. The thermodynamic framing goes beyond what the paper's own checks support

The paper borrows "van 't Hoff plot", "crossover temperature", "Clausius–Clapeyron", an "equilibrium two-state model" and phase-transition literature. Its own checks then show the following:

- the windowed repeat fraction does not meet the preregistered bimodality criterion (F1);
- K is not the rate ratio of a stationary chain: ln K rises by 1.3–2.1 between the two halves of a text (F2, Table F3);
- the exit hazard for long stretches is 0.28 of that for shorter ones, which is aging and not Markovian;
- T_× depends on the labeling and on L.

The paper hedges each of these honestly (Sec. II, last paragraph). The cumulative effect, though, is that "crossover temperature" is a property of a window of 512 tokens under one labeling, and K is an average over a process whose drift along the text is *larger than the penalty's effect*. At T*, the target change h* ḡ ≈ b·0.1/(T* T_×) is of order 1, against a drift of 1.3–2.1. The score-function result needs none of this structure, since it holds for any statistic. I recommend:

- dropping "equilibrium" and "Clausius–Clapeyron" from the main text, or framing them explicitly as a naive comparator;
- calling T_× what it is (the temperature at which the labeled entry and exit rates are equal over 512 tokens);
- **reporting the probability mass on the masked end-of-sequence token.** Forcing base models to write 512 tokens past a natural end of the article is a plausible driver of both the drift toward repetition and the late-text repetitive state. This bears directly on whether the "repetitive state" here is the loop that practitioners meet, where EOS is allowed. *(Would not change the prediction result. It could change the interpretation of the drift, of T_× and of the relevance of the phenomenology.)*

**A related remark on Sec. V.**

> "The compensator Q is the discrete-time counterpart of the escape-rate term in response theory for jump processes [0902.3955; 2005.10503], although that theory separates terms by time reversal, which autoregressive sampling lacks."

The escape-rate term is exactly the compensator (the Doob–Meyer compensator of the jump-counting process), so the correspondence is closer than a "counterpart". The Girsanov / likelihood-ratio response formula itself does not need time reversal. Time reversal is needed only for the further split into entropic and frenetic parts. As written, the sentence suggests that the formula used here differs in kind from the one for jump processes, which it does not. Please rephrase.

### M7. Fig. 3a gives an attribution, not a causal localization; a cheap experiment would make it causal

> "The derivative also shows that the response arises mainly at the minority of steps where the model is uncertain whether to choose a repeat candidate." (Abstract)

The decomposition Σ_t Cov(O, c_t − q_t) is the natural one, since the martingale increments are orthogonal. But "arises" is a causal word.

- **An exact first-order test is available.** Because {t : 0.1 ≤ q_t ≤ 0.9} is a predictable set, a penalty applied only at those steps has score Σ_{t∈set}(c_t − q_t)/T exactly. Its effect on ln K is therefore predicted, to first order, to be about 80% of the full effect. One penalized arm per model at h* would turn the attribution into a tested claim. *(Would not change the main conclusions; it would support or remove the last sentence of the abstract, which currently rests on an exploratory analysis.)*
- **The proportionality has a clear meaning that should be stated.** Each step contributes q_t(1 − q_t) × (E[O | c_t = 1, x_<t] − E[O | c_t = 0, x_<t]). Proportionality to the variance therefore means that the effect of one choice on O is about the same across q bins. Fig. 3a also shows it is about 1.5 times larger for q > 0.9 (response share about 0.2 against variance share about 0.13).

### M8. Statistical points

**(a) A small systematic excess over 1 is not discussed.** In the replication, six of seven point estimates are above 1, and the mean is 1.06. The first test gave 1.17 and 1.10. F4 gives 1.05. Appendix C says the range was set partly

> "from a synthetic check in which the labeling raised the ratio by about 10%."

Since the identity is exact at h = 0, how can the labeling raise the ratio? Any systematic departure must come from curvature in h, or from the finite-sample bias of a ratio of estimated quantities. Please explain the synthetic check, and say whether the ~5% excess is a second-order effect (see M3).

**(b) Sequential testing.** Confirmation came after an undecided first test, from a single preregistered replication. This is disclosed, but the error rate of the overall procedure ("confirm on the first or the second test") is higher than that of either test alone. Please also report the ratio pooled over all 160 articles, using the h* arm of the first test. *(Would not change the conclusion unless the pooled interval crosses 1.25, which the numbers suggest it will not.)*

**(c) "Did not drift systematically" is stronger than F10 supports.** Sec. IV B says

> "across temperatures the ratio did not drift systematically (Appendix F10)."

F10 reports a median change of 0.01 (−0.446 to 0.161) against a bound of ±0.45. Drifts of up to about 45% across the grid are not excluded, and for the larger models the result was undecided. Say "a drift larger than ±0.45 across the grid was excluded for the smaller models".

**(d) The "about 1%" rule is too loose to support claims of stability.**

> "Before computing, we called s_pen "about 1%" if the interval of the median lay within 0.5–2%."

That band is a factor of four wide. With it, 1.79% at 256 tokens counts as "about 1%", while the length dependence (a factor of 1.48 (1.37–1.60) between 256 and 512 tokens) is real. Table C1's "About 1% in all three judged cases" should not be read as length-independence. The text partly says this; Table C1 does not.

**(e) The claim about common random numbers is unverified.**

> "Penalized and unpenalized runs therefore shared their random numbers, which reduces the variance of their difference."

In autoregressive sampling the coupling decays once the two arms emit different tokens, which happens early at |h|/T ≈ 1. Either report the correlation between arms of the per-article ln K (or the variance of the difference with and without pairing), or drop the claim. The paired bootstrap is valid either way. *(Would not change the conclusions.)*

**(f) Per-model reliability of g.** GPT-2-XL's ratio was 1.60 (design per-temperature median), 1.17 (first test), 1.08 (replication) and 1.29 (ρ_out). For a reader who wants to use this on one model, the per-model precision matters more than the median. A short paragraph on per-model reliability would help.

### M9. Which "crossover temperature" is meant in Fig. 1, Table F1 and Table F3?

Table F1 says T is the "temperature of the test grid closest to T_× estimated from the unpenalized runs of the same test". For Pythia-1.4B, T = 0.62. But:

- Table I gives T_× = 0.57;
- the interpolated unpenalized crossing is 0.577 (Table F8);
- 0.57 is itself a grid point (T* = 0.47 ± 0.15).

The T_× used here is evidently the straight-line fit on the test grid, which the paper itself shows is biased by curvature (F6). The same happens for GPT-2-XL (0.53 against 0.48 and an interpolated 0.504), Qwen2.5-7B (0.60 against 0.55 and 0.564) and Mistral (0.49 against 0.44 and 0.464). Fig. 1a ("Pythia-1.4B at T = 0.62") and Fig. 1b ("closest to each model's crossover temperature") inherit this. Please use one definition of T_× throughout, preferably the interpolated crossing, and redo Fig. 1b and Tables F1 and F3 with it. *(Would not change the conclusions; it removes an inconsistency a careful reader will find.)*

### M10. Responsibility for the derivations (for the editor)

> "The derivations of Sec. III and Appendices G and H were likewise written by an AI assistant and checked by other AI instances, not by the author."

I checked the derivations of Sec. III and Appendix H (Sec. 0) and found them correct. The independent reimplementation that reproduced the ratios to four decimals is a valuable check, and I commend it. However, the author of a paper in this journal must be able to stand behind its central derivation, and these derivations are a few lines long. I ask the author to verify Sec. III personally and to revise the statement once that is done. Whether the code disclosure is acceptable as written is a matter of journal policy, and I leave it to the editor.

---

## 4. What could be removed or moved, and which caveats are essential

**Could be moved or cut without changing any conclusion:**

1. **s_temp (Table II, Sec. III C).** s_temp depends on the arbitrary energy assigned to a step (Table II caption: "Unlike s_pen, s_temp depends on the energy assigned to a step"), and it serves only the comparison with the two-state model. Move the s_temp column and most of Sec. III C to Appendix H. Keep one sentence saying that the temperature response also subtracts a compensator.
2. **Sec. III D, Appendix H, F7, Fig. S2 and the R questions of F9.** Together these take a great deal of space for a comparator that the paper itself shows is ill-defined ("a property of the chosen energy, not of the sampler"). One paragraph in the main text and a shortened appendix would do. The preregistered R test (Fig. S2a) can stay in the appendix.
3. **Appendix D and the ESS sentence in Sec. V.** Replace them with the analytic log-weight variance (M3), or delete them.
4. **Appendix G (truncated sampling).** It is untested with language models ("Nothing in this appendix has been tested with language models"). Reduce it to two sentences.
5. **F10, parts one and three.** One sentence each. Part three ("undecided because no alternative rule could be tested") reports that a robustness check had nothing to check.
6. **F6, "Penalized ln K at the design temperature".** This is a dense paragraph of numbers. Put the three-part decomposition in a small table or cut it.
7. **Sec. III A.** Drop the sentence on bounded stopping times ("The identity also holds with a bounded stopping time ... but we did not use one").
8. **Sec. V.** Move the note on model revisions to Appendix A only; it is already there.
9. **Table II.** Give ranges in the text and move the per-model table to the appendix.
10. **The abstract's last sentence.** It reports an exploratory attribution (M7). Drop it, or mark it as exploratory.

**Caveats the reader really needs, which should all appear in the main text:**

- g comes from the unpenalized arm of the same test, so the confirmatory tests test linearity at h*, and the design-only prediction (F4) is exploratory (M1).
- Confirmation came after one undecided test.
- Intervals are over articles, not models, and only two models are new.
- The states are a labeling, K averages over a strong drift along the text, and T_× depends on labeling and length.
- The T_× target was missed in four of nine models (M4).
- The multiplicative penalty was validated only below the strengths used in practice for five of six models (M5).
- s_pen depends on length, and probably on the switching timescale (M2).

---

## 5. Undefined terms, jargon, and passages that read as notes

**Terms that need definition or replacement:**

- **"identity" as a noun for the first-order prediction** (Fig. 3b legend "identity"; Fig. S2b "Score-function identity"; F10 "the identity's entry share"; F9 "the identity was closer"). An identity is exact; the prediction at finite h is not. Use "first-order prediction".
- **"compensator."** It is defined in passing. A sentence naming the Doob–Meyer decomposition, plus one line of intuition for ML readers, would help.
- **"static susceptibility of the path ensemble."** For physicists, "static susceptibility" suggests an equilibrium thermodynamic response. "Susceptibility of the tilted (biased) trajectory ensemble" is more accurate.
- **"units of m steps" (Appendix H).** m is not defined physically, and the reader is not told what a "unit" is.
- **Labels from the analysis records.** "record", "rule fixed", "Exploratory, record", and "arm" (clinical-trial vocabulary) are used without explanation outside Table C1.
- **"repeats" and "candidates", plus a third set for the multiplicative penalty.** Three closely related sets carry similar names. A three-row table (definition, role, example) in Sec. II would remove most of the confusion.
- **Overloaded symbols.** a is both the intercept of the ln K fit and the logit derivative a(v). s is the temperature score in Appendix B while s_pen and s_temp are fractions. σ is the score sum, which physicists will read as entropy production. S is the samples per article and S_t is the truncation set. R is used in F7 and F9 next to ρ, ρ_out, ρ_eq and r_C.

**Passages that read as notes or lists rather than connected prose:**

- **The abstract.** Fifteen sentences, most of which add one more result.
- **Sec. IV B, second paragraph.** Seven results in a row, each with its own interval.
- **The last two paragraphs of Sec. V.** A list of caveats with no argument connecting them.
- **Appendix F as a whole**, and especially F6 and F10. Short declarative sentences with numbers, few of which say why the number matters.
- **F10:** "They answer three questions raised in review." This reads as a note from an earlier round of review and should be removed.
- **Table C1:** "Other genres | F9 | Preregistered | See F9". The outcome column should state the outcome.

---

## 6. Common weaknesses: are they present?

- **A large promise backed by a small experiment.** Partly. The title promises to predict "how repetition penalties act". The evidence covers one switching statistic (ln K), one length (512), no truncation, base models only, English news only for the confirmatory tests, and one penalty (the additive one) that "is not an option in common libraries". The practical penalty was validated mostly below practical strengths. The instruction-tuned test was undecided for one model and not run for the other. The title should name the statistic, for example "...act on switching between repetitive and ordinary text".
- **A single run, model or dataset.**
  - Models: nine is adequate, but the confirmatory replication reused the seven design models.
  - Datasets: one for all confirmatory tests (BBC news, April–June 2025). The other genres (F9) were tested only with h = ±0.3 and with questions about the two-state model, and no intervals are given for the identity's ratios there (0.71–1.46).
  - Replications: one, after one undecided test.
  - Seeds: each configuration was generated once with fixed seeds, with four samples per article. The article bootstrap covers sampling variance, so this is acceptable.
- **Parameter choices without a stated reason or time of choice.**
  - *With reasons:* the range 0.8–1.25 (reason and time stated), L = 512 (reason and time stated), 4-grams for repeats and two-token context for candidates (reason stated), the ten-switch threshold (robustness shown).
  - *Without a reason:* the target shift of 0.10; the 32-token window and thresholds of 1/2 and 1/4 (robustness shown in F1, but neither reason nor time of choice is stated); the 64-token prompt; four samples per article (this choice limits the within-article covariances and the ESS); temperature spacing of 0.05; the 15-temperature range 0.20–0.90 and the eligibility window 0.3–0.8; the q bins at 0.1 and 0.9; the hazard bins 49–96 and ≥97 (partly justified).
- **No report of computational cost.** Cost is reported for the replication (about 22 L4 GPU-hours) and the second stage of the 7B test (about 6 A100 GPU-hours). It is missing for the design runs, the first test, the multiplicative and stronger-divisor runs, the instruction-tuned runs, the genre runs and the re-scoring. A total would help readers judge the claim in Sec. V that locating T_× is "comparable in cost to trying a few penalty strengths".
- **A broad conclusion from a narrow case.**
  - "which explains why the effect is small" (Sec. I) is broad and has no referent (M2).
  - "so the overshoot came from the curvature" (Sec. IV B) generalizes from the median to all models (M4).
  - The extension to logit biases, watermarking and truncated sampling (Sec. V, Appendix G) is clearly marked as untested, which is acceptable.
  - The 99% figure is hedged with "In our runs", which is good, but it depends on length and probably on the switching timescale, and this should be said next to it.

---

## 7. Could the author defend each design choice? Where they would struggle

Choices the author can clearly defend:

- the ratio of sums over temperatures (Appendix C gives the reason);
- the median over models, "so that no single model could decide it";
- the decision range;
- separating repeats from candidates.

Places where I expect the author to struggle if asked directly:

1. **"Why ln K, and not a quantity a user cares about?"** For example, the fraction of repeated tokens, or the probability that a continuation ends in a loop. The identity applies to any statistic, so the choice of ln K appears to come from the thermodynamic narrative and not from use. Reporting the prediction for one user-facing statistic would be cheap. *(Would not change the conclusions; it would widen their relevance.)*
2. **"What does the tilted response correspond to in practice, and what sets s_pen ≈ 1%?"** (M2.)
3. **"Why is the response linear up to |h|/T ≈ 2–4?"** At present the answer is "We attribute the contrast to scale, without having tested it" (M3).
4. **"How can the labeling raise the ratio by 10% if the identity is exact?"** (M8a.)
5. **"Why was the test grid centered on T*, which puts the unpenalized crossing at its edge? And why was the identity-based h* not cross-checked against the measured slope dT_×/dh (Fig. S1), which already flagged GPT-2-XL?"** (M4.)
6. **"Why within-article covariances for s_pen but pooled ones for the tests?"** F8 answers this adequately, but the reason should be in Sec. IV C.
7. **"Why 0.10 for the target shift?"** No reason is given.
8. **"Why compare with a two-state Boltzmann model whose prediction depends on an arbitrary energy gauge?"** The paper's own answer ("a natural shortcut") sits uneasily with Appendix H's conclusion that the comparison is "a property of the chosen energy, not of the sampler".
9. **"Can you explain the derivation of Sec. III B without notes?"** Given the statement in Appendix E, the editor may ask this.

---

## 8. Minor issues

1. **Sec. II:**
   > "Repeats define the states below and candidates define the penalty, so that the definition of the states does not depend on the penalty."

   The states would not depend on the penalty even if the same set were used. Appendix C gives the real reason ("keeps the events that define the states distinct from the events the penalty acts on"). Use that wording.
2. **Two definitions of usable temperatures.** Sec. II says "both kinds of switch occurred there at least ten times". Appendix C says "at least ten entries and ten exits in both arms". Make them consistent.
3. **F1 and Sec. IV A:**
   > "The high side is small because continuations start in the ordinary state"

   F1 also reports that one continuation of Pythia-1.4B "stayed repetitive throughout", so it started in the repetitive state. Say "almost all continuations start ...".
4. **Appendix A:** "Articles shorter than 576 tokens in a model's tokenizer were not used." State how many were excluded per model. If the article sets differ across models, the "same resample for all models" needs a remark.
5. **Appendix B:** "agreed ... to within 0.0015 and 0.0007 at most". Units? Per step or per text? Also, why is there any disagreement at all, given 64-bit sampling and bit-identical reruns? (Presumably batch shape.)
6. **Appendix B:** the heading "Temperature identity and the choice of energy" promises more about the choice of energy than the subsection contains.
7. **Table I:** models are listed neither by family nor by size (1.4, 1.3, 1.7, 1.5, 1.6, 1.1). Order them by parameter count, or group them by family.
8. **Table F1:** inconsistent decimals ("0.0093", "0.0075" next to "0.012").
9. **Fig. 1c:** the grey lines cannot be identified. Label them, or say in the caption that only Pythia-1.4B is identified. Showing q_t or the candidate marks in Fig. 1a would connect the figure to the theory.
10. **Fig. 2:** the GPT-Neo-1.3B multiplicative interval (0.87–2.71) dominates the axis. A note in the caption on the small ḡ_λ of that model would help.
11. **F9 / Table F12:** give intervals for the "Identity" column. These are the only identity results outside news.
12. **Discussion:** "with |h*|/T* between 0.7 and 2.1". Over the test grid the range is wider (about 0.5–3.4). State that the quoted range is at T*.
13. **Sec. III C:** "Over several temperatures we sum numerator and denominator with the weights of the equilibrium prediction g_C". This forward reference makes s_temp hard to parse on first reading. Define g_C first, or move the weighting to Appendix H.
14. **References:** citations given as bare arXiv numbers ([1904.09751]) mixed with author–year ([Glynn 1990]). Use the journal style throughout.
15. **Data availability:** "Generated continuations are available on request." These are the primary data and they are model outputs, not BBC text. Deposit them with the code.
16. **Sec. II:** "Sampling was plain softmax at temperatures between 0.19 and 0.90." Appendix F5 refers to a provisional T_× of 0.16 found from a 0.20–0.90 grid, so that T_× was extrapolated. Say so.
17. **Small wording points:**
    - "the realized change" and "the measured response" are used interchangeably; choose one.
    - "replication" is used for a test that also re-estimates g; "second preregistered test" is more accurate.
    - "Δc_T" and "Δc" are both used; define the subscript once.

---

## 9. Recommendation: **major revision**

**Strengths.**

- The theory is correct.
- The prospective design is unusually careful for this field: dated documents, hashed predictions, an independent reimplementation, honest reporting of an undecided first test and of failed secondary analyses.
- The central empirical result, that first-order response from unpenalized samples predicts the change in a switching statistic at |h|/T ≈ 1–2, is useful and clearly demonstrated for these models and this text type.
- I found no arithmetic inconsistencies.

**Why not minor revision.**

1. **The confirmatory tests are tests of linearity at h*, with g taken from the same arm** (M1). The truly predictive result (F4) is exploratory and sits in an appendix, while the title and abstract read as if it were the main result.
2. **The second headline, "cancels about 99%", lacks a mechanism for its size and a reader for whom the reference response matters** (M2). Its size very likely restates the slowness of the switching. The Introduction's "explains why the effect is small" has no referent, since the paper's own Appendix H shows that the natural shortcut is off by 45%, not 100×.
3. **The user-facing target (T_×) was missed in four of nine models, and the multiplicative penalty was validated at practical strengths in about one model** (M4, M5). The abstract and Sec. IV B understate both. The k = 8 failure may partly reflect the selection of usable temperatures.
4. **The thermodynamic framing is stronger than the paper's own checks allow** (M6). The main result does not need it.

**Expected effect of the requested changes.**

- Most requests (dwell times, analytic log-weight variance, a second-order estimate, pooled analysis over 160 articles, one T_× definition, EOS mass, correlation between arms) would not change the conclusions. They would turn speculation into results and make the headline numbers interpretable.
- Three could change conclusions or their scope:
  - recomputing the stronger multiplicative penalties on a fixed set of temperatures (could change "below the range at k = 8");
  - the gated-penalty experiment of M7 (could support or remove the abstract's last sentence);
  - a preregistered test of the design-time prediction h ḡ (could strengthen or narrow the main claim).

With these revisions, and a shorter, single-message abstract, I would expect to support publication.