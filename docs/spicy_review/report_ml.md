# Referee report

**Manuscript:** "Predicting how repetition penalties act in language models from unpenalized samples"
**Venue:** PRX Intelligence
**Referee background:** decoding and sampling methods, repetition and degeneration in generated text, evaluation of language models

---

## 1. What the paper asks, what was done, and the main result

**Question.** Can you predict, before running any penalized generation, how much a repetition penalty will change the way sampled text switches between repetitive and ordinary stretches? And why does a per-step penalty have the effect size it has?

**What was done.**
- The author labels every step of a 512-token continuation as "repetitive" or "ordinary" from a 32-token window of 4-gram repeats, using hysteresis. The switching statistic is ln K, the log of the entry rate divided by the exit rate.
- The penalty studied adds h to the logits of "repeat candidates": tokens that have previously followed the current last two tokens.
- The likelihood-ratio (score-function) identity gives the exact derivative of E[·] with respect to h at h = 0, as a covariance with σ = N − Q. This yields g(T) = ∂ln K/∂h, estimated from unpenalized samples only.
- Unpenalized design runs on 40 articles were used to choose h* so that the crossover temperature T_× (where ln K = 0) would drop by 0.10.
- Preregistered tests then compared the realized Δln K at h* with the first-order prediction h*·g(T), where g came from the unpenalized arm of the same test. The decision rule required the bootstrap interval of the median ratio to lie in 0.8–1.25.
- The paper also decomposes the response into the part that survives step-wise normalization (s_pen = g/g_N, compared with exponential tilting of whole texts) and attributes the response to steps by the candidate probability q_t.

**Main result.**
- The first test (40 design articles) was undecided.
- The preregistered replication on 120 new articles and seven base models (0.4–1.7B) gave a median ratio of 1.04 (0.98–1.15). A separate test on two ~7B models gave a mean of 0.96 (0.84–1.11).
- The HF-style multiplicative penalty followed the prediction at θ* ≈ 1.02–1.05 and 2λ*, but fell below the range at 8λ*.
- The step-wise response of ln K is about 1% of the response under whole-text exponential tilting.
- The intended shift of T_× was not reliably achieved per model.

## 2. The most interesting finding, and whether it opens the abstract

To me the most interesting finding is one the paper mentions only in passing, in the Discussion and Appendix D. A first-order derivative estimated from unpenalized samples predicts Δln K at |h*|/T* of 0.7–2.1. At the same penalties, exact importance reweighting of whole texts collapses: "reweighting the four unpenalized continuations of an article by their exact probability ratios put most of the weight on one or two of them, with an effective sample size of 1.3 to 2.1". So an aggregate count statistic stays nearly linear in a penalty that already makes the per-text likelihood ratio degenerate. This is a useful and non-obvious fact for anyone who tunes decoding hyperparameters. The paper then says "We attribute the contrast to scale, without having tested it."

The second candidate is the ~1% surviving fraction s_pen. Below (M3) I argue that its current framing, against whole-text tilting, is less informative than the paper suggests. A reframing would make it more interesting.

Is the main finding in the first two sentences of the abstract? Only partly.
- Sentence 1 is setup: "A repetition penalty in a language model is applied, and the distribution renormalized, at every sampling step."
- Sentence 2 states the claim in vague terms: "show that this first-order prediction holds at penalties strong enough to change the switching." The reader does not learn how strong (|h|/T up to ~2; multiplicative only to θ ≈ 1.1), how well (1.04), or for which statistic until sentences 10–12.
- The 99% figure arrives in sentence 7.
- The contrast with importance reweighting does not appear in the abstract at all.

The abstract is about 330 words, carries many symbols (h, N, Q, T, q_t) and numbers, and reads as a compressed paper rather than one message.

## 3. Major concerns

### M1. The registered test checks linearity on the same data with a lenient aggregate criterion. It does not show that the model-specific derivative adds predictive value.

The paper is candid that "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test" and that "they test whether the first-order prediction holds at h*."

Under that design, the score-function estimator is unbiased for the derivative by construction, apart from ratio-of-expectations bias. The empirical content is therefore that ln K is close to linear in h over [0, h*], at a 25% tolerance, on the median of seven models.

Before the tests, the author had also already seen agreement at h = ±0.3. The band was "set before the first test, from the agreement already seen on the design runs at h = ±0.3, with medians of 1.00 and 0.96". The prospective tests extrapolate h by a factor of only about 1.2–2.6 beyond this.

The criterion is too weak to separate the model-specific prediction from trivial alternatives. A rough check uses Table F5: ρ_constant = ρ_out·ḡ_m/2.4 predicts every model with a single constant g = 2.4 per unit h.
- For the seven smaller models this gives 1.12, 1.08, 1.10, 1.21, 0.79, 1.11 and 0.90, with a median of 1.10.
- That median lies inside 0.8–1.25.
- The constant is post hoc and I cannot compute its interval. Still, it suggests the median-over-models rule cannot tell a per-model derivative from a cross-model average.

**Requests.**
- (a) Report, on the replication data, the same ratio for three baselines:
  - the design-time ḡ (already in F4);
  - a single cross-model constant g;
  - a finite-difference slope from the existing design runs at h = ±0.3 with common random numbers, scaled to h*.

  This could change the practical conclusion: if the finite difference at equal compute is as accurate, the selling point "from unpenalized samples" becomes a convenience rather than an advantage.
- (b) Report a per-model verdict under the same rule. Four of the seven per-model intervals in Table I reach above 1.25; under the paper's own rule those models are "undecided". This would change how the abstract should be worded ("holds" vs. "holds on the median").
- (c) Report the estimator's variance against that of the finite difference at matched compute. This is the standard comparison for score-function versus pathwise or finite-difference estimators in ML. It would not change the correctness of the identity, but it would decide whether the method is worth using.

### M2. The practical relevance to decoding is not established, and the title promises more than was tested.

The title, "Predicting how repetition penalties act in language models", is broad. In fact:

- **The additive penalty is not used in practice.** The paper says: "This additive penalty is not an option in common libraries." However, the candidate set ("tokens that have previously followed the current last two tokens") is exactly the set banned by `no_repeat_ngram_size=3` in Hugging Face `generate`. So the additive penalty is a soft version of a widely used hard constraint, with the hard constraint at h → −∞. Penalties like DRY in several open-source inference stacks are also close relatives. Saying this would strengthen relevance. It also identifies the limit where the first-order prediction must fail, and that limit should be discussed.
- **The one penalty practitioners use works only below the practical range.** "The prediction therefore held for divisors up to about 1.1, the weakest values used in practice, and not reliably beyond." The predicted θ* of 1.02–1.05 is far below the 1.1–1.3 range that "correspond[s] … to 1.8 to 14 times the predicted ln θ*". The method is validated only for that weakest end of the range.
- **Presence and frequency penalties are untested.** "The widely used presence and frequency penalties were not tested." The presence penalty is additive on a context-dependent set (seen tokens), so the identity applies with no new theory. It is available in every major serving API. This is the single most useful extension. It could change the scope of the conclusions in either direction.
- **The regime is narrow.** Sampling is plain softmax at T = 0.19–0.90 with no top-p or top-k, on base models, with EOS masked. Instruction-tuned models are effectively untested: one test was undecided, 1.36 (0.81–2.40), and SmolLM2-Instruct had T_× = 0.16, so no test was possible. That T_× = 0.16 itself suggests switching into loops is mainly a base-model, low-temperature phenomenon. The Discussion should say this plainly.
- **The target statistic is unusual.** Practitioners measure repetition by rep-n or distinct-n rates, the fraction of outputs that end in a loop, or similar quantities, and they care about the quality cost. The paper's statement that "The shift of T_×, which a user would target" is unsupported; I know of no user who targets a crossover temperature.
- **Request.** Apply the same identity, prospectively or at least on the existing runs, to one or two standard degeneration metrics (e.g. mean repeat fraction; probability that the last 64 tokens are repetitive). Also report a quality proxy for the penalized arms, such as the perplexity of the continuations under a reference model. This would not change the ln K result. It would show whether the linearity is specific to ln K, which the Discussion says is untested ("We attribute the contrast to scale"). If linearity fails for a simple repeat fraction, the general claim in the title would need to be narrowed.

### M3. The "99% cancellation" is compared with an intervention nobody applies, and its interpretation is overstated.

The abstract says: "step-wise normalization therefore cancels about 99% of the first-order response the same penalty would produce on whole texts." The Introduction says the measurement "explains why the effect is small."

- **Tilting by exp(hN/T) is not "the same penalty … on whole texts" in any operational sense.** It is a sequence-level reweighting whose exponent grows with the length of the text. No decoder implements it, and no one would predict a penalty's effect with it.
- **"Small" has no yardstick.** At h* the penalty moves T_× by about 0.1 and ln K by about 1. That is not small for its purpose. It is small only relative to the tilted response.
- **The large Cov(O, N) ≈ Cov(O, Q) is largely mechanical.** "In the repetitive state the models chose a repeat candidate at almost every step" (92–94%), and q_t is near one there. Both N and Q are therefore close proxies for the time spent in the repetitive state, which is what the counts in K measure. The sentence "Each term of Q is fixed by the text before its step, yet Q covaries with the switching statistics almost as strongly as N does" presents an expected consequence of the construction as a surprise.
- **The figure depends on length.** s_pen rises to 1.8% at 256 tokens (F8), and "length and position are confounded."
- **A better framing exists.** By the paper's own decomposition N − E[N] = (N − Q) + Σν_t, 1/s_pen ≈ 70–140 measures how much the downstream consequences of a candidate choice (the later candidate choices it makes likely) outweigh the choice itself, in their covariance with ln K. Read this way, it is an amplification factor of self-reinforcing repetition, which connects directly to [2206.02369]. If the author can make that reading precise, it is a much more useful headline for an ML audience than "fraction of the tilted response."
- **Request.** Report s_pen for one statistic not built from the same tokens as N, for example a labeling based on 6-gram repeats with candidates on a 1-token context, or a non-repetition statistic. This would change the interpretation of "about 99%" (a general property of step-wise normalization vs. a property of this coupling), but not the main test.

Note also that the comparison in Sec. IV B, "the tilted response of Sec. IV C would give a ratio of about 0.01", and the equilibrium model's ρ_eq = 0.69 are two different "naive baselines". The reader is left unsure which is the intended foil.

### M4. Failures on the user-facing target are understated, and part of the GPT-2-XL anomaly was visible in the design data.

- **Per-model T_× shifts are far off.** Table F8, interpolated crossing, gives GPT-2-XL 2.41 (2.19–2.79), SmolLM2-1.7B 1.60 (1.43–1.81), BLOOM-1.1B 0.51 (0.39–0.63) and Qwen2.5-7B 1.53 (1.23–1.86). Three of nine are entirely outside the band and a fourth nearly so. Note also that the interpolated crossing was introduced after the tests.
- **The abstract phrasing is too soft.** "that temperature did not always move as intended" understates errors that run from half to 2.4 times the intended shift.
- **The paper says the cause is unknown, but there is a lead.** Sec. IV B says GPT-2-XL departed "for reasons we have not identified." Yet Table F10 shows the identity's median per-temperature ratio for GPT-2-XL on the design runs was already 1.60, the only value far from one. A rough slope implied by Table I, 0.10/|h*| ≈ 0.13 per unit h, also falls below the measured 0.21 in Table F10. (This ignores the curvature of T_× in h, so treat it as indicative only.)
- **Request.** Add the identity's predicted dT_×/dh to Fig. S2a alongside the equilibrium prediction. This would show whether the overshoot was predictable from pre-test data. It would not change the ln K result, but it would change the account of the T_× failures.
- **One statement in F6 does not hold per model.** F6 concludes "The overshoot of T_× therefore does not come from a misprediction of ln K at T*", but the per-model total deviation reaches 0.87 for GPT-2-XL. The statement is true only at the median. Please give the three-part decomposition per model.
- **On T_×, the identity is not shown to beat the equilibrium shortcut.**
  - The paper's own records (Table F9 vs. Table F8) give a straight-line T_× ratio with a median of 0.73 relative to the equilibrium prediction r_C, against 1.32 relative to the identity-based intent.
  - The distances from one are similar on a log scale (|ln 0.73| ≈ 0.31, |ln 1.32| ≈ 0.28). For the two larger models the equilibrium model is closer (0.67 and 1.26, against 2.25 and 2.10).
  - So the identity's advantage holds for ln K, not for T_×. The paper should say this explicitly, or report the equilibrium model on interpolated crossings too.

### M5. The two-state and thermodynamic framing is not supported by the paper's own checks, yet it carries a large share of the text.

The paper's checks undercut the framing:
- The windowed fraction "did not meet a bimodality criterion we had fixed in advance."
- Exit hazards fall by a factor of about 3.5 with dwell time (median 0.28).
- ln K drifts by 1.3–2.1 between the two halves of a text.

That drift is comparable to the entire designed intervention: h*·ḡ is about 1 in ln K. T_× is therefore a length- and position-specific summary, and a target defined on it is fragile.

Despite this, the paper builds a large apparatus on the framing: the van 't Hoff slope, "crossover temperature", a "Clausius–Clapeyron" analogue, s_temp, κ, r_C, ρ_eq, and Appendix H. The core contribution, the identity plus a prospective test of Δln K, needs none of it. I recommend presenting K as a summary of switching statistics from the start, keeping T_× only as the design device, and moving the equilibrium comparison to a short appendix (see Section 4). This would not change the conclusions; it would make them easier to find.

### M6. "Where the penalty acts" rests on decomposing the predicted response, not on a measurement.

The abstract states: "The derivative also shows that the response arises mainly at the minority of steps where the model is uncertain."
- Fig. 3a shows shares of the predicted response, Σ_t Cov(O, c_t − q_t)/T split by q_t.
- Proportionality to the conditional variance q(1 − q) is the natural null: it holds if a unit of (c_t − q_t) has the same downstream effect regardless of q.
- The finding "median of 0.96" therefore says that the per-unit influence does not depend much on q. That is fine, but it is weaker than "where the penalty acts."

**Request.** Run the obvious intervention: apply the penalty only at steps with 0.1 ≤ q_t ≤ 0.9. Then test whether about 80% of Δln K is recovered, as the first-order decomposition predicts, and whether text quality at confident steps is spared.
- This would turn Sec. IV D from a description into a tested claim.
- It would give practitioners a concrete design: a confidence-gated penalty.
- If it fails, the abstract's last sentence must go.

### M7. Statistical points that need tightening.

1. **The population of the inference is not stated.** The decision statistic is a median over seven heterogeneous models, with a bootstrap over articles: "the bootstrap intervals resample articles, not models." The interval is about these seven models on these articles, not about base models in general. Wording such as "holds" in the abstract and Discussion should say this.
2. **The ratios lean above one, and the cause is unexplained.**
   - Six of seven replication ratios exceed 1.
   - The first test gave 1.17 at h* and 1.10 at 2h*.
   - The ratio did not grow from h* to 2h*, so this does not look like a second-order effect in h. It looks like a constant bias.
   - Appendix C mentions "a synthetic check in which the labeling raised the ratio by about 10%." Please explain the mechanism, show the synthetic check, and say whether the ratio should be corrected for it. This could shift the replication point estimate toward ~0.95 and does not threaten the decision, but readers need to know why an "exact" identity gives a biased ratio.
3. **Per-temperature agreement is poor, and the no-drift claim is overstated.**
   - Fig. S2b shows per-temperature identity ratios from about 0.46 to 3.6. F10 gives a median half-width of 39% of the prediction at single temperatures.
   - The claim "across temperatures the ratio did not drift systematically" rests on a slope change of 0.01 with interval (−0.446 to 0.161). That lower end sits at the edge of a ±0.45 band. This is absence of evidence, not evidence of no drift.
   - Please rephrase. Also show the per-temperature ratios of the replication in a supplementary panel, since summing over temperatures can hide compensating errors.
4. **The linearity claim rests on the median.** F3 states "the change in ln K stayed close to linear in h up to |h| ≈ 1.6". The only model that reached |h| ≈ 1.6 (GPT-2-XL, 2h* = −1.58) had a ratio of 1.32 (1.00–2.00) at 2h*, against 1.17 at h*. Please restate it as a statement about the median.
5. **Interval calibration is not shown.** No coverage check of the percentile bootstrap for this ratio-of-sums statistic is reported. The synthetic-data self-tests in Appendix E could provide one cheaply. This would not change conclusions unless coverage is poor.
6. **Two coupling questions need answers.** Did the unpenalized arm of the first test reuse the design runs' seeds and articles at overlapping temperatures (e.g. 0.40–0.65)? Seeds "depended on the temperature, the sample and the batch". If so, h* (which divides by ḡ) and the test's g share noise, and the first test is not independent of the design. Please state this. It would only affect the interpretation of the first test.

### M8. Statements that conflict, or that a reader cannot reconcile.

- **Preregistration of the design-run comparison.**
  - Sec. IV B: "These documents did not cover Fig. S1, whose runs were made earlier."
  - Table C1 lists "Shift of T_× on the design runs (Fig. S2a) | F7 | Preregistered", and F7 says "One comparison was preregistered," using the same h = ±0.3 design runs.
  - Please state whether R was registered before those runs were generated.
- **Which T_× is meant.**
  - Table I gives T_× = 0.57 for Pythia-1.4B, 0.48 for GPT-2-XL, 0.44 for Mistral-7B and 0.55 for Qwen2.5-7B.
  - Table F1 gives the test temperature "closest to T_×" as 0.62, 0.53, 0.49 and 0.60.
  - Fig. 1a uses "Pythia-1.4B at T = 0.62" as the near-crossover example, while the interpolated crossing for that model is about 0.577.
  - The difference is the straight-line fit on the test grid against the design fit, but the reader must reconstruct this. Say which T_× is used where.
- **Inside or outside the grid.**
  - F6, straight-line fit: "For GPT-2-XL and the two larger models, the unpenalized T_× lay outside the test grid and was extrapolated."
  - F6, interpolation: "Every arm of every model crossed zero inside its grid."
  - Both are true for different estimators. One sentence would prevent confusion.
- **Direction of the entry/exit discrepancy.** Sec. IV D says: "It put slightly more of the response on the exit rate than was measured". In Fig. 3b, Mistral-7B-v0.3 goes the other way: the identity's entry share (~0.43) lies above the measured interval (~0.33–0.42).
- **A rounding mismatch.** Appendix H says ρ × g/g_C differs from ρ_eq "by up to 0.07, for GPT-2-XL". From Tables I and H1: 1.08 × 0.98 = 1.058 against 0.98, a difference of about 0.08.
- **Unequal comparison standards.** Sec. IV B says the equilibrium model "would have fallen outside the range for six of the nine models". The identity is not judged per model; four of seven per-model intervals reach above 1.25. Compare like with like: the equilibrium median is 0.69 (0.64–0.72), which fails the median rule.

### M9. Positioning: the theory is standard; the contribution is empirical.

The identity is the likelihood-ratio gradient. Its decomposition is the Doob decomposition of N, with the compensator Q. The tilted ensemble written step by step is the standard Doob transform. The paper cites these sources, which is good. Even so, the Introduction and Sec. III read as if a new theoretical tool were being introduced, and the physics vocabulary (static susceptibility, escape-rate terms, Clausius–Clapeyron) inflates the apparent novelty.

I checked the key steps and found them correct:
- ∂ log p_t/∂h = (c_t − q_t)/T;
- E[N − Q] = 0;
- Q − E[Q] = Σν_t;
- the first-order expansion of the tilted log-weight;
- the multiplicative derivative −|z|;
- the invariance of the temperature score to history-dependent shifts.

The paper should present itself as an empirical study of when a cheap first-order estimator is accurate for decoding interventions, with the decomposition as interpretation.

ML related work to add: `no_repeat_ngram_size`; presence and frequency penalties (OpenAI-style); DRY-type sequence penalties; contrastive search. More of the literature on tuning sampling hyperparameters would also help. None of this changes conclusions.

### M10. Verification and authorship.

Appendix E states that the derivations "were likewise written by an AI assistant and checked by other AI instances, not by the author", and that the code was not checked line by line by the author. The disclosure is commendable, and the safeguards are good: an independent reimplementation that reproduced the ratios to four decimals, bit-identical reruns, and synthetic self-tests. Nonetheless, the author must be able to defend every derivation in Sec. III and Appendices G–H. An editor will hold the author responsible for them. See Section 7 for where I expect difficulty.

Separately, "Generated continuations are available on request" is not adequate for a paper whose claims are statistical functionals of those continuations. Please deposit the continuations, or at least the per-step (c_t, q_t, repeat mark) arrays and the four counts per continuation; prompts can be withheld for licensing. This would not change conclusions, but it is needed for verification.

## 4. What could be removed or moved, and which caveats are essential

**Could be cut or moved to an appendix without changing the conclusions:**
- **Appendix H and most of F7.** This covers r_C, ρ_eq on halves, the κ factorization, energies −z and T·(−log p), and Fig. S2. One paragraph in the main text suffices: "a two-state Boltzmann shortcut overstates the response by ~45% with U = −log p and gives the wrong sign with −z". The factorization g/g_C = (s_pen/s_temp)κ is an identity by construction ("its content lies in the sizes of the factors") and adds little.
- **s_temp and Table II's second column.** They exist only to support the equilibrium comparison. The temperature-identity check in Appendix B can stay as one sentence.
- **F6:** the quadratic fit, and the three-part decomposition (median version). Keep a per-model table only if M4 is addressed.
- **F8:** the "pooled" column and the mixed convention. One sentence is enough.
- **F10, third part** ("no rule changed any temperature"). One sentence is enough.
- **F9 (other genres).** It is preregistered, but it tests the equilibrium shortcut's R on design runs, not the main claim. Keep only the identity column and the statement that code could not be measured.
- **Table F1's per-bin densities and the bimodality criterion.** Keep one sentence and the figure.
- **The instruction-tuned test.** One sentence.
- **Repetitions.** The bit-identical GPU reruns appear twice (Appendix A and E), and the definition of repeat candidates appears twice (Sec. II and Appendix B).
- **"The identity also holds with a bounded stopping time …, but we did not use one."** Delete.
- **Table C1** can stay as a ledger, but in the supplement.

**Caveats the reader really needs, ideally all in one limitations paragraph:**
1. g came from the same test's unpenalized arm, so the test checks first-order accuracy at h*. The genuinely out-of-sample prediction from design data (F4) is right at the median but off by up to ~30% per model (GPT-2-XL 1.29, Qwen2.5-7B 1.19, BLOOM 0.86).
2. Decisions are on the median (or the mean of two) over models; per-model results are often undecided.
3. The multiplicative penalty is validated only below practical strengths.
4. The intended shift of T_× was missed by factors of 0.5–2.4 for several models.
5. The tests used a single genre and language, base models of at most 7.6B parameters, plain sampling at low T with no truncation, and EOS masked.
6. The states are a labeling with strong drift and non-Markov persistence.

## 5. Undefined terms, jargon and note-like passages

**Undefined or underdefined:**
- "compensator" (defined in passing in Sec. III A, but the Doob–Meyer sense is assumed);
- "static susceptibility of the path ensemble";
- "finite-horizon Doob transform";
- "escape-rate term in response theory for jump processes";
- "if one slow mode dominated both responses" (App. H; "slow mode" is never defined);
- "independent units of m steps" and "free-energy difference per unit" (App. H);
- "the analogue of the Clausius–Clapeyron relation";
- "label bias" (cited but not explained for this setting);
- "first-order quantities, taken at zero penalty and zero tilt".

**Internal vocabulary a reader must memorize:** design temperature, design articles, replication articles, usable temperatures, "record", "rule fixed", ρ, ρ_out, ρ_eq, r_C, R, κ, s_pen, s_temp, g, g_N, g_Q, g_U, g_C, g_C^tilt, b, b_tilt, ḡ, ḡ_λ.

- A notation table is essential.
- Three token sets are used: repeats (the 4-gram marks), repeat candidates (the 2-token context), and seen tokens (multiplicative). They should be contrasted in a single small table.
- The symbol c_t (candidate indicator) and the notation Δc_T (difference of candidate rates between states) are too close.

**Note-like or list-like passages:**
- Most of Appendix F reads as an analysis log ("This analysis was exploratory. … Before computing, we called … "). "They answer three questions raised in review" (F10) refers to a previous review process and should not appear in a manuscript.
- The last two paragraphs of the Discussion are a sequence of one-line caveats rather than an argument.
- The abstract is a chain of facts without a single message.
- Sec. III C defines s_temp in one dense sentence: "Over several temperatures we sum numerator and denominator with the weights of the equilibrium prediction g_C of Sec. III D: s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u)". It uses quantities that are only defined in the next subsection.

## 6. Checklist

- **Large promise, small experiment.** Yes.
  - The title and the first sentence of the Discussion generalize to "repetition penalties". The support is an additive penalty not offered in libraries, plus one multiplicative penalty validated only at θ ≲ 1.1.
  - The Discussion's extension to "logit biases and watermarking biases" and to min-p and top-p is untested on language models. Appendix G is a 5-token, 5-step toy, and the paper says so ("Nothing in this appendix has been tested with language models").
  - The Introduction's statement that s_pen "explains why the effect is small" is a large interpretive claim on a single statistic.
- **Single run, model or dataset.**
  - Not a single model: nine base models were used.
  - A single dataset for all registered tests: BBC News, April–June 2025, English. The other genres were run only as design runs.
  - Each prospective test was run once, with one seed scheme and four samples per article. "The additive, multiplicative and larger-model tests share the same 120 articles".
- **Parameter choices without a stated reason or time of choice.**
  - The paper is better than most here: most choices were fixed in dated documents, and many labelings were checked.
  - No reason is given for: the target shift of 0.10 in T_×; four samples per prompt (which also limits the within-article covariances to 3 degrees of freedom and the importance-sampling analysis); the 64-token prompt; the 4-gram definition of a repeat; the T_× acceptance window 0.3–0.8; the θ* window 1–3; the "at least five temperatures" rule; the requirement that "Articles shorter than 576 tokens … were not used" (why does the article need 576 tokens when only 64 are used as a prompt?); and 35 rather than 40 code files in F9.
  - The 512-token length has a stated time ("fixed in the first experiments of the project") but no reason, and s_pen depends on it.
  - The Fig. S1 runs were made before the documents.
- **Computational cost.** Partly reported. There are about 22 L4 GPU-hours for the replication and about 6 A100 GPU-hours for the second stage of the larger-model test. There is no total, and no figures for the design runs, the first test, the four multiplicative strengths, the instruction-tuned runs or the genre runs. The claim "comparable in cost to trying a few penalty strengths" is not quantified; see M1(a) and M1(c).
- **Broad conclusion from a narrow case.** Yes.
  - The abstract's "where the penalty acts" comes from a decomposition of the prediction (M6).
  - "In our runs" in the 99% sentence is appropriate, but the Introduction drops that qualifier.

## 7. Could the author defend each design choice? Where they would struggle

Choices the paper defends well:
- summing over temperatures before dividing;
- the median so that "no single model could decide it";
- the log-symmetric band and the planned width of ~0.2;
- masking EOS for fixed length;
- the 2-token context for candidates, so the penalty acts at the step where a repetition would start and states stay distinct from penalized events.

Where I expect the author would struggle if asked by an editor:

1. **"Why ln K and T_×, rather than a repetition metric anyone uses?"** The honest answer appears to be the physics analogy. The paper's own checks then undercut the analogy (M5).
2. **"What intervention is exp(hN/T), and why should a practitioner care that 99% of it cancels?"** See M3. I also expect difficulty explaining, without notes, why Cov(O, Q) = Cov(O, Σν_t) and what ν_t is operationally.
3. **"Why should g come from the same test's unpenalized arm rather than the design data?"** And: "does your test distinguish your per-model derivative from a constant?" (M1).
4. **"Why does the labeling raise the ratio by ~10% in a synthetic check, and why are 6 of 7 ratios above one?"** (M7.2).
5. **"Why use the mean over two models for the larger-model test, and what population does an article bootstrap over seven fixed models generalize to?"**
6. **"Why is the multiplicative penalty linear only to θ ≈ 1.1, and why is GPT-Neo's ḡ_λ 13–22 times smaller?"** The answer "the divisor acts on raw logits" is a start. A plot of the logit-magnitude distribution of seen tokens per model would settle it.
7. **The Appendix H model:** what "independent units of m steps" are, why b = mΔE, and what κ ≈ 1 would mean physically. These are the derivations the author states were not personally checked. I would expect an editor to probe exactly here.
8. **Why the importance weights collapse while the first-order prediction holds.** This is the most interesting question in the paper, and the answer is currently "attributed to scale, without having tested it." A pooled self-normalized estimator across articles, which the paper "did not try", would answer it directly.

## 8. Minor issues

- **Citation style** mixes bare arXiv identifiers ("[2206.02369]") with author–year ("[Glynn 1990]"). Bare arXiv numbers are unreadable in running text; use one style, with names.
- **Abstract:** "the derivative of a text's log-probability with respect to h is (N − Q)/T" should say "the log-probability of a continuation under the penalized sampler".
- **Sec. II:** "Prior to division by T" and "before division by T" are both used; pick one.
- **Sec. II:** "We call one round of computing the next-token logits … a step" should come before the first use of "step" in the abstract and Sec. II.
- **Sec. IV B:** "A separate test of two 7-billion-parameter models": one has 7.6B and the other 7.2B; "two ~7B models" is fine.
- **Sec. IV B:** "For comparison, the tilted response of Sec. IV C would give a ratio of about 0.01": explain that this is ρ·s_pen, or the reader may expect ~100.
- **Sec. IV D:** "This proportionality is an empirical finding, not a consequence of the form of the score." State the null explicitly (equal downstream influence per unit of c_t − q_t).
- **Sec. V:** "The first-order prediction held at penalties with |h*|/T* between 0.7 and 2.1": BLOOM's value is 0.37/0.56 = 0.66; write "0.66–2.1" or "about 0.7".
- **Fig. 1a:** "chosen as the first output with two to six switches"; the panel shows exactly two. Fine, but "first" in what order?
- **Fig. 1b** caption: say that the y-axis is the fraction of windows per value of k/32.
- **Fig. 2:** the first-test rows show only the medians. Mark them visually as undecided (both upper ends 1.298 and 1.299), since the grey band makes them look inside.
- **Fig. 3b:** the dashed line at 0.5 is not explained in the caption.
- **Fig. S2b:** the y-axis reaches 3.6. Add per-model colouring, since "the points within a model are not independent."
- **Table I** caption: "(before division by temperature)" is repeated from the text. The caption is long; move definitions to the text.
- **Table F2:** the note that intervals differ "because this analysis used a different bootstrap seed" invites the question why a fixed seed was not reused. Reuse it.
- **Table F5:** ρ_out/ρ for SmolLM2 is listed as 0.95 (1.13/1.18 = 0.958) and for Qwen2.5-1.5B as 1.07 (1.02/0.94 = 1.085). Presumably rounding of unrounded values; state that the ratios use unrounded values.
- **Table F7:** the θ range at k = 8, "1.17–1.53", against θ* = 1.019: 1.019^8 = 1.162. Again rounding; fine if stated.
- **App. A:** "Weights were loaded from the main branch … The latest commits of all nine repositories predate our runs, so the runs used these revisions". The instruction-tuned models' revisions are not listed.
- **App. B:** "Covariances were population covariances pooled" vs. App. B "Exponential tilting": "divisor S − 1". Two conventions in one appendix; say once which is used where.
- **App. C:** "rounded to four decimals for generation", while Table I shows two decimals; fine, but the four-decimal values should be in the supplement.
- **App. F2:** "which takes at least nine steps": show the arithmetic (from 16/32 to below 8/32 needs at least nine non-repeat steps entering the window). Readers will check it.
- **App. F10:** "which a rule fixed beforehand read as no drift beyond that width" is awkward. Suggest: "which, by the rule fixed beforehand, counts as no drift larger than the width of the decision range."
- **App. H:** "U is singled out among energies only because the probability of a whole text at T = 1 is exactly exp(−U)". This is a good sentence; move it to the main text if the equilibrium comparison stays.
- **Typography:** "van 't Hoff" is correct throughout; "ln K" and "lnK" are mixed in figure labels and text ("ln K" in Fig. 1c is fine).
- **Data and code availability:** "archived as version 1.0" is good. Add the per-step arrays (M10).

## 9. Recommendation: major revision

The paper is careful and unusually transparent. The analysis plans and hashes were fixed in advance, the inconclusive first test is reported honestly, several labelings were checked, there is an independent reimplementation, and the limitations are stated rather than hidden. The core identity is correct, and the empirical finding that Δln K stays close to first order at |h|/T up to ~2 is useful. I would like to see it published in some form.

In its current form, however:
- **(i)** The registered test cannot separate the per-model derivative from simple baselines, and it decides only on a median over models (M1).
- **(ii)** Relevance to decoding practice is not established. The tested additive penalty is not used as such, the library penalty works only below practical strengths, presence and frequency penalties and instruction-tuned models are untested, and the target statistic is not one practitioners use (M2).
- **(iii)** The headline 99% cancellation is framed against an intervention no one applies, and its interpretation ("explains why the effect is small") is not supported (M3).
- **(iv)** Failures on the user-facing T_× target are understated (M4).
- **(v)** A large part of the text serves a two-state and thermodynamic framing that the paper's own checks contradict (M5).

**Requests most likely to change the conclusions:**
- the baseline comparisons and per-model verdicts (M1);
- one prospective test of the presence penalty or a standard repetition metric (M2);
- the confidence-gated penalty experiment (M6).

**Requests that change presentation rather than conclusions:** the reframing of s_pen, the cuts in Section 4, and the notation table.

With these, the paper would make a clear, practically useful claim of moderate scope: a cheap first-order estimator predicts how context-dependent logit penalties change repetition dynamics, together with the regime in which it does so. That claim would suit this venue.