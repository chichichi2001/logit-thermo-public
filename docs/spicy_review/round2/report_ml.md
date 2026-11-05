# Referee report

**Manuscript:** "Predicting how a repetition penalty changes switching into and out of repetitive text, from unpenalized samples"
**Referee expertise:** machine learning for language models (decoding and sampling, repetition, evaluation of generation)

---

## 1. Summary in my own words

**Question.** Can the effect of a repetition penalty on how generated text moves between repetitive and ordinary stretches be computed in advance, using only samples generated without the penalty?

**What the author did.** The author defines an additive logit penalty h on "repeat candidates", the tokens that previously followed the current last two tokens. The likelihood-ratio (score-function, REINFORCE-type) derivative of a text's log-probability at h = 0 is (N − Q)/T. Here N is the number of candidates chosen and Q is the sum of the per-step candidate probabilities q_t. Steps are labeled repetitive or ordinary with a 32-token window and hysteresis. From the labels the author forms K = entry rate / exit rate, and from the score the first-order response g = ∂ln K/∂h. From the design runs the author computes the h* that should lower the crossover temperature T_× (where K = 1) by 0.10. A preregistered test then compares the realized Δln K at h* with h*·g(T), where g is re-estimated from the unpenalized arm of the same test. Nine base models (0.4B to 7.6B parameters) continue BBC news articles at T ≈ 0.2 to 0.9, with no truncation, EOS masked and 512 tokens per continuation.

**Main result.** On 120 new articles, the median ratio over seven smaller models was 1.04 (0.98–1.15), inside the preregistered band 0.8–1.25. The mean over two 7B models was 0.96 (0.84–1.11). Per-model intervals mostly do not fall inside the band. A secondary, exploratory result: the step-wise response is about 1% of the response under whole-text exponential tilting (s_pen = 0.7–1.5%), and s_pen·τ ≈ 1, where τ is the switching relaxation time. Two further results: the multiplicative (HF `repetition_penalty`) prediction holds only at weak divisors (θ ≈ 1.02–1.11), and the targeted shift of T_× missed by a factor of 0.5 to 2.4 per model.

## 2. Most interesting finding, and whether the abstract leads with it

For an ML reader, the most interesting and best-supported finding is this. A linear extrapolation from unpenalized samples predicts the change in an aggregate switching statistic at penalties where |h|/T ≈ 0.7–2.1. That change is large: by my arithmetic from Table I and Table F5 (h*·ḡ), Δln K ≈ −0.9 to −1.6, so K falls by a factor of about 2.4 to 5. The ratio barely changes at 2h* (Appendix F3: 1.17 → 1.10). At these strengths, whole-text importance weights are already degenerate (Var log w ≈ 3–25). This is a locality result: the penalty acts through a window of roughly one relaxation time, not through the whole path. It is useful to practitioners. A close second is Fig. 3a: about 80% of the predicted response comes from the roughly 10% of candidate steps with 0.1 ≤ q_t ≤ 0.9, the decision points, not from steps inside loops.

**Neither is in the first two sentences of the abstract.** The first two sentences state the setting and the question ("A repetition penalty acts … renormalized each time. We ask whether its effect … can be computed from samples generated without it."). The answer (1.04) comes in sentence 6. The effect size (K changes about threefold) is never stated; the abstract only says "The penalties were strong enough to shift that ratio". The 99% cancellation, which the abstract presents as an explanation ("The same score explains why the effect is small"), is in my view the weakest-supported claim in the paper (see 3.4).

## 3. Major concerns

### 3.1 The title and abstract promise prediction; the registered test checks linearity

The author is admirably explicit that "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test. The test therefore asks whether the response is still first order at h*." The title ("Predicting how a repetition penalty changes …") and the abstract's framing ("We ask whether its effect … can be computed from samples generated without it") still read as an out-of-sample prediction. That prediction (Appendix F4, ρ_out) is:

- exploratory, not preregistered;
- undecided for the larger models (1.07, 0.91–1.35);
- not distinguishable from a derivative borrowed from other models: "With each model's ḡ replaced by the mean ḡ of the other six smaller models, the median ratio was 1.09 (0.99–1.21) … their difference, 0.04 (−0.10 to 0.11), did not exclude zero."

The last point matters most to a practitioner. At the level the paper tests (the median over models), a single model-agnostic constant ḡ ≈ 2.5 does as well as the model's own score. The paper states this in Sec. IV B and V, but the abstract does not. **Request:** say in the abstract that the registered tests check linearity, and that the model-specific value of the derivative was not demonstrated. *This would not change the conclusions; it aligns the framing with them.*

### 3.2 Practical relevance of the penalized quantity and of the penalty itself

- **The additive penalty.** The paper says "This additive penalty is not an option in common libraries." Its candidate set is, however, exactly the set that `no_repeat_ngram_size=3` bans: tokens that would complete a previously seen trigram. The tested penalty is therefore a soft version of a very widely used hard constraint (h → −∞), which first-order theory cannot reach. Separately, the presence penalty of commercial APIs is an additive logit shift on the set of tokens already seen. Its score has exactly the form (c_t − q_t) with a different set, and the frequency penalty is an additive shift proportional to counts. The paper says "The widely used presence and frequency penalties were not tested." Since the identity applies to them directly and the prediction needs only rescoring of existing unpenalized runs, testing at least the presence penalty is the cheapest way to make the result relevant to practice. *This could change the scope of the conclusions (positively or negatively). It would not change the core result.*
- **The multiplicative penalty works only below practical strengths.** "The divisors of 1.1–1.3 common in practice correspond, for these six models, to 1.8 to 14 times the predicted ln θ*", and at k = 4 the test was undecided (0.86, 0.77–0.99) and at k = 8 below the band (0.52). Most of the practical range is therefore outside the confirmed regime. The abstract says "only below the strengths common in practice", while Sec. IV B says "reach only the lower end of the values of 1.1 to 1.3". Please make these consistent; at k = 2 only GPT-2-XL reaches 1.11.
- **The statistic.** ln K under a hysteresis labeling with a 15-token look-ahead is not what users measure. Users measure rep-n or seq-rep-4 rates, distinct-n, and the fraction of outputs that end in a loop. The one practical-facing check (fraction of repeats, F4) passed overall. Its end-of-text version failed: "For the fraction in the last 64 steps alone, the median for the smaller models was 1.23 (1.11–1.38), with its upper end above the range." Loops at the end of a text are what users experience as degeneration, so this belongs in the main text next to "The first-order response also predicted the change in the fraction of steps marked as repeats". *Moving it to the main text would not change the conclusions, but it qualifies the claim that the prediction "carried over to the fraction of repeats".*
- **The regime.** The confirmed regime is narrow: base models of 7.6B parameters or fewer, T between 0.2 and 0.7, no top-k/top-p, EOS masked, a forced length of 512, and news text. The instruction-tuned test was undecided (Qwen) or could not be run (SmolLM2-Instruct, T_× = 0.16). Repetition penalties are most often used with greedy or beam decoding, or with chat models under truncation. The score-function approach does not exist for deterministic decoding, and this limitation is not stated. The introduction says repetition "persists in current systems", but no current-scale or instruction-tuned system was confirmed. **Request:** state the greedy-decoding limitation and narrow the introduction's framing. *This would not change the conclusions.*

### 3.3 Statistical inference over models

- The bootstrap resamples articles, not models ("the bootstrap intervals resample articles, not models", Sec. V). The interval 0.98–1.15 therefore describes article noise for these seven fixed models. It is not an interval for "a model". The per-model points (0.94–1.18) suggest that a model-level interval would still lie in the band, but this should be shown, for example with a leave-one-model-out median or a two-level bootstrap. *Unlikely to change the conclusions.*
- Per-model results are mostly undecided, not failed, because per-model intervals (width ≈ 0.3–0.45) are about as wide as the band (0.45). One exception deserves mention: SmolLM2-1.7B, 1.18 (1.02–1.42), excludes 1. In the first test, Pythia-1.4B was 0.75 (0.62–0.95) and moved to 1.03 (0.92–1.20) in the replication, two barely overlapping intervals on different article sets. Per-model biases of ±20% that are unstable across article sets suggest that between-article heterogeneity is larger than a 40-article bootstrap shows. Please comment.
- "For the two larger models we used the mean": the median of two numbers is their mean. Say so and drop the apparent asymmetry.
- The band 0.8–1.25 is generous for a first-order test. It was "set before the first test, from the agreement already seen on the design runs at h = ±0.3, with medians of 1.00 and 0.96". This is defensible, but the paper should state what the test can and cannot detect. A second-order correction of up to about 20% passes. The score function can also give the second derivative from the same unpenalized samples. The paper states "We did not estimate the second-order term", yet doing so would directly address whether the first test's 1.17 reflects curvature or bias in ĝ. *Would not change the conclusions; would strengthen "first order".*
- Summing over temperatures can hide errors that cancel. The F10 drift test passes only on the boundary ("−0.446 … just inside ±0.45 … this result does not exclude a drift of that size"). The threshold itself is lax: a ratio going from 0.8 to 1.25 across the grid would pass. Fig. S2b shows per-temperature ratios from about 0.5 to 3.6. Table C1 summarizes F10 as "No drift beyond the range width", which overstates the F10 text. Please align them.

### 3.4 The "99% cancellation" and the 1/τ explanation are weakly supported

- **What s_pen compares against.** g_N is the response to whole-text tilting by exp(hN/T), which no decoder implements. Across texts, N is dominated by the time spent in the repetitive state (candidate rate ≈ 94% vs ≈ 19%), whereas N − Q is a martingale with conditional variance Σq(1−q). Cov(O, N) ≈ Cov(O, Q) is therefore largely a consequence of Var(N) ≫ Var(N − Q). The sentence "The same score explains why the effect is small" will confuse ML readers, because the effect is not small in practice (K changes about threefold). It is small only relative to a hypothetical global reweighting. The equilibrium shortcut, which a reader might take as the "Boltzmann intuition", is off by about 1.45×, not 100× (ρ_eq = 0.69). The abstract should say what "small" is relative to, or drop the phrase.
- **The CV < 0.3 criterion cannot discriminate.** "Before computing, we had fixed that a coefficient of variation of s_pen τ below 0.3 across the smaller models would count as agreement with this reading. It was 0.14." From Table F11 and Table II, the CV of s_pen alone across the seven smaller models is about 0.17, and the CV of τ alone is about 0.09 (my calculation). Any roughly constant τ would therefore have passed: the criterion cannot tell s_pen ∝ 1/τ from s_pen = constant. The author's own rank correlation of 0.11 between s_pen and 1/τ confirms that the data carry no information about the 1/τ dependence across models. The honest statement is that s_pen ≈ 1% and τ ≈ 100 steps have reciprocal orders of magnitude.
- **Dependence on the labeling.** τ depends on the labeling (window, hysteresis), while "We did not estimate … s_pen under other labelings". The coincidence s_pen·τ ≈ 1 is therefore untested against the one choice most likely to break it.
- **Requests.**
  - (a) A stationary two-state Markov-chain calculation or simulation giving the expected s_pen·τ, so that 0.9–1.5 is compared with a prediction rather than with "≈ 1".
  - (b) A within-model test that varies τ, for example across temperatures or window lengths, and checks whether s_pen follows 1/τ.
  - (c) Otherwise, move the 1/τ reading to the Discussion as a conjecture and remove it from the abstract.

  *Request (b) could change this secondary conclusion. None of these changes the main test.*

### 3.5 The importance-reweighting comparison is overstated in the abstract

The abstract says: "at them reweighting the unpenalized texts by their exact probability ratios had already failed." The evidence (Appendix D) is a per-article effective sample size with four samples ("this estimate is itself noisy"), and "We did not try self-normalized weights pooled over articles." With 480 texts per temperature and Var log w ≈ 3.4 at the low end, pooled self-normalized IS would retain an ESS of order 480·e^(−3.4) ≈ 15. That is poor but not obviously a failure for every model. Sec. V also states that the explanation of the contrast was "without having tested it further." A cheap test is available from the stored per-step q_t: compute the second-order variance of the log-weight, (h/T)²Σ q(1−q), restricted to windows of about τ steps. If this is O(1) or smaller while the whole-text variance is 3–25, the locality explanation gains direct support. **Requests:** (a) soften "had already failed" to "per-article reweighting was degenerate"; (b) run the windowed-variance check. *(a) would not change the conclusions; (b) could support or refute the stated explanation.*

### 3.6 The design target (T_×) and the tested quantity (Δln K) differ

h* was designed to lower T_× by 0.10, but the decision is on Δln K. The T_× shift missed its target: 1.32 by straight-line fit and 0.87 by interpolation for the median, with per-model values from 0.51 to 2.41 and "No per-model interval of the interpolated ratio lay inside 0.8–1.25". The grid was centered on T* = T_× − 0.10, so the unpenalized crossover sits at the top edge of the grid, and for GPT-2-XL and both 7B models outside it ("was extrapolated"). Sec. II also says "Near T_×, ln K is close to linear in 1/T", but F6 reports that ln K lies "0.16–0.34 above the straight line" at the grid center in every model.

The post hoc explanation ("the spread was already visible in the design data") holds for GPT-2-XL. It does not hold for BLOOM-1.1B: the measured design slope × h* implied 0.97, yet BLOOM undershot to 0.51 (0.39–0.63). The text attributes this to g(T_×)/ḡ = 0.83, a second diagnostic that disagrees with the first for the same model. "Visible in the design data" is therefore too strong; it was visible for some models. *Rewording would not change the conclusions, but practitioners should be told plainly that the user-facing target was not controlled per model.*

### 3.7 Internal contradictions

1. **Who checked the derivations.** Sec. V says: "They were checked by tests, an independent reimplementation and other AI instances, and the author has since gone through the derivations of Sec. III (Appendix E)." Appendix E says: "The derivations of Sec. III and Appendices G and H were likewise written by an AI assistant and checked by other AI instances, not by the author." These directly contradict each other and must be reconciled. Given AI-authorship policies, the corrected statement matters for accountability.
2. **Whether the events are distinct.** Sec. II says "Repeats define the states below and candidates define the penalty, so that the definition of the states does not depend on the penalty", and Appendix C says the candidate choice "keeps the events that define the states distinct from the events the penalty acts on". But "Every repeat is also a chosen candidate". The events are nested, not distinct. The stated rationale also does not follow: the repeat marks would not depend on the penalty under either definition.
3. **Which bootstrap decisions were checked.** Sec. IV B says "Neither decision depended on the number of bootstrap resamples…", while Table C1 says "Unchanged in all four". Both are true, but the main text refers to two decisions and the table to four.
4. **Drift test.** Table C1 says "No drift beyond the range width" while F10 says "does not exclude a drift of that size" (see 3.3).
5. **Rounding in the ρ_eq check.** Appendix H says ρ × g/g_C "differs from ρ_eq by up to 0.07, for GPT-2-XL". From the displayed values, 1.08 × 0.98 = 1.06 against ρ_eq = 0.98, a difference of 0.08. This is probably rounding; please check.

### 3.8 Timing of some "rule fixed" thresholds

For F8, "we called s_pen 'about 1%' if the interval of the median lay within 0.5–2%." The baseline s_pen (Table II, 1.16%) was already known as a record. A fourfold band around a known value has little power to fail; the mixed-convention and length variants are near-certain to land in it. The same applies to the bimodality threshold ("below one fifth"), the CV thresholds 0.3 and 0.6, and the drift threshold ±0.45: no reasons are given. Please state for each exploratory rule what was known when it was fixed and why the threshold was chosen. *Would not change the main conclusions.*

### Arithmetic checks

I recomputed from the tables: T* = T_× − 0.10 for all nine models; the medians and means in Tables I, F4, F5, F6 and F7; ρ_out/ρ; s_pen·τ; the CV of 0.14; g/g_C = (s_pen/s_temp)κ; the multiplicative divisor ranges (1.8–14×); |h*|/T* (0.66–2.08); and the factor of 6 to 143 in w. All agree with the text, apart from the rounding issues noted above. Fig. S1 slopes read off the plot match Table F10 within about 0.02.

## 4. What could be removed or moved, and which caveats are essential

**Could be shortened or moved without changing the conclusions:**

- **The equilibrium two-state model** (Sec. III C, the Sec. V paragraph, Appendix H, Table H1, Table F9 with r_C and the "T_× ratio", F7 halves, Fig. S2, the three-energy comparison). It is a long rebuttal of a shortcut that ML readers do not use. One paragraph in the Discussion plus Appendix H with Table H1 would suffice. The quantities s_temp, κ, b_tilt, g_C^tilt, r_C and ρ_eq add about six symbols for a secondary point.
- **Fig. 3b** and the entry/exit-split difference (F10, second part) can go to the appendix. "About half and half" can stay as one sentence.
- **Appendix C robustness details** (five seeds, 10,000 resamples, BCa intervals, the pooled 160-article "record") can become one sentence. The pooled record "was not used for any decision" and can be deleted.
- **F6 "Penalized ln K at the design temperature"** (the three-part decomposition) can be cut to one sentence; its conclusion ("does not come from a misprediction of ln K at T*") is all the reader needs.
- **The F10 third analysis** (other rules for usable temperatures) can become one clause, since no rule changed any temperature.
- **The bimodality machinery** (Table F1 low/middle/high/middle-to-side columns) can be replaced by one sentence; the Sec. II caveat suffices.
- **The note under Table I** about 0.796 and 0.799: show three decimals or drop the note.
- **The bounded-stopping-time sentence in Sec. III A**: "The identity also holds with a bounded stopping time … but we did not use one." It is not used.
- **The main-text exploratory numbers.** Sec. IV B, IV C and IV D carry many exploratory figures (design-only per-model accuracy |ln ρ|, the slope comparison, 0.96 (0.91–0.98) for the variance proportionality). Most can go to the appendices.

**Caveats a reader really needs, ideally in the abstract or the first page:**

1. The registered tests estimate g from the same runs, so they test linearity. The out-of-sample prediction is exploratory and, at the median level, not distinguishable from a derivative shared across models.
2. Agreement holds for the median over models. Per-model tests are underpowered, and the intervals do not resample models.
3. The multiplicative penalty is confirmed only below practical strengths.
4. The T_× target was not controlled per model.
5. The regime: EOS masked, no truncation, base models, news text, 512 tokens.
6. K is a non-stationary summary (F2), not a two-state rate ratio.

## 5. Undefined terms, jargon, and passages that read as notes

**Undefined or underexplained for an ML audience:**

- **"compensator."** Defined in passing in Sec. III A ("Q is its compensator"). For ML readers, say "the expected number of candidate choices given the history (the normalizer's contribution to the score)".
- **"Doob transform", "tilted ensemble", "static susceptibility of the path ensemble".** Used in Sec. III B with no ML gloss.
- **"escape-rate term" and "entropic and frenetic parts".** Sec. V: "The further split of that response into entropic and frenetic parts uses time reversal, which autoregressive sampling lacks." "Frenetic" is never defined, and the sentence serves no purpose in the paper; delete it.
- **"van 't Hoff plot" and "Clausius–Clapeyron".** Acceptable in a physics venue, but one clause each ("the slope of ln K against 1/T") would help.
- **"the identity" as a noun for the prediction.** Used in "the identity's share", "the identity's ratio", "Identity" columns in Tables F10 and F12, and Fig. 3b's legend ("identity"). The identity is exact; what is tested is the first-order prediction built from it. Call it "first-order prediction" throughout.
- **"this reading" / "the reading"** (Sec. IV C, F8). It is never named precisely.
- **"tilted response".** Used in Sec. IV B ("the tilted response of Sec. IV C") before Sec. IV C.
- **"record", "rule fixed", "design articles", "usable temperatures".** Defined, but they accumulate. A reader must hold about 20 symbols (Table A1 plus s_temp, κ, b_tilt, g_C, g_C^tilt, r_C, ρ_eq, ρ_out, R, ḡ_λ, Δc, ΔE).

**Passages that read as notes or lists rather than prose:**

- F1: "In the repetitive and the ordinary state it was Pythia-410M 94% and 20%, Pythia-1.4B 94% and 20%, …" belongs in a table.
- Sec. V, limitations paragraph ("All prospective tests used … Texts had 512 tokens … The widely used presence and frequency penalties were not tested … The two states are a labeling of a continuum …") is a list of disconnected sentences. Group the items into scope limits, inference limits and labeling limits.
- Sec. IV B, paragraphs 2–4, compress three tests, robustness checks, a design-only analysis, the own/shared comparison and the repeat fraction into dense runs of numbers; the reader loses the thread.
- F6 "Why the shift missed" and "Penalized ln K at the design temperature" are numeric logs more than arguments.
- Appendix E reads as a process log. That is acceptable, but it must be made consistent with Sec. V (3.7).

## 6. Patterns of overreach

- **Large promise, small experiment: partly.** The title promises prediction of how a penalty changes switching, but the registered test covers one statistic (ln K), one labeling family, one domain (BBC news, April–June 2025), one replication of 120 articles, and a test of linearity, not prediction. The abstract's explanatory claim ("The same score explains why the effect is small") rests on a criterion that cannot discriminate (3.4).
- **Single run, model or dataset: yes, for dataset and replication.**
  - Every prospective test uses BBC news. The other genres (F9) were run only as design runs on four models, with 40 prompts each, and tested the equilibrium model, not the prospective prediction.
  - The additive, multiplicative and larger-model tests share the same 120 articles (acknowledged).
  - There is one seed path per configuration (shared seeds), and four samples per article.
  - The confirmation rests on one replication after an undecided first test (acknowledged, and handled well: "no further replication was planned").
- **Parameter choices without a stated reason or time:**
  - With reasons given: the 2-token candidate context, 512 tokens, the band 0.8–1.25, and the persistence cut-offs.
  - Without reasons:
    - the 32-token window and the 1/2 and 1/4 thresholds;
    - 4-grams for repeats;
    - 64-token prompts;
    - four samples per prompt (this choice matters for within-article covariances and for the per-article importance-sampling conclusion);
    - the 0.10 target shift;
    - the minimum of ten switches;
    - centering the grid on T* rather than between T* and T_×;
    - the bimodality, CV, drift and "about 1%" thresholds (see 3.8 on timing).
- **Computational cost: partly reported.** Appendix E gives about 22 L4 GPU-hours (replication, smaller models) and about 6 A100 GPU-hours (second stage, larger models). There is no total, and no cost for the design runs, first test, genre runs, multiplicative and instruction-tuned tests, or rescoring. The claim that prediction is "comparable in cost to trying a few penalty strengths" is not quantified. Trial and error at one temperature needs a few penalized runs, while the prediction needs unpenalized runs at seven temperatures.
- **Broad conclusion from a narrow case: in places.**
  - Sec. V: "The derivative of Sec. III A applies to any small shift added to the logits of a set of tokens, such as logit biases and watermarking biases". This is true as mathematics, but the adequacy of the first-order prediction was shown only for the two penalties. The same applies to truncated sampling, which was checked only in "a toy model with five tokens and five steps".
  - The introduction places the work against "current systems", but no instruction-tuned model was confirmed.

## 7. Could the author defend each design choice? Where they would struggle

**Defensible as written:**

- the ratio of sums over temperatures (Appendix C gives the reason);
- the median over models;
- the 2-token candidate context;
- EOS masking for comparability;
- common random numbers (and the twofold narrowing they bring);
- the equivalence-style decision rule.

**Where an author would most likely struggle if asked:**

1. **Why lower T_× by 0.10 and then test Δln K?** The design target and the tested quantity differ, and the target was missed per model. A practitioner would ask why the paper did not target a repetition rate.
2. **Why this additive penalty rather than the presence penalty**, which is additive, widely deployed and has the same score form? And why not relate it to `no_repeat_ngram_size=3`, its hard limit?
3. **What practical question does s_pen answer?** No decoder implements whole-text tilting. And why does a CV below 0.3 test s_pen ∝ 1/τ when s_pen alone already has a CV of about 0.17?
4. **Why center the test grid on T*,** which leaves the unpenalized crossover at or beyond the grid edge, the region the paper itself identifies as where the straight-line fit is least accurate?
5. **Why four samples per article?** This choice drives the "reweighting had already failed" conclusion, and pooled IS was not tried.
6. **Why the median and an article-only bootstrap** when the claim is about models?
7. **What does a first-order test with a ±25% band rule out?** In particular, why not estimate the second-order term, which the same score machinery provides?
8. **Derivations.** Given 3.7(1), the author may be asked to derive g = (d₀₁ − d₀) − (d₁₀ − d₁) and the Doob-transform statement in Sec. III B unaided.

## 8. Minor issues

- **Fig. 1a.** "chosen as the first output with two to six switches": the example shows one short repetitive stretch (about 35 steps), far shorter than τ ≈ 100. A more typical example would better illustrate "Repeats cluster into long stretches".
- **Fig. 1c.** Only Pythia-1.4B is identified; label the models or state that the grey lines are unlabeled. Several curves have only one or two points below zero, which visibly shows the edge problem of 3.6.
- **Fig. 1b.** State that the 33 values k/32 are the bins.
- **Fig. 3b.** The legend "identity" should read "first-order prediction".
- **Fig. S2a.** The seven points are unlabeled.
- **Table F12.** "0.31 (−0.02–0.90)": a dash after a minus sign is hard to read. Use "−0.02 to 0.90" (as in F3), and apply this throughout.
- **F10.** "0.01 (−0.446 to 0.161)": the precision is mixed.
- **Table C1.** "Undecided: no rule changed any temperature" is a confusing label for "no effect". Say "No effect (no alternative rule changed the set)".
- **Appendix A.** "Articles shorter than 576 tokens in a model's tokenizer were not used." Say whether any were dropped for any model. If so, the "same 120 articles" claims need qualifying.
- **Sec. II.** "Sampling was plain softmax at temperatures between 0.19 and 0.90." Note that 0.19 appears only for Mistral and 0.90 only in the 15-temperature screens.
- **Sec. V.** The phrase "|h*|/T* between 0.67 and 2.1" gives 0.66 for BLOOM with the displayed values (0.37/0.56). Also, at the temperatures actually used (T* ± 0.15), |h|/T reaches about 3.4 for GPT-2-XL, and about 7 at 2h* in the first test. The range quoted understates the strength tested.
- **Abstract length and density.** The abstract is about 250 words with about ten numbers. Suggested restructure: sentence 1 gives the finding (a first-order derivative from unpenalized samples predicted the median change in the entry/exit-rate ratio at penalties that changed it about threefold); sentence 2 gives why it is surprising (exact whole-text reweighting is degenerate at these strengths). Move the caveats into one sentence.
- **Abstract.** "the tokens that have previously followed the current last two tokens" is set off by commas and reads as part of the main clause. Use parentheses or a dash.
- **Sec. I.** "Kempton and Burrell described how local normalization distorts temperature and truncation" duplicates the reference in the previous paragraph.
- **Related work.** Two connections are missing. The ML reader will recognize ∂E[O]/∂h = Cov(O, Σ(c−q))/T as the policy-gradient / REINFORCE estimator for a logit bias. The candidate set (tokens that followed the current bigram earlier) is also what induction-style copying mechanisms predict. A sentence on each would help orientation.
- **Absolute effect sizes.** Please add a small table of absolute values at h* (repeat fraction, share of continuations ending in the repetitive state, distinct-n) with and without the penalty. *Would not change the conclusions; makes the effect concrete.*
- **Text quality.** The paper motivates the work with "How strong a penalty must be is usually found by trial", but penalties are chosen by trading off quality. One quality proxy (for example, perplexity under a larger reference model, or MAUVE against the true article continuations, which the author has) would show whether h* is a useful setting. *Would not change the conclusions about first-order accuracy.*
- **EOS mass.** "the probability of the end-of-sequence token, which was not recorded": the author already rescored the replication texts (Appendix B), so the masked EOS mass per step can be recovered cheaply. It would show how artificial the forced-length regime is, especially for the second half of the text, where ln K > 0 (F2). *Would not change the tests; informs the scope.*
- **Data availability.** The generated continuations are "available on request". They are outputs of open models and could be released without the 64-token BBC prompts. *Would not change the conclusions.*

## 9. Recommendation: **major revision**

**Strengths.**

- The core experiment is careful and unusually transparent. It has dated preregistration, hash-checked predictions, common random numbers, an honest report of an undecided first test followed by a single committed replication, an independent reimplementation that reproduces the ratios to four decimals, and many failures reported plainly: the T_× target, the multiplicative penalty at practical strengths, the instruction-tuned models, and the non-bimodal states.
- The central result is solid within its regime and useful: a first-order, unpenalized-sample derivative predicts the median change in a switching statistic at penalties that change it about threefold, where exact reweighting is degenerate.
- Fig. 3a (the penalty acts at uncertain steps) is a nice decoding insight.

**Why not minor revision.**

1. The title and abstract frame the work as prediction, while the registered evidence tests linearity. The out-of-sample, model-specific value of the derivative was not demonstrated (3.1).
2. The secondary "explains why the effect is small / 1/τ" claim, which the abstract features, rests on a test that cannot discriminate and is untested against the labeling (3.4).
3. The practical relevance is limited and under-stated. The tested additive penalty is not deployed (though its relatives, the presence penalty and `no_repeat_ngram`, are and were not tested), the multiplicative penalty is confirmed only below practical strengths, the user-facing T_× target was missed per model, and greedy decoding is out of scope (3.2).
4. There is a direct contradiction about who checked the derivations (3.7).
5. Exploratory analyses and the equilibrium-model detour crowd the main line (Section 4).

**What would change my assessment.**

- Reframe the title and abstract around linearity and locality.
- Fix the contradictions.
- Either test or demote the 1/τ reading.
- Add the cheap windowed log-weight variance check.
- Ideally, test the presence penalty, which the existing machinery already supports.

None of the requested additions is likely to overturn the main replication result. The presence-penalty test and the within-model test of s_pen against τ could change the paper's scope and its secondary claim.