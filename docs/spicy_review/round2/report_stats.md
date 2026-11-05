# Referee report

**Manuscript:** "Predicting how a repetition penalty changes switching into and out of repetitive text, from unpenalized samples"
**Referee expertise:** statistics and research methods (study design, bootstrap, multiple comparisons, preregistration, replication)

I read the whole manuscript, including Appendices A–H, and looked at Figs. 1–3, S1 and S2. I recomputed the summary statistics I could from the tables (medians, means, products, ratios). Unless I say otherwise, the tables agree with each other and with the text to rounding. The points below are the ones that did not hold up, or that hold up only in a narrower sense than the text gives them.

---

## 1. Summary in my own words

**Question.** A repetition penalty is applied at every sampling step, and the distribution is renormalized after it. Can its effect on one summary of generated text be predicted from samples drawn without the penalty? The summary is K, the ratio of the rate of entering "repetitive" stretches to the rate of leaving them.

**What was done.** For an additive logit shift h on "repeat candidates" (tokens that earlier followed the current bigram), the score of a whole text at h = 0 is (N − Q)/T. N counts the candidates chosen and Q is its compensator. This gives an exact likelihood-ratio derivative g = ∂ ln K/∂h from unpenalized samples. The author used g to choose a penalty h* intended to lower the crossover temperature T_× (where K = 1) by 0.10. The realized change in ln K at h* was then compared with h*·g, summed over seven temperatures per model.

- A first preregistered test on the 40 design articles was undecided.
- A preregistered replication on 120 new BBC articles with seven base models (0.4–1.7B) gave a median ratio of 1.04 (0.98–1.15), inside the band 0.8–1.25.
- Two 7B models gave a mean of 0.96 (0.84–1.11).
- The multiplicative (HF-style) penalty passed at the predicted divisor and at twice ln θ*, was undecided at four times, and fell below the band at eight times.

The exploratory analyses report three further things. The step-wise response is about 1% of the response to tilting whole texts by exp(hN/T). That fraction times a switching relaxation time τ is about 1. And most of the predicted response comes from steps with intermediate q_t.

**Main result.** For one statistic (ln K), one labeling family, one corpus and one text length, the first-order response computed from the unpenalized arm agrees with the realized change at h*, on the median over models. Because g came from the same test's unpenalized arm, this is a test that the response is still linear at h*. It is not a test of out-of-sample prediction. The paper says this clearly in the abstract and in Sec. IV B.

## 2. The most interesting finding, and whether the abstract leads with it

To me the most interesting finding is that a linear-response (score-function) prediction stays usable at penalty strengths where exact likelihood-ratio reweighting of the same samples is degenerate. A second interesting finding is that step-wise normalization removes about 99% of the whole-text ("tilted") response. The first is supported, with the reservations below. The second is exploratory, and it is partly built into how the baseline g_N is defined (Major 5).

The first two sentences of the abstract do not state a finding:

> "A repetition penalty acts on a language model at every sampling step, and the distribution is renormalized each time. We ask whether its effect on how generated text switches into and out of repetitive stretches can be computed from samples generated without it."

They give the setup and the question. The confirmatory result does not appear until the sixth sentence. The abstract also carries many numbers: h = 0, 1.04, 0.8, 1.25, seven models, 99%, and "one part in the hundred". I suggest that sentence 2 state the confirmed result in plain terms. Something like: "a first-order prediction from unpenalized samples matched the realized change on the median over seven models, at penalties where exact reweighting of the same samples had failed." The 99% / 1/τ material should then be clearly marked as exploratory. At present it is presented as an explanation ("The same score explains why the effect is small").

## 3. Major concerns

### M1. The title and the word "predicting" claim more than the confirmatory test shows

The text says it plainly:

> "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test. The test therefore asks whether the response is still first order at h*."

The title, "Predicting how a repetition penalty changes … from unpenalized samples", and Sec. IV B, "Predicting a penalty before applying it", lead a reader to expect an out-of-sample prediction. The analysis that is genuinely out of sample, ρ_out (design-time ḡ applied to new articles), is exploratory. It gave 1.05 (0.93–1.20) for the smaller models and was undecided for the larger ones. The confirmatory evidence therefore supports "the response is linear up to h*". It does not support "the effect can be predicted before applying the penalty". The title and section heading should say what was actually tested.

*Requested change:* retitle, or make clear in the title or first sentences that the registered tests are linearity tests. This does not change the conclusions; it brings the framing into line with them.

### M2. The decision statistic cannot tell a model-specific prediction from a generic constant

The paper reports this but does not draw the consequence:

> "The median rule does not single out the model's own derivative. A ḡ averaged over the other six models also gave a median inside the range"

and in F4:

> "It was 0.062 with the model's own ḡ and 0.107 with the shared one, and their difference, 0.04 (−0.10 to 0.11), did not exclude zero"

Across models, ḡ varies by more than a factor of two (1.30–2.81, Table F5). Replacing it with the other models' average still passes the registered rule. So the registered rule cannot detect whether the identity captures anything specific to a model. A practitioner applies a penalty to one model, so this matters. The median over seven fixed models, with articles as the only resampling unit, answers a population-average question that no user faces.

There is a related point. The per-model statement is phrased in a way that may *understate* the evidence:

> "The agreement holds for the median over models rather than for each model."

The per-model replication intervals are about 0.25–0.40 wide. That is close to the full width of the band, 0.45. Most per-model "failures" are therefore failures of precision, not evidence of disagreement. Only SmolLM2-1.7B, 1.18 (1.02–1.42), excludes 1. On the log scale, the spread of the seven point estimates (SD ≈ 0.07) is no larger than a typical per-model standard error (≈ 0.08, read off the intervals). As written, the sentence invites the reading "it does not work per model". The data say "it was not shown per model". The two are different, and the paper should say which it means.

*Requested additions:*

- (a) An estimate of between-model heterogeneity in ln ρ_m. For example, use a random-effects estimate of τ², taking the cross-model error covariance from the joint article bootstrap. Alternatively, run a joint bootstrap test of equal ρ_m. *Could change conclusions:* yes, in either direction. If heterogeneity is about zero, the per-model statement should be softened and the claim strengthened. If it is not zero, the abstract should say the identity is biased for some models.
- (b) A per-model test–retest comparison of ρ_m between the first test and the replication. For example, Pythia-1.4B went from 0.75 (0.62–0.95) to 1.03 (0.92–1.20). The data already exist. *Could change conclusions:* possibly; it shows whether per-model departures are reproducible.
- (c) A secondary statistic that would penalize a generic constant, such as the median of |ln ρ_m|, reported alongside the registered one and labelled exploratory. *Does not change the registered decision*, but it changes how informative the decision is.

### M3. The pooled evidence leans above 1, and the band was set partly from a known bias

Across tests, the point estimates are mostly above 1:

- first test 1.17 (h*) and 1.10 (2h*)
- replication 1.04
- pooled, 160 articles: 1.09 (0.98–1.15)
- ρ_out 1.05
- fraction of repeats 1.06
- last 64 steps 1.23

The BCa intervals, which the paper uses only to confirm that the decisions are unchanged, say more than that:

> "bias-corrected and accelerated (BCa) intervals gave the same decisions: 0.95–1.10 for the replication, 0.84–1.10 for the larger models, and 1.04–1.47 and 0.94–1.27 for the first test at h* and 2h*."

On the 40 design articles at h*, the BCa interval excludes 1 (1.04–1.47). The large shift between the percentile and BCa intervals there (0.95–1.30 against 1.04–1.47), and the shift for the replication (0.98–1.15 against 0.95–1.10), both point to a bootstrap distribution that is skewed or biased relative to the point estimate. This is typical of ratio estimators. Meanwhile the band was justified partly by:

> "a synthetic check in which the labeling raised the ratio by about 10%."

If the labeling raises the ratio by about 10% even when the identity is exact, then the expected ratio under "the response is first order" is not 1. A median of 1.04–1.09 is then hard to interpret: it could be labeling bias, second-order terms, or noise. The synthetic check is described in one clause and never discussed again.

*Requested additions:*

- (a) Describe the synthetic check: what process was used, what the true ratio was, and why the labeling biases it.
- (b) Report the bootstrap bias estimate for each registered statistic.
- (c) Discuss whether the evidence as a whole points to a systematic excess of about 5–10%.

*Could change conclusions:* not the registered decisions. It could change the interpretation from "first order holds" to "first order holds up to a positive bias of about 5–10%, whose source is not identified".

### M4. "Close to linear up to |h| ≈ 1.6" is contradicted for the model that reached |h| ≈ 1.6

F3 says:

> "Doubling the penalty barely changed the ratio, so the change in ln K stayed close to linear in h up to |h| ≈ 1.6."

Only GPT-2-XL reaches |2h*| ≈ 1.6 (h* = −0.79). For that model the paper reports that the difference between the 2h* and h* ratios excluded zero: "+0.15 (0.005–0.47)". The same holds for GPT-Neo-1.3B, "+0.14 (0.04–0.24)". The claim rests on the difference of *medians*, −0.07 (−0.12 to 0.14). That is an across-model summary, and it cannot establish linearity "up to |h| ≈ 1.6" when the one model that reached that strength showed measurable curvature.

The Discussion has the same problem:

> "The first-order prediction held at penalties with |h*|/T* between 0.67 and 2.1."

First, "held" is an across-model statement presented as if it held per model. Second, |h|/T on the test grid reaches well beyond T*. For GPT-2-XL the grid goes down to T ≈ 0.23, where |h*|/T ≈ 3.4. The range quoted is narrower than the one actually used.

*Requested addition:* estimate the second-order term from the unpenalized samples, using the second-order score identity, which needs no new runs. Report it per model as an expected curvature at h*. The paper says it was not done ("We did not estimate the second-order term of the response"). Yet this is exactly the quantity whose smallness is being claimed. *Could change conclusions:* yes. It would replace an indirect argument by a direct one, and it could show where the per-model 2h* departures come from.

### M5. The s_pen ≈ 1/τ "reading" was checked with a criterion that could not fail

> "Before computing, we had fixed that a coefficient of variation of s_pen τ below 0.3 across the smaller models would count as agreement with this reading. It was 0.14."

From Tables II and F11, the CV of s_pen *alone* across the seven smaller models is about 0.18 (values 0.81–1.46%). The CV of τ alone is about 0.10 (90–122 steps). If s_pen and τ were *independent*, the CV of their product would be about √(0.18² + 0.10²) ≈ 0.20, which also passes the 0.3 threshold. The criterion therefore passes for any roughly constant τ near 100. The rank correlation the paper reports, 0.11, confirms there is no across-model relation. The paper concedes that "the reading fixes the size of s_pen, not its order across models". With a 512-token text and τ ≈ 100, though, "the size" amounts to an order-of-magnitude coincidence. It is also unclear from F8 whether the s_pen values were already known when the CV threshold was set. If they were, the criterion was set knowing that it would pass.

In addition, the baseline g_N is partly built in. In the repetitive state the models choose a candidate at 92–94% of steps; in the ordinary state, at 16–21% (F1). So N is close to a linear function of the time spent in the repetitive state, M₁. Tilting by exp(hN/T) is then very nearly a tilt on the labeled state itself. A large tilted response of ln K is expected for that reason alone. The ratio of a local response to the static susceptibility of a slowly relaxing observable is the standard ratio of a one-step variance to an integrated autocorrelation time. "1% ≈ 1/τ" may be that textbook relation rather than a new property of step-wise normalization. The paper should either relate it to this known result or show that it is something more.

The abstract states this exploratory, weakly tested reading as an explanation:

> "The same score explains why the effect is small. In our runs Q cancels about 99% of the response to reweighting whole texts. What survives is roughly one part in the hundred or so steps over which the switching relaxes."

*Requested additions:*

- (a) Compare s_pen with 1/τ_int, where τ_int is the integrated autocorrelation time of c_t (or of the state indicator) computed directly. 1/(k₀₁ + k₁₀) depends on the labeling.
- (b) Report s_pen under the other labelings ("We did not estimate … s_pen under other labelings").
- (c) Separate τ from the horizon: s_pen rises from 1.16% at L = 512 to 1.79% at L = 256, and the paper says length and position are confounded.
- (d) Give bootstrap intervals for each model's s_pen in Table II.

*Could change conclusions:* yes, for the "explains why the effect is small" claim. It would not change the registered tests.

### M6. The contrast with importance reweighting is argued, not shown

> "Whole-text reweighting therefore cannot work at these penalties."

The evidence (Appendix D) is (i) the effective sample size *per article*, computed from four samples, which is bounded between 1 and 4 and noisy, as the paper says, and (ii) a *second-order approximation* of Var(log w), applied at |h|/T up to 2–3, where higher cumulants matter. The fair comparison is self-normalized importance sampling pooled over the 480 continuations at each temperature, the same pool the score estimator uses. The paper says this was not tried: "We did not try self-normalized weights pooled over articles." With Var(log w) ≈ 3.4, the pooled ESS would be roughly 480·e^(−3.4) ≈ 16. With 24.6 it would be negligible. So the claim may well hold for most models, but it has not been demonstrated. The exact log w is available, so its empirical variance can also be reported directly.

*Requested addition:* pooled self-normalized IS estimates of Δ ln K at h*, with bootstrap intervals, alongside the first-order estimate; and the empirical Var(log w). *Could change conclusions:* possibly, for the models with the smallest variance (3.4). The abstract states this contrast as fact.

### M7. The practical claim about cost omits estimator variance

> "Locating T_× required unpenalized runs at several temperatures, comparable in cost to trying a few penalty strengths, but the same runs give the response to any such change."

F4 reports:

> "the standard deviation of ḡ was 0.21–0.32 per unit of h, and that of the measured slope 0.11–0.13, but the slope needs two penalized arms instead of one unpenalized arm."

For a single penalty, the score-function estimate has 2–3 times the standard deviation of a finite difference with common random numbers, so 4–9 times the variance. For equal precision it needs more samples than the two penalized arms it replaces. Fig. S2b shows the same thing: per temperature, the identity's ratios range from about 0.5 to 3.6, while the equilibrium ratios are tight. The numerator is shared, so this dispersion comes from noise in the estimate of g. The score approach pays off only when many different perturbations are evaluated from the same runs. The text should say this. *Does not change the conclusions; corrects a practical claim.*

### M8. Statements that contradict other statements in the paper

1. **Who checked the derivations.** Sec. V: "the author has since gone through the derivations of Sec. III (Appendix E)." Appendix E: "The derivations of Sec. III and Appendices G and H were likewise written by an AI assistant and checked by other AI instances, not by the author." These cannot both be current. A journal will expect the author to take responsibility for the derivations in Sec. III at least.
2. **Overshoot of T_×.** F6: "The overshoot of T_× therefore does not come from a misprediction of ln K at T*." The same paragraph reports "the total ranged from −0.48 for BLOOM-1.1B to 0.87 for GPT-2-XL". GPT-2-XL is the model with the largest overshoot, and its deviation of +0.87 in ln K is in the direction that lowers T_× further. The conclusion holds for the median, not for the model that motivated it.
3. **"Visible in the design data."** Sec. IV B: "we found that the spread was already visible in the design data." The shifts implied by the design data span 0.97–1.64 times the target (F6). The realized interpolated shifts span 0.51–2.41. The *ordering* matches almost exactly, but the realized spread is about 2.5 times wider. Say "the ordering, not the size, was visible".
4. **Pooled 160-article interval.** "Pooling the first test and the replication, 160 articles at h*, gave a median ratio of 1.09 (0.98–1.15)". This interval is identical to two decimals to the 120-article replication interval (0.98–1.15). With a third more articles one would expect it to be about 13% narrower. This may be correct, since the medians and the article sets differ, but please check it.
5. **Usable temperatures.** Sec. II: "An analysis uses a temperature only if both kinds of switch occurred there at least ten times." Appendix C adds "in both arms". Make the main text match.
6. **Article filter versus 480 continuations per model.** Appendix A: "Articles shorter than 576 tokens in a model's tokenizer were not used." If this ever removed an article for some models, the claim of "the same resample for all models" and the "480 unpenalized continuations" per model in F1 would not hold. State whether any article was removed for any model.

### M9. Choice of the equivalence band, the interval level, and the sample size

- The band 0.8–1.25 was "set before the first test, from the agreement already seen on the design runs". A band set from observed pilot agreement and expected precision measures precision rather than an accuracy that matters in practice. Please justify ±25% in terms a user would recognize. For example: what error in T_× or in the fraction of repeats does a 25% error in Δ ln K correspond to? Table F8 suggests that per model it corresponds to T_× shifts between 0.5 and 2.4 times the target.
- Inclusion of a 95% interval is a two-one-sided test at α = 0.025 per side. That is conservative and acceptable, but say it, and say that a 90% interval would be the conventional choice for equivalence testing.
- The replication's size came from a precision argument ("narrow it by about 1/√3, to about 0.2"), not a power calculation. Given the first test's point estimate of 1.17, an interval of width 0.2 centered there would not fit inside the band (it would need an upper end ≤ 1.25). The replication "succeeded" because the point estimate fell to 1.04. Please report the planned probability of a confirmatory outcome under a plausible true ratio, for example 1.05–1.10.
- The decision to replicate followed an undecided result. That is legitimate, and the paper reports it. Still, one look plus one conditional further look raises the overall chance of declaring equivalence when the true median lies just outside the band. A sentence on this would help.

*None of these changes the registered decisions. They change how much weight a reader should give them.*

### M10. Dependence among the "confirmations", and the inference unit

> "The additive, multiplicative and larger-model tests share the same 120 articles, and the bootstrap intervals resample articles, not models."

The multiplicative test also uses the replication's unpenalized arm as its control. The three confirmations are therefore not independent pieces of evidence, and the abstract should not suggest that they are. Because models are not resampled, every interval is conditional on these nine checkpoints. "The response stays first order" is a statement about these models. *A wording change; it does not alter the conclusions.*

### M11. Results that carry more weight in the text than their tests can bear

- **F10, drift across temperatures:** "0.01 (−0.446 to 0.161), just inside ±0.45 … which a rule fixed beforehand read as no drift beyond that width." An equivalence margin of ±0.45 on the change of the ratio across the grid is about the whole decision band. The interval touches the margin, and the paper itself says it "does not exclude a drift of that size". Call it uninformative.
- **F10, single temperatures:** "4 had an interval excluding zero, 8% against the 5% expected by chance, all four in the same direction". P(X ≥ 4 | n = 49, p = 0.05) ≈ 0.2, and the 49 pairs are not independent. Say which direction, or drop the sentence.
- **F9, other genres:** in the Discussion, "runs on novels, scientific papers and code are in Appendix F9" reads like support for the identity across genres. But F9's identity ratios "have no intervals and were not part of the decision", and they range from 0.71 to 1.46. Say in the Discussion that the genre runs tested only the equilibrium model.

## 4. What could be removed or moved, and which caveats readers need

**Could be moved to an appendix or shortened without changing the conclusions:**

- Sec. IV B, last paragraph, and much of F6 (the shift of T_×). Keep one sentence in the main text: the T_× target was missed per model, for a reason identified afterwards (averaging g over temperature). The quadratic-fit "record" and the three-part decomposition can stay in the appendix.
- Sec. IV D, second paragraph, and Fig. 3b (the entry/exit split). It is a secondary, exploratory result whose prediction was "less well" matched. An appendix figure is enough.
- Sec. III C, Discussion paragraph 3, and the Introduction's pointer to Appendix H all cover the equilibrium two-state shortcut. One mention in the main text is enough. Table F2's ρ_eq column, Table F9's r_C and "T_× ratio" columns, and the halves analysis in F7 are records with no bearing on the main claim and can be shortened considerably.
- F10's third analysis (other rules for usable temperatures). No rule changed any temperature, so the result is empty. The sentence "the rule fixed beforehand recorded this question as undecided because no alternative rule could be tested" confuses more than it informs. Remove it.
- F2, the entry ratio 0.77, which is not used anywhere.
- Appendix C's list of BCa and seed variants. One sentence is enough, except that the BCa interval for the first test (1.04–1.47) should be *kept* and discussed (M3).
- The Discussion sentence on "entropic and frenetic parts". It introduces undefined terms to say something does not apply.
- Model-revision pinning and the post-freeze import fix belong in Appendix A/C only (they already are) and need not be in the Discussion.

**Caveats a reader really needs (keep them in the main text, gathered in one place):**

1. The registered tests estimate g from the same runs, so they test linearity, not out-of-sample prediction.
2. The decision is on the median over seven fixed models. It does not distinguish the model's own derivative from a shared constant, and models are not resampled.
3. One corpus (BBC News), one length (512), one labeling family, EOS masked, no truncation.
4. The multiplicative penalty was confirmed only at strengths at or below the low end of common practice. Instruction-tuned models were undecided or not run.
5. s_pen and the 1/τ reading are exploratory, depend on length and covariance convention, and say nothing about the ordering of models.
6. Most point estimates lie above 1 (M3).

## 5. Undefined terms, jargon, and passages that read as notes

**Undefined or underdefined for a general reader:** "compensator" (defined in passing), "Doob transform", "tilted ensemble", "static susceptibility of the path ensemble", "escape-rate term", "entropic and frenetic parts" (never defined), "label bias", "van 't Hoff plot", "Clausius–Clapeyron", "κ". "Record" and "rule fixed" are defined only in the caption of Table C1, but "exploratory, record" is used earlier, in F3. Δc and ΔE are used in F7 (r_C) before they are defined in Appendix H.

**Notation load.** The paper uses ρ, ρ_out, ρ_eq, r_C, R, s_pen, s_temp, κ, g, ḡ, ḡ_λ, g_N, g_Q, g_U, g_C, g_C^tilt, b and b_tilt. Several appear only once. I suggest cutting the main-text notation to ρ, g, ḡ, s_pen and τ, and confining the rest to Appendix H. "R" and "r_C" are particularly easy to confuse.

**Passages that read as notes or lists:**

- The abstract: a chain of short declarative sentences with many numbers and no connecting argument.
- Sec. V, paragraph 5: a list of caveats run together.
- F1, first paragraph: nine models' rates listed inline ("Pythia-410M 94% and 20%, Pythia-1.4B 94% and 20%, …"); this belongs in a table.
- F6: "0.99, 1.33, 1.07, 1.25, 1.05, 1.64 and 0.97 times the intended 0.10, in the order of Table I" and the parallel list of g-ratios; this also belongs in a table.
- F10: "First, … Second, … Third, …" with no linking argument.
- The end of Discussion paragraph 1 jumps to jump processes without motivation.

**Other.** "Design temperature", "design articles" and "arm" should be defined at first use in Sec. II. The abstract's appositive "the tokens that have previously followed the current last two tokens" is hard to parse.

## 6. Checklist of common weaknesses

- **Large promise, small experiment.** Yes, in places. The title promises prediction; the confirmatory test is of linearity (M1). Sec. V says the derivative "applies to any small shift added to the logits of a set of tokens, such as logit biases and watermarking biases", and that "the same runs give the response to any such change". Neither was tested; the paper does say "We tested only the two repetition penalties". The abstract's "The same score explains why the effect is small" rests on an exploratory, non-discriminating check (M5).
- **Single run, model or dataset.** Several models (nine, plus two instruction-tuned), which is good. But one news source (one collection, April–June 2025), one language, one continuation length, four samples per prompt, one labeling family, and one replication. The 7B "test" is two models summarized by their mean. The genre runs (F9) test only the equilibrium model, without intervals for the identity.
- **Parameter choices without a stated reason or time of choice.** No reason is given for: the 4-gram repeat definition; the 32-token window; the thresholds 1/2 and 1/4; the 64-token prompt; four samples per article (which limits the within-article covariances to 3 degrees of freedom and makes the ESS in Appendix D meaningless); the 0.10 target shift; the usability threshold of ten switches; the design grid 0.40–0.70; the CV thresholds 0.3/0.6; the 0.5–2% "about 1%" band; and the ±0.45 drift margin. A reason is given for the 512-token length ("fixed in the first experiments of the project"), for the bigram context of candidates, and partly for the band 0.8–1.25. The *time* at which the labeling parameters (window, thresholds) were chosen is not stated.
- **No report of computational cost.** Partly reported (Appendix E): about 22 L4 GPU-hours for the replication and about 6 A100 GPU-hours for the second stage of the 7B test. Not reported: the design runs, the first test, the multiplicative and stronger-divisor tests, the instruction-tuned runs, the genre runs, the rescoring, or the project total. Please add a total, and the cost of the score-based approach against finite differences at equal precision (M7).
- **Broad conclusion from a narrow case.** Yes. "This shows that the response stays first order" generalizes from one statistic, one corpus and one length. "Whether the two-state model overstates the effect of a penalty is therefore a property of the chosen energy, not of the sampler" generalizes from seven small models and three energy choices. "Whole-text reweighting therefore cannot work at these penalties" generalizes from per-article ESS on four samples (M6).

## 7. Could the author defend each design choice? Where would it be hardest?

Several choices are well defended:

- the ratio of sums over temperatures, so that near-zero denominators do not dominate;
- resampling articles jointly across models and arms; the paired arms with common random numbers are correctly handled, and independent resampling widened the interval "about twofold";
- matching candidates on bigrams so that the definition of the states does not depend on the penalty;
- masking EOS so that the score is a sum over a fixed number of steps.

The preregistration practice is exemplary. Hashed predictions are checked by the generating notebooks, failures are reported (k = 8 below the band; instruction-tuned undecided), and the post-freeze change is disclosed.

I would expect the author to struggle with these questions:

1. **Why the median over models?** Users apply a penalty to one model, and the rule passes even with another model's derivative (M2). The answer "so that no single model could decide it" protects the test against outliers, but at the cost of being able to show anything model-specific.
2. **What exactly the synthetic check was** "in which the labeling raised the ratio by about 10%", and why a known bias of that size did not change the centre of the band (M3).
3. **Why CV < 0.3 is evidence for the 1/τ reading** when s_pen alone, or s_pen times an unrelated τ, would pass (M5).
4. **Why within-article covariances for s_pen but pooled covariances for the tests.** Why ln K rather than a quantity users measure. Why g was averaged over 0.40–0.70 when T_× was the target; this is the stated cause of the overshoot.
5. **Why four samples per article**, and why IS was judged per article rather than pooled (M6).
6. **The derivations in Sec. III B and Appendix H** (Doob transform, Σν_t, the factorization g/g_C = (s_pen/s_temp)κ). Given the contradiction in M8.1, an editor who asks the author to explain the tilted-ensemble argument without notes is where the paper is most exposed. I checked the decomposition N − E[N] = (N − Q) + Σν_t and the identity Q − E[Q] = Σν_t, and they are correct. The author should nonetheless be able to present them independently.
7. **External time-stamping.** The registrations are "committed … to a version-controlled repository". Please state whether the registrations were time-stamped by a third party (for example a DOI-minting archive or a public registry) before the data existed. Commit dates in a self-controlled repository can be rewritten. This does not change the conclusions; it supports the claim of preregistration.

## 8. Minor issues

1. Table I caption: rather than "The lower ends shown as 0.80 are 0.796 … and 0.799", print three decimals in the table, since these values decide the per-model statements.
2. Appendix H: "ρ × g/g_C differs from ρ_eq by up to 0.07, for GPT-2-XL". From the tabulated values, 1.08 × 0.98 − 0.98 ≈ 0.08. This is within rounding, but please check.
3. F8: "All three medians that had a decision rule fell within 0.5–2%". Four non-pooled conventions are reported (baseline, mixed, 384, 256). Say which three were judged.
4. Fig. 1a: "chosen as the first output with two to six switches". First in what order? The repetitive stretch shown lasts about 35 steps, well below τ ≈ 100 and below the dwell bins used in F2. A more typical example would serve better.
5. Fig. 3a: "share of the predicted response" can be negative within a bin (the q < 0.1 bin contains values below zero). State how negative contributions are counted.
6. Table II: s_pen is in % but s_pen τ treats it as a fraction. Say so in the caption.
7. Precision is inconsistent: "+0.15 (0.005–0.47)", "0.01 (−0.446 to 0.161)", "0.0093" in Table F1. Use two decimals or two significant figures throughout.
8. The citation style mixes arXiv identifiers ("[1904.09751]") with author–year ("[Glynn 1990; …]"). Use one style.
9. Appendix A lists revisions for the nine base models but not for the two instruction-tuned models.
10. Appendix A: the 120 replication articles were drawn "in three groups of 40". Say why. If the groups were generated separately, a check for consistency between groups would be a cheap internal replication (optional; would not change conclusions).
11. Appendix A: the WLS weights 1/(1/n₀₁ + 1/n₁₀) assume independent Poisson counts. Switches within a continuation are correlated, and K drifts along the text (F2). This affects only the fit of T_×, not the bootstrap inference, but say so.
12. Sec. IV A: "at almost every step" for 92–94% is fine; "about one step in five" for 16–21% is fine.
13. F5: say explicitly which g_λ enters the denominator of the multiplicative ratio: the one rescored from the design runs, or one rescored from the replication's unpenalized arm.
14. "the rule fixed beforehand, an upper end below 0.8" (F7) and "0.8003 just failed": a boundary case decided at the fourth decimal suggests reporting it as undecided rather than interpreting it.

## 9. Recommendation: **Major revision**

**Reasons.** The registered core is careful and unusually transparent:

- an exact score identity;
- hashed, pre-committed predictions and decision rules;
- full reporting of the inconclusive and negative tests;
- a paired design with common random numbers and a correct article-level bootstrap;
- an independent reimplementation that reproduced the estimator;
- release of the per-step quantities.

I believe the registered result: on these models, the median response at h* is linear to within the band.

The paper, however, claims more than that result supports, and contains internal contradictions:

1. The title and framing say "predicting", but the confirmatory test is of linearity (M1).
2. The decision statistic cannot tell model-specific prediction from a generic constant, and the per-model statement confuses lack of precision with disagreement (M2).
3. A known bias from the labeling and a consistent excess above 1 are not discussed (M3).
4. Linearity "up to |h| ≈ 1.6" is contradicted for the model that reached that strength (M4).
5. The explanatory "99% / one part in τ" claim in the abstract rests on a criterion that could not fail, and on a baseline partly built in by construction (M5).
6. The contrast with importance sampling is argued rather than shown (M6).
7. The practical cost claim ignores the estimator's variance (M7).
8. Several statements contradict others, including who checked the derivations (M8).

**Which requested additions could change the conclusions:**

- M2a/b (heterogeneity and per-model test–retest): could change the per-model claim, in either direction.
- M4 (second-order term): could change the range over which "first order" is claimed.
- M5a–c (τ_int, labelings, horizon): could remove or substantiate the "explains why the effect is small" claim.
- M6 (pooled IS): could weaken the "reweighting had already failed" contrast for some models.

**Which do not change the conclusions but are needed for accurate reporting:** M1, M3, M7, M8, M9, M10, M11, and the cuts listed in Section 4.

With these revisions, and with the main text cut back to the confirmatory result and a clearly labelled exploratory section, the paper would make a solid, honest methods contribution.
