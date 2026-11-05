# Answers on manuscript_main_text.md

## 1. Per-paragraph ratings (0 = reads as human, 3 = reads strongly as machine-generated)

Overall the prose is plain and restrained, and no paragraph reads strongly as machine-generated. The impression comes mostly from a few repeated patterns: one fact per sentence, chains of "Here X is ..." definitions, "X, not Y" disclaimers placed before anyone raises the point, and summaries that repeat earlier text.

| P | Rating | Reason (if 1 or more) |
|---|---|---|
| P1 | 2 | One fact per sentence for about 16 sentences, with definitions stacked mid-abstract ("Here N counts the candidates chosen, Q sums over steps the probability of choosing one, and T is the temperature"). It also has an abstract metaphor ("It removes the credit each choice would receive for the candidates it makes likely to be chosen later") and a closing list of results ("The widely used multiplicative penalty ...", "The derivative also shows ..."). |
| P2 | 1 | A string of short sentences, one citation each ("Adaptive sampling schemes have been designed partly to avoid it", "Training objectives that discourage repetition have been proposed", "Penalties derived from a model of expected frequencies have also been proposed"). |
| P3 | 1 | It sets up a straw man and dismisses it in a neat pair: "A natural shortcut would treat ..." / "Appendix H shows why this shortcut is not a reliable guide." |
| P4 | 2 | A contribution list of the kind a template produces: "We use this formula to predict ..., and test the predictions after registering them. With the same formula we measure ..., which explains why the effect is small, and locate where in a text a penalty acts." |
| P5 | 2 | Related work grouped in sets of three citations per field, ending in a stock positioning line: "Here we bring these ideas to a repetition penalty and test the resulting prediction prospectively." |
| P6 | 0 | |
| P7 | 1 | Short, flat justifications: "This additive penalty is not an option in common libraries." and "so that the definition of the states does not depend on the penalty." |
| P8 | 1 | It coins names in a set pattern: "We call the temperature at which K = 1 the crossover temperature T_×." / "We call these the usable temperatures." |
| P9 | 2 | Disclaimers placed before anyone raises the point, in "X, not Y" form: "The two states are a labeling, not distinct phases." / "K is also not the rate ratio of a stationary two-state Markov chain." |
| P10 | 0 | |
| P11 | 0 | |
| P12 | 1 | An aside about generality that the paper never uses: "The identity also holds with a bounded stopping time ..., but we did not use one." |
| P13 | 0 | |
| P14 | 1 | Elliptical definitions tacked on at the end: "Here a is the derivative of a token's logit with respect to λ, and E_t the expectation under the step-t distribution." |
| P15 | 1 | Another "Here ν_t is ..." definition chain. The opening "To see what the difference means" reads as human teaching, though. |
| P16 | 0 | |
| P17 | 1 | A naming sentence at the end of a derivation: "we call s_pen = g/g_N the fraction of the tilted response that survives step-wise normalization." |
| P18 | 0 | |
| P19 | 0 | |
| P20 | 1 | Dense chained clauses and a crisp contrast: "which therefore also subtracts a compensator. Exponential tilting by exp(−εU), with a small tilt ε, does not." |
| P21 | 1 | A balanced antithesis after a semicolon: "The prediction needs an energy for each step and changes with that choice; the exact identities and s_pen do not." |
| P22 | 0 | |
| P23 | 0 | |
| P24 | 0 | |
| P25 | 1 | Disclosure sentences piled one after another: "These documents did not cover Fig. S1, whose runs were made earlier." / "Of the two factors of the prediction, only h* was fixed beforehand, while g came from ..." |
| P26 | 2 | Robustness checks listed one per clause, each with an appendix pointer ("Under eight other labelings ... (Table F2), and across temperatures the ratio did not drift systematically (Appendix F10)"), plus counterfactual comparisons ("For comparison, the tilted response of Sec. IV C would give a ratio of about 0.01." / "would have fallen outside the range for six of the nine models"). |
| P27 | 1 | A tidy concluding line: "The prediction therefore held for divisors up to about 1.1, the weakest values used in practice, and not reliably beyond." |
| P28 | 1 | "which a user would target" is slightly formulaic. The rest is human (see Q2). |
| P29 | 0 | |
| P30 | 0 | |
| P31 | 2 | It restates the mechanism after giving the numbers ("Both fractions are first-order quantities, taken at zero penalty and zero tilt." / "Nearly all of the covariance of the switching counts with N comes through ..., and Q carries the same expected later choices (Sec. III B)"). |
| P32 | 2 | A list of robustness checks with stock phrasing: "changes them little", "consistent with fewer later steps following each choice", "Both fractions are exploratory (Table C1)." |
| P33 | 1 | It repeats a caveat already made in P21: "Unlike s_pen, s_temp depends on the energy assigned to a step." |
| P34 | 1 | A preemptive "X, not Y": "This proportionality is an empirical finding, not a consequence of the form of the score." |
| P35 | 1 | A slogan-like topic sentence: "The penalty acts at both ends of a repetitive stretch." |
| P36 | 0 | |
| P37 | 2 | It opens by recapping the results, then restates in "not ... alone" form ("The prediction depends on the signed difference ..., not on the number of candidates chosen alone"), then gives an analogy and immediately qualifies it ("although that theory separates terms by time reversal, which autoregressive sampling lacks"). |
| P38 | 1 | "We attribute the contrast to scale, without having tested it." The candor reads as human, but the hedge is attached in a formulaic way. |
| P39 | 0 | |
| P40 | 1 | A claim of generality followed by the caveat: "applies to any small shift ... such as ..." then "We tested only the two repetition penalties, without truncation." |
| P41 | 2 | A checklist of caveats in uniform declaratives, mostly repeated from earlier sections: "The two states are a labeling of a continuum, K averages over a drift along the text, and s_pen depends on the length of the text." / "The change in ln K was predicted, but the shift of T_× did not always match the intended 0.10 (Sec. IV B)." |
| P42 | 0 | |

## 2. Paragraphs that read most clearly as written by a person

- **P28**: it gives an untidy, specific admission: "Some smaller models, GPT-2-XL most of all, and the mean of the two larger models still departed from the target, for reasons we have not identified." It names a model and admits an unexplained failure, without smoothing it over.
- **P24 (Fig. 1 caption)**: it explains how the example was chosen: "chosen as the first output with two to six switches". This is a concrete, unglamorous detail of the kind a person who actually picked the example writes.
- **P22**: it opens like a teacher ("Consider a temperature where both kinds of stretch are common.") and uses plain proportions ("at about one step in five").
- **P42**: a candid, slightly awkward disclosure: "not line by line by the author", "Model revisions were not pinned in the code, although the revisions used can be identified".
- **P38**: "We attribute the contrast to scale, without having tested it." It owns an untested belief instead of hedging it away. It also gives an intuitive physical explanation ("so small changes at many steps compound").
- **P15**: it guides the reader through a derivation ("To see what the difference means, write ...").
- **P6 and P10–P13**: plain methods and derivation text, with no rhetorical framing.

What these have in common: concrete particulars, admissions of loose ends, and uneven sentence rhythm instead of one fact per sentence.

## 3. Deletable and repeated sentences

**Could be deleted without losing anything a reader needs**
- P12: "The identity also holds with a bounded stopping time, such as stopping at an end-of-sequence token or at a maximum length, whichever comes first, but we did not use one." (It is not used, and P40 already covers how far the method generalizes.)
- P15: "The step-wise response equals this tilted response minus Cov(O, Q)/T." (This follows directly from P13 and the sentence before it.)
- P20: "where E_t is the expectation under the step-t distribution at the current temperature" (already defined in P14; only "at the current temperature" is new).
- P23, as a separate paragraph: "The crossover temperature lies between 0.44 and 0.66 for the nine models". It could be merged into P22 or left to Table I.
- P24: "The crossover temperature is where ln K = 0." (Already defined in P8.)
- P33: "Unlike s_pen, s_temp depends on the energy assigned to a step." (Already said in P21.)
- P40: "We tested only the two repetition penalties, without truncation." (Already in P6 and in P41.)
- P41: "Decisions concerned the median over the seven smaller models or the mean over the two larger ones" (already in P25, P26 and Table I). Also "The two states are a labeling of a continuum, K averages over a drift along the text, and s_pen depends on the length of the text" (already in P9 and P32).

**Repeat something said elsewhere**
- The equilibrium shortcut is criticized four times: P3 "Appendix H shows why this shortcut is not a reliable guide."; P21 "The prediction needs an energy for each step and changes with that choice"; P26 "would have fallen outside the range for six of the nine models"; P39 "it overstated the effect by about 45%".
- Tilted ensembles are introduced twice: P5 "reweighting whole trajectories gives tilted ensembles" and P15 "in statistical physics the result is a tilted ensemble of trajectories".
- The derivative is estimated from the same test's runs, stated three times: P1 "With the derivative estimated from the unpenalized runs of the same test"; P25 "while g came from the unpenalized runs of the same test"; P41 "Because the tests estimated the derivative from their own unpenalized runs".
- The mechanism of the cancellation appears three times: P1 "It removes the credit each choice would receive for the candidates it makes likely to be chosen later"; P17 "exponential tilting also rewards each choice for the candidates it makes likely to be chosen later"; P31 "Nearly all of the covariance ... comes through the candidates that each choice makes likely to be chosen later".
- The T_× shift missed its target three times: P1 "that temperature did not always move as intended"; P28; P41 "the shift of T_× did not always match the intended 0.10".
- The role of N − Q is restated: P37 "The prediction depends on the signed difference between each choice and the model's own probability of choosing a candidate" repeats P11–P13.
- "widely used" penalty appears three times: P1 "The widely used multiplicative penalty", P7 "A widely used multiplicative penalty", P41 "The widely used presence and frequency penalties".
- P31 "For ln K, the compensator term thus cancels about 99% of the tilted term." repeats P1 "cancels about 99% of the first-order response".

## 4. Sentences that are hard on a first reading

- P1: "In our runs, step-wise normalization therefore cancels about 99% of the first-order response the same penalty would produce on whole texts." The abstract never says what a penalty "on whole texts" is (exponential tilting appears only in Sec. III B). "therefore" also asks the reader to supply the step from "Q covaries ... almost as strongly as N".
- P1: "It removes the credit each choice would receive for the candidates it makes likely to be chosen later." "It" is ambiguous, "credit" is an undefined metaphor, and the relative clauses are nested.
- P1: "The penalties were chosen to shift the temperature at which the entry and exit rates are equal, and that temperature did not always move as intended, although the change in the log ratio was predicted." The design goal is first mentioned here, near the end, and three clauses pull in different directions.
- P4: "The probability of a text is then, in general, not exp(−U(x)/T)/Z(T) for an additive energy U that is independent of T." The "not" sits in the middle, and Z is undefined.
- P4: "... which explains why the effect is small ..." No effect has been described as small yet.
- P7: "we use it because its exact score has the simplest form". "Score" is not defined until Sec. III A.
- P7: "Repeats define the states below and candidates define the penalty, so that the definition of the states does not depend on the penalty." The reason for the "so that" is not obvious, because a definition based on 4-grams would not depend on the penalty either way.
- P14: "this gives the exact derivative g(T) = ∂ ln K/∂h, where K is defined by ratios of expected counts (Appendix B)." This clashes with the empirical-rate definition of K in P8, and "exact" sits oddly next to a quantity that is estimated from samples.
- P15: "write N − E[N] as the sum of updates of the expected final count, E[N | x_≤t] − E[N | x_<t]. Each update is c_t − q_t plus ν_t." It relies on a martingale decomposition without naming it.
- P17: "To first order in h, after renormalization, the expected future factor adds (h/T) Σ_t ν_t to the log-weight of a text, a term that a step-wise penalty lacks." It stacks qualifiers before the main clause.
- P20: "Over several temperatures we sum numerator and denominator with the weights of the equilibrium prediction g_C of Sec. III D: ..." It refers forward to a quantity not yet defined.
- P20: "Because each step's expectation is subtracted, adding to U_t any constant that depends only on the history, for example replacing U_t by −z_t, leaves this score unchanged." The reader has to know that −z_t differs from −log p_t only by a log-normalizer that depends on the history.
- P25: "Each test compares the realized change in ln K at h* with the first-order prediction h* g(T), each summed over the usable temperatures before their ratio is taken." The two "each"s refer to different things.
- P28: "With T_× located by a straight-line fit of ln K against 1/T, as in the design, its median was 1.3 times the intended shift in the smaller models." "its" has to be traced back to "the shift of T_×" in the previous sentence.
- P32: "Using covariances pooled over articles for the step-wise response, as in the tests, or computing s_temp from the step-wise temperature identity instead of the slope b, changes them little." Two alternatives are packed into the subject, and the difference between "within articles" and "pooled" is never explained.
- P34: "Steps where the model almost surely chooses a candidate, with q_t above 0.9". "Almost surely" has a technical meaning (probability one) that does not match q_t > 0.9.
- P38: "We attribute the contrast to scale, without having tested it." "The contrast" is never stated. The reader has to infer it: the first-order prediction works while the importance weights collapse.
- P41: "Because the tests estimated the derivative from their own unpenalized runs, they test whether the first-order prediction holds at h*." The intended contrast (and not whether a derivative estimated elsewhere carries over) is left implicit.

## 5. Three changes that would most improve naturalness

1. **Say each point once.** The shortcut criticism (P3/P21/P26/P39), the same-run estimation caveat (P1/P25/P41), the "credit for later candidates" mechanism (P1/P17/P31), and the T_× miss (P1/P28/P41) each need one full statement plus at most one short back-reference. This is what most affects whether P41, P31, P33 and P37 read as machine-like. *Removes nothing a reader needs*, as long as one instance of each stays, including the caveats in the limitations paragraph.
2. **Define terms before use and break up the definition chains.** Introduce "score", "exponential tilting/whole-text response" and "compensator" before using them (P1, P4, P7). Remove the forward reference to g_C in P20. Fold the trailing "Here X is ..." sentences (P1, P14, P15, P20, P21) into the sentences where the symbols first appear. In the abstract, cut down the one-fact-per-sentence list. *Removes nothing in the main text.* If the abstract drops the multiplicative-penalty and T_× sentences, abstract-only readers lose those two results; the main text keeps them.
3. **Merge the robustness checks and the preemptive "X, not Y" disclaimers into the surrounding argument.** In P26 and P32, gather the checks (eight labelings, no drift over temperature, pooled vs within-article covariances, the 256-token length) into one sentence with pointers, or move them to the appendices. Rephrase "a labeling, not distinct phases", "an empirical finding, not a consequence ...", "not on the number of candidates chosen alone" as positive statements that connect to what comes before. State what "the contrast" in P38 is. *Moving the checks to appendices removes some evidence from the main text*: a reader who wants to judge robustness without opening the appendices loses it. Keep a one-clause pointer to avoid that.

## 6. Typos, grammar, missing words, inconsistent terms or notation, contradictions

**Inconsistent terms and notation**
- Model size: "nine open base models with 0.4 to 7.6 billion parameters" (P6) and "models of up to 7.6 billion parameters" (P41), but "two 7-billion-parameter models" (P26) and "preregistered test on 7-billion-parameter models" (P30).
- Model set: "nine open base models" (P6), yet "For two instruction-tuned models the test was inconclusive for one and could not be run for the other" (P27; also Fig. 2, P29). The setup never introduces these models.
- Names for the advance registration: "with the analysis registered in advance" (P1), "after registering them" (P4), "test the resulting prediction prospectively" (P5), "dated documents" (P25), "committed in advance" (P26), "preregistered test" (P29, P30), "In the registered tests" (P37), "All prospective tests" (P41).
- Names for the acceptance interval: "the registered range of 0.8 to 1.25" (P1), "inside the range" (P26, P27), "the preregistered range 0.8–1.25" (P29), "Decision band for the ratio: 0.8–1.25" (P30).
- "usable temperatures" (P8, P25, P30, P33) vs "test temperatures" (P36: "at the test temperatures"; P24: "at the test temperature closest to"). Their relationship is never stated.
- "smaller models" (P28, P33, P35) and "larger models" are used as group labels, but the setup never defines them. P6 only gives a single size range.
- s_pen is called a "fraction" (P17: "we call s_pen = g/g_N the fraction"; P31) but reported "as a percentage" (P33).
- Definition of K: P8 defines it from observed counts ("switches into the repetitive state per ordinary step"; "K = entry rate/exit rate"), while P14 says "where K is defined by ratios of expected counts".
- E_t is defined twice: P14 "E_t the expectation under the step-t distribution" and P20 "where E_t is the expectation under the step-t distribution at the current temperature".
- Conditioning dropped: "P_h(x) = Π_t p_t^h(x_t | x_<t)" then "the derivative of log p_t^h(x_t) at h = 0" (P10).
- Subscript inconsistency: "c_t − q_t is replaced by a_t − E_t[a]" and "Here a is the derivative of a token's logit" (P14).
- Candidates vs repeats: ν_t is defined through candidates ("the expected number of candidates chosen after step t", P15), but P17 explains it with "Entering a repetitive stretch makes many future repeats likely".
- Mixed citation styles: arXiv identifiers ("[2503.21929]") alongside author-year ("[Glynn 1990; Williams 1992; Warren & Allen 2012]"). Also "Kempton and Burrell" (P5) vs "Warren & Allen", "Wiseman & Rush" (P4, P5).
- Possible citation-ID typo: "[2310.10226]" (P2) and "[2301.10226]" (P40, watermarking) differ only in digit order. Check that the P2 identifier is the intended paper.
- One appendix cited for two different analyses: "across temperatures the ratio did not drift systematically (Appendix F10)" (P26) and "the interval of the difference excluded zero for two of the nine models (Appendix F10)" (P35; also P36). This may be correct, but it is worth checking.
- θ called a "divisor" (P27: "predicted the divisor θ", "divisors up to about 1.1"), although P7 says it "divides their positive logits by θ and multiplies their negative logits by θ".
- "two-state Boltzmann system" (P3) vs "two-state Boltzmann model" (P21) vs "equilibrium two-state model" (P26, P39). Minor.

**Grammar, wording and missing words**
- P14: "and E_t the expectation under the step-t distribution": the verb "is" is missing (elliptical).
- P1: "we replicated on a new set of news articles in seven open base models": "in" should be "with" or "for", and "replicated" has no object (replicated the test).
- P6: "centered on a design temperature chosen for that model (Sec. IV B), with no truncation such as top-k or top-p": misplaced modifier. "with no truncation" belongs with "Sampling was plain softmax".
- P8: "a step enters the repetitive state when the fraction reaches 1/2": the text enters the state, not the step.
- P34: "Steps where the model almost surely chooses a candidate, with q_t above 0.9": "almost surely" is misused as a technical term.
- P23: "The crossover temperature lies between 0.44 and 0.66": present tense, while the surrounding results are in the past tense. Minor.
- P9: "towards", while the rest uses American spelling ("labeled", "favoring", "renormalized"). Trivial.

**Wording that contradicts or strains against other sentences**
- P28 opens "The shift of T_× itself, which a user would target, was larger than intended", but then reports "The median then became 0.87, slightly below the target". The flat first claim holds only under the line-fit location. The abstract says the less specific "that temperature did not always move as intended".
- P1: "The derivative also shows that the response arises mainly at the minority of steps where the model is uncertain" vs P34: "This proportionality is an empirical finding, not a consequence of the form of the score." The abstract credits the derivative itself with what P34 calls an empirical finding.
- P31: "for temperature it was at most 2.3% (s_temp). For ln K, the compensator term thus cancels about 99% of the tilted term." For temperature the cancellation can be as low as about 97.7%, so "about 99%" fits s_pen (0.7–1.5%) but not both fractions, as the "thus" suggests.
- P17: "For h > 0, exponential tilting also rewards each choice ..." is stated for h > 0, while the penalties studied use "h < 0 discourages repetition" (P7). This is not a contradiction, but the abstract states the same mechanism without a sign ("removes the credit"), which leaves the sign convention unclear.
- P26: "its intervals at h* and at 2h* both reached just above 1.25". The procedure in P25 never mentions a 2h* condition, so it appears here unannounced.
