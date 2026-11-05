# Answers on manuscript_main_text.md

General impression first: the text has almost none of the usual hype vocabulary ("crucially", "novel", "remarkably", "landscape"). Where it reads as machine-generated, the cause is different: every sentence is about the same length and built the same way, terms are defined by stacking appositives ("X, the Y") and "We call ... the ...", parenthetical appendix pointers come one after another, and the same caveat comes back in several places. The most human passages are the ones that admit something specific and awkward about this particular study.

## 1. Per-paragraph ratings (0 = reads as human, 3 = reads strongly as machine-generated)

| P | Rating | Reason (for ratings of 1 or more) |
|---|---|---|
| P1 | 2 | A string of facts with almost no connective tissue. Each sentence carries one result or definition: "Here N counts the candidates chosen, Q sums over steps the probability of choosing one, and T is the temperature." "It removes the credit each choice would receive for the repeats it makes likely later" is a tidy summary-aphorism. The last sentence, "The derivative also shows that ...", is tacked on as a further item in the list. |
| P2 | 1 | The literature review is a run of short sentences of one shape, each ending in one citation: "Adaptive sampling schemes have been designed partly to avoid it", "Training objectives that discourage repetition have been proposed", "Penalties derived from ... have also been proposed." |
| P3 | 1 | A setup-then-contrast template: "These works describe generated text but do not predict how an intervention such as a penalty acts. A natural shortcut would ..." |
| P4 | 2 | The closing sentence lists contributions in a triple: "we measure how much ... survives step-wise normalization, which explains why the effect is small, and locate where in a text a penalty acts". |
| P5 | 2 | A field-by-field survey ("In sequence labeling, ... In statistical physics, ...") followed by the stock closer "Here we bring these ideas to a repetition penalty and test the resulting prediction prospectively." |
| P6 | 1 | A definition that holds back its key word until the very end: "We call one round of computing the next-token logits, adjusting them, normalizing them and sampling one token a step." |
| P7 | 1 | A justification sentence that restates the sentence before it: "Repeats define the states below, and candidates define the penalty. Keeping them separate makes the definition of the states independent of the penalty." Also the appositive "acts on a third set, every token seen before". |
| P8 | 1 | Stacked appositive definitions, "per ordinary step, the entry rate, and switches out of it per repetitive step, the exit rate", and two "We call ..." sentences in one paragraph ("We call the temperature ...", "We call these the usable temperatures"). |
| P9 | 2 | The "X, not Y" negation pattern repeated as a defensive list: "The two states are a labeling, not distinct phases", "K is also not the rate ratio of a stationary two-state Markov chain". |
| P10 | 0 | |
| P11 | 0 | (equation) |
| P12 | 1 | A semicolon aside that answers a question nobody asked: "this identity also holds with a bounded stopping time, such as an end-of-sequence token kept in the normalization, but we did not use one." |
| P13 | 0 | (equation) |
| P14 | 1 | Elliptical "Here ..." glossing: "Here a is the derivative of a token's logit with respect to λ, and E_t the expectation under the step-t distribution." |
| P15 | 1 | Naming the same object in two fields' vocabularies within one sentence: "exponential tilting in the language of statistics, and a tilted ensemble of trajectories in statistical physics", then "the static susceptibility of the path ensemble". Plus the tutorial cue "To see what the difference means". |
| P16 | 0 | (equation) |
| P17 | 2 | An intuition line placed before the formal content ("Exponential tilting rewards each choice also for the future repeats it makes likely"), an expectation hedge ("so this is where we expect the two to differ most"), and a "Writing ... , we call ..." definition to close. |
| P18 | 0 | |
| P19 | 0 | (equation) |
| P20 | 1 | A clipped contrast fragment, "Exponential tilting by exp(−εU) does not.", followed by a long definition that points forward to a later section ("with the weights of the equilibrium predictions of Sec. III D"). |
| P21 | 1 | The balanced semicolon contrast "The prediction needs an energy for each step and changes with that choice; the exact identities and s_pen do not." |
| P22 | 0 | |
| P23 | 0 | |
| P24 | 0 | |
| P25 | 1 | Another coinage by "We call": "We call T* = T_× − 0.10 the design temperature." Otherwise concrete. |
| P26 | 2 | Reads like a robustness checklist, each item with its own pointer: "For comparison, ... about 0.01", "The equilibrium two-state model, evaluated after the tests, would have ...", "Under eight other labelings ... (Table F2)", "across temperatures the ratio did not drift systematically (Appendix F10)". |
| P27 | 1 | The conclusion is announced with "therefore" ("The prediction therefore holds only for divisors up to about 1.1, the weakest values used in practice") although the paragraph never gives the numbers that would support it. Also the vague closer "did not decide or could not be run". |
| P28 | 0 | |
| P29 | 0 | (caption) |
| P30 | 0 | (caption) |
| P31 | 2 | Restates theory as if it were a finding: "Nearly all of the covariance of the switching counts with N comes through the future repeats that each choice makes likely, and Q carries the same expected future repeats (Sec. III B)." Also "Both fractions are first-order quantities, taken at zero penalty and zero tilt." |
| P32 | 2 | A list of sensitivity checks with "changes them little" and "consistent with fewer future repeats following each choice", ending in the label "Both fractions are exploratory (Table C1)." |
| P33 | 1 | An add-on caveat sentence in a caption: "Unlike s_pen, s_temp depends on the energy assigned to a step." |
| P34 | 1 | A carefully balanced pair of parallel sentences ("make up more than four-fifths ... but carry only about a fifth", "make up about a tenth ... but carry about four-fifths"), then a pre-emptive methodological disclaimer ("This proportionality does not follow from the score alone"). |
| P35 | 1 | "in the smaller and the larger models alike"; the result is reported in a fixed pattern of good news, then a qualifier, then a detail with a pointer. |
| P36 | 0 | (caption) |
| P37 | 2 | Opens by recapping results. Then "not on the number of repeats alone" (the "not X alone" pattern), then a related-work aside closed by a concessive clause, "although that theory separates terms by time reversal, which autoregressive sampling lacks." |
| P38 | 1 | "We attribute the contrast to scale, without having tested it" reads as human. The explanation after it is very neat and symmetrical, though: "For a whole text, ... so small changes at many steps compound. ln K is instead ... so its response stays close to linear." |
| P39 | 1 | A semicolon-balanced pair of outcomes: "it overstated the effect by about 45%; with the logits it understated it, or predicted the wrong sign". |
| P40 | 2 | A generality claim with "any" and stacked "such as" examples ("applies to any small additive change to the logits, such as logit biases and watermarking biases"), followed by "It also carries over to ...", a caveat ("We tested only ..."), and a cost argument. |
| P41 | 2 | A long run of one-caveat sentences in identical cadence, most of which repeat earlier text (see Q3). The disclosure sentences at the end are specific and read as human. |

## 2. Paragraphs that read most clearly as written by a person

- **P28.** It opens with a blunt, reader-centered sentence, "A user would care about where T_× ends up." It reports an awkward result without spin ("it moved further than intended, by a median of 1.3 times the intended shift") and names a culprit ("GPT-2-XL most of all"). It ends by admitting ignorance: "for reasons we have not identified."
- **P24 (Fig. 1 caption).** The selection rule "chosen as the first output with two to six switches" is the kind of specific, slightly fussy disclosure a person adds after being burned once by cherry-picking accusations.
- **P25.** "These documents did not cover Fig. S1, whose runs were made earlier." and "Of the two factors of the prediction, only h* was fixed beforehand, while g came from the unpenalized runs of the same test." These volunteer weaknesses specific to this study rather than generic caveats.
- **P22.** "Consider a temperature where both kinds of stretch are common." This is a direct invitation to the reader, and the description that follows is concrete ("at about one step in five").
- **P38**, for "We attribute the contrast to scale, without having tested it.", and the end of **P41**, for "They were checked by tests, an independent reimplementation and other AI instances, not line by line by the author." Both are unusual, specific admissions.
- The non-idiomatic "did not decide" (P1, P26, P27, P41) is also a human fingerprint. A language model would almost always write "was inconclusive". It is still worth fixing (Q6), but it does not read as machine text.

## 3. Deletable and repeated sentences

**Could be deleted without losing anything a reader needs**
- P5: "The formula is standard in simulation and reinforcement learning." The citations in P4 already say this.
- P7: "Keeping them separate makes the definition of the states independent of the penalty." It follows directly from the previous sentence.
- P12: "this identity also holds with a bounded stopping time, such as an end-of-sequence token kept in the normalization, but we did not use one." This could go to an appendix.
- P15: "and a tilted ensemble of trajectories in statistical physics". P5 already makes this link.
- P17: "Entering a repetitive stretch makes many future repeats likely, so this is where we expect the two to differ most." This is a speculative intuition that is never tested.
- P26: "For comparison, the tilted response of Sec. IV C would give a ratio of about 0.01." This points forward to Sec. IV C and restates it. Also "The equilibrium two-state model, evaluated after the tests, would have fallen outside the range for six of the nine models (Appendix H)." P39 and Appendix H cover it.
- P31: "Both fractions are first-order quantities, taken at zero penalty and zero tilt." P17 and P20 ("at ε = 0") already define this. Also the last sentence, "Nearly all of the covariance ... (Sec. III B)", which restates P17.
- P32: "Pooling the covariances of the step-wise response over articles, or computing s_temp from the step-wise temperature identity instead of the slope b, changes them little." This belongs in an appendix.
- P33: "Unlike s_pen, s_temp depends on the energy assigned to a step." This duplicates P21 and P39.
- P37: the first sentence, "In the registered tests, the derivative estimated from unpenalized samples predicted the change in switching, ...", recaps Sec. IV B and is repeated again in P41.
- P40: "We tested only the two repetition penalties, without truncation." This repeats P6 ("with no truncation") and P41.
- P1: the final sentence, "The derivative also shows that the response arises mainly at the minority of steps where the model is uncertain whether to repeat." It is a secondary result and could be cut from the abstract.

**Sentences that repeat something said elsewhere**
- Two states are a labeling: P9 "The two states are a labeling, not distinct phases." / "The balance drifts towards repetition along a text", and P41 "The two states are a labeling of a continuum, K averages over a drift along the text".
- T_× shift: P1 "the temperature at which the entry and exit rates are equal, which the design targeted, did not always move as intended", P28 "for reasons we have not identified", and P41 "The shift of T_× itself, which a user would target, did not always match the intended 0.10".
- Multiplicative penalty limit: P1 "followed the prediction only up to the weakest strengths used in practice", P27 "holds only for divisors up to about 1.1, the weakest values used in practice", and P41 "the prediction held only up to the weakest divisors used in practice (Appendix F5)".
- Instruction-tuned models: P27 "For two instruction-tuned models the test did not decide or could not be run." and P41 "For instruction-tuned models the test did not decide".
- Derivative taken from the same test: P1 "With the derivative estimated from the unpenalized runs of the same test", P25 "g came from the unpenalized runs of the same test", and P41 "Because the tests estimated the derivative from their own unpenalized runs".
- Exploratory carry-over: P37 "in an exploratory analysis a prediction from the design articles alone carried over to new articles (Appendix F4)" and P41 "A prediction made before any test data existed was examined only in an exploratory analysis (Appendix F4)".
- Length dependence: P32 "Shorter texts keep a larger fraction, 1.8% for the first 256 tokens" and P41 "s_pen depends on the length of the text".
- Unreliable equilibrium shortcut: P3 "Appendix H shows why this shortcut is not a reliable guide", P21 "The prediction needs an energy for each step and changes with that choice", P33 "Unlike s_pen, s_temp depends on the energy assigned to a step", and P39 "depends on the energy assigned to a step, which the sampler does not define".
- Future-repeats intuition: P1 "It removes the credit each choice would receive for the repeats it makes likely later", P17 "Exponential tilting rewards each choice also for the future repeats it makes likely", and P31 "comes through the future repeats that each choice makes likely".
- "Not the number of repeats alone": P37 "The prediction depends on how far each choice departed from the model's own probability of repeating" restates the N − Q identity of P1 and P11.
- E_t is defined twice: P14 "E_t the expectation under the step-t distribution" and P20 "where E_t is the expectation under the step-t distribution at the current temperature".

## 4. Sentences that are hard on a first reading

- P1: "A repetition penalty in a language model is applied, and the distribution renormalized, at every sampling step." The gapped verb ("the distribution [is] renormalized") interrupts the main clause.
- P1: "Although the change in the log ratio was predicted, the temperature at which the entry and exit rates are equal, which the design targeted, did not always move as intended." The subject is buried under two relative clauses. The reader has not been told that the design targeted a temperature at all.
- P1: "It removes the credit each choice would receive for the repeats it makes likely later." This is a metaphor ("credit") for an operation that has not been described yet.
- P4: "With the same formula we measure how much of a whole-text, Boltzmann-like response survives step-wise normalization, which explains why the effect is small, and locate where in a text a penalty acts." There are three actions, and "the effect is small" refers to an effect the reader has not yet been told is small.
- P6: "We call one round of computing the next-token logits, adjusting them, normalizing them and sampling one token a step." This is a garden-path sentence: "a step" comes only at the very end.
- P7: "before division by T, it divides their positive logits by θ and multiplies their negative logits by θ". "Before" appears twice in close succession with different meanings ("seen before", "before division").
- P8: "At each temperature we count switches into the repetitive state per ordinary step, the entry rate, and switches out of it per repetitive step, the exit rate." The appositives look like list items.
- P12: "Q is its compensator, the sum of the probabilities of choosing a candidate given the text so far". "Compensator" is a martingale term of art used without warning, and "its" has to be traced back to N.
- P14: "Applied to the four counts that define K". The four counts are never listed (P8 defines K from rates). The phrase "where K is defined by ratios of expected counts" also quietly shifts the definition.
- P15: "It gives the response Cov(O, N)/T, the static susceptibility of the path ensemble." This is physics jargon with no gloss.
- P17: "Its step probabilities are multiplied by exp(hc_t/T) and by the expected future factor E[exp(hN_{>t}/T) | x_≤t], with N_{>t} the candidates chosen after step t, and renormalized." The final "and renormalized" has drifted far from its verb. Also "this is where we expect the two to differ most": the referents of "this" and "the two" are unclear.
- P20: "Because each step's expectation is subtracted, adding to U_t any constant that depends only on the history, for example replacing U_t by −z_t, leaves this score unchanged." The reader must already know that −z_t and −log p differ only by a history-dependent log-normalizer.
- P20: "Over several temperatures we sum numerator and denominator with the weights of the equilibrium predictions of Sec. III D, s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u)." It points forward to a later section, uses symbols (b, g_C^tilt) that are never defined, and redefines a symbol defined one sentence earlier.
- P21: "The prediction needs an energy for each step and changes with that choice; the exact identities and s_pen do not." What "that choice" means is clear only after reading P39.
- P26: "its intervals at h* and at 2h* both reached just above 1.25". 2h* has not been introduced, and the reader does not know why a test was run at 2h*.
- P26: "A separate test repeated the whole procedure for two 7-billion-parameter models, where the rule applied to the mean ratio, 0.96 (0.84–1.11)." "Where" is an odd connector, and the number hangs at the end.
- P27: "The prediction therefore holds only for divisors up to about 1.1". The predicted θ* values that make "about 1.1" follow are not given, so the "therefore" cannot be checked.
- P28: "With T_× from a straight-line fit of ln K against 1/T, as in the design, it moved further than intended, by a median of 1.3 times the intended shift in the smaller models." The opening phrase delays the subject, "it" is ambiguous, and three modifiers are stacked.
- P32: "These fractions use covariances within articles." "Within articles" is not explained, so the next sentence about "pooling ... over articles" is hard to place.
- P34: "four-fifths" is used twice in consecutive sentences for different things (share of steps, then share of response), which invites misreading. "This proportionality does not follow from the score alone, which only bounds a step's contribution by the square root of its variance" leans on an unstated Cauchy-Schwarz argument.
- P35: "the difference excluded zero for two of the nine models". Elliptical: it means the interval of the difference excluded zero.
- P37: "although that theory separates terms by time reversal, which autoregressive sampling lacks". This is too compressed for readers outside response theory.
- P38: "There, the exact probability ratios of the unpenalized samples concentrated on few samples, with an effective sample size of 1.3 to 2.1 out of four per prompt" and "We attribute the contrast to scale". The contrast itself (importance weights degenerate while the first-order prediction holds) is never stated. "Four per prompt" is also the first mention that each prompt had four samples.
- P40: "Locating T_× required unpenalized runs at several temperatures, comparable in cost to trying a few penalty strengths, but the same runs give the response to any such change." The point of the cost comparison is unclear.

## 5. Three changes that would most improve how natural the prose reads

1. **State each caveat once.** Keep each limitation in a single place, either where it first arises or in P41, not in both. This applies to the labeling/drift caveat (P9 and P41), the T_× shift (P1, P28, P41), the multiplicative limit (P1, P27, P41), the instruction-tuned models (P27, P41), "exploratory" (P32, P34, P37, P41), and no truncation (P6, P40, P41). Repetition in identical cadence is the strongest machine-like signal in the text. *Does it remove anything a reader needs?* No, as long as one copy of each caveat survives and stays as specific as now.
2. **Loosen the definition machinery.** Replace the stacked appositives ("..., the entry rate, ...", "..., the static susceptibility of the path ensemble") and the many "We call ... the ..." and "Here X is ..., and Y the ..." sentences with fewer, connected sentences that give the reason before the term. Do the same for the long strings of parenthetical appendix pointers. *Does it remove anything a reader needs?* No. Every definition stays; only the sentence architecture changes. Several sentences in Q4 would become easier at the same time.
3. **Move robustness checklists and side comparisons out of the results paragraphs.** Candidates are P26 "For comparison ...", the post-hoc equilibrium comparison, "Under eight other labelings ...", "did not drift systematically", and P32's pooling/identity checks. Move them to the appendices or compress them into one sentence. P31's and P37's restatements of the theory can also go. *Does it remove anything a reader needs?* Mostly no, since the details are already in the appendices. One sentence must stay in the main text: "Five of the nine per-model intervals reached above 1.25". It is a substantive qualification of the headline result. The labeling-robustness result should also survive in at least one clause, because the states are admitted to be an arbitrary labeling.

(A smaller fix: replace "did not decide" (four times) with a standard term such as "was inconclusive". This removes nothing.)

## 6. Typos, grammar, missing words, inconsistent terms or notation, contradictions

**Grammar and missing words**
- P6: "from nine open base models between 0.4 and 7.6 billion parameters". A word is missing ("with between", or "of 0.4 to 7.6 billion parameters").
- P7: "This additive penalty is not an option of common libraries". The preposition should be "in common libraries".
- P1, P26, P27, P41: "did not decide" (e.g. "After a first test that did not decide"). Non-idiomatic for an inconclusive test.
- P1: "replicated on new news articles". "new news" is an awkward repetition.
- P3: "Several studies have therefore described generated text in the language of statistical physics". "Therefore" claims a causal link that is not there; the preceding sentence is only a definition.
- P15: "is exponential tilting in the language of statistics, and a tilted ensemble of trajectories in statistical physics". A zeugma: one "is" serves two predicates of different kinds.
- P17: "Exponential tilting rewards each choice also for the future repeats it makes likely". "Also" is misplaced ("also rewards each choice for ...").
- P17: "with N_{>t} the candidates chosen after step t". This should be "the number of candidates chosen after step t".
- P28: "it moved further than intended". For physical or numeric distance, "farther" is the usual choice (minor).
- P35: "the difference excluded zero". The interval of the difference excluded zero.

**Undefined or inconsistent notation**
- P20, P21, P32: **b** is used ("s_temp = b/b_tilt", "g_C(T) = (b/ΔE) Δc_T/T", "instead of the slope b") but never defined. It is presumably the van 't Hoff slope ∂ ln K/∂β.
- P20: **ε** appears in "Exponential tilting by exp(−εU) does not." before it is defined ("at ε = 0" comes later).
- P20: **g_C^tilt** in "Σ_u g_C^tilt(T_u)" is never defined.
- P20: **s_temp** is defined twice in different ways, "s_temp = b/b_tilt" and "s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u)".
- P10 vs P34: the per-step score is "(c_t − q_t)/T" in P10 but "Each step contributes c_t − q_t to the score" in P34. The 1/T is missing.
- P10: "p_t^h(x_t | x_<t)" vs "log p_t^h(x_t)" in the next sentence. The conditioning is dropped.
- P14: "c_t − q_t is replaced by a_t − E_t[a]" vs "Here a is the derivative of a token's logit". a_t and a are used without stating that a_t is a at the sampled token.
- P14 vs P20: E_t is defined twice, the second time with an added qualifier ("at the current temperature").
- P8 vs P14: K is defined as "entry rate/exit rate" (empirical rates) in P8 but as "defined by ratios of expected counts" in P14.
- P14: "the four counts that define K". P8 never names four counts.
- P38: "out of four per prompt". The number of samples per prompt is never given in Setup (P6).
- Citation style is mixed: arXiv IDs ("[2406.05335]") alongside author-year ("[Glynn 1990; Williams 1992; Warren & Allen 2012]"). One reference is cited by name and by ID at once: "Kempton and Burrell described ... [2503.21929]".
- Supplementary labels are mixed: "Fig. S1" next to "Table F2", "Table C1" and "Appendix F10". Appendix F10 is cited both for "across temperatures the ratio did not drift systematically" (P26) and for the entry/exit split difference (P35, P36). This may be intended, but it looks like a possible mis-pointer.

**Inconsistent terms**
- **Repeat vs candidate.** P7 explicitly separates them ("Repeats define the states below, and candidates define the penalty"), but later text slides between the two:
  - P3 "treat the penalty as an energy per repeat" vs P21 "with the penalty as an energy per chosen candidate";
  - P1 "the repeats it makes likely later", P17 "the future repeats it makes likely", P31 "the future repeats that each choice makes likely", and P32 "fewer future repeats following each choice", although the mathematics (N_{>t}, ν_t) counts candidates chosen, not repeats;
  - P34 "Steps at which the model has almost decided to repeat, with q_t above 0.9", P1 "uncertain whether to repeat", and P37 "the model's own probability of repeating", although q_t is the probability of choosing a candidate.
- **Summation range.** P25 "summed over the usable temperatures" vs P30 "summed over the test temperatures".
- **Preregistration wording.** "registered in advance" (P1), "registered range" (P1), "dated documents" (P25), "committed in advance" (P26), "preregistered" (P29, P30), "registered tests" (P41), "prospectively" (P5), "Prospective tests" (P29).
- **Model groups.** "seven open base models" / "smaller models" / "first seven models", and "two larger models" / "7-billion-parameter models", vs P6 and P41 "up to 7.6 billion parameters". "7-billion" and "7.6 billion" are used loosely for the same group.
- **The whole-text response.** It is called "whole-text, Boltzmann-like response" (P4), "tilted response" (P26, P31), and "the first-order response the same penalty would produce on whole texts" (P1).

**Wording that contradicts or is in tension with other sentences**
- P28 "for reasons we have not identified" vs P41 "for reasons we have not fully identified".
- P27 "For two instruction-tuned models the test did not decide or could not be run." vs P41 "For instruction-tuned models the test did not decide". P41 drops "could not be run".
- P9 "The windowed repeat fraction has peaks near zero and one" vs P22 "a narrow peak at exactly one".
- P9 and P41 "a labeling of a continuum" vs P22's description of two distinct peaks with "a shallow trough in between". This is not strictly contradictory, but "continuum" sits uneasily with a two-peaked distribution.
- P20 says that changing the energy assigned to a step "leaves this score unchanged", while P33 says "Unlike s_pen, s_temp depends on the energy assigned to a step". The two are reconcilable (only the tilted denominator and the g_C weights depend on the energy), but the text never says so.
- P28 "This brought the median to 0.87, so curvature accounts for most of the overshoot". Going from 1.3 to 0.87 removes the whole overshoot and turns it into a 13% undershoot, so "most of the overshoot" understates the change.
- P26: "Five of the nine per-model intervals reached above 1.25 (Table I)" appears inside the paragraph about the seven-model replication, before the two-model test is introduced. "Nine" therefore mixes the two tests, and the two-model test was decided on a different statistic (the mean).
- P30 "(120 articles in each test)" vs P25 and P26, where the first test used "the 40 articles from which h* had been computed". This is consistent only if "each test" means the two tests in Table I, which the caption does not say.
- P1 presents the 99% cancellation as a derived result ("Step-wise normalization therefore cancels about 99% ..."), while P32 says "Both fractions are exploratory (Table C1)". The abstract does not flag the result as exploratory.
- P26 and P39: P26 says the equilibrium model "would have fallen outside the range for six of the nine models" without saying which energy was used, while P39 says the outcome depends on that choice ("With −log p as the energy it overstated the effect by about 45%; with the logits it understated it"). P26 cannot be interpreted without knowing which energy it used.
