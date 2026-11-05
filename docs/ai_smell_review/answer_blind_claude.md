# Blind review of manuscript_main_text.md (answers to the five questions)

## 1. Per-paragraph rating (0 = reads as written by a person, 3 = reads strongly as machine-generated)

| Para | Rating | Reason (for ratings of 1 or more) |
|---|---|---|
| P1 | 2 | Same-shaped declarative sentences, each holding one fact, then a run of caveats added at the end ("The widely used multiplicative penalty followed the prediction only at...", "The temperature at which the entry and exit rates are equal did not always move as intended.", "The derivative also shows that..."). Very dense definitions in the middle ("the tokens that have previously followed the current last two tokens"), and an odd coined adjective ("After an undecided first test"). |
| P2 | 2 | A catalogue of citations in one passive pattern, one sentence per source: "have been designed partly to avoid it", "have been proposed", "have also been proposed". Few links between the sentences. |
| P3 | 1 | Set phrasing that comes back almost word for word in P39: "A natural shortcut would treat the penalty as an energy..." / "is not a reliable guide". The causal "therefore" in "Several studies have therefore described" is weakly motivated. |
| P4 | 1 | The last sentence is a three-part summary of contributions with a mismatched agent: "The same formula shows why the effect is small, by measuring how much ... and locates where in a text a penalty acts." Tenses mix ("We use ... and we tested"). |
| P5 | 2 | Related work as a list of "X are standard in Y [refs]" sentences ("are standard in simulation", "are standard in statistical physics"), closed by a stock "Our contribution is..." sentence that repeats P4. |
| P6 | 1 | Definition template with the term at the end: "We call one round of computing the next-token logits, adjusting them, normalizing them and sampling one token a step." Points forward to notation not yet defined ("including q_t", "design temperature"). |
| P7 | 1 | Balanced justification sentences that answer a question before anyone asks it: "Repeats define the states below, and candidates define the penalty. Keeping them separate makes the definition of the states independent of the penalty." |
| P8 | 1 | A chain of "We call ..." definitions and definitions placed in apposition ("the entry rate", "the exit rate", "We call the temperature ...", "We call these the usable temperatures"). |
| P9 | 1 | Opens with the "X, not Y" pattern ("The two states are a labeling, not distinct phases."), then stacks disclaimers. The specific content (bimodality criterion fixed in advance, drift along the text) offsets this. |
| P10 | 0 | |
| P11 | 0 | (equation) |
| P12 | 1 | A defensive aside that the reader does not need: "the identity also holds with a bounded stopping time, such as an end-of-sequence token kept in the normalization, but we did not use one." |
| P13 | 0 | (equation) |
| P14 | 0 | |
| P15 | 1 | Names the same object in two fields: "is exponential tilting in the language of statistics, and a tilted ensemble of trajectories in statistical physics"; three appositives in one sentence ("plus ν_t, the change at step t ..., a martingale increment"). |
| P16 | 0 | (equation) |
| P17 | 1 | A third cross-field gloss ("In machine-learning terms, this is the difference between..."), and the slogan "rewards each choice also for the future repeats it makes likely", which recurs in P1, P31 and P37. |
| P18 | 0 | |
| P19 | 0 | (equation) |
| P20 | 1 | A contrast added to head off an objection ("The penalty score (N − Q)/T of Sec. III A contains no energy and does not depend on this choice."), repeated in P21 and in the Table II caption. |
| P21 | 1 | Neat closing antithesis: "The prediction needs an energy for each step and changes with that choice; the exact identities and s_pen do not." |
| P22 | 0 | |
| P23 | 0 | |
| P24 | 0 | |
| P25 | 1 | Restates the design after giving it: "Passing therefore requires the first-order prediction to hold at a penalty large enough to matter." The interrupted syntax "it predicts, from unpenalized runs on 40 design articles, the penalty h* that..." |
| P26 | 1 | A string of reassurances at the end: "Under eight other labelings ... the median stayed inside the range", "across temperatures the ratio did not drift systematically", "A separate test repeated the whole procedure..."; a near-tautology: "Individual models had wider intervals". |
| P27 | 0 | |
| P28 | 0 | |
| P29 | 0 | |
| P30 | 0 | |
| P31 | 2 | Re-explains the mechanism from P17 with a tidy closing clause: "Q carries the same expected future repeats, so the two terms cancel, leaving only the response to the choices themselves." A misplaced definitional sentence ("Each fraction is a ratio of derivatives at h = 0."). |
| P32 | 2 | Vague reassurance with no numbers ("Other conventions and the local form of s_temp give the same picture", "consistent with fewer future repeats following each choice") and in-house jargon ("Both fractions are exploratory records"). |
| P33 | 0 | |
| P34 | 1 | Mirror-image fractions that read as composed for effect: "more than four-fifths of the steps ... only about a fifth of the predicted response ... about a tenth of those steps but carry about four-fifths". |
| P35 | 1 | A bookend summary that restates the opening sentence: "A penalty thus both makes repetition harder to start and makes loops break more readily." |
| P36 | 0 | |
| P37 | 3 | Recaps the abstract almost sentence for sentence, in polished parallel antitheses: "Step-wise normalization keeps the effect small, because it changes what a penalty is. Acting on whole texts, a penalty would also reward ... Acting step by step, it reaches only the choice itself"; "not on the number of repeats alone"; then a new related-work point appears at the end ("the discrete-time counterpart of the escape-rate term"). |
| P38 | 1 | A smooth explanation that has not been tested ("so small changes at many steps compound", "so its response stays close to linear"). The candid "We attribute the contrast to scale, without having tested it." pulls the other way. |
| P39 | 2 | Repeats P3's wording ("A natural shortcut ... is not a reliable guide") and ends on an aphorism: "Its verdict is a property of the chosen energy, which the sampler does not define." |
| P40 | 2 | The opening connective has no clear antecedent ("In the same way, the derivative applies to any small additive change"). A broad "any" claim and a selling pivot: "The same runs, however, give the response to any such change and show where in the text it arises." |
| P41 | 1 | Checklist cadence, one limitation per sentence. The admissions themselves are specific and unflattering, which reads as human ("Model revisions were not pinned in the code", "not line by line by the author"). |

## 2. Paragraphs that read most clearly as written by a person

- **P22**: concrete, physical description with varied adjectives: "a high peak near zero, a narrow peak at exactly one and a shallow trough in between", and a plain summary "Most outputs switch between the two kinds at least once."
- **P27**: short, uneven sentences that report an unflattering result directly: "At twice the predicted strength the ratio was still 1.00, but at eight times it fell below the range. So the prediction holds only for divisors up to about 1.1, at the lower end of the values used in practice." It also has a sentence opening with "So" and a terse "did not decide or could not be run".
- **P28**: it starts from the user's concern ("A user would care about where T_× ends up."), reports an overshoot, explains part of it with an analysis done after the tests, and leaves the rest open: "GPT-2-XL most of all ... for reasons we have not identified."
- **P26 (first half)**: "did not decide, because both of its intervals reached just above 1.25" is a specific, slightly awkward, honest account of a near miss.
- **P24 (caption)**: the selection rule "chosen as the first output with two to six switches" is the kind of procedural detail a person who made the figure writes down.
- **P38 (one sentence)** and **P41 (last three sentences)**: "We attribute the contrast to scale, without having tested it."; "Model revisions were not pinned in the code, although the revisions used can be identified"; "not line by line by the author". These admissions are idiosyncratic. They work against the authors' interest and are not phrased to reassure.

What these share: specific numbers or rules tied to what was actually done, admitted failures or unknowns left unresolved, uneven sentence lengths, and no closing summary sentence.

## 3. Deletable sentences and repeated content

**Could be deleted without losing information a reader needs**
- P25: "Passing therefore requires the first-order prediction to hold at a penalty large enough to matter." (follows from the previous two sentences; said again in P1 and P37)
- P26: "Individual models had wider intervals (Table I)." (per-model intervals are necessarily wider than the interval of the median over seven models)
- P31: "Each fraction is a ratio of derivatives at h = 0." (already set by the definitions in P17 and P20)
- P35: "A penalty thus both makes repetition harder to start and makes loops break more readily." (restates "The penalty acts at both ends of a repetitive stretch" and the entry/exit split)
- P12: the clause "the identity also holds with a bounded stopping time, such as an end-of-sequence token kept in the normalization, but we did not use one" (not used anywhere; could go to an appendix)
- P5: "Our contribution is a tested first-order prediction ..., together with a measurement of how much of the whole-text response step-wise normalization removes." (repeats the last two sentences of P4)
- P37, first two sentences: "The effect of a repetition penalty on the switching statistics of repetition can be computed from samples generated without it. In the prospective tests, this first-order prediction matched the realized change at penalties large enough to move T_×." (repeat the abstract and Sec. IV B)
- P28: "A user would care about where T_× ends up." is optional motivation, though it helps the paragraph read naturally.

**Sentences that repeat something said elsewhere**
- The mechanism behind the cancellation is stated four times: P1 "It removes the credit each choice would receive for the repeats it makes likely later."; P17 "Exponential tilting rewards each choice also for the future repeats it makes likely."; P31 "The tilted response credits each choice also with the future repeats it makes likely (Sec. III B) ... leaving only the response to the choices themselves."; P37 "Acting on whole texts, a penalty would also reward each choice for the repeats it makes likely later. Acting step by step, it reaches only the choice itself".
- "about 99%" is stated three times: P1, P31 ("Thus the compensator cancels about 99% of the first term."), P37.
- The equilibrium shortcut: P3 "A natural shortcut would treat the penalty as an energy per repeat ... Appendix H shows why this shortcut is not a reliable guide."; P21; P39 "A natural shortcut, an equilibrium two-state model that treats the penalty as an energy, is not a reliable guide." (P39 is the only place with the numbers, "about 45%" and "the wrong sign", so it is the one to keep.)
- That the energy choice matters for s_temp but not s_pen appears three times: P20 "The penalty score (N − Q)/T ... does not depend on this choice."; P21 "the exact identities and s_pen do not."; P33 caption "Unlike s_pen, s_temp depends on the energy assigned to a step."
- The derivative is estimated from the same test's unpenalized runs: P1 "With the derivative estimated from the unpenalized runs of the same test", P25 "Only h* was fixed beforehand, and g was estimated from the unpenalized runs of the same test.", P41 "Because the tests estimated the derivative from their own unpenalized runs, they test whether the first-order prediction holds at h*." (P41 adds the consequence, so it is the most useful of the three.)
- The multiplicative penalty works only at low strength: P1, P27, P41 ("For the multiplicative penalty, the prediction held only for divisors below those common in practice (Appendix F5).").
- T_× did not move as intended: P1, P28, P41 ("did not always match the intended 0.10, for reasons we have not fully identified"). Note that P28 says "not identified" and P41 says "not fully identified".
- Instruction-tuned models: P27 and P41.
- States are a labeling, and K averages over drift: P9 and P41.
- "where in a text the penalty acts": P1 (last sentence), P4, P34, P40.
- Table I and Fig. 2 captions (P29, P30) redefine h*, the ratio and the band from P25. Captions are expected to stand alone, so this is acceptable.

Repetition in the limitations paragraph (P41) is expected, since that is where caveats should be collected. The repetitions in P31, P37 and P39 of points made in P17 and P3 are the ones that add nothing.

## 4. Sentences that are hard on a first reading, and why

- **P1** "We show that a first-order prediction of its effect on how text switches between repetitive and ordinary stretches, computed from samples generated without the penalty, holds at penalties large enough to shift that switching." The subject and the verb ("prediction ... holds") are about 25 words apart, with two modifiers in between.
- **P1** "For an additive penalty h on repeat candidates, the tokens that have previously followed the current last two tokens, the derivative of a text's log-probability with respect to h, at h = 0, is (N − Q)/T." Three appositive or parenthetical phrases come before the verb, and the gloss on "repeat candidates" reads at first like the subject.
- **P1** "After an undecided first test" uses "undecided" for a test, which is unusual; its meaning (the interval straddled the boundary) only becomes clear in P26.
- **P1** "Each term of Q is fixed by the text before its step, yet Q covaries with the switching statistics almost as strongly as N does." The "yet" depends on an expectation the reader has not been given, namely why being predictable should mean weak covariance.
- **P4** "The same formula shows why the effect is small, by measuring how much of a whole-text, Boltzmann-like response survives ..., and locates where in a text a penalty acts." A formula "measuring" is a mismatched agent, and the reader has not yet been told that the effect is small.
- **P6** "centered on that model's design temperature (Appendix A)" and "including q_t". Both terms are used before they are defined: the design temperature only in P25, q_t only in P7.
- **P12 / P30 / P36** "the identity" is used as a name ("the identity also holds", "the identity's first-order prediction") without being introduced as such. The same object is also called "the derivative", "the score", "the first-order prediction" and "the exact derivative g(T)".
- **P14** "where K is defined by ratios of expected counts". P8 defined K as a ratio of rates computed from counts, so the reader has to work out that this is a slightly different (population-level) definition.
- **P15** "Each update is c_t − q_t plus ν_t, the change at step t in the expected number of chosen candidates after step t, a martingale increment." Two stacked appositives, and two different "after step t / at step t" time references in one phrase.
- **P17** "To first order in h, after both factors are renormalized, the second adds (h/T) Σ_t ν_t to the log-weight of a text". "the second" refers back to a factor named two sentences earlier.
- **P20** "Summed over temperatures, we define s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u), with the two equilibrium predictions of Appendix H; at a single temperature this equals b/b_tilt." The definition uses objects (g_C, g_C^tilt) that are introduced only in the next subsection and in an appendix. The reader also cannot see why a temperature fraction is built from equilibrium predictions when b/b_tilt was just defined directly.
- **P20** "adding to U_t any constant that depends only on the history, for example replacing U_t by −z_t, leaves this score unchanged". The reader has to supply the fact that −log p and −z differ by a log-normalizer that depends only on the history.
- **P21** "g_C(T) = (b/ΔE) Δc_T/T". The subscript C is not explained.
- **P25** "it predicts, from unpenalized runs on 40 design articles, the penalty h* that should lower T_× by 0.10" (the object is split from the verb), and "Fig. S1 itself rests on runs made before these documents" (it is unclear whether this is a caveat or a reassurance).
- **P26** "because both of its intervals reached just above 1.25". The main text has not said there were two intervals. Only the Fig. 2 caption reveals that they were for h* and 2h*.
- **P26** "This range would have rejected the tilted response of Sec. IV C and, for six of the nine models, the equilibrium two-state model, whose ratio we computed after the tests". It refers forward to a section not yet read, and the "for six of the nine models" insertion splits the list.
- **P27** "At twice the predicted strength ... at eight times it fell below the range. So the prediction holds only for divisors up to about 1.1". The reader cannot convert "twice" and "eight times" (of θ, of ln θ, or of θ − 1?) into the stated limit of 1.1.
- **P31** "Thus the compensator cancels about 99% of the first term." "the first term" refers back to Cov(O, N)/T in P13, eighteen paragraphs earlier.
- **P32** "Other conventions and the local form of s_temp give the same picture, and shorter texts keep more". "Other conventions" (of what?), "the local form" and "keep more" (of what?) are all undefined in the main text. "exploratory records" is internal jargon.
- **P34** "The form of the score does not imply this proportionality, because it bounds a step's contribution only by the square root of its variance." It is unclear what "it" refers to, and the argument (a Cauchy-Schwarz-type bound) is left implicit.
- **P37** "although that theory separates terms by time reversal, which autoregressive sampling lacks" is a dense technical aside introduced in the discussion without background.
- **P38** "There, the exact probability ratios of the unpenalized samples concentrated on few samples, with an effective sample size of 1.3 to 2.1 out of four per prompt". "There" is ambiguous, and the reader is not told why this matters: reweighting the samples directly would have failed, while the first-order prediction did not.
- **P40** "In the same way, the derivative applies to any small additive change to the logits". It is unclear what "the same way" refers to, since P39 is about the equilibrium shortcut.
- Inconsistent terms that slow reading: "prompts" (P1) vs "articles" (elsewhere) for the same unit; "registered in advance" / "preregistered" / "committed in advance" / "fixed ... in dated documents".

## 5. Three changes that would most improve how naturally the prose reads

1. **State each main point once, in its natural place, and turn the Discussion's opening into something other than a recap.** The whole-text versus step-by-step mechanism ("the future repeats it makes likely") appears in P1, P17, P31 and P37; "about 99%" appears in P1, P31 and P37; "A natural shortcut ... is not a reliable guide" appears in P3 and P39; and the energy-dependence contrast appears in P20, P21 and the P33 caption. Keep the derivation in P17, the numbers in P31 and the shortcut's numbers in P39, and cut the other restatements, including the closing summary sentences (P35 "A penalty thus both makes ...", P31's last clause). *Removes nothing a reader needs*, provided one full statement of each point remains. The escape-rate remark in P37 is new content and should not be lost in the cut.

2. **Define before use, give each object one name, and untangle the stacked-appositive sentences.** Move or briefly define "design temperature" and q_t before P6 uses them. Introduce "the identity" explicitly or replace it with one consistent term (derivative / score / first-order prediction). Use "articles" throughout. Say in P26 what the two intervals were. Say what "the first term" in P31 is. Make P27's "twice / eight times" convertible to "about 1.1". Split P1's long sentences, and P15, P17 and P20, so that subject and verb sit close together. *Removes nothing*; it adds a few words in places and helps readers who are not experts in the theory.

3. **Break the list-like cadence and drop the aphoristic or reassuring filler.** P2 and P5 are catalogues of one-citation sentences of the same shape; the abstract's tail and P41 are one-fact-per-sentence checklists. Connecting sentences by their logical relations (why one line of work led to the next; grouping limitations by kind: scope, statistics, construct validity, provenance) would read more naturally. Lines such as "because it changes what a penalty is", "Its verdict is a property of the chosen energy, which the sampler does not define", "In the same way," and the number-free reassurance "give the same picture" (P32) add polish but no information. *Removes nothing a reader needs* for the epigrams and the transition. For P32, the vague reassurance should either point to specific numbers in the appendix or be cut; cutting it only loses an unquantified claim that the main text does not rely on. All limitations in P41 should be kept, regrouped rather than deleted.
