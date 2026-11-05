# Reading of the abstract and main text

## 1. Paragraph-by-paragraph ratings

I rate the impression made by the prose, not the likelihood of a particular author. A 1 is a slight impression, a 2 a sustained one, and a 3 a strong one. For the displayed equations I use 0 because there is no prose feature on which to base a higher rating. I treat caption shorthand as ordinary caption writing.

| Paragraph | Rating | Reason |
|---|---:|---|
| P1 | 2 | “After a first test that did not decide” and “the temperature at which the entry and exit rates are equal, which the design targeted” feel compressed to fit a checklist of results into the abstract. |
| P2 | 1 | “have been designed partly to avoid it”, “have been proposed”, and “have also been proposed” make the literature survey sound mechanically assembled. |
| P3 | 0 | The proposed shortcut gives the paragraph a specific question to pursue. |
| P4 | 1 | “measure how much of a whole-text, Boltzmann-like response survives step-wise normalization, which explains why the effect is small, and locate where in a text a penalty acts” piles three purposes onto one verb phrase. |
| P5 | 1 | “The formula is standard”, “has been studied in several fields”, and “Here we bring these ideas” form a generic frame around a much more specific account of prior work. |
| P6 | 0 | Matter-of-fact sampling details. |
| P7 | 0 | The author anticipates a real confusion between the event being counted and the tokens being penalized. |
| P8 | 0 | The thresholds, asymmetric window, and minimum number of switches make this read like an operational definition. |
| P9 | 0 | The aging of a repetitive stretch is a particularly concrete reason to qualify the interpretation of K. |
| P10 | 0 | Conventional lead-in to a displayed equation. |
| P11 | 0 | Displayed identity. |
| P12 | 0 | The explanation of what is known before the token is drawn does real work. |
| P13 | 0 | Displayed identity. |
| P14 | 0 | Compact mathematical exposition. |
| P15 | 0 | The argument follows the count from before to after one token is observed. |
| P16 | 0 | Displayed identity. |
| P17 | 0 | The example of entering a repetitive stretch motivates the future factor. |
| P18 | 0 | Conventional setup for the temperature derivative. |
| P19 | 0 | Displayed identity. |
| P20 | 1 | “The corresponding fraction for temperature” and “with the weights of the equilibrium predictions” make the transition sound smooth while leaving the reader to supply the link between the two formulas. |
| P21 | 0 | A short statement of a competing prediction and its dependence on an energy choice. |
| P22 | 0 | The distinction between a peak near zero and a peak exactly at one reads like an observation. |
| P23 | 0 | A single numerical result. |
| P24 | 0 | Caption with a specific rule for choosing the example. |
| P25 | 0 | The distinction between the earlier runs, the fixed penalty, and the later estimate of the derivative sounds carefully considered. |
| P26 | 1 | “For comparison”, “Under eight other labelings”, and “A separate test” turn a paragraph about the replication into a succession of results with little indication of their relative importance. |
| P27 | 0 | Concrete progression through the tested strengths. |
| P28 | 0 | The unexpected overshoot, the subsequent interpolation, and the named outlier give the paragraph an individual voice. |
| P29 | 0 | Conventional figure key. |
| P30 | 0 | Conventional table key. |
| P31 | 1 | “Compared with what the same penalty would do if it reweighted whole texts” is immediately followed by “compares each step-wise response with the response under exponential tilting”; the comparison is announced twice before we get its size. |
| P32 | 0 | The 256-token comparison gives the length qualification a concrete scale. |
| P33 | 0 | Conventional table key, including a qualification about the energy choice. |
| P34 | 0 | The contrast between the share of steps and the share of response is plainly drawn. |
| P35 | 0 | It reports where a successful prediction performs less well. |
| P36 | 0 | The last sentence tells the reader exactly which uncertainty the plotted interval represents. |
| P37 | 1 | “In the registered tests” and “in an exploratory analysis” give the opening the cadence of another abstract, just before a new point about jump processes. |
| P38 | 0 | “We attribute the contrast to scale, without having tested it” sounds like an author identifying the limit of their own explanation. |
| P39 | 0 | The comparison includes both the magnitude of the error and the possibility of the wrong sign. |
| P40 | 0 | The practical cost in unpenalized runs gives the discussion of generality a concrete limit. |
| P41 | 2 | “The confirmation rests on”, “Decisions concerned”, “All prospective tests”, and “For instruction-tuned models” start successive qualifications in the same register; design limits, outcomes, and provenance end up sounding equally weighted. |

## 2. Paragraphs with the clearest individual voice

- **P28.** “Because ln K curves (Fig. 1c), we also located each crossing by interpolation after the tests.” I can reconstruct what prompted the second measurement, when the authors decided to make it, and why they kept the original result. “GPT-2-XL most of all” identifies a case that still resisted the explanation.
- **P25.** “These documents did not cover Fig. S1, whose runs were made earlier.” This interrupts an otherwise tidy account to tell me exactly which evidence the registration covered. I would keep it.
- **P9.** “The balance drifts towards repetition along a text, and repetitive stretches become harder to leave the longer they last”. The second observation is especially effective. I can picture two stretches with the same label but different histories, and see why one exit rate cannot describe both.
- **P38.** “We attribute the contrast to scale, without having tested it.” The admission comes immediately before a plausible explanation. It tells me how much weight the author personally puts on that explanation, which matters more here than a general caution would.

## 3. Deletions and repetitions

**Sentences I would delete outright.**

| Paragraph | Exact sentence | Reason |
|---|---|---|
| P5 | “The difference between step-wise and whole-text normalization has been studied in several fields.” | The next three sentences name the fields and give the relevant results. |
| P28 | “A user would care about where T_× ends up.” | The next sentence reports where it ended up, and P25 already explains why that quantity was targeted. |
| P31 | “Compared with what the same penalty would do if it reweighted whole texts, the responses predicted and confirmed above are small.” | The next two sentences specify the comparison and give its magnitude. Starting there would make the result easier to find. |

**Repetitions for which placement matters.**

- **P7:** “Keeping them separate makes the definition of the states independent of the penalty.” This largely spells out “Repeats define the states below, and candidates define the penalty.” I would keep the motivation, since without it the mismatch can look accidental. The two statements could share a sentence.
- **P1/P12 and P1/P31:** “Each term of Q is fixed by the text before its step” reappears as “each q_t is fixed by the text before step t”; “Step-wise normalization therefore cancels about 99% of the first-order response” reappears as “Thus the compensator term Cov(O, Q)/T cancels about 99% of the tilted term Cov(O, N)/T.” Both recurrences belong in the body because the abstract should be readable on its own.
- **P8/P24:** “The crossover temperature is where ln K = 0.” This is the caption equivalent of “We call the temperature at which K = 1 the crossover temperature T_×.” Keep the caption definition so the figure can be read independently. The same consideration applies to the decision range in P25, P29, and P30.
- **P25/P37/P41:** “Because the tests estimated the derivative from their own unpenalized runs, they test whether the first-order prediction holds at h*.” The underlying fact already appears as “only h* was fixed beforehand, while g came from the unpenalized runs of the same test.” Keep one explicit reminder in the discussion. P41's “A prediction made before any test data existed was examined only in an exploratory analysis (Appendix F4).” can be dropped if P37 keeps “a prediction from the design articles alone carried over to new articles (Appendix F4)” and its exploratory qualifier.
- **P27/P41:** “For the multiplicative penalty, the prediction held only up to the weakest divisors used in practice (Appendix F5).” P27 gives both the conclusion and the approximate divisor. This whole P41 sentence could go. In the preceding P41 sentence, the information about presence and frequency penalties is new and should stay.
- **P9/P32/P41 and P28/P41:** “The two states are a labeling of a continuum, K averages over a drift along the text, and s_pen depends on the length of the text.” The three cautions have already been explained, but the length dependence deserves to stay somewhere alongside the 99% result. “The shift of T_× itself, which a user would target, did not always match the intended 0.10, for reasons we have not fully identified (Sec. IV B).” repeats P28. I would retain the distinction between successfully predicting the change in ln K and missing the target for T_× once in the discussion. It changes what readers can claim the procedure accomplished.

## 4. Passages that are hard on a first reading

| Paragraph | Exact wording | Where I had to stop |
|---|---|---|
| P1 | “the realized change divided by the prediction had a median of 1.04, with its bootstrap interval over articles inside the registered range of 0.8 to 1.25” | What changed is ln K; models supply the ratios whose median is reported, whereas articles supply the resamples. Neither distinction is locally explicit. |
| P6–P8 | “the tokens that have previously followed the current last two tokens”, and “the fraction of repeats in a 32-token window” | Three-token candidate eligibility can use the prompt, four-token repeat labels use the continuation, and the multiplicative penalty has a third eligibility rule. The reader has to carry the first two distinctions into the window definition. “Each label therefore uses up to 15 later tokens” is worth keeping. |
| P8/P14 | “Applied to the four counts that define K” | P8 gives two rates. To follow P14 I have to reconstruct entries, ordinary steps, exits, and repetitive steps, and then notice the shift from observed to expected counts. |
| P15–P17 | “Each update is c_t − q_t plus ν_t.” and “Q − E[Q] = Σ_t ν_t” | q_t is known before the draw, but ν_t is an update learned from that draw. The equality is for sums over the path. A reader looking for a step-by-step equality between q_t and ν_t will miss the argument. The intuitive opening of P17 arrives just after it would have helped most. |
| P20–P21 | “Over several temperatures we sum numerator and denominator with the weights of the equilibrium predictions of Sec. III D” | The passage moves from a local slope ratio to a ratio of sums of g_C terms. The weighting, the sign convention, and the meaning of the tilted superscript all have to be reconstructed. |
| P25 | “Each test compares the realized change in ln K at h* with the first-order prediction h* g(T), summed over the usable temperatures.” | “summed” could attach just to the prediction. A ratio of two sums and an average of per-temperature ratios can differ. P30 repeats the ambiguity. |
| P26 | “The equilibrium two-state model, evaluated after the tests, would have fallen outside the range for six of the nine models (Appendix H). Five of the nine per-model intervals reached above 1.25 (Table I).” | At first the five intervals sound like intervals for the equilibrium shortcut. The text also moves from the seven-model replication to nine models before introducing the separate pair. |
| P28 | “With T_× from a straight-line fit of ln K against 1/T, as in the design, it moved further than intended” | “it” is the fitted crossing, and “further” compares displacements. The subsequent “This brought the median to 0.87” is a ratio of displacements, not a temperature. |
| P32 | “Pooling the covariances of the step-wise response over articles” | P32 has just said the covariances are within articles. Combining within-article estimates and computing a covariance on pooled observations are different operations. I cannot tell from this wording which check was made. |
| P34/P36 | “For these steps, the share of the response divided by the share of the variance had a median of 0.96” | “these steps” means the intermediate-q_t bin; the basis of the median is less clear. The caption names nine models and multiple temperatures without saying which are pooled before the median. |
| P38 | “There, the exact probability ratios of the unpenalized samples concentrated on few samples” | Ratios are attached to individual paths. What appears to concentrate is the total reweighting mass. It also takes the next sentences to see why difficulty reweighting paths and success predicting an aggregate statistic can coexist. |

## 5. Three changes to improve naturalness

1. **Give the prediction and its test a stable order.** At P1 and P25–P26, introduce the fixed penalty, the derivative estimated on the test's controls, the change in ln K, and only then the median ratio and its interval. The pivotal words are “only h* was fixed beforehand”. Putting them before the result lets a reader understand what was predicted in advance. **Information lost:** none. Keep both the 1.04 result and the imperfect crossover shift.
2. **Put the physical meaning at the point where the algebra first needs it.** Move the idea in P17's “the future repeats it makes likely” beside P15's “plus ν_t”, while keeping the more precise “expected number of candidates chosen” for the mathematical object. When P20 introduces “the weights of the equilibrium predictions”, give the reader the slope and weighting convention there. **Information lost:** none. The identities and the future-factor argument should stay.
3. **Shorten the discussion by removing repeated conclusions and grouping like qualifications.** Keep the design/test distinction in one place, put the English-news and model limits together, and separate the final provenance disclosure from the list of scientific limits. Start with the cuts in §3, especially P41's second account of “the weakest divisors used in practice”. **Information lost:** only duplicate statements. Keep the shared articles, the unit of resampling, and the code-checking disclosure; none can be inferred from a generic limitations sentence.

## 6. Copyediting, notation, and consistency

I found no clear misspelling. The following are the wording, mathematical notation, and cross-paragraph issues I could identify in the supplied text. Some are definite mismatches; others need clarification before one could call the underlying claims contradictory.

### Wording, grammar, and scope

| Paragraph | Exact wording | Issue |
|---|---|---|
| P1/P26/P27/P41 | “After a first test that did not decide”; “did not decide: its intervals”; “the test did not decide or could not be run”; “the test did not decide” | Recurrent awkward use of *decide* without saying what was decided. P26 gives an unambiguous account of the inconclusive outcome. P27 also needs to associate the outcomes with the relevant models if they differed. |
| P3 | “where z are the logits” | Number and indexing are awkward: exp(z/T) appears to concern one token's scalar logit, whereas “are the logits” treats z as the collection. Component notation or an explicit vector convention would resolve it. |
| P6 | “We sampled continuations of news articles published after the models were released, from nine open base models between 0.4 and 7.6 billion parameters” | The phrase about the source of the samples is separated from “sampled”, and “models between 0.4 and 7.6 billion parameters” lacks a clear link such as “with”. |
| P6 | “Each prompt was the first 64 tokens of an article, continued for 512 tokens.” | The participle can modify “article” or “prompt”; what has length 512 is the newly generated continuation. |
| P7/P12 | “We write c_t = 1 if the sampled token is a candidate” and “N is the number of candidates chosen.” | The zero case of the indicator is implicit. The count is of candidate selections, including repeated selections of the same token, rather than distinct candidates. |
| P7 | “acts on a third set, every token seen before: before division by T, it divides their positive logits” | The sentence shifts from a distributed singular to a plural referent, and the adjacent uses of “before” have different meanings. |
| P12 | “with a bounded stopping time, such as an end-of-sequence token kept in the normalization” | A token is not a stopping time, and allowing EOS does not by itself bound generation length. A capped stopping time is needed for the example as worded. |
| P17/P1/P31 | “rewards each choice also for the future repeats it makes likely”; “the repeats it makes likely later”; “the future repeats that each choice makes likely” | “also” sits awkwardly in P17. More importantly, the exact identity concerns future **candidate selections**. “repeat” already has the narrower repeated-4-gram meaning in P7. The distinction matters when interpreting the 99% cancellation. “rewards” also describes the sign for positive h; with a negative h the same future count is penalized. |
| P17 | “with N_{>t} the candidates chosen after step t” | The exponential needs the **number** of candidates chosen, not the candidates themselves. |
| P20 | “The step-wise response to temperature, which is what the van 't Hoff slope measures” | The variable of differentiation in P18–P19 is inverse temperature β. A derivative with respect to T would differ in both sign and scale. |
| P26 | “A separate test repeated the whole procedure for two 7-billion-parameter models, where the rule applied to the mean ratio, 0.96 (0.84–1.11).” | “where” appears to modify the models, although the mean-ratio rule belongs to the test. |
| P26/P27/P35 | “Under eight other labelings of the states the median stayed inside the range”; “At the predicted divisors the median ratio over seven models was 1.01, inside the range”; “the difference excluded zero” | P25's decision concerns the bootstrap **interval**, not just the point estimate. A median inside the band is weaker evidence. In P35 it is the interval for the difference, not “the difference”, that can exclude zero. |
| P27 | “at eight times it fell below the range” and “The prediction therefore holds only for divisors up to about 1.1” | The tested settings bracket a loss of agreement. They do not locate a sharp upper limit or establish failure at every larger divisor. |
| P28 | “Individual models, GPT-2-XL most of all, and the larger models still departed from the target” | The two larger models are also individual models. The coordination obscures whether the first group means individual *smaller* models. |
| P31 | “Thus the compensator term Cov(O, Q)/T cancels about 99% of the tilted term Cov(O, N)/T.” | The displayed percentage concerns the derivative of ln K, assembled from four count responses (P14). Written for arbitrary O, the sentence suggests the same percentage for any statistic or each count separately. The percentage is established here for the combined response. |
| P32 | “Shorter texts keep a larger fraction, 1.8% for the first 256 tokens” | Two fractions have just been discussed. The singular does not say which one is 1.8%. |
| P34 | “the model has almost decided to repeat” | q_t is the probability of a candidate selection, not necessarily of a repeat under the labeling rule. The same shorthand recurs in P1's “uncertain whether to repeat” and P37's “probability of repeating”. |
| P37 | “how far each choice departed from the model's own probability of repeating” | “how far” suggests magnitude alone; c_t − q_t is a **signed** difference. |
| P38 | “the exact probability ratios of the unpenalized samples concentrated on few samples” | The object that concentrates is the weight carried by a few sample paths; using “samples” for both the paths being weighted and the small subset carrying that weight obscures the statement. “Each step changes it only slightly, so its response stays close to linear.” reads as a demonstrated inference, although the same paragraph says “without having tested it”. |
| P38 | “ln K is instead the log of a ratio of counts summed over many steps and texts.” | P8 defines K as a ratio of **rates**, each with its own exposure count. Reading this as the number of entries divided by the number of exits gives the wrong statistic. |
| P40 | “It also carries over to truncated sampling such as min-p [2407.01082] or top-p [1904.09751] if q_t is taken within the truncated set” | Renormalizing q_t inside the set covers a region where the retained support stays fixed. At a threshold or tie where an arbitrarily small perturbation changes support, the derivative need not exist. The stated condition alone is insufficient. Likewise, “any small additive change to the logits” requires centering the actual logit perturbation; the indicator formula in P10 is specific to a common shift on a candidate set. |

### Notation and cross-paragraph checks

| Paragraphs | Exact wording | Issue or reconciliation |
|---|---|---|
| P12–P14 | “the response at h = 0 is therefore” followed by “∂E[O]/∂h = Cov(O, N)/T − Cov(O, Q)/T.” | The prose supplies the evaluation point; the stand-alone equation omits it and the h-dependence of the expectation. P14's “g(T) = ∂ ln K/∂h” likewise denotes the derivative at zero. For the multiplicative formula, the corresponding baseline is λ = 0. |
| P18–P21/P33 | “s_temp = b/b_tilt”; “s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u)”; “g_C(T) = (b/ΔE) Δc_T/T” | b is never explicitly defined by an equation or identified as a fitted slope at its first occurrence. The local ratio and the multi-temperature ratio use the same name. The second ratio introduces g_C before its definition and leaves g_C^tilt implicit. P21's “difference in mean energy per step between the states” also leaves the subtraction order implicit. |
| P19/P33 | “with U = Σ_t U_t(x_t)” versus “with U = −log p at T = 1” | P19 defines a path sum. P33 can agree if p means the entire text probability, but elsewhere −log p denotes a token energy. This is a consequential missing subscript or missing identification of p. |
| P20/P39 | “replacing U_t by −z_t” versus “with the logits it understated it” | The second phrase leaves open whether the shortcut used z_t or −z_t as the energy. The sign should be explicit, particularly in a sentence about a wrong-sign prediction. |
| P23/P24/P26/P30 | “for the nine models”; “for unpenalized continuations of the 120 replication articles”; “six of the nine models”; “First seven models: preregistered replication. Last two models: preregistered test on 7-billion-parameter models.” | Nine is consistent for the combined studies, seven for the replication. The same articles were reused, so P24 need not be factually wrong. P24 and the middle of P26 nevertheless read as if the replication itself covered all nine. |
| P25/P30 | “summed over the usable temperatures” versus “summed over the test temperatures” | The caption drops the usability restriction. They coincide if all the relevant temperatures are usable; P8 explicitly makes inclusion conditional. Neither phrase clearly states that the numerator and denominator are summed separately. |
| P26/P30 | “on the 40 articles from which h* had been computed” versus “(120 articles in each test)” | The latter is accurate for the two tests tabulated according to P30, but would contradict P26 if “each test” meant the first test too. Limit the quantifier to the rows shown. |
| P27/P29/P41 | “for each preregistered test” versus “the test did not decide or could not be run” (P27) and “For instruction-tuned models the test did not decide” (P41) | P27 distinguishes an inconclusive outcome from an infeasible test; P41 loses the latter. P29 refers readers to Appendix F5 for the instruction-tuned models. Whether those tests were preregistered is not stated, so the figure's coverage cannot be settled from the supplied text. |
| P27/P41 | “For two instruction-tuned models the test did not decide or could not be run.” versus “All prospective tests used English news articles and base models of up to 7.6 billion parameters.” | These are compatible if the instruction-tuned work was outside the prospective series. The main text should specify its status if that is the intended distinction. |
| P34/P11/P36 | “Each step contributes c_t − q_t to the score, a term with variance q_t(1 − q_t).” versus “(N − Q)/T” and “share of the variance Σ q_t(1 − q_t)” | The contribution to the score as defined is (c_t − q_t)/T. Its variance **conditional on the history** is q_t(1 − q_t)/T². P34 and the caption describe the numerator. The constant cancels from shares at fixed T, but a pooling convention matters across temperatures. |
| P34 | “the score alone, which only bounds a step's contribution by the square root of its variance” | For a covariance with O, the elementary bound also contains the standard deviation of O. For ln K there are coefficients for the component counts. The text needs a normalization or a proportionality factor for the bound as written. |
| P35 | “It put slightly more of the response on the exit rate than was measured” versus P36's “Share of the realized change in ln K that comes from the change in entry rate” | The two statements can agree because the shares are complements. The reader must reverse the sign of the prediction error when moving between the prose and the plotted share. |

I see no conflict between P25's “moved T_× in the direction of h” and P28's “moved further than intended”: the first concerns direction in earlier runs, the second the magnitude of the target shift. Nor does P26's median interval “inside the range” contradict “Five of the nine per-model intervals reached above 1.25”. Those intervals describe different quantities.
