# 論文の下書き（Nature Machine Intelligence 向け → PRX Intelligence に組み直す予定）v0.7（2026-11-03 22:4x JST、A2-25・A2-26 を反映。v0.6 は 11:2x、編集者役のレビューと A2-24 を反映。v0.5 は 08:0x。v0.1 は 15:5x、v0.2 は 16:0x、v0.3 は 16:5x、v0.4 は 11-02 17:5x。前の版は git の履歴に残っている）

- 骨組み：`docs/PAPER_OUTLINE_NMI.md`（§12 に編集者役のレビュー）。先行研究：`docs/PRIOR_WORK_2026-10-31.md`・`docs/PRIOR_WORK_2026-11-02.md`。
- オーナー（2026-11-02 16:55 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）（A2-22 の結果を下書きに入れ、GPU の実験のゲートを作る流れへの返事）
- この版：編集者役のレビュー（§12）、A2-21（文全体と 1 歩ずつの比べ）、A2-22（融点の動き・入る速さと出る速さ・重み付け）を入れて書き直した。残り：Methods（見出しだけ）、図、表 1、文献の形。割る型の罰と指示調整済みのモデル（A2-23）は v0.5 で入れた。
- **仮の題**（2026-11-03 13:4x、A2-25 の結果を見てから本決まり）：**Repetition penalties shift language model outputs less than an equilibrium reading predicts**
- 決めていないこと：「we」で書くか（著者は 1 人）。

## v0.6 → v0.7 で直したこと（2026-11-03 22:4x JST）

- **組み直しの前の版の名札**：この v0.7（コミット 4026593）を、GitHub のブランチ `archive/nmi-draft-v0.7` として残した（タグ `nmi-draft-v0.7` も手元に付けたが、この環境からは GitHub にタグを送れなかったので、ブランチで代わりにした）。取り出し方：`git checkout archive/nmi-draft-v0.7 -- docs/PAPER_DRAFT_NMI.md`。
- オーナー（2026-11-03 22:48 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）

- オーナー（2026-11-03 22:45 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）（今日の結果を本文に入れる作業）
- 題：E「Repetition penalties shift language model outputs less than an equilibrium reading predicts」に本決まり（A2-25 で 8 倍まで予言どおりではなかったので、F は並べ直さない。骨組み §14 の決まりどおり）。
- 結果「Predicting…」の割る型の段落に、A2-25（予言の 2・4・8 倍）を 3 文で入れた。限界の割る型の文も書きかえた。補足 S7 に表と段落を足した。
- 融点の行きすぎ：「主に ln K の曲がりのため」を消した。A2-26 の XQ23 で、設計の温度では罰ありの ln K が中央値で狙いの近くにあり、行きすぎは ln K のずれからは説明できなかったため。「原因は分かっていない」と書いた（結果と考察の 2 か所）。
- 2 つの状態の節：A2-26 の XQ22（くり返しは長く続くほど抜けにくい）を 1 文で入れ、K を「位置と、抜けやすさの違うくり返しをならした比」と書いた。
- 考察の非定常の段落：A2-26 の XQ20・XQ21（前半・後半でも、分け方 8 通りでも、平衡の読みは言いすぎる）を 1 文で入れた。
- 限界：温度ごとに組み立てた平衡の読み（A2-26 の形）では GPT-2-XL と Mistral-7B がほとんど言いすぎないこと、くり返しは 2 状態のマルコフ鎖ではないこと、エネルギーの取り方（A2-27 の結果待ちの〔〕）を入れた。

## 文献の確かめ（2026-11-03 13:3x JST、経緯を知らない調べ役が 15 本をすべて開いた）

- 15 本すべて実在し、番号の取り違えはなかった。14 本は本文の主張を支える。直したこと：(1) CTRL [1909.05858] は「生成したトークンを θ で割る」だけで、正負での扱い分けと書き出しを含めることは Hugging Face の実装のもの → 両方を引く形に。「most often used in practice」を支える文献はない → 「A widely used penalty」に弱めた。(2) [2512.04419] は「強さの調整が要る」は支えるが「logit を下げる」「試行錯誤」は言っていない → 文を直し、罰そのものには CTRL を引く。(3)「くり返しは低温の秩序相」は [2406.05335; 2503.06330] だけが言っている → その文にこの 2 本を付けた。
- 足した文献（調べ役が挙げたもの。番号は投稿前に開いて確かめる）：transformers [1910.03771]、データの LatestEval（Li, Guerin & Lin, AAAI 2024）、min-p [2407.01082]、透かし [2301.10226]。モデルの論文（Pythia 2304.01373、SmolLM2 2502.02737、Qwen2.5 2412.15115、BLOOM 2211.05100、Mistral 2310.06825、GPT-2・GPT-Neo）は表 1 か Methods に付ける。
- 残り：1312.7480 は手引き（原論文ではない）。原論文を足すか考える。bootstrap（Efron）、有効な標本数（Kish）は必要なら。

## v0.4 → v0.5 で直したこと（2026-11-03 08:0x JST）

- オーナー（2026-11-03 07:59 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）（A2-23 の結果を下書きに足すことへの返事）
- A2-23（`docs/A2_23_RESULT.md`）：XQ12（割る型の罰）は予言どおり、XQ13（指示調整版）ははっきりしない。
- 要旨に 1 文（割る型の罰でも範囲の内）。新しい数は入れていない。約 150 → 約 165 語。投稿前に NMI の上限で数え直し、要るなら削る。
- はじめにの最後の段落に 1 文。
- 結果「Predicting a penalty」に 1 段落。数字の予算を守るため、新しい数は本文に入れず、補足 S7 に置いた。
- 考察：最初の文と実用の段落の主語を「足す型の罰」から広げた。ただし「確かめた強さで」と限った（θ* が 1.02〜1.05 で、ふだん使われる 1.1〜1.3 より弱いため）。
- 限界：「足す型の罰だけ」を外し、割る型は弱い強さでしか確かめていないこと、指示調整版では決められなかったことを書いた。
- 補足 S7 の下書きを足した。
- 新しい文献：割る型の罰を最初に出した論文として [1909.05858]（CTRL）。2026-11-03 13:3x に確かめた（下の「文献の確かめ」）。

## v0.3 → v0.4 で直したこと（2026-11-02 17:5x JST）

- オーナー（2026-11-02 17:44 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）
- オーナー（2026-11-02 17:47 JST、原文）：（オーナーの承認・発言。原文は非公開の記録にある）
- 結果の節に「数字の予算」をかけた（1 つの節に 3〜4 個まで。残りは図・表・補足へ）。数えた数（0・1・2 と図の番号を除く）：2 つの状態 14 → 4、平衡の読み 12 → 2、予言 25 → 5（0.10、0.8〜1.25、40、120、1.04）、どこで効くか 11 → 0、考察 4 → 2。本文は約 2,200 → 約 2,000 語。
- 有効数字を 2〜3 桁に（1.78 → 1.8、1.02 → 1.0）。
- 文の長さ（18:1x、オーナー 18:09（オーナーの承認・発言。原文は非公開の記録にある））：35 語を超える文 20 本（最長 81 語）を「1 文に 1 つのこと」で割った。要旨から考察まで 110 文、平均 19.6 語、最長 34 語。要旨は約 150 語。
- 設定の数（64・512・4 本・32 歩の窓・しきい値）は Methods へ。
- 投稿前の必須：文献をすべて開いて、実在すること・題と番号が合うこと・本文の主張を支えることを確かめる（ICLR 2026 で、AI が作った架空の文献が問題になった）。いまの [arXiv 番号] は調査役が開いたものと、Claude が開いたものが混ざっている。

## v0.2 → v0.3 で直したこと

- 要旨：150 語以内に。最初の 2 文に発見（罰はモデルが迷っていた歩で効く → 罰なしの標本から予言できる、平衡の読みは言いすぎる）。「every model's point estimate」をやめた（判定は中央値だったため）。
- 物理の言葉を控えた（オーナー 16:13 の〔オーナーの言葉〕）：crystal・liquid → repetitive state・ordinary state、freezing・melting の速さ → entry rate・exit rate、field・couples to → 使わない。melting temperature は一度だけ、たとえとして定義。エンタルピーの比べと協同性の段落は補足へ。化学の説明の段落は削除。
- 構成：はじめにと考察の小見出しを外す（NMI の形）。2.4（7B）を 2.3 にまとめた。新しい節「Where the penalty acts」（A2-22 の XQ11 と状態ごとの寄与）。限界は考察の最後の段落に。AI がコードを書いたことは Methods へ。
- 前もっての確かめの時系列を 1 文で書いた（±0.3 の罰ありの生成 → 恒等式の確かめ → 予言の手順を決めて封 → h* での生成）。
- 前もって書いた「平衡の読みならこう効く」との比べを 2.3 に入れた（9 本すべてで帯の外）。
- 融点の動き（A2-22 の XQ10）は 2.3 に 1 文と補足の表。
- 言いすぎを直した：「We know of no method」→「We are not aware of work that predicts …」、「some forms of steering」を外した、「identifies where」→ 新しい節で示したので「shows where」、考察の最初の文の主語を狭めた。

---

## Abstract（v0.8）

Sampling from a language model at temperature T has the Boltzmann form, and repetitive text has been described as an ordered phase in this picture. We show that the equilibrium reading this invites overstates how a repetition penalty shifts the balance between repetitive and ordinary text, about twofold in the median across nine open models. The reason is that sampling normalizes at every step. By the score-function identity, a penalty then acts in proportion to the model's uncertainty about whether to repeat, and little at each of the many steps where it already expects to. The identity predicts the response from unpenalized samples alone, which lets it be tested in advance. In a preregistered replication on new prompts, after an inconclusive first test, the response fell within the preregistered range. So did the responses of two 7-billion-parameter models and of the common multiplicative penalty at the predicted strengths.

（v0.8、2026-11-03 11:1x。13:3x：題の監査役（物理）の指摘で「hardly at all」を「little at each of」に（q ＞ 0.9 の歩は全体では効きの約 5 分の 1 を持つので）。編集者役の指摘（いちばん面白い点が 3 番目、2 文目が 1 文目から導けない、定義の前の言葉）と A2-24 の XQ14（迷い度に比例して効く）を入れて組み替えた。たとえの外れとその理由から始め、「melting temperature」を要旨から外した。「about twofold」は ln K（2 つの状態の釣り合い）での値。）

---

<!-- Introduction（NMI では見出しなし） -->

Language models that sample one token at a time sometimes fall into loops, repeating a phrase or a sentence until the end of the output. The problem is old and persists in current systems [1904.09751; 2012.14660; 2206.02369; 2512.12895]. The usual remedy is to penalize tokens that would repeat earlier text [1909.05858], with a strength that has to be tuned for each setting [2512.04419]. We are not aware of work that predicts, before applying it, how strongly a repetition penalty will act on a given model.

Sampling at temperature T draws each token with probability proportional to exp(logit/T), the same form as a Boltzmann distribution. Several studies have described generated text in the language of statistical physics, with phases and transitions as the temperature changes [2406.05335; 2503.06330; 2405.17088; 2606.06238]. Repetition, in this picture, belongs to a low-temperature ordered phase [2406.05335; 2503.06330]. If the analogy held quantitatively, equilibrium thermodynamics would also tell us how a repetition penalty acts. The penalty would add an energy proportional to the number of copied tokens, and its effect would follow from quantities measured without it.

We test this directly. We treat repetitive and ordinary stretches of generated text as two states, count how often sampling switches between them, and find a temperature at which the two states are equally likely. For small penalties this temperature shifts linearly with the strength of the penalty, but by only about half of what the equilibrium reading predicts.

The reason is that sampling normalizes the distribution afresh at every step, so the text as a whole does not follow a single Boltzmann distribution [2503.21929]. For such step-wise processes there is an exact identity. The change in the average of any output statistic, per unit change of the penalty, equals its covariance with the derivative of the text's log-probability with respect to the penalty. This is the score-function, or REINFORCE, identity [Glynn 1990; Williams 1992], known in non-equilibrium physics as a trajectory-based response formula [0902.3955; 1312.7480]. For repetition, what matters is not the number of copied tokens. It is the sum, over steps, of whether a copy candidate was chosen minus the probability the model assigned to the candidates at the sampling temperature. Inside a repetitive stretch the model already expects to copy, so copying there carries almost no weight.

Because the identity needs only unpenalized samples and their token probabilities, it can predict the effect of a penalty before the penalty is applied. We tested such predictions, recorded in advance, on nine open models between 0.4 and 7 billion parameters.

（約 490 語。）

---

## Results

### Generated text switches between two states

We sampled continuations of recent news articles from nine open base models between 0.4 and 7 billion parameters, at temperatures between 0.2 and 0.9, with plain softmax sampling and no truncation (Table 1; Methods).

To define repetition without reference to any particular penalty, we call a sampled token a copy if the four tokens ending with it already occurred, in the same order, earlier in the generated continuation. The penalty acts on a related, slightly broader set. At every step, the copy candidates are the tokens that have previously followed the current last two tokens, anywhere in the prompt or the continuation. q_t denotes the total probability that the model, at the sampling temperature, assigns to the candidates. The penalty we study adds a constant h (negative for a penalty) to the logits of the copy candidates before dividing by the temperature.

Copies cluster into long stretches in which the text repeats a phrase or a sentence, separated by stretches of ordinary text with occasional short copies. At a temperature where both kinds of stretch are common, the fraction of copies in a 32-token window has a high peak near zero, a narrow peak at exactly one and a shallow trough in between (Fig. 1b). Between 74% and 89% of outputs switch between the two kinds at least once (Fig. 1a shows an example). We therefore label each step as belonging to a repetitive or an ordinary state, using two thresholds on the windowed fraction (Methods). In the repetitive state the models choose a copy candidate at about nine steps in ten, in the ordinary state at about two in ten. The labelling is a choice and the temperatures reported below depend on it, but every comparison uses the same labelling.

Sampling moves in both directions between the states. At each temperature we count the entry rate, the number of switches into the repetitive state per step spent in the ordinary state, and the exit rate, defined the other way round. Their ratio K measures the balance between the two states. Around the temperature at which K = 1, ln K is close to linear in 1/T, as in the van 't Hoff plot of a two-state chemical equilibrium; over wider ranges it curves (Fig. 1c). By analogy we call the temperature at which K = 1 the melting temperature. It lies between 0.44 and 0.66 for the nine models. The balance is not constant along a text: near the melting temperature, ln K is higher by between 1.3 and 2.1 in the second half of the generated tokens than in the first. Repetitive stretches also become harder to leave the longer they last. After 97 or more steps in the repetitive state, the exit rate per step was a fifth to a third of that after 49 to 96 steps (exploratory). K is therefore an average over positions and over stretches of different persistence, not a ratio of two constant rates. Temperature always means the sampling temperature, and the analogy describes switching statistics; it is not a claim that the generated text is in equilibrium.

### The equilibrium reading overstates the effect of a penalty

Penalizing or favouring the copy candidates in the seven smaller models moved the melting temperature in proportion to h (Supplementary Fig. S1). Controls with synthetic data and a second labelling show that the shift is not produced by the labelling itself (Supplementary Section S2).

Suppose the text were an equilibrium two-state system in which the penalty adds h to the energy of every chosen copy candidate. The shift would then follow from the Clausius–Clapeyron relation: per unit of h, the melting temperature would move by T_m Δc / ΔE. Here Δc is the difference between the two states in the rate at which a copy candidate is chosen, and ΔE is the difference in their average energy per step. We take the energy of a step to be the negative log-probability of the sampled token at temperature 1. All three quantities can be measured without the penalty. For every model the measured shift was only about half to two-thirds of this prediction (Fig. 2a), and the same held with novels and scientific papers as prompts (Supplementary Section S3).

The score-function identity accounts for the difference. Temperature by temperature, the equilibrium form overstated the measured response of ln K to h by a median factor of 1.8. The identity, applied to the same unpenalized samples, matched it with a median ratio of 1.0 (Fig. 2b).

### Predicting a penalty before applying it

The identity turns into a prediction. After the experiments above, and before generating any text with a predicted penalty, we fixed the procedure, the analysis code and the decision criterion. From unpenalized generations at seven temperatures we estimated each model's melting temperature and the response of ln K to h, and from these the penalty h* that should lower the melting temperature by 0.10. We then generated text with and without h* around the predicted new melting temperature and compared the realized change in ln K with the first-order prediction. The criterion was that the bootstrap interval of the median ratio, measured over predicted, lie between 0.8 and 1.25. Because the identity is exact at h = 0, the test asks whether its first-order prediction, estimated from finite samples, still holds at a penalty large enough to matter. We also recorded r_C, the factor by which the equilibrium reading's prediction exceeded the identity's; it was about two. The test could therefore also tell the two readings apart.

The first test, on the 40 articles from which h* had been computed, did not decide, at either h* or 2h*, because the interval reached just above the upper limit (Supplementary Section S4). We then committed in advance to a single replication on 120 new articles. The median ratio was 1.04, inside the preregistered range. Measured against the equilibrium prediction, expressed as a change in ln K, every model's ratio was below 0.8 (Fig. 3), although no decision rule was set for this comparison. A second preregistered test, which repeated the whole procedure for two 7-billion-parameter models whose penalties differed by a factor of two, also fell inside the range (Fig. 3; Table 1). In a later exploratory check, the replication stayed inside the range for each of eight other choices of window length and thresholds in the labelling (Supplementary Section S2).

A widely used penalty is multiplicative. Introduced with CTRL [1909.05858] and implemented in the Hugging Face transformers library [1910.03771], it divides the positive logits of every token that has already appeared, including the prompt, and multiplies the negative ones. The identity applies to it unchanged; only the derivative of the logits with respect to the penalty differs. We repeated the preregistered procedure for the seven smaller models with this penalty, predicting the divisor from the same unpenalized samples. The median ratio again fell inside the preregistered range (Supplementary Section S7). For six of the seven models the predicted divisors were close to one, weaker than the values commonly used. A further preregistered test raised the divisor to two, four and eight times the predicted strength, θ up to 1.53. The median ratio was 1.00 at twice the strength, 0.86 at four times, which did not decide, and 0.52 at eight times, below the range. The first-order prediction thus holds up to θ of about 1.1, and stronger divisors act less than it predicts (Supplementary Section S7). For an instruction-tuned model prompted through its chat template, the test did not decide: the interval was wide and crossed the upper limit (Supplementary Section S7). A second instruction-tuned model stayed more often in the ordinary state than in the repetitive one even at the lowest temperature we sampled, so no prediction could be made for it.

For a user, what matters is where the melting temperature ends up, and here the first-order prediction was less accurate. In a further preregistered comparison, the melting temperature moved further than intended: by a median of 1.3 times the intended shift in the smaller models, with an interval that did not decide, and by about twice in the two larger models. At the design temperature itself, the penalized ln K was close to its target in the median, so the overshoot does not come from a misprediction of ln K there (exploratory). We have not identified its cause; ln K is curved in 1/T in all nine models, which a straight-line estimate of the melting temperature ignores. For three models the unpenalized melting temperature also had to be extrapolated beyond the test temperatures. Measured as a shift of the melting temperature, the equilibrium reading still overstated the response for seven of the nine models, but by less than in ln K (Supplementary Section S5).

### Where the penalty acts

The identity also says where in a text the penalty acts. Each step contributes c_t − q_t to the score, where c_t indicates whether a copy candidate was chosen. This term fluctuates with variance q_t(1 − q_t). Steps at which the model has almost decided to copy, with q_t above 0.9, make up more than four-fifths of the steps with copy candidates. They fluctuate little and carry only about a fifth of the predicted response. Steps with q_t between 0.1 and 0.9 make up about a tenth of those steps but carry about four-fifths of the response, close to their share of the variance (Fig. 4a). The penalty thus acts in proportion to the model's uncertainty about whether to copy. This follows from the identity rather than being a separate finding, but it explains the failure of the equilibrium reading, which weights every chosen candidate equally.

This does not mean that the penalty only prevents loops from forming. We compared entry and exit rates with and without h*. About half of the change in ln K came from a lower entry rate and half from a higher exit rate, in the smaller and the larger models alike. The identity's own split of the predicted response agreed to within about 0.1, although in most models it put slightly more of the response on the exit rate (Fig. 4b; Supplementary Section S5). A penalty therefore both makes repetition harder to start and makes loops break more readily, and both effects arise mainly at the minority of steps where the model is uncertain.

---

## Discussion

For an additive penalty on copy candidates, and for the common multiplicative penalty at the strengths tested, the change in the balance between the two states can be predicted to first order from samples generated without the penalty. The resulting shift of the melting temperature was predicted less well, for reasons we have not identified. The prediction depends on how much each choice departed from the model's own probability of choosing a copy candidate, not on the number of candidates chosen. The equilibrium reading fails in a specific direction: it overstates the response. It credits the penalty with acting on every chosen candidate, but choosing a candidate inside a repetitive stretch was already almost certain and barely changes with a modest change in the logits.

Comparing step-wise and whole-sequence interventions makes this more precise. From the same unpenalized samples one can compute exactly how ln K would respond if each output were reweighted as a whole. Reweighting by the number of copies mimics an equilibrium penalty; reweighting by total energy mimics an equilibrium change of temperature. For the smaller models the Clausius–Clapeyron form approximately holds between these two whole-sequence responses. The actual sampler, which normalizes at every step, retains only one or two per cent of either whole-sequence response. The overstatement of the equilibrium reading is the ratio of these two small remainders (Supplementary Section S6). The whole-sequence responses are exact computations; what is interpretive is the attribution of the overstatement to the slightly smaller remainder for the penalty.

The balance between the two states drifts towards repetition along a text, so the generated text is not stationary. One might suspect that this, rather than the step-wise normalization, defeats the equilibrium reading. The whole-sequence comparison argues against that. It uses the same non-stationary samples, and for the smaller models the Clausius–Clapeyron form approximately holds between the two whole-sequence responses. Only the step-wise sampler departs from it. In exploratory checks, the equilibrium reading also overstated the response within each half of the generated tokens and under each of eight other labellings. The identity stayed within about 10% in each half (Supplementary Section S6).

Practically, the identity estimates the effect of a change to the logits, additive or multiplicative, from runs that have already been made. Locating the melting temperature required unpenalized runs at several temperatures, comparable in cost to trying a few penalty strengths. The advantage is that the same runs give the response to any such change and show where in the text it comes from. Reweighting the unpenalized samples by their exact probability ratio, the natural non-linear alternative, fails at these penalties because the weight concentrates on one or two samples per prompt (Methods). Other interventions that add to the logits at every step, such as logit biases and watermarking biases [2301.10226], share the step-wise normalization, and the identity applies to them formally. Whether it also predicts their effects on long-range statistics is untested. The identity carries over to truncated sampling such as min-p [2407.01082] or top-p [1904.09751] if q_t is taken within the truncated set. A first-order prediction then fails only through steps at which the penalty moves a token across the truncation boundary. Such steps can be ruled out one by one from the margins of the unpenalized logits (Supplementary Section S9), but we have not tested this with language models.

The study has clear limits. All prompts were English news articles, and the nine models are base models released by 2024 with at most 7 billion parameters. We did not test truncated sampling such as top-p with language models. For the multiplicative penalty, the first-order prediction held up to θ of about 1.1, the lower end of the values commonly used, and overstated the effect of stronger divisors. For instruction-tuned models the test did not decide: one model gave an interval too wide to decide, and the other could not be predicted.

The prediction is first order. It predicts the change in ln K better than the resulting shift of the melting temperature, and the confirmation of it rests on one preregistered replication and one test on two larger models, after an undecided first test. The overstatement by the equilibrium reading is a separate finding. It appears in all seven models in which we varied the penalty directly and, measured in ln K, in all nine prospective tests. Measured as a shift of the melting temperature it appears in seven of the nine, and for Mistral-7B-v0.3 its direction depends on which unpenalized data are used (Supplementary Sections S5 and S6). When the equilibrium prediction is built temperature by temperature from the replication's own unpenalized samples, instead of from the earlier runs, it is close to the measured response for GPT-2-XL and Mistral-7B-v0.3 (Supplementary Section S6). 〔A2-27 の結果待ち：エネルギーの取り方を −z や T × (−log p_T) に変えたときの 1 文〕 Both findings rest on a labelling of repetition, and K and the melting temperature are averages over the 512 generated tokens, along which the balance drifts. Repetitive stretches are not states of a two-state Markov chain, since leaving them becomes rarer the longer they last. Bootstrap intervals resample articles, not models.

---

## Figure legends

（図は `paper/figures/fig1〜4.pdf`、描くコードは `scripts/paper_figures.py`。図 1 だけ生のデータを読み、図 2〜4 は results/ の JSON から描く。）

**Fig. 1 | Generated text switches between two states.** **a**, Copy marks (grey ticks) and labelled state (line; up, repetitive) along one unpenalized continuation of Pythia-1.4B at T = 0.62, chosen as the first output with two to six switches. **b**, Distribution of the fraction of copies in a 32-token window at the test temperature closest to each model's melting temperature, for unpenalized continuations of the 120 replication articles. Grey, eight models; blue, Pythia-1.4B. Dotted lines, the thresholds for leaving (1/4) and entering (1/2) the repetitive state. **c**, ln K, the log ratio of entry rate to exit rate, against inverse temperature for the same runs. The melting temperature is where ln K = 0.

**Fig. 2 | The equilibrium reading overstates the effect of the penalty.** **a**, Measured shift of the melting temperature per unit h in the seven smaller models, from runs at h = −0.3, 0 and +0.3 on 40 articles, against the Clausius–Clapeyron prediction from unpenalized runs. All models lie below the diagonal. **b**, Measured response of ln K to h divided by the prediction of the score-function identity (left) and of the equilibrium reading (right), one point per model and temperature (7 × 7). Bars, medians. Exploratory data; the points within a model are not independent.

**Fig. 3 | Prospective tests.** Ratio of the realized change in ln K to the first-order prediction, with 95% bootstrap intervals over articles (blue), for each preregistered test. Diamonds, the test statistic (median or mean) with its interval. For the first test only the statistics are shown, for h* and 2h* (per-model values in Supplementary Section S4). Orange squares, the ratio divided by r_C, approximately the realized change relative to the equilibrium prediction recorded before the test, in ln K. Grey band, the preregistered range 0.8–1.25. Logarithmic axis.

**Fig. 4 | Where the penalty acts.** **a**, For steps with copy candidates, split by the model's probability q_t of choosing one (below 0.1, 0.1–0.9, above 0.9): share of the steps, share of the variance Σ q_t(1 − q_t) and share of the identity's predicted response, for the nine models at the test temperatures. **b**, Share of the realized change in ln K that comes from the change in entry rate (blue, 95% interval) and the share predicted by the identity (open diamonds). The rest comes from the exit rate. In most models the identity puts slightly more of the response on the exit rate.

## Table 1

**Table 1 | Models and prospective tests.** Melting temperature T_m: unpenalized, from the fit used for the prediction. h*: predicted additive penalty on the logits of copy candidates (before division by temperature) that should lower T_m by 0.10. Ratio: realized change in ln K divided by the identity's first-order prediction, summed over the test temperatures, with a 95% bootstrap interval over articles (120 articles in each test). r_C: factor by which the equilibrium reading, recorded before the test, exceeds the identity's prediction. Ratio / r_C: approximately the realized change divided by the equilibrium prediction, both expressed as changes in ln K. Rows 1–7: preregistered replication. Rows 8–9: preregistered test on 7-billion-parameter models. Decision band for the ratio: 0.8–1.25 (median over rows 1–7; mean over rows 8–9).

| Model | Parameters (billions) | T_m | h* | Ratio | 95% interval | r_C | Ratio / r_C |
|---|---|---|---|---|---|---|---|
| Pythia-410M | 0.41 | 0.60 | −0.40 | 1.12 | 0.97–1.33 | 1.5 | 0.73 |
| Pythia-1.4B | 1.4 | 0.57 | −0.50 | 1.03 | 0.92–1.20 | 2.0 | 0.51 |
| GPT-Neo-1.3B | 1.3 | 0.63 | −0.39 | 1.03 | 0.91–1.18 | 1.7 | 0.61 |
| SmolLM2-1.7B | 1.7 | 0.51 | −0.53 | 1.18 | 1.02–1.42 | 2.2 | 0.53 |
| Qwen2.5-1.5B | 1.5 | 0.60 | −0.47 | 0.94 | 0.80–1.17 | 2.2 | 0.42 |
| GPT-2-XL | 1.6 | 0.48 | −0.79 | 1.08 | 0.93–1.26 | 2.9 | 0.37 |
| BLOOM-1.1B | 1.1 | 0.66 | −0.37 | 1.04 | 0.88–1.26 | 1.5 | 0.69 |
| Median (rows 1–7) | | | | 1.04 | 0.98–1.15 | | 0.53 |
| Qwen2.5-7B | 7.6 | 0.55 | −0.67 | 0.88 | 0.73–1.10 | 3.4 | 0.26 |
| Mistral-7B-v0.3 | 7.2 | 0.44 | −0.35 | 1.04 | 0.86–1.30 | 1.7 | 0.62 |
| Mean (rows 8–9) | | | | 0.96 | 0.84–1.11 | | |

（出どころ：results/A2_18/hstar.json・results/A2_20/hstar7b.json（T_m、h*、r_C）、results/A2_19/a2_19_summary.json・results/A2_20/a2_20_summary.json（比と区間）。Ratio / r_C は比 ÷ r_C で、A2-22 の結果ノートの文（0.26〜0.73・中央値 0.53）と一致。ただし A2-22 のゲートが前もって書いた記録は融点の動きでの比 ΔT_m ÷（−0.10 × r_C）（JSON の ratio_over_rC、0.53〜1.26）で、別の量。両方を補足 S5 に載せる（2026-11-03 の新鮮な目のレビューで見つかった）。大きさ（パラメータの数）は公開の値で、投稿前にモデルカードで確かめる。）

## Methods

（v0.5 の下書き、2026-11-03 08:2x JST。設定の数は、ゲート文書とコードから調べ役のサブエージェントが file:line つきで拾い、Claude が要所をコードで確かめた。数の出どころは各段落の後ろの（）に。投稿前にもう一度突き合わせる。）

### Models, prompts and sampling

We used nine open base models: Pythia-410M and Pythia-1.4B (EleutherAI/pythia-410m, EleutherAI/pythia-1.4b), GPT-Neo-1.3B (EleutherAI/gpt-neo-1.3B), SmolLM2-1.7B (HuggingFaceTB/SmolLM2-1.7B), Qwen2.5-1.5B and Qwen2.5-7B (Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-7B), GPT-2-XL (openai-community/gpt2-xl), BLOOM-1.1B (bigscience/bloom-1b1) and Mistral-7B-v0.3 (mistralai/Mistral-7B-v0.3). The instruction-tuned models in Supplementary Section S7 were Qwen/Qwen2.5-1.5B-Instruct and HuggingFaceTB/SmolLM2-1.7B-Instruct. Weights were loaded from the Hugging Face Hub without pinning a revision, which we list as a limitation for exact reproduction.

Prompts came from recent BBC News articles in the RealTimeData/bbc_news_alltime collection [LatestEval, Li et al. 2024] (April to June 2025). We split each article into paragraphs, dropped paragraphs shorter than 60 characters and boilerplate lines such as image captions, and kept articles of at least 4,000 characters. The first set of 40 articles was drawn with a fixed seed after sorting the pool by a hash of the article link. A second set of 120 articles, in three groups of 40, was drawn later from the same pool after removing duplicates and the first 40. The two sets do not overlap.

The prompt was the first 64 tokens of the cleaned article in each model's own tokenizer, with a beginning-of-sequence token only where the tokenizer adds one by default. Each prompt was continued for 512 tokens, with four samples per prompt. Sampling drew each token from softmax(z/T), where z are the logits, with no top-k or top-p truncation. End-of-sequence tokens, the <|endoftext|> token and unused padding entries of the vocabulary were excluded, so that every continuation had the same length. Articles shorter than 576 tokens in a model's tokenizer were not used. All computation used 32-bit floating point with TF32 disabled, and sampling probabilities were computed in 64-bit floating point.

Random seeds depended on the temperature, the sample and the batch, but not on the penalty. Penalized and unpenalized runs therefore shared their random numbers, which reduces the variance of their difference. Some runs were inadvertently repeated on separate L4 GPUs, and the repeats gave bit-identical outputs.

（scripts/a2_2_collect.py・a2_19_collect.py・a2_gpre_collect.py・a2_explore_dsc2.py。プールの大きさはゲートと結果ノートで食い違う（2,898 と 1,852）ので、本文には書かない。）

### Copies, states and rates

We marked step t of a continuation as a copy if the four tokens ending at t had occurred, in the same order, earlier in the same continuation, overlapping occurrences included. The prompt is not used for this mark. We then computed the fraction of copies in a sliding window over steps k to k + 31 and assigned it to step k + 16. States were labelled for steps 16 to 496 with hysteresis. A step enters the repetitive state when the windowed fraction reaches 1/2 and returns to the ordinary state when it falls below 1/4. The first labelled step is repetitive if its window fraction is at least 1/2 and ordinary otherwise.

At each temperature we summed four counts over samples and articles. N₀ and N₁ are the numbers of steps in the ordinary and the repetitive state. n₀₁ is the number of switches from ordinary to repetitive, and n₁₀ the number of switches back. The entry rate is n₀₁/N₀ and the exit rate n₁₀/N₁, and

ln K = ln(n₀₁/N₀) − ln(n₁₀/N₁).

All four counts are taken over pairs of consecutive labelled steps, classified by the state of the first step of the pair. A temperature was used only if it had at least ten entries and ten exits, and a fit needed at least three usable temperatures.

We fitted ln K = a + b/T by weighted least squares over the usable temperatures, with weights 1/(1/n₀₁ + 1/n₁₀), the inverse of the approximate variance of ln K. The melting temperature is T_m = −b/a, defined when b > 0 and a < 0. Fits used seven temperatures spaced by 0.05. For the seven smaller models these were 0.40 to 0.70. For the two larger models we first fitted 15 temperatures from 0.20 to 0.90, then refitted on the seven consecutive temperatures centred near that estimate.

（scripts/a2_explore_onset.py の copy_marks、a2_8_vh.py の phases と fit_vh、a2_18_analyze.py の seq_events と lnK_from、a2_20_predict.py。）

### Copy candidates and penalties

The penalty acts on copy candidates. At each step these are the tokens that have previously followed the current last two tokens, anywhere in the prompt or the continuation. The additive penalty adds h to their logits before division by the temperature, so that negative h discourages repetition. q_t is the probability of the candidate set under softmax(z/T), and c_t indicates whether the sampled token was a candidate. The marks used for labelling and the candidates used for the penalty are related but not identical. This keeps the definition of the states independent of the penalty.

The multiplicative penalty follows the implementation in the Hugging Face transformers library [1910.03771], which adapts the penalty of CTRL [1909.05858]. Before temperature scaling, the logit of every token that has appeared earlier, including in the prompt, is divided by θ if it is positive and multiplied by θ otherwise.

（scripts/a2_7_collect.py の CopyIndexN（n ＝ 3、書き出しを含む）、a2_7b_collect.py の B7_N、a2_explore_dsc2.py、a2_23_collect.py。）

### Score-function identity

For token-by-token sampling the log-probability of a continuation x is Σ_t log p_t(x_t), where p_t depends on the penalty. For the additive penalty, ∂ log p_t(x_t)/∂h = (c_t − q_t)/T. Hence, for any statistic Y of the continuation, with the sum running over all 512 generated steps,

∂E[Y]/∂h at h = 0 equals Cov(Y, σ)/T, with σ = Σ_t (c_t − q_t),

where the covariance is taken over unpenalized samples. We applied this to each of the four counts N₀, n₀₁, N₁ and n₁₀ of a continuation. With d_i = Cov(e_i, σ)/(T E[e_i]) for count e_i, the predicted response is

g(T) = ∂ln K/∂h = (d₀₁ − d₀) − (d₁₀ − d₁).

Covariances were population covariances pooled over all samples and articles at a temperature, with article weights for the bootstrap. For the multiplicative penalty, with λ = ln θ, the derivative of a logit at λ = 0 is −|z| for tokens already seen and zero otherwise. The same identity then holds with σ replaced by Σ_t A_t, where A_t = a(x_t) − Σ_v p_t(v) a(v) and a(v) = −|z_v| for seen tokens. The identity is exact; the prediction of a finite penalty from it is first order.

（scripts/a2_18_analyze.py の g_identity、a2_16_analyze.py、a2_23_collect.py。）

### Response to a fixed penalty

For the seven smaller models we generated continuations of the first 40 articles at seven temperatures from 0.40 to 0.70, with h = −0.3, 0 and +0.3. From these runs we fitted T_m at each h and took the slope of T_m against h as the measured shift. At each temperature the measured response of ln K was (ln K(+0.3) − ln K(−0.3))/0.6. The unpenalized arm of these runs also served to predict h* and θ*.

（scripts/a2_gpre_collect.py、a2_16_analyze.py の records_dlnK。）

### Equilibrium prediction

The equilibrium reading treats the penalty as an energy h per chosen copy candidate in a two-state system. The Clausius–Clapeyron relation then gives the shift of the melting temperature per unit h as T_m Δc/ΔE. Δc is the difference between the repetitive and ordinary states in the rate at which a candidate is chosen. ΔE is the difference in mean energy per step, where the energy of a step is −log p of the sampled token at T = 1. ΔE is taken as the ordinary-state mean minus the repetitive-state mean. Both were averaged over the usable temperatures of the unpenalized runs. Temperature by temperature, the equilibrium form of the response of ln K is n Δc_T/T, with Δc_T measured at that temperature (Fig. 2b). Here n = b/ΔE is the ratio of the van 't Hoff slope to the energy difference. For each prospective test we recorded, before the test, the factor r_C by which this reading exceeds the identity's prediction for the chosen h*.

（scripts/a2_18_predict.py・a2_20_predict.py の s_clapeyron と r_clapeyron。r_C は小さい 7 本で 1.5〜2.9、7B で 3.4 と 1.7。）

### Prospective tests

Each test followed the same procedure, fixed in a dated document before any penalized text was generated. From unpenalized runs we estimated a, b and T_m, and the mean ḡ of g(T) over the fitting temperatures (over the temperatures with a defined g, at least five, for the larger models). The predicted penalty was h* = −(a + b/T*)/ḡ with T* = T_m − 0.10, rounded to four decimals for generation. We then generated unpenalized and penalized continuations at seven temperatures, T* rounded to 0.01 plus or minus up to 0.15 in steps of 0.05, on the same articles and with shared random numbers.

For each model, Δ(T) = ln K(T, h*) − ln K(T, 0) was compared with the prediction P(T) = h* g(T), where g was computed from the unpenalized arm of the same test. The ratio was ρ = Σ_u Δ(T_u)/Σ_u P(T_u) over the usable temperatures u. Usable temperatures needed at least ten entries and ten exits in both arms. They were fixed at the point estimate and kept in every bootstrap resample. A model was dropped if fewer than two temperatures were usable, and the median was computed only if at most two of the seven models were dropped. The test statistic was the median of ρ over the seven smaller models, or the mean over the two larger ones. Intervals were 2.5–97.5 percentile intervals from 1,000 bootstrap resamples of articles, using the same resample for all models and both arms.

The decision rule was set before each test. The prediction was confirmed if the 95% interval of the statistic lay within 0.8 to 1.25. It was called larger or smaller than predicted if the interval lay entirely above 1.25 or below 0.8, and reversed if it lay entirely below 0. It was undecided otherwise, or if the statistic could not be computed in more than 10% of resamples. For the larger models, T_m was first fitted on 15 temperatures from 0.20 to 0.90. The fit was then repeated on the seven consecutive temperatures centred on the grid point nearest that estimate, and re-centred once if the estimate moved. A model was not predicted if a fit failed or the 15-temperature estimate fell outside 0.3 to 0.8. It was also not predicted if the final T_m fell outside its seven temperatures, if fewer than five temperatures gave g, if ḡ was not positive, or if the test grid would reach below 0.10.

There were three tests of the additive penalty. The first used the 40 articles from which h* had been computed, with arms at h* and 2h*, and both were undecided. A single replication on the 120 new articles was then committed to, with its decision rule, before any of its data existed. The third test applied the whole procedure, including the estimation of T_m, to the two 7-billion-parameter models on the same 120 articles. The test of instruction-tuned models (Supplementary Section S7) followed the rules for the larger models. For the multiplicative penalty, ∂ln K/∂λ was computed by re-scoring the unpenalized runs that gave h*, without generating new text, and averaged over their seven temperatures to give ḡ_λ. With the same a, b and T*, the prediction was λ* = −(a + b/T*)/ḡ_λ and θ* = e^{λ*}, rounded to four decimals. A model was not predicted if ḡ_λ was not negative or θ* fell outside 1 to 3. The unpenalized arm of the replication served as the control, and the decision rule was unchanged.

Before each test we committed the procedure, the analysis code with its SHA-256 hashes and the decision rule to a version-controlled repository. Predictions were written to files whose hashes were recorded before penalized generation, and the notebooks that generated penalized text checked these hashes before running. One post-freeze change was made to analysis code, a missing import statement in the script for the whole-sequence analysis; it did not change any computation.

（docs/A2_18_GATE.md・A2_19_GATE.md・A2_20_GATE.md・A2_23_GATE.md。ゲートの凍結は日本時間の 10 分刻みで記録。A2-18・A2-19 のハッシュは 16 桁の頭だけ、A2-20 からは全部。）

### Whole-sequence responses

For a continuation let N = Σ_t c_t, Q = Σ_t q_t and E = Σ_t U_t, with U_t the energy of step t at T = 1. Reweighting each unpenalized continuation as a whole by exp(hN/T) would change ln K at rate g_N, computed as g(T) above with σ replaced by N. The step-wise penalty has σ = N − Q, so g = g_N − g_Q exactly when all three use the same covariance convention. A whole-sequence change of temperature gives the enthalpy-like quantity ΔH_whole = −T g_E. For these comparisons, covariances were computed within articles and multiplied by S/(S − 1) with S = 4 samples.

（scripts/a2_21_analyze.py。）

### Importance reweighting

Reweighting unpenalized continuations by their exact probability ratio gives log-weights

log w = Σ_t [ (h/T) c_t − log(1 − q_t + q_t e^{h/T}) ].

We measured the effective sample size per article as (Σw)²/Σw² over its four samples, averaged over temperatures and articles. At h*, it was between 1.3 and 2.1 out of 4.

（scripts/a2_22_analyze.py。）

### Exploratory analyses after the prospective tests

After the prospective tests we ran three further analyses of the unpenalized and penalized runs of the replication and of the larger models, fixing their definitions and decision rules in a dated document before running them. First, we split the score σ into the contributions of steps with q_t below 0.1, between 0.1 and 0.9, and above 0.9, computed the identity's response for each part, and compared each part's share of the response with its share of the variance Σ q_t(1 − q_t). Second, at the test temperature closest to each model's melting temperature, we examined the distribution of the windowed copy fraction, for all outputs and for outputs that switch state at least once. Third, we repeated the replication's ratio with windows of 16, 32 and 64 tokens and thresholds of (1/2, 1/4), (0.6, 0.3) and (0.4, 0.2), recomputing the states, the counts and g for each choice. We also compared ln K between the first and second halves of the labelled steps. These analyses are reported as exploratory (Supplementary Section S2).

### Compute

Generation for the smaller models ran on NVIDIA L4 GPUs and for the 7-billion-parameter models on NVIDIA A100 (40 GB) GPUs, through a cloud notebook service. Each prospective test of the seven smaller models took roughly 10–20 L4 GPU-hours. The prediction itself needs only the unpenalized runs and a few minutes of CPU time. All analyses ran on CPU.

（時間はゲートの見込みと通知の時刻からの概算。投稿前に、npz と run_meta の秒数から足し直す。）

### Code and AI assistance

The author designed the questions, the tests and the decision rules. The analysis and data-collection code was written by an AI assistant (Claude, Anthropic) under the author's direction, and the author did not check it line by line. The code was checked in four ways. First, each analysis ran self-tests on synthetic data with known answers before touching real data. Second, a separate AI instance without access to the project history reviewed each design and its code before data collection. Third, quantities computed by later scripts were required to match earlier recorded results to within 10⁻⁹. Fourth, runs that had been inadvertently repeated on separate GPUs were compared bit for bit. Finally, a separate AI instance that had not seen the code reimplemented the estimator from the Methods text alone; on the replication data it reproduced each model's ratio and the median to four decimal places.

（著者は 1 行ずつは確かめていない、をはっきり書く。中心の推定量を別に一から書き直して一致を見るかは未定。）

### Data and code availability

The BBC article texts cannot be redistributed. We release, for every article, the collection month and SHA-256 hashes of its link and text, together with the code that rebuilds the prompt sets from the public collection. All code, gate documents, sealed predictions and summary results are available in the project repository. Generated continuations are available on request.

（記事の識別子の JSON（docs_meta）は今は Drive にだけある。公開のときにリポジトリに入れる。生成した文の公開は、BBC の文を含むのでオーナーと相談。）

## 補足（Supplementary）の予定

- S1 表 1 の全部の数（モデルごとの融点、h*、比、区間）
- S2 ものさしの確かめ（空の作り物の見かけ、続けて写すものさし）と、A2-24 の探索（分け方 9 通りの表、融点に近い温度の分布と前もって決めた 2 つの山の決まりを満たさなかったこと、前半と後半の ln K、q で分けた効き）。図 S1（罰 ±0.3 での融点、もとの図 2a）
- S3 文章の種類（小説・論文・コード）
- S4 1 回目の確かめ（40 本、h* と 2h*）
- S5 融点の動き（A2-22 の XQ10 の表と、狙いからのずれ）
- S6 文全体と 1 歩ずつの比べ（A2-21）、エネルギーから見た状態の差と協同性。A2-26 の探索：温度ごとに組み立てた平衡の読みの比（9 本の中央値 0.69、GPT-2-XL 0.98・Mistral-7B 0.96）、前半・後半（0.73・0.59、恒等式は 1.10・0.91）、分け方 8 通り（0.63〜0.74）、くり返しから出る割合と長さ（0.28）、融点の行きすぎの内訳（設計の温度での ln K は狙いの近く、ln K は 9 本すべてで 1/T に対して上に凸）。A2-27 のエネルギーの取り方（結果待ち）
- S7 割る型の罰と指示調整済みのモデル（A2-23。下書きは下）
- S8 2 次の項と厳密に解ける模型（8.36・8.41・8.42）
- S9 切り詰めつきのサンプリング（Min-p・top-p）への恒等式の持ち込み（本文は考察の 2 文だけ。2026-11-03 13:1x、オーナー〔オーナーの言葉〕）：切り詰めたあとの q で成り立つこと、有限の罰では候補集合の入れかわりが 1 次の外に出ること、Min-p の十分条件（マージン ＞ |h|/T。足す型の罰は logit を同じ向きにしか動かさないので、PROJECT_A の `notes/MARGIN_STABILITY.md` の定理 1 の片側の形（osc ＝ |h|/T）が使える。定理 1 は短いので Claude が自分で確かめた。ほかのサンプラーの係数は AI が証明し AI が監査したもので、人の確認はない）。作り物の数え上げ：`scripts/check_truncated_identity.py`、`docs/TRUNCATION_NOTE.md`。

確かめた点（v0.2 から引き継ぎ）：状態ごとの写す割合（9 本）、2 つの山（63 のうち 61）、ln K の曲がり、罰の定義のコード。v0.3 で足した数の出どころ：前もって書いた平衡の読みの倍率 1.5〜2.9（A2-18 の予言のファイル）、平衡の読みに対する比 0.37〜0.73（A2-22、小さい 7 本。7B は 0.26・0.62）、融点の着地 9 本のうち 6 本（A2-22）、状態ごとの寄与 61〜95%（A2-22）、入る側の割合 0.48・0.44（A2-22）、恒等式の f_入 との差 約 0.1 以内（A2-22 で 0.00〜0.12）、文全体の応答の残り 1〜2%（A2-21 で 0.7〜2.2%）。

### Supplementary Section S7: Multiplicative penalty and instruction-tuned models

**Multiplicative penalty.** We used the definition of the widely used implementation in the Hugging Face transformers library: before temperature scaling, the logit z of every token that has appeared earlier, including the prompt, becomes z/θ if z > 0 and zθ otherwise. With λ = ln θ, the derivative of a logit with respect to λ at λ = 0 is −|z| for seen tokens and zero otherwise, and the identity gives ∂ln K/∂λ from unpenalized samples. We rescored the unpenalized samples of the earlier experiments without generating new text, predicted θ* that lowers the melting temperature by 0.10, and fixed these predictions before sampling with θ*. The control arm and the decision rule were those of the preregistered replication (120 articles, ratio of realized to predicted change in ln K, bootstrap interval of the median within 0.8–1.25).

| Model | θ* | Ratio | 95% interval |
|---|---|---|---|
| Pythia-410M | 1.021 | 0.99 | 0.85–1.20 |
| Pythia-1.4B | 1.027 | 1.12 | 0.94–1.39 |
| GPT-Neo-1.3B | 1.508 | 1.32 | 0.87–2.71 |
| SmolLM2-1.7B | 1.039 | 1.02 | 0.81–1.36 |
| Qwen2.5-1.5B | 1.022 | 0.94 | 0.79–1.18 |
| GPT-2-XL | 1.054 | 1.01 | 0.82–1.28 |
| BLOOM-1.1B | 1.019 | 0.91 | 0.76–1.12 |
| Median | | 1.01 | 0.92–1.14 |

The predicted divisors were 1.02–1.05 for six models and 1.51 for GPT-Neo-1.3B, whose ln K responded 13–22 times less to the divisor. The divisor acts on raw logits, so its effect depends on their scale in each model. By the same first-order reasoning, the values of 1.1–1.3 that are common in practice (CTRL suggests about 1.2 [1909.05858]) would move the melting temperature by much more than 0.10 in the six models.

**Stronger divisors.** In a further preregistered test (Supplementary gate A2-25), λ = ln θ was set to k times the predicted λ* with k = 2, 4 and 8, rounded to four decimals, for the same 120 articles and temperatures, with the replication's unpenalized arm as control. Divisors above 3 were not run, so GPT-Neo-1.3B was tested at k = 2 only and the decision used the other six models at every k.

| k | θ (six models) | Median ratio | 95% interval | Decision |
|---|---|---|---|---|
| 1 | 1.02–1.05 | 1.00 | 0.91–1.11 | (replication, for reference) |
| 2 | 1.04–1.11 | 1.00 | 0.90–1.12 | within the range |
| 4 | 1.08–1.24 | 0.86 | 0.77–0.99 | undecided |
| 8 | 1.17–1.53 | 0.52 | 0.46–0.67 | below the range |

At k = 8 the penalty suppressed repetition so strongly that Pythia-410M kept only one usable temperature and dropped out, and three other models kept two or three. Models differed in how fast the ratio fell: SmolLM2-1.7B reached 0.24 at k = 8, whereas both Pythia models stayed near 1 up to k = 4. GPT-Neo-1.3B at k = 2 (θ = 2.27) gave 1.64 (95% interval 1.11–3.59), the only response larger than predicted.

**Instruction-tuned models.** Prompts were wrapped in each model's chat template, with the user turn "Continue this news article:" followed by the first 64 tokens of the article. For Qwen2.5-1.5B-Instruct the unpenalized melting temperature was 0.65 and the predicted additive penalty h* = −0.33. The ratio was 1.36 with a 95% interval of 0.81–2.40, which neither confirms nor rejects the prediction. For SmolLM2-1.7B-Instruct the provisional melting temperature, 0.16, lay outside the range in which our procedure makes a prediction (0.3–0.8), so no test was run.
