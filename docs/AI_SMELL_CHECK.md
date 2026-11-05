# 論文の「AI 臭さ」の定義と検出（PRX の原稿 v0.5）

作成：2026-11-05 09:50（Claude）。オーナーの依頼（2026-11-05 09:45）「まるで肉を香味野菜と一緒に煮て臭みを抜くように、文章からとことんAI臭さを消したいな。というわけでまずはあんちすろっぷのスキルと、それ以外の資料……インターネットとかそれ関連の論文？とかからAIらしさを調べてきて定義して、臭みを検出してほしい！それから臭みをどう消すかかんがえよう！」

この文書は、定義・検出・消し方の候補までをまとめたもの。本文はまだ直していない。数えるスクリプトは `scripts/ai_smell_check.py`、結果は `results/paper_checks/ai_smell.json`。対象は `paper/review_copy_v05.md`（英語だけの写し）。表の行は数えず、図表の説明（caption）は文の長さなどから外した。語数は、要旨 303、本文（I〜V 章）4,014、付録 7,333。

## 1. 出典

- あんちすろっぷのスキル（anti-slop-paper）と PROJECT_C の `docs/PAPER_WRITING_GUIDE.md`：査読者が挙げたスロップの特徴（結論に効かない実験、数字の多さ、言いすぎ・ことわりだらけ、箇条書きの vibe report）、数字の予算（結果の節に 3〜4 個）、名詞化が多い・ふくらんで抽象的な文体。
- Kobak et al., "Delving into LLM-assisted writing in biomedical publications through excess vocabulary"（arXiv:2406.07016、Science Advances）：2024 年の PubMed の要旨の少なくとも 13.5% が LLM を通っていた。急に増えた語は 66% が動詞、14% が形容詞。名前の出ている語は delves、underscores、showcasing、potential、findings、crucial と、下限の推定に使った across、additionally、comprehensive、crucial、enhancing、exhibited、insights、notably、particularly、within。
- Liang et al., "Mapping the Increasing Use of LLMs in Scientific Papers"（arXiv:2404.01268）：計算機科学の論文で LLM が手を入れた割合がいちばん大きく、最大 17.5%。
- Reinhart et al., "Do LLMs write like humans? Variation in grammatical and rhetorical styles"（arXiv:2410.16107）：指示に従うよう調整したモデルは、文の後ろにつく現在分詞の句（", revealing …"）を人の 2〜5 倍、名詞化を 1.5〜2 倍、主語になる that 節、句の並列を多く使う。「情報を詰めこんだ、名詞の多い文体」。
- Wikipedia: Signs of AI writing（WikiProject AI Cleanup）：意義の誇張（stands as、testament、pivotal、key role）、浅い分析の -ing の句（highlighting、ensuring、fostering）、宣伝の口調、あいまいな出典、「課題と展望」型のまとめ、AI の語彙（Additionally、delve、crucial、underscore、tapestry、showcasing）、is の言いかえ（serves as、stands as）、否定の並列（Not only … but、It's not just …, it's …）。ただし「しるしは証拠ではない」「表面のしるしだけ消しても中身の問題は直らない」と注意している。
- Freeburg, "The Last Fingerprint: How Markdown Training Shapes LLM Prose"（arXiv:2603.27006）：em ダッシュは、マークダウンを禁じても残る。
- 「Signs of AI writing」のまとめ（jetwong.neocities.org）：三つ並べの癖、否定からの切り返し、型どおりの段落（主題文・根拠・まとめ）、まとめのくり返し、演出の短い文（"The result?"）、ぼかしの連鎖、文の長さが均一（burstiness が低い）。
- 注意：AI 検出器は誤りが多く、英語を母語としない書き手の文を AI と誤判定しやすいことが知られている（Liang et al. 2023, "GPT detectors are biased against non-native English writers", Patterns）。この調べは「検出器に通す」ためではなく、読み手が受ける印象を良くするためのもの。

## 2. 定義（5 つの層）

| 層 | 臭み | 数え方 | 出典 |
|---|---|---|---|
| L1 単語 | AI が増やした語・決まり文句 | 上の語の一覧と、"plays a role"、"it is worth noting" などの句 | Kobak、Wikipedia |
| L2 文の形 | 文末の -ing の句、is の言いかえ、否定と対比（not X but Y、X, not Y）、三つ並べ、em ダッシュ、コロン、セミコロン、名詞化 | 正規表現 | Reinhart、Wikipedia、Freeburg、まとめ |
| L3 運び | 35 語を超える文、6 語以下の演出の短い文、まとめの文頭（Thus など）、文頭の単調さ、文の長さのばらつき | 文に分けて数える | スキル、まとめ |
| L4 中身 | 本文の数字の多さ、ぼかし語、記録の言葉（registered、dated document、rule fixed）、付録への参照の多さ | 節ごとに数える | スキル、ガイド |
| L5 Claude の癖 | "What X is, is Y" 型の種明かし、強調の only・exactly、先回りの否定（"This does not mean …"、"…, not a test of …"） | 正規表現と目視 | **出典なし。この原稿を書いた AI（僕）の見立て** |

## 3. 検出の結果

### 典型的な AI のしるしは、ほとんどない
- L1：Kobak の語は within（23）と across（4）だけで、どちらも数学の文でふつうに使う意味。delve、crucial、pivotal、showcasing、Moreover、Furthermore、serves as は 0。
- L2：文末の -ing の句は本文で 5 個（1.25/千語）。Reinhart がいちばん強いしるしに挙げた形は少ない。em ダッシュは 0。三つ並べは 2。

### 臭みが強いところ（本文）
1. **数字が多い（L4）**：本文の数字（1 桁の整数と文献番号は除く）は 52 個。節ごとに、IV B が 45、IV D が 21、V（考察）が 13、IV A が 9、IV C が 9。ガイドの予算（結果の節に 3〜4 個）を大きく超える。IV B の段落 2 は、1.17、1.10、1.25、1.04、0.98–1.15、0.01、0.96–1.08、0.01、−0.446、0.161、0.96、0.84–1.11、1.05、0.99–1.11、1.07 と、1 段落に 15 個以上。
2. **記録の言葉（L4）**：本文 4,800 語ほど（表の説明を含む）の中に、"registered / in advance / fixed before / dated document / beforehand" が 18 回、"exploratory" が 7 回、付録への参照が 33 回（6.8/千語）。「記録の書き方をそのまま論文に持ち込む」臭み。
3. **先回りの否定（L5）**："The agreement is not an artifact of …"、"This is a decomposition …, not a test of …"、"This does not mean that …"、"These are ratios of derivatives at h = 0; we did not apply …"、"These intervals do not include …"。査読の回を重ねるうちに、まだ聞かれていない反論に答える文が積もった。
4. **演出の短い文と種明かし（L3・L5）**："The ingredients are known."、"The difference has a direct meaning."、"A step-wise penalty lacks it."、"The study has clear limits."、"What it removes is …"、"What we add is …"、"For a user, what matters is …"。
5. **文頭の単調さ（L3）**：本文 206 文のうち 57 文（28%）が "The" で始まる。付録も 402 文のうち 110 文（27%）。
6. **セミコロン（L2）**：本文の地の文で 16 文（文の 8%）。"A; B" で 2 つの文をつなぐ形が多い（"Repeats define the states below; candidates define the penalty."）。
7. **要旨（L3・L4）**：14 文。数字は 1.04、0.8–1.25、99% の 3 種類で予算内。ただし定義（h、N、Q、T）を要旨で説明しているため長い。最後の文 "By this prediction, the response arises mainly at …" は意味が取りにくい。

### 数の目安についての注意
人が書いた物理の論文での同じ数（"The" で始まる文の割合、セミコロンの数、名詞化の数）は、手元に基準がない。ここでの判断は「出典にある向き」と「節どうしの比較」と「目で読んだ印象」による。名詞化は本文で 33.6/千語だが、人の論文と比べられないので判定しない。

## 4. 消し方の候補（オーナーと決める）

- **A. 数字を減らす**（効き目が大きい）：IV B を、判定に使った数（1.04 と区間、7B の 0.96）だけにし、ほかは表 I・図 2・付録へ移す。IV D と考察も同じ。
- **B. 記録の言葉を 1 か所に**：「事前登録」の説明は II 章か IV B の頭で 1 回だけ言い、各文の "registered in advance"、"exploratory, rule fixed" は表 C1 に任せる。付録への参照も、段落に 1 つまでにする。
- **C. 先回りの否定を減らす**：本当に読者が誤解しそうなもの（2 状態は相ではない、など）だけ残し、ほかは消すか限界の節へ。
- **D. 演出の文と種明かしをふつうの文に**："The ingredients are known." → 前後の文に溶かす。"What it removes is X" → "It removes X."
- **E. 文頭と文のつなぎを散らす**："The …" で始まる文を減らし、セミコロンの文を 2 文に分けるか、because・while でつなぐ。
- **F. 確かめ**：直したあと、経緯を知らない AI と、できれば人にも、段落ごとに「AI が書いたと感じるところ」を聞く。書いた本人（この会話の AI）は判定役にしない。

## 5. A〜E を当てた結果（2026-11-05 09:58）

オーナー（2026-11-05 09:55）「うん！まずはAからEまで全部やって！」。下書き docs/PAPER_DRAFT_PRX.md の要旨と I〜V 章を直した（付録はそのまま）。置きかえの手順は、英語の写しの段落を下書きの中で 1 回だけあることを確かめてから置きかえる形にした。消した数字はすべて表・図・付録に残っている（F1、F3、F5、F6、F10、表 I、表 F2）。

| 数 | 前 | 後 |
|---|---|---|
| 本文の語数（表・図の説明を除く） | 4,014 | 3,553 |
| 本文の数字（1 桁の整数と文献番号は除く） | 52（IV B だけで 45） | 32（IV B 20、IV D 4、IV C 4、IV A 3） |
| "registered / in advance / fixed before / dated document" | 18 | 11 |
| "exploratory" | 7 | 4 |
| 地の文のセミコロン | 16 | 4 |
| "The" で始まる文 | 57 / 206（28%） | 31 / 189（16%） |
| 種明かしの形（What X is …） | 要旨 1、本文 1 | 0 |
| 演出の短い文（The ingredients are known. など 4 つ） | 4 | 0 |

- A：IV B の段落 2 は、最初の試験の 1.17・1.10、傾けた応答の 0.01、ラベルを変えた 0.96–1.08、温度ごとの 0.01（−0.446〜0.161）、設計だけの予測の 1.05・1.07 を付録へ。IV A は 74–89%・92–94%・16–21% を言葉に（most、almost every step、about one step in five）。IV C は 512・256・1.8%・1.2% を外した。IV D は判定の幅（0.8–1.25、0.33–0.67）と分け方の数を外し、0.96 だけ残した。考察の限界の段落から 1.32・0.87（IV B にある）を外した。
- B：事前登録の説明は IV B の頭の 1 回に寄せ、ほかは "exploratory; Table C1" の短い形にした。要旨と I 章の 1 回ずつは残した。
- C：先回りの否定のうち、"The agreement is not an artifact of …"、"This is a decomposition …, not a test of …"、"This does not mean that …"、"These intervals do not include …" を消すか、肯定の文（"The penalty acts at both ends of a repetitive stretch."）にした。"The two states are a labeling, not distinct phases." は読者が本当に誤解しそうなので残した。
- D："The ingredients are known."、"The difference has a direct meaning."、"A step-wise penalty lacks it."、"The study has clear limits."、"What it removes is …"、"What we add is …"、"For a user, what matters is …"、要旨の最後の "By this prediction, …" を、ふつうの文にした。
- E：セミコロンの文を 2 文か and・but でつなぐ形に。"The" で始まる文のうち 17 文の頭を変えた（意味は変えない）。

残っているもの：強調の only（本文 17）、付録への参照（本文と図表の説明で 31）。only は多くが意味の上で必要（"only at strengths below …" など）なので、F の確かめで指摘されたものだけ直す。

## 6. F（経緯を知らない読み手）の答えと、2 回目の直し（2026-11-05 10:42）

読み手は 2 人。この会話の AI が立てた、2 つのファイルしか見ていない AI（docs/ai_smell_review/answer_blind_claude.md）と、オーナーが頼んだ 外部の AI レビュー役（High。docs/ai_smell_review/answer_external_ai.md）。頼み方の文と段落番号つきの本文は同じフォルダにある。

二人が 2 点以上をつけた段落：P1（要旨、2・2）、P31（IV C、2・2）、P37（考察の最初、3・2）。片方だけが 2 点以上：P2、P5（外部の AI レビュー役 は 3）、P20、P32、P39、P40、P41。二人とも「人が書いたように読める」としたのは P28（T_× が狙いより動き、理由が分からないと認めた段落）。

二人とも指したので直したところ：同じ仕組み（後の繰り返しの分の得点）を 4 回言っていたのを III B と要旨に寄せた（IV C は 1 文、考察からは消した）。IV D の締めの文、考察の「because it changes what a penalty is」と対句、IV C の「give the same picture」「keep more」を具体的に。要旨の主語と動詞の距離、III B の ν_t の文、"the second"、s_temp の定義の順番（g_C を III D の前に使っていた）。

外部の AI レビュー役 が見つけた事実のずれ（直した）：限界の段落の「判定はモデルの中央値」→ 7 モデルは中央値、2 モデルは平均。臭み抜き 1 回目で書いた「どちらの割合も h = 0 での微分の比」は温度に当てはまらない → 「ゼロの罰とゼロの傾きでの一次の量」。「実用値の下端」と「一般的な値より下」の言い方のずれ → 要旨・IV B・限界・送り状を「the weakest strengths/divisors used in practice」にそろえた。

経緯を知らない AI が見つけた用語のずれ（直した）：Setup で定義前に使っていた design temperature と q_t、要旨の prompts と本文の articles、同じものを the identity / the derivative と呼び分けていた図表の説明、最初の試験の「two intervals」が何かを本文で言っていなかったこと、掛け算型の「twice / eight times」が何の倍か。

残したもの：III A の「停止の時刻があっても成り立つ」の一文（経緯を知らない AI は消せると言い、外部の AI レビュー役 は 0 点。物理の査読者が気にする点なので残す）。限界の段落の並べ方（二人とも「チェックリストのよう」と言ったが、一つずつ具体的な開示なので順番だけ保つ）。

本文の語数 3,553 → 3,450、"The" で始まる文 31/179。

## 7. 2 回目の読み手と、3 回目の直し（2026-11-05 10:57）

オーナー（2026-11-05 10:43）「もーいちどよんでもらお！論文は小説と違って誤字脱字が許されなさそうだし文体次第じゃあ嫌われそうだもん！」。読み手は新しく立てた AI と、新しいチャットの 外部の AI レビュー役（High）。質問に 6「誤字・文法・抜けた語・用語や記号の不ぞろい・ほかの文と食いちがう文」を足した。外部の AI レビュー役 への受け渡しは、オーナーのスマホを通さず、非公開リポジトリの枝 blind-review-2（2 つのファイルだけ。commit fa1eaa7）で行い、答えも同じ枝に置いてもらった（commit 28f7762）。答えは docs/ai_smell_review/round2/。

点数：経緯を知らない AI は 3 点が 0、外部の AI レビュー役 は 2 点が要旨と限界の段落だけで、ほかは 0〜1。

直した中身のまちがい：b が本文で定義されていなかった（1 回目の臭み抜きで消していた）→ III C で b = ∂ ln K/∂β と定義。「9 モデル中 5 つの区間が 1.25 を超えた」→ 追試の 7 モデルで 4 つ。「曲がりが行きすぎの大部分を説明」→ 0.87 は目標をわずかに下回るので「行きすぎは曲がりから来ていた」。限界の段落の「すべての前向きの試験は base モデル」→ 指示調整モデルの試験も前向きだったので「models」に。g_C^tilt が未定義 → 定義。ΔE と Δc の引き算の向きを付録 H に合わせて明記。スコアへの 1 ステップの寄与は (c_t − q_t)/T で、分散は q_t(1 − q_t)/T²。

直した言葉：「繰り返し」と「候補」の混同（要旨、I、III B、IV C、IV D、V）、did not decide → inconclusive、N_{>t} は「数」、c_t = 0 の場合、z は 1 トークンの logit、「not an option of」→「in」、「new news」、4 本ずつ生成したことを Setup に、停止の時刻の例、判定は区間で決まることを IV B と IV D の文に、「summed」は分子と分母を別々に和を取ってから比、表 I の「each test」は表の 2 つの試験、指示調整モデルは「一つは決まらず、一つは走らせられなかった」、限界の段落の重複（掛け算型、設計だけの予測、T_× のずれ）をまとめ、出どころの開示を別の段落にした。35 語を超える文を分けた。

残したもの：III A の停止の時刻の一文、要旨の長さ（17 文。要旨の数字は 1.04、0.8〜1.25、99% のまま）。
