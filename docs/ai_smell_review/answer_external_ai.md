11s考えました

本文で最も機械生成らしく読めたのは P5 です。先行研究を一文ずつ紹介する間、各研究が今回の問いにどうつながるのかを読者が補う必要があります。逆に、最も人が書いた文章らしく読めたのは P28 です。予測した温度移動が大きすぎたため、著者が交点の求め方を変えて調べ直し、それでも説明できなかったモデルを名指ししています。違いを生むのは専門用語の量より、なぜ著者が次の一手を取ったのかが文から分かるかでした。

以下は添付された英文だけから受けた文体上の印象です。P41 の開示はコードと導出に関するものなので、本文の採点根拠には使っていません。

1. 段落ごとの評価

0 は人が書いた文章として自然に読める、1 は一部に機械生成らしい言い回しがある、2 は段落全体でその印象が目立つ、3 はその印象が強い、という尺度です。数式だけの段落の 0 は、文体を判断できる散文がほぼなく、加点する根拠がないという意味です。

段落	点	理由

P1	2	“a first-order prediction of its effect on how text switches between repetitive and ordinary stretches, computed from samples generated without the penalty, holds” は、何を予測し、何から計算し、何が成立したのかを一つの主語に詰め込みすぎています。終盤も結果が一文ずつ同じ重さで並びます。
P2	1	“have been designed partly to avoid it”, “have been proposed”, “have also been proposed” と、関連研究を紹介する受動文が繰り返されます。冒頭のループの描写は自然です。
P3	1	“A natural shortcut would treat” で仮説を紹介した直後、“Appendix H shows why this shortcut is not a reliable guide.” と結論を予告する運びがやや定型的です。
P4	1	“The same formula shows why the effect is small, by measuring how much” から “and locates where in a text a penalty acts” まで、一つの式を主語に論文の見どころをいくつも背負わせています。
P5	3	“are standard in simulation and reinforcement learning”, “are standard in statistical physics”, “has been steered”, “Our contribution is” と続きます。個々の先行研究の関係を説明する前に、分野紹介から貢献宣言までの型が見えてしまいます。
P6	0	マスクの対象、プロンプト長、生成長の順に、実験を再現する人が必要とする情報を素直に示しています。
P7	1	“Repeats define the states below, and candidates define the penalty. Keeping them separate makes the definition of the states independent of the penalty.” と、隣り合う二文で同じ区別を言い直すため、少し説明過多に聞こえます。ただし区別自体は重要です。
P8	0	“Each label therefore uses up to 15 later tokens.” のように、窓の仕様を読者にとって重要な時間上の意味に結びつけています。
P9	0	事前に決めた二峰性の基準を満たさなかったこと、長い反復ほど抜けにくいことを、解釈上の制限に直接結びつけています。
P10	0	系列の確率から一ステップの微分へ移る短い導出です。
P11	0	数式と記号の定義です。
P12	0	実際に用いた固定長と、同じ恒等式が使える停止条件の範囲を書き分けています。
P13	0	直前で導いた応答の数式です。
P14	0	式を適用する四つのカウントと、乗法ペナルティで置き換わる項を簡潔に指定しています。
P15	1	“in the language of statistics, and a tilted ensemble of trajectories in statistical physics” と、一つの操作に二分野の名前を並べる部分だけが、導出の途中で総説調に聞こえます。
P16	0	導出の結論を独立させた数式です。
P17	1	“This is a finite-horizon Doob transform” の後、 “In machine-learning terms, this is the difference” と再び分野を切り替えて命名するため、段落後半が用語の対応表のように読めます。
P18	0	逆温度とエネルギーを導入し、そのまま次の式に渡しています。
P19	0	温度スコアと補償項を定義する数式です。
P20	2	“The corresponding fraction for temperature compares” に続いて “with the two equilibrium predictions of Appendix H” と進みます。傾き、応答、比、温度にわたる和が、何を何と比較しているかを確かめる間もなく同じ調子で導入されます。
P21	0	近似式がエネルギーの選択に依存するという、後で実際に確かめられる違いを明示しています。
P22	0	“a narrow peak at exactly one and a shallow trough in between” は、図で何を見たのかが分かる具体的な描写です。
P23	0	九つのモデルで得た範囲を一文で報告しています。
P24	0	図注の “chosen as the first output with two to six switches” は、例をどう選んだかまで読者に伝えています。
P25	1	“the derivative of Sec. III A needs only unpenalized samples, it predicts, from unpenalized runs” と入力条件を言い直し、“large enough to matter” で意義を抽象的にまとめるところにやや作為を感じます。事前固定の時系列は具体的です。
P26	0	最初のテストが決着しなかった理由を、二つの区間が上限をわずかに超えたことまで含めて報告しています。
P27	0	予測した強度、その二倍、八倍で何が起きたかを順に追える結果報告です。
P28	0	補間を試した理由、その結果、なお説明できないモデルまで、著者の判断の順序が見えます。
P29	0	図注として、点・区間・判定帯の意味と、最初のテストだけ表示が異なる事情を伝えています。
P30	0	前半七モデルで中央値、後半二モデルで平均を使う違いまで、表を読むのに必要な情報を示しています。
P31	2	“the future repeats it makes likely”, “through those future repeats”, “the same expected future repeats” が続きます。0.7–1.5% という明瞭な結果の後で、同じ機序を三度説明する調子が目立ちます。
P32	1	“give the same picture”, “shorter texts keep more”, “exploratory records” はどれも曖昧です。本文の他の箇所に比べ、何が同じで、何の割合が大きいのかを読者に補わせています。
P33	0	表がなくても、二つの割合の定義と、温度側だけがエネルギーの選択に依存する点を読めます。
P34	0	応答の約 4/5 が候補のあるステップの約 1/10 に集まるという観測と、それが分散式だけでは保証されない理由を区別しています。
P35	1	“The penalty acts at both ends of a repetitive stretch.” と “A penalty thus both makes repetition harder to start and makes loops break more readily.” が段落を挟んで同じ役割を果たし、まとめが一回多く感じられます。
P36	0	実測シェアの区間と、予測との差についての不確実性を区別する、目的の明確な図注です。
P37	2	“Acting on whole texts” と “Acting step by step” の対句に加え、“because it changes what a penalty is” という大きな言い回しを重ね、すでに導出・実測した機序をもう一度演出しています。
P38	0	“We attribute the contrast to scale, without having tested it.” と、観測に対する自分たちの説明をどこまで確かめたか率直に示しています。
P39	0	エネルギーの取り方で約 45% の過大評価から符号違いまで起こる、という比較が具体的です。
P40	1	“In the same way” と “give the response to any such change and show where in the text it arises” は、適用範囲を広げて利点を再掲する定型的な結びに聞こえます。計算費用への言及には中身があります。
P41	2	“All prospective tests used”, “For instruction-tuned models”, “For the multiplicative penalty”, “Model revisions were not pinned” と、対象範囲、既出の結果、再現性、作業分担が同じ重さで続きます。個々の開示は具体的ですが、段落全体はチェックリストのように読めます。


2. とくに人が書いた文章らしく読める段落

- P28 が最も明瞭です。“Because ln K curves (Fig. 1c), we also located each crossing by interpolation after the tests.” には、図で気づいたこと、選び直した方法、それが事後の判断だったことが一文に収まっています。“GPT-2-XL most of all” と例外を名指しするので、中央値が改善した後も何が説明できずに残ったかが分かります。
- P25 と P26 は一続きで読むと自然です。“Fig. S1 itself rests on runs made before these documents.” は、図まで事前登録に従って作ったと読者が誤解する、まさにその箇所で時系列を訂正しています。続く “both of its intervals reached just above 1.25” からは、なぜ新しい記事で一度だけ追試したのかが伝わります。
- P34 の “The form of the score does not imply this proportionality” は、数式を見た読者が抱きそうな疑問にその場で答えています。分散の式から必ず言えることと、このデータでたまたま近かった二つの割合を、著者が自分で区別した跡があります。
- P38 の “We attribute the contrast to scale, without having tested it.” は、著者自身がどの説明を有力と考え、どこまで確かめたかを短く示しています。この限定は残したいです。

3. 削除できる文と、残す価値のある反復

次の四文は、丸ごと削っても読者に必要な情報が残ります。

箇所	削除候補の全文	情報が残る場所

P34 冒頭	“The derivative also says where in a text the penalty acts.”	節見出しがすでに問いを示し、直後の “Each step contributes” から具体的な答えが始まります。
P35 末尾	“A penalty thus both makes repetition harder to start and makes loops break more readily.”	同段落の “a lower entry rate and half from a higher exit rate” が両方向の効果を、寄与の大きさまで含めて示しています。
P37 中盤	“Step-wise normalization keeps the effect small, because it changes what a penalty is.”	直後の二文が、何が変わるのかと、応答の約 99% が除かれることの両方を説明しています。
P28 冒頭	“A user would care about where T_× ends up.”	直後に意図した移動量と実測値の比較があり、P25 は 0.10 という目標を、P41 はそれが利用者の狙う量であることを明示しています。応答比から温度移動へ話を切り替える合図として残す選択もあります。


内容が重なる文でも、置かれた場所によっては必要です。

箇所	重なる表現	判断

P1・P17・P31・P37	P1 の “the repeats it makes likely later”、P17 の “the future repeats it makes likely”、P31 の “those future repeats”、P37 の “the repeats it makes likely later”	要旨の直観、理論での導出、結果の解釈にはそれぞれ役割があります。先に減らしたいのは P31 内部の言い直しと、P37 での再説明です。
P25・P41	“g was estimated from the unpenalized runs of the same test” と “the tests estimated the derivative from their own unpenalized runs”	残す価値があります。 事前に固定した \(h^*\) と、各テスト自身の無罰則データから求めた導関数を混同すると、何を前向きに検証したのかが変わります。
P27・P41	“at the lower end of the values used in practice” と “below those common in practice”	実用上の強度に関する結果を再掲しています。ただし「実用値の下端」と「一般的な値より下」には差があるため、残すなら主張の強さをそろえたい箇所です。
P9・P41	“We use K as a summary of switching statistics, averaged over positions” と “K averages over a drift along the text”	これは解釈上の重要な制約です。P41 の同じ文にある “s_pen depends on the length of the text” は別の情報なので、文全体の削除は勧めません。


図表のキャプションにある定義や判定帯の再掲は、図表だけを読む読者に必要です。本文中の重複とは区別しました。

4. 初読でつまずく文

箇所	原文の該当部分	つまずく理由

P1	“its effect on how text switches between repetitive and ordinary stretches, computed from samples generated without the penalty, holds at penalties large enough to shift that switching”	“computed” が修飾するのは遠くにある “prediction” です。“holds” まで来て初めて主語が確定し、“shift that switching” に至っても、変わるのが遷移率なのか交差温度なのかをまだ選べません。
P15	“Each update is c_t − q_t plus ν_t, the change at step t in the expected number of chosen candidates after step t, a martingale increment.”	「時点 \(t\) で変わる期待」と「時点 \(t\) より後に数える候補」が入れ子です。最後の “a martingale increment” が更新全体と \(\nu_t\) のどちらを指すのかも、語順からは取りにくいです。
P17	“after both factors are renormalized, the second adds (h/T) Σ_t ν_t to the log-weight of a text”	“the second” が何かを二文前まで戻って探す必要があります。そのうえ、一ステップの二つの因子から、系列全体の対数重みの和へ一度に移ります。
P20	“Summed over temperatures, we define s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u), with the two equilibrium predictions of Appendix H; at a single temperature this equals b/b_tilt.”	直前までの \(b\) と \(b_{\rm tilt}\) から、説明のない \(g_C\) と \(g_C^{\rm tilt}\) に切り替わります。\(g_C\) の本文中の定義は次段落です。単一温度の比と複数温度での集約の関係を、使う前に示してほしいところです。
P25	“Only h* was fixed beforehand, and g was estimated from the unpenalized runs of the same test.”	少し前に “we fixed the procedure, the analysis code and the decision criterion” とあるため、“Only” が何にかかるかで一瞬止まります。意図は、積 \(h^*g(T)\) の二つの因子のうち、値を事前に固定したのは \(h^*\) だけ、ということだと読めます。
P26	“This range would have rejected the tilted response of Sec. IV C and, for six of the nine models, the equilibrium two-state model, whose ratio we computed after the tests (Appendix H).”	登録した判定はモデルをまとめた比についてなのに、途中で九モデル中六モデルという個別比較へ移ります。さらに事前の判定帯と事後計算が一文に入り、正式な判定と参考比較の境目が見えにくくなります。
P31	“Each fraction is a ratio of derivatives at h = 0.”	直前にはペナルティと温度の二つの割合があります。\(h=0\) はペナルティ側の説明として読めますが、P20 の温度側には \(\beta\) と、傾斜を与える \(\varepsilon=0\) が出てきます。“Each” の方が意図より広い可能性があり、これは文体だけでなく記号の適用範囲として確認したい点です。
P32	“Other conventions and the local form of s_temp give the same picture, and shorter texts keep more”	“keep more” の目的語がなく、短い文では何が増えるのかを読み返すことになります。ここで比較しているのは、正規化後に残る応答の割合です。
P34	“For these steps the two shares nevertheless agreed closely, with a median ratio of 0.96”	段落にはステップ数、分散、予測応答の三つのシェアがあります。“these steps” は中間の \(q_t\) 帯、“the two shares” は分散と応答のシェアだとたどれますが、0.96 の分子と分母の向きまでは文からすぐに分かりません。
P41	“Decisions concerned the median over models, and the bootstrap intervals resample articles, not models.”	後半は大切で明瞭です。一方、前半は全判定が中央値に基づくように読め、P30 の “median over the first seven models; mean over the last two” と合いません。七モデルの追試と二モデルの別試験を明示的に区別したい箇所です。


5. 自然さを最も改善する三つの変更

1. P1 では、検証した問いと結果を先に読めるようにする。 現状は “a first-order prediction” と動詞 “holds” の間に、予測対象と推定方法の両方が入っています。さらに、実現した \(\ln K\) の変化は予測できた一方、目標にした \(T_×\) の移動はモデルによって外れた、という大切な違いが要旨の前後に分かれています。この二つの成果を読み分けられる順に整理するのが、最も効果の大きい変更です。必要な情報の削除はありません。 同じテストの無罰則ランから導関数を求めたことも残します。

2. P5 を研究分野の列挙から、今回の問いに至る説明に変える。 読者が知りたいのは、既知の尤度比公式で何が計算でき、局所正規化と系列全体の正規化の違いについて、今回初めて何を測ったのかです。その関係を軸に先行研究を配置し、Doob 変換の分野間の対応づけは、実際にその変換を使う P17 にまとめられます。必要な情報の削除はありません。 先行研究の帰属と、本論文が標準公式に加えた実証上の貢献を残せます。

3. 結果と考察の各段落で、一度説明したことを締めの文でもう一度言う箇所を減らす。 とくに P31・P35・P37 です。P35 なら、入口と出口が約半分ずつ寄与した実測値に続いて、「予測は出口の寄与をやや大きく見積もり、九モデル中二つで差の区間がゼロを含まなかった」という箇所で終われます。そこで終える方が、予測がどこまで当たったのかが印象に残ります。削るのは既出の言い直しだけです。 P37 の設計記事から新記事への探索的予測、P41 の記事単位の再標本化や AI 利用の開示など、その場所にしかない情報は残します。