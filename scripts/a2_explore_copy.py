"""探索：書き進めたあとに裾が細るのは「結晶」（写し歩）か「ガラス」（写していない歩でも自信が固まる）か。
docs/A2_EXPLORE_T0_FEATURES.md §5 の候補 C の「手」。A2-2・A2-3 の生のファイルを見直す探索で、確かめには使わない。

scripts/a2_2_collect.py を実行したのと同じ名前空間で実行する（load_docs・split_doc・step_stats・valid_mask・h2_of・
load_model・TEMPS・PROMPT・CONT を使う）。CPU だけで動く（モデルを読むのは検量線の確かめの Qwen2.5-0.5B だけ）。

写し歩：その歩のトークンで終わる 4 トークンの並びが、それより前（書き出しを含む）にすでに出てきている歩。
新しい歩：それ以外。

位置の帯ごとに、記事ごと（生成は 2 本をまとめる）に次の分け方をする。分け方は恒等式（記事ごとに足すと f に一致する）：
  f ＝ ガラスの分 ＋ 結晶の分
  ガラスの分 ＝ ℓ_gen（新しい歩の平均）− ℓ_tf（人の文の新しい歩の平均）
  結晶の分   ＝ c_gen ×（ℓ_gen（写し歩）− ℓ_gen（新しい歩））− c_hum ×（ℓ_tf（写し歩）− ℓ_tf（新しい歩））
  （c は写し歩の割合。写し歩がない記事ではその項は 0。）

読み方（データを開く前に決める。探索の目安で、判定ではない）：
- 見るところ：温度 0.7 と 0.8、位置の帯 128〜512、記事の平均で f ≤ −0.2 のとき。
- ガラスの割合 ＝ ガラスの分 ÷ f。0.5 以上なら「ガラス寄り」、0.25 以下なら「結晶寄り」、その間は「混ざり」。
- 検量線（人工の結晶）：人の続きの前半 256 トークンを、後半にそのまま写した文を Qwen2.5-0.5B に読ませる（記事 10 本）。
  元の人の文と比べて、帯 128〜512 で写し歩の割合が 0.4 以上、かつガラスの割合が 0.5 未満（結晶の分が半分以上）なら
  「手は結晶を結晶と言える」。通らなければ、この手で「結晶」「ガラス」という言葉は使わない。
  ガラスの人工の標準は作れていない（限界として書く）。
"""
import hashlib, json, math, time
import numpy as np

N_GRAM = 4
BINS = [(1, 8), (8, 32), (32, 128), (128, 512)]
LOOK_T = [0.7, 0.8]
CAL_DOCS, CAL_HALF = 10, 256
CAL_MODEL = ('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B')
BBC_CONDITIONS = [  # (名前, npz のパス（フォルダからの相対）, tokenizer の Hugging Face の ID)
    ('pythia-1b', 'A2_2/pythia-1b.npz', 'EleutherAI/pythia-1b'),
    ('qwen2.5-0.5b', 'A2_2/qwen2.5-0.5b.npz', 'Qwen/Qwen2.5-0.5B'),
    ('qwen2.5-0.5b-inst', 'A2_2/qwen2.5-0.5b-inst.npz', 'Qwen/Qwen2.5-0.5B-Instruct'),
    ('qwen2.5-1.5b', 'A2_2/qwen2.5-1.5b.npz', 'Qwen/Qwen2.5-1.5B'),
    ('olmo2-1b', 'A2_2/olmo2-1b.npz', 'allenai/OLMo-2-0425-1B'),
    ('smollm2-1.7b', 'A2_2/smollm2-1.7b.npz', 'HuggingFaceTB/SmolLM2-1.7B'),
    ('smollm2-1.7b-inst', 'A2_2/smollm2-1.7b-inst.npz', 'HuggingFaceTB/SmolLM2-1.7B-Instruct'),
    ('qwen2.5-7b@bbc', 'A2_3/qwen2.5-7b@bbc.npz', 'Qwen/Qwen2.5-7B'),
    ('olmo2-7b@bbc', 'A2_3/olmo2-7b@bbc.npz', 'allenai/OLMo-2-1124-7B'),
]


def copy_flags(hist, cont, n=N_GRAM):
    """hist：書き出しのトークン、cont：続き（長さ L）→ (L,) の bool。続きの各歩が写し歩か。"""
    seq = list(hist) + list(cont)
    P = len(hist)
    seen, flags = set(), np.zeros(len(cont), bool)
    for i in range(n - 1, len(seq)):
        g = tuple(seq[i - n + 1:i + 1])
        if i >= P:
            flags[i - P] = g in seen
        seen.add(g)
    return flags


def split_means(ell, flags):
    """ell・flags：同じ形の配列（まとめる歩）→（新しい歩の平均, 写し歩の平均, 写し歩の割合, 全体の平均）。"""
    ell = np.asarray(ell, np.float64).ravel(); flags = np.asarray(flags, bool).ravel()
    c = float(flags.mean()) if len(flags) else float('nan')
    new = float(ell[~flags].mean()) if (~flags).any() else float('nan')
    cp = float(ell[flags].mean()) if flags.any() else float('nan')
    return new, cp, c, float(ell.mean())


def decompose(gen_ell, gen_flags, tf_ell, tf_flags):
    """1 記事・1 温度・1 帯。gen_*：(S, 帯の長さ)、tf_*：(帯の長さ,)。→ dict（f・ガラス・結晶・写しの割合）。"""
    gn, gc, cg, gm = split_means(gen_ell, gen_flags)
    tn, tc, ch, tm = split_means(tf_ell, tf_flags)
    if not (np.isfinite(gn) and np.isfinite(tn)):
        return None  # 新しい歩がない（全部写し）。この記事・帯は外す
    glass = gn - tn
    crystal = (cg * (gc - gn) if cg > 0 else 0.0) - (ch * (tc - tn) if ch > 0 else 0.0)
    f = gm - tm
    assert abs(glass + crystal - f) < 1e-9, (glass, crystal, f)
    return dict(f=f, glass=glass, crystal=crystal, c_gen=cg, c_hum=ch)


def summarize(rows):
    """記事ごとの dict のリスト → 平均（分け方は平均で足し算が成り立つ）と、f の中央値。"""
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    out = {k: float(np.mean([r[k] for r in rows])) for k in ['f', 'glass', 'crystal', 'c_gen', 'c_hum']}
    out['f_median'] = float(np.median([r['f'] for r in rows]))
    out['n_docs'] = len(rows)
    return out


def reading(s):
    if s is None or s['f'] > -0.2:
        return '（f が −0.2 より上なので読まない）'
    share = s['glass'] / s['f']
    return 'ガラス寄り' if share >= 0.5 else '結晶寄り' if share <= 0.25 else '混ざり'


def analyze_condition(Z, prompts, humans, temps):
    """Z：npz、prompts・humans：使った記事（use の順）のトークン列。→ {T: {帯: 要約}}。"""
    toks, gen, tf = Z['toks'], Z['gen'][0], Z['tf'][0]          # (nT,S,nD,L)、(nT,S,nD,L)、(nD,nT,L)
    nT, S, nD, L = toks.shape
    hum_flags = [copy_flags(prompts[j], humans[j]) for j in range(nD)]
    res = {}
    for k, T in enumerate(temps):
        gflags = np.zeros((S, nD, L), bool)
        for s in range(S):
            for j in range(nD):
                gflags[s, j] = copy_flags(prompts[j], toks[k, s, j])
        res[T] = {}
        for a, b in BINS:
            rows = [decompose(gen[k, :, j, a:b], gflags[:, j, a:b], tf[j, k, a:b], hum_flags[j][a:b]) for j in range(nD)]
            res[T][f'{a}-{b}'] = summarize(rows)
    return res


def calibration(docs, device='cpu'):
    """人工の結晶：人の続きの前半 CAL_HALF トークンを後半に写した文。Qwen2.5-0.5B が読む。"""
    import torch
    name, hf_id = CAL_MODEL
    tok, model = load_model(hf_id, device)
    valid, _ = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    res = {T: {f'{a}-{b}': [] for a, b in BINS} for T in [0.7, 1.0]}
    used = 0
    for d in docs:
        p = split_doc(tok, d['text'])
        if p is None:
            continue
        prompt, human = p
        art = human[:CAL_HALF] + human[:CAL_HALF]
        ells = {}
        with torch.no_grad():
            for key, cont in [('art', art), ('hum', human)]:
                x = torch.tensor([prompt + cont], device=device)
                lg = model(x).logits[0, len(prompt) - 1: len(prompt) - 1 + CONT].float()
                ells[key] = {T: step_stats(lg, T, valid, H2)[0].double().cpu().numpy() for T in [0.7, 1.0]}
        fa, fh = copy_flags(prompt, art), copy_flags(prompt, human)
        for T in [0.7, 1.0]:
            for a, b in BINS:
                res[T][f'{a}-{b}'].append(decompose(ells['art'][T][None, a:b], fa[None, a:b], ells['hum'][T][a:b], fh[a:b]))
        used += 1
        if used >= CAL_DOCS:
            break
    out = {T: {k: summarize(v) for k, v in bins.items()} for T, bins in res.items()}
    s = out[1.0]['128-512']
    ok = bool(s is not None and s['c_gen'] >= 0.4 and s['f'] < 0 and s['glass'] / s['f'] < 0.5)
    return dict(summary=out, passed=ok, n_docs=used)


def run_all(base, docs, temps):
    """base：MyDrive/logit-thermo。→ dict。"""
    from transformers import AutoTokenizer
    out = {'calibration': calibration(docs), 'conditions': {}}
    for name, rel, hf_id in BBC_CONDITIONS:
        t0 = time.time()
        Z = np.load(f'{base}/{rel}')
        tok = AutoTokenizer.from_pretrained(hf_id)
        use = [int(i) for i in Z['use']]
        pairs = [split_doc(tok, docs[i]['text']) for i in use]
        assert all(p is not None for p in pairs)
        assert len(pairs[0][0]) == int(Z['prompt_len']), (name, len(pairs[0][0]), int(Z['prompt_len']))
        prompts = [p[0] for p in pairs]; humans = [p[1] for p in pairs]
        res = analyze_condition(Z, prompts, humans, temps)
        look = {str(T): reading(res[T]['128-512']) for T in LOOK_T}
        out['conditions'][name] = dict(by_T={str(T): v for T, v in res.items()}, reading=look, seconds=round(time.time() - t0, 1))
        print(name, look, f'{time.time() - t0:.0f} 秒')
    return out


def self_test():
    """torch も Hugging Face も使わない確かめ。"""
    # 写し歩の判定
    assert copy_flags([1, 2, 3], [4, 1, 2, 3, 4, 5]).tolist() == [False, False, False, False, True, False]
    assert copy_flags([], [7, 7, 7, 7, 7]).tolist() == [False, False, False, False, True]
    # 分け方が恒等式（足すと f）になる
    rng = np.random.default_rng(0)
    for _ in range(200):
        L = int(rng.integers(5, 40))
        ge = rng.normal(size=(2, L)); gf = rng.random((2, L)) < rng.random()
        te = rng.normal(size=L); tfl = rng.random(L) < rng.random()
        r = decompose(ge, gf, te, tfl)
        if r is not None:
            assert abs(r['glass'] + r['crystal'] - r['f']) < 1e-9
    # 作り物：写し歩だけ自信が強い（ℓ が低い）→ 結晶寄り。新しい歩でも ℓ が低い → ガラス寄り
    L = 384
    gf = np.zeros((2, L), bool); gf[:, 128:] = True
    tfl = np.zeros(L, bool)
    te = np.zeros(L)
    ge_crystal = np.where(gf, -3.0, 0.0)
    ge_glass = np.full((2, L), -1.0)
    rc = decompose(ge_crystal, gf, te, tfl); rg = decompose(ge_glass, np.zeros((2, L), bool), te, tfl)
    assert reading({**rc, 'f_median': rc['f'], 'n_docs': 1}) == '結晶寄り', rc
    assert reading({**rg, 'f_median': rg['f'], 'n_docs': 1}) == 'ガラス寄り', rg
    return 'OK'


if __name__ == '__main__':
    print(self_test())
