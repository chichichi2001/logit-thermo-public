"""探索の続き（2026-10-28 夜）。CPU だけ。確かめには使わない。

scripts/a2_2_collect.py と scripts/a2_explore_copy.py を実行したのと同じ名前空間で実行する
（load_docs・split_doc・copy_flags・decompose・summarize・reading・BINS・LOOK_T・BBC_CONDITIONS を使う）。

part1(base)：探索 C の手を確かめ直す。
  (a) 写し歩を決める並びの長さを 3 と 8 に変えて、ガラスの割合を出し直す（ニュースの 9 条件。検量線はしない）。
      読み方は探索 C と同じ（温度 0.7・0.8、128〜512 トークン目、f ≤ −0.2 のときだけ。ガラスの割合 0.5 以上で
      「ガラス寄り」、0.25 以下で「結晶寄り」）。4 のときの読み方と、どれだけ食い違うかを数える。
  (b) A2-3 の 6 条件で、位置の帯（1〜8・8〜32・32〜128・128〜512）ごとの f（記事の中央値）と、帯ごとに f が 0 を
      横切る温度（ε なしの単純な横切り）を出す（探索 §2 と同じ計算）。
part2()：候補 A（生成の長さ）の下見。2025-01〜2025-06 の文章で、書き出し 64 ＋ 続き 2,048 トークン（Qwen2.5 の
  トークナイザーで数える）以上ある文書が何本あるかを、データセット・月ごとに数える。中身は表示しない・保存しない。
  データセットの読み込みに失敗したら、その理由を記録して次へ進む。
"""
import json, math, time
import numpy as np

LONG_MONTHS = ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06']
LONG_SOURCES = ['RealTimeData/bbc_news_alltime', 'RealTimeData/wikitext_alltime', 'RealTimeData/arxiv_alltime']
LONG_NEED = 64 + 2048
LONG_TOKENIZER = 'Qwen/Qwen2.5-1.5B'
A23 = ['qwen2.5-1.5b@bbc-a100', 'qwen2.5-7b@bbc', 'olmo2-7b@bbc', 'qwen2.5-1.5b@code', 'smollm2-1.7b@code', 'pythia-1b@code']


def analyze_condition_n(Z, prompts, humans, temps, n):
    """a2_explore_copy.analyze_condition と同じ計算を、写しの並びの長さ n で。"""
    toks, gen, tf = Z['toks'], Z['gen'][0], Z['tf'][0]
    nT, S, nD, L = toks.shape
    hum = [copy_flags(prompts[j], humans[j], n) for j in range(nD)]
    res = {}
    for k, T in enumerate(temps):
        gf = np.zeros((S, nD, L), bool)
        for s in range(S):
            for j in range(nD):
                gf[s, j] = copy_flags(prompts[j], toks[k, s, j], n)
        res[T] = {f'{a}-{b}': summarize([decompose(gen[k, :, j, a:b], gf[:, j, a:b], tf[j, k, a:b], hum[j][a:b])
                                         for j in range(nD)]) for a, b in BINS}
    return res


def simple_cross(y, temps):
    lt = np.log(np.asarray(temps)); y = np.asarray(y, float)
    for i in range(len(y) - 1):
        if y[i] < 0 <= y[i + 1]:
            return float(math.exp(lt[i] + (0 - y[i]) / (y[i + 1] - y[i]) * (lt[i + 1] - lt[i])))
    return float('nan')


def part1(base, docs):
    from transformers import AutoTokenizer
    out = {'copy_n': {}, 'a23_positions': {}}
    for name, rel, hf_id in BBC_CONDITIONS:
        t0 = time.time()
        Zr = np.load(f'{base}/{rel}')
        Z = {k: Zr[k] for k in ['toks', 'gen', 'tf', 'use', 'temps', 'prompt_len']}
        temps = [float(t) for t in Z['temps']]
        tok = AutoTokenizer.from_pretrained(hf_id)
        pairs = [split_doc(tok, docs[int(i)]['text']) for i in Z['use']]
        assert len(pairs[0][0]) == int(Z['prompt_len'])
        prompts = [p[0] for p in pairs]; humans = [p[1] for p in pairs]
        row = {}
        for n in [3, 8]:
            res = analyze_condition_n(Z, prompts, humans, temps, n)
            row[f'n{n}'] = dict(by_T={str(T): res[T] for T in temps}, reading={str(T): reading(res[T]['128-512']) for T in LOOK_T})
        out['copy_n'][name] = row
        print(name, {k: v['reading'] for k, v in row.items()}, f'{time.time() - t0:.0f} 秒')
    for name in A23:
        Zr = np.load(f'{base}/A2_3/{name}.npz')
        temps = [float(t) for t in Zr['temps']]
        g = Zr['gen'][0].astype(np.float64).mean(1).transpose(1, 0, 2) - Zr['tf'][0].astype(np.float64)   # (nD, nT, L)
        bins = {f'{a}-{b}': np.median(g[:, :, a:b].mean(-1), 0) for a, b in BINS}
        out['a23_positions'][name] = {k: dict(f=v.tolist(), cross=simple_cross(v, temps)) for k, v in bins.items()}
    return out


def part2():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LONG_TOKENIZER)
    out = {}
    for src in LONG_SOURCES:
        out[src] = {}
        for m in LONG_MONTHS:
            t0 = time.time()
            try:
                ds = load_dataset(src, m, split='train')
                cols = [c for c in ds.column_names if getattr(ds.features[c], 'dtype', None) == 'string']
                field = max(cols, key=lambda c: np.mean([len(x or '') for x in ds[c][:50]]))   # いちばん長い文字の列を本文とみなす
                texts = ds[field]
                chars = np.array([len(x or '') for x in texts])
                cand = [x for x in texts if x and len(x) >= 6000][:2000]                           # 2,112 トークンの見込みに届くものだけ、最大 2,000 本数える
                ntok = np.array([len(tok(x, add_special_tokens=False).input_ids) for x in cand]) if cand else np.array([])
                out[src][m] = dict(field=field, n_docs=int(len(texts)), n_ge_6000_chars=int((chars >= 6000).sum()), capped=bool((chars >= 6000).sum() > 2000), chars_median=float(np.median(chars)) if len(chars) else None,
                                   n_ge_need=int((ntok >= LONG_NEED).sum()), n_ge_1088=int((ntok >= 64 + 1024).sum()),
                                   ntok_quartiles=[float(np.percentile(ntok, q)) for q in (25, 50, 75)] if len(ntok) else None,
                                   seconds=round(time.time() - t0, 1))
            except Exception as e:
                out[src][m] = dict(error=f'{type(e).__name__}: {str(e)[:300]}')
            print(src, m, {k: v for k, v in out[src][m].items() if k != 'ntok_quartiles'})
    return out
