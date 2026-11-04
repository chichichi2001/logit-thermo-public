"""A2-6F（ゲート docs/A2_6F_GATE.md）：しらふ同士の床と、家族の差の天井。データを開く前に固定する。CPU。
scripts/a2_2_collect.py（split_doc）・scripts/a2_2_analyze.py（per_doc_feedback・feedback_stats）・
scripts/a2_explore_copy.py（copy_flags・decompose）・scripts/a2_5_analyze.py（doc_parts・G_at）を実行したのと同じ名前空間で実行する。

床は 3 種類（どれも GPU を使わず、手元のデータから出す）：
- 組の床：同じモデル・同じ記事・同じコードで、生成の種だけ違う 2 回の差（A2-2 の種 20261027 と、A2-5/A2-6 の種 20261102）。4 組。
- 種の半分の床：1 回の中の 2 本のサンプル（1 本目と 2 本目）を分けて、それぞれで測った差。A2-2・A2-5・A2-6 の全部の回。
- 記事の半分の床：40 本の記事を、使った順で偶数番目と奇数番目の 20 本ずつに分けて測った差（G07 は 30 本以上ないと読まないので出さない）。
半分どうしの差は、1 回まるごとどうしの差よりおよそ √2 倍大きくなる（半分は雑音の分散が 2 倍）ので、√2 で割って比べる。
    analyze6f(logit-thermo のフォルダ, docs) -> dict
"""
import json, math
import numpy as np

F6_PAIRS = [  # (名前, 古い種のフォルダ, 新しい種のフォルダ, トークナイザー)
    ('pythia-1b', 'A2_2', 'A2_5', 'EleutherAI/pythia-1b'),
    ('qwen2.5-0.5b', 'A2_2', 'A2_6', 'Qwen/Qwen2.5-0.5B'),
    ('qwen2.5-1.5b', 'A2_2', 'A2_6', 'Qwen/Qwen2.5-1.5B'),
    ('smollm2-1.7b', 'A2_2', 'A2_6', 'HuggingFaceTB/SmolLM2-1.7B'),
]
F6_RUNS = {  # 半分の床に使う全部の回：フォルダ → [(名前, トークナイザー)]
    'A2_2': [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B'), ('qwen2.5-0.5b-inst', 'Qwen/Qwen2.5-0.5B-Instruct'),
             ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'),
             ('smollm2-1.7b-inst', 'HuggingFaceTB/SmolLM2-1.7B-Instruct')],
    'A2_5': [(n, 'EleutherAI/pythia-1b') for n in ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
                                                   'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000']],
    'A2_6': [('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('qwen2.5-3b', 'Qwen/Qwen2.5-3B'),
             ('smollm2-135m', 'HuggingFaceTB/SmolLM2-135M'), ('smollm2-360m', 'HuggingFaceTB/SmolLM2-360M'),
             ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'), ('qwen3-0.6b', 'Qwen/Qwen3-0.6B-Base'), ('qwen3-1.7b', 'Qwen/Qwen3-1.7B-Base'),
             ('qwen3-4b', 'Qwen/Qwen3-4B-Base')],
}
F6_KEYS = ['T0_logratio', 'F07', 's', 'G07']
F6_STEPS = ['pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000', 'pythia-1b']
F6_EQ = 1.05            # A2-6 の Q9「動かない」の線（家族の中の T_0 の最大 ÷ 最小）
F6_CEIL_X = 3.0         # 天井が床の何倍あれば「家族の差は床よりずっと大きい」と言うか


def G_one_temp(Z, prompts, humans, k):
    """A2-5 の doc_parts と G_at を、温度 k だけについて同じ手順で計算する（速くするため）。"""
    toks, gen, tf = Z['toks'], Z['gen'][0], Z['tf'][0]
    nT, S, nD, L = toks.shape
    a, b = A5_BAND
    g, f = [], []
    for j in range(nD):
        hum = copy_flags(prompts[j], humans[j])
        gfl = np.stack([copy_flags(prompts[j], toks[k, s, j]) for s in range(S)])
        r = decompose(gen[k, :, j, a:b], gfl[:, a:b], tf[j, k, a:b], hum[a:b])
        if r is not None:
            g.append(r['glass']); f.append(r['f'])
    n = len(f)
    if n == 0:
        return float('nan'), float('nan'), 0
    mg, mf = float(np.mean(g)), float(np.mean(f))
    return (mg / mf if (n >= A5_MIN_DOCS and mf <= A5_FMAX) else float('nan')), mf, n


def sub(Z, samples=None, docs=None):
    """サンプルや記事を選んだ Z（tf：(4, nD, nT, L)、gen：(4, nT, S, nD, L)、toks：(nT, S, nD, L)）。"""
    Z2 = dict(Z)
    if samples is not None:
        Z2['gen'] = Z2['gen'][:, :, samples]; Z2['toks'] = Z2['toks'][:, samples]
    if docs is not None:
        Z2['tf'] = Z2['tf'][:, docs]; Z2['gen'] = Z2['gen'][:, :, :, docs]; Z2['toks'] = Z2['toks'][:, :, docs]
    return Z2


def measures(Z, prompts, humans, with_G=True):
    temps = [float(t) for t in Z['temps']]
    k07 = temps.index(0.7)
    fd = per_doc_feedback(Z['tf'][0], Z['gen'][0])
    fs = feedback_stats(fd, temps, np.random.default_rng(0))
    F = np.median(fd, 0)
    G07, f07, n07 = G_one_temp(Z, prompts, humans, k07) if with_G else (float('nan'), float('nan'), 0)
    return dict(T0=fs['T0'], F07=float(F[k07]), s=float((F[temps.index(0.9)] - F[temps.index(0.8)]) / 0.1),
                G07=G07, f07_band=f07, n07=n07)


def diffs(a, b):
    lr = abs(math.log(a['T0'] / b['T0'])) if (np.isfinite(a['T0']) and np.isfinite(b['T0'])) else float('nan')
    return dict(T0_logratio=lr, F07=abs(a['F07'] - b['F07']), s=abs(a['s'] - b['s']), G07=abs(a['G07'] - b['G07']))


def p95(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.percentile(v, 95)) if len(v) else float('nan')


def floor_from_vals(vals, a25, a26, halves_seed=None, halves_doc=None):
    """vals：{名前: (古い種の measures, 新しい種の measures)}。a25・a26：A2-5・A2-6 の要約（dict）。
    halves_seed・halves_doc：{回の名前: (半分 1 の measures, 半分 2 の measures)}。"""
    res = {'pairs': {}}
    d = {k: [] for k in F6_KEYS}
    for n, (a, b) in vals.items():
        row = dict(old=a, new=b, **diffs(a, b))
        res['pairs'][n] = row
        for k in d:
            d[k].append(row[k])
    res['pair_floor_max'] = {k: float(np.nanmax(v)) for k, v in d.items()}
    res['pair_floor_median'] = {k: float(np.nanmedian(v)) for k, v in d.items()}
    judge = dict(res['pair_floor_max'])
    for tag, hv in (('seed_half', halves_seed or {}), ('doc_half', halves_doc or {})):
        rows = {n: diffs(a, b) for n, (a, b) in hv.items()}
        res[tag] = rows
        res[tag + '_p95'] = {k: p95([r[k] for r in rows.values()]) for k in F6_KEYS}
        res[tag + '_max'] = {k: (float(np.nanmax([r[k] for r in rows.values()])) if rows and np.isfinite([r[k] for r in rows.values()]).any() else float('nan')) for k in F6_KEYS}
        res[tag + '_n_finite'] = {k: int(np.isfinite([r[k] for r in rows.values()]).sum()) for k in F6_KEYS}
        for k in F6_KEYS:
            v = res[tag + '_p95'][k] / math.sqrt(2)
            if np.isfinite(v):
                judge[k] = max(judge[k], v) if np.isfinite(judge[k]) else v
    res['floor_max'] = judge          # 判定に使う床：組の床のいちばん大きい値と、半分の床の 95% 点 ÷ √2 のうち大きいほう
    # F1：T_0 の床は、A2-6 の「動かない」の線（5%）の内側か
    fr = math.exp(res['floor_max']['T0_logratio'])
    res['F1'] = '床は 5% の線の内側' if fr < F6_EQ else '床が 5% の線に届く'
    res['F1_detail'] = dict(floor_ratio=fr, line=F6_EQ)
    # F2：家族の差の天井（A2-6 の家族ごとの T_0 の幾何平均の、いちばん離れた 2 家族）は床の何倍か
    gm = {g: v['geo_mean'] for g, v in (a26 or {}).get('fam_T0', {}).items()}
    if len(gm) >= 2 and res['floor_max']['T0_logratio'] > 0:
        ceil = math.log(max(gm.values()) / min(gm.values()))
        x = ceil / res['floor_max']['T0_logratio']
        res['F2'] = '天井は床の 3 倍以上' if x >= F6_CEIL_X else '天井は床の 3 倍より小さい'
        res['F2_detail'] = dict(ceiling_logratio=ceil, times_floor=x, fam_geo_mean=gm)
    else:
        res['F2'] = '読めない'; res['F2_detail'] = dict(fam_geo_mean=gm)
    # F3：A2-5 の学びの軸で、隣どうしの G07 の差は床（G07 の最大）を超えるか
    G = [(a25 or {}).get('models', {}).get(n, {}).get('G07', float('nan')) for n in F6_STEPS]
    steps = []
    for i in range(len(G) - 1):
        diff = G[i + 1] - G[i]
        w = '床の上' if (np.isfinite(diff) and abs(diff) > res['floor_max']['G07']) else ('床の中' if np.isfinite(diff) else '読めない')
        steps.append(dict(frm=F6_STEPS[i], to=F6_STEPS[i + 1], diff=float(diff), word=w))
    res['F3_steps'] = steps
    res['F3'] = sum(s['word'] == '床の上' for s in steps)
    first, last = G[0], G[-1]
    res['F3_whole'] = dict(diff=float(last - first), word='床の上' if abs(last - first) > res['floor_max']['G07'] else '床の中')
    return res


def analyze6f(base, docs):
    from transformers import AutoTokenizer
    vals = {}
    for n, da, db, hf in F6_PAIRS:
        tok = AutoTokenizer.from_pretrained(hf)
        pairs = [split_doc(tok, d['text']) for d in docs]
        Zs = [(lambda z: {k: z[k] for k in z.files})(np.load(f'{base}/{dd}/{n}.npz')) for dd in (da, db)]
        uses = [[int(u) for u in Z['use']] for Z in Zs]
        assert uses[0] == uses[1], ('使った記事が 2 回で違います', n)
        prompts = [pairs[i][0] for i in uses[0]]; humans = [pairs[i][1] for i in uses[0]]
        vals[n] = tuple(measures(Z, prompts, humans) for Z in Zs)
    a25 = json.load(open(f'{base}/A2_5/a2_5_summary.json'))
    try:
        a26 = json.load(open(f'{base}/A2_6/a2_6_summary.json'))
    except FileNotFoundError:
        a26 = None
    hs, hd = {}, {}
    for folder, runs in F6_RUNS.items():
        for n, hf in runs:
            tok = AutoTokenizer.from_pretrained(hf)
            pairs = [split_doc(tok, d['text']) for d in docs]
            Z = (lambda z: {k: z[k] for k in z.files})(np.load(f'{base}/{folder}/{n}.npz'))
            use = [int(u) for u in Z['use']]
            prompts = [pairs[i][0] for i in use]; humans = [pairs[i][1] for i in use]
            assert Z['gen'].shape[2] == 2, ('サンプルが 2 本ではありません', folder, n)
            hs[f'{folder}/{n}'] = tuple(measures(sub(Z, samples=[s_]), prompts, humans) for s_ in (0, 1))
            ev, od = list(range(0, len(use), 2)), list(range(1, len(use), 2))
            hd[f'{folder}/{n}'] = tuple(measures(sub(Z, docs=ix), [prompts[i] for i in ix], [humans[i] for i in ix], with_G=False)
                                        for ix in (ev, od))
    return floor_from_vals(vals, a25, a26, hs, hd)


def self_test_a26f():
    mk = lambda T0, F07, s, G: dict(T0=T0, F07=F07, s=s, G07=G, f07_band=-1.0, n07=40)
    vals = {'pythia-1b': (mk(0.852, -0.88, 3.9, 0.29), mk(0.841, -0.90, 4.1, 0.33)),
            'qwen2.5-0.5b': (mk(0.858, -2.0, 6.4, 0.40), mk(0.866, -2.1, 6.0, 0.41))}
    a25 = {'models': {'pythia-1b@step8000': {'G07': 0.20}, 'pythia-1b@step24000': {'G07': 0.27},
                      'pythia-1b@step64000': {'G07': 0.30}, 'pythia-1b': {'G07': 0.33}}}
    a26 = {'fam_T0': {'A': {'geo_mean': 0.80}, 'B': {'geo_mean': 0.90}}}
    r = floor_from_vals(vals, a25, a26)
    assert abs(r['floor_max']['G07'] - 0.04) < 1e-9, r['floor_max']
    # 半分の床が組の床より大きければ、判定にはそちら（95% 点 ÷ √2）を使う
    hs = {f'r{i}': (mk(0.85, -1.0, 4.0, 0.30), mk(0.85, -1.0, 4.0, 0.30 + 0.01 * i)) for i in range(20)}
    r2 = floor_from_vals(vals, a25, a26, halves_seed=hs)
    assert abs(r2['floor_max']['G07'] - max(0.04, np.percentile([0.01 * i for i in range(20)], 95) / math.sqrt(2))) < 1e-9, r2['floor_max']
    # 交点が出ない半分は外して数える
    hd = {'x': (mk(float('nan'), -1.0, 4.0, float('nan')), mk(0.85, -1.0, 4.0, float('nan')))}
    r3 = floor_from_vals(vals, a25, a26, halves_doc=hd)
    assert r3['doc_half_n_finite']['T0_logratio'] == 0 and r3['floor_max']['T0_logratio'] == r3['pair_floor_max']['T0_logratio']
    assert r['F1'] == '床は 5% の線の内側', r['F1_detail']
    assert r['F2'] == '天井は床の 3 倍以上', r['F2_detail']
    assert r['F3'] == 1 and r['F3_steps'][0]['word'] == '床の上' and r['F3_steps'][1]['word'] == '床の中', r['F3_steps']
    assert r['F3_whole']['word'] == '床の上'
    assert floor_from_vals(vals, a25, None)['F2'] == '読めない'
    return 'OK'
