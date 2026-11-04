"""探索 8.40 の分析：PROJECT_G の下見（罰の置き場所）。データを開く前に書いた（2026-11-01 06:3x JST、時計で確かめた）。CPU、判定しない。
scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases）を実行したのと同じ名前空間で実行する。

量（モデル × 温度 × 罰の組）：
  くり返し rep ＝ 写しの印（4 つ組がそれより前に出ていた）の割合、歩 16〜495、1 本ごとに出して平均（ものさしに頼らない量）。
  副作用 cost ＝ 1 歩あたりの KL の平均（罰をかけない歩は 0、512 歩すべてで割る。単位 nat ÷ 歩）。
  記録：結晶の割合（窓のものさし）、罰をかけた歩の割合、選んだトークンの U の平均（温度 1・罰なし）、写しの印がない歩の U の平均。
効き目の比べ（同じ副作用で）：
  「いつも」の曲線 ＝ (0, rep(罰なし)) と 3 つの強さの (cost, rep) を cost の順に並べ、直線でつないだもの（外挿しない）。
  比べる腕 m（芽を摘む・溶かす）の強さ h の点 (cost_m, rep_m) について、D ＝ 「いつも」の曲線の cost_m での rep − rep_m。
  D ＞ 0 なら、同じ副作用で m のほうがくり返しを多く減らす。cost_m が「いつも」の範囲を超えたら出ない（nan）。
  効率の比 E（記録）＝ (rep(罰なし) − rep_m) ÷ (rep(罰なし) − 「いつも」の曲線の cost_m での rep)。
  2 つの温度の D の平均を、そのモデルの D とする。
区間：記事の選び直し 1,000 回（種 20261145。モデルの中で、温度・罰の組すべてに同じ選び方）。出ない回が 1 割を超えたら「はっきりしない」。

問い（モデルごと、強さ h ＝ −1.0）：
  PQ1（芽を摘む vs いつも）：「芽を摘むほうが効く」：D の区間の下 ＞ 0。「いつもの罰のほうが効く」：区間の上 ＜ 0。それ以外「はっきりしない」。
  PQ2（溶かす vs いつも）：同じ決め方で「溶かすほうが効く」「いつもの罰のほうが効く」「はっきりしない」。
  全体：3 つのモデルが同じ言葉ならその言葉、そうでなければ「モデルで分かれる・はっきりしない」。
"""
import numpy as np

PA_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
PA_TEMPS = [0.5, 0.6]
PA_H = [-0.5, -1.0, -2.0]
PA_MAIN_H = -1.0
PA_BOOT, PA_SEED = 1000, 20261145
PA_FAIL = 0.10


def pa_tag(mode, h):
    return 'none' if mode == 'none' else f'{mode}_h{int(round(-h * 10)):02d}'


def pa_configs():
    return [('none', 0.0)] + [(m, h) for m in ('all', 'grow', 'melt') for h in PA_H]


def seq_stats(Z):
    """Z：1 つの組の npz → 記事ごと (nT, nD, 7) の和：[rep の和, 本数, cost の和, 結晶の歩, 相がつく歩, 罰の歩, 歩]、と U の記録 (nT, nD, 4)：[U 和, 歩, 写しの印のない歩の U 和, その歩]。"""
    toks, kl, app, U = Z['toks'], Z['kl'].astype(np.float64), Z['app'], Z['U'].astype(np.float64)
    nT, S, nD, L = toks.shape
    out = np.zeros((nT, nD, 7)); uo = np.zeros((nT, nD, 4))
    for k in range(nT):
        for j in range(nD):
            for s in range(S):
                c = copy_marks(toks[k, s, j])
                st, _ = phases(c)
                out[k, j] += [c[16:496].mean(), 1, kl[k, s, j].mean(), st.sum(), len(st), app[k, s, j].sum(), L]
                nc = c == 0
                uo[k, j] += [U[k, s, j].sum(), L, U[k, s, j][nc].sum(), nc.sum()]
    return out, uo


def curve_interp(xs, ys, x):
    """xs・ys（点の並び）を xs の順に直線でつなぎ、x での値。範囲の外は nan。"""
    o = np.argsort(xs); xs, ys = np.asarray(xs)[o], np.asarray(ys)[o]
    if not np.isfinite(x) or x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        return float('nan')
    return float(np.interp(x, xs, ys))


def point_model(A):
    """A：{tag: (nT, 7)}（記事を足したもの）→ 温度ごとの rep・cost と、D・E。"""
    res = {}
    rep = {t: A[t][:, 0] / A[t][:, 1] for t in A}
    cost = {t: A[t][:, 2] / A[t][:, 1] for t in A}
    res['rep'] = {t: v.tolist() for t, v in rep.items()}
    res['cost'] = {t: v.tolist() for t, v in cost.items()}
    res['cry'] = {t: (A[t][:, 3] / A[t][:, 4]).tolist() for t in A}
    res['app_frac'] = {t: (A[t][:, 5] / A[t][:, 6]).tolist() for t in A}
    for mode in ('grow', 'melt'):
        for h in PA_H:
            tg = pa_tag(mode, h)
            Ds, Es = [], []
            for k in range(len(PA_TEMPS)):
                xs = [0.0] + [cost[pa_tag('all', hh)][k] for hh in PA_H]
                ys = [rep['none'][k]] + [rep[pa_tag('all', hh)][k] for hh in PA_H]
                ra = curve_interp(xs, ys, cost[tg][k])
                Ds.append(ra - rep[tg][k])
                den = rep['none'][k] - ra
                Es.append((rep['none'][k] - rep[tg][k]) / den if np.isfinite(den) and den > 1e-6 else float('nan'))
            res[f'D_{tg}_T'] = Ds
            res[f'D_{tg}'] = float(np.mean(Ds)) if np.all(np.isfinite(Ds)) else float('nan')
            res[f'E_{tg}_T'] = Es
    return res


def word_D(ci, fail, pos):
    if fail > PA_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if ci[0] > 0:
        return pos
    if ci[1] < 0:
        return 'いつもの罰のほうが効く'
    return 'はっきりしない'


def analyze_model(S, seed=PA_SEED, boot=PA_BOOT):
    """S：{tag: (nT, nD, 7)} → 点・区間・言葉。"""
    pt = point_model({t: v.sum(1) for t, v in S.items()})
    nD = S['none'].shape[1]
    rng = np.random.default_rng(seed)
    keys = [f'D_{pa_tag(m, h)}' for m in ('grow', 'melt') for h in PA_H]
    bs = {k: [] for k in keys}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        b = point_model({t: np.tensordot(v, w, axes=([1], [0])) for t, v in S.items()})
        for k in keys:
            bs[k].append(b[k])
    ci, fail = {}, {}
    for k in keys:
        v = np.array(bs[k], float)
        fail[k] = float(np.mean(~np.isfinite(v)))
        ci[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
    g, m = f'D_{pa_tag("grow", PA_MAIN_H)}', f'D_{pa_tag("melt", PA_MAIN_H)}'
    words = dict(PQ1=word_D(ci[g], fail[g], '芽を摘むほうが効く'), PQ2=word_D(ci[m], fail[m], '溶かすほうが効く'))
    return dict(point=pt, ci=ci, fail=fail, words=words)


def analyze_pork(base, models=PA_MODELS, seed=PA_SEED, boot=PA_BOOT):
    res = {'models': {}}
    for name in models:
        S, Ur, uses = {}, {}, []
        for mode, h in pa_configs():
            tg = pa_tag(mode, h)
            Z = np.load(f'{base}/{name}@{tg}.npz')
            assert [round(float(t), 4) for t in Z['temps']] == PA_TEMPS and str(Z['mode']) == mode and abs(float(Z['h']) - h) < 1e-9, (name, tg)
            if mode == 'none':
                assert not Z['app'].any() and float(np.abs(Z['kl']).max()) == 0.0, (name, '罰なしで罰がかかっている')
            uses.append(np.asarray(Z['use']).tolist())
            S[tg], Ur[tg] = seq_stats(Z)
        assert all(u == uses[0] for u in uses), name
        r = analyze_model(S, seed, boot)
        r['U_record'] = {t: dict(U_mean=(v[..., 0].sum(1) / v[..., 1].sum(1)).tolist(), U_nocopy=(v[..., 2].sum(1) / v[..., 3].sum(1)).tolist()) for t, v in Ur.items()}
        res['models'][name] = r
    w1 = [res['models'][n]['words']['PQ1'] for n in models]; w2 = [res['models'][n]['words']['PQ2'] for n in models]
    res['verdict'] = dict(PQ1=(w1[0] if len(set(w1)) == 1 and w1[0] != 'はっきりしない' else 'モデルで分かれる・はっきりしない', dict(zip(models, w1))),
                          PQ2=(w2[0] if len(set(w2)) == 1 and w2[0] != 'はっきりしない' else 'モデルで分かれる・はっきりしない', dict(zip(models, w2))))
    return res


def _synth_S(nD=40, eff_grow=2.0, eff_melt=0.5, rng=None):
    """作り物の (nT, nD, 7)：「いつも」は rep ＝ 0.4 − 0.10 × cost/0.01、芽を摘むは効率 eff_grow 倍（同じ cost で eff 倍減る）、溶かすは eff_melt 倍。
    cost は「いつも」h ＝ −0.5・−1・−2 で 0.004・0.010・0.025、芽を摘む・溶かすはその 0.4 倍。記事ごとのぶれ ±0.02。"""
    rng = np.random.default_rng(0) if rng is None else rng
    S = {}
    base = 0.4
    for mode, h in pa_configs():
        tg = pa_tag(mode, h)
        c_all = {-0.5: 0.004, -1.0: 0.010, -2.0: 0.025}
        cost = 0.0 if mode == 'none' else (c_all[h] if mode == 'all' else 0.4 * c_all[h])
        eff = {'none': 0, 'all': 1.0, 'grow': eff_grow, 'melt': eff_melt}[mode]
        rep = base - eff * 10 * cost
        arr = np.zeros((2, nD, 7))
        for k in range(2):
            noise = rng.normal(0, 0.02, nD)
            arr[k, :, 0] = (rep + noise) * 4; arr[k, :, 1] = 4; arr[k, :, 2] = cost * 4
            arr[k, :, 3:] = 1
        S[tg] = arr
    return S


def self_test_pork_an():
    # 1) 直線つなぎ
    assert abs(curve_interp([0, 1, 2], [1.0, 0.5, 0.0], 1.5) - 0.25) < 1e-12 and np.isnan(curve_interp([0, 1], [1, 0], 1.5))
    # 2) 芽を摘むが 2 倍効く・溶かすが半分 → PQ1「芽を摘むほうが効く」、PQ2「いつもの罰のほうが効く」。D は解析解と合う
    r = analyze_model(_synth_S(), boot=200)
    # 芽を摘む h ＝ −1：cost 0.004、rep ＝ 0.4 − 2 × 0.04 ＝ 0.32。「いつも」の曲線の cost 0.004 は rep 0.36 → D ＝ 0.04
    assert abs(r['point']['D_grow_h10'] - 0.04) < 0.01, r['point']['D_grow_h10']
    assert abs(r['point']['D_melt_h10'] - (0.36 - 0.38)) < 0.01, r['point']['D_melt_h10']
    assert r['words']['PQ1'] == '芽を摘むほうが効く' and r['words']['PQ2'] == 'いつもの罰のほうが効く', r['words']
    # 3) 同じ効率なら「はっきりしない」
    r2 = analyze_model(_synth_S(eff_grow=1.0, eff_melt=1.0, rng=np.random.default_rng(3)), boot=200)
    assert r2['words']['PQ1'] == 'はっきりしない', (r2['words'], r2['ci']['D_grow_h10'])
    # 4) 写しの印の割合：くり返しの区間を入れた列で、印の割合が区間の長さに合う
    rng = np.random.default_rng(1)
    seq = rng.integers(100, 30000, 512); seq[200:400] = np.tile(rng.integers(100, 30000, 20), 10)
    c = copy_marks(seq)
    assert abs(c[16:496].mean() - (400 - 220) / 480) < 0.02, c[16:496].mean()
    return 'OK'
