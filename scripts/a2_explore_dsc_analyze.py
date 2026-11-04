"""探索 8.33 の分析：文章の DSC（温度の往復）で、凍る温度と溶ける温度のずれ（ヒステリシス）を測る（CPU、判定しない）。
データを開く前に書いた（2026-10-31 13:4x）。scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases）を実行したのと同じ名前空間で実行する。

相：2 つの状態（A2-8 と同じ：写しの印、中心の窓 32、入口 0.5・出口 0.25）。歩 16〜L−16 にラベルがつく。
温度の箱：0.35〜0.85 を 0.05 ずつ 10 個。片道ごと（冷やす道・温める道）に、箱ごとの結晶の割合 φ を出す。
量（モデル × 場 × 速さ × 順番ごと）：
  T_half：φ が「基準の高さ」を横切る温度（高い温度から下へ見て、はじめて φ ≥ 基準になる箱と、その 1 つ上の箱のあいだを直線でつなぐ）。
          基準の高さ ＝ 2 つの道の φ の最大の半分（腕ごと）。
  ループの面積 ＝ Σ（φ_温める − φ_冷やす）× 0.05（正ならヒステリシス）。
  幅 ＝ T_half(温める) − T_half(冷やす)、真ん中 ＝ 2 つの平均。
順番の組み合わせ：'ch'（冷やすが先）と 'hc'（温めるが先）。「後ろの位置ほど結晶ができやすい」効きは、ch では温める道を、hc では冷やす道を
  結晶寄りにするので、ヒステリシス ＝（面積_ch ＋ 面積_hc）÷ 2、位置の効き ＝（面積_ch − 面積_hc）÷ 2 と読む（幅・真ん中も同じ）。
エネルギー：箱ごとに、液体の歩と結晶の歩の U（−ln p、温度 1・場なし）の平均。ΔH_熱量計(T) ＝ U_液体 − U_結晶（2 つの道・2 つの順番を足す）。
  キルヒホッフ：ΔH_熱量計(T) を T に対して直線で当てはめた傾き ＝ ΔC（回数の重み。液体・結晶ともに 200 歩以上の箱だけ）。
  見かけの比熱：道ごとの U の平均の、箱ごとの差 ÷ 0.05（記録だけ）。
比べる相手（データを開く前に分かっている値、8.30 の一定温度の融点）：Pythia-1B 0.517・0.595・0.678、OLMo-2-1B 0.496・0.550・0.620（h ＝ −0.3・0・＋0.3）。
区間：記事の選び直し 1,000 回（種 20261134、モデルの中で全部の腕に同じ選び方）。
"""
import numpy as np

DA_EDGES = np.linspace(0.35, 0.85, 11)
DA_CENT = (DA_EDGES[:-1] + DA_EDGES[1:]) / 2
DA_NB = 10
DA_W = 0.05
DA_MIN_STEPS = 200
DA_BOOT, DA_SEED = 1000, 20261134
DA_TM_FIELD = {'pythia-1b': {'hm03': 0.5174, 'h0': 0.5947, 'hp03': 0.6781}, 'olmo2-1b': {'hm03': 0.4962, 'h0': 0.5497, 'hp03': 0.6204}}
DA_HS = ['hm03', 'h0', 'hp03']
DA_ARMS = [(o, ha, 'slow') for ha in DA_HS for o in ('ch', 'hc')] + [(o, 'h0', 'fast') for o in ('ch', 'hc')]


def da_bin(T):
    return np.clip(np.searchsorted(DA_EDGES, T, side='right') - 1, 0, DA_NB - 1)


def dsc_doc_stats(toks, U, Ts, order):
    """toks・U：(S, nD, L)、Ts：(L,) → (nD, 2, 10, 5)：[道 0 ＝ 冷やす・1 ＝ 温める] × 箱 × [結晶の歩, 歩, U 液体の和, 液体の歩, U 結晶の和]。"""
    S, nD, L = toks.shape
    n = L // 2
    leg = np.zeros(L, int)
    first, second = (0, 1) if order == 'ch' else (1, 0)
    leg[:n] = first; leg[n:] = second
    b = da_bin(Ts)
    out = np.zeros((nD, 2, DA_NB, 5))
    for j in range(nD):
        for s in range(S):
            st, t = phases(copy_marks(toks[s, j]))
            x = st.astype(float); u = U[s, j, t].astype(float)
            lg, bb = leg[t], b[t]
            key = lg * DA_NB + bb
            for col, val in ((0, x), (1, np.ones_like(x)), (2, u * (1 - x)), (3, 1 - x), (4, u * x)):
                out[j, :, :, col] += np.bincount(key, weights=val, minlength=2 * DA_NB).reshape(2, DA_NB)
    return out


def t_half(phi, level):
    """phi：箱ごとの結晶の割合（低い温度 → 高い温度）。高い温度から下へ見て、はじめて phi ≥ level になる温度（直線でつなぐ）。"""
    for k in range(DA_NB - 1, -1, -1):
        if np.isfinite(phi[k]) and phi[k] >= level:
            if k == DA_NB - 1:
                return float(DA_CENT[k])
            a, b = phi[k], phi[k + 1]
            if not np.isfinite(b) or a == b:
                return float(DA_CENT[k])
            return float(DA_CENT[k] + (a - level) / (a - b) * DA_W)
    return float('nan')


def arm_point(st):
    """st：(2, 10, 5)（記事を足したもの）→ 腕の量。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = st[:, :, 0] / st[:, :, 1]
    level = 0.5 * np.nanmax(phi)
    tc, th = t_half(phi[0], level), t_half(phi[1], level)
    area = float(np.nansum((phi[1] - phi[0]) * DA_W))
    return dict(phi_cool=phi[0].tolist(), phi_heat=phi[1].tolist(), level=float(level), T_cool=tc, T_heat=th,
                width=th - tc, center=(th + tc) / 2, area=area)


def kirchhoff(st_sum):
    """st_sum：(2, 10, 5)（道・順番・腕を足したもの）→ ΔH_熱量計(T) と ΔC（直線の傾き）。"""
    tot = st_sum.sum(0)
    nl, nc = tot[:, 3], tot[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        dh = tot[:, 2] / nl - tot[:, 4] / nc
    ok = (nl >= DA_MIN_STEPS) & (nc >= DA_MIN_STEPS) & np.isfinite(dh)
    res = dict(dH_T=[float(x) if o else None for x, o in zip(dh, ok)], n_liq=nl.tolist(), n_cry=nc.tolist())
    if ok.sum() >= 3:
        w = 1.0 / (1.0 / nl[ok] + 1.0 / nc[ok])
        X = np.stack([np.ones(ok.sum()), DA_CENT[ok]], 1)
        a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * dh[ok]))
        res.update(dC=float(b), dH_at_055=float(a + b * 0.55))
    else:
        res.update(dC=float('nan'), dH_at_055=float('nan'))
    return res


def model_point(stats):
    """stats：{(順番, 場, 速さ): (nD, 2, 10, 5)} → モデルの量。"""
    arms = {k: arm_point(v.sum(0)) for k, v in stats.items()}
    comb = {}
    for ha, rate in [(h, 'slow') for h in DA_HS] + [('h0', 'fast')]:
        a, b = arms[('ch', ha, rate)], arms[('hc', ha, rate)]
        comb[f'{ha}_{rate}'] = {k: (a[k] + b[k]) / 2 for k in ('area', 'width', 'center')}
        comb[f'{ha}_{rate}'].update({f'pos_{k}': (a[k] - b[k]) / 2 for k in ('area', 'width', 'center')})
    kir = {ha: kirchhoff(sum(stats[(o, ha, 'slow')].sum(0) for o in ('ch', 'hc'))) for ha in DA_HS}
    kir['all'] = kirchhoff(sum(v.sum(0) for k, v in stats.items() if k[2] == 'slow'))
    # 見かけの比熱（道ごと、h ＝ 0・ゆっくり、2 つの順番を足す）
    s0 = sum(stats[(o, 'h0', 'slow')].sum(0) for o in ('ch', 'hc'))
    with np.errstate(divide='ignore', invalid='ignore'):
        Ubar = (s0[:, :, 2] + s0[:, :, 4]) / s0[:, :, 1]
    capp = (np.diff(Ubar, axis=1) / DA_W).tolist()
    return dict(arms={'|'.join(k): v for k, v in arms.items()}, comb=comb, kirchhoff=kir, U_bar_h0=Ubar.tolist(), C_app_h0=capp)


def analyze_dsc(base, name):
    stats, uses = {}, []
    for o, ha, rate in DA_ARMS:
        Z = np.load(f'{base}/{name}@{o}_{ha}_{rate}.npz')
        assert str(Z['order']) == o and str(Z['rate']) == rate, (name, o, ha, rate)
        stats[(o, ha, rate)] = dsc_doc_stats(Z['toks'], Z['U'], np.asarray(Z['Ts']), o)
        uses.append(np.asarray(Z['use']).tolist())
    assert all(u == uses[0] for u in uses), name
    pt = model_point(stats)
    pt['Tm_field'] = DA_TM_FIELD.get(name)
    nD = next(iter(stats.values())).shape[0]
    rng = np.random.default_rng(DA_SEED)
    keys = [(c, q) for c in pt['comb'] for q in ('area', 'width', 'center', 'pos_area')]
    bs = {k: [] for k in keys}; bk = {ha: [] for ha in DA_HS + ['all']}
    for _ in range(DA_BOOT):
        idx = rng.integers(0, nD, nD)
        b = model_point({k: v[idx] for k, v in stats.items()})
        for c, q in keys:
            bs[(c, q)].append(b['comb'][c][q])
        for ha in bk:
            bk[ha].append(b['kirchhoff'][ha]['dC'])
    pc = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    pt['ci'] = {f'{c}|{q}': pc(v) for (c, q), v in bs.items()}
    pt['ci_dC'] = {ha: pc(v) for ha, v in bk.items()}
    pt['boot_nan'] = {f'{c}|{q}': float(np.mean(~np.isfinite(np.array(v, float)))) for (c, q), v in bs.items()}
    return pt


def self_test_dsc_an():
    """作り物：冷やす道では T < 0.5 で結晶になり、温める道では T > 0.6 まで結晶が残る（周期 23）。液体の U ＝ 3、結晶の U ＝ 0.5。
    → T_cool ≈ 0.5、T_heat ≈ 0.6、面積 ≈ 0.1、両方の相がある箱の ΔH_熱量計 ≈ 2.5。'hc' の順でも同じ形に数えること。
    キルヒホッフの傾きは、作った数えで別に確かめる。"""
    rng = np.random.default_rng(3)
    n = 640
    out = {}
    for order in ('ch', 'hc'):
        Ts = np.concatenate([np.linspace(0.85, 0.35, n), np.linspace(0.35, 0.85, n)] if order == 'ch'
                            else [np.linspace(0.35, 0.85, n), np.linspace(0.85, 0.35, n)])
        toks = np.zeros((2, 3, 2 * n), np.int64); U = np.full((2, 3, 2 * n), 3.0, np.float32)
        for s in range(2):
            for j in range(3):
                seq = rng.integers(100, 30000, 2 * n)
                if order == 'ch':
                    a = int(np.argmax(Ts < 0.5)); b = n + int(np.argmax(Ts[n:] > 0.6))
                    segs = [(a, b)]
                else:
                    b0 = int(np.argmax(Ts[:n] > 0.6)); a1 = n + int(np.argmax(Ts[n:] < 0.5))
                    segs = [(0, b0), (a1, 2 * n)]
                for a, b in segs:
                    blk = rng.integers(100, 30000, 23)
                    seq[a:b] = np.tile(blk, (b - a) // 23 + 1)[:b - a]
                    U[s, j, a + 23:b] = 0.5
                toks[s, j] = seq
        st = dsc_doc_stats(toks, U, Ts, order).sum(0)
        r = arm_point(st)
        out[order] = r
        assert abs(r['T_cool'] - 0.5) < 0.04 and abs(r['T_heat'] - 0.6) < 0.04, (order, r['T_cool'], r['T_heat'])
        assert 0.06 < r['area'] < 0.14, (order, r['area'])
        k = kirchhoff(st)
        ok = [x for x in k['dH_T'] if x is not None]
        assert ok and all(abs(x - 2.5) < 0.3 for x in ok), (order, k['dH_T'])
    # キルヒホッフ：ΔH ＝ 2.5 ＋ 1.0 ×（T − 0.55）になるように作った数え → 傾き 1.0
    st = np.zeros((2, DA_NB, 5))
    for k_ in range(DA_NB):
        dh = 2.5 + 1.0 * (DA_CENT[k_] - 0.55)
        st[0, k_] = [300, 600, 300 * (1.0 + dh), 300, 300 * 1.0]
    k = kirchhoff(st)
    assert abs(k['dC'] - 1.0) < 1e-9 and abs(k['dH_at_055'] - 2.5) < 1e-9, k
    return 'OK'
