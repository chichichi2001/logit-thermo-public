"""探索 8.38 の分析：DSC その 2（1.5 まで温める）。データを開く前に書いた（2026-10-31 17:0x）。CPU、判定しない。
scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases）を実行したのと同じ名前空間で実行する。

相：2 つの状態（A2-8 と同じ窓のものさし、8.33 と同じ）。
道：冷やす道 ＝ 歩 0〜639（4 本の腕で同じトークン）。温める道 ＝ base・penheat は 640〜1279、hold は 960〜1599、fastheat は 640〜959。
温度の箱：冷やす道 0.35〜0.85（0.05 ずつ 10 個）、温める道 0.35〜1.5（0.05 ずつ 23 個）。
量（モデル × 腕）：
  φ_温(T)：温める道の、箱ごとの結晶の割合。φ0 ＝ 温める道のいちばん低い箱の φ。
  T_melt：低い温度から上へ見て、はじめて φ < φ0/2 になる温度（その箱と 1 つ下の箱のあいだを直線でつなぐ）。横切らなければ打ち切り（nan、打ち切りの印つき）。
  吸う熱（溶ける潜熱、1 トークンあたり）：Σ_箱 ΔH_熱量計(T) ×（φ(箱) − φ(次の箱)）（φ が減るぶんだけ。増えるぶんは別に「凍った熱」として足す）。
  出す熱（凍る潜熱）：冷やす道で同じように（8.33 結果 4 と同じ）。
  熱の収支：（吸う熱 ÷ 溶けた割合）÷（出す熱 ÷ 凍った割合）。1 なら、凍るときに出した熱と溶けるときに吸った熱が、結晶 1 トークンあたりで同じ。
  ものさしの較正：窓のものさしは入口と出口の決め方が違う（履歴つき）ので、作り物（真の収支 1）でもずれる。その作り物の値（synth_point）も一緒に記録し、実データの収支はそれと比べて読む。
  ΔH_熱量計(T)：温める道（base）で、箱ごとの液体の歩と結晶の歩の U の平均の差（両方 200 歩以上の箱）。冷やす道の値とあわせて、0.35〜1.5 のキルヒホッフ（直線の傾き ΔC）。
  対になった比べ（base との差）：hold（熟成）・fastheat（速く温める）・penheat（強い罰）の T_melt − base の T_melt、溶けた割合の差。
  penheat：温める道の結晶の歩で、場をかける前の候補の確率 q0 とかけた後の q の平均（罰がどれだけ確率を動かしたか）。
区間：記事の選び直し 500 回（種 20261138、モデルの中で 4 本の腕に同じ選び方）。
"""
import numpy as np

D2A_ARMS = ['base', 'hold', 'fastheat', 'penheat']
D2A_COOL_E = np.linspace(0.35, 0.85, 11)
D2A_HEAT_E = np.round(np.arange(0.35, 1.5 + 1e-9, 0.05), 4)
D2A_MIN = 200
D2A_BOOT, D2A_SEED = 500, 20261138


def heat_slice(arm):
    return {'base': slice(640, 1280), 'penheat': slice(640, 1280), 'hold': slice(960, 1600), 'fastheat': slice(640, 960)}[arm]


def binidx(T, edges):
    return np.clip(np.searchsorted(edges, T, side='right') - 1, 0, len(edges) - 2)


def doc_bins(toks, U, Ts, sl, edges, extra=None):
    """toks・U：(S, nD, L) → (nD, 箱, 6)：[結晶の歩, 歩, U 液体の和, 液体の歩, U 結晶の和, extra の結晶での和]（歩 sl のうち、ラベルがつく歩だけ）。"""
    S, nD, L = toks.shape
    nb = len(edges) - 1
    out = np.zeros((nD, nb, 6))
    b = binidx(Ts, edges)
    lo, hi = sl.start, sl.stop
    for j in range(nD):
        for s in range(S):
            st, t = phases(copy_marks(toks[s, j]))
            m = (t >= lo) & (t < hi)
            x = st[m].astype(float); tt = t[m]; u = U[s, j, tt].astype(float); bb = b[tt]
            ex = extra[s, j, tt].astype(float) if extra is not None else np.zeros_like(x)
            for col, val in ((0, x), (1, np.ones_like(x)), (2, u * (1 - x)), (3, 1 - x), (4, u * x), (5, ex * x)):
                out[j, :, col] += np.bincount(bb, weights=val, minlength=nb)
    return out


def phi_of(st):
    with np.errstate(divide='ignore', invalid='ignore'):
        return st[:, 0] / st[:, 1]


def dH_of(st):
    with np.errstate(divide='ignore', invalid='ignore'):
        d = st[:, 2] / st[:, 3] - st[:, 4] / st[:, 0]
    ok = (st[:, 3] >= D2A_MIN) & (st[:, 0] >= D2A_MIN)
    return np.where(ok, d, np.nan)


def t_melt(phi, edges):
    c = (edges[:-1] + edges[1:]) / 2
    ok = np.isfinite(phi)
    if not ok.any():
        return float('nan'), True
    p0 = phi[np.argmax(ok)]
    lvl = p0 / 2
    for k in range(1, len(phi)):
        if np.isfinite(phi[k]) and phi[k] < lvl:
            a, b = phi[k - 1], phi[k]
            if not np.isfinite(a) or a == b:
                return float(c[k]), False
            return float(c[k - 1] + (a - lvl) / (a - b) * (c[k] - c[k - 1])), False
    return float('nan'), True


def latent(phi, dH, decreasing=True):
    """箱ごとの φ と ΔH → φ が減るぶん（decreasing）か増えるぶんの Σ ΔH × |Δφ|（ΔH は 2 つの箱の平均、どちらか nan ならある方）、と |Δφ| の和。"""
    tot, frac = 0.0, 0.0
    for k in range(len(phi) - 1):
        a, b = phi[k], phi[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        d = a - b if decreasing else b - a
        if d <= 0:
            continue
        h = np.nanmean([dH[k], dH[k + 1]]) if (np.isfinite(dH[k]) or np.isfinite(dH[k + 1])) else np.nan
        if np.isfinite(h):
            tot += h * d; frac += d
    return tot, frac


def kirchhoff2(dHc, dHh):
    """冷やす道と温める道の ΔH_熱量計(T) を合わせて、温度にまっすぐ当てはめた傾き（箱の真ん中の温度）。"""
    cc = (D2A_COOL_E[:-1] + D2A_COOL_E[1:]) / 2; ch = (D2A_HEAT_E[:-1] + D2A_HEAT_E[1:]) / 2
    x = np.r_[cc, ch]; y = np.r_[dHc, dHh]; ok = np.isfinite(y)
    if ok.sum() < 3:
        return float('nan'), float('nan')
    b, a = np.polyfit(x[ok], y[ok], 1)
    return float(b), float(a + b * 0.55)


def point(cool, heat):
    """cool：(nD, 10, 6)（base の冷やす道）、heat：{腕: (nD, 23, 6)} → 量。"""
    sc = cool.sum(0); phic = phi_of(sc); dHc = dH_of(sc)
    sh = {a: v.sum(0) for a, v in heat.items()}
    dHh = dH_of(sh['base'])
    res = {'phi_cool': phic.tolist(), 'dH_cool': dHc.tolist(), 'dH_heat_base': dHh.tolist()}
    res['dC'], res['dH055'] = kirchhoff2(dHc, dHh)
    # 箱の ΔH が出ない（片方の相が 200 歩に足りない）ところは、キルヒホッフの直線で埋める（直線もなければ、出た箱の平均）
    allv = np.r_[dHc, dHh]; mean_dH = float(np.nanmean(allv)) if np.isfinite(allv).any() else float('nan')
    line = lambda e: (res['dH055'] + res['dC'] * ((e[:-1] + e[1:]) / 2 - 0.55)) if np.isfinite(res['dC']) else np.full(len(e) - 1, mean_dH)
    dHc_f = np.where(np.isfinite(dHc), dHc, line(D2A_COOL_E)); dHh_f = np.where(np.isfinite(dHh), dHh, line(D2A_HEAT_E))
    exo, fexo = latent(phic, dHc_f, decreasing=True)               # 冷やす道は温度の低い箱が先なので、温度の高い方へ φ が減る ＝ 凍った量
    res.update(exo=exo, exo_frac=fexo, exo_per=exo / fexo if fexo > 0 else float('nan'))
    for a, s in sh.items():
        ph = phi_of(s); tm, cens = t_melt(ph, D2A_HEAT_E)
        dh_use = np.where(np.isfinite(dH_of(s)), dH_of(s), dHh_f)
        endo, fendo = latent(ph, dh_use, decreasing=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            q0q = (s[:, 5].sum() / s[:, 0].sum()) if s[:, 0].sum() > 0 else float('nan')
        res[a] = dict(phi_heat=ph.tolist(), T_melt=tm, censored=cens, endo=endo, endo_frac=fendo,
                      endo_per=endo / fendo if fendo > 0 else float('nan'),
                      balance=(endo / fendo) / res['exo_per'] if (fendo > 0 and np.isfinite(res['exo_per'])) else float('nan'),
                      melted=float(ph[np.isfinite(ph)][0] - np.nanmin(ph[np.isfinite(ph)])) if np.isfinite(ph).any() else float('nan'),
                      extra_mean_cry=q0q)
    for a in ('hold', 'fastheat', 'penheat'):
        res[f'd_Tmelt_{a}'] = res[a]['T_melt'] - res['base']['T_melt']
        res[f'd_melted_{a}'] = res[a]['melted'] - res['base']['melted']
    return res


def analyze_dsc2(base_dir, name):
    Z = {a: np.load(f'{base_dir}/{name}@{a}.npz') for a in D2A_ARMS}
    uses = [np.asarray(Z[a]['use']).tolist() for a in D2A_ARMS]
    assert all(u == uses[0] for u in uses), name
    for a in D2A_ARMS[1:]:
        assert (Z[a]['toks'][:, :, :640] == Z['base']['toks'][:, :, :640]).all(), (name, a, '冷やす道が base と違う')
    cool = doc_bins(Z['base']['toks'], Z['base']['U'], np.asarray(Z['base']['Ts']), slice(0, 640), D2A_COOL_E)
    heat = {}
    for a in D2A_ARMS:
        extra = Z[a]['q'] - Z[a]['q0'] if a == 'penheat' else None
        heat[a] = doc_bins(Z[a]['toks'], Z[a]['U'], np.asarray(Z[a]['Ts']), heat_slice(a), D2A_HEAT_E, extra)
    pt = point(cool, heat)
    # penheat の q0・q の平均（結晶の歩）
    nD = cool.shape[0]
    rng = np.random.default_rng(D2A_SEED)
    keys = ['exo_per', 'dC', 'dH055'] + [f'd_Tmelt_{a}' for a in ('hold', 'fastheat', 'penheat')] + [f'd_melted_{a}' for a in ('hold', 'fastheat', 'penheat')]
    bs = {k: [] for k in keys}; bsa = {(a, k): [] for a in D2A_ARMS for k in ('T_melt', 'endo_per', 'balance', 'melted')}
    for _ in range(D2A_BOOT):
        idx = rng.integers(0, nD, nD)
        b = point(cool[idx], {a: v[idx] for a, v in heat.items()})
        for k in keys:
            bs[k].append(b[k])
        for a in D2A_ARMS:
            for k in ('T_melt', 'endo_per', 'balance', 'melted'):
                bsa[(a, k)].append(b[a][k])
    pc = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(np.array(v, float)).any() else [float('nan')] * 2
    pt['ci'] = {k: pc(v) for k, v in bs.items()}
    pt['ci_arm'] = {f'{a}|{k}': pc(v) for (a, k), v in bsa.items()}
    pt['censored_frac'] = {a: float(np.mean(~np.isfinite(np.array(bsa[(a, 'T_melt')], float)))) for a in D2A_ARMS}
    return pt


def synth_point(seed=8, n_doc=12):
    """作り物：冷やす道で T < 0.5（記事ごとに ±0.04）から結晶（周期 23）、温める道で T > 1.0（±0.04）で溶ける。液体の U ＝ 3、結晶の U ＝ 0.5（真の ΔH ＝ 2.5、
    真の収支 ＝ 1）。窓のものさし（入口 0.5・出口 0.25 の履歴つき）が、凍るときと溶けるときの潜熱をどれだけゆがめるかの「ものさしの較正」にも使う。"""
    rng = np.random.default_rng(seed)
    Ts = np.r_[np.linspace(0.85, 0.35, 640), np.linspace(0.35, 1.5, 640)]
    toks = np.zeros((2, n_doc, 1280), np.int64); U = np.full((2, n_doc, 1280), 3.0, np.float32)
    for s in range(2):
        for j in range(n_doc):
            ta, tb = 0.5 + 0.02 * (j % 5 - 2), 1.0 + 0.02 * (j % 5 - 2)
            a = int(np.argmax(Ts < ta)); b = 640 + int(np.argmax(Ts[640:] > tb))
            seq = rng.integers(100, 30000, 1280)
            blk = rng.integers(100, 30000, 23); seq[a:b] = np.tile(blk, (b - a) // 23 + 1)[:b - a]
            U[s, j, a + 23:b] = 0.5
            toks[s, j] = seq
    cool = doc_bins(toks, U, Ts, slice(0, 640), D2A_COOL_E)
    hb = doc_bins(toks, U, Ts, slice(640, 1280), D2A_HEAT_E)
    return point(cool, {k: hb for k in D2A_ARMS})


def self_test_dsc2_an():
    """作り物で：T_melt ≈ 1.0、凍る・溶ける潜熱が真の値（2.5）の 6〜12 割、収支が 0.6〜1.2（窓のものさしのゆがみのぶん）、対の差 ＝ 0。"""
    r = synth_point()
    assert abs(r['base']['T_melt'] - 1.0) < 0.05, r['base']['T_melt']
    assert 1.5 < r['exo_per'] < 3.0 and 1.5 < r['base']['endo_per'] < 3.0, (r['exo_per'], r['base']['endo_per'])
    assert 0.6 < r['base']['balance'] < 1.2, r['base']['balance']
    assert r['d_Tmelt_hold'] == 0.0
    return 'OK'
