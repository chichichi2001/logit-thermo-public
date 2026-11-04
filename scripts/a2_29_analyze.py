"""A2-29 の分析（CPU、探索）v0.2。ゲート：docs/A2_29_GATE.md。scripts/a2_22_run.py の load22ns() の名前空間で実行する
（seq_events・g_identity・rates・identity_split・A18_MIN_EV・A18_MIN_T・A18_BAND がある）。
XQ29：モデルごとの比 r_u ＝ D_u/P_u が、使う温度の端から端までで動く量 Δr（直線の当てはめ）。7 本の中央値。
XQ30：Δf ＝ f_id − f_meas（同じ選び直しの中で）。XQ31：使う温度の選び方 B〜E での複製の ρ の中央値。
v0.2（新鮮な目のチェックを受けて）：XQ29 を「区間が 0 を外す割合」から「温度で向きのそろったずれの大きさ」に。中央値は A2-19 と同じく
出ないモデル 2 本まで許す。有限でない値は NaN にそろえる。XQ31 は A と温度が変わらない選び方を「試していない」として判定から外す。
自己テストは作り物の腕で analyze29 を最後まで通す。
"""
import numpy as np

A29_SEED, A29_BOOT, A29_FAIL = 20261129, 1000, 0.10
A29_DR = 0.45                     # XQ29 の幅：帯 0.8〜1.25 の幅
G_FN = None                       # 自己テストだけで使う差しかえ（None なら g_identity）
SPLIT_FN = None                   # 同じく（None なら identity_split）


def _fin(x):
    x = np.array(x, float)
    x[~np.isfinite(x)] = np.nan
    return x


def per_temp(A0, A1, h, w):
    """1 本のモデル → 温度ごとの D（実現の Δln K）、P（h* g）、入る側の変化、両方の腕の入る回・出る回。"""
    lf0, lm0, f0, m0 = rates(A0['ev'], w)
    lf1, lm1, f1, m1 = rates(A1['ev'], w)
    D = _fin((lf1 - lm1) - (lf0 - lm0))
    g = G_FN(A0, w) if G_FN is not None else g_identity(A0['ev'], A0['sig'], A0['temps'], w)
    return dict(D=D, P=_fin(h * np.array(g, float)), ent=_fin(lf1 - lf0), cnt=np.array([f0, m0, f1, m1], float))


def select(cnt, rule):
    """使う温度の選び方。cnt：(4, nT) ＝ [c0 の入る回, c0 の出る回, h1 の入る回, h1 の出る回]。"""
    thr, rows = {'A': (A18_MIN_EV, [0, 1, 2, 3]), 'E': (A18_MIN_EV, [0, 1, 2, 3]), 'B': (A18_MIN_EV, [0, 1]),
                 'C': (5, [0, 1, 2, 3]), 'D': (20, [0, 1, 2, 3])}[rule]
    return [k for k in range(cnt.shape[1]) if all(cnt[r, k] >= thr for r in rows)]


def rho_of(pt, use):
    u = np.array(use, int)
    if len(u) < A18_MIN_T or not (np.all(np.isfinite(pt['D'][u])) and np.all(np.isfinite(pt['P'][u]))) or pt['P'][u].sum() == 0:
        return np.nan
    return float(pt['D'][u].sum() / pt['P'][u].sum())


def dr_of(pt, use, temps):
    """比 r_u ＝ D_u/P_u を温度に直線で当て、使う温度の端から端までで動く量（傾き × 温度の幅）。"""
    u = np.array(use, int)
    if len(u) < A18_MIN_T:
        return np.nan
    r = pt['D'][u] / pt['P'][u]; T = np.array(temps, float)[u]
    if not np.all(np.isfinite(r)):
        return np.nan
    return float(np.polyfit(T, r, 1)[0] * (T.max() - T.min()))


def f_pair(A0, pt, use, w):
    """f_meas（実測の入る側の割合）と f_id（恒等式の入る側の割合、c0 だけから）。"""
    u = np.array(use, int)
    if len(u) < A18_MIN_T:
        return np.nan, np.nan
    Du = pt['D'][u]
    fm = float(pt['ent'][u].sum() / Du.sum()) if np.all(np.isfinite(Du)) and np.all(np.isfinite(pt['ent'][u])) and Du.sum() != 0 else np.nan
    g_in, g_out = (SPLIT_FN(A0, w) if SPLIT_FN is not None else identity_split(A0, w)[:2])
    g = np.array(g_in, float) + np.array(g_out, float)
    fi = float(np.array(g_in, float)[u].sum() / g[u].sum()) if np.all(np.isfinite(g[u])) and g[u].sum() != 0 else np.nan
    return fm, fi


def _ci(v):
    v = _fin(v)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [np.nan, np.nan]


def _agg(v, how):
    """小さい 7 本は中央値（A2-19 と同じく、出ないモデルが 2 本までなら残りで）、7B の 2 本は平均（両方そろったときだけ）。"""
    v = _fin(v); ok = np.isfinite(v)
    if how == 'median':
        return float(np.median(v[ok])) if ok.sum() >= len(v) - 2 and ok.sum() > 0 else np.nan
    return float(np.mean(v)) if ok.all() else np.nan


def word29(ci, fail):
    if fail > A29_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if -A29_DR <= ci[0] and ci[1] <= A29_DR:
        return '温度ごとの比は帯の幅を越えて動かない'
    if ci[0] > A29_DR or ci[1] < -A29_DR:
        return '温度で向きのそろったずれ'
    return 'はっきりしない'


def word30(ci, fail):
    if fail > A29_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if ci[1] < 0:
        return '恒等式は出る側に寄せる'
    if ci[0] > 0:
        return '恒等式は入る側に寄せる'
    return '全体として向きのそろったずれは見えない'


def word31(rows):
    """rows：{選び方: dict(ci, fail, tested)}。試していない（A と同じ温度）選び方と、はっきりしない選び方は判定に入れない。"""
    ok = {k: r for k, r in rows.items() if r['tested'] and r['fail'] <= A29_FAIL and np.all(np.isfinite(r['ci']))}
    untested = [k for k, r in rows.items() if not r['tested']]
    unclear = [k for k, r in rows.items() if r['tested'] and k not in ok]
    if not ok:
        return 'はっきりしない', [], unclear, untested
    out = [k for k, r in ok.items() if r['ci'][0] < A18_BAND[0] or r['ci'][1] > A18_BAND[1]]
    return ('選び方で変わる' if out else 'どの選び方でも帯の内'), out, unclear, untested


def analyze29(small, big, seed=A29_SEED, boot=A29_BOOT):
    """small・big：{モデル: dict(c0=prep22 の材料, h1=…, h=h*)}。記事の並びは 9 本で同じ（120 本）。"""
    data = {**small, **big}
    names = list(data)
    nD = data[names[0]]['c0']['ev'].shape[2]
    assert all(d['c0']['ev'].shape[2] == nD and d['h1']['ev'].shape[2] == nD for d in data.values())
    one = np.ones(nD)
    rules = ('A', 'B', 'C', 'D', 'E')
    T = {m: data[m]['c0']['temps'] for m in names}
    P0 = {m: per_temp(d['c0'], d['h1'], d['h'], one) for m, d in data.items()}
    U = {m: {r: select(P0[m]['cnt'], r) for r in rules} for m in names}
    point = {m: dict(rho={r: rho_of(P0[m], U[m][r]) for r in rules}, dr=dr_of(P0[m], U[m]['A'], T[m]),
                     f=f_pair(data[m]['c0'], P0[m], U[m]['A'], one)) for m in names}
    rng = np.random.default_rng(seed)
    Eb = {m: [] for m in names}; DR = {m: [] for m in names}; FM = {m: [] for m in names}; FI = {m: [] for m in names}
    RHO = {m: {r: [] for r in rules} for m in names}; Ediff = {m: 0 for m in names}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        for m in names:
            d = data[m]; pt = per_temp(d['c0'], d['h1'], d['h'], w)
            Eb[m].append(pt['D'] - pt['P']); DR[m].append(dr_of(pt, U[m]['A'], T[m]))
            fm, fi = f_pair(d['c0'], pt, U[m]['A'], w); FM[m].append(fm); FI[m].append(fi)
            uE = select(pt['cnt'], 'E'); Ediff[m] += uE != U[m]['A']
            for r in rules:
                RHO[m][r].append(rho_of(pt, uE if r == 'E' else U[m][r]))
    res = {'models': {m: dict(use=U[m]['A'], rho=point[m]['rho']) for m in names}}
    # XQ29：判定は Δr。記録は温度ごとの e の区間など
    for grp, how, key in ((small, 'median', 'small'), (big, 'mean', 'big')):
        drb = np.array([DR[m] for m in grp])
        bs = [_agg(drb[:, b], how) for b in range(boot)]
        fail = float(np.mean(~np.isfinite(bs))); ci = _ci(bs)
        n_out = n_pairs = 0; pos = neg = 0
        for m in grp:
            e = np.array(Eb[m]); rows = []
            for k in U[m]['A']:
                v = _fin(e[:, k]); ci_e = _ci(v); fl = float(np.mean(~np.isfinite(v)))
                excl = bool(np.all(np.isfinite(ci_e)) and (ci_e[0] > 0 or ci_e[1] < 0))
                D, Pp = float(P0[m]['D'][k]), float(P0[m]['P'][k])
                rows.append(dict(k=int(k), T=float(T[m][k]), D=D, P=Pp, ratio=D / Pp if Pp != 0 else np.nan, e_ci=ci_e, fail=fl, excl=excl,
                                 halfwidth_over_P=float((ci_e[1] - ci_e[0]) / 2 / abs(Pp)) if Pp != 0 else np.nan,
                                 counts=P0[m]['cnt'][:, k].tolist()))
                n_pairs += 1; n_out += excl
                if excl:
                    pos += ci_e[0] > 0; neg += ci_e[1] < 0
            res['models'][m].update(per_T=rows, dr=point[m]['dr'], dr_ci=_ci(DR[m]), dr_fail=float(np.mean(~np.isfinite(_fin(DR[m])))),
                                    n_excl_pos=int(sum(r['excl'] and r['e_ci'][0] > 0 for r in rows)),
                                    n_excl_neg=int(sum(r['excl'] and r['e_ci'][1] < 0 for r in rows)))
        hw = [r['halfwidth_over_P'] for m in grp for r in res['models'][m]['per_T']]
        res[f'XQ29_{key}'] = dict(point=_agg([point[m]['dr'] for m in grp], how), ci=ci, fail=fail, word=word29(ci, fail),
                                  F=n_out / n_pairs if n_pairs else np.nan, n_pairs=n_pairs, n_excl=n_out, n_excl_pos=int(pos), n_excl_neg=int(neg),
                                  median_halfwidth_over_P=float(np.nanmedian(_fin(hw))) if hw else np.nan)
    # XQ30
    for grp, how, key in ((small, 'median', 'small'), (big, 'mean', 'big')):
        dfs = []
        for m in grp:
            fm0, fi0 = point[m]['f']
            dfb = _fin(np.array(FI[m]) - np.array(FM[m]))
            ci = _ci(dfb)
            res['models'][m].update(f_meas=fm0, f_id=fi0, df=fi0 - fm0, df_ci=ci, df_fail=float(np.mean(~np.isfinite(dfb))),
                                    df_excl0=bool(np.all(np.isfinite(ci)) and (ci[0] > 0 or ci[1] < 0)))
            dfs.append(dfb)
        dfs = np.array(dfs)
        bs = [_agg(dfs[:, b], how) for b in range(boot)]
        fail = float(np.mean(~np.isfinite(bs))); ci = _ci(bs)
        res[f'XQ30_{key}'] = dict(point=_agg([res['models'][m]['df'] for m in grp], how), ci=ci, fail=fail, word=word30(ci, fail),
                                  n_excl0=int(sum(res['models'][m]['df_excl0'] for m in grp)), n=len(grp))
    # XQ31
    for grp, how, key in ((small, 'median', 'small'), (big, 'mean', 'big')):
        rows = {}
        bA = [_agg([RHO[m]['A'][b] for m in grp], how) for b in range(boot)]
        for r in rules:
            bs = [_agg([RHO[m][r][b] for m in grp], how) for b in range(boot)]
            fail = float(np.mean(~np.isfinite(bs)))
            n_diff = int(sum(U[m][r] != U[m]['A'] for m in grp)) if r != 'E' else None
            frac_E = float(np.mean([Ediff[m] / boot for m in grp])) if r == 'E' else None
            tested = (n_diff > 0) if r != 'E' else (frac_E > 0)
            rows[r] = dict(point=_agg([point[m]['rho'][r] for m in grp], how), ci=_ci(bs), fail=fail, tested=bool(tested),
                           n_models_diff_from_A=n_diff, frac_resamples_diff_from_A=frac_E,
                           ratio_to_A_ci=_ci(_fin(np.array(bs) / np.array(bA))),
                           n_temps={m: len(U[m][r]) for m in grp} if r != 'E' else None)
        word, out, unclear, untested = word31({r: rows[r] for r in ('B', 'C', 'D', 'E')})
        res[f'XQ31_{key}'] = dict(rows=rows, word=word, out=out, unclear=unclear, untested=untested)
    return res


# ---------- 自己テスト ----------

def _toy_model(rng, temps, ratio, nD=60, S=4, h=-0.4):
    """作り物の 1 本：c0 と h1 の腕（ev だけ）と、実現 D_u ÷ 予言 P_u ＝ ratio[u] になるように決めた g。入る速さだけを変える。"""
    nT = len(temps)
    lk0 = np.linspace(-1.0, 1.0, nT); dK = np.full(nT, -0.4)
    def ev(lk):
        e = np.zeros((nT, S, nD, 4)); M = 2000.0; km = 0.02
        for k in range(nT):
            kf = km * np.exp(lk[k])
            e[k, :, :, 0] = M; e[k, :, :, 2] = M
            e[k, :, :, 1] = rng.poisson(M * kf, (S, nD)); e[k, :, :, 3] = rng.poisson(M * km, (S, nD))
        return e
    A0 = dict(ev=ev(lk0), sig=np.zeros((nT, S, nD)), temps=list(temps), parts={})
    A1 = dict(ev=ev(lk0 + dK), sig=np.zeros((nT, S, nD)), temps=list(temps), parts={})
    g = dK / (h * np.array(ratio, float))
    return dict(c0=A0, h1=A1, h=h), g


def self_test29(boot=60):
    global G_FN, SPLIT_FN
    rng = np.random.default_rng(11)
    temps = [0.35, 0.40, 0.45, 0.50, 0.55]
    cnt = np.array([[12, 30, 4, 25], [11, 30, 9, 25], [12, 6, 30, 25], [15, 7, 30, 19]], float)
    assert select(cnt, 'A') == [0, 3] and select(cnt, 'E') == [0, 3] and select(cnt, 'B') == [0, 1, 3] and select(cnt, 'C') == [0, 1, 3] \
        and select(cnt, 'D') == [], (select(cnt, 'A'), select(cnt, 'B'), select(cnt, 'C'), select(cnt, 'D'))
    try:
        for case, ratio, want in (('flat', [1.0] * 5, '温度ごとの比は帯の幅を越えて動かない'),
                                  ('tilt', [1.4, 1.2, 1.0, 0.8, 0.6], '温度で向きのそろったずれ')):
            small, big, gs = {}, {}, {}
            for i in range(7):
                d, g = _toy_model(rng, temps, ratio); small[f's{i}'] = d; gs[id(d['c0'])] = g
            for i in range(2):
                d, g = _toy_model(rng, temps, ratio); big[f'b{i}'] = d; gs[id(d['c0'])] = g
            G_FN = lambda A0, w: gs[id(A0)]
            SPLIT_FN = lambda A0, w: (gs[id(A0)], np.zeros(len(gs[id(A0)])))
            res = analyze29(small, big, boot=boot)
            assert res['XQ29_small']['word'] == want, (case, res['XQ29_small'])
            rhoA = [res['models'][m]['rho']['A'] for m in small]
            assert abs(np.median(rhoA) - 1) < 0.1, (case, rhoA)
            assert abs(res['models']['s0']['f_meas'] - 1) < 0.1 and abs(res['models']['s0']['f_id'] - 1) < 1e-9, \
                (res['models']['s0']['f_meas'], res['models']['s0']['f_id'])     # 出る回にもポアソンの揺れがあるので f_meas は 1 の近く
            assert res['XQ30_small']['word'] == '全体として向きのそろったずれは見えない', res['XQ30_small']
            assert res['XQ31_small']['word'] == 'はっきりしない' and set(res['XQ31_small']['untested']) == {'B', 'C', 'D', 'E'}, res['XQ31_small']
    finally:
        G_FN = SPLIT_FN = None
    assert word29([-0.3, 0.4], 0) == '温度ごとの比は帯の幅を越えて動かない' and word29([0.5, 0.9], 0) == '温度で向きのそろったずれ'
    assert word29([0.2, 0.9], 0) == 'はっきりしない' and word29([-0.1, 0.1], 0.2) == 'はっきりしない'
    assert word30([-0.2, -0.01], 0.0) == '恒等式は出る側に寄せる' and word30([-0.1, 0.05], 0.0) == '全体として向きのそろったずれは見えない'
    r = lambda lo, hi, t=True, f=0.0: dict(ci=[lo, hi], fail=f, tested=t)
    assert word31({'B': r(0.9, 1.2), 'C': r(0.85, 1.3)})[0] == '選び方で変わる'
    assert word31({'B': r(0.9, 1.2), 'C': r(np.nan, np.nan, f=0.5)})[:2] == ('どの選び方でも帯の内', [])
    assert word31({'B': r(0.9, 1.2, t=False), 'C': r(np.nan, np.nan, f=0.5)})[0] == 'はっきりしない'
    assert _agg([1, 2, 3, 4, 5, np.nan, np.inf], 'median') == 3.0 and np.isnan(_agg([1, 2, 3, 4, np.nan, np.nan, np.nan], 'median'))
    assert np.isnan(_agg([1.0, np.nan], 'mean')) and _ci([np.inf, -np.inf, 1.0, 2.0])[0] >= 1.0
    return 'A2-29 自己テスト OK'
