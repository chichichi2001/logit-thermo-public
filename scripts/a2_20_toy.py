"""A2-20 の記録の問い（XQ7-模型）：厳密に解ける「写しのはしご」（8.36）を 7B の第 1 段（h ＝ 0）のデータに合わせ、
第 2 段の h* での 1 次の予言の比 ρ₁ の見込みを、第 2 段のデータの前に出して封をする（2026-11-02 JST、CPU。判定には使わない）。

合わせる 5 つの数（8.36 と同じ。どれも第 1 段の h ＝ 0 のデータの、予言の窓の 7 つの温度から）：
  T_m・ΔH_vH：予言（scripts/a2_20_predict.py）の窓の当てはめの T_m0 と b。
  E_a,f：凍る速さ k_f ＝ 凍る回 ÷ 液体の歩 のアレニウス（ln k_f ＝ ln A − E_a,f ÷ T、窓の 7 つで重みなしの直線）。
  m_liq・m_cry：液体・結晶の歩のうち候補を選んだ割合。T_m0 をはさむ 2 つの格子の温度で出し、T について直線で内挿。
合わせ方：8.36（scripts/a2_explore_toy_fit.py）と同じ（m は 3・4・5・6・8 から、それぞれ 15 の出発点、範囲つきの最小二乗）。
  違うのは、ひな形のファントホッフ・アレニウスを、0.4〜0.7 ではなく窓の 7 つの温度で出すことだけ。
見込み：ひな形で、第 2 段の 7 つの温度ごとに Δ_模型(T) ＝ ln K(T, h*) − ln K(T, 0) と P_模型(T) ＝ h* × ∂ln K/∂h（h ＝ 0、差分 1e-4）を厳密に出し、
  ρ₁_模型 ＝ Σ Δ_模型 ÷ Σ P_模型。2 本の平均も出す。
"""
import numpy as np, json, warnings
warnings.filterwarnings('ignore')

T20_SC = dict(Tm=0.02, dH=0.2, Ea_f=0.1, m_liq=0.02, m_cry=0.02)
T20_MS = (3, 4, 5, 6, 8)


def toy_targets(tot, temps, window, Tm0, dH, liq_cry_pick):
    """tot：(nT, 16) 記事を足した窓の数え（0 液体の歩・1 凍る回・2 結晶の歩・3 溶ける回）。window：窓の番号。liq_cry_pick：(nT, 2) 液体・結晶の歩で候補を選んだ割合。"""
    w = list(window); Tw = np.array([temps[i] for i in w])
    kf = tot[w, 1] / tot[w, 0]
    bf, af = np.polyfit(1 / Tw, np.log(kf), 1)
    j = int(np.searchsorted(Tw, Tm0)); j = min(max(j, 1), len(Tw) - 1)
    t0, t1 = Tw[j - 1], Tw[j]; f = (Tm0 - t0) / (t1 - t0)
    lc = liq_cry_pick[w]
    m_liq = float((1 - f) * lc[j - 1, 0] + f * lc[j, 0]); m_cry = float((1 - f) * lc[j - 1, 1] + f * lc[j, 1])
    return dict(Tm=float(Tm0), dH=float(dH), Ea_f=float(-bf), m_liq=m_liq, m_cry=m_cry), [float(t) for t in Tw]


def liq_cry_pick_from(toks, pick):
    """(nT, S, nD, L) → 温度ごとの [液体の歩で候補を選んだ割合, 結晶の歩で候補を選んだ割合]（窓のものさしの相で）。"""
    nT = toks.shape[0]; out = np.zeros((nT, 2));
    for k in range(nT):
        n = np.zeros(2); c = np.zeros(2)
        for s in range(toks.shape[1]):
            for j in range(toks.shape[2]):
                st, _ = phases(copy_marks(toks[k, s, j]))
                pk = pick[k, s, j][:len(st)].astype(bool)
                for ph in (0, 1):
                    sel = st == ph
                    n[ph] += sel.sum(); c[ph] += (pk & sel).sum()
        out[k] = c / np.maximum(n, 1)
    return out


def fit_toy(tg, temps_fit, seed=0, starts=15):
    """8.36 と同じ合わせ方。ひな形のファントホッフ・アレニウスは temps_fit（窓の 7 つ）で出す。"""
    from scipy.optimize import least_squares
    MM = [8]
    mk = lambda x: Toy(a=x[0], b=x[1], Delta=np.exp(x[2]), B=np.exp(x[3]), rho=1 / (1 + np.exp(-x[4])), m=MM[0])

    def resid(x):
        t = mk(x)
        try:
            v = t.vh(temps_fit); pm = t.phase_means(v['Tm'] if np.isfinite(v['Tm']) else float(np.mean(temps_fit)))
        except Exception:
            return np.full(5, 1e3)
        got = dict(Tm=v['Tm'], dH=v['dH'], Ea_f=v['Ea_f'], m_liq=pm['m_liq'], m_cry=pm['m_cry'])
        r = np.array([(got[k] - tg[k]) / T20_SC[k] for k in T20_SC])
        return np.where(np.isfinite(r), r, 1e3)
    best, bm = None, None
    lo = np.array([-2, 0, np.log(0.3), 0, -6]); hi = np.array([4, 0.3, np.log(30), 35, 3])
    for mm in T20_MS:
        MM[0] = mm
        rng = np.random.default_rng(seed)
        for i in range(starts):
            r = least_squares(resid, lo + (hi - lo) * rng.random(5), bounds=(lo, hi), max_nfev=300)
            if best is None or r.cost < best.cost:
                best, bm = r, mm
    MM[0] = bm
    x = best.x
    par = dict(a=float(x[0]), b=float(x[1]), Delta=float(np.exp(x[2])), B=float(np.exp(x[3])), rho=float(1 / (1 + np.exp(-x[4]))), m=bm)
    return par, float(best.cost), [float(v) for v in resid(x)]


def toy_rho1(par, h, grid, e=1e-4):
    """ひな形で、第 2 段の温度ごとに Δ と P を厳密に出す → ρ₁_模型、温度ごとの値。"""
    t = Toy(par['a'], par['b'], par['Delta'], par['B'], par['rho'], m=par['m'])
    lnK = lambda T, hh: float(np.log(np.divide(*t.rates(T, hh))))
    D, P = [], []
    for T in grid:
        g = (lnK(T, e) - lnK(T, -e)) / (2 * e)
        D.append(lnK(T, h) - lnK(T, 0.0)); P.append(h * g)
    D, P = np.array(D), np.array(P)
    return float(D.sum() / P.sum()), D.tolist(), P.tolist()


def xq7_toy_word(ci, rho_toy):
    """記録の言葉：測った 2 本の平均の区間に、模型の見込みと 1（ずれなし）がそれぞれ入るか。"""
    inside = lambda x: bool(np.isfinite(x) and ci[0] <= x <= ci[1])
    a, b = inside(rho_toy), inside(1.0)
    if a and b:
        return '模型とも、ずれなしとも矛盾しない（見分けられない）'
    if a:
        return '模型と矛盾しない（ずれなしとは矛盾する）'
    if b:
        return 'ずれなしと矛盾しない（模型とは矛盾する）'
    return 'どちらとも矛盾する'


def self_test_t20(toy_fit_json='results/A2_explore/toy_fit.json'):
    # (1) 8.36 の合わせ方を、0.4〜0.7 の温度で回すと、8.36 の結果（toy_fit.json）と同じ値になる
    F = json.load(open(toy_fit_json))
    tg = dict(Tm=0.595, dH=3.32, Ea_f=-1.54, m_liq=0.185, m_cry=0.931)
    par, cost, _ = fit_toy(tg, [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    ref = F['pythia-1b']
    assert par['m'] == ref['params']['m'] and abs(cost - ref['cost']) < 1e-6, (par, cost, ref['cost'])
    # (2) ひな形の ρ₁：罰（h ＜ 0）で 1 を少し上回る（09:4x の計算と同じ）
    r, _, _ = toy_rho1(ref['params'], -0.5, [0.40, 0.45])
    assert 1.02 < r < 1.06, r
    # (3) 記録の言葉
    assert xq7_toy_word([0.9, 1.1], 1.04).startswith('模型とも') and xq7_toy_word([1.02, 1.2], 1.04).startswith('模型と矛盾しない')
    return 'OK'


def toy_for_model(path, pred, expect_temps=None):
    """第 1 段の npz（h ＝ 0）と、その予言（predict_model20 の出力）→ ひな形の 5 つの数・合わせた値・ρ₁_模型。"""
    Z = np.load(path)
    temps = [round(float(t), 4) for t in Z['temps']]
    assert expect_temps is None or temps == [round(float(t), 4) for t in expect_temps]
    cw, _ = arm_counts(Z['toks'], Z['U'], Z['pick'])
    tot = cw.sum(1)
    win = [temps.index(round(float(t), 4)) for t in pred['window']]
    lcp = liq_cry_pick_from(Z['toks'], Z['pick'])
    tg, Tw = toy_targets(tot, temps, win, pred['Tm0'], pred['b'], lcp)
    par, cost, res = fit_toy(tg, Tw)
    r, D, P = toy_rho1(par, round(float(pred['target']['hstar']), 4), [float(t) for t in pred['target']['grid']])
    return dict(targets=tg, window=Tw, params=par, cost=cost, resid=res, rho1_toy=r, D=D, P=P)
