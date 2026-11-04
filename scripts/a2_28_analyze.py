"""A2-28 の分析（CPU）。ゲート：docs/A2_28_GATE.md。
XQ26 設計のデータだけでの予言の比 ρ_out、XQ27 直線を使わない T_× の行きすぎの比 ρ_loc、XQ28 s_N の共分散の取り方と長さへの依存。
名前空間は a2_27_run.load27ns() の上で使う（counts10・tot・lnK_of・use_of・g_from・fit_vh・boot など）。"""
import numpy as np

A28_SEED, A28_BOOT = 20261128, 2000
A28_BAND = (0.8, 1.25)
A28_FAIL = 0.10
A28_LCUT = 256
A28_LREC = 384
A28_SN_BAND = (0.005, 0.02)   # XQ28：「約 1%」と言える幅（s_N の中央値の区間がこの内）
A28_EXTRAP = 0.05             # XQ27：格子の端から外へ内挿してよい幅（温度で 1 目盛り）


# ---------- XQ26：設計のデータだけでの予言 ----------

def rho_out(c0, c1, target, w, use):
    """Σ_u Δ ÷ (n_u × target)。target ＝ h ḡ（h は生成に使った、4 桁に丸めた h*。ḡ は設計の数。どちらも前もって決まった数）。→ (ρ_out, use)。"""
    if use is None or len(use) < A18_MIN_T:
        return np.nan, use
    t0, t1 = tot(c0, w), tot(c1, w)
    D = lnK_of(t1) - lnK_of(t0)
    u = np.array(use, int)
    if not np.all(np.isfinite(D[u])) or target == 0 or not np.isfinite(target):
        return np.nan, use
    return float(D[u].sum() / (len(u) * target)), use


# ---------- XQ27：直線を使わない T_× ----------

def cross_local(x, y, ref, extrap=None):
    """x ＝ 1/T、y ＝ ln K（使う温度だけ）。ln K が 1/T について上がる向き（y_i < 0 ≤ y_{i+1}、x 昇順）に 0 をまたぐ
    隣り合う 2 点の間で、1/T について直線に内挿した根の T（またぐ所が 2 つ以上なら T が ref にいちばん近いもの）。
    またぐ所がなく、extrap（温度の幅）を渡せば：端の 2 点の直線を外へ延ばした根が、端の温度から extrap 以内ならそれを使う
    （2 点の傾きが正のときだけ）。それもなければ nan。→ (T, 外挿したか)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x); x, y = x[o], y[o]
    roots = []
    for i in range(len(x) - 1):
        if np.isfinite(y[i]) and np.isfinite(y[i + 1]) and y[i] < 0 <= y[i + 1]:
            roots.append(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    if roots:
        T = 1.0 / np.array(roots)
        return float(T[np.argmin(np.abs(T - ref))]), False
    if extrap is None or len(x) < 2 or not np.all(np.isfinite(y)):
        return np.nan, False
    cands = []
    for (i0, i1) in ((0, 1), (len(x) - 2, len(x) - 1)):
        sl = (y[i1] - y[i0]) / (x[i1] - x[i0])
        if sl <= 0:
            continue
        xr = x[i0] - y[i0] / sl
        Tr, Tedge = 1.0 / xr, (1.0 / x[i0] if i0 == 0 else 1.0 / x[i1])
        if xr > 0 and abs(Tr - Tedge) <= extrap + 1e-12 and (xr < x[0] or xr > x[-1]):
            cands.append(Tr)
    if not cands:
        return np.nan, False
    cands = np.array(cands)
    return float(cands[np.argmin(np.abs(cands - ref))]), True


def quad_root(x, y, wt, ref):
    """重みつきの 2 次式 ln K ＝ a ＋ b x ＋ c x² の根（x ＝ 1/T）。使う温度の範囲の内にある実の根のうち、T が ref に近いもの。なければ nan。"""
    x, y, wt = np.asarray(x, float), np.asarray(y, float), np.asarray(wt, float)
    if len(x) < 4:
        return np.nan
    X = np.stack([np.ones_like(x), x, x * x], 1)
    a, b, c = np.linalg.solve(X.T @ (wt[:, None] * X), X.T @ (wt * y))
    r = np.roots([c, b, a]) if abs(c) > 1e-12 else np.array([-a / b])
    r = r[np.isreal(r)].real
    r = r[(r >= x.min()) & (r <= x.max())]
    if not len(r):
        return np.nan
    T = 1.0 / r
    return float(T[np.argmin(np.abs(T - ref))])


def tm_three(cw, temps, w, ref, use=None):
    """窓の数え cw（nT, nD, ≥10）→ (直線の T_×, T_loc, 2 次の根, 外挿したか, 使う温度)。
    使う温度は fit_vh の決め方（入る・出るが 10 以上）。use を渡せばそれに固定する。ref は根を選ぶ目安（前もって決まった温度）。"""
    g = fit_vh.__globals__; old = g['A8_TU']; g['A8_TU'] = [float(t) for t in temps]
    try:
        tt = np.einsum('kjc,j->kc', cw, w)[:, :10]
        f = fit_vh(tt, temps_use=use)
    finally:
        g['A8_TU'] = old
    idx = f['use_idx']
    if not f.get('ok'):
        return np.nan, np.nan, np.nan, False, idx
    Tm = float(f['Tm']) if np.isfinite(f.get('Tm', np.nan)) else np.nan
    x = 1.0 / np.array(temps, float)[idx]; y = np.array(f['lnK'])[idx]
    n01, n10 = tt[idx, 1], tt[idx, 3]
    wt = 1.0 / (1.0 / n01 + 1.0 / n10)
    Tl, ex = cross_local(x, y, ref, A28_EXTRAP)
    return Tm, Tl, quad_root(x, y, wt, ref), ex, idx


def overshoot(cw0, cw1, temps, w, refs, use=None):
    """refs ＝ (罰なしの腕の目安 ＝ 設計の T_×、罰ありの腕の目安 ＝ T*)。→ dict（直線・内挿・2 次それぞれの比 ＝ (T(h*) − T(0)) ÷ −0.10）と、使う温度の組。"""
    u0, u1 = (None, None) if use is None else use
    a0 = tm_three(cw0, temps, w, refs[0], u0); a1 = tm_three(cw1, temps, w, refs[1], u1)
    r = lambda i: (a1[i] - a0[i]) / -0.10
    return dict(lin=r(0), loc=r(1), quad=r(2), T0=list(a0[:3]), T1=list(a1[:3]), extrap=(bool(a0[3]), bool(a1[3]))), (a0[4], a1[4])


# ---------- XQ28：s_N ----------

def s_N_of(ev, N, Q, temps, w, use, within=True):
    """s_N ＝ Σ_u g ÷ Σ_u g_N、g ＝ g_N − g_Q（A2-27 の factors と同じ式。within で共分散の取り方）。"""
    if use is None or len(use) < A18_MIN_T:
        return np.nan
    gN = g_from(ev, N, temps, w, within=within); gQ = g_from(ev, Q, temps, w, within=within)
    u = np.array(use, int)
    den = gN[u].sum()
    if not np.isfinite(den) or den == 0:
        return np.nan
    return float((gN - gQ)[u].sum() / den)


def s_N_mixed(ev, N, Q, temps, w, use):
    """取り方の混ざった比：Σ_u g（まとめた共分散、前もっての確かめの g と同じ）÷ Σ_u g_N（記事の中、S/(S − 1) 倍）。"""
    if use is None or len(use) < A18_MIN_T:
        return np.nan
    g = g_from(ev, N - Q, temps, w, within=False)
    gN = g_from(ev, N, temps, w, within=True)
    u = np.array(use, int)
    den = gN[u].sum()
    if not np.isfinite(den) or den == 0:
        return np.nan
    return float(g[u].sum() / den)


def trunc_parts(C, U, pick, q, L):
    """続きの最初の L 歩だけで、数え（counts10）・N・Q を出し直す。"""
    c = counts10(C[..., :L], U[..., :L], pick[..., :L])
    return c, pick[..., :L].astype(float).sum(-1), q[..., :L].astype(float).sum(-1)


# ---------- 言葉 ----------

def word_band28(point, ci, fail, words, neg=None):
    """words ＝ (帯の内, 上, 下)。neg を渡せば点が 0 未満でその言葉。上から順：はっきりしない → 逆向き → 帯 → 上 → 下 → はっきりしない。"""
    lo, hi = A28_BAND
    if fail > A28_FAIL or not np.isfinite(point) or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if neg is not None and point < 0:
        return neg
    if lo <= ci[0] and ci[1] <= hi:
        return words[0]
    if ci[0] > hi:
        return words[1]
    if ci[1] < lo:
        return words[2]
    return 'はっきりしない'


W26 = ('予言どおり（2 割の内）', '予言より大きく動く', '予言より小さく動く')
W27 = ('狙いどおり（2 割の内）', '狙いより大きく動く', '狙いより小さく動く')
W28 = ('約 1% の内（0.5〜2%）', '2% より大きい', '0.5% より小さい')


def word28(point, ci, fail):
    lo, hi = A28_SN_BAND
    if fail > A28_FAIL or not np.isfinite(point) or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if lo <= ci[0] and ci[1] <= hi:
        return W28[0]
    if ci[0] > hi:
        return W28[1]
    if ci[1] < lo:
        return W28[2]
    return 'はっきりしない'


# ---------- 自己テスト ----------

def self_test28(seed=11):
    rng = np.random.default_rng(seed)
    temps = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    x = 1 / np.array(temps); a, b = -5.0, 2.75
    # (1) 直線の ln K：内挿の根 ＝ 直線の根
    y = a + b * x; Tlin = -b / a
    assert abs(cross_local(x, y, Tlin)[0] - Tlin) < 1e-9, 'self-test：直線で内挿の根がずれる'
    # (2) 曲がった ln K：内挿の根は本当の根に、直線の当てはめの根より近い。2 次の根は本当の根
    c = -0.6
    yc = a + b * x + c * (x - x.mean()) ** 2
    rt = [r.real for r in np.roots([c, b - 2 * c * x.mean(), a + c * x.mean() ** 2]) if np.isreal(r) and x.min() <= r.real <= x.max()]
    Ttrue = 1 / rt[0]
    X = np.stack([np.ones_like(x), x], 1); a1, b1 = np.linalg.lstsq(X, yc, rcond=None)[0]
    assert abs(cross_local(x, yc, Ttrue)[0] - Ttrue) < abs(-b1 / a1 - Ttrue), 'self-test：曲がりで内挿が直線より悪い'
    assert abs(quad_root(x, yc, np.ones_like(x), Ttrue) - Ttrue) < 1e-9, 'self-test：2 次の根が本当の根と合わない'
    # (3) 上がる向きにまたぐ所が 2 つ：ref に近いほう。下がる向きのまたぎは使わない。またがなければ nan
    xs = np.sort(x); y2 = np.array([-1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])   # x の昇順：上がる向きのまたぎが 2 つ（0→1 と 2→3）
    T01, T23 = 1 / (0.5 * (xs[0] + xs[1])), 1 / (0.5 * (xs[2] + xs[3]))
    assert abs(cross_local(xs, y2, T01)[0] - T01) < 1e-9 and abs(cross_local(xs, y2, T23)[0] - T23) < 1e-9, 'self-test：根の選び方'
    ydown = -(a + b * x)
    assert np.isnan(cross_local(x, ydown, 0.55)[0]), 'self-test：下がる向きのまたぎを使った'
    assert np.isnan(cross_local(x, np.abs(y) + 1, 0.55)[0]), 'self-test：根がないのに nan でない'
    # (4) 外挿：端から 1 目盛りの内なら使い、外なら nan
    yE = (a + 0.6) + b * x                                              # 根が端の少し外
    TE = -b / (a + 0.6)
    tE, exE = cross_local(x, yE, TE, A28_EXTRAP)
    assert (exE and abs(tE - TE) < 1e-9) if abs(TE - max(temps)) <= A28_EXTRAP and TE > max(temps) else True, 'self-test：外挿'
    yF = (a + 3.0) + b * x
    assert np.isnan(cross_local(x, yF, 1.0, A28_EXTRAP)[0]), 'self-test：遠すぎる外挿を使った'
    # (5) tm_three：作り物の数えで、直線の ln K なら T_loc が当てはめの T_× と 1e-3 で合い、A8_TU が戻る
    nD = 5
    lnK = a + b * x
    cw = np.zeros((7, nD, 16))
    for k in range(7):
        N0, N1, n10 = 4000.0, 4000.0, 200.0
        n01 = n10 * np.exp(lnK[k]) * N0 / N1
        cw[k, :, 0], cw[k, :, 1], cw[k, :, 2], cw[k, :, 3] = N0 / nD, n01 / nD, N1 / nD, n10 / nD
        cw[k, :, 4:10] = 1.0
    old = list(fit_vh.__globals__['A8_TU'])
    Tm, Tl, Tq, ex, idx = tm_three(cw, temps, np.ones(nD), Tlin)
    assert abs(Tm - Tlin) < 1e-3 and abs(Tl - Tlin) < 1e-3 and not ex, 'self-test：tm_three'
    assert list(fit_vh.__globals__['A8_TU']) == old, 'self-test：A8_TU が戻らない'
    # (6) 切り詰め：全長なら元と同じ。写しの印は前しか見ない
    nT, S, nD2, L = 2, 2, 3, 120
    C = (rng.random((nT, S, nD2, L)) < 0.4).astype(np.int8)
    U = rng.random((nT, S, nD2, L)); pick = rng.random((nT, S, nD2, L)) < 0.5; q = rng.random((nT, S, nD2, L))
    c_full = counts10(C, U, pick)
    c_tr, N_tr, Q_tr = trunc_parts(C, U, pick, q, L)
    assert np.array_equal(c_full, c_tr) and np.allclose(N_tr, pick.sum(-1)) and np.allclose(Q_tr, q.sum(-1)), 'self-test：切り詰めの全長が元と違う'
    toks = rng.integers(0, 6, (1, 2, 2, 80))
    Cm = marks_of(toks)
    assert np.array_equal(marks_of(toks[..., :50]), Cm[..., :50]), 'self-test：写しの印が後ろを見ている'
    # (7) s_N：作り物で、まとめた共分散と記事の中の共分散の式を手で確かめる
    ev = rng.random((1, 4, 6, 4)) * 10 + 1; Nn = rng.random((1, 4, 6)) * 5; Qn = Nn * 0.5
    w6 = np.ones(6)
    sN = s_N_of(np.repeat(ev, A18_MIN_T, 0), np.repeat(Nn, A18_MIN_T, 0), np.repeat(Qn, A18_MIN_T, 0), [0.5] * A18_MIN_T, w6, list(range(A18_MIN_T)))
    assert abs(sN - 0.5) < 1e-9, 'self-test：s_N（Q ＝ N/2 なら 0.5）'
    sM = s_N_mixed(np.repeat(ev, A18_MIN_T, 0), np.repeat(Nn, A18_MIN_T, 0), np.repeat(Qn, A18_MIN_T, 0), [0.5] * A18_MIN_T, w6, list(range(A18_MIN_T)))
    gp = g_from(ev, Nn - Qn, [0.5], w6, within=False)[0]; gw = g_from(ev, Nn, [0.5], w6, within=True)[0]
    assert abs(sM - gp / gw) < 1e-9, 'self-test：混ざった比'
    # (8) rho_out：手で計算した値
    t0 = np.zeros((7, 1, 1, 4)); t1 = np.zeros((7, 1, 1, 4))
    t0[..., :] = [100, 10, 100, 10]; t1[..., :] = [100, 5, 100, 10]
    r, _ = rho_out(t0, t1, np.log(0.5) * 1.0, np.ones(1), list(range(7)))
    assert abs(r - 1.0) < 1e-9, 'self-test：rho_out'
    # (9) 言葉の順
    assert word_band28(1.0, [0.9, 1.1], 0.0, W27) == W27[0]
    assert word_band28(1.6, [1.3, 1.9], 0.0, W27) == W27[1]
    assert word_band28(0.5, [0.4, 0.7], 0.0, W27) == W27[2]
    assert word_band28(1.0, [0.7, 1.3], 0.0, W27) == 'はっきりしない'
    assert word_band28(1.0, [0.9, 1.1], 0.2, W27) == 'はっきりしない'
    assert word_band28(-0.3, [-0.5, -0.1], 0.0, W26, neg='逆向き') == '逆向き'
    assert word28(0.011, [0.009, 0.013], 0.0) == W28[0] and word28(0.03, [0.025, 0.04], 0.0) == W28[1]
    assert word28(0.003, [0.002, 0.004], 0.0) == W28[2] and word28(0.01, [0.004, 0.015], 0.0) == 'はっきりしない'
    return 'OK'
