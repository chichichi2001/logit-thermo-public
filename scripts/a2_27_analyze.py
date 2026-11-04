"""A2-27 の分析（CPU）。ゲート：docs/A2_27_GATE.md。scripts/a2_26_run.py の load26ns() の名前空間で実行する
（counts10・tot・lnK_of・eq_pred・use_of・ratios・g_identity・g_from・boot・agg・wfit・A18_MIN_EV・A18_MIN_T など）。
XQ24 温度の恒等式（スコア −(U − U_c)）は、温度を変えたときの ln K の動きを予言するか。XQ25 平衡の読みの言いすぎは、エネルギーの取り方によらないか。
"""
import numpy as np

A27_SEED, A27_BOOT, A27_FAIL = 20261127, 1000, 0.10
A27_BAND = (0.8, 1.25)
A27_CONV = ('U', 'negz', 'TneglogpT')


# ---------- 数え：状態ごとのエネルギーの和を、取り方を変えて ----------

def energy_of(conv, Z27, U, temps):
    """取り方 → 歩ごとのエネルギー (nT, S, nD, L)。U は保存してある −log p_1、Z27 は採点し直しの量。"""
    if conv == 'U':
        return U
    if conv == 'negz':
        return -Z27['zx'].astype(np.float64)
    if conv == 'TneglogpT':                                             # T × (−log p_T) ＝ −z ＋ T lse_T（−z ＋ 歩ごとの定数の形。−log p_T そのままは β 倍ずれる。レビュー 1）
        T = np.array(temps, float)[:, None, None, None]
        return -Z27['zx'].astype(np.float64) + T * Z27['lseT'].astype(np.float64)
    raise ValueError(conv)


def with_energy(c0, C0, E, pick):
    """c0（counts10 の数え）の状態ごとのエネルギーの和（4〜7）だけを、エネルギー E で数え直したもの。"""
    c = c0.copy()
    c[..., 4:10] = counts10(C0, E, pick)[..., 4:10]
    return c


def rho_E_conv(cc0, c1, temps, h, w, use=None):
    """XQ25 の ρ_E：A2-26 の ratios と同じ式だが、ΔE ≤ 0 でも n ＝ b/ΔE をそのまま使う（向きが逆になるのを見るため。レビュー 2）。→ (ρ_E, use)。"""
    t0, t1 = tot(cc0, w), tot(c1, w)
    if use is None:
        use = use_of(t0, t1)
    if len(use) < A18_MIN_T:
        return np.nan, use
    u = np.array(use, int); T = np.array(temps, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dc = t0[:, 9] / t0[:, 7] - t0[:, 8] / t0[:, 5]
        dE = float(np.mean((t0[:, 4] / t0[:, 5] - t0[:, 6] / t0[:, 7])[u]))
    lk0 = lnK_of(t0)
    wt = 1.0 / (1.0 / t0[u, 1] + 1.0 / t0[u, 3])
    b = wfit(1.0 / T[u], lk0[u], wt, 1)[1]
    D = lnK_of(t1) - lk0
    if dE == 0 or not np.isfinite(dE):
        return np.nan, use
    E = h * (b / dE) * dc / T
    if not (np.all(np.isfinite(D[u])) and np.all(np.isfinite(E[u]))) or E[u].sum() == 0:
        return np.nan, use
    return float(D[u].sum() / E[u].sum()), use


# ---------- XQ24：温度の恒等式 ----------

def score_beta(Z27):
    """文ごとの β のスコア −Σ_t (U_t(x_t) − E_t[U_t]) (nT, S, nD)。"""
    return -(Z27['Ux'].astype(np.float64) - Z27['Uc'].astype(np.float64)).sum(-1)


def y_identity(c0, sb, temps, w, within=False):
    """温度ごとの予言 ∂ ln K/∂β（罰の g と同じ組み立て）。g_identity・g_from は Cov/(T E) なので、T を掛けて戻す。"""
    T = np.array(temps, float)
    if within:
        return T * g_from(c0[..., :4], sb, temps, w, within=True, unbiased=True)
    return T * np.array(g_identity(c0[..., :4], sb, temps, w))


def use_c0(t0):
    return [k for k in range(t0.shape[0]) if min(t0[k, 1], t0[k, 3]) >= A18_MIN_EV]


def rho_beta(c0, sb, temps, w, use=None):
    """→ (ρ_β, use)：使う温度の両端の ln K の差 ÷ 予言 y の台形の積分（β ＝ 1/T で）。"""
    t0 = tot(c0, w)
    if use is None:
        use = use_c0(t0)
    if len(use) < A18_MIN_T:
        return np.nan, use
    u = np.array(sorted(use, key=lambda k: 1.0 / temps[k]), int)       # β の小さい順
    beta = 1.0 / np.array(temps, float)[u]
    lk = lnK_of(t0)[u]; y = y_identity(c0, sb, temps, w)[u]
    if not (np.all(np.isfinite(lk)) and np.all(np.isfinite(y))):
        return np.nan, use
    pred = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(beta)))
    if pred == 0:
        return np.nan, use
    return float((lk[-1] - lk[0]) / pred), use


# ---------- 記録：3 つの因子 ----------

def factors(c0, conv_c0, sig_N, sig_Q, sb, Utot, temps, w, use):
    """s_N、s_U（局所：Σy/ΣΔH_全体、と THEORY §6 の足した版：Σg_C/Σg_C全体）、κ と、平衡の読みに対する比 g/g_C（記事の中の共分散でそろえる）。
    ratio ＝ (s_N/s_U_sum) κ が恒等的に成り立つ。s_U_local では成り立たない（Δc/T が温度で変わるため）。conv_c0 はエネルギーの取り方で数え直した c0、Utot はその取り方の文ごとの和。"""
    u = np.array(use, int); T = np.array(temps, float)
    ev = c0[..., :4]
    gN = g_from(ev, sig_N, temps, w); gQ = g_from(ev, sig_Q, temps, w); g = gN - gQ
    y = y_identity(c0, sb, temps, w, within=True)
    dHw = -T * g_from(ev, Utot, temps, w)                              # 文全体を e^{−δ ΣE} で傾けた ∂ ln K/∂δ（A2-21 の ΔH_全体 と同じ向き）
    t0 = tot(conv_c0, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        dc = t0[:, 9] / t0[:, 7] - t0[:, 8] / t0[:, 5]
        dE = float(np.mean((t0[:, 4] / t0[:, 5] - t0[:, 6] / t0[:, 7])[u]))
    gCw = dHw / dE * dc / T
    _, info = eq_pred(t0, temps, 1.0, use)
    gC = info['n'] * dc / T
    gCloc = y / dE * dc / T
    S = lambda a: float(np.sum(a[u]))
    return dict(s_N=S(g) / S(gN), s_U_local=S(y) / S(dHw), s_U_sum=S(gC) / S(gCw), kappa=S(gN) / S(gCw), ratio=S(g) / S(gC), ratio_local=S(g) / S(gCloc), dE=dE, n=info['n'])


# ---------- 言葉 ----------

def word24(pt, ci, fail):
    lo, hi = A27_BAND
    if fail > A27_FAIL or not (np.isfinite(pt) and np.all(np.isfinite(ci))):
        return 'はっきりしない'
    if pt < 0:
        return '逆向き'
    if lo <= ci[0] and ci[1] <= hi:
        return '予言どおり（2 割の内）'
    if ci[0] > hi:
        return '予言より大きく動く'
    if ci[1] < lo:
        return '予言より小さく動く'
    return 'はっきりしない'


def word25(rows):
    """rows：取り方ごとの dict(point, ci, fail)。並びの後ろの言葉を優先（ゲート §3）。"""
    if any(r['fail'] > A27_FAIL or not np.isfinite(r['point']) or not np.all(np.isfinite(r['ci'])) for r in rows):
        return 'はっきりしない'
    if any(r['point'] < 0 for r in rows):
        return '向きが逆になる取り方がある'
    if any(r['point'] >= 0.8 for r in rows):
        return '言いすぎでない取り方がある'
    if all(r['ci'][1] < 0.8 for r in rows):
        return 'どの取り方でも言いすぎ'
    return '点はどれも 0.8 未満'


# ---------- 自己テスト ----------

def self_test27():
    rng = np.random.default_rng(5)
    # (1) with_energy：E ＝ U なら c0 と同じ
    toks = rng.integers(0, 6, (3, 2, 4, 512)).astype(np.int64)
    toks[0, 0, 0, 200:] = np.tile([1, 2, 3, 4], 78)[:312]
    C = marks_of(toks); U = rng.random(toks.shape); pick = rng.random(toks.shape) < 0.3
    c0 = counts10(C, U, pick)
    assert np.allclose(with_energy(c0, C, U, pick), c0, rtol=0, atol=1e-9)
    c2 = with_energy(c0, C, U + 1.0, pick)
    assert np.allclose(c2[..., 4] - c0[..., 4], c0[..., 5]) and np.array_equal(c2[..., :4], c0[..., :4])
    # (2) energy_of：−log p_T ＝ −z/T ＋ lseT
    Z = dict(zx=np.full((3, 1, 1, 2), 2.0), lseT=np.full((3, 1, 1, 2), 1.0))
    e = energy_of('TneglogpT', Z, None, [0.5, 1.0, 2.0])
    assert np.allclose(e[:, 0, 0, 0], [-1.5, -1.0, 0.0])                # −z ＋ T lse_T
    # (2b) rho_E_conv：ΔE が正なら A2-26 の ratios の ρ_E と同じ値、ΔE の向きを逆にすると符号が逆（作り物の合計で、値が出ることも確かめる）
    tt7 = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]; Tq = np.array(tt7)
    lk0 = -3 + 1.6 / Tq
    def mkc(lk, uo, ur):
        c = np.zeros((7, 1, 1, 10))
        c[:, 0, 0] = np.stack([np.full(7, 1e6), 1e5 * np.exp(lk), np.full(7, 1e6), np.full(7, 1e5), np.full(7, uo), np.full(7, 1e6), np.full(7, ur), np.full(7, 1e6), np.full(7, 2e5), np.full(7, 9e5)], 1)
        return c
    a0, a1 = mkc(lk0, 1.5e6, 0.5e6), mkc(lk0 - 0.2, 1.5e6, 0.5e6)
    rA = ratios(a0, a1, np.zeros((7, 1, 1)), tt7, -0.3, np.ones(1))[0]
    rB = rho_E_conv(a0, a1, tt7, -0.3, np.ones(1))[0]
    want = (-0.2 * 7) / np.sum(-0.3 * 1.6 / 1.0 * 0.7 / Tq)
    assert np.isfinite(rB) and abs(rA - rB) < 1e-9 and abs(rB - want) < 1e-9, (rA, rB, want)
    rN = rho_E_conv(mkc(lk0, 0.5e6, 1.5e6), a1, tt7, -0.3, np.ones(1))[0]
    assert abs(rN + rB) < 1e-9, (rB, rN)
    assert np.isnan(ratios(mkc(lk0, 0.5e6, 1.5e6), a1, np.zeros((7, 1, 1)), tt7, -0.3, np.ones(1))[0])   # A2-26 の ratios は ΔE ≤ 0 で出ない
    # (3) rho_beta：ln K が β に線形で、y がその傾きなら比 1（台形は線形で厳密）。y_identity を差しかえて確かめる
    temps = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]; beta = 1 / np.array(temps)
    lk = -3 + 1.5 * beta
    cc = np.zeros((7, 1, 1, 10)); cc[:, 0, 0, :4] = np.stack([np.full(7, 1e7), 1e6 * np.exp(lk), np.full(7, 1e7), np.full(7, 1e6)], 1)
    g0 = globals()['y_identity']
    try:
        globals()['y_identity'] = lambda c, sb, t, w, within=False: np.full(len(t), 1.5)
        r, _ = rho_beta(cc, None, temps, np.ones(1))
        assert abs(r - 1) < 1e-9, r
        globals()['y_identity'] = lambda c, sb, t, w, within=False: np.full(len(t), 3.0)
        r, _ = rho_beta(cc, None, temps, np.ones(1))
        assert abs(r - 0.5) < 1e-9, r
    finally:
        globals()['y_identity'] = g0
    # (4) y_identity は、温度の向きが罰と同じ組み立て：スコアを入れかえると g_identity × T
    ev = rng.integers(1, 50, (2, 3, 5, 4)).astype(float); sb = rng.normal(0, 1, (2, 3, 5)); tt = [0.5, 0.7]
    cz = np.zeros((2, 3, 5, 10)); cz[..., :4] = ev
    assert np.allclose(y_identity(cz, sb, tt, np.ones(5)), np.array(tt) * np.array(g_identity(ev, sb, tt, np.ones(5))))
    # (5) 言葉
    assert word24(1.0, [0.9, 1.1], 0) == '予言どおり（2 割の内）' and word24(0.5, [0.4, 0.6], 0) == '予言より小さく動く' and word24(-0.2, [-0.5, 0.1], 0) == '逆向き'
    R = lambda p, lo, hi, f=0.0: dict(point=p, ci=[lo, hi], fail=f)
    assert word25([R(0.5, 0.4, 0.6)] * 3) == 'どの取り方でも言いすぎ'
    assert word25([R(0.5, 0.4, 0.6), R(0.7, 0.5, 0.9), R(0.5, 0.4, 0.6)]) == '点はどれも 0.8 未満'
    assert word25([R(0.5, 0.4, 0.6), R(0.9, 0.5, 1.2), R(0.5, 0.4, 0.6)]) == '言いすぎでない取り方がある'
    assert word25([R(0.5, 0.4, 0.6), R(-0.9, -1.5, 1.2), R(0.9, 0.4, 1.6)]) == '向きが逆になる取り方がある'
    assert word25([R(0.5, 0.4, 0.6, 0.2)] * 3) == 'はっきりしない'
    return 'OK'
