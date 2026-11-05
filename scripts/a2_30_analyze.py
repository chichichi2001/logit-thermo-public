"""A2-30 の分析（CPU、探索）。ゲート：docs/A2_30_GATE.md。scripts/a2_30_run.py の load30ns() の名前空間で実行する
（A2-18 の seq_events・lnK_from・g_identity・rho_model・word18・A18_MIN_EV・A18_MIN_T、A2-24 の marks_of などがある）。
XQ32 登録した判定の頑丈さ（10,000 回 × 5 つの種、BCa）、XQ33 予言の当たり方（モデルごと・またいだ平均・差分の傾き）、
XQ34 設計の ḡ もゆらした ρ_out、XQ35 繰り返しの割合への予言。R8〜R10 の記録の部品。"""
import numpy as np
from statistics import NormalDist

A30_SEEDS = (20261130, 20261131, 20261132, 20261133, 20261134)
A30_SEED33, A30_SEED34, A30_SEED35 = 20261135, 20261136, 20261137
A30_BOOT = 10000
A30_FAIL = 0.10
A30_BAND = (0.8, 1.25)
_N = NormalDist()


def agg30(v, how):
    """小さい 7 本は中央値（出ないモデル 2 本までは残りで）、7B の 2 本は平均（両方そろったときだけ）。"""
    v = np.array(v, float); ok = np.isfinite(v)
    if how == 'median':
        return float(np.median(v[ok])) if ok.sum() >= len(v) - 2 and ok.sum() > 0 else np.nan
    return float(np.mean(v)) if ok.all() else np.nan


def ci_of(bs):
    bs = np.array(bs, float)
    return [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))] if np.isfinite(bs).any() else [np.nan, np.nan]


def word_band30(ci, fail):
    """登録と同じ帯の言葉（A2-18 の word18 と同じ決まり）。"""
    if fail > A30_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if A30_BAND[0] <= ci[0] and ci[1] <= A30_BAND[1]:
        return '予言どおり（2 割の内）'
    if ci[0] > A30_BAND[1]:
        return '予言より大きく効く'
    if ci[1] < A30_BAND[0]:
        return '予言より小さく効く'
    return 'はっきりしない'


def resample(rng, nD):
    return np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)


# ---------- XQ32 ----------

def stat_rho(EE, names, arm, hs, grids, uses, w, how):
    return agg30([rho_model(EE[n], (arm, hs[n]), grids[n], w, uses[n])[0] for n in names], how)


def boot_stat(fn, nD, seed, n):
    """fn(w) → 統計。→ 選び直しの統計の列。"""
    rng = np.random.default_rng(seed)
    return np.array([fn(resample(rng, nD)) for _ in range(n)], float)


def bca_ci(fn, bs, nD, alpha=0.05):
    """BCa：偏り z0 は選び直しで点より小さい割合から、加速 a は記事を 1 本ずつ抜いた jackknife から。"""
    point = fn(np.ones(nD))
    b = bs[np.isfinite(bs)]
    if not np.isfinite(point) or len(b) == 0:
        return [np.nan, np.nan], dict(z0=np.nan, a=np.nan)
    p = np.clip(np.mean(b < point) + 0.5 * np.mean(b == point), 1.0 / len(b), 1 - 1.0 / len(b))
    z0 = _N.inv_cdf(p)
    jk = []
    for j in range(nD):
        w = np.ones(nD); w[j] = 0.0
        jk.append(fn(w))
    jk = np.array(jk, float); jk = jk[np.isfinite(jk)]
    d = jk.mean() - jk
    den = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _N.inv_cdf(q)
        out.append(float(np.nanpercentile(b, 100 * _N.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))
    return out, dict(z0=float(z0), a=a)


def xq32_word(words, registered):
    """words：{種: 言葉, 'bca': 言葉}。種で変われば「種で境目」、種ではすべて同じで BCa だけ変われば「区間の作り方で変わる」。"""
    changed = [k for k, v in words.items() if v != registered]
    if not changed:
        return '頑丈', changed
    if any(k != 'bca' for k in changed):
        return '種で境目', changed
    return '区間の作り方で変わる', changed


# ---------- XQ33・XQ34：設計の傾き ----------

def g_design(evd, sigd, temps, v):
    """設計の h ＝ 0 の腕 → 温度ごとの恒等式の g(T) と、その平均 ḡ（A2-18 の予言と同じ式。v は設計の記事の重み）。"""
    g = np.array(g_identity(evd, sigd, temps, v), float)
    return g, float(np.mean(g))


def fd_slope(evp, evm, w=None, use=None, dh=0.6):
    """設計の ±0.3 の腕 → 温度ごとの差分の傾きと、両方の腕で入る回・出る回が 10 以上の温度（全データで決めて固定）での平均。"""
    w = np.ones(evp.shape[2]) if w is None else w
    lp, fp, mp = lnK_from(evp, w); lm, fm, mm = lnK_from(evm, w)
    s = (lp - lm) / dh
    if use is None:
        use = [k for k in range(len(s)) if min(fp[k], mp[k], fm[k], mm[k]) >= A18_MIN_EV and np.isfinite(s[k])]
    v = s[np.array(use, int)] if use else np.array([])
    return s, use, (float(np.mean(v)) if len(v) and np.all(np.isfinite(v)) else np.nan)


def D_of(E, w):
    l0, _, _ = lnK_from(E['c0'][0], w); l1, _, _ = lnK_from(E['h1'][0], w)
    return l1 - l0


def rho_slope(D, use, h, slope):
    u = np.array(use, int)
    if len(u) < A18_MIN_T or not np.all(np.isfinite(D[u])) or not np.isfinite(slope) or slope == 0:
        return np.nan
    return float(D[u].sum() / (len(u) * h * slope))


def D_split(E, w0, w1):
    """2 つの腕を別々の重みで（R10）。"""
    l0, _, _ = lnK_from(E['c0'][0], w0); l1, _, _ = lnK_from(E['h1'][0], w1)
    return l1 - l0


def rho_split(E, h, temps, w0, w1, use):
    """R10：組にしない選び直し。D は腕ごとの重み、g は c0 の腕の重み。"""
    D = D_split(E, w0, w1); P = h * np.array(g_identity(E['c0'][0], E['c0'][1], temps, w0), float)
    u = np.array(use, int)
    if len(u) < A18_MIN_T or not (np.all(np.isfinite(D[u])) and np.all(np.isfinite(P[u]))) or P[u].sum() == 0:
        return np.nan
    return float(D[u].sum() / P[u].sum())


def loo_mean(gb, m):
    """1 本抜き：m 以外のモデルの ḡ の平均。"""
    return float(np.mean([v for k, v in gb.items() if k != m]))


def acc(rhos):
    """予言の当たり方：|ln ρ| の中央値（出ないモデル 2 本までは残りで）。"""
    r = np.array(rhos, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(r > 0, np.abs(np.log(r)), np.nan)
    return agg30(a, 'median')


def word33(ci, fail):
    if fail > A30_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if ci[0] > 0:
        return 'モデルごとの微分のほうがよく当てる'
    if ci[1] < 0:
        return 'モデルをまたいだ平均のほうがよく当てる'
    return '区別できない'


# ---------- XQ35：繰り返しの割合 ----------

def frac_parts(O0, O1, sig0, temps, h, w):
    """O：(nT, S, nD) 続きごとの繰り返しの割合。→ 温度ごとの実現の変化と、予言 h Cov(O, σ)/T（全サンプル・全記事でまとめた共分散、記事の重み w）。"""
    S = O0.shape[1]
    ww = np.repeat(w[None, :], S, 0).ravel(); W = ww.sum()
    dE, P = [], []
    for k, T in enumerate(temps):
        o0, o1, sg = O0[k].ravel(), O1[k].ravel(), sig0[k].ravel()
        m0, m1, ms = (ww * o0).sum() / W, (ww * o1).sum() / W, (ww * sg).sum() / W
        dE.append(m1 - m0)
        P.append(h * float((ww * (o0 - m0) * (sg - ms)).sum() / W) / T)
    return np.array(dE), np.array(P)


def rho_frac(O0, O1, sig0, temps, h, w, use):
    dE, P = frac_parts(O0, O1, sig0, temps, h, w)
    u = np.array(use, int)
    if len(u) < A18_MIN_T or P[u].sum() == 0:
        return np.nan
    return float(dE[u].sum() / P[u].sum())


# ---------- 記録 ----------

def rates_tau(ev, w):
    """→ 温度ごとの k01 ＝ 入る回/普通の歩、k10 ＝ 出る回/繰り返しの歩、τ ＝ 1/(k01 ＋ k10)。"""
    c = np.einsum('ksjc,j->kc', ev, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        k01, k10 = c[:, 1] / c[:, 0], c[:, 3] / c[:, 2]
        return k01, k10, 1.0 / (k01 + k10)


def logw_var(q, temps, h, use):
    """(h/T)² Σ_t q(1 − q) を続きごとに → 使う温度での中央値と四分位。"""
    v = []
    for k in use:
        qq = q[k].astype(float)
        v.append(((h / temps[k]) ** 2 * (qq * (1 - qq)).sum(-1)).ravel())
    v = np.concatenate(v)
    return dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))


def arm_corr(O0, O1, use):
    r = [float(np.corrcoef(O0[k].ravel(), O1[k].ravel())[0, 1]) for k in use]
    return dict(per_T=r, median=float(np.median(r)))


def closest(grid, T):
    """いちばん近い格子の温度（ちょうど真ん中なら低いほう）。"""
    g = np.array(sorted(grid), float)
    d = np.abs(g - T)
    return float(g[int(np.flatnonzero(np.isclose(d, d.min(), atol=1e-12))[0])])


def cv(v):
    v = np.array(v, float)
    return float(np.std(v) / np.mean(v))


def read_R8(c):
    return '説明と合う' if c < 0.3 else ('説明と合わない' if c > 0.6 else 'どちらとも言えない')


def dTdh_identity(g_T, temps, Tm, b):
    """恒等式の予言する T_× の動き：dT_×/dh ＝ g(T_×) T_×² / b（g は設計の g_T を温度で直線で内挿）。"""
    return float(np.interp(Tm, temps, g_T) * Tm ** 2 / b)


# ---------- 自己テスト ----------

def self_test30(seed=3):
    rng = np.random.default_rng(seed)
    # (a) 正規の作り物で BCa は百分位とほぼ同じ
    x = rng.normal(1.0, 0.3, 200)
    fn = lambda w: float((w * x).sum() / w.sum())
    bs = boot_stat(fn, len(x), 1, 2000)
    p, b = ci_of(bs), bca_ci(fn, bs, len(x))[0]
    assert abs(p[0] - b[0]) < 0.02 and abs(p[1] - b[1]) < 0.02, (p, b)
    # 歪んだ作り物で BCa は上に寄る（指数分布の平均）
    y = rng.exponential(1.0, 60)
    fy = lambda w: float((w * y).sum() / w.sum())
    by = boot_stat(fy, len(y), 2, 3000)
    py, bcy = ci_of(by), bca_ci(fy, by, len(y))[0]
    assert bcy[1] > py[1] and bcy[0] > py[0], (py, bcy)
    # (b) 繰り返しの割合の予言：O ＝ c ＋ β σ のとき、予言は h β Var(σ)/T
    nT, S, nD = 3, 4, 50
    sig = rng.normal(0, 2, (nT, S, nD)); O0 = 0.3 + 0.05 * sig
    dE, P = frac_parts(O0, O0 + 0.01, sig, [0.5, 0.6, 0.7], -0.4, np.ones(nD))
    for k, T in enumerate([0.5, 0.6, 0.7]):
        assert abs(P[k] - (-0.4) * 0.05 * np.var(sig[k].ravel()) / T) < 1e-12 and abs(dE[k] - 0.01) < 1e-12
    # (c) 1 本抜きは自分を含まない
    gb = {'a': 1.0, 'b': 2.0, 'c': 4.0}
    assert loo_mean(gb, 'a') == 3.0 and loo_mean(gb, 'c') == 1.5
    # (d) 言葉の境目
    assert word_band30([0.8, 1.25], 0.0) == '予言どおり（2 割の内）' and word_band30([0.79, 1.1], 0.0) == 'はっきりしない'
    assert word_band30([1.26, 1.5], 0.0) == '予言より大きく効く' and word_band30([0.5, 0.79], 0.0) == '予言より小さく効く'
    assert word_band30([0.9, 1.1], 0.11) == 'はっきりしない'
    assert word33([0.01, 0.2], 0) == 'モデルごとの微分のほうがよく当てる' and word33([-0.2, -0.01], 0) == 'モデルをまたいだ平均のほうがよく当てる'
    assert word33([-0.1, 0.1], 0) == '区別できない' and word33([0.1, 0.2], 0.2) == 'はっきりしない'
    assert xq32_word({'s1': 'x', 'bca': 'x'}, 'x') == ('頑丈', []) and xq32_word({'s1': 'x', 'bca': 'y'}, 'x') == ('区間の作り方で変わる', ['bca'])
    assert xq32_word({'s1': 'y', 'bca': 'x'}, 'x')[0] == '種で境目' and xq32_word({'s1': 'y', 'bca': 'y'}, 'x')[0] == '種で境目'
    assert closest([0.5, 0.6], 0.55) == 0.5 and closest([0.5, 0.6], 0.56) == 0.6
    assert read_R8(0.29) == '説明と合う' and read_R8(0.61) == '説明と合わない' and read_R8(0.45) == 'どちらとも言えない'
    assert abs(dTdh_identity([1.0, 3.0], [0.4, 0.6], 0.5, 2.0) - 2.0 * 0.25 / 2.0) < 1e-12
    assert agg30([1, 2, np.nan, np.nan, 3, 4, 5], 'median') == 3.0 and np.isnan(agg30([1, np.nan, np.nan, np.nan, 3, 4, 5], 'median'))
    assert np.isnan(agg30([1, np.nan], 'mean')) and agg30([1, 3], 'mean') == 2.0
    assert abs(acc([np.e, 1 / np.e, 1.0]) - 1.0) < 1e-12
    return 'OK'
