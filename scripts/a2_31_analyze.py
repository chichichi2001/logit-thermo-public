"""A2-31 の分析の部品。ゲート：docs/A2_31_GATE.md。a2_30_run.load30ns() の名前空間の中で exec する
（A2-18 の seq_events・lnK_from・rho_model、A2-30 の agg30・ci_of・word_band30・resample がある）。
"""
import numpy as np
from scipy.stats import chi2, spearmanr

A31_SEED36, A31_SEED37, A31_SEED_R17A, A31_SEED_R17B = 20261140, 20261141, 20261142, 20261143
A31_BOOT = 10000
A31_FAIL = 0.10
A31_P = 0.05
A31_COND = 1e6
A31_ESS_MIN = 30
A31_RESID = 2.0


# ---------- XQ37：文の確率の比そのもので重みをつける ----------

def logw_of(pick, q, h, T):
    """pick・q：(S, nD, L) → 続きごとの log w（罰ありの文の確率 ÷ 罰なしの文の確率）(S, nD)。"""
    q = q.astype(float); c = pick.astype(float); a = h / T
    return (a * c - np.log1p(q * np.expm1(a))).sum(-1)


def logw_check(pick, q, h, T):
    """同じ量を歩ごとの確率の対数の差として書いたもの（前提の確かめ用）。"""
    q = q.astype(float); c = pick.astype(bool); a = h / T
    Z = 1 - q + q * np.exp(a)
    lp1 = np.where(c, np.log(q * np.exp(a) / Z), np.log((1 - q) / Z))
    lp0 = np.where(c, np.log(q), np.log(1 - q))
    return (lp1 - lp0).sum(-1)


def lnK_w(ev_k, W):
    """ev_k：(S, nD, 4)、W：(S, nD) の重み → ln K。"""
    c = np.einsum('sjc,sj->c', ev_k, W)
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.log(c[1] / c[0]) - np.log(c[3] / c[2]))


def rho_IS(E, LW, use, a):
    """E：{'c0': (ev, sig), 'h1': …}、LW：(7, S, nD) の log w、a：記事の重み (nD)。→ (ρ_IS, Σ D, Σ Δ_IS)。"""
    l0, _, _ = lnK_from(E['c0'][0], a)
    l1, _, _ = lnK_from(E['h1'][0], a)
    sD = sIS = 0.0
    for k in use:
        lw = LW[k] - LW[k].max()
        W = np.exp(lw) * a[None, :]
        sD += l1[k] - l0[k]
        sIS += lnK_w(E['c0'][0][k], W) - l0[k]
    if not (np.isfinite(sD) and np.isfinite(sIS)) or sIS == 0:
        return np.nan, sD, sIS
    return float(sD / sIS), float(sD), float(sIS)


def ess_of(lw):
    lw = lw.ravel() - lw.max(); w = np.exp(lw)
    return float(w.sum() ** 2 / (w ** 2).sum())


# ---------- XQ36：モデルの間のばらつき ----------

def cochran(x, Sig):
    """一般化した Cochran の Q（GLS の平均のまわり）。→ dict。"""
    x = np.asarray(x, float); M = len(x)
    cond = float(np.linalg.cond(Sig))
    if not np.isfinite(cond) or cond > A31_COND:
        return dict(Q=np.nan, df=M - 1, p=np.nan, mu=np.nan, I2=np.nan, tau2=np.nan, cond=cond)
    Si = np.linalg.inv(Sig); one = np.ones(M)
    mu = float(one @ Si @ x / (one @ Si @ one))
    r = x - mu
    Q = float(r @ Si @ r)
    C = np.eye(M) - np.ones((M, M)) / M
    tau2 = max(0.0, float(np.var(x, ddof=1) - np.trace(C @ Sig) / (M - 1)))
    z = r / np.sqrt(np.diag(Sig))
    return dict(Q=Q, df=M - 1, p=float(chi2.sf(Q, M - 1)), mu=mu, tau2=tau2, cond=cond, z=z.tolist(), tau_up=tau_upper(x, Sig))


def Q_at(x, Sig, t2):
    Si = np.linalg.inv(Sig + t2 * np.eye(len(x))); one = np.ones(len(x))
    mu = one @ Si @ x / (one @ Si @ one); r = x - mu
    return float(r @ Si @ r)


def tau_upper(x, Sig, level=0.95):
    """Q-profile：モデルの間の分散 τ² の上側の限界（Q(τ²) ＝ χ²_{df}(0.025) を解く）。→ τ（ln ρ の標準偏差）。"""
    df = len(x) - 1; target = chi2.ppf((1 - level) / 2, df)
    if Q_at(x, Sig, 0.0) <= target:
        return 0.0
    lo, hi = 0.0, 1.0
    while Q_at(x, Sig, hi) > target:
        hi *= 2
        if hi > 1e4:
            return np.inf
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if Q_at(x, Sig, mid) > target else (lo, mid)
    return float(np.sqrt(hi))


def sig_robust(B):
    """記録用：分散は 2.5〜97.5% の幅 ÷ 3.92 の 2 乗、相関は順位相関から。"""
    from scipy.stats import spearmanr as _sp
    sd = (np.nanpercentile(B, 97.5, 0) - np.nanpercentile(B, 2.5, 0)) / 3.92
    rc = _sp(B).correlation if B.shape[1] > 2 else np.corrcoef(B.T)
    rc = 2 * np.sin(np.pi * np.asarray(rc) / 6)
    return np.outer(sd, sd) * rc


def word36(res, fail):
    if fail > A31_FAIL or not np.isfinite(res['p']):
        return 'はっきりしない'
    return 'ばらつきが見える' if res['p'] < A31_P else 'ばらつきは見えない'


def word37(ci, fail, ess):
    """登録と同じ帯の言葉。ただし「予言どおり」は、まとめた ESS（7 本の、使う温度の中央値の中央値）が A31_ESS_MIN 以上のときだけ。"""
    w = word_band30(ci, fail)
    if w == '予言どおり（2 割の内）' and not (ess >= A31_ESS_MIN):
        return '重み付けが信用できない'
    return w


def boot_logrho(fns, nD, seed, n):
    """fns：モデルごとの fn(w) → ρ。→ (n, M) の ln ρ（出ない回は nan）。"""
    rng = np.random.default_rng(seed)
    out = np.full((n, len(fns)), np.nan)
    for b in range(n):
        w = resample(rng, nD)
        for i, f in enumerate(fns):
            r = f(w)
            out[b, i] = np.log(r) if np.isfinite(r) and r > 0 else np.nan
    return out


# ---------- 自己テスト ----------

def _toy_chain(rng, n, L, h, q0=(0.2, 0.9), p_in=0.3, p_out=0.5):
    """作り物：状態 s、候補を選ぶ確率 q(s) に罰 h（T ＝ 1）。普通で候補を選ぶと p_in で入り、繰り返しで選ばないと p_out で出る。"""
    s = np.zeros(n, int); ev = np.zeros((n, 4)); pick = np.zeros((n, L), bool); qq = np.zeros((n, L))
    for t in range(L):
        q = np.where(s == 0, q0[0], q0[1]); qq[:, t] = q
        qh = q * np.exp(h) / (1 - q + q * np.exp(h))
        c = rng.random(n) < qh; pick[:, t] = c
        u = rng.random(n)
        enter = (s == 0) & c & (u < p_in); leave = (s == 1) & ~c & (u < p_out)
        ev[:, 0] += (s == 0); ev[:, 2] += (s == 1); ev[:, 1] += enter; ev[:, 3] += leave
        s = np.where(enter, 1, np.where(leave, 0, s))
    return ev, pick, qq


def self_test31():
    rng = np.random.default_rng(0)
    # (a) 自分で割る重み付けが、罰ありで別に作った ln K に近づく
    h = -0.4
    ev0, pk, qq = _toy_chain(rng, 40000, 60, 0.0)
    lw = logw_of(pk[None], qq[None], h, 1.0)[0]
    W = np.exp(lw - lw.max())
    lnIS = lnK_w(ev0[None], W[None])
    evh, _, _ = _toy_chain(rng, 40000, 60, h)
    lnh = lnK_w(evh[None], np.ones((1, 40000)))
    assert abs(lnIS - lnh) < 0.05, ('自己テスト (a)', lnIS, lnh)
    assert np.allclose(logw_of(pk[None, :50], qq[None, :50], 0.0, 1.0), 0.0)
    assert np.allclose(logw_of(pk[None, :50], qq[None, :50], h, 0.7), logw_check(pk[None, :50], qq[None, :50], h, 0.7), atol=1e-9)
    # (b) Cochran の Q：ばらつきなしで自由度ほど、ばらつきありで大きい
    M = 7; A = rng.normal(size=(M, M)); Sig = A @ A.T / M + 0.02 * np.eye(M)
    L = np.linalg.cholesky(Sig)
    Q0 = np.mean([cochran(0.1 + L @ rng.normal(size=M), Sig)['Q'] for _ in range(3000)])
    Q1 = np.mean([cochran(0.1 + L @ rng.normal(size=M) + rng.normal(0, 0.5, M), Sig)['Q'] for _ in range(3000)])
    assert abs(Q0 - 6) < 0.4 and Q1 > Q0 + 3, ('自己テスト (b)', Q0, Q1)
    # (c) 言葉の境目
    assert word36(dict(p=0.049), 0.0) == 'ばらつきが見える' and word36(dict(p=0.05), 0.0) == 'ばらつきは見えない'
    assert word36(dict(p=0.01), 0.11) == 'はっきりしない' and word36(dict(p=np.nan), 0.0) == 'はっきりしない'
    assert word37([0.81, 1.2], 0.0, 30) == '予言どおり（2 割の内）' and word37([0.81, 1.2], 0.0, 29.9) == '重み付けが信用できない'
    assert word37([0.5, 0.7], 0.0, 5) == '予言より小さく効く' and word37([0.81, 1.2], 0.11, 100) == 'はっきりしない'
    # (d) τ の上限：ばらつきが大きいほど上限も大きい
    x0 = 0.1 + L @ rng.normal(size=M)
    assert tau_upper(x0 + rng.normal(0, 1.0, M), Sig) > tau_upper(x0, Sig) >= 0.0
    return 'ok'
