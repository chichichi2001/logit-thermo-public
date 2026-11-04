"""探索 8.41：恒等式の 2 次の項（ビリアル展開のたとえ）。罰をかけていない出力だけから、ln K の h についての 2 階微分 g₂ を出す（2026-11-02 JST、CPU）。
判定はしない（もう見たデータでの探索）。よさそうなら、A2-20 の第 2 段のデータの前に、記録の問いとして封をする。

式（どの「文の関数」Y でも成り立つ。罰 h は候補の logit に足し、温度 T で割る）：
  1 本の文の重みの h による対数微分（スコア）は S/T、S ＝ Σ_t（選んだか − q_t）。
  q_t の h による微分は q_t(1 − q_t)/T なので、S の h による微分は −V/T、V ＝ Σ_t q_t(1 − q_t)。
  E_h[Y]' ＝ E[Y S]/T（＝ Cov(Y, S)/T。E[S] ＝ 0）
  E_h[Y]'' ＝ E[Y (S² − V)]/T²（＝ Cov(Y, S² − V)/T²。E[S² − V] ＝ 0）
  ln E[Y] の 2 階微分 ＝ E''/E − (E'/E)²。
  ln K ＝ ln E[凍る回] − ln E[液体の歩] − ln E[溶ける回] ＋ ln E[結晶の歩] なので、g₂ はこの 4 つを同じ符号で足したもの（g と同じ組み立て）。
予言の 2 次の形：Δ ln K ≈ h g ＋ (h²/2) g₂。

あわせて：作り物の「窓のものさしは比を 1 割高く読む」は、本当の状態の ln K の傾き（g_真）と比べたときの話で、
本物の分析の ρ（分子も分母も同じ窓のものさしで数えた ln K と、その同じ数えに恒等式を当てたもの）には当てはまらない（恒等式はどの関数にも正確）。
"""
import numpy as np


def seq_events2(toks, pick, q):
    """seq_events（A2-18）に V ＝ Σ q(1 − q) を足したもの → ev (nT, S, nD, 4)、sig (nT, S, nD)、var (nT, S, nD)。"""
    ev, sig = seq_events(toks, pick, q)
    qq = q.astype(float)
    return ev, sig, (qq * (1 - qq)).sum(-1)


def g12_identity(ev, sig, var, temps, w=None):
    """h ＝ 0 の腕 → 温度ごとの (g, g₂)。g は g_identity と同じ（重み w は記事ごと）。"""
    S_, nD = ev.shape[1], ev.shape[2]
    w = np.ones(nD) if w is None else w
    ww = np.repeat(w[None, :], S_, 0).ravel()
    W = ww.sum()
    out1, out2 = [], []
    for k, T in enumerate(temps):
        s = sig[k].ravel(); v = var[k].ravel()
        m = lambda x: (ww * x).sum() / W
        u = s - m(s); r = s * s - v; r = r - m(r)
        d1, d2 = [], []
        for i in range(4):
            e = ev[k, :, :, i].ravel(); me = m(e)
            if me <= 0:
                d1.append(np.nan); d2.append(np.nan); continue
            E1 = m((e - me) * u) / T; E2 = m((e - me) * r) / (T * T)
            d1.append(E1 / me); d2.append(E2 / me - (E1 / me) ** 2)
        out1.append((d1[1] - d1[0]) - (d1[3] - d1[2])); out2.append((d2[1] - d2[0]) - (d2[3] - d2[2]))
    return np.array(out1), np.array(out2)


# ---------- 作り物で確かめる（くり返しの鎖。候補 ＝ 1 つ前のトークン） ----------
def toy_sample(N, L, V, T, h, beta, base, rng):
    """N 本の文を生成 → toks (N, L)、pick (N, L)、q (N, L)。logit ＝ base ＋ beta·[x ＝ 1 つ前]、候補（1 つ前と同じトークン）に h を足して T で割る。"""
    toks = np.zeros((N, L), int); pick = np.zeros((N, L), bool); q = np.zeros((N, L))
    prev = rng.integers(0, V, N)
    for t in range(L):
        z = np.repeat(base[t][None, :], N, 0)
        z[np.arange(N), prev] += beta + h
        z = z / T; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        q[:, t] = p[np.arange(N), prev]
        c = p.cumsum(1); x = (rng.random(N)[:, None] < c).argmax(1)
        pick[:, t] = x == prev; toks[:, t] = x; prev = x
    return toks, pick, q


def toy_functionals(pick):
    """くり返しの 2 つの状態（連続 3 回以上のくり返しを「結晶」）→ [液体の歩, 凍る回, 結晶の歩, 溶ける回]（1 本ごと）。"""
    N, L = pick.shape
    run = np.zeros((N, L), int)
    for t in range(L):
        run[:, t] = np.where(pick[:, t], (run[:, t - 1] if t else 0) + 1, 0)
    st = run >= 3
    a, b = st[:, :-1], st[:, 1:]
    return np.stack([(~a).sum(1), ((~a) & b).sum(1), a.sum(1), (a & ~b).sum(1)], 1).astype(float)


def self_test_g2(N=40000, L=120, V=12, T=0.7, beta=1.5, seed=3):
    """h ＝ 0 だけから出した ln K の 1 次・2 次の予言を、h ≠ 0 で本当に生成した ln K と比べる。2 次を足すと当たりがよくなること。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, (L, V))
    tk, pk, q = toy_sample(N, L, V, T, 0.0, beta, base, rng)
    ev = toy_functionals(pk)[None, :, None, :].transpose(0, 1, 2, 3)          # (1, N, 1, 4)
    sig = (pk - q).sum(1)[None, :, None]; var = (q * (1 - q)).sum(1)[None, :, None]
    g, g2 = g12_identity(ev, sig, var, [T])
    lnK = lambda e: np.log(e[:, 1].sum() / e[:, 0].sum()) - np.log(e[:, 3].sum() / e[:, 2].sum())
    l0 = lnK(ev[0, :, 0])
    rows = []
    for h in (-0.8, -0.4, 0.4, 0.8):
        _, pkh, _ = toy_sample(N, L, V, T, h, beta, base, np.random.default_rng(seed + 100 + int(h * 10)))
        d = lnK(toy_functionals(pkh)) - l0
        p1 = h * g[0]; p2 = p1 + 0.5 * h * h * g2[0]
        rows.append((h, d, p1, p2))
    return dict(g=float(g[0]), g2=float(g2[0]), rows=rows)
