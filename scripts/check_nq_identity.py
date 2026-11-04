"""σ ＝ N − Q の導出（docs/THEORY_NQ.md）を、小さな作り物の自己回帰モデルで、全部の文を数え上げて（厳密に）確かめる。CPU、乱数の作り物だけ（実データは使わない）。

  python3 scripts/check_nq_identity.py

確かめること（どれも中心差分 と 式 の差の最大を出す）
  (1) 文 1 本ごとの スコア：∂ log P_h(x)/∂h ＝ (N − Q)/T（h ＝ 0）。
  (2) 罰（歩ごと）への応答：d E[O]/dh ＝ Cov(O, N)/T − Cov(O, Q)/T。E[N − Q] ＝ 0。
  (3) 文全体を傾けたとき（P_0 e^{hN/T}/Z）の応答：d E[O]/dh ＝ Cov(O, N)/T（平衡の揺らぎと応答の関係の形）。(2) との差は −Cov(O, Q)/T。
  (4) 温度（β ＝ 1/T、歩ごと）への応答：d E[O]/dβ ＝ −Cov(O, U − U_comp)、U_comp ＝ Σ_t E_t[U_t]。U の取り方（−log p_1 か −z か）によらない。
      文全体を e^{−δ ΣU} で傾けた応答 −Cov(O, U) は、U の取り方で変わる。
  (5) 割る型の罰：∂ log P/∂λ ＝ Σ_t (a(x_t) − E_t a)/T、a ＝ 見たことのあるトークンで −|z|、λ ＝ ln θ。
  (6) ドゥーブの分け方：N − E[N] ＝ (N − Q) ＋ Σ_t δ_t、δ_t ＝ E[N_{>t} | x_≤t] − E[N_{>t} | x_<t]（この先の写しの見込みの更新）。
      したがって Cov(O, Q) ＝ Cov(O, Σ_t δ_t)：文全体を傾けた応答と歩ごとの罰の応答の差は、ちょうど「この先の見込み」の項。
"""
import itertools
import numpy as np

V, L, T0 = 5, 5, 0.7
rng = np.random.default_rng(23)
TABLE = rng.normal(0, 2.0, (V, V, V))
PROMPT = [0, 1, 2]
SEQS = list(itertools.product(range(V), repeat=L))


def cands(hist):
    a, b = hist[-2], hist[-1]
    return {hist[i + 2] for i in range(len(hist) - 2) if hist[i] == a and hist[i + 1] == b}


def logits(hist, h=0.0, lam=0.0):
    z = TABLE[hist[-2], hist[-1]].copy()
    for v in cands(hist):
        z[v] += h
    if lam != 0.0:
        th = np.exp(lam)
        for v in set(hist):
            z[v] = z[v] / th if z[v] > 0 else z[v] * th
    return z


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def path_stats(h=0.0, beta=1.0 / T0, lam=0.0):
    """全部の文について：log P、N、Q、U（−log p_1）・U_comp、Uz（−z）・Uz_comp、A・A_comp（割る型）、O の 2 つ。"""
    out = []
    for seq in SEQS:
        hist = list(PROMPT); lp = 0.0; N = Q = U = Uc = Uz = Uzc = A = Ac = 0.0
        for x in seq:
            z0 = TABLE[hist[-2], hist[-1]]
            z = logits(hist, h, lam)
            p = softmax(beta * z)
            lp += np.log(p[x])
            C = cands(hist)
            N += x in C; Q += sum(p[v] for v in C)
            u1 = -np.log(softmax(z0))                                    # 温度 1・罰なしの −log p
            U += u1[x]; Uc += float(p @ u1)
            Uz += -z0[x]; Uzc += float(p @ (-z0))
            seen = set(hist); a = np.array([-abs(z0[v]) if v in seen else 0.0 for v in range(V)])
            A += a[x]; Ac += float(p @ a)
            hist.append(x)
        s = hist[len(PROMPT):]
        g = [tuple(s[i:i + 3]) for i in range(len(s) - 2)]
        O1 = float(len(g) != len(set(g)))                                # 3 つ組のくり返しがあるか（なめらかでない）
        O2 = float(s[-1] in cands(hist[:-1]))                            # 最後の歩で候補を選んだか
        out.append((lp, N, Q, U, Uc, Uz, Uzc, A, Ac, O1, O2))
    return np.array(out)


def cov(P, a, b):
    ma, mb = P @ a, P @ b
    return float(P @ ((a - ma) * (b - mb)))


if __name__ == '__main__':
    eps = 1e-5
    S0 = path_stats(); P0 = np.exp(S0[:, 0]); assert abs(P0.sum() - 1) < 1e-12
    lp, N, Q, U, Uc, Uz, Uzc, A, Ac, O1, O2 = S0.T
    worst = {}

    # (1) スコア
    Sp, Sm = path_stats(h=eps), path_stats(h=-eps)
    fd = (Sp[:, 0] - Sm[:, 0]) / (2 * eps)
    worst['(1) スコア ＝ (N − Q)/T'] = float(np.max(np.abs(fd - (N - Q) / T0)))
    worst['(2) E[N − Q] ＝ 0'] = float(abs(P0 @ (N - Q)))

    # (2)(3) 罰への応答と、文全体を傾けた応答
    for nm, O in (('O1', O1), ('O2', O2)):
        dO = (np.exp(Sp[:, 0]) @ O - np.exp(Sm[:, 0]) @ O) / (2 * eps)
        rhs = cov(P0, O, N) / T0 - cov(P0, O, Q) / T0
        worst[f'(2) 罰の応答 {nm}'] = abs(dO - rhs)
        tilt = lambda hh: (P0 * np.exp(hh * N / T0)) / (P0 @ np.exp(hh * N / T0))
        dT = (tilt(eps) @ O - tilt(-eps) @ O) / (2 * eps)
        worst[f'(3) 文全体を傾けた応答 {nm} ＝ Cov(O,N)/T'] = abs(dT - cov(P0, O, N) / T0)
        print(f'  {nm}：罰の応答 {dO:+.6f}、Cov(O,N)/T {cov(P0, O, N) / T0:+.6f}、−Cov(O,Q)/T {-cov(P0, O, Q) / T0:+.6f}、Q が打ち消す割合 {cov(P0, O, Q) / cov(P0, O, N):.3f}')

    # (4) 温度
    b0 = 1.0 / T0
    Bp, Bm = path_stats(beta=b0 + eps), path_stats(beta=b0 - eps)
    fdb = (Bp[:, 0] - Bm[:, 0]) / (2 * eps)
    worst['(4) β のスコア ＝ −(U − U_comp)'] = float(np.max(np.abs(fdb + (U - Uc))))
    worst['(4) β のスコア ＝ −(Uz − Uz_comp)'] = float(np.max(np.abs(fdb + (Uz - Uzc))))
    for nm, O in (('O1', O1), ('O2', O2)):
        dO = (np.exp(Bp[:, 0]) @ O - np.exp(Bm[:, 0]) @ O) / (2 * eps)
        worst[f'(4) 温度の応答 {nm}'] = abs(dO + cov(P0, O, U - Uc))
        print(f'  {nm}：温度の応答 {dO:+.6f}、文全体の傾け −Cov(O,U) {-cov(P0, O, U):+.6f}（U ＝ −log p_1）、{-cov(P0, O, Uz):+.6f}（U ＝ −z）')

    # (5) 割る型
    Lp, Lm = path_stats(lam=eps), path_stats(lam=-eps)
    fdl = (Lp[:, 0] - Lm[:, 0]) / (2 * eps)
    worst['(5) 割る型のスコア ＝ (A − A_comp)/T'] = float(np.max(np.abs(fdl - (A - Ac) / T0)))

    # (6) ドゥーブの分け方（この先の写しの見込み W(prefix) ＝ E[その先の歩の c の和 | prefix] を、後ろから数え上げる）
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def W(prefix):
        hist = list(PROMPT) + list(prefix)
        if len(prefix) == L:
            return 0.0
        p = softmax(TABLE[hist[-2], hist[-1]] / T0); C = cands(hist)
        return float(sum(p[v] * ((v in C) + W(prefix + (v,))) for v in range(V)))
    D = np.zeros(len(SEQS))
    for i, seq in enumerate(SEQS):
        for t in range(L):
            hist = list(PROMPT) + list(seq[:t]); p = softmax(TABLE[hist[-2], hist[-1]] / T0)
            D[i] += W(tuple(seq[:t + 1])) - float(sum(p[v] * W(tuple(seq[:t]) + (v,)) for v in range(V)))
    worst['(6) N − E[N] ＝ (N − Q) ＋ Σδ'] = float(np.max(np.abs((N - P0 @ N) - ((N - Q) + D))))
    for nm, O in (('O1', O1), ('O2', O2)):
        worst[f'(6) Cov(O,Q) ＝ Cov(O,Σδ) {nm}'] = abs(cov(P0, O, Q) - cov(P0, O, D))
    print('差の最大（中心差分 対 式）')
    for k, v in worst.items():
        print(f'  {k}：{v:.2e}')
    assert all(v < 1e-6 for v in worst.values()), worst
    print('OK')
