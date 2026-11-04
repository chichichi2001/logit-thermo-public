"""A1 の事前確認（本物の logit は使わない）。

1. theta*(K, V)：p_r ∝ 1/r（Zipf の傾き 1）の分布での圧縮因子 Zc(K)。
   傾き a の Zipf を温度 T = a にすると、ちょうど p_r ∝ 1/r になるので、
   「Zc(K) が theta* を横切る温度 T_on」は、純粋な Zipf では a と一致する。
2. 合成の分布（純粋な Zipf、ガウス型の logit、頭だけ尖った Zipf）で、
   A1 の手順（a の当てはめ → T_on）を通したとき、T_on / a がどうなるか。
"""
import numpy as np

V = 151_665          # Qwen2.5 のトークナイザーの長さ（metadata.json で本番前に確かめる）
K = 1000             # 壁の位置（残すトークンの数）
R_LO, R_HI = 100, 10_000   # 裾の傾きを当てはめる順位の範囲
T_GRID = np.exp(np.linspace(np.log(0.2), np.log(8.0), 121))


def zc_curve(logits, K=K):
    z = np.sort(logits)[::-1].astype(np.float64)
    out = []
    for T in T_GRID:
        w = np.exp((z - z[0]) / T)
        Pk = w[:K].sum()
        out.append(K * np.log1p(w[K] / Pk))
    return np.array(out)


def tail_slope(logits):
    z = np.sort(logits)[::-1].astype(np.float64)
    logp = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
    r = np.arange(R_LO, R_HI + 1)
    x, y = np.log(r), logp[r - 1]
    slope, icpt = np.polyfit(x, y, 1)
    resid = y - (slope * x + icpt)
    r2 = 1 - resid.var() / y.var()
    return -slope, r2


def t_on(zc, theta):
    above = np.nonzero(zc >= theta)[0]
    if len(above) == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return T_GRID[0]
    # 対数温度で線形補間
    t0, t1 = np.log(T_GRID[i - 1]), np.log(T_GRID[i])
    f = (theta - zc[i - 1]) / (zc[i] - zc[i - 1])
    return float(np.exp(t0 + f * (t1 - t0)))


rank = np.arange(1, V + 1, dtype=np.float64)
theta = zc_curve(-np.log(rank))[np.argmin(np.abs(T_GRID - 1.0))]
# T_GRID に 1.0 がちょうど無いかもしれないので、直接計算もする
w = 1.0 / rank
theta_exact = K * np.log1p(w[K] / w[:K].sum())
print(f"theta*(K={K}, V={V}) = {theta_exact:.6f}")

rng = np.random.default_rng(20261027)
cases = []
for a in [0.7, 1.0, 1.5, 2.5]:
    cases.append((f"Zipf a={a}", -a * np.log(rank)))
for s in [1.5, 2.5, 3.5]:
    cases.append((f"Gauss sigma={s}", rng.normal(0, s, V)))
for a in [1.0, 1.5]:
    z = -a * np.log(rank)
    z[:5] += 8.0   # 頭の 5 個だけ飛び出た（迷いの少ない位置のまね）
    cases.append((f"Zipf a={a} + spiky head", z))

print(f"{'case':28s} {'a_fit':>7s} {'R2':>6s} {'T_on':>7s} {'T_on/a':>7s}")
for name, z in cases:
    a_fit, r2 = tail_slope(z)
    ton = t_on(zc_curve(z), theta_exact)
    print(f"{name:28s} {a_fit:7.3f} {r2:6.3f} {ton:7.3f} {ton / a_fit:7.3f}")


# ---- 変種：裾だけの圧縮因子と、壁の近くでの傾き ----
H = R_LO          # 頭として除く順位（1..H-1 を除く）
L_LO, L_HI = 300, 3000   # 壁 K=1000 のまわりで傾きを測る範囲


def zc_tail_curve(logits, K=K):
    z = np.sort(logits)[::-1].astype(np.float64)
    out = []
    for T in T_GRID:
        w = np.exp((z - z[H - 1]) / T)
        Pk = w[H - 1:K].sum()
        out.append(K * np.log1p(w[K] / Pk))
    return np.array(out)


def local_slope(logits):
    z = np.sort(logits)[::-1].astype(np.float64)
    r = np.arange(L_LO, L_HI + 1)
    slope, _ = np.polyfit(np.log(r), z[r - 1], 1)
    return -slope


w = 1.0 / rank
theta_tail = K * np.log1p(w[K] / w[H - 1:K].sum())
print(f"\ntheta_tail*(K={K}, head<{H}) = {theta_tail:.6f}")
print(f"{'case':28s} {'a_loc':>7s} {'Ton_tl':>7s} {'ratio':>7s}")
for name, z in cases:
    a_loc = local_slope(z)
    ton = t_on(zc_tail_curve(z), theta_tail)
    print(f"{name:28s} {a_loc:7.3f} {ton:7.3f} {ton / a_loc:7.3f}")
