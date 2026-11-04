"""A2-1 の計算：一歩の「融点」T_half（頭と裾の確率がちょうど半分ずつになる温度）。

T_half(H)：順位 H 以降の確率の合計が 1/2 を超える最低の温度。
頭（順位 1〜H−1）と裾（順位 H〜）の自由エネルギーが等しくなる温度で、化学の融点と同じ考え方。
log(裾の確率 ÷ 頭の確率) が 0 を横切る温度を、対数温度で直線補間して求める。
"""
import numpy as np

T_GRID2 = np.exp(np.linspace(np.log(0.05), np.log(20.0), 241))
HEADS = (30, 100, 300)


def _lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def tail_log_odds(zs, H, T):
    """zs：大きい順に並べた logit。log(順位 H 以降の確率 ÷ 順位 1〜H−1 の確率)。"""
    return _lse(zs[H - 1:] / T) - _lse(zs[:H - 1] / T)


def t_half(zs, H):
    """(T_half, 端の印)。端の印：None・'low'（最低温ですでに裾が上）・'high'（最高温でも頭が上）。"""
    lo = np.array([tail_log_odds(zs, H, T) for T in T_GRID2])
    above = np.nonzero(lo > 0)[0]
    if len(above) == 0:
        return np.nan, 'high'
    i = above[0]
    if i == 0:
        return np.nan, 'low'
    t0, t1 = np.log(T_GRID2[i - 1]), np.log(T_GRID2[i])
    f = (0 - lo[i - 1]) / (lo[i] - lo[i - 1])
    return float(np.exp(t0 + f * (t1 - t0))), None
