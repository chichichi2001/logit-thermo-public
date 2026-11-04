"""問い1（混ぜると傾きが 1 に近づくか）が、計算だけで決まってしまわないかの確認（本物の logit は使わない）。

傾き a = 1.4 のべきの分布を N 個作り、位置ごとに「どのトークンが上位か」の並びを変えて平均する。
並びの共有の度合い（全部同じ／少しずつずれる／完全にばらばら）で、平均の傾きがどう変わるかを見る。
"""
import numpy as np
V, A, N = 151_665, 1.4, 400
rng = np.random.default_rng(11)
rank = np.arange(1, V + 1, dtype=np.float64)
base = rank ** (-A); base /= base.sum()

def slope(p, lo=100, hi=10_000):
    ps = np.sort(p)[::-1]; r = np.arange(lo, hi + 1)
    return -np.polyfit(np.log(r), np.log(ps[r - 1]), 1)[0]

def mix(order_fn):
    m = np.zeros(V)
    for _ in range(N):
        m[order_fn()] += base
    return m / N

# 好み（どのトークンが上に来やすいか）を共有したうえで、位置ごとに揺らす
pref = rng.permutation(V)
def shared_noise(s):
    def f():
        score = -np.log(np.arange(1, V + 1)) + rng.normal(0, s, V)   # 共有の順位に雑音
        return pref[np.argsort(-score)]
    return f

print(f"1 つの分布の傾き = {slope(base):.3f}")
print(f"完全に同じ並び            -> 平均の傾き {slope(mix(lambda: pref)):.3f}")
for s in [0.5, 1.0, 2.0, 4.0]:
    print(f"共有の並び＋雑音 σ={s:<4}    -> 平均の傾き {slope(mix(shared_noise(s))):.3f}")
print(f"完全にばらばら            -> 平均の傾き {slope(mix(lambda: rng.permutation(V))):.3f}")
