"""ガウス型か Zipf 型かを見分ける 2 つの目印を、合成データで確かめる（本物の logit は使わない）。

- g = a_loc ÷ (0.36 × σ_rob)：ガウス型なら 1 くらい。
- c = （順位 1,000〜10,000 の傾き）÷（順位 100〜1,000 の傾き）：Zipf なら 1、ガウス型なら 1 より大きい（深いほど急）。
σ_rob = （75% 点 − 25% 点）÷ 1.349（外れ値に強い広がり。ガウス分布なら標準偏差と同じ）。
"""
import numpy as np
V = 151_665
rank = np.arange(1, V + 1, dtype=np.float64)

def slope(z, lo, hi):
    r = np.arange(lo, hi + 1)
    return -np.polyfit(np.log(r), z[r - 1], 1)[0]

def marks(logits):
    z = np.sort(logits)[::-1]
    q25, q75 = np.percentile(z, [25, 75])
    s_rob = (q75 - q25) / 1.349
    a_loc = slope(z, 300, 3000)
    return s_rob, a_loc, a_loc / (0.36 * s_rob), slope(z, 1000, 10000) / slope(z, 100, 1000)

rng = np.random.default_rng(7)
print(f"{'case':26s} {'sigma_rob':>9s} {'a_loc':>7s} {'g':>6s} {'c':>6s}")
for name, z in [("Gauss sigma=2", rng.normal(0, 2, V)), ("Gauss sigma=3", rng.normal(0, 3, V)),
                ("Zipf a=1", -np.log(rank)), ("Zipf a=1.5", -1.5 * np.log(rank)),
                ("Gauss sigma=3 + 20 spikes", np.concatenate([rng.normal(0, 3, V - 20), rng.normal(25, 3, 20)]))]:
    s, a, g, c = marks(z)
    print(f"{name:26s} {s:9.3f} {a:7.3f} {g:6.2f} {c:6.2f}")
