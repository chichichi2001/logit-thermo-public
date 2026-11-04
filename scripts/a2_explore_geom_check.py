"""探索 8.24 の確かめ：窓のラベル（8.21・8.22）そのものが「C → A → L」の溶け方を作っていないか（2026-10-30 18:1x、CPU、判定しない）。
scripts/a2_explore_onset.py・a2_explore_hidden.py・a2_explore_three.py を実行したのと同じ名前空間で実行する。
1) 作り物：ランダムなトークンの途中に、周期 p のきれいな結晶をいきなり入れて、いきなり終わらせる（p ＝ 3〜29、各 30 本）。
   本当は L → C → L と直接行き来しているので、ラベルが C → A や A → L を数えたら、それは窓の形が作った見かけ。
2) 本物（A2-10 の「なし」、温度 0.55・0.6）：C の窓のいちばん合うずれ（周期）と、C から出るときの行き先を、周期 11 歩以下・12 歩以上に分けて数える。
"""
import numpy as np


def geom_synth(seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for per in (3, 5, 7, 11, 17, 23, 29):
        res = {k: 0 for k in ('LC', 'LA', 'AC', 'CA', 'CL', 'AL')}
        for _ in range(30):
            s = rng.integers(100, 30000, 512)
            a = int(rng.integers(100, 200)); b = a + int(rng.integers(120, 220))
            s[a:b] = np.tile(rng.integers(100, 30000, per), 600)[:b - a]
            _, tr = seq_counts(s)
            for k in res:
                res[k] += tr.get(k, 0)
        out[per] = res
    return out


def lab_and_lag(seq):
    c = copy_marks(seq).astype(float); s = np.asarray(seq); out = []; cur = 'L'
    for a in range(0, len(s) - HD_WIN + 1, HD_STEP):
        w = s[a:a + HD_WIN]; r = c[a:a + HD_WIN].mean()
        m = [np.mean(w[p:] == w[:-p]) for p in range(2, HD_PMAX + 1)]
        x = label(r, max(m))
        if x != '-':
            cur = x
        out.append((cur, 2 + int(np.argmax(m))))
    return out


def geom_real(toks, temps, use=(0.55, 0.6)):
    res = {'short': {'CA': 0, 'CL': 0}, 'long': {'CA': 0, 'CL': 0}}; lags = []
    for T in use:
        k = list(temps).index(T)
        for s in range(toks.shape[1]):
            for j in range(toks.shape[2]):
                L = lab_and_lag(toks[k, s, j])
                for (x, lx), (y, _) in zip(L[:-1], L[1:]):
                    if x == 'C':
                        lags.append(lx)
                        if y != 'C':
                            res['short' if lx <= 11 else 'long'][x + y] += 1
    lags = np.array(lags)
    return dict(exits=res, lag_median=float(np.median(lags)), frac_lag_le11=float(np.mean(lags <= 11)))
