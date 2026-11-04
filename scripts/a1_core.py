"""A1（ゲート v0.3）の計算。ノートブックにはこのファイルの中身をそのまま埋め込む。

1 行（1 位置の logit、使う番号だけ）ごとに row_metrics() で数字を出し、
summarize() で J1・J2（裾の型の判定）と R1〜R5（記録）をまとめる。
すべて float64。温度は T = 1 の logit を T で割る。
"""
import numpy as np

K = 1000                 # 壁の位置
H = 100                  # 頭として除く順位（1..H-1 を除く）
WIN = (100, 10_000)      # a_win の窓
LOC = (300, 3_000)       # a_loc の窓
S_WIN = (100, 1_001)     # R1 の s の窓
C_SHALLOW, C_DEEP = (100, 1_000), (1_000, 10_000)
GAUSS_K = 0.36           # ガウス型での a_loc ÷ σ（§2.5）
T_GRID = np.exp(np.linspace(np.log(0.2), np.log(8.0), 121))
G_BANDS = ((0.85, 1.15),)   # J1：[0.85, 1.15] がガウス型
C_BANDS = ((0.9, 1.15),)    # J2：[0.9, 1.15] が Zipf 型
BOOT_N, BOOT_SEED = 2000, 20261027


def thresholds(V):
    """p ∝ 1/r（語彙 V）での Zc_full・Zc_tail の値。"""
    w = 1.0 / np.arange(1, V + 1, dtype=np.float64)
    return K * np.log1p(w[K] / w[:K].sum()), K * np.log1p(w[K] / w[H - 1:K].sum())


def _slope(zs, lo, hi):
    r = np.arange(lo, hi + 1)
    return -np.polyfit(np.log(r), zs[r - 1], 1)[0]


def _t_on(zc, theta):
    """(T_on, 端の印)。端の印は None・'low'（最低温ですでに超えている）・'high'（最高温でも届かない）。"""
    above = np.nonzero(zc >= theta)[0]
    if len(above) == 0:
        return np.nan, 'high'
    i = above[0]
    if i == 0:
        return np.nan, 'low'
    t0, t1 = np.log(T_GRID[i - 1]), np.log(T_GRID[i])
    f = (theta - zc[i - 1]) / (zc[i] - zc[i - 1])
    return float(np.exp(t0 + f * (t1 - t0))), None


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def row_metrics(z, theta_full, theta_tail):
    """z：1 行の logit（使う番号だけ、float64）。"""
    zs = np.sort(z)[::-1]
    out = {}
    # 傾き
    r = np.arange(WIN[0], WIN[1] + 1)
    x, y = np.log(r), zs[r - 1]
    sl, ic = np.polyfit(x, y, 1)
    resid = y - (sl * x + ic)
    out['a_win'] = -sl
    out['r2_win'] = 1 - resid.var() / y.var()
    out['a_loc'] = _slope(zs, *LOC)
    out['s'] = _slope(zs, *S_WIN) / out['a_loc']
    out['c'] = _slope(zs, *C_DEEP) / _slope(zs, *C_SHALLOW)
    q25, q75 = np.percentile(z, [25, 75])
    out['sigma_rob'] = (q75 - q25) / 1.349
    out['g'] = out['a_loc'] / (GAUSS_K * out['sigma_rob'])
    # 頭の強さ（T = 1）
    out['h'] = (_logsumexp(zs[:H - 1]) - _logsumexp(zs[H - 1:])) / np.log(10)
    # 圧縮因子（上位 K+1 個だけ使う）
    top = zs[:K + 1]
    zf, zt = [], []
    for T in T_GRID:
        w = np.exp((top - top[0]) / T)
        zf.append(K * np.log1p(w[K] / w[:K].sum()))
        zt.append(K * np.log1p(w[K] / w[H - 1:K].sum()))
    out['t_on_full'], out['edge_full'] = _t_on(np.array(zf), theta_full)
    out['t_on_tail'], out['edge_tail'] = _t_on(np.array(zt), theta_tail)
    # 比熱 C(T) = Var(E) / T²、E = −z
    cs = []
    zc0 = zs - zs[0]
    for T in T_GRID:
        w = np.exp(zc0 / T)
        p = w / w.sum()
        m1 = (p * zs).sum()
        cs.append(((p * zs * zs).sum() - m1 * m1) / (T * T))
    i = int(np.argmax(cs))
    out['t_c'] = float(T_GRID[i]) if 0 < i < len(T_GRID) - 1 else np.nan
    out['edge_c'] = None if 0 < i < len(T_GRID) - 1 else ('low' if i == 0 else 'high')
    return out


def _band(v, lo, hi, names):
    return names[0] if v < lo else (names[1] if v <= hi else names[2])


G_NAMES = ('ガウスより緩い', 'ガウス型', 'Zipf寄り')
C_NAMES = ('浅いほうが急', 'Zipf型', 'ガウス型')


def _spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return float(spearmanr(a[ok], b[ok]).correlation)


def _q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else None


def _dist(v):
    return {k: _q(v, p) for k, p in (('p10', 10), ('p25', 25), ('median', 50), ('p75', 75), ('p90', 90))}


def _judge(values, prompts, lo, hi, names):
    values, prompts = np.asarray(values, float), np.asarray(prompts)
    med = float(np.median(values))
    ids = np.unique(prompts)
    groups = [values[prompts == i] for i in ids]
    rng = np.random.default_rng(BOOT_SEED)
    meds = [np.median(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]))
            for _ in range(BOOT_N)]
    ci = (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))
    bands = sorted({_band(ci[0], lo, hi, names), _band(med, lo, hi, names), _band(ci[1], lo, hi, names)},
                   key=names.index)
    share = {n: float(np.mean([_band(v, lo, hi, names) == n for v in values])) for n in names}
    return {'median': med, 'ci95_prompt_boot': ci, 'band_of_median': _band(med, lo, hi, names),
            'clear': len(bands) == 1, 'bands_touched_by_ci': bands, 'share_rows': share}


def summarize(rows, prompts):
    """rows：row_metrics の結果のリスト、prompts：各行のプロンプトの番号。"""
    col = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in rows], float)
    edge = lambda k: [r[k] for r in rows]
    g, c = col('g'), col('c')
    a_loc, a_win = col('a_loc'), col('a_win')
    ton_t, ton_f, s, h, tc = col('t_on_tail'), col('t_on_full'), col('s'), col('h'), col('t_c')
    ratio_t, ratio_f = ton_t / a_loc, ton_f / a_win
    n = len(rows)
    frac = lambda k, v: float(sum(e == v for e in edge(k)) / n)
    return {
        'rows': n, 'prompts': int(len(np.unique(prompts))),
        'J1_g': _judge(g, prompts, *G_BANDS[0], G_NAMES),
        'J2_c': _judge(c, prompts, *C_BANDS[0], C_NAMES),
        'R1': {'ratio_tail': _dist(ratio_t), 's': _dist(s),
               'median_ratio_minus_s': _q(ratio_t - s, 50), 'spearman_ratio_s': _spearman(ratio_t, s),
               'spearman_ratio_c': _spearman(ratio_t, c),
               'v02_H1_check': {'median_ratio': _q(ratio_t, 50), 'spearman_a_loc_t_on_tail': _spearman(a_loc, ton_t)}},
        'R2': {'spearman_h_ratio_full': _spearman(h, ratio_f), 'ratio_full': _dist(ratio_f), 'h': _dist(h)},
        'R3': {'spearman_t_c_t_on_tail': _spearman(tc, ton_t), 'median_t_c_over_t_on_tail': _q(tc / ton_t, 50),
               't_c': _dist(tc)},
        'R4': {'a_win': _dist(a_win), 'a_loc': _dist(a_loc), 'r2_win': _dist(col('r2_win')),
               'sigma_rob': _dist(col('sigma_rob'))},
        'R5': {'tail_low': frac('edge_tail', 'low'), 'tail_high': frac('edge_tail', 'high'),
               'full_low': frac('edge_full', 'low'), 'full_high': frac('edge_full', 'high'),
               'c_low': frac('edge_c', 'low'), 'c_high': frac('edge_c', 'high')},
    }
