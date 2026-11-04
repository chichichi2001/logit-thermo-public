"""探索 8.26：対応状態（温度 ÷ 融点）でそろえると、塩の効きは家系ごとに 1 本の線に乗るか（2026-10-30 22:5x、CPU、判定しない）。
scripts/a2_13_analyze.py を実行した名前空間で実行する。A2-13 の 9 モデル（Pythia 6・SmolLM2 3）と、A2-10 の OLMo・Qwen（「偽薬」はサンプル 2）。
温度ごとに v(T) ＝ log(k_AL(偽薬) ÷ k_AL(なし))（両方で A → L が 8 回以上の温度だけ）を出し、θ ＝ T ÷ T_m で並べる。
家系ごとに v ＝ α ＋ β × θ を回数の重みで当てはめ、θ ＝ 0.9 と 1.0 での値を比べる。
"""
import numpy as np


def corr_states(base):
    out = {}
    for m in C13_MODELS:
        arms_s = C13_SALT[m]
        tn = sum(seg_counts_by_doc(load13(base, m, a)[0]) for a in C13_BASE).sum(1)
        ta = sum(seg_counts_by_doc(load13(base, m, a)[0]) for a in arms_s).sum(1)
        two = sum(counts2(load13(base, m, a)[0]) for a in C13_BASE).sum(1)
        f = fit_vh(two)
        Tm = f.get('Tm') if f.get('ok') else float('nan')
        pts = []
        for k, T in enumerate(C13_TEMPS):
            if tn[k, 4] >= 8 and ta[k, 4] >= 8:
                v = float(np.log((ta[k, 4] / ta[k, 1]) / (tn[k, 4] / tn[k, 1])))
                w = float(1 / (1 / ta[k, 4] + 1 / tn[k, 4]))
                pts.append(dict(T=T, theta=T / Tm, v=v, w=w))
        out[m] = dict(Tm=Tm, pts=pts)
    fam = {'pythia': [m for m, _ in C13_PYTHIA], 'smollm2': C13_SMOL, 'olmo_qwen': C13_REF}
    fits = {}
    for f, ms in fam.items():
        P = [p for m in ms for p in out[m]['pts']]
        x = np.array([p['theta'] for p in P]); y = np.array([p['v'] for p in P]); w = np.array([p['w'] for p in P])
        X = np.stack([np.ones_like(x), x], 1)
        a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        r = y - X @ np.array([a, b])
        fits[f] = dict(alpha=float(a), beta=float(b), at_0_9=float(a + 0.9 * b), at_1_0=float(a + b),
                       theta_range=[float(x.min()), float(x.max())], wrms=float(np.sqrt(np.sum(w * r * r) / np.sum(w))), n=len(P))
    return dict(models=out, fits=fits)
