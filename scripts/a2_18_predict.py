"""A2-18 の予言：場をかけていない（h ＝ 0）データだけから、「融点を ΔT だけ下げるのに要る罰の強さ h*」を計算する（2026-11-01 08:xx JST、CPU）。
この出力（h*）を封をして固定し、そのあとで h* の生成を回す。A2-16 と同じ名前空間（scripts/a2_16_run.py の load16ns：copy_marks・phases・fit_vh・
A8_TU ＝ 0.4〜0.7 の 7 つの温度・a2_16_analyze の arm_counts）で実行する。

使う量（どれも G-pre の h ＝ 0 の腕だけ。±0.3 の腕は使わない）：
  ファントホッフの当てはめ（窓のものさし、A2-16 と同じ）：ln K(T) ＝ a ＋ b ÷ T（b ＝ ΔH_vH）、T_m0 ＝ −b ÷ a。
  恒等式の場の効き g(T) ＝ ∂ln K/∂h（h ＝ 0）：温度ごとに、凍る・溶けるの回数と Σ（選んだか − q）の共分散から（A2-16 の records_dlnK と同じ式）。
    7 つの温度の値に直線を当てはめる（重みなし）：g(T) ＝ c0 ＋ c1 T。
  予言：ln K(T, h) ≈ a ＋ b ÷ T ＋ h g(T)。融点を T* ＝ T_m0 ＋ ΔT にする h は、ln K(T*, h*) ＝ 0 から h* ＝ −(a ＋ b ÷ T*) ÷ g(T*)。
  ΔT は −0.10 と −0.20 の 2 つ。
この形を選んだ理由（すでに見た A2-16 のデータでの確かめ、2026-11-01 08:2x）：h ＝ −0.3 の腕の融点の動きを、h ＝ 0 だけから予言すると、
  予言 ÷ 実測が 7 モデルで 0.83〜1.21（真ん中 1.02）。融点で傾き g を 1 回だけ測って直線で動かす形（0.92〜1.28、真ん中 1.15）より当たった。
対抗の予言（記録）：写した数 M を共役な量とする平衡のクラペイロン（A2-16 の予言の傾き T_m0 Δm ÷ 熱量計の ΔH）で、同じ h* のとき融点がどれだけ動くか。
  動いた量 ÷ ΔT を r_C とする（A2-16 では、この傾きは測った傾きの 1.5〜2 倍だった）。
"""
import json
import numpy as np

P18_MODELS = ['pythia-410m', 'pythia-1.4b', 'gpt-neo-1.3b', 'smollm2-1.7b', 'qwen2.5-1.5b', 'gpt2-1.5b', 'bloom-1b1']
P18_DT = [-0.10]
P18_GRID = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]     # 生成の温度：T*（0.01 に丸める）のまわり


def identity_g(Z0):
    """h ＝ 0 の腕（toks・pick・q、形 (7, S, nD, L)）→ 温度ごとの恒等式の ∂ln K/∂h（A2-16 の records_dlnK と同じ式）。"""
    pick, q = Z0['pick'].astype(float), Z0['q'].astype(float)
    sig = (pick - q).sum(-1)
    out = []
    for k, T in enumerate(G16_TEMPS):
        ev = np.zeros((4,) + sig.shape[1:])
        for s in range(sig.shape[1]):
            for j in range(sig.shape[2]):
                st, _ = phases(copy_marks(Z0['toks'][k, s, j]))
                a, b = st[:-1], st[1:]
                ev[:, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
        sg = sig[k].ravel()
        d = [float(np.mean((ev[i].ravel() - ev[i].mean()) * (sg - sg.mean())) / (T * ev[i].mean())) if ev[i].mean() > 0 else float('nan') for i in range(4)]
        out.append((d[1] - d[0]) - (d[3] - d[2]))
    return np.array(out)


def hstar_from(a, b, c, Tstar):
    g = c[0] + c[1] * Tstar
    return float(-(a + b / Tstar) / g) if g > 0 else float('nan')


def predict_model(base, name):
    Z = np.load(f'{base}/{name}@h0.npz')
    assert [round(float(t), 4) for t in Z['temps']] == G16_TEMPS and abs(float(Z['h'])) < 1e-12, name
    Z0 = {k: Z[k] for k in ('toks', 'pick', 'U', 'q', 'q0')}
    cw, _ = arm_counts(Z0['toks'], Z0['U'], Z0['pick'])
    tot = cw.sum(1)
    f = fit_vh(tot[:, :10])
    assert f.get('ok') and np.isfinite(f['Tm']), (name, '当てはめが出ない')
    b = float(f['dH_vH']); Tm0 = float(f['Tm']); a = -b / Tm0
    gT = identity_g(Z0)
    c = [float(np.mean(gT)), 0.0]                                        # ḡ（定数）
    k = np.array(f['use_idx'], int)
    dm = float(np.mean(tot[k, 11] / tot[k, 7] - tot[k, 10] / tot[k, 5]))
    sC = Tm0 * dm / f['dH_cal']                                         # 対抗：平衡のクラペイロンの傾き
    res = dict(Tm0=Tm0, a=a, b=b, dH_cal=float(f['dH_cal']), dm=dm, g_T=gT.tolist(), g_coef=[float(c[0]), float(c[1])], s_clapeyron=sC, targets={})
    for dT in P18_DT:
        Ts = Tm0 + dT
        h = hstar_from(a, b, c, Ts)
        Tc = round(Ts, 2)
        res['targets'][f'{dT:+.2f}'] = dict(dT=dT, Tstar=Ts, hstar=h, grid=[round(Tc + x, 2) for x in P18_GRID],
                                             r_clapeyron=float(sC * h / dT) if np.isfinite(h) else float('nan'))
    return res


def predict_all(base, names=P18_MODELS):
    return {n: predict_model(base, n) for n in names}


def self_test_p18():
    # (1) h* の解：ln K ＝ a ＋ b/T ＋ h g(T) が T* で 0 になる
    a, b, c = -6.0, 3.6, (2.0, 0.5)
    for Ts in (0.45, 0.55):
        h = hstar_from(a, b, c, Ts)
        assert abs(a + b / Ts + h * (c[0] + c[1] * Ts)) < 1e-12
    assert hstar_from(a, b, c, 0.5) < 0                                  # T* ＜ T_m0（0.6）なら罰（h ＜ 0）
    # (2) 格子は 7 つで、T* のまわり
    g = [round(round(0.4712, 2) + x, 2) for x in P18_GRID]
    assert len(g) == 7 and g[3] == 0.47
    return 'OK'
