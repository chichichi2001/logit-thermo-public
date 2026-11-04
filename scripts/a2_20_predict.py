"""A2-20 の予言：7B 級のモデルで、罰をかけていない（h ＝ 0）データだけから、融点を 0.10 下げる罰 h* を出す（2026-11-02 JST、CPU）。
A2-18 の予言（scripts/a2_18_predict.py `72b2ed5d…`）と同じ式・同じ数え方。違うのは温度の並びだけ：
  7B の融点がどこにあるか分からないので、第 1 段は 0.20〜0.90 の 15 の温度（0.05 おき）で回す。
  そこから「窓」として 7 つの続いた温度を選び、その 7 つで A2-18 と同じ当てはめをする。
窓の選び方（データの前に固定）：
  ① 15 の温度ぜんぶでファントホッフの当てはめ（窓のものさし）をして、仮の融点 T_all を出す。
  ② T_all にいちばん近い格子の温度を真ん中にして、前後 3 つずつ、計 7 つ（端にかかるときは、端から 7 つ）。
  ③ その 7 つの温度で、A2-18 と同じ当てはめ（ln K(T) ＝ a ＋ b ÷ T、T_m0 ＝ −b ÷ a）と、恒等式の g(T) の平均 ḡ。
  ④ T* ＝ T_m0 − 0.10、h* ＝ −(a ＋ b ÷ T*) ÷ ḡ、第 2 段の温度は T*（0.01 に丸める）± 0.15 の 7 つ。
  ②′ ③ の融点にいちばん近い格子の温度が ② の真ん中と違えば、そこを真ん中に 7 つを選び直して ③ をもう一度（選び直しは 1 回だけ）。
  ḡ は窓の 7 つのうち g が出る温度（5 つ以上）の平均。
使えないとき（データの前に固定）：① か ③ の当てはめが出ない、T_all が 0.30〜0.80 の外、③ の T_m0 が窓の外、g が出る温度が 5 つ未満、ḡ ≤ 0、第 2 段の温度の最小が 0.10 未満。
  → そのモデルは「予言できない」として記録し、予備のモデル（Granite-3.1-8B）で第 1 段からやり直す（ゲート §2）。
A2-16 と同じ名前空間（scripts/a2_16_run.py の load16ns と scripts/a2_18_predict.py、scripts/a2_18_analyze.py の _fit）で実行する。
"""
import json
import numpy as np

A20_T1 = [round(0.20 + 0.05 * i, 2) for i in range(15)]                     # 第 1 段の温度：0.20〜0.90
A20_DT = -0.10
A20_GRID = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
A20_TALL = (0.30, 0.80)
A20_MINT = 0.10
A20_MIN_G = 5


def identity_g_t(Z0, temps):
    """identity_g（A2-18）と同じ式で、温度の並びを渡せるようにしたもの。Z0 の温度の軸は temps と同じ順。"""
    pick, q = Z0['pick'].astype(float), Z0['q'].astype(float)
    sig = (pick - q).sum(-1)
    out = []
    for k, T in enumerate(temps):
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


def choose_window(temps, T_all, n=7):
    """T_all にいちばん近い格子の温度を真ん中に 7 つ（端にかかるときは端から 7 つ）→ 番号の並び。"""
    c = int(np.argmin(np.abs(np.array(temps) - T_all)))
    lo = min(max(c - n // 2, 0), len(temps) - n)
    return list(range(lo, lo + n))


def predict_from_counts(tot, temps, gT_fn):
    """tot：(15, 16) 記事を足した窓の数え。gT_fn(idx) → 窓の温度での g(T)。→ (結果, 使えない理由 or None)。"""
    f_all = _fit(tot, temps)
    res = dict(temps1=list(temps), fit_all=dict(ok=bool(f_all.get('ok')), Tm=f_all.get('Tm'), use_idx=f_all.get('use_idx')))
    if not (f_all.get('ok') and np.isfinite(f_all['Tm'])):
        return res, '15 の温度で当てはめが出ない'
    T_all = float(f_all['Tm']); res['T_all'] = T_all
    if not (A20_TALL[0] <= T_all <= A20_TALL[1]):
        return res, f'仮の融点 {T_all:.3f} が {A20_TALL} の外'
    win = choose_window(temps, T_all)
    f = _fit(tot[win], [temps[i] for i in win])
    if f.get('ok') and np.isfinite(f['Tm']):                                  # 窓の融点で選び直すのは 1 回だけ（広い当てはめの曲がりで T_all がずれたとき）
        win2 = choose_window(temps, float(f['Tm']))
        if win2 != win:
            res['window_first'] = [temps[i] for i in win]
            win = win2
            f = _fit(tot[win], [temps[i] for i in win])
    tw = [temps[i] for i in win]
    res['window'] = tw
    if not (f.get('ok') and np.isfinite(f['Tm'])):
        return res, '窓の 7 つで当てはめが出ない'
    b = float(f['dH_vH']); Tm0 = float(f['Tm']); a = -b / Tm0
    if not (tw[0] <= Tm0 <= tw[-1]):
        return res, f'融点 {Tm0:.3f} が窓 {tw[0]}〜{tw[-1]} の外'
    gT = np.asarray(gT_fn(win), float)
    nfin = int(np.isfinite(gT).sum())
    gbar = float(np.nanmean(gT)) if nfin >= A20_MIN_G else float('nan')      # g が出ない温度（結晶の歩が 0 など）は平均から外す。5 つ未満なら予言しない
    k = np.array(f['use_idx'], int); tw_tot = tot[win]
    dm = float(np.mean(tw_tot[k, 11] / tw_tot[k, 7] - tw_tot[k, 10] / tw_tot[k, 5]))
    sC = Tm0 * dm / f['dH_cal']
    res.update(Tm0=Tm0, a=a, b=b, dH_cal=float(f['dH_cal']), dm=dm, g_T=gT.tolist(), gbar=gbar, s_clapeyron=sC, use_idx=f['use_idx'])
    if not np.isfinite(gbar):
        return res, f'g が出る温度が {nfin} 個（{A20_MIN_G} 個未満）'
    if not gbar > 0:
        return res, f'ḡ ＝ {gbar:.3f} が 0 以下'
    Ts = Tm0 + A20_DT
    h = float(-(a + b / Ts) / gbar)
    Tc = round(Ts, 2); grid = [round(Tc + x, 2) for x in A20_GRID]
    res['target'] = dict(dT=A20_DT, Tstar=Ts, hstar=h, grid=grid, r_clapeyron=float(sC * h / A20_DT))
    if grid[0] < A20_MINT:
        return res, f'第 2 段の温度の最小 {grid[0]} が {A20_MINT} 未満'
    return res, None


def predict_model20(path, name, expect_temps=A20_T1):
    Z = np.load(path)
    temps = [round(float(t), 4) for t in Z['temps']]
    assert expect_temps is None or temps == [round(float(t), 4) for t in expect_temps], (name, '温度の並びが違う', temps)
    assert Z['toks'].shape[1:] == (4, 40, 512) and Z['toks'].shape[0] == len(temps), (name, Z['toks'].shape)
    assert abs(float(Z['h'])) < 1e-12, (name, 'h ≠ 0')
    Z0 = {k: Z[k] for k in ('toks', 'pick', 'U', 'q', 'q0')}
    assert float(np.abs(Z0['q'].astype(float) - Z0['q0'].astype(float)).max()) == 0.0, (name, 'h ＝ 0 で q と q0 が違う')
    cw, _ = arm_counts(Z0['toks'], Z0['U'], Z0['pick'])
    tot = cw.sum(1)
    sub = lambda win: {k: v[win] for k, v in Z0.items()}
    res, why = predict_from_counts(tot, temps, lambda win: identity_g_t(sub(win), [temps[i] for i in win]))
    res['name'] = name; res['ok'] = why is None; res['why'] = why
    return res


def self_test_p20(gpre_base=None, hstar_json='results/A2_18/hstar.json'):
    # (1) 窓の選び方
    T = A20_T1
    assert choose_window(T, 0.61) == list(range(5, 12))                    # 0.60 が真ん中 → 0.45〜0.75
    assert choose_window(T, 0.21) == list(range(0, 7)) and choose_window(T, 0.89) == list(range(8, 15))
    assert choose_window([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], 0.6) == list(range(7))
    # (2) A2-18 の予言をそのまま再現する：G-pre の 7 つの温度（窓 ＝ 全部）で、hstar.json の T_m0・h*・格子と一致する
    if gpre_base is not None:
        P = json.load(open(hstar_json))
        for n in P:
            Z = np.load(f'{gpre_base}/{n}@h0.npz')
            Z0 = {k: Z[k] for k in ('toks', 'pick', 'U', 'q', 'q0')}
            cw, _ = arm_counts(Z0['toks'], Z0['U'], Z0['pick'])
            tot = cw.sum(1)
            f = _fit(tot, G16_TEMPS); b = float(f['dH_vH']); Tm0 = float(f['Tm']); a = -b / Tm0
            gT = identity_g_t(Z0, G16_TEMPS); gbar = float(np.mean(gT))
            Ts = Tm0 + A20_DT; h = float(-(a + b / Ts) / gbar)
            t = P[n]['targets']['-0.10']
            assert abs(Tm0 - P[n]['Tm0']) < 1e-12 and abs(h - t['hstar']) < 1e-12, (n, Tm0, h)
            assert [round(round(Ts, 2) + x, 2) for x in A20_GRID] == t['grid'], n
            assert np.allclose(gT, P[n]['g_T'], rtol=0, atol=1e-12), n
    # (3) 作り物の数え：融点 0.55 の 2 状態を作り、窓がそのまわりに来て、T_m0 が 0.55 の近くに出る
    rng = np.random.default_rng(5)
    tot = np.zeros((15, 16))
    for i, t in enumerate(T):
        K = np.exp(4.0 * (1 / t - 1 / 0.55))                                   # ln K ＝ 4 (1/T − 1/0.55)：低い温度で凍る向き（b ＞ 0）
        N0 = N1 = 20000.0
        n01 = 2000.0; n10 = n01 / K
        tot[i, :4] = [N0, n01, N1, n10]; tot[i, 4:10] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        tot[i, 5] = tot[i, 7] = 1.0; tot[i, 4] = 1.3; tot[i, 6] = 0.3; tot[i, 10:12] = [0.2, 0.9]
    res, why = predict_from_counts(tot, T, lambda win: np.full(len(win), 2.0))
    assert why is None and res['window'][3] == 0.55 and abs(res['Tm0'] - 0.55) < 1e-6, (why, res.get('window'), res.get('Tm0'))
    assert res['target']['hstar'] < 0 and res['target']['grid'][3] == 0.45
    # (3b) g が 2 つ出ない温度があっても予言する。3 つ出なければ止まる
    gn = lambda win: np.array([np.nan, np.nan] + [2.0] * (len(win) - 2))
    assert predict_from_counts(tot, T, gn)[1] is None
    assert predict_from_counts(tot, T, lambda win: np.array([np.nan] * 3 + [2.0] * (len(win) - 3)))[1] is not None
    # (3c) 選び直し：T_all が窓の端にずれても、窓の融点のまわりに選び直す
    r3, _ = predict_from_counts(tot, T, lambda win: np.full(len(win), 2.0))
    assert 'window_first' not in r3
    # (4) 使えないとき：融点が範囲の外
    tot2 = tot.copy()
    for i, t in enumerate(T):
        tot2[i, 3] = tot2[i, 1] / np.exp(4.0 * (1 / t - 1 / 0.25))
    _, why2 = predict_from_counts(tot2, T, lambda win: np.full(len(win), 2.0))
    assert why2 is not None, '範囲の外の融点を止められない'
    return 'OK'
