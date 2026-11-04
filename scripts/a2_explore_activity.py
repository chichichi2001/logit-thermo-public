"""探索 8.17：A2-10 のデータから、活量（ln a）・過剰エンタルピー・相互作用の W を拾う（2026-10-30 16:5x、CPU、判定しない）。
scripts/a2_10_analyze.py を（その前の部品ごと）実行したのと同じ名前空間で実行する（counts_arm10・fit_vh・C10A_*・A8_TU を使う）。

考え方：塩（偽薬）は液体の側にだけいるとすると、ln K(塩あり, T) − ln K(なし, T) ＝ ln a（液体の活量）＝ ln(1 − x) ＋ ln γ。
- ln a を 1/T に対して当てはめた傾き ＝ 溶媒の部分モル過剰エンタルピー h^E（ナット、1 回の相転移あたり。ln(1 − x) は x が温度で
  変わらなければ傾きに入らない）。これは ΔH_vH(塩あり) − ΔH_vH(なし) と同じもの。
- 偽薬の濃さは「1 位を禁止する歩の割合」で固定なので、取り上げる確率の量 m(T) は温度で変わる（低い温度ほど多い）。x ∝ m(T) とみなす
  補正版として、ln a ÷ m(T) の 1/T での傾きも出す（理想なら温度によらず一定 ＝ 傾き 0）。
- 濃さでの曲がり：同じ温度で ln a ＝ −κ x ＋ W' x² を当てはめる（x ＝ 濃さ）。理想なら W' ＝ 0。W' > 0（下がり方が鈍る）なら、
  溶けたものと溶媒が嫌い合う向き（正の過剰ギブズエネルギー）。正則溶液なら W' ＝ W ÷ T。
区間は記事の選び直し（500 回、種 20261118）。
"""
import numpy as np

ACT_BOOT, ACT_SEED = 500, 20261118


def lnK_tot(tot):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(tot[:, 1] / tot[:, 0]) - np.log(tot[:, 3] / tot[:, 2])


def wls(x, y, w):
    X = np.stack([np.ones_like(x), x], 1)
    return np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))


def activity_once(cnt, mass, w, use):
    """→ {腕: dict(h=傾き, h_mass=量あたりの傾き, lna=[温度ごと])}、{温度: (κ, W')}。use：共通に使う温度の番号。"""
    tot = {a: np.tensordot(cnt[a], w, axes=([1], [0])) for a in cnt}
    lk = {a: lnK_tot(tot[a]) for a in tot}
    T = np.array(C10A_TEMPS)
    out = {}
    for a in C10A_PL:
        lna = lk[a] - lk['none']
        idx = [i for i in use if np.isfinite(lna[i])]
        n = tot[a]; n0 = tot['none']
        var = 1 / n[idx, 1] + 1 / n[idx, 3] + 1 / n0[idx, 1] + 1 / n0[idx, 3]
        ww = 1 / var
        x = 1 / T[idx]
        h = wls(x, lna[idx], ww)[1] if len(idx) >= 3 else np.nan
        m = np.array([mass[a][i] for i in idx])
        hm = wls(x, lna[idx] / m, ww * m * m)[1] if len(idx) >= 3 else np.nan
        out[a] = dict(h=float(h), h_mass=float(hm), lna=[float(v) for v in lna], idx=idx)
    curv = {}
    for i in use:
        ys = [out[a]['lna'][i] for a in C10A_PL]
        if all(np.isfinite(ys)):
            xs = np.array([C10A_DOSE[a] for a in C10A_PL])
            A = np.stack([-xs, xs * xs], 1)
            kap, Wp = np.linalg.lstsq(A, np.array(ys), rcond=None)[0]
            curv[C10A_TEMPS[i]] = (float(kap), float(Wp))
    return out, curv


def activity10(d):
    res = {}
    for m in C10A_MODELS:
        cnt, mass = {}, {}
        for a in ['none'] + C10A_PL:
            Z = np.load(f'{d}/{m}@{a}.npz')
            cnt[a] = counts_arm10(Z['toks'])
            mass[a] = Z['g_ban_mass'].astype(np.float64).mean(axis=(1, 2, 3)).tolist()
        tot0 = {a: cnt[a].sum(1) for a in cnt}
        use = [i for i in range(len(C10A_TEMPS)) if all(tot0[a][i, 1] >= A8_MIN_EV and tot0[a][i, 3] >= A8_MIN_EV for a in cnt)]
        nD = cnt['none'].shape[1]
        pt, curv = activity_once(cnt, mass, np.ones(nD), use)
        rng = np.random.default_rng(ACT_SEED)
        bh = {a: [] for a in C10A_PL}; bhm = {a: [] for a in C10A_PL}; bW = []
        for _ in range(ACT_BOOT):
            w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
            o, c = activity_once(cnt, mass, w, use)
            for a in C10A_PL:
                if np.isfinite(o[a]['h']): bh[a].append(o[a]['h'])
                if np.isfinite(o[a]['h_mass']): bhm[a].append(o[a]['h_mass'])
            if c:
                bW.append(float(np.mean([v[1] for v in c.values()])))
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None
        res[m] = dict(use_temps=[C10A_TEMPS[i] for i in use], arms={a: dict(h=pt[a]['h'], h_ci=ci(bh[a]), h_mass=pt[a]['h_mass'], h_mass_ci=ci(bhm[a]),
                                                                           lna=pt[a]['lna']) for a in C10A_PL},
                      curv_by_T=curv, Wp_mean=float(np.mean([v[1] for v in curv.values()])) if curv else None, Wp_mean_ci=ci(bW),
                      kappa_mean=float(np.mean([v[0] for v in curv.values()])) if curv else None,
                      mass_by_T={a: mass[a] for a in C10A_PL})
    return res
