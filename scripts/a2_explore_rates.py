"""探索：偽薬（塩）は凍る速さ k_f を下げるのか、溶ける速さ k_m を上げるのか（A2-10 のデータ、2026-10-30 17:3x、CPU、判定しない）。
scripts/a2_10_analyze.py を（その前の部品ごと）実行したのと同じ名前空間で実行する（counts_arm10・C10A_* を使う）。
温度ごとに、log(k_f(偽薬) ÷ k_f(なし)) と log(k_m(偽薬) ÷ k_m(なし)) を出す（ln K の変化 ＝ 前者 − 後者）。
区間は記事の選び直し（500 回、種 20261121）。凍る回・溶ける回が全部の腕でそれぞれ 10 回以上の温度だけ使う。
"""
import numpy as np

RT_BOOT, RT_SEED = 500, 20261121


def rates_once(cnt, w, use):
    tot = {a: np.tensordot(cnt[a], w, axes=([1], [0])) for a in cnt}
    with np.errstate(divide='ignore', invalid='ignore'):
        kf = {a: tot[a][:, 1] / tot[a][:, 0] for a in tot}
        km = {a: tot[a][:, 3] / tot[a][:, 2] for a in tot}
        occ = {a: tot[a][:, 2] / (tot[a][:, 0] + tot[a][:, 2]) for a in tot}
    out = {}
    for a in C10A_PL:
        df = np.log(kf[a] / kf['none'])[use]; dm = np.log(km[a] / km['none'])[use]
        out[a] = dict(dlog_kf=float(np.nanmean(df)), dlog_km=float(np.nanmean(dm)),
                      by_T_kf=np.log(kf[a] / kf['none']).tolist(), by_T_km=np.log(km[a] / km['none']).tolist())
    return out, kf, km, occ


def rates10(d):
    res = {}
    for m in C10A_MODELS:
        cnt = {a: counts_arm10(np.load(f'{d}/{m}@{a}.npz')['toks']) for a in ['none', 'none_b'] + C10A_PL}
        tot0 = {a: cnt[a].sum(1) for a in cnt}
        use = [i for i in range(len(C10A_TEMPS)) if all(tot0[a][i, 1] >= A8_MIN_EV and tot0[a][i, 3] >= A8_MIN_EV for a in cnt)]
        nD = cnt['none'].shape[1]
        pt, kf, km, occ = rates_once(cnt, np.ones(nD), use)
        # 種の床（なし（別の種））も同じように
        with np.errstate(divide='ignore', invalid='ignore'):
            fb = dict(dlog_kf=float(np.nanmean(np.log(kf['none_b'] / kf['none'])[use])), dlog_km=float(np.nanmean(np.log(km['none_b'] / km['none'])[use])))
        rng = np.random.default_rng(RT_SEED)
        bs = {a: ([], []) for a in C10A_PL}
        for _ in range(RT_BOOT):
            w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
            o, *_ = rates_once(cnt, w, use)
            for a in C10A_PL:
                bs[a][0].append(o[a]['dlog_kf']); bs[a][1].append(o[a]['dlog_km'])
        ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        res[m] = dict(use_temps=[C10A_TEMPS[i] for i in use], floor=fb,
                      arms={a: dict(dlog_kf=pt[a]['dlog_kf'], dlog_kf_ci=ci(bs[a][0]), dlog_km=pt[a]['dlog_km'], dlog_km_ci=ci(bs[a][1]),
                                    by_T_kf=pt[a]['by_T_kf'], by_T_km=pt[a]['by_T_km']) for a in C10A_PL},
                      kf_none=kf['none'].tolist(), km_none=km['none'].tolist(), occ={a: occ[a].tolist() for a in occ})
    return res
