"""探索：凝固点降下（束一的性質）の下見。2026-10-29 夕方。確かめには使わない（予想なし）。CPU。
scripts/a2_2_analyze.py（per_doc_feedback・feedback_stats）と scripts/a2_4_analyze.py（stats_fixed）を実行したのと同じ名前空間で実行する。

発想（オーナー 2026-10-29 18:06）：（オーナーの承認・発言。原文は非公開の記録にある）
問い：融点の下がり方 log(T_0(邪魔) ÷ T_0(なし)) は、邪魔の「種類」によらず「量」だけで決まるか。
  邪魔の種類：写しの禁止（n ＝ 8・6・4・3・2。n ＝ 4 は A2-4 の block）と、A2-4 の偽薬（写しかけでない歩の 1 位を同じ割合で禁止）。
  邪魔の量：禁止したトークンの、その温度の分布での確率の合計（g_ban_mass）の、全歩・全サンプル・全記事の平均。
            温度 0.8・0.85・0.9（T_0 のまわり）の平均を主に使い、温度ごとの値も残す。
T_0 は A2-7 と同じ（温度 8 つ、ε は「なし」で出したもの）。交点が出ないときは記録だけ。
"""
import math
import numpy as np

CL_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
CL_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
CL_MASS_T = [0.8, 0.85, 0.9]
CL_DOSES = [('n8', 'A2_7', '@n8'), ('n6', 'A2_7', '@n6'), ('n4', 'A2_4', '@block'), ('n3', 'A2_7', '@n3'), ('n2', 'A2_7', '@n2'),
            ('placebo', 'A2_4', '@placebo')]


def run_colligative(base):
    res = {}
    for m in CL_MODELS:
        Zn = np.load(f'{base}/A2_4/{m}@none.npz')
        t4 = [float(t) for t in Zn['temps']]
        kk = [t4.index(t) for t in CL_TEMPS]
        tf = Zn['tf'][0][:, kk]
        fd0 = per_doc_feedback(tf, Zn['g_ell'][kk])
        eps = feedback_stats(fd0, CL_TEMPS, np.random.default_rng(0))['eps']
        s0 = stats_fixed(fd0, CL_TEMPS, eps)
        row = {'none': dict(T0=s0['T0'], cls=s0['cls'], mass=0.0)}
        for dose, folder, suf in CL_DOSES:
            Z = np.load(f'{base}/{folder}/{m}{suf}.npz')
            tz = [float(t) for t in Z['temps']]
            kz = [tz.index(t) for t in CL_TEMPS]
            g = Z['g_ell'][kz]
            s = stats_fixed(per_doc_feedback(tf, g), CL_TEMPS, eps)
            bm = Z['g_ban_mass'][kz].astype(np.float64)                 # (8, S, nD, L)
            mass_T = bm.mean(axis=(1, 2, 3))
            mass = float(np.mean([mass_T[CL_TEMPS.index(t)] for t in CL_MASS_T]))
            top1 = Z['g_banned_top1'][kz].astype(np.float64).mean(axis=(1, 2, 3))
            row[dose] = dict(T0=s['T0'], cls=s['cls'], mass=mass, mass_by_T=mass_T.tolist(),
                             top1_rate_by_T=top1.tolist(),
                             dlogT0=(math.log(s['T0'] / s0['T0']) if np.isfinite(s['T0']) and np.isfinite(s0['T0']) else float('nan')))
        # 写しの禁止の点（交点が出たものだけ）で、原点を通る直線 dlogT0 ＝ −K × 量 を当てはめ、偽薬がそれに乗るかを見る
        pts = [(row[d]['mass'], row[d]['dlogT0']) for d, _, _ in CL_DOSES if d != 'placebo' and np.isfinite(row[d]['dlogT0'])]
        if len(pts) >= 2:
            x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
            K = float(-(x @ y) / (x @ x))
            pl = row['placebo']
            pred = -K * pl['mass']
            row['fit'] = dict(K=K, n_points=len(pts), placebo_pred=pred, placebo_obs=pl['dlogT0'],
                              resid_rms=float(np.sqrt(np.mean((y + K * x) ** 2))))
        res[m] = row
    return res


def self_test_colligative():
    x = np.array([0.1, 0.2, 0.4]); y = -0.5 * x
    K = float(-(x @ y) / (x @ x))
    assert abs(K - 0.5) < 1e-12
    return 'OK'
