"""探索 8.28：比熱の山と T_0（2026-10-31 09:2x、CPU、判定しない）。
材料：A2_explore/tcal_all.npz（A2-2・A2-5・A2-6 の 25 本、人の文章を読ませたときの Σ_t(⟨z⟩_T − z_y)、記事 40 × 61 の温度 0.4〜2.5）。
1 トークンあたりのエネルギー U(T) ＝ −(記事の平均) ÷ 512（E ＝ −logit、定数のずれは比熱に効かない）。比熱 C(T) ＝ dU/dT
（log T の格子で、となり 5 点の 2 次式の傾き ÷ T）。山の温度は、最大の点のまわり 3 点の 2 次式の頂点。T_0 は A2-11 の表（scripts/a2_11_probe_univ.py の A11_T0）。
"""
import ast
import numpy as np


def heatcap(path_npz, path_a11='scripts/a2_11_probe_univ.py'):
    t = ast.parse(open(path_a11).read())
    T0 = [ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'A11_T0'][0]
    Z = np.load(path_npz)
    out = {}
    for k in sorted({k.split('|')[0] for k in Z.files if '|' in k}):
        tc, TT = Z[k + '|tcal'], Z[k + '|tcal_T']
        fin = np.isfinite(tc).all(1)
        U = -tc[fin].mean(0) / 512.0
        lt = np.log(TT)
        C = np.full(len(TT), np.nan)
        for i in range(2, len(TT) - 2):
            c = np.polyfit(lt[i - 2:i + 3], U[i - 2:i + 3], 2)
            C[i] = (c[1] + 2 * c[0] * lt[i]) / TT[i]
        ip = int(np.nanargmax(C))
        c = np.polyfit(lt[ip - 1:ip + 2], C[ip - 1:ip + 2], 2)
        Tpk = float(np.exp(-c[1] / (2 * c[0])))
        out[k] = dict(T_peak=Tpk, C_max=float(np.nanmax(C)), T0=T0[k][0], ratio=Tpk / T0[k][0], n_docs=int(fin.sum()),
                      C_at_T0_over_max=float(np.interp(np.log(T0[k][0]), lt[2:-2], C[2:-2]) / np.nanmax(C)),
                      curve_T=TT[2:-2].tolist(), curve_C=C[2:-2].tolist())
    r = np.array([v['ratio'] for v in out.values()])
    a = np.array([[v['T_peak'], v['T0']] for v in out.values()])
    return dict(runs=out, ratio_mean=float(r.mean()), ratio_sd=float(r.std()), corr_Tpeak_T0=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
