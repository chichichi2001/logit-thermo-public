"""探索 8.30 の分析：場 h で融点はどれだけ動くか、クラウジウス・クラペイロンの式と合うか（CPU、判定しない）。
データを開く前に書いた（2026-10-31 13:1x）。scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases・doc_counts・fit_vh）・
a2_13_analyze.py（counts2・A8_TU ＝ C13_TEMPS）を実行したのと同じ名前空間で実行する。

考え方（R ＝ 1）：p ∝ exp((logit ＋ h × [写しの候補]) ÷ T) なので、エネルギーに −h × M（M ＝ 写しの候補を選んだ歩の数）が足される。
2 つの状態の ln K ＝ ΔH ÷ T − ΔS ＋ h × ΔM ÷ T（ΔM ＝ 1 回の相転移で増える M）。よって
  (1) T_m(h) ＝ (ΔH ＋ h ΔM) ÷ ΔS、dT_m/dh ＝ ΔM ÷ ΔS（クラペイロン）
  (2) 温度ごとに ∂ln K/∂h ＝ ΔM ÷ T、つまり ΔM_速さ ＝ T × ∂ln K/∂h
ΔM の見積もり（場をかけない h ＝ 0 の腕だけから）：ΔM_数え ＝ n × Δm。Δm ＝ 結晶の歩で候補を選んだ割合 − 液体の歩で候補を選んだ割合（1 トークンあたり）、
n ＝ 協同性の単位（A2-8：Pythia-1B 3.26、OLMo-2-1B 2.83）。
予想の傾き：dT_m/dh（予想）＝ n × Δm × T_m(0) ÷ ΔH_vH(0)。これを、3 つの h の T_m に引いた直線の傾き（測った値）と比べる。
区間：記事の選び直し 1,000 回（種 20261132、3 つの腕で同じ選び方。温度は点のときと同じものを使う）。
"""
import numpy as np

FA_N = {'pythia-1b': 3.264, 'olmo2-1b': 2.826}           # A2-8 の n（results/A2_8/a2_8_summary.json の point）
FA_H = [('hm03', -0.3), ('h0', 0.0), ('hp03', 0.3)]
FA_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
FA_BOOT, FA_SEED = 1000, 20261132


def load_field(base, name, arm):
    Z = np.load(f'{base}/{name}@{arm}.npz')
    assert [round(float(t), 4) for t in Z['temps']] == FA_TEMPS, (name, arm)
    return Z['toks'], Z['pick'], np.asarray(Z['use']).tolist(), float(Z['h'])


def field_counts(toks, pick):
    """toks・pick：(7, S, nD, L) → cnt (7, nD, 10)（counts2 と同じ）, pk (7, nD, 4) ＝ [液体で候補を選んだ歩, 液体の歩, 結晶で候補を選んだ歩, 結晶の歩]。"""
    nT, S, nD, L = toks.shape
    cnt = counts2(toks)
    pk = np.zeros((nT, nD, 4))
    for k in range(nT):
        for j in range(nD):
            for s in range(S):
                st, t = phases(copy_marks(toks[k, s, j]))
                p = pick[k, s, j][t]
                pk[k, j] += [p[st == 0].sum(), (st == 0).sum(), p[st == 1].sum(), (st == 1).sum()]
    return cnt, pk


def field_point(cnts, pks, n, temps_use=None, k_use=None):
    """cnts・pks：{腕: (7, nD, …)}（記事の軸は足す前）→ 点の値。temps_use：腕ごとの使う温度（選び直しで点と同じにする）。"""
    out = {}
    fits = {}
    for arm, h in FA_H:
        tot = cnts[arm].sum(1)
        f = fit_vh(tot, temps_use=None if temps_use is None else temps_use[arm])
        fits[arm] = f
        out[arm] = dict(h=h, Tm=f.get('Tm', float('nan')) if f.get('ok') else float('nan'),
                        dH_vH=f.get('dH_vH', float('nan')), dS_vH=f.get('dS_vH', float('nan')), use_idx=f['use_idx'], lnK=f['lnK'])
    hs = np.array([h for _, h in FA_H]); tms = np.array([out[a]['Tm'] for a, _ in FA_H])
    slope = float(np.polyfit(hs, tms, 1)[0]) if np.all(np.isfinite(tms)) else float('nan')
    p0 = pks['h0'].sum(1)
    k0 = out['h0']['use_idx'] if k_use is None else k_use
    with np.errstate(divide='ignore', invalid='ignore'):
        m_liq = p0[k0, 0].sum() / p0[k0, 1].sum(); m_cry = p0[k0, 2].sum() / p0[k0, 3].sum()
    dm = float(m_cry - m_liq)
    pred = n * dm * out['h0']['Tm'] / out['h0']['dH_vH'] if out['h0']['dH_vH'] > 0 else float('nan')
    # 速さの確かめ：温度ごとの T × (ln K(＋0.3) − ln K(−0.3)) ÷ 0.6（両方の腕で使える温度）
    lp, lm = np.array(out['hp03']['lnK']), np.array(out['hm03']['lnK'])
    both = sorted(set(out['hp03']['use_idx']) & set(out['hm03']['use_idx']))
    dM_kin_T = {FA_TEMPS[k]: float(FA_TEMPS[k] * (lp[k] - lm[k]) / 0.6) for k in both}
    dM_kin = float(np.mean(list(dM_kin_T.values()))) if dM_kin_T else float('nan')
    return dict(arms=out, slope=slope, dm=dm, m_liq=float(m_liq), m_cry=float(m_cry), n=n, dM_count=float(n * dm), pred_slope=float(pred),
                ratio=float(slope / pred) if np.isfinite(pred) and pred != 0 else float('nan'),
                dM_kin_T=dM_kin_T, dM_kin=dM_kin, k0=list(map(int, k0)))


def analyze_field(base, models=('pythia-1b', 'olmo2-1b')):
    res = {}
    for name in models:
        cnts, pks, uses = {}, {}, []
        for arm, h in FA_H:
            toks, pick, use, hh = load_field(base, name, arm)
            assert abs(hh - h) < 1e-9, (name, arm, hh)
            cnts[arm], pks[arm] = field_counts(toks, pick); uses.append(use)
        assert uses[0] == uses[1] == uses[2], name
        pt = field_point(cnts, pks, FA_N[name])
        tu = {a: pt['arms'][a]['use_idx'] for a, _ in FA_H}
        nD = cnts['h0'].shape[1]
        rng = np.random.default_rng(FA_SEED)
        bs = {'slope': [], 'pred_slope': [], 'ratio': [], 'dm': [], 'dM_kin': []}
        for _ in range(FA_BOOT):
            idx = rng.integers(0, nD, nD)
            b = field_point({a: cnts[a][:, idx] for a in cnts}, {a: pks[a][:, idx] for a in pks}, FA_N[name], temps_use=tu, k_use=pt['k0'])
            for k in bs:
                bs[k].append(b[k])
        pt['ci'] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
        pt['boot_fail'] = float(np.mean(~np.isfinite(np.array(bs['slope']))))
        res[name] = pt
    return res


def self_test_field_an():
    """作り物：液体はでたらめ、結晶は周期 23 の写し。pick は結晶で 1、液体で 0.1 の割合。→ Δm ≈ 0.9 になること。"""
    rng = np.random.default_rng(5)
    toks = np.zeros((7, 2, 3, 512), np.int64); pick = np.zeros_like(toks, dtype=bool)
    for k in range(7):
        for s in range(2):
            for j in range(3):
                seq = list(rng.integers(100, 30000, 150)) + list(np.tile(rng.integers(100, 30000, 23), 20)[:200])
                seq += list(rng.integers(100, 30000, 512 - len(seq)))
                toks[k, s, j] = seq
                pick[k, s, j] = rng.random(512) < 0.1
                pick[k, s, j, 150 + 23:350] = True
    cnt, pk = field_counts(toks, pick)
    p = pk.sum((0, 1))
    m_liq, m_cry = p[0] / p[1], p[2] / p[3]
    assert m_liq < 0.2 and m_cry > 0.8, (m_liq, m_cry)
    assert cnt[:, :, 1].sum() == 42 and cnt[:, :, 3].sum() == 42, (cnt[:, :, 1].sum(), cnt[:, :, 3].sum())
    return 'OK'
