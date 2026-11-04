"""A2-11（ゲート docs/A2_11_GATE.md）：(C) 協同性の単位 n は写しの印の大きさのせいか、(D) T_m ÷ T_0 はモデルを越えてそろうか。CPU。
生成も読み直しもしない。scripts/a2_explore_onset.py（copy_marks）と scripts/a2_8_vh.py（phases・doc_counts・fit_vh・A8_TU・A8_SOURCES・
A8_MODELS・A8_T0）を実行したのと同じ名前空間で実行する。データを開く前に固定する。

(C) 材料：A2-8 の読み直し（`A2_8/{name}.npz`。トークンと U ＝ −log p₁ と HT）。写しの印を「g トークンの組がそれより前（続きの中）に出ていたら 1」
    として g ＝ 3・4・6 で作り直し、A2-8 と同じ決め方（中心の窓 32、入口 0.5・出口 0.25、凍る回・溶ける回がそれぞれ 10 回以上の温度）で
    T_m・ファントホッフの ΔH・熱量計の ΔH・n を出す。g ＝ 4 は A2-8 と同じ数になる（確かめる）。
(D) 材料：A2-2・A2-5・A2-6 の生成（`A2_2/{name}.npz` など。温度 0.6〜1.5 の 15 個 × サンプル 2 × 記事 40 × 512 歩）、探索 8.12 と同じ 25 本。
    写しの印 g ＝ 4 で T_m を出す。温度は 0.6 からしかないので、これを T_m⁺（0.6 以上の温度だけで出した T_m）と呼ぶ。
    A2-8 の 3 モデルでは、0.6 以上だけにすると T_m は 0.04〜0.05 高く出た（0.567 → 0.608 など）。T_0 は各実験の結果（固定の数）。
"""
import math
import numpy as np

A11_G = [3, 4, 6]
A11_RUNS = {
    'A2_2': ['pythia-1b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-inst', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b', 'smollm2-1.7b-inst'],
    'A2_5': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
             'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000'],
    'A2_6': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
             'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'],
}
A11_T22 = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
A11_T0 = {'A2_2/pythia-1b': (0.852045, 0.810311, 0.869373), 'A2_2/qwen2.5-0.5b': (0.857873, 0.827184, 0.887921),
          'A2_2/qwen2.5-0.5b-inst': (0.880881, 0.829638, 0.922522), 'A2_2/qwen2.5-1.5b': (0.910086, 0.861524, 0.956799),
          'A2_2/olmo2-1b': (0.800822, 0.775764, 0.856064), 'A2_2/smollm2-1.7b': (0.800452, 0.756107, 0.81886),
          'A2_2/smollm2-1.7b-inst': (0.814278, 0.788919, 0.857462), 'A2_5/pythia-70m': (0.847192, 0.830408, 0.880878),
          'A2_5/pythia-160m': (0.829054, 0.803702, 0.839713), 'A2_5/pythia-410m': (0.849135, 0.81199, 0.863422),
          'A2_5/pythia-1b': (0.841417, 0.81333, 0.870235), 'A2_5/pythia-1.4b': (0.845578, 0.803406, 0.8815),
          'A2_5/pythia-2.8b': (0.818831, 0.79674, 0.852551), 'A2_5/pythia-1b@step8000': (0.848006, 0.825036, 0.874433),
          'A2_5/pythia-1b@step24000': (0.82072, 0.819876, 0.8601), 'A2_5/pythia-1b@step64000': (0.858904, 0.825706, 0.875041),
          'A2_6/qwen2.5-0.5b': (0.847806, 0.832104, 0.879202), 'A2_6/qwen2.5-1.5b': (0.925402, 0.86499, 0.972219),
          'A2_6/qwen2.5-3b': (0.913934, 0.869474, 0.984563), 'A2_6/smollm2-135m': (0.809256, 0.76964, 0.831557),
          'A2_6/smollm2-360m': (0.794828, 0.75636, 0.827773), 'A2_6/smollm2-1.7b': (0.798879, 0.759564, 0.838186),
          'A2_6/qwen3-0.6b': (0.895509, 0.85676, 0.943189), 'A2_6/qwen3-1.7b': (0.925114, 0.88927, 1.01637),
          'A2_6/qwen3-4b': (0.928481, 0.901447, 1.01439)}
A11_PYTHIA_SIZE = [('A2_5/pythia-70m', 70e6), ('A2_5/pythia-160m', 160e6), ('A2_5/pythia-410m', 410e6), ('A2_5/pythia-1b', 1.0e9),
                   ('A2_5/pythia-1.4b', 1.4e9), ('A2_5/pythia-2.8b', 2.8e9)]
A11_RHO_CRIT = 0.886                  # 6 点のスピアマンの両側 5% の境（|ρ| ≥ 0.886）
A11_CQ1_BINS = [(0.8, '印と逆に小さくなる（0.8 倍未満）'), (1.25, 'ほぼ動かない（0.8〜1.25 倍）'), (1.5, '少し動く（1.25〜1.5 倍）'),
                (float('inf'), '印と一緒に大きくなる（1.5 倍以上）')]
A11_DQ1_BINS = [(0.05, 'そろう（ばらつき 5% 未満）'), (0.10, 'ゆるくそろう（5〜10%）'), (float('inf'), 'そろわない（10% 以上）')]


def copy_marks_g(seq, g):
    """続きのトークン列 → c。t 歩目で終わる g トークンの組が、それより前（続きの中）に出ていたら 1。g ＝ 4 は copy_marks と同じ。"""
    s = np.asarray(seq, dtype=np.int64)
    L = len(s)
    c = np.zeros(L, np.int8)
    if L < g:
        return c
    gr = np.stack([s[i:L - g + 1 + i] for i in range(g)], axis=1)
    _, first, inv = np.unique(gr, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    c[g - 1:] = (first[inv] < np.arange(len(gr))).astype(np.int8)
    return c


def build_counts_g(Z, g):
    """A2-8 の build_counts と同じ。写しの印だけ g トークンの組にする。"""
    nD = len(Z['use'])
    out = np.zeros((len(A8_TU), nD, 10))
    for key, _, temps in A8_SOURCES:
        toks, U, HT = Z[f'{key}_toks'], Z[f'{key}_U'], Z[f'{key}_HT']
        for k, T in enumerate(temps):
            ku = A8_TU.index(T)
            for j in range(nD):
                C = np.stack([copy_marks_g(toks[k, s, j], g) for s in range(toks.shape[1])])
                out[ku, j] += doc_counts(C, U[k, :, j].astype(np.float64), HT[k, :, j].astype(np.float64))
    return out


def counts_tokens(toks, temps, g=4):
    """toks：(nT, S, nD, L) → (17, nD, 10)（U・HT は 0）。"""
    nT, S, nD, L = toks.shape
    out = np.zeros((len(A8_TU), nD, 10))
    z = np.zeros((S, L))
    for k, T in enumerate(temps):
        ku = A8_TU.index(T)
        for j in range(nD):
            C = np.stack([copy_marks_g(toks[k, s, j], g) for s in range(S)])
            out[ku, j] += doc_counts(C, z, z)
    return out


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / math.sqrt((rx @ rx) * (ry @ ry)))


def bin_word(x, bins):
    if not np.isfinite(x):
        return 'はっきりしない'
    for e, w in bins:
        if x < e:
            return w


def analyze11(base):
    res = {'C': {}, 'D': {}}
    # (C)
    for name, _ in A8_MODELS:
        Z = dict(np.load(f'{base}/A2_8/{name}.npz'))
        row = {}
        for g in A11_G:
            p = fit_vh(build_counts_g(Z, g).sum(1))
            row[str(g)] = {k: p.get(k) for k in ('ok', 'Tm', 'dH_vH', 'dH_cal', 'n', 'use_idx')}
        res['C'][name] = row
    ratios = [res['C'][m]['6']['n'] / res['C'][m]['3']['n'] for m, _ in A8_MODELS
              if res['C'][m]['6'].get('n') and res['C'][m]['3'].get('n') and np.isfinite(res['C'][m]['6']['n']) and np.isfinite(res['C'][m]['3']['n'])
              and res['C'][m]['3']['n'] > 0]
    res['C_ratio_n6_n3'] = ratios
    res['CQ1'] = bin_word(float(np.median(ratios)), A11_CQ1_BINS) if len(ratios) >= 2 else 'はっきりしない'
    stable = 0
    for m, _ in A8_MODELS:
        tms = [res['C'][m][str(g)]['Tm'] for g in A11_G]
        if all(t is not None and np.isfinite(t) for t in tms) and max(tms) / min(tms) <= 1.05:
            stable += 1
    res['CQ2'] = stable
    # (D)
    for folder, names in A11_RUNS.items():
        for n in names:
            key = f'{folder}/{n}'
            Z = np.load(f'{base}/{folder}/{n}.npz')
            assert [float(t) for t in Z['temps']] == A11_T22, key
            p = fit_vh(counts_tokens(Z['toks'], A11_T22).sum(1))
            tm = p.get('Tm') if p['ok'] else float('nan')
            t0 = A11_T0[key][0]
            res['D'][key] = dict(Tm_plus=tm, dH_vH=p.get('dH_vH'), use_temps=[A8_TU[i] for i in p['use_idx']], T0=t0,
                                 ratio=(tm / t0 if np.isfinite(tm) else float('nan')))
    rs = [v['ratio'] for v in res['D'].values() if np.isfinite(v['ratio'])]
    res['D_n_runs'] = len(rs)
    res['D_ratio_mean'] = float(np.mean(rs)) if rs else float('nan')
    res['D_ratio_cv'] = float(np.std(rs) / np.mean(rs)) if len(rs) >= 5 else float('nan')
    res['DQ1'] = bin_word(res['D_ratio_cv'], A11_DQ1_BINS) if len(rs) >= 5 else 'はっきりしない'
    sz = [(s, res['D'][k]['Tm_plus']) for k, s in A11_PYTHIA_SIZE if np.isfinite(res['D'][k]['Tm_plus'])]
    if len(sz) == 6:
        rho = spearman(np.log([s for s, _ in sz]), [t for _, t in sz])
        res['D_pythia_rho'] = rho
        res['DQ2'] = '大きいほど T_m が下がる' if rho <= -A11_RHO_CRIT else '大きいほど T_m が上がる' if rho >= A11_RHO_CRIT else 'はっきりしない'
    else:
        res['D_pythia_rho'] = float('nan')
        res['DQ2'] = 'はっきりしない'
    return res


def self_test_a11():
    rng = np.random.default_rng(0)
    for _ in range(20):
        sq = rng.integers(0, 6, 80)
        assert (copy_marks_g(sq, 4) == copy_marks(sq)).all()
    assert copy_marks_g([1, 2, 3, 1, 2, 3], 3).tolist() == [0, 0, 0, 0, 0, 1]
    assert copy_marks_g([1, 2, 3, 1, 2, 3], 6).tolist() == [0] * 6
    assert spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) == -1.0
    assert bin_word(1.0, A11_CQ1_BINS) == 'ほぼ動かない（0.8〜1.25 倍）' and bin_word(2.0, A11_CQ1_BINS) == '印と一緒に大きくなる（1.5 倍以上）'
    assert bin_word(0.03, A11_DQ1_BINS) == 'そろう（ばらつき 5% 未満）' and bin_word(0.2, A11_DQ1_BINS) == 'そろわない（10% 以上）'
    assert set(A11_T0) == {f'{f}/{n}' for f, ns in A11_RUNS.items() for n in ns} and len(A11_T0) == 25
    assert all(T in A8_TU for T in A11_T22)
    return 'OK'
