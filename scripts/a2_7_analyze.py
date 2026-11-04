"""A2-7（ゲート docs/A2_7_GATE.md）の分析：滴定。データを開く前に固定する。CPU。
scripts/a2_2_analyze.py（pav・crossing_mono・feedback_stats・per_doc_feedback・NBOOT）、scripts/a2_4_analyze.py（stats_fixed・paired・
verdict_inf・copy_frac）、scripts/a2_5_analyze.py（spearman）を実行したのと同じ名前空間で実行する。

濃さの並び（薄い → 濃い）：なし・n8・n6・n4・n3・n2。なし と n4 は A2-4（A2_4/{モデル}@none.npz・@block.npz）の、A7_TEMPS の 8 つの温度だけを使う。
ε はモデルごとに 1 つ：「なし」の 8 つの温度のデータで feedback_stats が出す ε。
"""
import math
import numpy as np

A7_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
A7_DOSES = ['none', 'n8', 'n6', 'n4', 'n3', 'n2']
A7_TEMPS_A = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
A7_BOOT, A7_BOOT_SEED = 1000, 20261107
A7_FLOOR_T = 0.6          # 「大きく下がる（格子の外）」のときに、並びの計算で使う T_0（格子の下の端）


def load_fds(d4, d7, m):
    Zn = np.load(f'{d4}/{m}@none.npz')
    t4 = [float(t) for t in Zn['temps']]
    kk = [t4.index(t) for t in A7_TEMPS_A]
    tf = Zn['tf'][0][:, kk]                                        # (nD, 8, L)
    use = [int(u) for u in Zn['use']]
    fds, extra = {}, {}
    for dose in A7_DOSES:
        if dose in ('none', 'n4'):
            Z = Zn if dose == 'none' else np.load(f'{d4}/{m}@block.npz')
            g = Z['g_ell'][kk]
            toks = Z['toks'][kk]
        else:
            Z = np.load(f'{d7}/{m}@{dose}.npz')
            assert [float(t) for t in Z['temps']] == A7_TEMPS_A, '温度の並びが違います'
            g = Z['g_ell']; toks = Z['toks']
        assert [int(u) for u in Z['use']] == use, ('使った記事が違います', m, dose)
        fds[dose] = per_doc_feedback(tf, g)
        extra[dose] = dict(toks=toks, prompts=Z['prompts'])
    return fds, extra


def dose_T0(fd, eps):
    s = stats_fixed(fd, A7_TEMPS_A, eps)
    T0 = s['T0']
    rank_T0 = T0 if np.isfinite(T0) else (A7_FLOOR_T if s['cls'] == 'below' else float('nan'))
    return s, rank_T0


def analyze7(d4, d7):
    res = {'models': {}}
    for m in A7_MODELS:
        fds, extra = load_fds(d4, d7, m)
        eps = feedback_stats(fds['none'], A7_TEMPS_A, np.random.default_rng(0))['eps']
        row = {'eps': eps, 'doses': {}}
        ys = []
        for dose in A7_DOSES:
            s, rT = dose_T0(fds[dose], eps)
            k07 = A7_TEMPS_A.index(0.7)
            toks, prm = extra[dose]['toks'][k07], extra[dose]['prompts']
            c4 = float(np.mean([copy_frac(prm[j], toks[s_, j], 4) for s_ in range(toks.shape[0]) for j in range(toks.shape[1])]))
            c3 = float(np.mean([copy_frac(prm[j], toks[s_, j], 3) for s_ in range(toks.shape[0]) for j in range(toks.shape[1])]))
            d = dict(T0=s['T0'], cls=s['cls'], T0_rank=rT, copy4_07=c4, copy3_07=c3)
            if dose != 'none':
                p = paired(fds[dose], fds['none'], A7_TEMPS_A, eps, seed=1)
                d.update(d_vs_none=p, word=verdict_inf(p, s['cls'], True))
            row['doses'][dose] = d
            ys.append(rT)
        # TQ2：濃くするほど T_0 が下がるか（6 点の並び）
        xs = list(range(len(A7_DOSES)))
        r = spearman(xs, ys) if np.all(np.isfinite(ys)) else float('nan')
        rng = np.random.default_rng(A7_BOOT_SEED)
        nD = fds['none'].shape[0]
        neg, okn = 0, 0
        for _ in range(A7_BOOT):
            idx = rng.integers(0, nD, nD)
            yb = [dose_T0(fds[dz][idx], eps)[1] for dz in A7_DOSES]
            if np.all(np.isfinite(yb)):
                okn += 1
                neg += int(spearman(xs, yb) < 0)
        ok_frac = okn / A7_BOOT
        word = ('濃いほど下がる' if (np.isfinite(r) and r <= -0.9 + 1e-9 and ok_frac >= 0.9 and neg / A7_BOOT >= 0.95)
                else 'はっきりしない')
        row.update(rho=r, boot_ok=ok_frac, frac_neg=neg / A7_BOOT, TQ2_word=word)
        res['models'][m] = row
    res['TQ1_olmo'] = res['models']['olmo2-1b']['doses']['n2']['word']
    res['TQ1_qwen'] = res['models']['qwen2.5-1.5b']['doses']['n2']['word']
    res['TQ1_pythia_record'] = res['models']['pythia-1b']['doses']['n2']['word']
    res['TQ2'] = sum(res['models'][m]['TQ2_word'] == '濃いほど下がる' for m in A7_MODELS)
    return res


def self_test_a27_analyze():
    # 作り物：T_0 が濃さとともに下がる記事ごとの f（8 温度）。濃さごとに交点を 0.86 → 0.70 へ
    rng = np.random.default_rng(0)
    t = np.array(A7_TEMPS_A)
    def fd_of(T0, n=40, noise=0.05):
        F = np.where(t < T0, (t - T0) * 12, (t - T0) * 5)
        return F[None, :] + rng.normal(0, noise, (n, len(t)))
    T0s = [0.86, 0.84, 0.81, 0.78, 0.74, 0.70]
    fds = {d: fd_of(T) for d, T in zip(A7_DOSES, T0s)}
    eps = feedback_stats(fds['none'], A7_TEMPS_A, np.random.default_rng(0))['eps']
    ys = [dose_T0(fds[d], eps)[1] for d in A7_DOSES]
    assert spearman(list(range(6)), ys) <= -0.9, ys
    p = paired(fds['n2'], fds['none'], A7_TEMPS_A, eps, seed=1)
    assert verdict_inf(p, 'cross', True) == '大きく下がる', p
    # 格子の外（全部の温度で F > ε）は T_0 ＝ 0.6 として並べる
    s, rT = dose_T0(fd_of(0.4), eps)
    assert s['cls'] == 'below' and rT == A7_FLOOR_T
    return 'OK'
