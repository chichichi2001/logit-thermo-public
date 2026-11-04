"""A2-13（ゲート docs/A2_13_GATE.md）の分析。CPU。データを開く前に固定する。
scripts/a2_explore_onset.py（copy_marks）、scripts/a2_8_vh.py の phases・doc_counts・fit_vh と A8_ の定数、scripts/a2_explore_three.py（rates・vh_step・
three_state）、scripts/a2_explore_seg.py（seg_counts_by_doc）を実行したのと同じ名前空間で実行する。fit_vh が読む温度の並び A8_TU は、
ここで A2-13 の 7 つの温度に置きかえる。

いちばんの量：塩の効き S_AL。温度 0.4〜0.7 のそれぞれで v ＝ log(k_AL(偽薬 30%) ÷ k_AL(なし)) を出し、重み 1 ÷ (1/偽薬の A→L の回数 ＋ 1/なしの A→L の回数)
で平均したもの。k_AL ＝ A → L の回数 ÷ A にいた歩数（8.25 のものさし）。「なし」は none ＋ none_b、「偽薬 30%」は pl300 ＋ pl300_b を足す（サンプル 4）。
平均に入れる温度は、点のときに両方で A → L が 8 回以上あった温度（選び直しでも同じ温度を使う）。
"""
import math
import numpy as np

C13_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
C13_SALT_T = [0, 1, 2, 3, 4, 5, 6]
C13_PYTHIA = [('pythia-70m', 70e6), ('pythia-160m', 160e6), ('pythia-410m', 410e6), ('pythia-1b', 1.0e9),
              ('pythia-1.4b', 1.4e9), ('pythia-2.8b', 2.8e9)]
C13_SMOL = ['smollm2-135m', 'smollm2-360m', 'smollm2-1.7b']
C13_REF = ['olmo2-1b', 'qwen2.5-1.5b']          # 記録だけ。A2-10 のトークン（「偽薬 30%」はサンプル 2 だけ）を 7 つの温度に切って使う
C13_MODELS = [m for m, _ in C13_PYTHIA] + C13_SMOL + C13_REF
C13_BASE = ['none', 'none_b']
C13_SALT = {m: (['pl300'] if m in C13_REF else ['pl300', 'pl300_b']) for m in C13_MODELS}
C13_MIN = 8
C13_BOOT, C13_SEED = 1000, 20261124
C13_FLAT = 0.10         # PQ1：大きさ 10 倍あたりの傾きの「ほとんど変わらない」の幅
C13_FAM = 0.15          # PQ2：家系の平均の差の「同じくらい」の幅
C13_KEYS = {'LA': (3, 0), 'AL': (4, 1), 'AC': (5, 1), 'CA': (6, 2), 'LC': (7, 0), 'CL': (8, 2)}   # (回数の列, いた歩数の列)
A8_TU = C13_TEMPS


def load13(base, m, arm):
    folder = 'A2_10' if m in C13_REF else 'A2_13'
    Z = np.load(f'{base}/{folder}/{m}@{arm}.npz')
    temps = [round(float(t), 4) for t in Z['temps']]
    idx = [temps.index(t) for t in C13_TEMPS]
    return Z['toks'][idx], np.asarray(Z['use']).tolist()


def counts2(toks):
    """2 つの状態（A2-8 の相）の数え：(7, nD, 10)。U・HT は 0（T_m と ΔH_vH だけ出す）。"""
    nT, S, nD, L = toks.shape
    out = np.zeros((nT, nD, 10)); z = np.zeros((S, L))
    for k in range(nT):
        for j in range(nD):
            C = np.stack([copy_marks(toks[k, s, j]) for s in range(S)])
            out[k, j] += doc_counts(C, z, z)
    return out


def use_temps(tn, ta, key):
    i, _ = C13_KEYS[key]
    return [k for k in C13_SALT_T if tn[k, i] >= C13_MIN and ta[k, i] >= C13_MIN]


def salt_eff(tn, ta, key, use):
    """tn・ta：(7, 9) の足した数（「なし」と腕）。→ use の温度で、回数の重みで平均した log(速さの比)。出なければ nan。"""
    i, o = C13_KEYS[key]
    if len(use) == 0:
        return float('nan')
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.log((ta[use, i] / ta[use, o]) / (tn[use, i] / tn[use, o]))
        w = 1.0 / (1.0 / ta[use, i] + 1.0 / tn[use, i])
    if not (np.all(np.isfinite(v)) and np.all(w > 0)):
        return float('nan')
    return float(np.sum(w * v) / np.sum(w))


def slope_log(sizes, ys):
    x = np.log10(np.asarray(sizes)); y = np.asarray(ys)
    X = np.stack([np.ones_like(x), x], 1)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def pq1_word(b, ci, fail):
    lo, hi = ci
    if fail > 0.1 or not (np.isfinite(b) and np.isfinite(lo) and np.isfinite(hi)):
        return 'はっきりしない'
    if lo > 0 and b >= C13_FLAT:
        return '大きいほど上がる（塩で A が溶けやすくなる）'
    if hi < 0 and b <= -C13_FLAT:
        return '大きいほど下がる（塩で A が溶けにくくなる）'
    if lo >= -C13_FLAT and hi <= C13_FLAT:
        return 'ほとんど変わらない'
    return 'はっきりしない'


def pq2_word(d, ci, fail):
    lo, hi = ci
    if fail > 0.1 or not (np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi)):
        return 'はっきりしない'
    if lo > 0 and d >= C13_FAM:
        return 'SmolLM2 のほうが溶けやすい（Pythia の家系の癖らしい）'
    if hi < 0 and d <= -C13_FAM:
        return 'Pythia のほうが溶けやすい'
    if lo >= -C13_FAM and hi <= C13_FAM:
        return '同じくらい（家系の癖ではなさそう）'
    return 'はっきりしない'


def pct(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else [float('nan'), float('nan')]


def family_stats(S, keep):
    """S：{モデル: S_AL}、keep：使うモデル（点のときに S_AL が出たもの）。→ (Pythia の傾き, SmolLM2 の平均 − Pythia の平均)。
    Pythia が 4 つ未満、SmolLM2 が 2 つ未満なら、それぞれ nan。"""
    py = [(m, s) for m, s in C13_PYTHIA if m in keep]
    sm = [m for m in C13_SMOL if m in keep]
    ys = [S[m] for m, _ in py]; zs = [S[m] for m in sm]
    ok_y = len(py) >= 4 and all(np.isfinite(ys)); ok_z = len(sm) >= 2 and all(np.isfinite(zs))
    b = slope_log([s for _, s in py], ys) if ok_y else float('nan')
    d = float(np.mean(zs) - np.mean(ys)) if ok_y and ok_z else float('nan')
    return b, d


def analyze13(base):
    seg, two = {}, {}
    use0 = None
    for m in C13_MODELS:
        arms = C13_BASE + C13_SALT[m]
        got = {a: load13(base, m, a) for a in arms}
        toks = {a: g[0] for a, g in got.items()}
        uses = {tuple(g[1]) for g in got.values()}
        assert len(uses) == 1, (m, '腕ごとに記事が違う')
        u = uses.pop()
        if use0 is None:
            use0 = u
        assert u == use0, (m, '記事がモデルで違う（選び直しを同じ記事で組めない）')
        seg[m] = {a: seg_counts_by_doc(toks[a]) for a in arms}
        two[m] = {a: counts2(toks[a]) for a in arms}
    nD = seg[C13_MODELS[0]]['none'].shape[1]
    assert all(seg[m][a].shape[1] == nD for m in C13_MODELS for a in seg[m])
    res = {'models': {}, 'temps': C13_TEMPS}
    use = {}
    for m in C13_MODELS:
        cb = sum(seg[m][a] for a in C13_BASE); cs = sum(seg[m][a] for a in C13_SALT[m])
        tn, ta = cb.sum(1), cs.sum(1)
        use[m] = {k: use_temps(tn, ta, k) for k in C13_KEYS}
        t0, t1 = seg[m]['none'].sum(1), seg[m]['none_b'].sum(1)
        row = {'salt': {k: salt_eff(tn, ta, k, use[m][k]) for k in C13_KEYS}, 'use': use[m],
               'seed_floor_none': {k: salt_eff(t0, t1, k, use_temps(t0, t1, k)) for k in C13_KEYS},
               'AL_events_base': tn[:, 4].tolist(), 'AL_events_salt': ta[:, 4].tolist()}
        if len(C13_SALT[m]) == 2:
            s0, s1 = seg[m]['pl300'].sum(1), seg[m]['pl300_b'].sum(1)
            row['seed_floor_salt'] = {k: salt_eff(s0, s1, k, use_temps(s0, s1, k)) for k in C13_KEYS}
        k55 = C13_TEMPS.index(0.55); o = tn[k55, :3] / tn[k55, :3].sum()
        row['frac_0.55'] = dict(L=float(o[0]), A=float(o[1]), C=float(o[2]))
        T = {k: int(tn[:, C13_KEYS[k][0]].sum()) for k in C13_KEYS}
        row['trans_base'] = T
        row['C_exit_to_L'] = T['CL'] / max(1, T['CL'] + T['CA']); row['C_entry_from_L'] = T['LC'] / max(1, T['LC'] + T['AC'])
        st = three_state(cb, C13_TEMPS)
        row['steps'] = {s: {k: st[s].get(k) for k in ('ok', 'dH', 'Tm', 'dH_ci', 'idx')} for s in ('LA', 'AC')}
        lc = vh_step(tn, C13_TEMPS, 'LC', 'CL', 7, 8)
        row['steps']['LC'] = {k: lc.get(k) for k in ('ok', 'dH', 'Tm', 'idx')}
        f0 = fit_vh(sum(two[m][a] for a in C13_BASE).sum(1)); f1 = fit_vh(sum(two[m][a] for a in C13_SALT[m]).sum(1))
        row['two'] = dict(Tm_base=f0.get('Tm'), Tm_salt=f1.get('Tm'), dH_vH=f0.get('dH_vH'), use_base=f0['use_idx'], use_salt=f1['use_idx'])
        if f0.get('ok') and f1.get('ok') and np.isfinite(f0['Tm']) and np.isfinite(f1['Tm']):
            K = -math.log(f1['Tm'] / f0['Tm']) / 0.30
            row['two'].update(K=K, dH_col=f0['Tm'] / K if K > 0 else None,
                              col_over_vH=(f0['Tm'] / K) / f0['dH_vH'] if K > 0 and f0['dH_vH'] > 0 else None)
        res['models'][m] = row
    # 記事の選び直し（全部のモデル・腕で同じ選び方）
    S = {m: res['models'][m]['salt']['AL'] for m in C13_MODELS}
    keep = [m for m in C13_MODELS if np.isfinite(S[m])]
    res['dropped'] = [m for m in [x for x, _ in C13_PYTHIA] + C13_SMOL if m not in keep]
    rng = np.random.default_rng(C13_SEED)
    bs = {m: [] for m in C13_MODELS}; bb, bd = [], []
    for _ in range(C13_BOOT):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        cur = {}
        for m in C13_MODELS:
            tn = sum(np.tensordot(seg[m][a], w, axes=([1], [0])) for a in C13_BASE)
            ta = sum(np.tensordot(seg[m][a], w, axes=([1], [0])) for a in C13_SALT[m])
            cur[m] = salt_eff(tn, ta, 'AL', use[m]['AL'])
            bs[m].append(cur[m])
        b, d = family_stats(cur, keep)
        bb.append(b); bd.append(d)
    for m in C13_MODELS:
        res['models'][m]['S_AL'] = dict(point=S[m], ci=pct(bs[m]), fail=float(np.mean(~np.isfinite(bs[m]))))
    b, d = family_stats(S, keep)
    fb, fd = float(np.mean(~np.isfinite(bb))), float(np.mean(~np.isfinite(bd)))
    res['PQ1'] = dict(slope_per_decade=b, ci=pct(bb), fail=fb, word=pq1_word(b, pct(bb), fb))
    res['PQ2'] = dict(diff_smol_minus_pythia=d, ci=pct(bd), fail=fd, word=pq2_word(d, pct(bd), fd),
                      mean_pythia=float(np.mean([S[m] for m, _ in C13_PYTHIA if m in keep])) if keep else None,
                      mean_smol=float(np.mean([S[m] for m in C13_SMOL if m in keep])) if any(m in keep for m in C13_SMOL) else None)
    return res


def self_test_a13():
    """作り物で、塩の効きの平均・傾き・家系の差と、言葉の決め方を確かめる。"""
    assert pq1_word(0.2, [0.05, 0.35], 0.0).startswith('大きいほど上がる')
    assert pq1_word(-0.2, [-0.35, -0.05], 0.0).startswith('大きいほど下がる')
    assert pq1_word(0.02, [-0.08, 0.09], 0.0) == 'ほとんど変わらない'
    assert pq1_word(0.05, [-0.2, 0.3], 0.0) == 'はっきりしない' and pq1_word(0.3, [0.1, 0.5], 0.2) == 'はっきりしない'
    assert pq2_word(0.3, [0.1, 0.5], 0).startswith('SmolLM2') and pq2_word(-0.3, [-0.5, -0.1], 0).startswith('Pythia')
    assert pq2_word(0.02, [-0.1, 0.12], 0).startswith('同じくらい') and pq2_word(0.1, [-0.1, 0.3], 0) == 'はっきりしない'
    rng = np.random.default_rng(0)
    nD = 40
    def fake(eff):
        """(7, nD, 9) の数：L・A・C にいた歩、A → L は 1 記事 1 温度あたり 200 × exp(eff) 回。"""
        out = np.zeros((7, nD, 9))
        out[:, :, 0] = 3000; out[:, :, 1] = 2000; out[:, :, 2] = 500
        out[:, :, 4] = rng.poisson(200 * math.exp(eff), (7, nD))
        out[:, :, 3] = rng.poisson(30, (7, nD))
        return out
    tn, ta = fake(0.0).sum(1), fake(0.5).sum(1)
    u = use_temps(tn, ta, 'AL')
    assert u == C13_SALT_T and abs(salt_eff(tn, ta, 'AL', u) - 0.5) < 0.05, salt_eff(tn, ta, 'AL', u)
    S = {m: 0.2 * math.log10(s / 1e9) + 0.3 for m, s in C13_PYTHIA}
    S.update({m: 0.6 for m in C13_SMOL})
    allm = [m for m, _ in C13_PYTHIA] + C13_SMOL
    b, d = family_stats(S, allm)
    assert abs(b - 0.2) < 1e-9 and abs(d - (0.6 - np.mean([S[m] for m, _ in C13_PYTHIA]))) < 1e-12
    b2, d2 = family_stats(S, [m for m in allm if m != 'pythia-70m'])     # 1 つ抜けても、残りで出す
    assert abs(b2 - 0.2) < 1e-9 and np.isfinite(d2)
    b3, d3 = family_stats(S, ['pythia-70m', 'pythia-160m', 'pythia-410m'] + C13_SMOL)   # Pythia が 3 つでは出さない
    assert not np.isfinite(b3) and not np.isfinite(d3)
    return 'OK'
