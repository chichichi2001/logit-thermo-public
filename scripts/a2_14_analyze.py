"""A2-14（ゲート docs/A2_14_GATE.md）の分析。CPU。データを開く前に固定する（2026-10-30 22:4x）。
scripts/a2_13_analyze.py を実行した名前空間（その前提の onset・a2_8_vh・three・seg も）で実行する。use_temps・salt_eff・counts2・C13_TEMPS・A8_TU を使う。

やること：モデルごとに S_AL（A2-13 と同じ定義）と 2 つの状態の融点 T_m（「なし」2 本）を出す。Pile で学習していない 5 つ
（SmolLM2-135M・360M・1.7B、OLMo-2-1B、Qwen2.5-1.5B）で、S_AL ＝ a ＋ b × T_m の直線を引き（最小二乗）、ほかのモデルの「ずれ」
（S_AL − 直線の値）を出す。家系ごとのずれ ＝ その家系のモデルのずれの平均。
"""
import numpy as np

C14_FOLDER = {**{m: 'A2_13' for m in ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
                                      'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b']},
              **{m: 'A2_14' for m in ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped',
                                      'gpt-neo-125m', 'gpt-neo-1.3b', 'gpt-neo-2.7b', 'olmo2-1b', 'qwen2.5-1.5b']}}
C14_FAM = {'pythia': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b'],
           'dedup': ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped'],
           'neo': ['gpt-neo-125m', 'gpt-neo-1.3b', 'gpt-neo-2.7b'],
           'ref': ['smollm2-135m', 'smollm2-360m', 'smollm2-1.7b', 'olmo2-1b', 'qwen2.5-1.5b']}
C14_MODELS = [m for f in ('pythia', 'dedup', 'neo', 'ref') for m in C14_FAM[f]]
C14_ARMS_B, C14_ARMS_S = ['none', 'none_b'], ['pl300', 'pl300_b']
C14_BOOT, C14_SEED = 1000, 20261125
C14_BAND = 0.15


def load14(base, m, arm):
    Z = np.load(f'{base}/{C14_FOLDER[m]}/{m}@{arm}.npz')
    temps = [round(float(t), 4) for t in Z['temps']]
    assert temps == C13_TEMPS, (m, temps)
    return Z['toks'], np.asarray(Z['use']).tolist()


def tm_of(cnt2_base, temps_use=None):
    """→ (T_m, 使った温度の番号)。選び直しでは、点のときの温度の番号を temps_use で渡す。"""
    f = fit_vh(cnt2_base, temps_use=temps_use)
    return (f['Tm'] if f.get('ok') and np.isfinite(f.get('Tm', np.nan)) else float('nan')), f['use_idx']


def line_offsets(S, Tm, keep):
    """keep：使うモデル。→ (a, b, {家系: ずれの平均})。直線は ref で S・T_m が出たもの 4 つ以上、家系は 2 つ以上のときだけ。"""
    ref = [m for m in C14_FAM['ref'] if m in keep and np.isfinite(S[m]) and np.isfinite(Tm[m])]
    if len(ref) < 4:
        return float('nan'), float('nan'), {f: float('nan') for f in ('pythia', 'dedup', 'neo')}
    x = np.array([Tm[m] for m in ref]); y = np.array([S[m] for m in ref])
    X = np.stack([np.ones_like(x), x], 1)
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    off = {}
    for f in ('pythia', 'dedup', 'neo'):
        ms = [m for m in C14_FAM[f] if m in keep]
        r = [S[m] - (a + b * Tm[m]) for m in ms]
        off[f] = float(np.mean(r)) if len(ms) >= 2 and all(np.isfinite(r)) else float('nan')
    return float(a), float(b), off


def eq_word(d, ci, fail):
    lo, hi = ci
    if fail > 0.1 or not (np.isfinite(d) and np.isfinite(lo) and np.isfinite(hi)):
        return 'はっきりしない'
    if hi < 0 and d <= -C14_BAND:
        return 'Pythia と同じく低い（塩で A が溶けにくい）'
    if lo > 0 and d >= C14_BAND:
        return 'ほかの家系より高い'
    if lo >= -C14_BAND and hi <= C14_BAND:
        return 'ほかの家系と同じくらい'
    return 'はっきりしない'


def analyze14(base):
    seg, two = {}, {}
    use0 = None
    for m in C14_MODELS:
        got = {a: load14(base, m, a) for a in C14_ARMS_B + C14_ARMS_S}
        us = {tuple(g[1]) for g in got.values()}
        assert len(us) == 1, m
        u = us.pop()
        use0 = u if use0 is None else use0
        assert u == use0, (m, '記事がモデルで違う')
        seg[m] = {a: seg_counts_by_doc(g[0]) for a, g in got.items()}
        two[m] = {a: counts2(got[a][0]) for a in C14_ARMS_B}
    nD = seg[C14_MODELS[0]]['none'].shape[1]
    res = {'models': {}}
    use, S, Tm, tuse = {}, {}, {}, {}
    for m in C14_MODELS:
        tn = sum(seg[m][a] for a in C14_ARMS_B).sum(1); ta = sum(seg[m][a] for a in C14_ARMS_S).sum(1)
        use[m] = use_temps(tn, ta, 'AL')
        S[m] = salt_eff(tn, ta, 'AL', use[m])
        Tm[m], tuse[m] = tm_of(sum(two[m][a] for a in C14_ARMS_B).sum(1))
        t0, t1 = seg[m]['none'].sum(1), seg[m]['none_b'].sum(1)
        k55 = C13_TEMPS.index(0.55); o = tn[k55, :3] / tn[k55, :3].sum()
        res['models'][m] = dict(S_AL=S[m], Tm=Tm[m], Tm_temps=tuse[m], use=use[m], AL_base=tn[:, 4].tolist(), AL_salt=ta[:, 4].tolist(),
                                seed_floor=salt_eff(t0, t1, 'AL', use_temps(t0, t1, 'AL')),
                                salt_LA=salt_eff(tn, ta, 'LA', use_temps(tn, ta, 'LA')),
                                frac_055=dict(L=float(o[0]), A=float(o[1]), C=float(o[2])))
    keep = [m for m in C14_MODELS if np.isfinite(S[m]) and np.isfinite(Tm[m])]
    res['dropped'] = [m for m in C14_MODELS if m not in keep]
    a, b, off = line_offsets(S, Tm, keep)
    rng = np.random.default_rng(C14_SEED)
    bo = {f: [] for f in off}; bab = []
    for _ in range(C14_BOOT):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        Sb, Tb = {}, {}
        for m in keep:
            tn = sum(np.tensordot(seg[m][x], w, axes=([1], [0])) for x in C14_ARMS_B)
            ta = sum(np.tensordot(seg[m][x], w, axes=([1], [0])) for x in C14_ARMS_S)
            Sb[m] = salt_eff(tn, ta, 'AL', use[m])
            Tb[m] = tm_of(sum(np.tensordot(two[m][x], w, axes=([1], [0])) for x in C14_ARMS_B), tuse[m])[0]
        ab, bb, ob = line_offsets(Sb, Tb, keep)
        bab.append((ab, bb))
        for f in ob:
            bo[f].append(ob[f])
    res['line'] = dict(a=a, b=b, ci_b=pct([x[1] for x in bab]))
    res['offsets'] = {f: dict(point=off[f], ci=pct(bo[f]), fail=float(np.mean(~np.isfinite(bo[f])))) for f in off}
    for q, f in (('EQ1', 'neo'), ('EQ2', 'dedup')):
        o = res['offsets'][f]
        res[q] = dict(family=f, word=eq_word(o['point'], o['ci'], o['fail']), **o)
    return res


def self_test_a14():
    assert eq_word(-0.3, [-0.5, -0.1], 0).startswith('Pythia と同じく') and eq_word(0.3, [0.1, 0.5], 0).startswith('ほかの家系より')
    assert eq_word(0.02, [-0.1, 0.12], 0).startswith('ほかの家系と同じ') and eq_word(-0.1, [-0.3, 0.1], 0) == 'はっきりしない'
    assert eq_word(-0.3, [-0.5, -0.1], 0.2) == 'はっきりしない'
    Tm = {m: 0.5 + 0.02 * i for i, m in enumerate(C14_MODELS)}
    S = {m: 3.0 * Tm[m] - 1.4 for m in C14_MODELS}
    for m in C14_FAM['neo']:
        S[m] -= 0.4
    a, b, off = line_offsets(S, Tm, C14_MODELS)
    assert abs(b - 3.0) < 1e-9 and abs(off['neo'] + 0.4) < 1e-9 and abs(off['dedup']) < 1e-9 and abs(off['pythia']) < 1e-9
    a2, b2, off2 = line_offsets(S, Tm, [m for m in C14_MODELS if m not in C14_FAM['ref'][:2]])
    assert not np.isfinite(b2)                      # 直線の元が 3 つでは引かない
    return 'OK'
