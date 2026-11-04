"""A2-15（ゲート docs/A2_15_GATE.md）の分析。CPU。データを開く前に固定する（2026-10-31 08:4x）。
scripts/a2_14_analyze.py を実行した名前空間（その前提の a2_13_analyze・onset・a2_8_vh・three・seg も）で実行する。
load14・tm_of・eq_word・pct・use_temps・salt_eff・counts2・seg_counts_by_doc を使い、モデルと家系の表だけを広げる。
直線は A2-14 と同じ Pile で学習していない 5 つ（SmolLM2-135M・360M・1.7B、OLMo-2-1B、Qwen2.5-1.5B）。
"""
import numpy as np

C14_FOLDER.update({m: 'A2_15' for m in ['gpt2-124m', 'gpt2-355m', 'gpt2-774m', 'gpt2-1.5b', 'bloom-560m', 'bloom-1b1', 'redpajama-3b']})
C15_FAM = dict(C14_FAM)
C15_FAM['gpt2'] = ['gpt2-124m', 'gpt2-355m', 'gpt2-774m', 'gpt2-1.5b']
C15_FAM['ln_other'] = ['redpajama-3b', 'bloom-560m', 'bloom-1b1']
C15_OFF = ('pythia', 'dedup', 'neo', 'gpt2', 'ln_other')
C15_MODELS = [m for f in C15_OFF + ('ref',) for m in C15_FAM[f]]
C15_BOOT, C15_SEED = 1000, 20261126


def line_offsets15(S, Tm, keep):
    ref = [m for m in C15_FAM['ref'] if m in keep and np.isfinite(S[m]) and np.isfinite(Tm[m])]
    if len(ref) < 4:
        return float('nan'), float('nan'), {f: float('nan') for f in C15_OFF}
    x = np.array([Tm[m] for m in ref]); y = np.array([S[m] for m in ref])
    a, b = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y, rcond=None)[0]
    off = {}
    for f in C15_OFF:
        ms = [m for m in C15_FAM[f] if m in keep]
        r = [S[m] - (a + b * Tm[m]) for m in ms]
        off[f] = float(np.mean(r)) if len(ms) >= 2 and all(np.isfinite(r)) else float('nan')
    return float(a), float(b), off


def analyze15(base):
    seg, two = {}, {}
    use0 = None
    for m in C15_MODELS:
        got = {a: load14(base, m, a) for a in C14_ARMS_B + C14_ARMS_S}
        us = {tuple(g[1]) for g in got.values()}
        assert len(us) == 1, m
        u = us.pop()
        use0 = u if use0 is None else use0
        assert u == use0, (m, '記事がモデルで違う')
        seg[m] = {a: seg_counts_by_doc(g[0]) for a, g in got.items()}
        two[m] = {a: counts2(got[a][0]) for a in C14_ARMS_B}
    nD = seg[C15_MODELS[0]]['none'].shape[1]
    res = {'models': {}}
    use, S, Tm, tuse = {}, {}, {}, {}
    for m in C15_MODELS:
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
    keep = [m for m in C15_MODELS if np.isfinite(S[m]) and np.isfinite(Tm[m])]
    res['dropped'] = [m for m in C15_MODELS if m not in keep]
    a, b, off = line_offsets15(S, Tm, keep)
    rng = np.random.default_rng(C15_SEED)
    bo = {f: [] for f in off}; bb = []
    for _ in range(C15_BOOT):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        Sb, Tb = {}, {}
        for m in keep:
            tn = sum(np.tensordot(seg[m][x], w, axes=([1], [0])) for x in C14_ARMS_B)
            ta = sum(np.tensordot(seg[m][x], w, axes=([1], [0])) for x in C14_ARMS_S)
            Sb[m] = salt_eff(tn, ta, 'AL', use[m])
            Tb[m] = tm_of(sum(np.tensordot(two[m][x], w, axes=([1], [0])) for x in C14_ARMS_B), tuse[m])[0]
        ab, b_, ob = line_offsets15(Sb, Tb, keep)
        bb.append(b_)
        for f in ob:
            bo[f].append(ob[f])
    res['line'] = dict(a=a, b=b, ci_b=pct(bb))
    res['offsets'] = {f: dict(point=off[f], ci=pct(bo[f]), fail=float(np.mean(~np.isfinite(bo[f])))) for f in off}
    for q, f in (('FQ1', 'gpt2'), ('FQ2', 'ln_other')):
        o = res['offsets'][f]
        res[q] = dict(family=f, word=eq_word(o['point'], o['ci'], o['fail']), **o)
    return res


def self_test_a15():
    Tm = {m: 0.5 + 0.01 * i for i, m in enumerate(C15_MODELS)}
    S = {m: 3.0 * Tm[m] - 1.4 for m in C15_MODELS}
    for m in C15_FAM['gpt2']:
        S[m] -= 0.3
    a, b, off = line_offsets15(S, Tm, C15_MODELS)
    assert abs(b - 3.0) < 1e-9 and abs(off['gpt2'] + 0.3) < 1e-9 and abs(off['ln_other']) < 1e-9 and abs(off['pythia']) < 1e-9
    assert len(C15_MODELS) == 24 and len(set(C15_MODELS)) == 24
    return 'OK'
