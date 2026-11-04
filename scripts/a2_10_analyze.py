"""A2-10（ゲート docs/A2_10_GATE.md）の分析。CPU。データを開く前に固定する。
scripts/a2_2_analyze.py（if False: 版。crossing_mono・per_doc_feedback・feedback_stats）、scripts/a2_4_analyze.py（stats_fixed）、
scripts/a2_7b_analyze.py（slope0・log_q）、scripts/a2_explore_onset.py（copy_marks）、scripts/a2_8_vh.py（phases・doc_counts・fit_vh）を
実行したのと同じ名前空間で実行する。fit_vh が読む温度の並び A8_TU は、ここで A2-10 の 14 の温度に置きかえる。
"""
import math
import numpy as np

C10A_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
C10A_T0_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]      # T_0 はこの 8 つだけで（A2-7c と同じ格子。ε が低い温度でふくらまないように）
C10A_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
C10A_ARMS = ['none', 'none_b', 'pl075', 'pl150', 'pl300']
C10A_PL = ['pl075', 'pl150', 'pl300']
C10A_DOSE = {'none': 0.0, 'none_b': 0.0, 'pl075': 0.075, 'pl150': 0.15, 'pl300': 0.30}
C10A_BOOT, C10A_SEED = 1000, 20261117
C10A_R = 1.5
C10A_Q1_BINS = [(0.05, 'ほとんど下がらない（log で 0.05 未満、5% 未満）'), (0.15, '少し下がる（log で 0.05〜0.15、5〜14%）'),
                (0.30, 'かなり下がる（log で 0.15〜0.30、14〜26%）'), (float('inf'), '大きく下がる（log で 0.30 以上、26% 以上）')]
A8_TU = C10A_TEMPS


def counts_arm10(toks):
    """toks：(14, S, nD, L) → (14, nD, 10)（U・HT は 0。T_m だけ出す）。"""
    nT, S, nD, L = toks.shape
    out = np.zeros((nT, nD, 10))
    z = np.zeros((S, L))
    for k in range(nT):
        for j in range(nD):
            C = np.stack([copy_marks(toks[k, s, j]) for s in range(S)])
            out[k, j] += doc_counts(C, z, z)
    return out


def load10(d, m):
    Zt = np.load(f'{d}/{m}@tf.npz')
    assert [float(t) for t in Zt['temps']] == C10A_TEMPS
    i0 = [C10A_TEMPS.index(t) for t in C10A_T0_TEMPS]
    tf = Zt['tf'][:, i0]
    fds, cnt, mass, top1 = {}, {}, {}, {}
    for a in C10A_ARMS:
        Z = np.load(f'{d}/{m}@{a}.npz')
        assert [float(t) for t in Z['temps']] == C10A_TEMPS
        fds[a] = per_doc_feedback(tf, Z['g_ell'][i0])
        cnt[a] = counts_arm10(Z['toks'])
        bm = Z['g_ban_mass'].astype(np.float64)
        mass[a] = bm.mean(axis=(1, 2, 3)).tolist()                 # 温度ごとの、取り上げた確率の量
        top1[a] = Z['g_banned_top1'].astype(np.float64).mean(axis=(1, 2, 3)).tolist()
    return fds, cnt, mass, top1


def fit_once10(fds, cnt, eps, w, use):
    """w：記事の重み（選び直し）。use：{腕: 使う温度の番号}（点のときに、その腕の凍る回・溶ける回がそれぞれ 10 回以上の温度）。
    → {腕: (y_m, y_0)}、全部の腕で T_m と T_0 が出たか。"""
    idx = np.repeat(np.arange(len(w)), w.astype(int))
    T0n = crossing_mono(np.median(fds['none'][idx], 0), C10A_T0_TEMPS, eps)
    fm = {a: fit_vh(np.tensordot(cnt[a], w, axes=([1], [0])), temps_use=use[a]) for a in C10A_ARMS}
    Tmn = fm['none']['Tm'] if fm['none']['ok'] else float('nan')
    out, ok_m, ok_0 = {}, True, True
    for a in C10A_ARMS:
        if a == 'none':
            continue
        tm = fm[a]['Tm'] if fm[a]['ok'] else float('nan')
        t0 = crossing_mono(np.median(fds[a][idx], 0), C10A_T0_TEMPS, eps)
        ym = math.log(tm / Tmn) if np.isfinite(tm) and np.isfinite(Tmn) else float('nan')
        y0 = math.log(t0 / T0n) if np.isfinite(t0) and np.isfinite(T0n) else float('nan')
        ok_m &= np.isfinite(ym); ok_0 &= np.isfinite(y0)
        out[a] = (ym, y0)
    return out, ok_m, ok_0


def slopes10(out):
    """偽薬の 3 本で、原点を通る直線の傾き K_m・K_0（y ＝ −K × 濃さ）。T_0 は出た腕だけ（2 本未満なら nan）。"""
    xm = [C10A_DOSE[a] for a in C10A_PL if np.isfinite(out[a][0])]
    ym = [out[a][0] for a in C10A_PL if np.isfinite(out[a][0])]
    x0 = [C10A_DOSE[a] for a in C10A_PL if np.isfinite(out[a][1])]
    y0 = [out[a][1] for a in C10A_PL if np.isfinite(out[a][1])]
    Km = slope0(xm, ym) if len(xm) == 3 else float('nan')
    K0 = slope0(x0, y0) if len(x0) >= 2 else float('nan')
    return Km, K0, x0


def q1_word10(y, ci):
    if not (np.isfinite(y) and ci is not None and np.isfinite(ci[0])):
        return 'はっきりしない'
    if ci[1] >= 0:
        return 'はっきりしない（0 をまたぐ）' if ci[0] < 0 else '上がる'
    for edge, w in C10A_Q1_BINS:
        if -y < edge:
            return w


def q2_word10(lr, ci, fail):
    if fail > 0.1 or not np.isfinite(lr) or ci is None:
        return 'はっきりしない'
    lo, hi = ci
    if hi < 0 and lr <= -math.log(C10A_R) + 1e-12:
        return 'T_m のほうが速く下がる'
    if lo >= -math.log(C10A_R) and hi <= math.log(C10A_R):
        return '同じくらい'
    if lo > 0 and lr >= math.log(C10A_R) - 1e-12:
        return 'T_0 のほうが速く下がる（海水型）'
    return 'はっきりしない'


def analyze10(d):
    res = {'models': {}, 'temps': C10A_TEMPS, 'T0_temps': C10A_T0_TEMPS}
    for m in C10A_MODELS:
        fds, cnt, mass, top1 = load10(d, m)
        eps = feedback_stats(fds['none'], C10A_T0_TEMPS, np.random.default_rng(0))['eps']
        use = {a: fit_vh(cnt[a].sum(1))['use_idx'] for a in C10A_ARMS}   # 濃い偽薬では T_m が下がり、使える温度がずれるので腕ごとに
        nD = cnt['none'].shape[1]
        out, okm, ok0 = fit_once10(fds, cnt, eps, np.ones(nD), use)
        Km, K0, x0 = slopes10(out)
        lr = log_q(K0, Km)                                          # log(K_0 ÷ K_m)
        row = dict(eps=eps, use_temps={a: [C10A_TEMPS[i] for i in use[a]] for a in C10A_ARMS}, K_m=Km, K_0=K0, logR=lr, arms_T0=x0,
                   arms={a: dict(dose=C10A_DOSE[a], mass_by_T=mass[a], top1_by_T=top1[a],
                                 Tm=(fit_vh(cnt[a].sum(1), temps_use=use[a]).get('Tm')), T0=stats_fixed(fds[a], C10A_T0_TEMPS, eps)['T0'],
                                 dH_vH=fit_vh(cnt[a].sum(1), temps_use=use[a]).get('dH_vH'),
                                 y_m=(out[a][0] if a != 'none' else 0.0), y_0=(out[a][1] if a != 'none' else 0.0)) for a in C10A_ARMS})
        rng = np.random.default_rng(C10A_SEED)
        b_ym, b_lr, fail_m, fail_r = {a: [] for a in C10A_ARMS if a != 'none'}, [], 0, 0
        for _ in range(C10A_BOOT):
            w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
            o, om, o0 = fit_once10(fds, cnt, eps, w, use)
            if not om:
                fail_m += 1
            for a in b_ym:
                if np.isfinite(o[a][0]):
                    b_ym[a].append(o[a][0])
            km, k0, _ = slopes10(o)
            v = log_q(k0, km)
            if not (np.isfinite(km) and np.isfinite(k0)):
                fail_r += 1
            elif not np.isnan(v):
                b_lr.append(v)
        pc = lambda v: ([float(np.percentile(v, 2.5, method='lower')), float(np.percentile(v, 97.5, method='higher'))] if len(v) >= 20 else None)
        row['ci_y_m'] = {a: pc(v) for a, v in b_ym.items()}
        row['ci_logR'] = pc(b_lr)
        row['boot_fail_m'], row['boot_fail_R'] = fail_m / C10A_BOOT, fail_r / C10A_BOOT
        row['Q1_word'] = q1_word10(out['pl300'][0], row['ci_y_m']['pl300']) if row['boot_fail_m'] <= 0.1 else 'はっきりしない'
        row['Q2_word'] = q2_word10(lr, row['ci_logR'], row['boot_fail_R'])
        # 記録：凝固点降下の式から逆算した ΔH（x ＝ 濃さ、x ＝ T_m のまわりで取り上げた確率の量）
        Tmn = fit_vh(cnt['none'].sum(1), temps_use=use['none']).get('Tm')
        if np.isfinite(Km) and Km > 0 and Tmn and np.isfinite(Tmn):
            row['dH_col_dose'] = Tmn / Km
            mm = [float(np.mean([mass[a][i] for i in use[a]])) for a in C10A_PL]
            Kmass = slope0(mm, [out[a][0] for a in C10A_PL])
            row['dH_col_mass'] = Tmn / Kmass if Kmass > 0 else float('nan')
        res['models'][m] = row
    ys = [res['models'][m]['arms']['pl300']['y_m'] for m in C10A_MODELS]
    words = [res['models'][m]['Q1_word'] for m in C10A_MODELS]
    ok = [y for y, w in zip(ys, words) if not w.startswith('はっきりしない')]
    if len(ok) >= 2:
        med = float(np.median(ok))
        res['Q1'] = next(w for e, w in C10A_Q1_BINS if -med < e) if med < 0 else '上がる'
    else:
        res['Q1'] = 'はっきりしない'
    res['Q2'] = sum(res['models'][m]['Q2_word'] == 'T_m のほうが速く下がる' for m in C10A_MODELS)
    return res


def self_test_a10():
    """作り物：T_m は濃さ 30% で −20%（K_m ≈ 0.74）、T_0 は −3%（K_0 ≈ 0.1）→ Q1「かなり」、Q2「T_m のほうが速く」。"""
    import tempfile
    rng = np.random.default_rng(0)
    L, nD, S = 512, 40, 2
    t = np.array(C10A_T0_TEMPS)

    def chain(Tm):
        toks = np.zeros((len(C10A_TEMPS), S, nD, L), np.int32)
        for k, T in enumerate(C10A_TEMPS):
            lnK = 5.0 / T - 5.0 / Tm
            km = 0.01 * math.exp(-lnK / 2); kf = 0.01 * math.exp(lnK / 2)
            for s in range(S):
                for j in range(nD):
                    st, seq, per = 0, [], list(rng.integers(100, 30000, 7))
                    for tt in range(L):
                        if tt > 0:
                            st = (1 if rng.random() < kf else 0) if st == 0 else (0 if rng.random() < km else 1)
                        seq.append(per[tt % 7] if st else int(rng.integers(100, 30000)))
                    toks[k, s, j] = seq
        return toks
    with tempfile.TemporaryDirectory() as dd:
        for m in C10A_MODELS:
            np.savez(f'{dd}/{m}@tf.npz', tf=np.zeros((nD, len(C10A_TEMPS), 24), np.float32), temps=np.array(C10A_TEMPS))
            for a in C10A_ARMS:
                x = C10A_DOSE[a]
                Tm, T0 = 0.56 * math.exp(-0.74 * x), 0.86 * math.exp(-0.1 * x)
                F = np.where(np.array(C10A_TEMPS) < T0, (np.array(C10A_TEMPS) - T0) * 12, (np.array(C10A_TEMPS) - T0) * 5)
                g = np.broadcast_to(F[:, None, None, None], (len(C10A_TEMPS), S, nD, 24)) + rng.normal(0, 0.15, (len(C10A_TEMPS), S, nD, 24))
                np.savez(f'{dd}/{m}@{a}.npz', temps=np.array(C10A_TEMPS), toks=chain(Tm), g_ell=g.astype(np.float32),
                         g_ban_mass=np.full((len(C10A_TEMPS), S, nD, 24), x * 0.5, np.float32), g_banned_top1=np.zeros((len(C10A_TEMPS), S, nD, 24), bool))
            break
        # 3 モデル分は時間がかかるので、1 モデル分を名前だけ変えて置く
        import shutil, glob, os
        for m in C10A_MODELS[1:]:
            for p in glob.glob(f'{dd}/{C10A_MODELS[0]}@*.npz'):
                shutil.copy(p, p.replace(C10A_MODELS[0], m))
        r = analyze10(dd)
    row = r['models']['pythia-1b']
    assert row['Q1_word'] == 'かなり下がる（log で 0.15〜0.30、14〜26%）', (row['arms']['pl300'], row['ci_y_m'])
    assert row['Q2_word'] == 'T_m のほうが速く下がる', (row['K_m'], row['K_0'], row['ci_logR'])
    assert r['Q1'] == 'かなり下がる（log で 0.15〜0.30、14〜26%）' and r['Q2'] == 3, (r['Q1'], r['Q2'])
    assert q2_word10(math.log(2), [0.2, 1.0], 0) == 'T_0 のほうが速く下がる（海水型）' and q2_word10(0.0, [-0.2, 0.3], 0) == '同じくらい'
    return 'OK'
