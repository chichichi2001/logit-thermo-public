"""A2-24 の実行（CPU、探索）。
  自己テスト：python3 scripts/a2_24_run.py --self-test
  分析：python3 scripts/a2_24_run.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_24/a2_24_summary.json
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_22_run.py').read().split("\nif __name__")[0])

SHA24 = ('scripts/a2_24_analyze.py', 'scripts/a2_24_run.py') + SHA22


def load24ns():
    ns = load22ns()
    exec(open('scripts/a2_24_analyze.py').read(), ns)
    return ns


if __name__ == '__main__':
    ns = load24ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test24']()); sys.exit()
    f19, f20, out = sys.argv[1:4]
    assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA24}
    print('自己テスト', ns['self_test24'](), flush=True)
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']; S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    S22 = json.load(open('results/A2_22/a2_22_summary.json')); T22 = {**S22['small']['models'], **S22['big']['models']}
    SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
    D = {}
    for m in SMALL + BIG:
        t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
        S = S19[m] if m in SMALL else S20[m]
        f = f19 if m in SMALL else f20
        Z0 = load_arm(f, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f, m, 'h1', t['grid'], S['h'])
        C0, C1 = ns['marks_of'](Z0['toks']), ns['marks_of'](Z1['toks'])
        sig0 = (Z0['pick'].astype(float) - Z0['q'].astype(float)).sum(-1)
        ev0, ev1 = ns['events_of'](C0, 32, 0.5, 0.25), ns['events_of'](C1, 32, 0.5, 0.25)
        # 前提：元の分け方で Δln K が記録（D）と一致
        one = np.ones(ev0.shape[2])
        l0, *_ = ns['lnK_from'](ev0, one); l1, *_ = ns['lnK_from'](ev1, one)
        u = S['use']
        assert np.allclose((l1 - l0)[u], np.array(S['D'])[u], atol=1e-9), (m, '前提：Δln K が記録と合わない')
        D[m] = dict(temps=[float(x) for x in t['grid']], h=S['h'], use=u, rC=t['r_clapeyron'], C0=C0, C1=C1, sig0=sig0,
                    ev0=ev0, ev1=ev1, pick0=Z0['pick'], q0=Z0['q'].astype(float), rho_ci=S['ci'], rho=S['rho'])
        print('読み込み', m, flush=True)
    nD = D[SMALL[0]]['ev0'].shape[2]
    res = dict(sha256=sha)

    # ---- XQ14 ----
    def share_fn(lo, hi):
        P = {m: ns['q_parts'](D[m]['pick0'], D[m]['q0'], lo, hi) for m in D}
        for m in D:   # 前提：3 つの和の g が g_identity と一致
            g_ref = np.array(ns['g_identity'](D[m]['ev0'], D[m]['sig0'], D[m]['temps'], np.ones(nD)))
            g_sum = sum(ns['g_part'](D[m]['ev0'], v, D[m]['temps'], np.ones(nD)) for v in P[m].values())
            assert np.allclose(g_sum, g_ref, atol=1e-9, equal_nan=True), m
        return P
    P = share_fn(*ns['A24_QBINS'])
    lo, hi = ns['A24_QBINS']
    enr = lambda m, w: ns['q_share'](D[m]['ev0'], P[m], D[m]['temps'], D[m]['use'], w)['unsure'] / ns['var_share'](None, D[m]['q0'], D[m]['use'], w, lo, hi)
    b14 = ns['boot'](lambda m, w, use: (enr(m, w), D[m]['use']), SMALL + BIG, nD, 'median', 2)
    b14['word'] = ns['word14'](b14['ci'], b14['fail'])
    b14['unsure_share'] = ns['boot'](lambda m, w, use: (ns['q_share'](D[m]['ev0'], P[m], D[m]['temps'], D[m]['use'], w)['unsure'], D[m]['use']), SMALL + BIG, nD, 'median', 2)
    rec14 = {}
    for m in D:
        q, cand = D[m]['q0'], D[m]['q0'] > 0
        lo, hi = ns['A24_QBINS']
        rec14[m] = dict(share=ns['q_share'](D[m]['ev0'], P[m], D[m]['temps'], D[m]['use'], np.ones(nD)),
                        var_share_unsure=ns['var_share'](None, D[m]['q0'], D[m]['use'], np.ones(nD), lo, hi),
                        step_frac={'sure_no': float((q < lo).mean()), 'unsure': float(((q >= lo) & (q <= hi)).mean()), 'sure_copy': float((q > hi).mean())},
                        step_frac_with_candidates={'sure_no': float(((q < lo) & cand).sum() / cand.sum()), 'unsure': float(((q >= lo) & (q <= hi)).sum() / cand.sum()), 'sure_copy': float((q > hi).sum() / cand.sum())})
    alt = {}
    for lo, hi in ns['A24_QBINS_ALT']:
        Pa = share_fn(lo, hi)
        alt[f'{lo}-{hi}'] = {m: ns['q_share'](D[m]['ev0'], Pa[m], D[m]['temps'], D[m]['use'], np.ones(nD)) for m in D}
    res['XQ14'] = dict(b14, records=rec14, alt_bins=alt)
    print('XQ14', b14['word'], round(b14['point'], 3), [round(x, 3) for x in b14['ci']], flush=True)

    # ---- XQ15 ----
    rec15, nb = {}, 0
    hists = {}
    for m in D:
        temps = D[m]['temps']; k = int(np.argmin(np.abs(np.array(temps) - T22[m]['Tm_c0'])))
        sw, ao, ar = ns['seq_kinds'](D[m]['C0'][k])
        hk = ns['window_hist'](D[m]['C0'][k]); bm = ns['bimodal'](hk)
        hs = ns['window_hist'](D[m]['C0'][k], only=sw); bs_ = ns['bimodal'](hs)
        both = bool(bm['bimodal'] and bs_['bimodal']); nb += both
        rec15[m] = dict(T=temps[k], Tm_c0=T22[m]['Tm_c0'], hist=hk.tolist(), all=bm, hist_switching=hs.tolist(), switching=bs_, both=both,
                        frac_switching=float(sw.mean()), frac_all_ordinary=float(ao.mean()), frac_all_repetitive=float(ar.mean()))
        hists[m] = [ns['window_hist'](D[m]['C0'][kk]).tolist() for kk in range(len(temps))]
    res['XQ15'] = dict(n_bimodal=int(nb), word=ns['word15'](nb), records=rec15)
    print('XQ15', res['XQ15']['word'], nb, flush=True)

    # ---- XQ16 ----
    rows, rows_big, rows_fixed = {}, {}, {}
    for (W, en, ex) in sorted(ns['A24_LABELS'], key=lambda L: L != (32, 0.5, 0.25)):   # 元の分け方を最初に（前提の確かめを早く）
        E = {m: (ns['events_of'](D[m]['C0'], W, en, ex), ns['events_of'](D[m]['C1'], W, en, ex)) for m in D}
        fn = lambda m, w, use: ns['rho_label'](E[m][0], D[m]['sig0'], E[m][1], D[m]['h'], D[m]['temps'], w, use)
        key = f'W{W}_{en}_{ex}'
        rows[key] = ns['boot'](fn, SMALL, nD, 'median', 2)
        if key == 'W32_0.5_0.25':
            assert abs(rows[key]['point'] - json.load(open('results/A2_19/a2_19_summary.json'))['XQ1']['median']) < 1e-9, '前提：元の分け方の中央値が A2-19 と合わない'
        rows[key]['use'] = {m: ns['rho_label'](E[m][0], D[m]['sig0'], E[m][1], D[m]['h'], D[m]['temps'], np.ones(nD))[1] for m in SMALL}
        rows_fixed[key] = ns['boot'](lambda m, w, use: ns['rho_label'](E[m][0], D[m]['sig0'], E[m][1], D[m]['h'], D[m]['temps'], w, D[m]['use']), SMALL, nD, 'median', 2)
        rows_big[key] = ns['boot'](fn, BIG, nD, 'mean', 0)
        print('XQ16', key, round(rows[key]['point'], 3), [round(x, 3) for x in rows[key]['ci']], '7B', round(rows_big[key]['point'], 3), flush=True)
    orig = 'W32_0.5_0.25'
    res['XQ16'] = dict(word=ns['word16']([v for k, v in rows.items() if k != orig]), small=rows, small_fixed_use=rows_fixed, big=rows_big)
    print('XQ16', res['XQ16']['word'], flush=True)

    # ---- 記録 ----
    res['R1'] = {m: dict(lnK=dict(point=D[m]['rho'] / D[m]['rC'], ci=[x / D[m]['rC'] for x in D[m]['rho_ci']]),
                         Tm=dict(point=T22[m]['ratio_over_rC'], ci=[x / D[m]['rC'] for x in T22[m]['ratio_ci']])) for m in D}
    r2 = {}
    for m in D:
        k = rec15[m]['T']; kk = D[m]['temps'].index(k); one = np.ones(nD)
        halves = [ns['lnK_from'](ns['events_of'](D[m]['C0'][kk:kk + 1], 32, 0.5, 0.25, half=hh), one)[0][0] for hh in (0, 1)]
        r2[m] = dict(T=k, lnK_first=float(halves[0]), lnK_second=float(halves[1]))
    res['R2'] = r2
    C = D['pythia-1.4b']['C0']; k = D['pythia-1.4b']['temps'].index(rec15['pythia-1.4b']['T'])
    ex = None
    for j in range(C.shape[2]):
        st, t = ns['states_of'](C[k, 0, j], 32, 0.5, 0.25)
        if 2 <= int(np.abs(np.diff(st)).sum()) <= 6:
            ex = dict(T=D['pythia-1.4b']['temps'][k], article_index=j, marks=C[k, 0, j].astype(int).tolist(), states=st.astype(int).tolist(), steps=t.astype(int).tolist()); break
    res['R3'] = dict(hists=hists, temps={m: D[m]['temps'] for m in D}, example=ex)
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False)
    print('保存しました', out)
