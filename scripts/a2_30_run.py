"""A2-30 の実行（CPU）。ゲート：docs/A2_30_GATE.md。
  自己テスト：python3 scripts/a2_30_run.py --self-test
  前提だけ：  python3 scripts/a2_30_run.py <G-pre> <A2-18> <A2-19> <A2-20 第 1 段> <A2-20 第 2 段> <出力> --dry
  分析：      python3 scripts/a2_30_run.py <G-pre> <A2-18> <A2-19> <A2-20 第 1 段> <A2-20 第 2 段> results/A2_30/a2_30_summary.json
標準出力には自己テストと前提の確かめだけを出す（新しい点や区間は出さない）。
"""
import sys, os, json, hashlib, time
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_28_run.py').read().split("\nif __name__")[0])

SHA30 = ('scripts/a2_30_analyze.py', 'scripts/a2_30_run.py', 'docs/A2_30_GATE.md') + SHA28


def load30ns():
    ns = load28ns()
    exec(open('scripts/a2_30_analyze.py').read(), ns)
    return ns


def arm_E(ns, Z, want_O=False):
    ev, sig = ns['seq_events'](Z['toks'], Z['pick'], Z['q'])
    if not want_O:
        return ev, sig, None, None
    C = ns['marks_of'](Z['toks']).astype(float)
    return ev, sig, C.sum(-1) / 512.0, C[..., -64:].mean(-1)


if __name__ == '__main__':
    ns = load30ns()
    print('自己テスト', ns['self_test30'](), flush=True)
    if sys.argv[1] == '--self-test':
        sys.exit()
    fg, f18, f19, f20a, f20, out = sys.argv[1:7]
    DRY = '--dry' in sys.argv
    if not DRY:
        assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
        assert os.path.isdir(os.path.dirname(out)), ('出力のフォルダがありません', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA30}
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    R18 = json.load(open('results/A2_18/a2_18_summary.json'))
    S19 = json.load(open('results/A2_19/a2_19_summary.json')); S20 = json.load(open('results/A2_20/a2_20_summary.json'))
    R28 = json.load(open('results/A2_28/a2_28_summary.json'))
    S22 = json.load(open('results/A2_22/a2_22_summary.json')); T22 = {**S22['small']['models'], **S22['big']['models']}
    R27 = json.load(open('results/A2_27/a2_27_summary.json'))['R6']['U']['s_N']['models']
    SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
    t0 = time.time()
    # ---- 読み込み ----
    R, G, F, DS = {}, {}, {}, {}
    for m in SMALL + BIG:
        t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
        S = S19['models'][m] if m in SMALL else S20['models'][m]
        f = f19 if m in SMALL else f20
        Z0 = load_arm(f, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f, m, 'h1', t['grid'], S['h'])
        assert Z0['toks'].shape[-1] == 512
        e0, s0, O0, P0 = arm_E(ns, Z0, True); e1, s1, O1, P1 = arm_E(ns, Z1, True)
        R[m] = dict(E={'c0': (e0, s0), 'h1': (e1, s1)}, O0=O0, O1=O1, O640=P0, O641=P1, q0=Z0['q'], h=S['h'], grid=[float(x) for x in t['grid']],
                    use=list(S['use']), toks0=Z0['toks'] if m in SMALL else None)
        print('読み込み', m, round(time.time() - t0), flush=True)
    for m in SMALL:
        t = H18[m]['targets']['-0.10']
        E = {}
        for arm in ('c0', 'h1', 'h2'):
            Z = np.load(f'{f18}/{m}@{arm}.npz')
            assert [round(float(x), 4) for x in Z['temps']] == [round(float(x), 4) for x in t['grid']], (m, arm)
            E[arm] = ns['seq_events'](Z['toks'], Z['pick'], Z['q'])
        F[m] = dict(E=E, h={a: R18['models'][m][a]['h'] for a in ('h1', 'h2')}, use={a: R18['models'][m][a]['use'] for a in ('h1', 'h2')})
        Z = {a: np.load(f'{fg}/{m}@{a}.npz') for a in ('h0', 'hm03', 'hp03')}
        for a, hv in (('h0', 0.0), ('hm03', -0.3), ('hp03', 0.3)):
            assert [round(float(x), 4) for x in Z[a]['temps']] == ns['G16_TEMPS'] and abs(float(Z[a]['h']) - hv) < 1e-9, (m, a)
        G[m] = {a: ns['seq_events'](Z[a]['toks'], Z[a]['pick'], Z[a]['q']) for a in Z}
        # R15：最初の試験の c0 と設計の h ＝ 0 の、重なる温度で、続きのトークンの列がまるごと同じ割合（割合だけを残す）
        Zc = np.load(f'{f18}/{m}@c0.npz'); tg = [round(float(x), 4) for x in Zc['temps']]
        same = {}
        for k, T in enumerate(ns['G16_TEMPS']):
            if round(T, 4) in tg:
                kk = tg.index(round(T, 4))
                same[str(T)] = float(np.mean(np.all(Zc['toks'][kk] == Z['h0']['toks'][k], -1)))
        G[m]['R15'] = same
        print('読み込み（最初の試験・設計）', m, round(time.time() - t0), flush=True)
    G7 = {}
    for m in BIG:
        Z = np.load(f'{f20a}/{m}@s1.npz')
        tt = [round(float(x), 4) for x in Z['temps']]; win = [round(float(x), 4) for x in H20[m]['window']]
        idx = [tt.index(x) for x in win]
        assert abs(float(Z['h'])) < 1e-12
        G7[m] = dict(E=ns['seq_events'](Z['toks'][idx], Z['pick'][idx], Z['q'][idx]), temps=[float(x) for x in win])
        print('読み込み（7B の設計）', m, round(time.time() - t0), flush=True)
    one = {m: np.ones(R[m]['E']['c0'][0].shape[2]) for m in R}
    nD = R[SMALL[0]]['E']['c0'][0].shape[2]; nD18 = F[SMALL[0]]['E']['c0'][0].shape[2]; nDg = G[SMALL[0]]['h0'][0].shape[2]
    assert nD == 120 and nD18 == 40 and nDg == 40
    # ---- 前提の確かめ ----
    for m in SMALL + BIG:
        r, u, D, P = ns['rho_model'](R[m]['E'], ('h1', R[m]['h']), R[m]['grid'], one[m])
        S = S19['models'][m] if m in SMALL else S20['models'][m]
        assert u == R[m]['use'] and abs(r - S['rho']) < 1e-9, (m, '前提：ρ・使う温度が記録と合わない')
    for m in SMALL:
        for a in ('h1', 'h2'):
            r, u, _, _ = ns['rho_model'](F[m]['E'], (a, F[m]['h'][a]), H18[m]['targets']['-0.10']['grid'], np.ones(nD18))
            assert u == F[m]['use'][a] and abs(r - R18['models'][m][a]['rho']) < 1e-9, (m, a, '前提：最初の試験が記録と合わない')
        g, gb = ns['g_design'](*G[m]['h0'], ns['G16_TEMPS'], np.ones(nDg))
        assert np.allclose(g, H18[m]['g_T'], atol=1e-9) and abs(gb - H18[m]['g_coef'][0]) < 1e-9, (m, '前提：設計の g が記録と合わない')
        D = ns['D_of'](R[m]['E'], one[m])
        ro = ns['rho_slope'](D, R[m]['use'], R[m]['h'], gb)
        assert abs(ro - R28['XQ26']['records'][m]['rho_out']) < 1e-9, (m, '前提：ρ_out が A2-28 と合わない')
    for m in BIG:
        g, gb = ns['g_design'](*G7[m]['E'], G7[m]['temps'], np.ones(G7[m]['E'][0].shape[2]))
        assert np.allclose(g, H20[m]['g_T'], atol=1e-9) and abs(gb - H20[m]['gbar']) < 1e-9, (m, '前提：7B の設計の g が記録と合わない')
        D = ns['D_of'](R[m]['E'], one[m])
        assert abs(ns['rho_slope'](D, R[m]['use'], R[m]['h'], gb) - R28['XQ26']['records'][m]['rho_out']) < 1e-9, (m, '前提：7B の ρ_out が A2-28 と合わない')
    # 1,000 回・登録の種で登録の区間を出し直す
    st = lambda EE, names, arm, hs, grids, uses, how: (lambda w: ns['stat_rho'](EE, names, arm, hs, grids, uses, w, how))
    E19 = {m: R[m]['E'] for m in SMALL}; E20 = {m: R[m]['E'] for m in BIG}
    fn19 = st(E19, SMALL, 'h1', {m: R[m]['h'] for m in SMALL}, {m: R[m]['grid'] for m in SMALL}, {m: R[m]['use'] for m in SMALL}, 'median')
    fn20 = st(E20, BIG, 'h1', {m: R[m]['h'] for m in BIG}, {m: R[m]['grid'] for m in BIG}, {m: R[m]['use'] for m in BIG}, 'mean')
    E18 = {m: F[m]['E'] for m in SMALL}; g18 = {m: H18[m]['targets']['-0.10']['grid'] for m in SMALL}
    fn18 = {a: st(E18, SMALL, a, {m: F[m]['h'][a] for m in SMALL}, g18, {m: F[m]['use'][a] for m in SMALL}, 'median') for a in ('h1', 'h2')}
    chk = [(fn19, nD, ns['A19_SEED_BOOT'], S19['XQ1']['ci']), (fn20, nD, ns['A20_SEED_BOOT'], S20['XQ7']['ci'])]
    # A2-18 は 1 回の選び直しで h1 と h2 を同じ w で出す（選び直しの列は同じ）
    chk += [(fn18[a], nD18, ns['A18_SEED'], R18['verdict'][a]['ci']) for a in ('h1', 'h2')]
    for fn, n_, sd, ci in chk:
        assert np.allclose(ns['ci_of'](ns['boot_stat'](fn, n_, sd, 1000)), ci, atol=1e-9), ('前提：登録の区間を再現しない', sd)
    print('前提 OK', round(time.time() - t0), flush=True)
    if DRY:
        sys.exit()

    res = dict(sha256=sha)
    # ---- XQ32 ----
    reg = {'replication': (fn19, nD, S19['XQ1']['word']), 'big7b': (fn20, nD, S20['XQ7']['word']),
           'first_h1': (fn18['h1'], nD18, R18['verdict']['h1']['word']), 'first_h2': (fn18['h2'], nD18, R18['verdict']['h2']['word'])}
    x32 = {}
    for key, (fn, n_, regw) in reg.items():
        point = fn(np.ones(n_))
        rows, words, bsets = {}, {}, {}
        for sd in ns['A30_SEEDS']:
            bs = ns['boot_stat'](fn, n_, sd, ns['A30_BOOT']); bsets[sd] = bs
            ci = ns['ci_of'](bs); fl = float(np.mean(~np.isfinite(bs)))
            rows[str(sd)] = dict(ci=ci, fail=fl, word=ns['word_band30'](ci, fl), bias=float(np.nanmean(bs) - point))
            words[str(sd)] = rows[str(sd)]['word']
        b0 = bsets[ns['A30_SEEDS'][0]]
        bci, bpar = ns['bca_ci'](fn, b0, n_)
        fl0 = float(np.mean(~np.isfinite(b0)))
        words['bca'] = ns['word_band30'](bci, fl0)
        w, changed = ns['xq32_word'](words, regw)
        lo = [rows[str(s)]['ci'][0] for s in ns['A30_SEEDS']]; hi = [rows[str(s)]['ci'][1] for s in ns['A30_SEEDS']]
        x32[key] = dict(point=point, registered_word=regw, percentile=rows, bca=dict(ci=bci, **bpar, word=words['bca']), word=w, changed=changed,
                        mc_sd_lo=float(np.std(lo)), mc_sd_hi=float(np.std(hi)))
        print('XQ32 計算', key, round(time.time() - t0), flush=True)
    res['XQ32'] = x32
    # ---- XQ33・XQ34（設計と追試の記事を別々に選び直す） ----
    oneg = np.ones(nDg)
    gb = {m: ns['g_design'](*G[m]['h0'], ns['G16_TEMPS'], oneg)[1] for m in SMALL}
    fd0 = {m: ns['fd_slope'](G[m]['hp03'][0], G[m]['hm03'][0]) for m in SMALL}
    fduse = {m: fd0[m][1] for m in SMALL}

    def slopes_of(v):
        g_ = {m: ns['g_design'](*G[m]['h0'], ns['G16_TEMPS'], v)[1] for m in SMALL}
        return {'out': g_, 'xm': {m: ns['loo_mean'](g_, m) for m in SMALL},
                'fd': {m: ns['fd_slope'](G[m]['hp03'][0], G[m]['hm03'][0], v, fduse[m])[2] for m in SMALL}}

    def rhos(w, sl):
        D = {m: ns['D_of'](R[m]['E'], w) for m in SMALL}
        return {k: [ns['rho_slope'](D[m], R[m]['use'], R[m]['h'], sl[k][m]) for m in SMALL] for k in sl}
    sl0 = slopes_of(oneg); pt = rhos(np.ones(nD), sl0)
    same = [S19['models'][m]['rho'] for m in SMALL]
    rng = np.random.default_rng(ns['A30_SEED33'])
    d_xm, d_fd, med = [], [], {k: [] for k in ('out', 'xm', 'fd')}
    sdg = {m: {'out': [], 'fd': []} for m in SMALL}
    for _ in range(ns['A30_BOOT']):
        w = ns['resample'](rng, nD); v = ns['resample'](rng, nDg)
        sl = slopes_of(v); r = rhos(w, sl)
        a = {k: ns['acc'](r[k]) for k in r}
        d_xm.append(a['xm'] - a['out']); d_fd.append(a['fd'] - a['out'])
        for k in med:
            med[k].append(ns['agg30'](r[k], 'median'))
        for m in SMALL:
            sdg[m]['out'].append(sl['out'][m]); sdg[m]['fd'].append(sl['fd'][m])
    d_xm, d_fd = np.array(d_xm), np.array(d_fd)
    ci = ns['ci_of'](d_xm); fl = float(np.mean(~np.isfinite(d_xm)))
    cixm = ns['ci_of'](med['xm']); flxm = float(np.mean(~np.isfinite(np.array(med['xm'], float))))
    res['XQ33'] = dict(point=ns['acc'](pt['xm']) - ns['acc'](pt['out']), ci=ci, fail=fl, word=ns['word33'](ci, fl),
                       acc={k: ns['acc'](v) for k, v in pt.items()} | {'same': ns['acc'](same)},
                       fd_minus_out=dict(point=ns['acc'](pt['fd']) - ns['acc'](pt['out']), ci=ns['ci_of'](d_fd), fail=float(np.mean(~np.isfinite(d_fd)))),
                       xm_median=dict(point=ns['agg30'](pt['xm'], 'median'), ci=cixm, fail=flxm, word=ns['word_band30'](cixm, flxm)),
                       fd_median=dict(point=ns['agg30'](pt['fd'], 'median'), ci=ns['ci_of'](med['fd'])),
                       models={m: dict(rho_same=S19['models'][m]['rho'], rho_out=pt['out'][i], rho_xm=pt['xm'][i], rho_fd=pt['fd'][i],
                                       gbar=sl0['out'][m], gbar_xm=sl0['xm'][m], fd=sl0['fd'][m], fd_use=fduse[m], fd_T=fd0[m][0].tolist(),
                                       sd_gbar=float(np.nanstd(sdg[m]['out'])), sd_fd=float(np.nanstd(sdg[m]['fd']))) for i, m in enumerate(SMALL)})
    print('XQ33 計算', round(time.time() - t0), flush=True)
    # XQ34：小さい 7 本は中央値、7B は平均（7B の設計は A2-20 第 1 段の 40 記事）
    rng = np.random.default_rng(ns['A30_SEED34'])
    nD7 = G7[BIG[0]]['E'][0].shape[2]
    bs, bs7, bm = [], [], {m: [] for m in SMALL + BIG}
    for _ in range(ns['A30_BOOT']):
        w = ns['resample'](rng, nD); v = ns['resample'](rng, nDg); v7 = ns['resample'](rng, nD7)
        r = [ns['rho_slope'](ns['D_of'](R[m]['E'], w), R[m]['use'], R[m]['h'], ns['g_design'](*G[m]['h0'], ns['G16_TEMPS'], v)[1]) for m in SMALL]
        r7 = [ns['rho_slope'](ns['D_of'](R[m]['E'], w), R[m]['use'], R[m]['h'], ns['g_design'](*G7[m]['E'], G7[m]['temps'], v7)[1]) for m in BIG]
        for m, x in zip(SMALL + BIG, r + r7):
            bm[m].append(x)
        bs.append(ns['agg30'](r, 'median')); bs7.append(ns['agg30'](r7, 'mean'))
    pt7 = [R28['XQ26']['records'][m]['rho_out'] for m in BIG]
    ci = ns['ci_of'](bs); fl = float(np.mean(~np.isfinite(np.array(bs, float))))
    ci7 = ns['ci_of'](bs7); fl7 = float(np.mean(~np.isfinite(np.array(bs7, float))))
    res['XQ34'] = dict(point=ns['agg30'](pt['out'], 'median'), ci=ci, fail=fl, word=ns['word_band30'](ci, fl),
                       big=dict(point=ns['agg30'](pt7, 'mean'), ci=ci7, fail=fl7, word=ns['word_band30'](ci7, fl7)),
                       models={m: dict(point=(pt['out'][SMALL.index(m)] if m in SMALL else pt7[BIG.index(m)]), ci=ns['ci_of'](bm[m])) for m in SMALL + BIG})
    print('XQ34 計算', round(time.time() - t0), flush=True)
    # ---- XQ35 ----
    x35 = {}
    for grp, how, key, sd, a0, a1 in ((SMALL, 'median', 'small', ns['A30_SEED35'], 'O0', 'O1'), (BIG, 'mean', 'big', ns['A30_SEED35'], 'O0', 'O1'),
                                      (SMALL, 'median', 'small_last64', ns['A30_SEED35'], 'O640', 'O641'), (BIG, 'mean', 'big_last64', ns['A30_SEED35'], 'O640', 'O641')):
        fr = lambda m, w, a0=a0, a1=a1: ns['rho_frac'](R[m][a0], R[m][a1], R[m]['E']['c0'][1], R[m]['grid'], R[m]['h'], w, R[m]['use'])
        fn = lambda w: ns['agg30']([fr(m, w) for m in grp], how)
        rng = np.random.default_rng(sd)
        bsm = {m: [] for m in grp}; bsa = []
        for _ in range(ns['A30_BOOT']):
            w = ns['resample'](rng, nD)
            r = [fr(m, w) for m in grp]
            for m, x in zip(grp, r):
                bsm[m].append(x)
            bsa.append(ns['agg30'](r, how))
        ci = ns['ci_of'](bsa); fl = float(np.mean(~np.isfinite(bsa)))
        x35[key] = dict(point=fn(np.ones(nD)), ci=ci, fail=fl, word=ns['word_band30'](ci, fl),
                        models={m: dict(point=fr(m, np.ones(nD)), ci=ns['ci_of'](bsm[m])) for m in grp})
    res['XQ35'] = x35
    print('XQ35 計算', round(time.time() - t0), flush=True)
    # ---- 記録 ----
    eqw = {}
    R26 = json.load(open('results/A2_26/a2_26_summary.json'))['full']['models']
    for m in SMALL + BIG:
        c = R26[m]['ci']; eqw[m] = ns['word_band30'](c, 0.0)
    res['R7'] = dict(replication=S19['XQ1']['per_model'], big7b=S20['XQ7']['per_model'], equilibrium_same_rule=eqw)
    r8 = {}
    for m in SMALL + BIG:
        k01, k10, tau = ns['rates_tau'](R[m]['E']['c0'][0], one[m])
        u = R[m]['use']; sp = R27.get(m, {}).get('point')
        r8[m] = dict(k01=k01[u].tolist(), k10=k10[u].tolist(), tau=tau[u].tolist(), tau_median=float(np.median(tau[u])),
                     s_pen=sp, s_pen_ci=R27.get(m, {}).get('ci'), s_pen_times_tau=(sp * float(np.median(tau[u])) if sp is not None else None))
    from scipy.stats import spearmanr
    sp = [r8[m]['s_pen'] for m in SMALL]; it = [1.0 / r8[m]['tau_median'] for m in SMALL]
    c8 = ns['cv']([r8[m]['s_pen_times_tau'] for m in SMALL])
    res['R8'] = dict(models=r8, spearman_s_pen_vs_inv_tau_small=float(spearmanr(sp, it).correlation), cv_s_pen_times_tau_small=c8, reading=ns['read_R8'](c8))
    res['R9'] = {m: ns['logw_var'](R[m]['q0'], R[m]['grid'], R[m]['h'], R[m]['use']) for m in SMALL + BIG}
    # R10：組にした区間（登録）と、腕を別々に選び直した区間の幅
    rng = np.random.default_rng(20261138)
    bsp, bsi = [], []
    for _ in range(ns['A30_BOOT']):
        w0 = ns['resample'](rng, nD); w1 = ns['resample'](rng, nD)
        bsp.append(fn19(w0))
        bsi.append(ns['agg30']([ns['rho_split'](R[m]['E'], R[m]['h'], R[m]['grid'], w0, w1, R[m]['use']) for m in SMALL], 'median'))
    cp, ci_ = ns['ci_of'](bsp), ns['ci_of'](bsi)
    res['R10'] = dict(paired_ci=cp, independent_ci=ci_, width_ratio=float((ci_[1] - ci_[0]) / (cp[1] - cp[0])))
    r11 = {}
    for m in SMALL + BIG:
        lin = T22[m]['Tm_c0']; loc = R28['XQ27']['records'][m]['T0'][0]
        r11[m] = dict(T_lin=lin, T_loc=loc, closest_lin=ns['closest'](R[m]['grid'], lin), closest_loc=ns['closest'](R[m]['grid'], loc))
        r11[m]['differs'] = r11[m]['closest_lin'] != r11[m]['closest_loc']
    res['R11'] = dict(models=r11, n_differ=int(sum(v['differs'] for v in r11.values())))
    R16s = json.load(open('results/A2_16/a2_16_summary.json'))['models']
    res['R13'] = {m: dict(identity=ns['dTdh_identity'](H18[m]['g_T'], ns['G16_TEMPS'], H18[m]['Tm0'], H18[m]['b']), measured=R16s[m]['point']['slope_w'],
                          Tm0=H18[m]['Tm0'], hstar=H18[m]['targets']['-0.10']['hstar']) for m in SMALL}
    for m in SMALL:
        r = res['R13'][m]; r['measured_over_identity'] = r['measured'] / r['identity']; r['implied_shift_ratio'] = abs(r['hstar']) * r['measured'] / 0.10
    X23 = json.load(open('results/A2_26/a2_26_summary.json'))['XQ23']
    res['R14'] = {k: X23[k]['models'] for k in ('base', 'g', 'identity', 'total')}
    res['R15'] = {m: G[m]['R15'] for m in SMALL}
    # R16：最初の試験の 2h* と h* の比の差（同じ選び直し）
    rng = np.random.default_rng(20261139)
    pm = {m: (lambda w, m=m: ns['rho_model'](F[m]['E'], ('h2', F[m]['h']['h2']), g18[m], w, F[m]['use']['h2'])[0] - ns['rho_model'](F[m]['E'], ('h1', F[m]['h']['h1']), g18[m], w, F[m]['use']['h1'])[0]) for m in SMALL}
    dmed, dm_ = [], {m: [] for m in SMALL}
    for _ in range(ns['A30_BOOT']):
        w = ns['resample'](rng, nD18)
        dmed.append(fn18['h2'](w) - fn18['h1'](w))
        for m in SMALL:
            dm_[m].append(pm[m](w))
    res['R16'] = dict(point=fn18['h2'](np.ones(nD18)) - fn18['h1'](np.ones(nD18)), ci=ns['ci_of'](dmed),
                      models={m: dict(point=pm[m](np.ones(nD18)), ci=ns['ci_of'](dm_[m])) for m in SMALL})
    json.dump(clean28(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print('保存しました', out, round(time.time() - t0))
