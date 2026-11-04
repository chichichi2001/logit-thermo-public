"""A2-26 の実行（CPU、探索）。ゲート：docs/A2_26_GATE.md。
  自己テスト：python3 scripts/a2_26_run.py --self-test
  分析：python3 scripts/a2_26_run.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_26/a2_26_summary.json
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_24_run.py').read().split("\nif __name__")[0])

SHA26 = ('scripts/a2_26_analyze.py', 'scripts/a2_26_run.py') + SHA24
W20 = {'all': 'どちらの半分でも言いすぎ', 'points': '点はどちらも 0.8 未満', 'not_all': '言いすぎでない半分がある', 'unclear': 'はっきりしない'}
W21 = {'all': 'どの分け方でも言いすぎ', 'points': '点はどれも 0.8 未満', 'not_all': '言いすぎでない分け方がある', 'unclear': 'はっきりしない'}
W22 = ('ほぼ一定', '長くとどまるほど出にくい', '長くとどまるほど出やすい', 'はっきりしない')
W23 = {'base': '主に罰なしの ln K のずれ', 'g': '主に g の温度変化', 'both': '両方', 'neither': 'どちらとも言えない', 'unclear': 'はっきりしない'}


def load26ns():
    ns = load24ns()
    exec(open('scripts/a2_26_analyze.py').read(), ns)
    return ns


if __name__ == '__main__':
    ns = load26ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test24'](), ns['self_test26']()); sys.exit()
    f19, f20, out = sys.argv[1:4]
    assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    assert os.path.isdir(os.path.dirname(out)), ('出力のフォルダがありません', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA26}
    print('自己テスト', ns['self_test26'](), flush=True)
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']; S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    S22 = json.load(open('results/A2_22/a2_22_summary.json')); T22 = {**S22['small']['models'], **S22['big']['models']}
    SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
    ALL = SMALL + BIG
    ORIG = (32, 0.5, 0.25)
    D = {}
    for m in ALL:
        t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
        S = S19[m] if m in SMALL else S20[m]
        f = f19 if m in SMALL else f20
        Z0 = load_arm(f, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f, m, 'h1', t['grid'], S['h'])
        C0, C1 = ns['marks_of'](Z0['toks']), ns['marks_of'](Z1['toks'])
        sig0 = (Z0['pick'].astype(float) - Z0['q'].astype(float)).sum(-1)
        temps = [float(x) for x in t['grid']]
        c0 = ns['counts10'](C0, Z0['U'], Z0['pick']); c1 = ns['counts10'](C1, Z1['U'], Z1['pick'])
        nD = c0.shape[2]; one = np.ones(nD)
        # 前提 1：Δln K が記録（D）と一致
        u = S['use']
        dl = ns['lnK_of'](ns['tot'](c1, one)) - ns['lnK_of'](ns['tot'](c0, one))
        assert np.allclose(dl[u], np.array(S['D'])[u], rtol=0, atol=1e-9), (m, '前提：Δln K が記録と合わない')
        assert np.array_equal(c0[..., :4], ns['events_of'](C0, *ORIG)), (m, '前提：数えが events_of と合わない')
        # 前提 2：状態ごとの和が A2-16 の arm_counts（窓）と一致
        cw, _ = ns['arm_counts'](Z0['toks'], Z0['U'], Z0['pick'])
        assert np.allclose(c0.sum(1)[..., 4:10], cw[..., [4, 5, 6, 7, 10, 11]], rtol=0, atol=1e-9), (m, '前提：U・候補の数えが arm_counts と合わない')
        # 前提 3：前半＋後半 ＝ 全文
        h0 = [ns['counts10'](C, Z['U'], Z['pick'], half=0) for C, Z in ((C0, Z0), (C1, Z1))]
        h1 = [ns['counts10'](C, Z['U'], Z['pick'], half=1) for C, Z in ((C0, Z0), (C1, Z1))]
        assert np.allclose(h0[0] + h1[0], c0, rtol=0, atol=1e-9) and np.allclose(h0[1] + h1[1], c1, rtol=0, atol=1e-9), (m, '前提：前半＋後半が全文と合わない')
        D[m] = dict(t=t, temps=temps, h=S['h'], use=u, C0=C0, C1=C1, Z0U=Z0['U'], Z0p=Z0['pick'], Z1U=Z1['U'], Z1p=Z1['pick'],
                    sig0=sig0, c0=c0, c1=c1, half=[(h0[0], h0[1]), (h1[0], h1[1])])
        print('読み込み', m, flush=True)
    nD = D[SMALL[0]]['c0'].shape[2]
    res = dict(sha256=sha)

    def rat_fn(get):
        def fn(m, w, use):
            a, b = get(m)
            r = ns['ratios'](a, b, D[m]['sig0'], D[m]['temps'], D[m]['h'], w, use)
            return r[0], r[2]
        return fn

    def rho_fn(get):
        def fn(m, w, use):
            a, b = get(m)
            r = ns['ratios'](a, b, D[m]['sig0'], D[m]['temps'], D[m]['h'], w, use)
            return r[1], r[2]
        return fn

    def info(a, b, m):
        one = np.ones(nD); t0, t1 = ns['tot'](a, one), ns['tot'](b, one)
        use = ns['use_of'](t0, t1)
        if len(use) < ns['A18_MIN_T']:
            return dict(use=use)
        _, inf = ns['eq_pred'](t0, D[m]['temps'], D[m]['h'], use)
        return dict(use=use, **inf)

    # ---- 全文・元の分け方（共通の量の記録。図 2b と同じ種類の比） ----
    full = ns['boot'](rat_fn(lambda m: (D[m]['c0'], D[m]['c1'])), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
    full['info'] = {m: info(D[m]['c0'], D[m]['c1'], m) for m in ALL}
    full['rho'] = ns['boot'](rho_fn(lambda m: (D[m]['c0'], D[m]['c1'])), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
    assert abs(ns['agg']([full['rho']['models'][m]['point'] for m in SMALL], 'median', 2) - json.load(open('results/A2_19/a2_19_summary.json'))['XQ1']['median']) < 1e-9, '前提：小さい 7 本の ρ の中央値が A2-19 と合わない'
    res['full'] = full
    print('全文 ρ_E', round(full['point'], 3), [round(x, 3) for x in full['ci']], 'ρ', round(full['rho']['point'], 3), flush=True)

    # ---- XQ20 ----
    rows20 = []
    rec20 = {}
    for hh, nm in ((0, 'first'), (1, 'second')):
        r = ns['boot'](rat_fn(lambda m, hh=hh: D[m]['half'][hh]), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
        r['rho'] = ns['boot'](rho_fn(lambda m, hh=hh: D[m]['half'][hh]), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
        r['info'] = {m: info(*D[m]['half'][hh], m) for m in ALL}
        rec20[nm] = r; rows20.append(r)
        print('XQ20', nm, round(r['point'], 3), [round(x, 3) for x in r['ci']], 'ρ', round(r['rho']['point'], 3), flush=True)
    res['XQ20'] = dict(word=W20[ns['word_over'](rows20)], **rec20)
    print('XQ20', res['XQ20']['word'], flush=True)

    # ---- XQ21 ----
    rows21 = {}
    for (W, en, ex) in ns['A24_LABELS']:
        key = f'W{W}_{en}_{ex}'
        if (W, en, ex) == ORIG:
            rows21[key] = dict(point=full['point'], ci=full['ci'], fail=full['fail'], models=full['models']); continue
        cc = {m: (ns['counts10'](D[m]['C0'], D[m]['Z0U'], D[m]['Z0p'], W, en, ex), ns['counts10'](D[m]['C1'], D[m]['Z1U'], D[m]['Z1p'], W, en, ex)) for m in ALL}
        rows21[key] = ns['boot'](rat_fn(lambda m: cc[m]), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
        print('XQ21', key, round(rows21[key]['point'], 3), [round(x, 3) for x in rows21[key]['ci']], flush=True)
    alt = [v for k, v in rows21.items() if k != 'W32_0.5_0.25']
    res['XQ21'] = dict(word=W21[ns['word_over'](alt)], rows=rows21)
    print('XQ21', res['XQ21']['word'], flush=True)

    # ---- XQ22 ----
    HZ, rec22 = {}, {}
    for m in ALL:
        temps = D[m]['temps']; k = int(np.argmin(np.abs(np.array(temps) - T22[m]['Tm_c0'])))
        rep, od = ns['hazard_counts'](D[m]['C0'][k])
        HZ[m] = (rep, od)
        cr = rep.sum(0)
        rec22[m] = dict(T=temps[k], rep_counts=cr[:4].tolist(), ord_counts=od.sum(0)[:4].tolist(), dwell_hist=ns['dwell_hist'](D[m]['C0'][k]),
                        mean_pos_short=float(cr[4] / cr[0]) if cr[0] else None, mean_pos_long=float(cr[5] / cr[2]) if cr[2] else None)
    b22 = ns['boot'](lambda m, w, use: (ns['hazard_ratio'](HZ[m][0], w), None), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
    b22['word'] = ns['word_band'](b22['ci'], b22['fail'], W22)
    b22['entry'] = ns['boot'](lambda m, w, use: (ns['hazard_ratio'](HZ[m][1], w), None), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
    res['XQ22'] = dict(b22, records=rec22)
    print('XQ22', b22['word'], round(b22['point'], 3), [round(x, 3) for x in b22['ci']], '入る', round(b22['entry']['point'], 3), flush=True)

    # ---- XQ23 ----
    def dsplit(m, w, part):
        t = D[m]['t']; temps = D[m]['temps']
        k = int(np.argmin(np.abs(np.array(temps) - t['Tstar'])))
        H = H18[m] if m in SMALL else H20[m]
        gbar = H['g_coef'][0] if m in SMALL else H['gbar']
        if m in SMALL:
            assert H['g_coef'][1] == 0.0, (m, 'ḡ が温度によらない形ではない')
        assert abs(D[m]['h'] - t['hstar']) < 1e-4, (m, 'h が h* と合わない（生成は 4 桁に丸めた h）')
        g = np.array(ns['g_identity'](D[m]['c0'][..., :4], D[m]['sig0'], temps, w))
        return ns['design_split'](ns['tot'](D[m]['c0'], w), ns['tot'](D[m]['c1'], w), g, temps, k, H['a'], H['b'], gbar, D[m]['h'])[part]
    b23 = {nm: ns['boot'](lambda m, w, use, i=i: (dsplit(m, w, i), None), ALL, nD, 'median', 2, seed=ns['A26_SEED'])
           for i, nm in enumerate(('base', 'g', 'identity', 'total'))}
    b23['word'] = W23[ns['word23'](b23['base'], b23['g'])]
    curv = {}
    for m in ALL:
        t = D[m]['t']; temps = D[m]['temps']; k = int(np.argmin(np.abs(np.array(temps) - t['Tstar'])))
        H = H18[m] if m in SMALL else H20[m]
        cv, dd = ns['curv_split'](ns['tot'](D[m]['c0'], np.ones(nD)), temps, k, H['a'], H['b'], D[m]['use'])
        curv[m] = dict(T=temps[k], Tstar=t['Tstar'], curvature=float(cv), data=float(dd), A2_22_ratio=T22[m].get('ratio'), A2_22_ratio_over_rC=T22[m].get('ratio_over_rC'))
    b23['curv_records'] = curv
    res['XQ23'] = b23
    print('XQ23', b23['word'], {k: (round(b23[k]['point'], 3), [round(x, 3) for x in b23[k]['ci']]) for k in ('base', 'g', 'identity', 'total')}, flush=True)

    # ---- R4 ----
    S21 = json.load(open('results/A2_21/a2_21_summary.json'))
    M21 = {**S21['small']['models'], **S21['big']['models']}
    kap = np.array([M21[m]['kappa'] for m in ALL]); rr = np.array([M21[m]['r'] for m in ALL])
    rng = np.random.default_rng(ns['A26_SEED'])
    idx = rng.integers(0, len(ALL), (10000, len(ALL)))
    res['R4'] = dict(models={m: {k: M21[m][k] for k in ('kappa', 'ci', 'r', 'n', 'n_whole', 'kappa_mixed') if k in M21[m]} for m in ALL},
                     kappa_median=float(np.median(kap)), kappa_model_ci=np.percentile(np.median(kap[idx], 1), [2.5, 97.5]).tolist(),
                     r_median=float(np.median(rr)), r_model_ci=np.percentile(np.median(rr[idx], 1), [2.5, 97.5]).tolist(),
                     note='モデルの選び直しだけ（記事の選び直しは A2-21 のモデルごとの区間を並べた）。小さい 7 本は G-pre（A2-16）のデータ、7B の 2 本は A2-20 のデータで、データの出どころが混ざる')
    print('R4 κ', round(res['R4']['kappa_median'], 3), [round(x, 3) for x in res['R4']['kappa_model_ci']], flush=True)

    # ---- R5 ----
    ev = {m: (D[m]['c0'][..., :4], D[m]['c1'][..., :4]) for m in SMALL}
    fn5 = lambda m, w, use: ns['rho_label'](ev[m][0], D[m]['sig0'], ev[m][1], D[m]['h'], D[m]['temps'], w, use)
    cis = [ns['boot'](fn5, SMALL, nD, 'median', 2, seed=ns['A26_SEED'] + i)['ci'] for i in range(10)]
    cis = np.array(cis)
    res['R5'] = dict(cis=cis.tolist(), sd_lo=float(cis[:, 0].std(ddof=1)), sd_hi=float(cis[:, 1].std(ddof=1)),
                     A2_19_ci=json.load(open('results/A2_19/a2_19_summary.json'))['XQ1']['ci'])
    print('R5 区間の端の標準偏差', round(res['R5']['sd_lo'], 4), round(res['R5']['sd_hi'], 4), flush=True)
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False)
    print('保存しました', out)
