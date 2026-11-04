"""A2-28 の実行（CPU）。ゲート：docs/A2_28_GATE.md。
  自己テスト：python3 scripts/a2_28_run.py --self-test
  分析：python3 scripts/a2_28_run.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_28/a2_28_summary.json
  試運転：最後に --dry を付けると、選び直しを 3 回にし、数字を表示せず、出力を書かない（前提の確かめと、最後まで回ることだけ）。
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_27_run.py').read().split("\nif __name__")[0])

SHA28 = ('scripts/a2_28_analyze.py', 'scripts/a2_28_run.py') + SHA27


def load28ns():
    ns = load27ns()
    exec(open('scripts/a2_28_analyze.py').read(), ns)
    return ns


def clean28(o):
    if isinstance(o, dict):
        return {str(k): clean28(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean28(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


if __name__ == '__main__':
    ns = load28ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test28']()); sys.exit()
    f19, f20, out = sys.argv[1:4]
    DRY = len(sys.argv) > 4 and sys.argv[4] == '--dry'
    say = (lambda *a, **k: None) if DRY else print
    assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    assert os.path.isdir(os.path.dirname(out)), ('出力のフォルダがありません', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA28}
    print('自己テスト', ns['self_test28'](), flush=True)
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']; S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    S22 = json.load(open('results/A2_22/a2_22_summary.json')); T22 = {**S22['small']['models'], **S22['big']['models']}
    R6 = json.load(open('results/A2_27/a2_27_summary.json'))['R6']['U']['s_N']['models']
    SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
    ALL = SMALL + BIG
    D = {}
    for m in ALL:
        t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
        S = S19[m] if m in SMALL else S20[m]
        f = f19 if m in SMALL else f20
        temps = [float(x) for x in t['grid']]
        Z0 = load_arm(f, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f, m, 'h1', t['grid'], S['h'])
        C0, C1 = ns['marks_of'](Z0['toks']), ns['marks_of'](Z1['toks'])
        c0 = ns['counts10'](C0, Z0['U'], Z0['pick']); c1 = ns['counts10'](C1, Z1['U'], Z1['pick'])
        nD = c0.shape[2]; one = np.ones(nD)
        # XQ26 の前提：設計の予言の数
        if m in SMALL:
            gc = H18[m]['g_coef']
            assert abs(gc[1]) < 1e-12, (m, '前提：G-pre の g の当てはめが温度によらない形でない')
            gbar, a, b = gc[0], H18[m]['a'], H18[m]['b']
        else:
            gbar, a, b = H20[m]['gbar'], H20[m]['a'], H20[m]['b']
        assert abs(t['hstar'] * gbar + (a + b / t['Tstar'])) < 1e-6, (m, '前提：h* ḡ が −(a ＋ b/T*) と合わない')
        target = S['h'] * gbar                     # 生成に使った（4 桁に丸めた）h × 設計の ḡ
        Tm0 = H18[m]['Tm0'] if m in SMALL else H20[m]['Tm0']
        # 前提：Δ と ρ が A2-19・A2-20 を再現
        sig0 = (Z0['pick'].astype(float) - Z0['q'].astype(float)).sum(-1)
        u = S['use']
        assert ns['use_of'](ns['tot'](c0, one), ns['tot'](c1, one)) == list(u), (m, '前提：使う温度が A2-19・A2-20 と合わない')
        dl = ns['lnK_of'](ns['tot'](c1, one)) - ns['lnK_of'](ns['tot'](c0, one))
        assert np.allclose(dl[u], np.array(S['D'])[u], rtol=0, atol=1e-9), (m, '前提：Δ が記録と合わない')
        _, rho, _ = ns['ratios'](c0, c1, sig0, temps, S['h'], one, u)
        assert abs(rho - S['rho']) < 1e-9, (m, '前提：ρ が A2-19・A2-20 と合わない')
        # XQ27 の前提：直線の比が A2-22 を再現
        cw0, _ = ns['arm_counts'](Z0['toks'], Z0['U'], Z0['pick']); cw1, _ = ns['arm_counts'](Z1['toks'], Z1['U'], Z1['pick'])
        cw0, cw1 = cw0.sum(1) if cw0.ndim == 4 else cw0, cw1.sum(1) if cw1.ndim == 4 else cw1
        refs = (Tm0, t['Tstar'])
        ov, uu = ns['overshoot'](cw0, cw1, temps, one, refs)
        assert abs(ov['lin'] - T22[m]['ratio']) < 1e-9, (m, '前提：直線の比が A2-22 と合わない')
        # XQ28：基準の s_N が A2-27 の R6 を再現（小さい 7 本）
        N, Q = Z0['pick'].astype(float).sum(-1), Z0['q'].astype(float).sum(-1)
        u6 = ns['use_of'](ns['tot'](c0, one), ns['tot'](c1, one))
        sN = ns['s_N_of'](c0[..., :4], N, Q, temps, one, u6, within=True)
        if m in SMALL:
            assert abs(sN - R6[m]['point']) < 1e-9, (m, '前提：s_N が A2-27 の R6 と合わない')
        tr = {}
        for L in (ns['A28_LCUT'], ns['A28_LREC'], 512):
            ca, Na, Qa = ns['trunc_parts'](C0, Z0['U'], Z0['pick'], Z0['q'], L)
            cb = ns['counts10'](C1[..., :L], Z1['U'][..., :L], Z1['pick'][..., :L])
            tr[L] = dict(c0=ca, c1=cb, N=Na, Q=Qa)
        u512 = ns['use_of'](ns['tot'](tr[512]['c0'], one), ns['tot'](tr[512]['c1'], one))
        assert np.array_equal(tr[512]['c0'], c0) and np.array_equal(tr[512]['c1'], c1) and np.allclose(tr[512]['N'], N) and np.allclose(tr[512]['Q'], Q) and u512 == u6, (m, '前提：L ＝ 512 の切り詰めが元と合わない')
        assert np.array_equal(ns['marks_of'](Z0['toks'][..., :ns['A28_LCUT']]), C0[..., :ns['A28_LCUT']]), (m, '前提：写しの印が切り詰めで変わる')
        D[m] = dict(temps=temps, c0=c0, c1=c1, target=target, use=u, cw0=cw0, cw1=cw1, use27=uu, refs=refs, Tstar=t['Tstar'], Tm0=Tm0,
                    N=N, Q=Q, u6=u6, tr=tr, rho=S['rho'], ratio22=T22[m]['ratio'])
        print('読み込み', m, flush=True)
    nD = D[SMALL[0]]['c0'].shape[2]
    res = dict(sha256=sha)
    B = dict(seed=ns['A28_SEED'], n=3 if DRY else ns['A28_BOOT'])

    # ---- XQ26 ----
    f26 = lambda m, w, use: ns['rho_out'](D[m]['c0'], D[m]['c1'], D[m]['target'], w, D[m]['use'])
    x26 = ns['boot'](f26, SMALL, nD, 'median', 2, **B)
    x26['word'] = ns['word_band28'](x26['point'], x26['ci'], x26['fail'], ns['W26'], neg='逆向き')
    x26['big'] = ns['boot'](f26, BIG, nD, 'mean', 0, **B)
    x26['records'] = {m: dict(rho_out=f26(m, np.ones(nD), None)[0], rho=D[m]['rho'], target=D[m]['target'],
                              out_over_rho=f26(m, np.ones(nD), None)[0] / D[m]['rho']) for m in ALL}
    res['XQ26'] = x26
    say('XQ26', x26['word'], round(x26['point'], 3), [round(v, 3) for v in x26['ci']], flush=True)

    # ---- XQ27 ----
    ov = lambda m, w: ns['overshoot'](D[m]['cw0'], D[m]['cw1'], D[m]['temps'], w, D[m]['refs'], D[m]['use27'])[0]
    f27 = lambda key: (lambda m, w, use: (ov(m, w)[key], use))
    x27 = ns['boot'](f27('loc'), SMALL, nD, 'median', 0, **B)          # 7 本すべてで T_loc が出ないと中央値を取らない
    pts = [ov(m, np.ones(nD))['loc'] for m in SMALL]
    x27['word'] = 'はっきりしない' if not all(np.isfinite(pts)) else ns['word_band28'](x27['point'], x27['ci'], x27['fail'], ns['W27'])
    x27['missing'] = [m for m, v in zip(SMALL, pts) if not np.isfinite(v)]
    x27['big'] = ns['boot'](f27('loc'), BIG, nD, 'mean', 0, **B)
    x27['lin'] = ns['boot'](f27('lin'), SMALL, nD, 'median', 0, **B)
    x27['quad'] = ns['boot'](f27('quad'), SMALL, nD, 'median', 2, **B)
    rec = {}
    for m in ALL:
        o = ov(m, np.ones(nD))
        rec[m] = dict(o, loc_h1_minus_Tstar=o['T1'][1] - D[m]['Tstar'], loc_c0_minus_Tm0=o['T0'][1] - D[m]['Tm0'], ratio22=D[m]['ratio22'])
    x27['records'] = rec
    res['XQ27'] = x27
    say('XQ27', x27['word'], round(x27['point'], 3), [round(v, 3) for v in x27['ci']], x27['missing'], flush=True)

    # ---- XQ28 ----
    LC, LR = str(ns['A28_LCUT']), str(ns['A28_LREC'])
    def sN(m, w, kind, use=None):
        d = D[m]
        if kind in ('base', 'pooled', 'mixed'):
            uu = d['u6'] if use is None else use
            ev = d['c0'][..., :4]
            if kind == 'mixed':
                return ns['s_N_mixed'](ev, d['N'], d['Q'], d['temps'], w, uu)
            return ns['s_N_of'](ev, d['N'], d['Q'], d['temps'], w, uu, within=(kind == 'base'))
        tr = d['tr'][int(kind)]
        uL = ns['use_of'](ns['tot'](tr['c0'], np.ones(nD)), ns['tot'](tr['c1'], np.ones(nD))) if use is None else use
        return ns['s_N_of'](tr['c0'][..., :4], tr['N'], tr['Q'], d['temps'], w, uL, within=True)
    x28 = {}
    for kind in ('base', 'mixed', LC):
        r = ns['boot'](lambda m, w, use, kind=kind: (sN(m, w, kind), use), SMALL, nD, 'median', 2, **B)
        r['word'] = ns['word28'](r['point'], r['ci'], r['fail'])
        r['big'] = {m: sN(m, np.ones(nD), kind) for m in BIG}
        x28[kind] = r
        say('XQ28', kind, r['word'], round(r['point'], 4), [round(v, 4) for v in r['ci']], flush=True)
    x28['records'] = {kind: ns['boot'](lambda m, w, use, kind=kind: (sN(m, w, kind), use), SMALL, nD, 'median', 2, **B) for kind in ('pooled', LR)}
    def common(m):
        tr = D[m]['tr'][int(LC)]
        uL = ns['use_of'](ns['tot'](tr['c0'], np.ones(nD)), ns['tot'](tr['c1'], np.ones(nD)))
        return sorted(set(uL) & set(D[m]['u6']))
    def rL(m, w, use):
        cu = common(m); b0 = sN(m, w, 'base', cu)
        return ((sN(m, w, LC, cu) / b0) if (np.isfinite(b0) and b0 != 0) else np.nan), use
    x28['records']['L256_over_base_common_temps'] = ns['boot'](rL, SMALL, nD, 'median', 2, **B)
    res['XQ28'] = x28
    if DRY:
        print('試運転：最後まで回りました（数字は表示せず、出力は書いていません）'); sys.exit()
    json.dump(clean28(res), open(out, 'w'), ensure_ascii=False)
    print('保存しました', out)
