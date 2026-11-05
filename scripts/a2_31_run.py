"""A2-31 の実行（CPU）。ゲート：docs/A2_31_GATE.md。
  自己テスト：python3 scripts/a2_31_run.py --self-test
  前提だけ：  python3 scripts/a2_31_run.py <A2-18> <A2-19> <A2-20 第 2 段> <出力> --dry
  分析：      python3 scripts/a2_31_run.py <A2-18> <A2-19> <A2-20 第 2 段> results/A2_31/a2_31_summary.json
標準出力には自己テストと前提の確かめだけを出す（新しい点や区間は出さない）。
"""
import sys, os, json, hashlib, time
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_30_run.py').read().split("\nif __name__")[0])

SHA31 = ('scripts/a2_31_analyze.py', 'scripts/a2_31_run.py', 'docs/A2_31_GATE.md') + SHA30


def load31ns():
    ns = load30ns()
    exec(open('scripts/a2_31_analyze.py').read(), ns)
    return ns


if __name__ == '__main__':
    ns = load31ns()
    print('自己テスト', ns['self_test31'](), flush=True)
    if sys.argv[1] == '--self-test':
        sys.exit()
    f18, f19, f20, out = sys.argv[1:5]
    DRY = '--dry' in sys.argv
    if not DRY:
        assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
        assert os.path.isdir(os.path.dirname(out)), ('出力のフォルダがありません', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA31}
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    R18 = json.load(open('results/A2_18/a2_18_summary.json'))
    S19 = json.load(open('results/A2_19/a2_19_summary.json')); S20 = json.load(open('results/A2_20/a2_20_summary.json'))
    SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
    t0 = time.time()
    R, F = {}, {}
    for m in SMALL + BIG:
        t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
        S = S19['models'][m] if m in SMALL else S20['models'][m]
        f = f19 if m in SMALL else f20
        Z0 = load_arm(f, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f, m, 'h1', t['grid'], S['h'])
        e0, s0 = ns['seq_events'](Z0['toks'], Z0['pick'], Z0['q']); e1, s1 = ns['seq_events'](Z1['toks'], Z1['pick'], Z1['q'])
        grid = [float(x) for x in t['grid']]
        LW = np.stack([ns['logw_of'](Z0['pick'][k], Z0['q'][k], S['h'], T) for k, T in enumerate(grid)])
        LWc = ns['logw_check'](Z0['pick'][0][:1, :2], Z0['q'][0][:1, :2], S['h'], grid[0])
        assert np.allclose(LW[0][:1, :2], LWc, atol=1e-9), (m, '前提：log w の 2 つの書き方が合わない')
        for k, T in enumerate(grid):
            assert np.allclose(ns['logw_of'](Z0['pick'][k], Z0['q'][k], 0.0, T), 0.0), (m, k, '前提：h ＝ 0 で log w が 0 でない')
        V2 = np.stack([(S['h'] / T) ** 2 * (Z0['q'][k].astype(float) * (1 - Z0['q'][k].astype(float))).sum(-1) for k, T in enumerate(grid)])
        R[m] = dict(E={'c0': (e0, s0), 'h1': (e1, s1)}, h=S['h'], grid=grid, use=list(S['use']), LW=LW, V2=V2)
        del Z0, Z1
        print('読み込み', m, round(time.time() - t0), flush=True)
    for m in SMALL:
        t = H18[m]['targets']['-0.10']
        E = {}
        for arm in ('c0', 'h1'):
            Z = np.load(f'{f18}/{m}@{arm}.npz')
            assert [round(float(x), 4) for x in Z['temps']] == [round(float(x), 4) for x in t['grid']], (m, arm)
            E[arm] = ns['seq_events'](Z['toks'], Z['pick'], Z['q'])
        F[m] = dict(E=E, h=R18['models'][m]['h1']['h'], use=R18['models'][m]['h1']['use'], grid=[float(x) for x in t['grid']])
    # ---- 前提 ----
    nD = R[SMALL[0]]['E']['c0'][0].shape[2]; nD18 = F[SMALL[0]]['E']['c0'][0].shape[2]
    for m in SMALL + BIG:
        r, u, _, _ = ns['rho_model'](R[m]['E'], ('h1', R[m]['h']), R[m]['grid'], np.ones(nD))
        rec = S19['models'][m] if m in SMALL else S20['models'][m]
        assert u == R[m]['use'] and abs(r - rec['rho']) < 1e-9, (m, '前提：ρ・使う温度が記録と合わない')
    for m in SMALL:
        r, u, _, _ = ns['rho_model'](F[m]['E'], ('h1', F[m]['h']), F[m]['grid'], np.ones(nD18))
        assert u == F[m]['use'] and abs(r - R18['models'][m]['h1']['rho']) < 1e-9, (m, '前提：最初の試験が記録と合わない')
    print('前提の確かめ：通った', round(time.time() - t0), flush=True)
    if DRY:
        sys.exit()
    # ---- XQ36 と R18（ばらつき） ----
    res = dict(sha256=sha)
    fr = lambda E, h, grid, use: (lambda w: ns['rho_model'](E, ('h1', h), grid, w, use)[0])
    fns = [fr(R[m]['E'], R[m]['h'], R[m]['grid'], R[m]['use']) for m in SMALL]
    x = np.log([f(np.ones(nD)) for f in fns])
    B = ns['boot_logrho'](fns, nD, ns['A31_SEED36'], ns['A31_BOOT'])
    ok = np.all(np.isfinite(B), 1); fail = float(1 - ok.mean())
    Sig = np.cov(B[ok].T); c = ns['cochran'](x, Sig)
    Sr = ns['sig_robust'](B[ok]); cr = ns['cochran'](x, Sr)
    res['XQ36'] = dict(models=SMALL, logrho=x.tolist(), cov=Sig.tolist(), fail=fail, **c, word=ns['word36'](c, fail),
                       flagged=[m for m, z in zip(SMALL, c['z']) if abs(z) > ns['A31_RESID']],
                       robust=dict(Q=cr['Q'], p=cr['p'], tau_up=cr['tau_up'], word=ns['word36'](cr, fail)))
    print('XQ36 終わり', round(time.time() - t0), flush=True)
    # ---- XQ37 と R18（まとめた重み付け） ----
    recs = {}
    for m in SMALL + BIG:
        ess = [ns['ess_of'](R[m]['LW'][k]) for k in R[m]['use']]
        vlw = [float(np.var(R[m]['LW'][k].ravel(), ddof=1)) for k in R[m]['use']]
        ev2 = [float(np.mean(R[m]['V2'][k])) for k in R[m]['use']]
        recs[m] = dict(ess_median=float(np.median(ess)), ess=ess, var_logw_median=float(np.median(vlw)), EV2_median=float(np.median(ev2)))
    pts = {m: ns['rho_IS'](R[m]['E'], R[m]['LW'], R[m]['use'], np.ones(nD)) for m in SMALL}
    rng = np.random.default_rng(ns['A31_SEED37'])
    bs, bm = [], []
    for _ in range(ns['A31_BOOT']):
        w = ns['resample'](rng, nD)
        v = [ns['rho_IS'](R[m]['E'], R[m]['LW'], R[m]['use'], w)[0] for m in SMALL]
        bm.append(v); bs.append(ns['agg30'](v, 'median'))
    bs = np.array(bs, float); bm = np.array(bm, float); fail = float(np.mean(~np.isfinite(bs)))
    ess_all = float(np.median([recs[m]['ess_median'] for m in SMALL]))
    ci = ns['ci_of'](bs)
    for i, m in enumerate(SMALL):
        recs[m].update(rho_IS=pts[m][0], sumD=pts[m][1], sumIS=pts[m][2], ci=ns['ci_of'](bm[:, i]), fail=float(np.mean(~np.isfinite(bm[:, i]))))
        recs[m]['word'] = ns['word37'](recs[m]['ci'], recs[m]['fail'], recs[m]['ess_median'])
    res['XQ37'] = dict(point=ns['agg30']([pts[m][0] for m in SMALL], 'median'), ci=ci, fail=fail, ess_all=ess_all,
                       word=ns['word37'](ci, fail, ess_all), models=recs)
    print('XQ37 終わり', round(time.time() - t0), flush=True)
    # ---- R17（最初の試験と追試の再現） ----
    f19s = [fr(R[m]['E'], R[m]['h'], R[m]['grid'], R[m]['use']) for m in SMALL]
    f18s = [fr(F[m]['E'], F[m]['h'], F[m]['grid'], F[m]['use']) for m in SMALL]
    B19 = ns['boot_logrho'](f19s, nD, ns['A31_SEED_R17A'], ns['A31_BOOT'])
    B18 = ns['boot_logrho'](f18s, nD18, ns['A31_SEED_R17B'], ns['A31_BOOT'])
    x19 = np.log([f(np.ones(nD)) for f in f19s]); x18 = np.log([f(np.ones(nD18)) for f in f18s])
    Dd = B19 - B18
    mods = {m: dict(d=float(x19[i] - x18[i]), ci=ns['ci_of'](Dd[:, i]), fail=float(np.mean(~np.isfinite(Dd[:, i]))),
                    rho_rep=float(np.exp(x19[i])), rho_first=float(np.exp(x18[i]))) for i, m in enumerate(SMALL)}
    excl = [m for m in SMALL if mods[m]['ci'][0] > 0 or mods[m]['ci'][1] < 0]
    res['R17'] = dict(models=mods, excluding_zero=excl, reading='再現しない' if len(excl) <= 1 else '再現するずれがある')
    res['time_s'] = round(time.time() - t0)
    json.dump(clean28(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print('書き出し', out, hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16], flush=True)
