"""A2-27 の分析の実行（CPU）。ゲート：docs/A2_27_GATE.md。
  自己テスト：python3 scripts/a2_27_run.py --self-test
  分析：python3 scripts/a2_27_run.py <A2-19 のフォルダ> <A2-27 のフォルダ> results/A2_27/a2_27_summary.json
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_26_run.py').read().split("\nif __name__")[0])

SHA27 = ('scripts/a2_27_analyze.py', 'scripts/a2_27_run.py', 'scripts/a2_27_collect.py') + SHA26


def load27ns():
    ns = load26ns()
    exec(open('scripts/a2_27_analyze.py').read(), ns)
    return ns


def load_rescore(folder, m, temps, use):
    """A2-27 の 3 つの組を記事の向きにつなぐ。温度・使う記事が A2-19 と同じかを確かめる。"""
    out = {k: [] for k in ('Ux', 'Uc', 'zx', 'zc', 'lseT', 'q0r')}
    chk = []
    for g in range(3):
        Z = np.load(f'{folder}/{m}@c0_g{g}.npz')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in temps] and int(Z['group']) == g, (m, g)
        for k in out:
            out[k].append(Z[k])
        chk.append(dict(U_max=float(Z['U_max']), U_mean=float(Z['U_mean']), q_max=float(Z['q_max']), seconds=float(Z['seconds'])))
    return {k: np.concatenate(v, 2) for k, v in out.items()}, chk


if __name__ == '__main__':
    ns = load27ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test26'](), ns['self_test27']()); sys.exit()
    f19, f27, out = sys.argv[1:4]
    assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    assert os.path.isdir(os.path.dirname(out)), ('出力のフォルダがありません', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA27}
    print('自己テスト', ns['self_test27'](), flush=True)
    H18 = json.load(open('results/A2_18/hstar.json'))
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']
    S26 = json.load(open('results/A2_26/a2_26_summary.json'))
    SMALL = list(ns['G16_MODELS'])
    ORIG = (32, 0.5, 0.25)
    D = {}
    for m in SMALL:
        t = H18[m]['targets']['-0.10']; S = S19[m]
        Z0 = load_arm(f19, m, 'c0', t['grid'], 0.0); Z1 = load_arm(f19, m, 'h1', t['grid'], S['h'])
        temps = [float(x) for x in t['grid']]
        Z27, chk = load_rescore(f27, m, temps, None)
        # 前提：採点し直しの U・q0 が保存してあるものと合う（Notebook の確かめを、ここでもう一度）
        dU = np.abs(Z27['Ux'].astype(np.float64) - Z0['U'].astype(np.float64))
        assert float(dU.max()) <= 0.01 and float(dU.mean()) <= 1e-4, (m, '前提：U が合わない')
        assert float(np.abs(Z27['q0r'].astype(np.float64) - Z0['q'].astype(np.float64)).max()) <= 1e-3, (m, '前提：q0 が合わない')
        C0, C1 = ns['marks_of'](Z0['toks']), ns['marks_of'](Z1['toks'])
        U0 = Z0['U'].astype(np.float64)
        c0 = ns['counts10'](C0, U0, Z0['pick']); c1 = ns['counts10'](C1, Z1['U'], Z1['pick'])
        sig0 = (Z0['pick'].astype(float) - Z0['q'].astype(float)).sum(-1)
        nD = c0.shape[2]
        D[m] = dict(temps=temps, h=S['h'], C0=C0, U0=U0, pick0=Z0['pick'], c0=c0, c1=c1, sig0=sig0,
                    N=Z0['pick'].astype(float).sum(-1), Q=Z0['q'].astype(float).sum(-1), sb=ns['score_beta'](Z27), Z27=Z27, chk=chk,
                    conv={cv: ns['with_energy'](c0, C0, ns['energy_of'](cv, Z27, U0, temps), Z0['pick']) for cv in ns['A27_CONV']},
                    Etot={cv: ns['energy_of'](cv, Z27, U0, temps).sum(-1) for cv in ns['A27_CONV']})
        # 前提：取り方 (a) の ρ_E が A2-26 の全文の値と一致
        r_a = ns['rho_E_conv'](D[m]['conv']['U'], c1, temps, S['h'], np.ones(nD))[0]
        assert abs(r_a - S26['full']['models'][m]['point']) < 1e-9, (m, '前提：取り方 (a) の ρ_E が A2-26 と合わない')
        print('読み込み', m, flush=True)
    nD = D[SMALL[0]]['c0'].shape[2]
    res = dict(sha256=sha, checks={m: D[m]['chk'] for m in SMALL})

    # ---- XQ24 ----
    b24 = ns['boot'](lambda m, w, use: ns['rho_beta'](D[m]['c0'], D[m]['sb'], D[m]['temps'], w, use), SMALL, nD, 'median', 2, seed=ns['A27_SEED'])
    b24['word'] = ns['word24'](b24['point'], b24['ci'], b24['fail'])
    rec24 = {}
    one = np.ones(nD)
    for m in SMALL:
        temps = D[m]['temps']; t0 = ns['tot'](D[m]['c0'], one); lk = ns['lnK_of'](t0); beta = 1 / np.array(temps)
        y = ns['y_identity'](D[m]['c0'], D[m]['sb'], temps, one)
        o = np.argsort(beta)
        rec24[m] = dict(use=ns['use_c0'](t0), y=y.tolist(), lnK=lk.tolist(), beta=beta.tolist(),
                        slope_fd=[float((lk[o][i + 1] - lk[o][i]) / (beta[o][i + 1] - beta[o][i])) for i in range(len(o) - 1)])
    b24['records'] = rec24
    res['XQ24'] = b24
    print('XQ24', b24['word'], round(b24['point'], 3), [round(x, 3) for x in b24['ci']], flush=True)

    # ---- XQ25 ----
    rows = {}
    for cv in ns['A27_CONV']:
        rows[cv] = ns['boot'](lambda m, w, use, cv=cv: ns['rho_E_conv'](D[m]['conv'][cv], D[m]['c1'], D[m]['temps'], D[m]['h'], w, use),
                              SMALL, nD, 'median', 2, seed=ns['A27_SEED'])
        rows[cv]['info'] = {m: ns['eq_pred'](ns['tot'](D[m]['conv'][cv], one), D[m]['temps'], 1.0,
                                             ns['use_of'](ns['tot'](D[m]['c0'], one), ns['tot'](D[m]['c1'], one)))[1] for m in SMALL}
        print('XQ25', cv, round(rows[cv]['point'], 3), [round(x, 3) for x in rows[cv]['ci']], flush=True)
    res['XQ25'] = dict(word=ns['word25'](list(rows.values())), rows=rows)
    print('XQ25', res['XQ25']['word'], flush=True)

    # ---- R6・R7 ----
    r6 = {}
    for cv in ns['A27_CONV']:
        per = {}
        for key in ('s_N', 's_U_local', 's_U_sum', 'kappa', 'ratio', 'ratio_local'):
            def fn(m, w, use, key=key, cv=cv):
                u = ns['use_of'](ns['tot'](D[m]['c0'], w), ns['tot'](D[m]['c1'], w)) if use is None else use
                if len(u) < ns['A18_MIN_T']:
                    return np.nan, u
                f = ns['factors'](D[m]['c0'], D[m]['conv'][cv], D[m]['N'], D[m]['Q'], D[m]['sb'], D[m]['Etot'][cv], D[m]['temps'], w, u)
                return f[key], u
            per[key] = ns['boot'](fn, SMALL, nD, 'median', 2, seed=ns['A27_SEED'])
        r6[cv] = per
        print('R6', cv, {k: round(v['point'], 4) for k, v in per.items()}, flush=True)
    res['R6'] = r6
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False)
    print('保存しました', out)
