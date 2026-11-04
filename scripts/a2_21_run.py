"""A2-21 の実行（CPU）。
  自己テスト：python3 scripts/a2_21_run.py --self-test
  分析：python3 scripts/a2_21_run.py <G-pre のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_21/a2_21_summary.json
"""
import sys, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_20_run.py').read().split("\nif __name__")[0])

SHA21 = ('scripts/a2_21_analyze.py', 'scripts/a2_21_run.py', 'scripts/a2_explore_g2.py', 'scripts/a2_8_vh.py', 'scripts/a2_explore_onset.py',
         'scripts/a2_16_run.py', 'scripts/a2_18_run.py', 'scripts/a2_20_run.py') + SHA_FILES


def load21ns():
    ns = load20ns()
    exec(open('scripts/a2_explore_g2.py').read(), ns)
    exec(open('scripts/a2_21_analyze.py').read(), ns)
    return ns


def prep(ns, Z):
    toks, pick, q, U = Z['toks'], Z['pick'], Z['q'], Z['U']
    ev, _ = ns['seq_events'](toks, pick, q)
    cw, _ = ns['arm_counts'](toks, U, pick)
    return dict(ev=ev, N=pick.astype(float).sum(-1), Q=q.astype(float).sum(-1), E=U.astype(float).sum(-1), cw=cw, temps=[float(t) for t in Z['temps']])


def cat_groups(folder, m, grid):
    """A2-19 の load_groups と同じ確かめをして、c0 の 3 つの組を記事の向きにつなぐ（U も）。"""
    out = {k: [] for k in ('toks', 'pick', 'q', 'U')}
    for g in range(3):
        Z = np.load(f'{folder}/{m}@c0_g{g}.npz')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in grid], (m, g)
        assert abs(float(Z['h'])) < 1e-12 and int(Z['group']) == g
        assert Z['toks'].shape[0] == 7 and Z['toks'].shape[3] == 512 and not (Z['pick'] & ~Z['cand']).any()
        assert float(np.abs(Z['q'].astype(float) - Z['q0'].astype(float)).max()) == 0.0
        for k in out:
            out[k].append(Z[k])
    res = {k: np.concatenate(v, 2) for k, v in out.items()}
    res['temps'] = np.array(grid, float)
    return res



if __name__ == '__main__':
    ns = load21ns()
    if sys.argv[1] == '--self-test':
        print(json.dumps(ns['self_test21'](), ensure_ascii=False)); sys.exit()
    gp, s2, out = sys.argv[1:4]
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA21}
    print(json.dumps(ns['self_test21'](), ensure_ascii=False), flush=True)
    S16 = json.load(open('results/A2_16/a2_16_summary.json'))['models']
    small = {}
    for m in ns['G16_MODELS']:
        Z = np.load(f'{gp}/{m}@h0.npz')
        assert [round(float(t), 4) for t in Z['temps']] == ns['G16_TEMPS'] and float(Z['h']) == 0.0
        small[m] = prep(ns, Z)
        # 前提の確かめ：n と Δm が A2-16 と一致
        one = np.ones(small[m]['ev'].shape[2])
        c = ns['clap'](small[m]['cw'], small[m]['temps'], one)
        assert c['ok'] and abs(c['n'] - S16[m]['point']['n']) < 1e-9 and abs(c['dm'] - S16[m]['point']['dm']) < 1e-9, (m, c, S16[m]['point'])
        byT = S16[m]['records']['by_T']
        gS_pool = ns['g_from'](small[m]['ev'], small[m]['N'] - small[m]['Q'], small[m]['temps'], one, within=False)
        for i, r in enumerate(byT):                                       # 温度ごとの g_C と、まとめた g_S が A2-16 の記録と一致
            if r['dlnK_M_thermo'] is not None and np.isfinite(c['gC'][i]):
                assert abs(c['gC'][i] - r['dlnK_M_thermo']) < 1e-9, (m, i)
            if r['dlnK_identity'] is not None:
                assert abs(gS_pool[i] - r['dlnK_identity']) < 1e-9, (m, i, gS_pool[i], r['dlnK_identity'])
        print('前提 OK', m, flush=True)
    H = json.load(open('results/A2_20/hstar7b.json'))['models']
    big = {m: prep(ns, cat_groups(s2, m, H[m]['target']['grid'])) for m in ns['A21_BIG']}
    print('読み込み OK', flush=True)
    A = ns['analyze21'](small, 'median')
    B = ns['analyze21'](big, 'mean')
    for m in small:
        A['models'][m]['R_A2_16'] = S16[m]['point']['R']
    S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    for m in big:
        p = B['models'][m]; s = S20[m]
        if not p['ok']:
            continue
        u = sorted(set(s['use']) & set(p['use']))                         # A2-20 の ρ の温度と、ここで点で選んだ温度の共通部分
        D = np.array(s['D']); gC = np.array(p['gC']); gCw = np.array(p['gC_whole']); gN = np.array(p['gN']); gQ = np.array(p['gQ'])
        p['record_temps'] = u
        p['meas_over_C'] = float(D[u].sum() / (s['h'] * gC[u].sum()))
        p['meas_over_Cwhole'] = float(D[u].sum() / (s['h'] * gCw[u].sum()))
        p['one_minus_r_same_temps'] = float(1 - gQ[u].sum() / gN[u].sum())
    res = dict(small=A, big=B, sha256=sha)
    json.dump(clean(res), open(out, 'w'), indent=1, ensure_ascii=False)
    print('XQ8-小', A['word'], round(A['point'], 3), [round(x, 3) for x in A['ci']], 'fail', A['fail'])
    print('XQ8-7B', B['word'], round(B['point'], 3), [round(x, 3) for x in B['ci']], 'fail', B['fail'])
    for g in (A, B):
        for m, p in g['models'].items():
            print(f"{m:16s} κ全体={p.get('kappa', float('nan')):.3f} {[round(x, 2) for x in p['ci']]} κ混ぜ={p.get('kappa_mixed', float('nan')):.3f} r={p.get('r', float('nan')):.3f} n={p.get('n', float('nan')):.2f} n全体={p.get('n_whole', float('nan')):.2f}")
