"""A2-22 の実行（CPU）。
  自己テスト：python3 scripts/a2_22_run.py --self-test
  分析：python3 scripts/a2_22_run.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_22/a2_22_summary.json
"""
import sys, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_21_run.py').read().split("\nif __name__")[0])

SHA22 = ('scripts/a2_22_analyze.py', 'scripts/a2_22_run.py') + SHA21


def load22ns():
    ns = load21ns()
    exec(open('scripts/a2_22_analyze.py').read(), ns)
    return ns


def load_arm(folder, m, arm, grid, h):
    """A2-19 の load_groups と同じ確かめで、3 つの組を記事の向きにつなぐ（U も）。"""
    out = {k: [] for k in ('toks', 'pick', 'q', 'U')}
    for g in range(3):
        Z = np.load(f'{folder}/{m}@{arm}_g{g}.npz')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in grid], (m, arm, g)
        assert abs(float(Z['h']) - h) < 1e-6 and int(Z['group']) == g, (m, arm, g, float(Z['h']))
        assert Z['toks'].shape[0] == 7 and Z['toks'].shape[3] == 512 and not (Z['pick'] & ~Z['cand']).any()
        if abs(h) < 1e-12:
            assert float(np.abs(Z['q'].astype(float) - Z['q0'].astype(float)).max()) == 0.0
        for k in out:
            out[k].append(Z[k])
    res = {k: np.concatenate(v, 2) for k, v in out.items()}
    res['temps'] = np.array(grid, float)
    return res


if __name__ == '__main__':
    ns = load22ns()
    if sys.argv[1] == '--self-test':
        print(json.dumps(ns['self_test22'](), ensure_ascii=False)); sys.exit()
    f19, f20, out = sys.argv[1:4]
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA22}
    print(json.dumps(ns['self_test22'](), ensure_ascii=False), flush=True)
    H18 = json.load(open('results/A2_18/hstar.json'))
    H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']
    S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    small, big = {}, {}
    for m in ns['G16_MODELS']:
        t = H18[m]['targets']['-0.10']
        assert abs(t['hstar'] - S19[m]['h']) < 1e-3, (m, t['hstar'], S19[m]['h'])
        small[m] = dict(c0=ns['prep22'](load_arm(f19, m, 'c0', t['grid'], 0.0)), h1=ns['prep22'](load_arm(f19, m, 'h1', t['grid'], S19[m]['h'])),
                        h=S19[m]['h'], rC=t['r_clapeyron'], Tstar=t['Tstar'])
        print('読み込み', m, flush=True)
    for m in ns['A21_BIG']:
        t = H20[m]['target']
        assert abs(t['hstar'] - S20[m]['h']) < 1e-3
        big[m] = dict(c0=ns['prep22'](load_arm(f20, m, 'c0', t['grid'], 0.0)), h1=ns['prep22'](load_arm(f20, m, 'h1', t['grid'], S20[m]['h'])),
                      h=S20[m]['h'], rC=t['r_clapeyron'], Tstar=t['Tstar'])
        print('読み込み', m, flush=True)
    # 前提の確かめ：Δln K が A2-19・A2-20 の記録（D）と一致
    for S, D_ in ((S19, small), (S20, big)):
        for m, d in D_.items():
            p = ns['model_point'](d['c0'], d['h1'], d['h'], np.ones(d['c0']['ev'].shape[2]))
            assert np.allclose(np.array(p['dlnK'])[S[m]['use']], np.array(S[m]['D'])[S[m]['use']], atol=1e-9), m
    print('前提 OK', flush=True)
    A = ns['analyze22'](small, 'median'); B = ns['analyze22'](big, 'mean')
    json.dump(clean(dict(small=A, big=B, sha256=sha)), open(out, 'w'), indent=1, ensure_ascii=False)
    for nm, g in (('小', A), ('7B', B)):
        print(f"XQ10-{nm}", g['XQ10']['word'], round(g['XQ10']['point'], 3), [round(x, 3) for x in g['XQ10']['ci']], 'fail', g['XQ10']['fail'])
        print(f"XQ11-{nm}", g['XQ11']['word'], round(g['XQ11']['point'], 3), [round(x, 3) for x in g['XQ11']['ci']], 'fail', g['XQ11']['fail'])
    for g in (A, B):
        for m, p in g['models'].items():
            print(f"{m:16s} 比={p['ratio']:.2f} {[round(x, 2) for x in p['ratio_ci']]} f入={p['f_entry']:.2f} {[round(x, 2) for x in p['f_entry_ci']]} "
                  f"恒等式のf入={p['f_entry_identity']:.2f} 状態={ {k: round(v, 2) for k, v in p['state_share'].items()} } "
                  f"平衡比={p['ratio_over_rC']:.2f} T_m(h1)−T*={p['Tm_h1_minus_Tstar']:+.3f} 重み付け={ {k: (round(v['meas_over_is'], 2), round(v['ess'], 2)) for k, v in p['is'].items()} }")
