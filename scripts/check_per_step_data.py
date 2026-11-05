"""公開した歩ごとの数だけから、登録した試験の比 ρ を出し直して、記録と照らし合わせる（トークンは使わない）。
  python3 scripts/check_per_step_data.py data/per_step
追試（A2-19）と 7B（A2-20）は 3 つの組を記事の向きにつなぐ。最初の試験（A2-18）は h* と 2h*。
"""
import sys, json
import numpy as np
sys.path.insert(0, 'scripts')
ns = {}
exec(open('scripts/a2_18_analyze.py').read(), ns)


def arm(path_fmt, groups):
    ev, sig = [], []
    for g in groups:
        Z = np.load(path_fmt.format(g=g))
        ev.append(Z['events'].astype(float)); sig.append(Z['N'] - Z['Q'])
    return np.concatenate(ev, 2), np.concatenate(sig, 2)


if __name__ == '__main__':
    D = sys.argv[1]
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    R18 = json.load(open('results/A2_18/a2_18_summary.json'))
    S19 = json.load(open('results/A2_19/a2_19_summary.json')); S20 = json.load(open('results/A2_20/a2_20_summary.json'))
    worst = 0.0
    for m, rec in S19['models'].items():
        E = {a: arm(f'{D}/replication_A2_19/{m}@{a}_g{{g}}.npz', range(3)) for a in ('c0', 'h1')}
        r, u, _, _ = ns['rho_model'](E, ('h1', rec['h']), H18[m]['targets']['-0.10']['grid'], np.ones(E['c0'][0].shape[2]))
        assert u == rec['use']; worst = max(worst, abs(r - rec['rho']))
    for m, rec in S20['models'].items():
        E = {a: arm(f'{D}/larger_models_stage2_A2_20/{m}@{a}_g{{g}}.npz', range(3)) for a in ('c0', 'h1')}
        r, u, _, _ = ns['rho_model'](E, ('h1', rec['h']), H20[m]['target']['grid'], np.ones(E['c0'][0].shape[2]))
        assert u == rec['use']; worst = max(worst, abs(r - rec['rho']))
    for m in S19['models']:
        for a in ('h1', 'h2'):
            E = {b: arm(f'{D}/first_test_A2_18/{m}@{b}.npz', [0]) for b in ('c0', a)}
            rec = R18['models'][m][a]
            r, u, _, _ = ns['rho_model'](E, (a, rec['h']), H18[m]['targets']['-0.10']['grid'], np.ones(E['c0'][0].shape[2]))
            assert u == rec['use']; worst = max(worst, abs(r - rec['rho']))
    print('登録の ρ との差の最大', worst)
    assert worst < 1e-6
