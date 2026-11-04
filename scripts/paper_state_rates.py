import sys, json, numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_26_run.py').read().split("if __name__ == '__main__':")[0])
ns = load26ns()
f19, f20 = sys.argv[1:3]
H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']; S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
SMALL, BIG = list(ns['G16_MODELS']), list(ns['A21_BIG'])
out = {}
for m in SMALL + BIG:
    t = H18[m]['targets']['-0.10'] if m in SMALL else H20[m]['target']
    S = S19[m] if m in SMALL else S20[m]
    Z0 = load_arm(f19 if m in SMALL else f20, m, 'c0', t['grid'], 0.0)
    C0 = ns['marks_of'](Z0['toks'])
    c0 = ns['counts10'](C0, Z0['U'], Z0['pick'])
    tt = ns['tot'](c0, np.ones(c0.shape[2]))[S['use']].sum(0)
    out[m] = dict(ordinary=float(tt[8] / tt[5]), repetitive=float(tt[9] / tt[7]))
    print(m, round(out[m]['ordinary'], 3), round(out[m]['repetitive'], 3), flush=True)
json.dump(out, open(sys.argv[3], 'w'), indent=1)
