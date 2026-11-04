import sys, os, numpy as np, json
sys.path.insert(0, 'scripts')
import a2_2_analyze as A
TEMPS = np.array([0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5])
TT = np.exp(np.linspace(np.log(0.4), np.log(2.5), 61))
nD, S, L = 40, 2, 512
def make(d, T0s, Tcals, seed=0):
    os.makedirs(d, exist_ok=True); r = np.random.default_rng(seed)
    for (name, fam, kind) in A.MODELS:
        T0, Tc = T0s[name], Tcals[name]
        tf = np.zeros((4, nD, 15, L), np.float32); gen = np.zeros((4, 15, S, nD, L), np.float32)
        for q in range(4):
            base = -5 + 3*np.log(TEMPS)[None,:,None] + r.normal(0, 1, (nD, 15, L))
            tf[q] = base
            f = 2.0*(np.log(TEMPS) - np.log(T0))[None,None,:,None] + r.normal(0, 1, (S, nD, 15, L)) + r.normal(0,0.2,(1,nD,1,1))
            gen[q] = (base[None] + f).transpose(2, 0, 1, 3)
        tcal = -(np.log(TT) - np.log(Tc))[None, :]*100 + r.normal(0, 1, (nD, 61))
        np.savez_compressed(f'{d}/{name}.npz', tf=tf, gen=gen, tcal=tcal, tcal_T=TT, a=np.full((nD, L), 1.4 + 0.1*r.normal()),
                            h=np.ones((nD, L)), toks=r.integers(0, 1000, (15, S, nD, L)).astype(np.int32), temps=TEMPS, H2=50, tok_len=50000)
    for rd, wr in A.CROSS:
        out = np.stack([r.normal(-3, 0.1, (15, S, nD)), r.normal(-4, 0.1, (15, S, nD))])
        np.savez_compressed(f'{d}/cross_{rd}_reads_{wr}.npz', out=out, temps=TEMPS)
names = [m[0] for m in A.MODELS]
Tc = dict(zip(names, [0.95, 1.0, 1.25, 1.05, 0.9, 1.0, 1.3]))
base = '<scratch>'
for label, T0 in [('E holds (T0=Tcal)', Tc), ('E fails (T0=1.3*Tcal)', {k: 1.3*v if 'inst' in k else v*1.0 for k, v in Tc.items()}),
                  ('E fails (T0 const 1.15)', {k: 1.15 for k in Tc})]:
    d = f'{base}/{label[:7].replace(" ","_")}{len(label)}'
    make(d, T0, Tc)
    r = A.analyze(d)
    j = r['judge_H']
    print('==', label)
    print('  T0 rec:', {n: round(r['models'][n]['fb_H']['T0'], 3) for n in names})
    print('  Tcal rec:', {n: round(r['models'][n]['Tcal'], 3) for n in names})
    print('  eps', round(r['models']['pythia-1b']['fb_H']['eps'], 3), 'ci', [round(x,3) for x in r['models']['pythia-1b']['fb_H']['T0_ci']])
    print('  E1', j['E1']['verdict'], [(round(p['d_cal'],3), round(p['d_0'],3)) for p in j['E1']['pairs']])
    print('  E2', j['E2']['verdict'], {k: round(v,3) for k, v in j['E2']['mean_err'].items()}, 'undec', j['undecidable'])
    print('  P2', j['P2']['verdict'], round(j['P2']['err_x1'],4), round(j['P2']['err_x3'],4), ' Q6', r['Q6']['verdict'])

print('\n== P2 check: F depends only on the one-step state (should hold) vs only on T (should not)')
import a2_2_analyze as A
for label, mode in [('P2 holds', 'state'), ('P2 broken', 'temp')]:
    d = f'{base}/p2_{mode}'; os.makedirs(d, exist_ok=True); r = np.random.default_rng(1)
    for j, (name, fam, kind) in enumerate(A.MODELS):
        sh = [0.8, 0.9, 1.25, 1.0, 1.1, 0.95, 1.35][j]
        tf = np.zeros((4, nD, 15, L), np.float32); gen = np.zeros((4, 15, S, nD, L), np.float32)
        for q in range(4):
            state = 3*np.log(TEMPS/sh)
            base_ = -5 + state[None,:,None] + r.normal(0, 1, (nD, 15, L))
            tf[q] = base_
            Fv = np.tanh(state) if mode == 'state' else np.tanh(3*np.log(TEMPS))
            f = Fv[None,None,:,None] + r.normal(0, 1, (S, nD, 15, L))
            gen[q] = (base_[None] + f).transpose(2, 0, 1, 3)
        tcal = -(np.log(TT) - np.log(sh))[None, :]*100 + r.normal(0, 1, (nD, 61))
        np.savez_compressed(f'{d}/{name}.npz', tf=tf, gen=gen, tcal=tcal, tcal_T=TT, a=np.full((nD, L), 1.4), h=np.ones((nD, L)),
                            toks=r.integers(0, 1000, (15, S, nD, L)).astype(np.int32), temps=TEMPS, H2=50, tok_len=50000)
    for rd, wr in A.CROSS:
        np.savez_compressed(f'{d}/cross_{rd}_reads_{wr}.npz', out=np.stack([r.normal(-3,.1,(15,S,nD)), r.normal(-4,.1,(15,S,nD))]), temps=TEMPS)
    j = A.analyze(d)['judge_H']
    print(' ', label, j['P2']['verdict'], 'x1', round(j['P2']['err_x1'],4), 'x3', round(j['P2']['err_x3'],4), 'outside x3', round(j['P2']['outside_x3'],3), '| E2', j['E2']['verdict'], 'undec', j['undecidable']['E2_P2_undecidable'])
