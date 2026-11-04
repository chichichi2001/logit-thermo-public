"""A2-3 の分析コードを作り物のデータで確かめる（本物のデータを開く前）。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
ns = {}
exec(open(os.path.join(os.path.dirname(__file__), 'a2_2_analyze.py'), encoding='utf-8').read(), ns)
exec(open(os.path.join(os.path.dirname(__file__), 'a2_3_analyze.py'), encoding='utf-8').read(), ns)
TEMPS = np.array([0.6,0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5]); TT = np.exp(np.linspace(np.log(0.4), np.log(2.5), 61))
nD, S, L = 40, 2, 512
def make(path, T0, Tc=1.0, seed=0, noise=1.0, flat=False):
    r = np.random.default_rng(seed)
    tf = np.zeros((4, nD, 15, L), np.float32); gen = np.zeros((4, 15, S, nD, L), np.float32)
    for q in range(4):
        base = -5 + 3*np.log(TEMPS)[None,:,None] + r.normal(0, noise, (nD, 15, L)); tf[q] = base
        slope = 0.0 if flat else 2.0
        f = slope*(np.log(TEMPS) - np.log(T0))[None,None,:,None] + r.normal(0, noise, (S, nD, 15, L)) + r.normal(0,0.2,(1,nD,1,1))
        gen[q] = (base[None] + f).transpose(2, 0, 1, 3)
    tcal = -(np.log(TT) - np.log(Tc))[None, :]*100 + r.normal(0, 1, (nD, 61))
    np.savez_compressed(path, tf=tf, gen=gen, tcal=tcal, tcal_T=TT, toks=r.integers(0,1000,(15,S,nD,L)).astype(np.int32),
                        temps=TEMPS, nll=r.uniform(0.5,2,nD), ext=np.array(['.py']*nD), chunk=40, use=np.arange(nD), revision='syn')
base = sys.argv[1]
for label, T0s in [('all in', dict(q7=0.90, o7=0.85, qc=0.88, sc=0.82, pc=0.86)),
                   ('qwen7b high', dict(q7=1.10, o7=0.85, qc=0.88, sc=0.82, pc=0.86)),
                   ('code all high', dict(q7=0.90, o7=0.85, qc=1.15, sc=1.12, pc=1.2))]:
    d = os.path.join(base, label.replace(' ', '_')); os.makedirs(d, exist_ok=True)
    make(f'{d}/qwen2.5-1.5b@bbc-a100.npz', 0.91, seed=1); make(f'{d}/a22.npz', 0.91, seed=1)
    for n, k in [('qwen2.5-7b@bbc','q7'),('olmo2-7b@bbc','o7'),('qwen2.5-1.5b@code','qc'),('smollm2-1.7b@code','sc'),('pythia-1b@code','pc')]:
        make(f'{d}/{n}.npz', T0s[k], seed=hash(n) % 1000)
    r = ns['analyze3'](d, f'{d}/a22.npz')
    print('==', label, '| main', r['overall_main'], '| ref', r['overall_ref'], '| size', r['axis_size'], '| text', r['axis_text'], '| Q3', r['Q3_code_in_band'], '| Q4', r['Q4'], '| a100', r['a100_check']['ok'])
    print('   ', {n: (round(c['fb_H']['T0'],3), c['main'], c['ref']) for n, c in r['conditions'].items()})
