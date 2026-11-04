import sys, json, numpy as np; GP=sys.argv[1]; sys.argv=sys.argv[:1]
exec(open('scripts/a2_20_run.py').read().split("\nif __name__")[0])
ns=load20ns(); exec(open('scripts/a2_explore_toy.py').read(), ns); exec(open('scripts/a2_20_toy.py').read(), ns)
TOY=json.load(open('results/A2_explore/toy_small7.json')); G16=[0.4,0.45,0.5,0.55,0.6,0.65,0.7]
R=json.load(open('results/A2_explore/models_compare.json'))
rows=R['rows']; ex=[r for r in rows if not (r['model']=='pythia-1.4b' and r['arm'].startswith('A2-18'))]
chi=lambda rs,m: sum(((r['rho']-r[m])/r['sd'])**2 for r in rs)
print('χ² without Pythia-1.4B A2-18 arms:', round(chi(ex,'ladder'),1), round(chi(ex,'bell'),1), 'n', len(ex))
a19=[r for r in rows if r['arm']=='A2-19 h1']; print('χ² A2-19 only:', round(chi(a19,'ladder'),1), round(chi(a19,'bell'),1))
names=list(TOY); rng=np.random.default_rng(5); D=[]; T=[]; B=[[] for _ in range(1000)]
EE={}
for n in names:
    Z={a:np.load(f'{GP}/{n}@{a}.npz') for a in ('h0','hm03','hp03')}
    E={a:ns['seq_events'](Z[a]['toks'],Z[a]['pick'],Z[a]['q']) for a in Z}; EE[n]=E
nD=40; one=np.ones(nD)
def diff(n,w):
    E=EE[n]; rm=ns['rho_model']({'c0':E['h0'],'hm03':E['hm03']},('hm03',-0.3),G16,w)[0]; rp=ns['rho_model']({'c0':E['h0'],'hp03':E['hp03']},('hp03',0.3),G16,w)[0]; return rm-rp
for n in names:
    d=diff(n,one); t=ns['toy_rho1'](TOY[n]['params'],-0.3,G16)[0]-ns['toy_rho1'](TOY[n]['params'],0.3,G16)[0]; D.append(d); T.append(t)
    print(n,'測った ρ(−0.3)−ρ(+0.3)',round(d,3),'はしご',round(t,3))
med=[]
for b in range(1000):
    w=np.bincount(rng.integers(0,nD,nD),minlength=nD).astype(float); med.append(np.median([diff(n,w) for n in names]))
print('median measured',round(float(np.median(D)),3),'CI',[round(x,3) for x in np.nanpercentile(med,[2.5,97.5])],'ladder median',round(float(np.median(T)),3),'bell 0')
