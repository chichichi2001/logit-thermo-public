"""探索 8.36：写しのはしごのひな形を、Pythia-1B・OLMo-2-1B の 5 つの数（T_m・ΔH_vH・E_a,f・液体と結晶の候補を選ぶ割合）に合わせる（多数の出発点・範囲つきの最小二乗、m は 3〜8 から選ぶ）。"""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import least_squares
exec(open('scripts/a2_explore_toy.py').read())
TARGETS = {'pythia-1b': dict(Tm=0.595, dH=3.32, Ea_f=-1.54, m_liq=0.185, m_cry=0.931),
           'olmo2-1b': dict(Tm=0.550, dH=2.95, Ea_f=-1.26, m_liq=0.153, m_cry=0.906)}
SC = dict(Tm=0.02, dH=0.2, Ea_f=0.1, m_liq=0.02, m_cry=0.02)
MM=[8]
def mk(x): a,b,lD,lB,lr = x; return Toy(a=a,b=b,Delta=np.exp(lD),B=np.exp(lB),rho=1/(1+np.exp(-lr)),m=MM[0])
def resid(x, tg):
    t=mk(x)
    try:
        v=t.vh(); pm=t.phase_means(v['Tm'] if np.isfinite(v['Tm']) else 0.55)
    except Exception: return np.full(5,1e3)
    got=dict(Tm=v['Tm'],dH=v['dH'],Ea_f=v['Ea_f'],m_liq=pm['m_liq'],m_cry=pm['m_cry'])
    r=np.array([(got[k]-tg[k])/SC[k] for k in tg]); return np.where(np.isfinite(r),r,1e3)
out={}
for name,tg in TARGETS.items():
    best=None; bm=None
    for mm in (3,4,5,6,8):
      MM[0]=mm
      rng=np.random.default_rng(0); lo=np.array([-2,0,np.log(0.3),0,-6]); hi=np.array([4,0.3,np.log(30),35,3])
      for i in range(15):
        x0=lo+(hi-lo)*rng.random(5)
        r=least_squares(resid,x0,args=(tg,),bounds=(lo,hi),max_nfev=300)
        if best is None or r.cost<best.cost: best=r; bm=mm
      print(name,'m',mm,'best so far',round(best.cost,3),bm,flush=True)
    MM[0]=bm
    t=mk(best.x); rep=toy_report(t)
    par=dict(a=float(best.x[0]),b=float(best.x[1]),Delta=float(np.exp(best.x[2])),B=float(np.exp(best.x[3])),rho=float(1/(1+np.exp(-best.x[4]))),m=bm)
    print(name,'cost',round(best.cost,4),'params',{k:round(v,3) for k,v in par.items()})
    print({k:(round(v,3) if isinstance(v,float) else [round(float(x),3) for x in v]) for k,v in rep.items()})
    out[name]=dict(params=par,cost=float(best.cost),report={k:(float(v) if isinstance(v,float) else [float(x) for x in v]) for k,v in rep.items()})
json.dump(out,open('results/A2_explore/toy_fit.json','w'),indent=1)
