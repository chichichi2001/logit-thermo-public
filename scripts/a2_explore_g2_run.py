"""8.41 の実行：python3 scripts/a2_explore_g2_run.py <A2-19 のフォルダ> <A2-18 のフォルダ> <出力 json>（探索、判定しない）。"""
import sys, json, ast, numpy as np
A19, A18, OUT = sys.argv[1:4]; sys.argv = sys.argv[:1]
exec(open('scripts/a2_20_run.py').read().split("\nif __name__")[0])
ns=load20ns(); exec(open('scripts/a2_explore_g2.py').read(), ns)
tree=ast.parse(open('scripts/a2_18_collect.py').read()); J={(n,a):(h,t) for n,a,h,t in next(ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign) and getattr(x.targets[0],'id','')=='A18_JOBS')}
names=['pythia-410m','pythia-1.4b','gpt-neo-1.3b','smollm2-1.7b','qwen2.5-1.5b','gpt2-1.5b','bloom-1b1']
def load(base, n, a, groups):
    h,t=J[(n,a)]
    if groups:
        tk,pk,q=ns['load_groups'](base,n,a,h,t)
    else:
        Z=np.load(f'{base}/{n}@{a}.npz'); tk,pk,q=Z['toks'],Z['pick'],Z['q']
    return ns['seq_events2'](tk,pk,q)
def rhos(E0, E1, h, temps, w):
    l0,f0,m0=ns['lnK_from'](E0[0],w); l1,f1,m1=ns['lnK_from'](E1[0],w)
    use=[k for k in range(len(temps)) if min(f0[k],m0[k],f1[k],m1[k])>=10]
    g,g2=ns['g12_identity'](E0[0],E0[1],E0[2],temps,w)
    u=np.array(use); D=(l1-l0)[u]; P1=(h*g)[u]; P2=(h*g+0.5*h*h*g2)[u]
    return D.sum()/P1.sum(), D.sum()/P2.sum(), float(np.mean(g2[u]/g[u]))
out={}
for tag,base,groups,arms in (('A2-19',A19,True,['h1']),('A2-18',A18,False,['h1','h2'])):
    for n in names:
        E0=load(base,n,'c0',groups); nD=E0[0].shape[2]; w=np.ones(nD); temps=J[(n,'c0')][1]
        for a in arms:
            E1=load(base,n,a,groups); h=J[(n,a)][0]
            r1,r2,ratio=rhos(E0,E1,h,temps,w)
            out[(tag,a,n)]=(r1,r2,ratio,h)
            print(tag,a,n,f'h={h:+.3f} rho1={r1:.3f} rho2={r2:.3f} mean(g2/g)={ratio:+.3f}', flush=True)
for key in (('A2-19','h1'),('A2-18','h1'),('A2-18','h2')):
    v=[out[(key[0],key[1],n)] for n in names]
    print(key,'median rho1',round(float(np.median([x[0] for x in v])),3),'median rho2',round(float(np.median([x[1] for x in v])),3))
json.dump({'|'.join(k):v for k,v in out.items()},open(OUT,'w'))
