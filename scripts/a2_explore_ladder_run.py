"""探索 8.37 の実行：8.30 の場のデータで、はしごを測り、ものさし（m ＝ 4・8・16）をそろえて、データとひな形をくらべる。"""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
exec(open('scripts/a2_explore_ladder.py').read())
B='<scratch>'
out={}
for name in ['pythia-1b','olmo2-1b']:
    Z={a:np.load(f'{B}/{name}@{a}.npz') for a in ('hm03','h0','hp03')}; temps=[float(t) for t in Z['h0']['temps']]
    H={'hm03':-0.3,'h0':0.0,'hp03':0.3}
    res={}
    for m in (4,8,16):
        dat={a:[] for a in H}; ch={a:[] for a in H}; Md={a:[] for a in H}; Mc={a:[] for a in H}; lads=[]
        for k,T in enumerate(temps):
            lad=measure_ladder(Z['h0']['cand'][k], Z['h0']['pick'][k]); lads.append(lad)
            r0=rho0_profile(lad)
            for a,h in H.items():
                c,M=ruler_counts(Z[a]['pick'][k], m); dat[a].append(c.sum((0,1))); Md[a].append(M.mean())
                ec,EM=chain_counts(lad['rho'], p_field(lad['p'],T,h), r0, m=m); N=c.shape[0]*c.shape[1]
                ch[a].append(ec*N); Mc[a].append(EM)
        R={}
        for a in H:
            R[a]=dict(data=vh_fit(np.array(dat[a]),temps), chain=vh_fit(np.array(ch[a]),temps))
        sl=lambda src:(R['hp03'][src].get('Tm',np.nan)-R['hm03'][src].get('Tm',np.nan))/0.6
        res[m]=dict(vh=R, slope_data=sl('data'), slope_chain=sl('chain'),
                    M_data={a:[float(x) for x in Md[a]] for a in H}, M_chain={a:[float(x) for x in Mc[a]] for a in H},
                    kf_data={a:(np.array(dat[a])[:,1]/np.array(dat[a])[:,0]).tolist() for a in H}, kf_chain={a:(np.array(ch[a])[:,1]/np.array(ch[a])[:,0]).tolist() for a in H},
                    km_data={a:(np.array(dat[a])[:,3]/np.array(dat[a])[:,2]).tolist() for a in H}, km_chain={a:(np.array(ch[a])[:,3]/np.array(ch[a])[:,2]).tolist() for a in H})
        if m==8:
            # ゆらぎと応答、記事の違い、アヴラミ
            Rd=[];Rc=[];Rmix=[];nd=[];npos=[];ncon=[]
            for k,T in enumerate(temps):
                lad=lads[k]; r0=rho0_profile(lad)
                Mh0=ruler_counts(Z['h0']['pick'][k])[1]; slope_d=(ruler_counts(Z['hp03']['pick'][k])[1].mean()-ruler_counts(Z['hm03']['pick'][k])[1].mean())/0.6
                Rd.append(slope_d/(Mh0.var()/T))
                e=0.05; sp=(chain_counts(lad['rho'],p_field(lad['p'],T,e),r0)[1]-chain_counts(lad['rho'],p_field(lad['p'],T,-e),r0)[1])/(2*e)
                mean,var=chain_M_moments(lad['rho'],lad['p'],r0); Rc.append(sp/(var/T))
                means=[];vars_=[];slopes=[]
                for j,f in enumerate(lad['rho0_doc']):
                    rj=np.clip(r0*f,0,1)
                    mj,vj=chain_M_moments(lad['rho'],lad['p'],rj); means.append(mj); vars_.append(vj)
                    slopes.append((chain_counts(lad['rho'],p_field(lad['p'],T,e),rj)[1]-chain_counts(lad['rho'],p_field(lad['p'],T,-e),rj)[1])/(2*e))
                vmix=np.mean(vars_)+np.var(means); Rmix.append(np.mean(slopes)/(vmix/T))
                ff=first_freeze_data(Z['h0']['pick'][k]).ravel()
                Sd=np.array([np.mean((ff<0)|(ff>=t)) for t in range(1,LD_L+1)])
                nd.append(avrami(Sd)); npos.append(avrami(chain_survival(lad['rho'],lad['p'],r0))); ncon.append(avrami(chain_survival(lad['rho'],lad['p'],rho0_profile(lad,positional=False))))
            res[m].update(R_data=Rd,R_chain=Rc,R_chain_docmix=Rmix,avrami_data=nd,avrami_chain_pos=npos,avrami_chain_const=ncon,
                          ladder=[dict(rho=l['rho'].tolist(),p=l['p'].tolist(),rho0_pos=l['rho0_pos'].tolist(),rho0_doc_sd=float(np.std(l['rho0_doc']))) for l in lads])
    out[name]=res
    for m in (4,8,16):
        r=res[m]; v=r['vh']
        print(name,'m',m,'Tm data/chain', [ (round(v[a]['data'].get('Tm',np.nan),3), round(v[a]['chain'].get('Tm',np.nan),3)) for a in H], 'dH h0', round(v['h0']['data'].get('dH',np.nan),2), round(v['h0']['chain'].get('dH',np.nan),2),
              'slope', round(r['slope_data'],3), round(r['slope_chain'],3))
    r=res[8]
    print('  M h0 data', [round(x) for x in r['M_data']['h0']], 'chain', [round(x) for x in r['M_chain']['h0']])
    print('  M hp03 data', [round(x) for x in r['M_data']['hp03']], 'chain', [round(x) for x in r['M_chain']['hp03']])
    print('  M hm03 data', [round(x) for x in r['M_data']['hm03']], 'chain', [round(x) for x in r['M_chain']['hm03']])
    print('  Ea_f/Ea_m data', round(r['vh']['h0']['data']['Ea_f'],2), round(r['vh']['h0']['data']['Ea_m'],2), 'chain', round(r['vh']['h0']['chain']['Ea_f'],2), round(r['vh']['h0']['chain']['Ea_m'],2))
    print('  R data', [round(x,4) for x in r['R_data']], 'chain', [round(x,3) for x in r['R_chain']], 'docmix', [round(x,3) for x in r['R_chain_docmix']])
    print('  avrami data', [round(x,2) for x in r['avrami_data']], 'chain pos', [round(x,2) for x in r['avrami_chain_pos']], 'const', [round(x,2) for x in r['avrami_chain_const']])
    print('  p_k T0.55', [round(x,3) for x in r['ladder'][3]['p'][:12]], 'rho_k', [round(x,3) for x in r['ladder'][3]['rho'][:6]], 'rho0_pos', [round(x,3) for x in r['ladder'][3]['rho0_pos'][:16]], 'doc sd', round(r['ladder'][3]['rho0_doc_sd'],3))
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer)): o=o.item()
    if isinstance(o,float) and not np.isfinite(o): return None
    return o
json.dump(clean(out),open('results/A2_explore/ladder_summary.json','w'),ensure_ascii=False,indent=1); print('done')
