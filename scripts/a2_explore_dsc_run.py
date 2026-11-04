"""探索 8.33 の実行：固定した分析（scripts/a2_explore_dsc_analyze.py）を DSC のデータに当てる。"""
import sys, json, hashlib, numpy as np, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'<scratch>')
from load import load13ns
ns=load13ns(); exec(open('scripts/a2_explore_dsc_analyze.py').read(), ns); print(ns['self_test_dsc_an']())
B='<scratch>'
out={'sha':hashlib.sha256(open('scripts/a2_explore_dsc_analyze.py','rb').read()).hexdigest()}
for m in ['pythia-1b','olmo2-1b']:
    t=time.time(); r=ns['analyze_dsc'](B,m); out[m]=r; print(m, round(time.time()-t),'s')
    for c,v in r['comb'].items():
        print(' ',c,{k:round(x,3) for k,x in v.items()}, 'ci area', [round(x,3) for x in r['ci'][c+'|area']], 'center', [round(x,3) for x in r['ci'][c+'|center']], 'width', [round(x,3) for x in r['ci'][c+'|width']])
    for k,a in r['arms'].items(): print('  arm',k,'T_cool',round(a['T_cool'],3),'T_heat',round(a['T_heat'],3),'area',round(a['area'],3),'level',round(a['level'],3))
    print('  Tm_field', r['Tm_field'])
    for ha,kv in r['kirchhoff'].items(): print('  kirch',ha,'dC',round(kv['dC'],2) if kv['dC']==kv['dC'] else None,'dH55',round(kv['dH_at_055'],3) if kv['dH_at_055']==kv['dH_at_055'] else None,'dH_T',[None if x is None else round(x,2) for x in kv['dH_T']])
    print('  ci_dC', {k:[round(x,2) for x in v] for k,v in r['ci_dC'].items()})
    a=r['arms']; print('  phi h0 ch cool',[round(x,2) for x in a['ch|h0|slow']['phi_cool']]); print('  phi h0 ch heat',[round(x,2) for x in a['ch|h0|slow']['phi_heat']])
    print('  phi h0 hc cool',[round(x,2) for x in a['hc|h0|slow']['phi_cool']]); print('  phi h0 hc heat',[round(x,2) for x in a['hc|h0|slow']['phi_heat']])
    print('  Ubar h0 cool',[round(x,3) for x in r['U_bar_h0'][0]]); print('  Ubar h0 heat',[round(x,3) for x in r['U_bar_h0'][1]])
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer)): o=o.item()
    if isinstance(o,float) and not np.isfinite(o): return None
    return o
json.dump(clean(out),open('results/A2_explore/dsc_summary.json','w'),ensure_ascii=False,indent=1); print('done')
