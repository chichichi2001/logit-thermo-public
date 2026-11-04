"""探索 8.35 の実行：固定した分析（scripts/a2_explore_fq_analyze.py）を q のデータに当てる。"""
import sys, json, hashlib, numpy as np, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'<scratch>')
from load import load13ns
ns=load13ns(); exec(open('scripts/a2_explore_fq_analyze.py').read(), ns); print(ns['self_test_fq_an']())
B='<scratch>'
out={'sha':hashlib.sha256(open('scripts/a2_explore_fq_analyze.py','rb').read()).hexdigest()}
for m in ['pythia-1b','olmo2-1b']:
    t=time.time(); r=ns['analyze_fq'](B+'/fq', B+'/field', m); out[m]=r
    print(m, round(time.time()-t), 'Tm_h0', round(r['Tm_h0'],4))
    print(' pred_dM', [round(x,1) for x in r['pred_dM']]); print(' meas_dM', [round(x,1) for x in r['meas_dM']]); print(' ratio', [round(x,3) for x in r['ratio_dM']], 'ci lo', [round(x,2) for x in r['ci']['ratio_dM'][0]], 'hi', [round(x,2) for x in r['ci']['ratio_dM'][1]])
    print(' R_pred', [round(x,4) for x in r['R_pred']])
    print(' dlnkf', [round(x,2) for x in r['dlnkf']], 'dlnkm', [round(x,2) for x in r['dlnkm']], 'dM_kin_pred', [round(x,2) for x in r['dM_kin_pred']])
    print(' slope_rw', round(r['slope_rw'],4), r['ci']['slope_rw'], 'rw03', round(r['slope_rw03'],4), r['ci']['slope_rw03'], 'Tm_rw', {k:round(v,4) for k,v in r['Tm_rw'].items()})
    print(' ess', {k:[round(x,3) for x in v] for k,v in r['ess'].items()})
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer)): o=o.item()
    if isinstance(o,float) and not np.isfinite(o): return None
    return o
json.dump(clean(out),open('results/A2_explore/fq_summary.json','w'),ensure_ascii=False,indent=1); print('done')
