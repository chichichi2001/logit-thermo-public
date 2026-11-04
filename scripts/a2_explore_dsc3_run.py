"""探索 8.39 の実行：固定した分析（scripts/a2_explore_dsc3_analyze.py、sha e66a736a…）を DSC その 3 のデータに当てる。"""
import sys, json, hashlib, numpy as np, warnings, time; warnings.filterwarnings('ignore'); sys.path.insert(0,'<scratch>')
from load import load13ns
ns=load13ns(); exec(open('scripts/a2_explore_dsc3_analyze.py').read(), ns)
B='<scratch>'
out={'sha':hashlib.sha256(open('scripts/a2_explore_dsc3_analyze.py','rb').read()).hexdigest()}
assert out['sha'].startswith('e66a736a')
out['calib']=ns['self_test_dsc3_an']()
for m,sub in [('pythia-1b','pythia'),('olmo2-1b','olmo')]:
    t=time.time(); out[m]=ns['analyze_dsc3'](f'{B}/{sub}',m); print(m, round(time.time()-t),'s')
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer,np.bool_)): o=o.item()
    if isinstance(o,float) and not np.isfinite(o): return None
    return o
json.dump(clean(out),open('results/A2_explore/dsc3_summary.json','w'),ensure_ascii=False,indent=1); print('done')
