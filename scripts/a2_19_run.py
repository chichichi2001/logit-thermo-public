"""A2-19 の実行：A2-18 と同じ名前空間（a2_18_run.load18ns）に scripts/a2_19_analyze.py を足して当てる。
使い方：python3 scripts/a2_19_run.py <A2_19 の npz のフォルダ> <A2_18 の npz のフォルダ（記録の 160 本だけに使う）> <出力の json>
        python3 scripts/a2_19_run.py --self-test"""
import sys, json, hashlib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
from a2_18_run import load18ns
from a2_16_run import clean

if __name__ == '__main__':
    ns = load18ns()
    exec(open('scripts/a2_19_analyze.py').read(), ns)
    if sys.argv[1] == '--self-test':
        print(ns['self_test18']()['rho1.0']['h1']['word'], ns['self_test19']()); sys.exit()
    b19, b18, out = sys.argv[1:4]
    shas = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in
            ('scripts/a2_19_analyze.py', 'scripts/a2_19_collect.py', 'scripts/a2_18_analyze.py', 'scripts/a2_18_collect.py', 'results/A2_18/hstar.json')}
    print({k: v[:16] for k, v in shas.items()})
    st = ns['self_test18']()
    res, EE, hs, grids = ns['analyze19'](b19, ns['A18_JOBS'])
    res['sha256'] = shas
    print(json.dumps(clean(res['XQ1']), ensure_ascii=False, indent=1))
    if EE and b18 != '-':
        res['record_pooled_160'] = ns['pooled_record'](EE, b18, ns['A18_JOBS'], hs, grids)
        print('160 本（記録）：', clean(res['record_pooled_160']))
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
