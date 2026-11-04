"""探索 8.40 の実行：固定した分析（scripts/a2_explore_pork_analyze.py、sha 1cef4beb…）を PROJECT_G の下見のデータに当てる。
名前空間は A2-16 と同じ（copy_marks・phases）。使い方：python3 scripts/a2_explore_pork_run.py <npz のフォルダ> <出力の json>"""
import sys, json, hashlib, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
from a2_16_run import load16ns, clean

if __name__ == '__main__':
    base, out = sys.argv[1], sys.argv[2]
    sha = hashlib.sha256(open('scripts/a2_explore_pork_analyze.py', 'rb').read()).hexdigest()
    assert sha.startswith('1cef4beb'), sha
    ns = load16ns()
    exec(open('scripts/a2_explore_pork_analyze.py').read(), ns)
    t = time.time(); st = ns['self_test_pork_an'](); print('self_test_pork_an OK', round(time.time() - t), 's')
    t = time.time(); res = ns['analyze_pork'](base); print('analyze', round(time.time() - t), 's')
    res['sha256'] = sha; res['self_test'] = st
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(clean(res['verdict']), ensure_ascii=False, indent=1))
