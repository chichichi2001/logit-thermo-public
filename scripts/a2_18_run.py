"""A2-18 の実行：A2-16 と同じ名前空間に a2_18_predict.py と a2_18_analyze.py を足して当てる。
使い方：python3 scripts/a2_18_run.py <A2_18 の npz のフォルダ> <G-pre の npz のフォルダ（記録の比べの相手だけに使う）> <出力の json>
        python3 scripts/a2_18_run.py --self-test"""
import sys, json, hashlib, warnings, ast
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
from a2_16_run import load16ns, clean


def load18ns():
    ns = load16ns()
    for f in ('scripts/a2_18_predict.py', 'scripts/a2_18_analyze.py'):
        exec(open(f).read(), ns)
    # 生成の仕事の表（h と温度）は、集めのコードから書き写さずに読む（torch を読み込まないように、式だけ取り出す）
    tree = ast.parse(open('scripts/a2_18_collect.py').read())
    ns['A18_JOBS'] = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'A18_JOBS')
    return ns


if __name__ == '__main__':
    ns = load18ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test_p18']()); print(json.dumps(clean(ns['self_test18']()), ensure_ascii=False, indent=1)); sys.exit()
    b18, bg, out = sys.argv[1:4]
    shas = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in
            ('scripts/a2_18_analyze.py', 'scripts/a2_18_predict.py', 'scripts/a2_18_collect.py', 'scripts/a2_16_analyze.py', 'results/A2_18/hstar.json')}
    print({k: v[:16] for k, v in shas.items()})
    st = ns['self_test18']()
    res = ns['analyze18'](b18, bg); res['sha256'] = shas; res['self_test'] = st
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(clean(res['verdict']), ensure_ascii=False, indent=1))
