"""A2-16 の実行：固定した分析（scripts/a2_16_analyze.py）を G-pre のデータに当てる。名前空間の組み立ては、探索で使ってきたものと同じ。
使い方：python3 scripts/a2_16_run.py <G-pre の npz のフォルダ> <出力の json>"""
import ast, sys, json, hashlib, warnings
import numpy as np
warnings.filterwarnings('ignore')


def load16ns():
    ns = {'np': np}
    exec(open('scripts/a2_explore_onset.py').read(), ns)
    src = open('scripts/a2_8_vh.py').read(); tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(n2.id.startswith('A8_') for t in n.targets for n2 in ast.walk(t) if isinstance(n2, ast.Name)))
            or (isinstance(n, ast.FunctionDef) and n.name in ('phases', 'doc_counts', 'fit_vh'))]
    exec(compile(ast.Module(keep, []), 'a2_8', 'exec'), ns)
    for f in ['scripts/a2_explore_three.py', 'scripts/a2_explore_seg.py', 'scripts/a2_13_analyze.py', 'scripts/a2_explore_ladder.py', 'scripts/a2_16_analyze.py']:
        exec(open(f).read(), ns)
    return ns


def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)): o = o.item()
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


if __name__ == '__main__':
    base, out = sys.argv[1], sys.argv[2]
    ns = load16ns()
    shas = {f: hashlib.sha256(open(f'scripts/{f}', 'rb').read()).hexdigest() for f in
            ('a2_16_analyze.py', 'a2_explore_onset.py', 'a2_8_vh.py', 'a2_13_analyze.py', 'a2_explore_ladder.py')}
    print({k: v[:16] for k, v in shas.items()})
    st = ns['self_test16'](); print('self_test16 OK')
    res = ns['analyze16'](base)
    res['sha256'] = shas; res['self_test'] = st
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(clean(res['verdict']), ensure_ascii=False, indent=1))
