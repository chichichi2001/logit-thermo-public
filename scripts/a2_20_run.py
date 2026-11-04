"""A2-20 の実行（CPU）。
  予言：python3 scripts/a2_20_run.py --predict <第 1 段の npz のフォルダ> results/A2_20/hstar7b.json
  予備の予言（本命が予言できないときだけ）：python3 scripts/a2_20_run.py --predict-backup <フォルダ> results/A2_20/hstar7b_backup.json
  分析：python3 scripts/a2_20_run.py <第 2 段のフォルダ> <A2-19 のフォルダ> results/A2_20/a2_20_summary.json
  自己テスト：python3 scripts/a2_20_run.py --self-test [G-pre のフォルダ]
"""
import sys, os, json, hashlib, ast
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_18_run.py').read().split("\nif __name__")[0])


def load20ns():
    ns = load18ns()
    for f in ('scripts/a2_19_analyze.py', 'scripts/a2_20_predict.py', 'scripts/a2_20_analyze.py'):
        exec(open(f).read(), ns)
    tree = ast.parse(open('scripts/a2_20_collect.py').read())
    for k in ('A20_HF', 'A20_BACKUP'):
        ns[k] = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == k)
    return ns


SHA_FILES = ('scripts/a2_20_predict.py', 'scripts/a2_20_analyze.py', 'scripts/a2_20_collect.py', 'scripts/a2_19_analyze.py', 'scripts/a2_18_analyze.py',
             'scripts/a2_18_predict.py', 'scripts/a2_16_analyze.py')

if __name__ == '__main__':
    ns = load20ns()
    shas = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA_FILES}
    print({k: v[:16] for k, v in shas.items()})
    if sys.argv[1] == '--self-test':
        print(ns['self_test_p20'](sys.argv[2] if len(sys.argv) > 2 else None)); print(ns['self_test20']()); print(ns['self_test19']()); sys.exit()
    if sys.argv[1] in ('--predict', '--predict-backup'):
        base, out = sys.argv[2:4]
        names = ns['A20_HF'] if sys.argv[1] == '--predict' else ns['A20_BACKUP']
        assert not os.path.exists(out), ('予言の出力はもうあります（上書きしない）', out)
        res = {'models': {n: ns['predict_model20'](f'{base}/{n}@s1.npz', n) for n in names}, 'sha256': shas}
        json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
        for n, r in res['models'].items():
            print(n, 'ok' if r['ok'] else r['why'], {k: r.get(k) for k in ('T_all', 'window', 'Tm0', 'gbar')}, r.get('target'))
        sys.exit()
    b20, b19, out = sys.argv[1:4]
    tree = ast.parse(open('scripts/a2_18_collect.py').read())
    jobs19 = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'A18_JOBS')
    hp = ['results/A2_20/hstar7b.json', 'results/A2_20/hstar7b_backup.json']
    for f in hp:
        if os.path.exists(f):
            shas[f] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    res, *_ = ns['analyze20'](b20, hp, b19, jobs19)
    res['sha256'] = shas
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(clean({k: res[k] for k in ('XQ7', 'missing') if k in res}), ensure_ascii=False, indent=1))
    print('9 本（記録）：', res.get('record_9models'))
