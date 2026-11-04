"""XQ7-模型の見込みを出す（第 1 段のあと、第 2 段の前）：python3 scripts/a2_20_toy_run.py <第 1 段のフォルダ> results/A2_20/toy7b.json"""
import sys, os, json, hashlib, numpy as np
S1, OUT = sys.argv[1:3]; sys.argv = sys.argv[:1]
exec(open('scripts/a2_20_run.py').read().split("\nif __name__")[0])
ns = load20ns(); exec(open('scripts/a2_explore_toy.py').read(), ns); exec(open('scripts/a2_20_toy.py').read(), ns)
assert not os.path.exists(OUT), ('もうあります（上書きしない）', OUT)
P = json.load(open('results/A2_20/hstar7b.json'))['models']
res = {}
for n, pred in P.items():
    if not pred.get('ok'):
        continue
    r = ns['toy_for_model'](f'{S1}/{n}@s1.npz', pred, expect_temps=ns['A20_T1'])
    res[n] = r
    print(n, 'ρ₁_模型', round(r['rho1_toy'], 4), 'cost', round(r['cost'], 2), {k: round(v, 3) for k, v in r['targets'].items()}, 'm', r['params']['m'], flush=True)
out = dict(models=res, mean_rho1_toy=float(np.mean([r['rho1_toy'] for r in res.values()])),
           sha256={f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in ('scripts/a2_20_toy.py', 'scripts/a2_explore_toy.py', 'results/A2_20/hstar7b.json')})
json.dump(out, open(OUT, 'w'), indent=1, ensure_ascii=False)
print('2 本の平均 ρ₁_模型', round(out['mean_rho1_toy'], 4))
