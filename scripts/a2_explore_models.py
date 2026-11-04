"""探索 8.42：罰の強さによる比 ρ₁ の変わり方を、いくつかの模型で説明できるか（2026-11-02 10:xx JST、CPU、判定しない。もう見たデータだけ）。
実行：python3 scripts/a2_explore_models.py <G-pre のフォルダ> <出力 json>
使うデータ（小さい 7 本、どれももう見たもの）：
  G-pre の h ＝ −0.3・＋0.3（温度 0.4〜0.7、記事 40 本）→ ここで ρ₁ と区間を出す（対照は h ＝ 0 の腕、A2-18 の rho_model と同じ数え）。
  A2-18 の h*・2h*（40 本）、A2-19 の h*（120 本）→ それぞれの結果の json から ρ と区間。
模型：
  ベル模型：凍る・溶ける速さが h について指数関数でまっすぐ → ln K は h の 1 次 → ρ₁ ＝ 1（どの h でも）。
  はしご模型（8.36）：G-pre の h ＝ 0 に合わせたひな形（results/A2_explore/toy_small7.json）で、それぞれの腕の h と温度で ρ₁ を厳密に。
比べ方：腕ごとの ρ₁ の測った値と模型の値の差を、区間の幅から出したばらつき（95% 区間の幅 ÷ 3.92）で割った 2 乗の和（χ²）。小さいほうが合う。
"""
import sys, json, ast, numpy as np
GP, OUT = sys.argv[1:3]; sys.argv = sys.argv[:1]
exec(open('scripts/a2_20_run.py').read().split("\nif __name__")[0])
ns = load20ns(); exec(open('scripts/a2_explore_toy.py').read(), ns); exec(open('scripts/a2_20_toy.py').read(), ns)
tree = ast.parse(open('scripts/a2_18_collect.py').read())
J = {(n, a): (h, t) for n, a, h, t in next(ast.literal_eval(x.value) for x in tree.body if isinstance(x, ast.Assign) and getattr(x.targets[0], 'id', '') == 'A18_JOBS')}
names = ['pythia-410m', 'pythia-1.4b', 'gpt-neo-1.3b', 'smollm2-1.7b', 'qwen2.5-1.5b', 'gpt2-1.5b', 'bloom-1b1']
TOY = json.load(open('results/A2_explore/toy_small7.json'))
S18 = json.load(open('results/A2_18/a2_18_summary.json'))['models']; S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']
G16 = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
rng = np.random.default_rng(20261142)
rows = []
for n in names:
    Z = {a: np.load(f'{GP}/{n}@{a}.npz') for a in ('h0', 'hm03', 'hp03')}
    E = {a: ns['seq_events'](Z[a]['toks'], Z[a]['pick'], Z[a]['q']) for a in Z}
    nD = E['h0'][0].shape[2]
    for a, h in (('hm03', -0.3), ('hp03', 0.3)):
        EE = {'c0': E['h0'], a: E[a]}
        r, use, _, _ = ns['rho_model'](EE, (a, h), G16, np.ones(nD))
        bs = [ns['rho_model'](EE, (a, h), G16, np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float), use)[0] for _ in range(300)]
        ci = list(np.nanpercentile(bs, [2.5, 97.5]))
        rows.append(dict(model=n, arm=f'G-pre {a}', h=h, temps=G16, rho=r, ci=ci))
    for a in ('h1', 'h2'):
        rows.append(dict(model=n, arm=f'A2-18 {a}', h=J[(n, a)][0], temps=J[(n, a)][1], rho=S18[n][a]['rho'], ci=S18[n][a]['ci']))
    rows.append(dict(model=n, arm='A2-19 h1', h=J[(n, 'h1')][0], temps=J[(n, 'h1')][1], rho=S19[n]['rho'], ci=S19[n]['ci']))
for r in rows:
    r['ladder'] = ns['toy_rho1'](TOY[r['model']]['params'], r['h'], r['temps'])[0]
    r['bell'] = 1.0
    r['sd'] = (r['ci'][1] - r['ci'][0]) / 3.92
    print(f"{r['model']:14s} {r['arm']:12s} h={r['h']:+.3f} 測った {r['rho']:.3f} [{r['ci'][0]:.2f},{r['ci'][1]:.2f}]  はしご {r['ladder']:.3f}  ベル 1.000", flush=True)
chi = {m: float(sum(((r['rho'] - r[m]) / r['sd']) ** 2 for r in rows)) for m in ('ladder', 'bell')}
arms = sorted({r['arm'] for r in rows})
summ = {a: dict(median_measured=float(np.median([r['rho'] for r in rows if r['arm'] == a])), median_ladder=float(np.median([r['ladder'] for r in rows if r['arm'] == a])),
                median_h=float(np.median([r['h'] for r in rows if r['arm'] == a]))) for a in arms}
print('χ²（35 の腕）：', {k: round(v, 1) for k, v in chi.items()})
for a, v in summ.items():
    print(a, {k: round(x, 3) for k, x in v.items()})
json.dump(dict(rows=rows, chi2=chi, by_arm=summ), open(OUT, 'w'), indent=1, ensure_ascii=False)
