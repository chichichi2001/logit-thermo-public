"""A2-23 の実行（CPU）。
  自己テスト：python3 scripts/a2_23_run.py --self-test
  (A) の予言：python3 scripts/a2_23_run.py --predict-mult <G-pre のフォルダ> <第 1 段のフォルダ> results/A2_23/pred_mult.json
  (D) の予言：python3 scripts/a2_23_run.py --predict-it <第 1 段のフォルダ> results/A2_23/pred_it.json
  分析：python3 scripts/a2_23_run.py <A2-19 のフォルダ> <第 1 段のフォルダ> <第 2 段のフォルダ> results/A2_23/a2_23_summary.json
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_22_run.py').read().split("\nif __name__")[0])

SHA23 = ('scripts/a2_23_analyze.py', 'scripts/a2_23_run.py', 'scripts/a2_23_collect.py') + SHA22


def load23ns():
    ns = load22ns()
    exec(open('scripts/a2_23_analyze.py').read(), ns)
    return ns


def load_groups23(folder, m, arm, grid, h, U=False):
    out = {k: [] for k in ('toks', 'pick', 'q') + (('U',) if U else ())}
    for g in range(3):
        Z = np.load(f'{folder}/{m}@{arm}_g{g}.npz')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in grid], (m, arm, g)
        assert abs(float(Z['h']) - h) < 1e-6 and int(Z['group']) == g, (m, arm, g)
        assert Z['toks'].shape[0] == 7 and Z['toks'].shape[3] == 512 and not (Z['pick'] & ~Z['cand']).any()
        for k in out:
            out[k].append(Z[k])
    return {k: np.concatenate(v, 2) for k, v in out.items()}


if __name__ == '__main__':
    ns = load23ns()
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA23 if os.path.exists(f)}
    if sys.argv[1] == '--self-test':
        print(ns['self_test23'](), ns['self_test22']()); sys.exit()
    if sys.argv[1] == '--predict-mult':
        gp, s1, out = sys.argv[2:5]
        assert not os.path.exists(out), ('予言の出力はもうあります（上書きしない）', out)
        H18 = json.load(open('results/A2_18/hstar.json'))
        res = {'models': {m: ns['predict_mult'](np.load(f'{gp}/{m}@h0.npz'), np.load(f'{s1}/{m}@rs_gpre.npz'), H18[m], m) for m in ns['G16_MODELS']}, 'sha256': sha}
        json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
        for m, r in res['models'].items():
            print(m, 'ok' if r['ok'] else r['why'], {k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in ('gbar', 'lam', 'theta', 'max_dU')})
        sys.exit()
    if sys.argv[1] == '--predict-it':
        s1, out = sys.argv[2:4]
        assert not os.path.exists(out), ('予言の出力はもうあります（上書きしない）', out)
        names = ['qwen2.5-1.5b-it', 'smollm2-1.7b-it']
        res = {'models': {n: ns['predict_model20'](f'{s1}/{n}@s1.npz', n) for n in names}, 'sha256': sha}
        json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
        for n, r in res['models'].items():
            print(n, 'ok' if r['ok'] else r['why'], {k: r.get(k) for k in ('T_all', 'window', 'Tm0', 'gbar')}, r.get('target'))
        sys.exit()
    f19, s1, s2, out = sys.argv[1:5]
    PA = json.load(open('results/A2_23/pred_mult.json'))['models']
    PD = json.load(open('results/A2_23/pred_it.json'))['models']
    for f in ('results/A2_23/pred_mult.json', 'results/A2_23/pred_it.json'):
        sha[f] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
    # (A)
    EA = {}
    for m, r in PA.items():
        if not r.get('ok'):
            continue
        Z0 = load_groups23(f19, m, 'c0', r['grid'], 0.0)
        Z1 = {k: [] for k in ('toks', 'pick', 'q')}
        for g in range(3):
            Z = np.load(f'{s2}/{m}@m1_g{g}.npz')
            assert abs(float(Z['strength']) - round(r['theta'], 4)) < 1e-9 and str(Z['kind']) == 'gen_mult' and int(Z['group']) == g
            assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in r['grid']]
            for k in Z1:
                Z1[k].append(Z[k])
        Z1 = {k: np.concatenate(v, 2) for k, v in Z1.items()}
        As = []
        for g in range(3):
            Zr = np.load(f'{s1}/{m}@rs_c0_g{g}.npz')
            assert [round(float(t), 4) for t in Zr['temps']] == [round(float(t), 4) for t in r['grid']] and int(Zr['group']) == g and str(Zr['kind']) == 'rs'
            As.append(Zr['A_sum'])
        As = np.concatenate(As, 2).astype(float)
        EA[m] = (ns['seq_events'](Z0['toks'], Z0['pick'], Z0['q']), ns['seq_events'](Z1['toks'], Z1['pick'], Z1['q']), As, float(np.log(round(r['theta'], 4))), r['grid'])
        print('読み込み (A)', m, flush=True)
    miss_A = [m for m in PA if not PA[m].get('ok')]
    if len(miss_A) >= 4:                                                    # ゲート §2：4 本以上予言できなければ XQ12 は「予言できない」
        A = dict(word='予言できない', point=float('nan'), ci=[float('nan')] * 2, fail=1.0, models={})
    else:
        A = ns['boot_agg'](lambda m, w, use: ns['rho_mult'](EA[m][0], EA[m][1], EA[m][2], EA[m][3], EA[m][4], w, use), list(EA), 120, 'median')
    # (D)
    ED = {}
    for n, r in PD.items():
        if not r.get('ok'):
            continue
        h, grid = round(float(r['target']['hstar']), 4), r['target']['grid']
        Z0 = load_groups23(s2, n, 'c0', grid, 0.0); Z1 = load_groups23(s2, n, 'h1', grid, h)
        ED[n] = ({'c0': ns['seq_events'](Z0['toks'], Z0['pick'], Z0['q']), 'h1': ns['seq_events'](Z1['toks'], Z1['pick'], Z1['q'])}, h, grid)
        print('読み込み (D)', n, flush=True)
    if not ED:                                                              # ゲート §2：2 本とも予言できなければ XQ13 は「予言できない」
        D = dict(word='予言できない', point=float('nan'), ci=[float('nan')] * 2, fail=1.0, models={})
    else:
        D = ns['boot_agg'](lambda n, w, use: ns['rho_model'](ED[n][0], ('h1', ED[n][1]), ED[n][2], w, use), list(ED), 120, 'mean')
    json.dump(clean(dict(A=A, D=D, missing_A=[m for m in PA if not PA[m].get('ok')], missing_D=[n for n in PD if not PD[n].get('ok')], sha256=sha)),
              open(out, 'w'), ensure_ascii=False, indent=1)
    for nm, g in (('XQ12（割る型の罰）', A), ('XQ13（指示調整版）', D)):
        if g:
            print(nm, g['word'], g['point'], g['ci'], 'fail', g['fail'])
            for m, p in g['models'].items():
                print('  ', m, round(p['rho'], 3), [round(x, 2) for x in p['ci']], p['word'])
