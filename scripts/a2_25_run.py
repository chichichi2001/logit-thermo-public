"""A2-25 の分析（CPU）。ゲート：docs/A2_25_GATE.md。
  自己テスト：python3 scripts/a2_25_run.py --self-test
  分析：python3 scripts/a2_25_run.py <A2-19 のフォルダ> <A2-23 第 1 段のフォルダ> <A2-23 第 2 段のフォルダ> <A2-25 のフォルダ> results/A2_25/a2_25_summary.json
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_23_run.py').read().split("\nif __name__")[0])
exec(open('scripts/a2_25_jobs.py').read())

A25_SEED = 20261125
SHA25 = ('scripts/a2_25_run.py', 'scripts/a2_25_jobs.py') + SHA23


def compact_pred():
    P = json.load(open('results/A2_23/pred_mult.json'))['models']
    return json.dumps({'models': {n: ({'ok': True, 'theta': r['theta'], 'grid': r['grid']} if r['ok'] else {'ok': False}) for n, r in P.items()}},
                      sort_keys=True, separators=(',', ':'))


A25_MAIN_EXCLUDE = ('gpt-neo-1.3b',)                                   # 主の中央値は、どの倍率でも同じ 6 本（GPT-Neo は 2 倍だけなので別に記録）


def word25(res):
    """boot_agg の結果 → 言葉（ゲート §2）。
    ・中央値が点で出ず、出ないモデルのほとんど（全部）が「使える温度が 2 つ未満」のためなら「くり返しが減りすぎて測れない」。
    ・中央値が出ても、出ないモデルが 2 本なら「予言どおり」は使わない（落ちたモデルで結果が偏りうるため）→「はっきりしない」。"""
    miss = [m for m, v in res['models'].items() if not np.isfinite(v['rho'])]
    few = [m for m in miss if len(res['models'][m]['use']) < 2]
    if not np.isfinite(res['point']):
        return 'くり返しが減りすぎて測れない' if miss and len(few) == len(miss) else 'はっきりしない'
    if len(miss) >= 2 and res['word'] == '予言どおり（2 割の内）':
        return 'はっきりしない'
    return res['word']


def self_test25(ns):
    PMC = compact_pred()
    assert hashlib.sha256(PMC.encode()).hexdigest() == 'bc9e1f4897a505ba52c3196db00ce26356ebaf41f37e0e07f36ab6d66fbf13f3', '予言の短い JSON が A2-23 と違う'
    J = a25_jobs(PMC)
    assert len(J) == 57, len(J)
    names = {(j[1], j[2]) for j in J}
    assert ('gpt-neo-1.3b', 'm2') in names and ('gpt-neo-1.3b', 'm4') not in names and ('gpt-neo-1.3b', 'm8') not in names
    assert all(1.0 < j[3] <= 3.0 for j in J)
    assert abs(a25_theta(1.0214336067424572, 8) - 1.1846) < 1e-12
    # m1（A2-23）の θ と一致：k ＝ 1 なら θ* の丸め
    assert a25_theta(1.0214336067424572, 1) == round(1.0214336067424572, 4)
    nan = float('nan')
    mk = lambda rs, us: {f'm{i}': dict(rho=r, use=u) for i, (r, u) in enumerate(zip(rs, us))}
    assert word25(dict(point=nan, word='はっきりしない', models=mk([nan, nan, nan, 1.0], [[0], [], [1], [0, 1]]))) == 'くり返しが減りすぎて測れない'
    assert word25(dict(point=nan, word='はっきりしない', models=mk([nan, nan, nan, 1.0], [[0, 1], [], [1], [0, 1]]))) == 'はっきりしない'
    assert word25(dict(point=1.0, word='予言どおり（2 割の内）', models=mk([1.0, 1.0, nan, nan], [[0, 1]] * 2 + [[0]] * 2))) == 'はっきりしない'
    assert word25(dict(point=1.0, word='予言どおり（2 割の内）', models=mk([1.0, 1.0, nan], [[0, 1]] * 2 + [[0]]))) == '予言どおり（2 割の内）'
    return 'OK'


if __name__ == '__main__':
    ns = load23ns()
    if sys.argv[1] == '--self-test':
        print(ns['self_test23'](), self_test25(ns)); sys.exit()
    f19, s1, s2, f25, out = sys.argv[1:6]
    assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA25 if os.path.exists(f)}
    sha['results/A2_23/pred_mult.json'] = hashlib.sha256(open('results/A2_23/pred_mult.json', 'rb').read()).hexdigest()
    print('自己テスト', self_test25(ns), flush=True)
    PA = json.load(open('results/A2_23/pred_mult.json'))['models']
    J = a25_jobs(compact_pred())
    E0, AS = {}, {}
    for m, r in PA.items():
        if not r.get('ok'):
            continue
        Z0 = load_groups23(f19, m, 'c0', r['grid'], 0.0)
        E0[m] = ns['seq_events'](Z0['toks'], Z0['pick'], Z0['q'])
        As = []
        for g in range(3):
            Zr = np.load(f'{s1}/{m}@rs_c0_g{g}.npz')
            assert [round(float(t), 4) for t in Zr['temps']] == [round(float(t), 4) for t in r['grid']] and int(Zr['group']) == g and str(Zr['kind']) == 'rs'
            As.append(Zr['A_sum'])
        AS[m] = np.concatenate(As, 2).astype(float)
    def arm(folder, m, tag, theta, grid):
        Z1 = {k: [] for k in ('toks', 'pick', 'q')}
        for g in range(3):
            Z = np.load(f'{folder}/{m}@{tag}_g{g}.npz')
            assert abs(float(Z['strength']) - theta) < 1e-9 and str(Z['kind']) == 'gen_mult' and int(Z['group']) == g, (m, tag, g)
            assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in grid]
            for k in Z1:
                Z1[k].append(Z[k])
        Z1 = {k: np.concatenate(v, 2) for k, v in Z1.items()}
        return ns['seq_events'](Z1['toks'], Z1['pick'], Z1['q'])
    # 前提 1：仕事の表（a25_jobs）と、ここで読む（モデル, 倍率）の組が同じ
    want = {(j[1], int(j[2][1:])) for j in J}
    have = {(m, k) for m, r in PA.items() if r.get('ok') for k in A25_K if a25_theta(r['theta'], k) <= A25_THETA_MAX}
    assert want == have, ('仕事の表と分析の組が違う', want ^ have)
    sha['results/A2_23/a2_23_summary.json'] = hashlib.sha256(open('results/A2_23/a2_23_summary.json', 'rb').read()).hexdigest()
    S23 = json.load(open('results/A2_23/a2_23_summary.json'))['A']
    EK, TH = {}, {}
    for m, r in PA.items():
        if not r.get('ok'):
            continue
        for k in (1,) + tuple(A25_K):
            th = round(float(r['theta']), 4) if k == 1 else a25_theta(r['theta'], k)
            if th <= A25_THETA_MAX:
                EK[(m, k)] = arm(s2 if k == 1 else f25, m, f'm{k}', th, r['grid']); TH[(m, k)] = th
        print('読み込み', m, flush=True)
    fn = lambda k, fix=None: (lambda m, w, use: ns['rho_mult'](E0[m], EK[(m, k)], AS[m], float(np.log(TH[(m, k)])), PA[m]['grid'], w, use if fix is None else fix[m]))
    nD = E0[next(iter(E0))][0].shape[2]
    # 前提 2：k ＝ 1、7 本で、A2-23 の記録（点・区間・言葉）を再現する
    all7 = [m for m in PA if PA[m].get('ok')]
    b1 = ns['boot_agg'](fn(1), all7, nD, 'median', seed=ns['A23_SEED_BOOT'])
    assert abs(b1['point'] - S23['point']) < 1e-9 and np.allclose(b1['ci'], S23['ci'], atol=1e-9) and b1['word'] == S23['word'], '前提：A2-23 の判定を再現しない'
    for m in all7:
        assert abs(b1['models'][m]['rho'] - S23['models'][m]['rho']) < 1e-9, m
    print('前提 OK（A2-23 を再現）', flush=True)
    main = [m for m in all7 if m not in A25_MAIN_EXCLUDE]
    res = dict(sha256=sha, main_models=main, by_k={}, gpt_neo={}, fixed_use={})
    for k in (1,) + tuple(A25_K):
        b = ns['boot_agg'](fn(k), main, nD, 'median', seed=A25_SEED)
        b['word25'] = word25(b); b['theta'] = {m: TH[(m, k)] for m in main}
        for m in main:
            _, f1, m1_ = ns['lnK_from'](EK[(m, k)][0], np.ones(nD))
            b['models'][m]['entries_h'] = f1.tolist(); b['models'][m]['exits_h'] = m1_.tolist()
        res['by_k'][str(k)] = b
        print(f'k={k}', b['word25'], b['point'], b['ci'], 'fail', b['fail'], flush=True)
        for m in main:
            r_ = b['models'][m]['rho']
            print('   ', m, TH[(m, k)], round(r_, 3) if np.isfinite(r_) else 'nan', [round(x, 2) for x in b['models'][m]['ci']], 'use', b['models'][m]['use'])
    # 記録：GPT-Neo（k ＝ 1・2）
    for k in (1, 2):
        b = ns['boot_agg'](fn(k), ['gpt-neo-1.3b'], nD, 'median', seed=A25_SEED)
        res['gpt_neo'][str(k)] = dict(theta=TH[('gpt-neo-1.3b', k)], rho=b['models']['gpt-neo-1.3b']['rho'], ci=b['models']['gpt-neo-1.3b']['ci'], use=b['models']['gpt-neo-1.3b']['use'])
    # 記録：どの倍率でも使える温度だけに固定した比（倍率ごとに使う温度が変わる効きを外す）
    fix = {m: sorted(set.intersection(*[set(res['by_k'][str(k)]['models'][m]['use']) for k in (1,) + tuple(A25_K)])) for m in main}
    for k in (1,) + tuple(A25_K):
        b = ns['boot_agg'](fn(k, fix), main, nD, 'median', seed=A25_SEED)
        res['fixed_use'][str(k)] = dict(point=b['point'], ci=b['ci'], use=fix, models={m: b['models'][m]['rho'] for m in main})
        print(f'fixed-use k={k}', b['point'], b['ci'], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    print('保存しました', out)
