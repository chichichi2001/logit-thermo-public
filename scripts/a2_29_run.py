"""A2-29 の実行（CPU）。ゲート：docs/A2_29_GATE.md。
  自己テスト：python3 scripts/a2_29_run.py --self-test
  前提だけ：  python3 scripts/a2_29_run.py --dry <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ>
  分析：      python3 scripts/a2_29_run.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> results/A2_29/a2_29_summary.json
標準出力には自己テストと前提の確かめだけを出す（点や区間は出さない）。
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_22_run.py').read().split("\nif __name__")[0])

SHA29 = ('scripts/a2_29_analyze.py', 'scripts/a2_29_run.py', 'docs/A2_29_GATE.md') + SHA22


def load29ns():
    ns = load22ns()
    exec(open('scripts/a2_29_analyze.py').read(), ns)
    return ns


def clean29(o):
    if isinstance(o, dict):
        return {str(k): clean29(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean29(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == '__main__':
    ns = load29ns()
    print(ns['self_test29'](), flush=True)
    if sys.argv[1] == '--self-test':
        sys.exit()
    dry = sys.argv[1] == '--dry'
    args = sys.argv[2:] if dry else sys.argv[1:]
    f19, f20 = args[0], args[1]
    if not dry:
        out = args[2]
        assert not os.path.exists(out), ('出力はもうあります（上書きしない）', out)
    H18 = json.load(open('results/A2_18/hstar.json')); H20 = json.load(open('results/A2_20/hstar7b.json'))['models']
    S19 = json.load(open('results/A2_19/a2_19_summary.json'))['models']; S20 = json.load(open('results/A2_20/a2_20_summary.json'))['models']
    S22 = json.load(open('results/A2_22/a2_22_summary.json')); T22 = {**S22['small']['models'], **S22['big']['models']}
    small, big = {}, {}
    for m in ns['G16_MODELS']:
        t = H18[m]['targets']['-0.10']
        small[m] = dict(c0=ns['prep22'](load_arm(f19, m, 'c0', t['grid'], 0.0)), h1=ns['prep22'](load_arm(f19, m, 'h1', t['grid'], S19[m]['h'])), h=S19[m]['h'])
    for m in ns['A21_BIG']:
        t = H20[m]['target']
        big[m] = dict(c0=ns['prep22'](load_arm(f20, m, 'c0', t['grid'], 0.0)), h1=ns['prep22'](load_arm(f20, m, 'h1', t['grid'], S20[m]['h'])), h=S20[m]['h'])
    # 前提の確かめ（ゲート §2）
    for g in range(3):                                                    # 記事の並びが 9 本・2 つの腕で同じ（npz の use ＝ 記事の番号）
        ref = None
        for f, ms in ((f19, ns['G16_MODELS']), (f20, ns['A21_BIG'])):
            for m in ms:
                for arm in ('c0', 'h1'):
                    u = np.load(f'{f}/{m}@{arm}_g{g}.npz')['use']
                    ref = u if ref is None else ref
                    assert np.array_equal(u, ref), ('記事の並びが違う', m, arm, g)
    for S, grp in ((S19, small), (S20, big)):
        for m, d in grp.items():
            one = np.ones(d['c0']['ev'].shape[2])
            pt = ns['per_temp'](d['c0'], d['h1'], d['h'], one)
            u = ns['select'](pt['cnt'], 'A')
            assert u == S[m]['use'], (m, '使う温度が記録と違う', u, S[m]['use'])
            assert np.allclose(pt['D'][u], np.array(S[m]['D'])[u], atol=1e-9, rtol=0), (m, 'D')
            assert np.allclose(pt['P'][u], np.array(S[m]['P'])[u], atol=1e-9, rtol=0), (m, 'P')
            assert abs(ns['rho_of'](pt, u) - S[m]['rho']) < 1e-9, (m, 'rho')
            fm, fi = ns['f_pair'](d['c0'], pt, u, one)
            assert abs(fm - T22[m]['f_entry']) < 1e-9 and abs(fi - T22[m]['f_entry_identity']) < 1e-9, (m, 'f')
    print('前提 OK（D・P・ρ・使う温度・f_meas・f_id が記録と 1e-9 で一致）', flush=True)
    if dry:
        sys.exit()
    sha = {f: hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in SHA29}
    res = ns['analyze29'](small, big)
    res['sha256'] = sha
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(clean29(res), open(out, 'w'), indent=1, ensure_ascii=False)
    print('書き出し', out, '（点・区間・言葉はファイルの中。ここには出さない）', flush=True)
