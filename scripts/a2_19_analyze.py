"""A2-19 の分析：A2-18 の XQ1 を、新しい記事 120 本でもう一度だけ（2026-11-01 20:xx JST、データを集める前に書いた。CPU）。
A2-18 の分析（scripts/a2_18_analyze.py `5b55c3df…`、凍結）の関数をそのまま使う（seq_events・rho_model・word18・_median_or_nan・analyze_all）。
決め方は A2-18 の XQ1 と同じ：7 モデルの ρ の中央値の区間が 0.8〜1.25 に収まれば「予言どおり（2 割の内）」、など（word18）。
違うのは、記事が新しい 120 本（3 つの組をつないだもの）であることと、2h* の腕がないことだけ。
区間：記事の選び直し 1,000 回（種 20261192。7 モデルと 2 つの腕に同じ選び方）。
記録：A2-18 の 40 本とあわせた 160 本での中央値と区間（判定には使わない）。
"""
import os, json
import numpy as np

A19_SEED_BOOT = 20261192


def load_groups(base, name, arm, h, temps, ngroups=3):
    """3 つの組の npz を記事の向き（3 番目の軸）につなぐ → seq_events の材料。"""
    toks, pick, q = [], [], []
    for g in range(ngroups):
        Z = np.load(f'{base}/{name}@{arm}_g{g}.npz')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in temps], (name, arm, g)
        assert abs(float(Z['h']) - h) < 1e-9 and int(Z['group']) == g
        assert Z['toks'].shape[0] == 7 and Z['toks'].shape[3] == 512 and not (Z['pick'] & ~Z['cand']).any()
        if abs(h) < 1e-12:
            assert float(np.abs(Z['q'].astype(float) - Z['q0'].astype(float)).max()) == 0.0
        toks.append(Z['toks']); pick.append(Z['pick']); q.append(Z['q'])
    return np.concatenate(toks, 2), np.concatenate(pick, 2), np.concatenate(q, 2)


def analyze19(base19, jobs, seed=A19_SEED_BOOT, boot=1000, base18=None):
    J = {(n, a): (h, t) for n, a, h, t in jobs}
    names = list(dict.fromkeys(n for n, _, _, _ in jobs))
    EE, hs, grids, nD, missing = {}, {}, {}, None, []
    for n in names:
        if not all(os.path.exists(f'{base19}/{n}@{a}_g{g}.npz') for a in ('c0', 'h1') for g in range(3)):
            missing.append(n); continue
        E = {}
        for a in ('c0', 'h1'):
            h, t = J[(n, a)]
            tk, pk, q = load_groups(base19, n, a, h, t)
            E[a] = seq_events(tk, pk, q)
        nd = E['c0'][0].shape[2]
        assert E['h1'][0].shape[2] == nd
        nD = nd if nD is None else nD
        assert nd == nD, ('記事の数がモデルで違う', n, nd, nD)
        E['h2'] = E['h1']                                                   # analyze_all は 2 つの腕を回すので、h1 を写しておく（h2 の結果は使わない）
        EE[n] = E; hs[n] = {'h1': J[(n, 'h1')][0], 'h2': J[(n, 'h1')][0]}; grids[n] = J[(n, 'c0')][1]
    res = {'missing': missing}
    if len(EE) >= 5:
        r = analyze_all(EE, hs, grids, nD, seed, boot)
        res['XQ1'] = r['verdict']['h1']
        res['models'] = {n: r['models'][n]['h1'] for n in EE}
    else:
        res['XQ1'] = dict(word='はっきりしない', note='そろったモデルが 4 つ以下')
    res['nD'] = nD
    return res, EE, hs, grids


def pooled_record(EE19, base18, jobs, hs, grids, seed=A19_SEED_BOOT, boot=1000):
    """記録：A2-18 の 40 本（c0・h1）と A2-19 の 120 本をつないだ 160 本での中央値と区間（判定には使わない）。"""
    J = {(n, a): (h, t) for n, a, h, t in jobs}
    EE = {}
    for n, E in EE19.items():
        E18 = {}
        for a in ('c0', 'h1'):
            h, t = J[(n, a)]
            Z = np.load(f'{base18}/{n}@{a}.npz')
            E18[a] = seq_events(Z['toks'], Z['pick'], Z['q'])
        EE[n] = {a: (np.concatenate([E18[a][0], E[a][0]], 2), np.concatenate([E18[a][1], E[a][1]], 2)) for a in ('c0', 'h1')}
        EE[n]['h2'] = EE[n]['h1']
    nD = next(iter(EE.values()))['c0'][0].shape[2]
    r = analyze_all(EE, hs, grids, nD, seed, boot)
    return dict(median=r['verdict']['h1']['median'], ci=r['verdict']['h1']['ci'], word=r['verdict']['h1']['word'], nD=nD)


def self_test19(boot=40):
    """作り物の 3 つの組の npz（形だけ本物と同じ、中身は G-pre 風の乱数）で、読み込み → つなぐ → 判定まで動くこと。数の正しさは self_test18 で見ている。"""
    import tempfile
    rng = np.random.default_rng(3)
    tmp = tempfile.mkdtemp()
    temps = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    jobs = []
    for m in range(5):
        for a, h in (('c0', 0.0), ('h1', -0.4)):
            jobs.append((f'm{m}', a, h, temps))
            for g in range(3):
                pk = rng.random((7, 2, 4, 512)) < 0.5
                np.savez_compressed(f'{tmp}/m{m}@{a}_g{g}.npz', temps=np.array(temps), h=h, arm=a, group=g, use=np.arange(4),
                                    toks=rng.integers(0, 50, (7, 2, 4, 512)).astype(np.int32), cand=pk, pick=pk,
                                    q=np.zeros((7, 2, 4, 512), np.float32), q0=np.zeros((7, 2, 4, 512), np.float32))
    res, EE, _, _ = analyze19(tmp, jobs, boot=boot)
    assert res['nD'] == 12 and set(EE) == {f'm{m}' for m in range(5)} and res['missing'] == []
    assert 'word' in res['XQ1']
    os.remove(f'{tmp}/m4@h1_g2.npz')
    res2, _, _, _ = analyze19(tmp, jobs, boot=boot)
    assert res2['missing'] == ['m4'] and res2['XQ1']['word'] == 'はっきりしない'
    return 'OK'
