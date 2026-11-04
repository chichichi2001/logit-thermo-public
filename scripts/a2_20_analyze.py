"""A2-20 の分析：7B 級の 2 本で、罰をかけていない出力だけからの予言が当たるか（2026-11-02 JST、データを集める前に書いた。CPU）。
A2-18 の分析（scripts/a2_18_analyze.py `5b55c3df…`、凍結）の seq_events・rho_model・word18・analyze_all と、A2-19 の load_groups（`66db0e8b…`）を使う。
決め方（XQ7）：2 本の ρ の平均（＝ 2 本の中央値）の区間が 0.8〜1.25 に収まれば「予言どおり（2 割の内）」。言葉は word18 と同じ。
  区間：記事の選び直し 1,000 回（種 20261202。2 本と 2 つの腕に同じ選び方。120 本はどのモデルでも同じ記事）。
  どちらかのモデルの ρ が出ない回は平均を出さない。出ない回が 1 割を超えたら「はっきりしない」。
  第 2 段がそろったモデルが 2 本でなければ「はっきりしない」。
記録（判定に使わない）：モデルごとの ρ と区間、A2-19 の 7 本と合わせた 9 本の中央値と区間（同じ 120 本で選び直す）。
"""
import os, json
import numpy as np

A20_SEED_BOOT = 20261202


def _events(base, name, h, temps):
    E = {}
    for a, hh in (('c0', 0.0), ('h1', h)):
        tk, pk, q = load_groups(base, name, a, hh, temps)
        E[a] = seq_events(tk, pk, q)
    return E


def models20(paths):
    """a2_20_collect.py の a20_models2 と同じ決まり（予備は、本命が予言できないときだけ枠に入る）。"""
    P = json.load(open(paths[0]))['models']
    use = {n: r for n, r in P.items() if r.get('ok')}
    if len(paths) > 1 and os.path.exists(paths[1]) and len(use) < len(P):
        for n, r in json.load(open(paths[1]))['models'].items():
            if r.get('ok') and len(use) < len(P):
                use[n] = r
    return use, [n for n, r in P.items() if not r.get('ok')]


def analyze20(base20, hstar_json, base19=None, jobs19=None, seed=A20_SEED_BOOT, boot=1000):
    paths = [hstar_json] if isinstance(hstar_json, str) else list(hstar_json)
    P, not_pred = models20(paths)
    EE, hs, grids, missing = {}, {}, {}, list(not_pred)
    for n, r in P.items():
        h, grid = round(float(r['target']['hstar']), 4), [float(t) for t in r['target']['grid']]
        if not all(os.path.exists(f'{base20}/{n}@{a}_g{g}.npz') for a in ('c0', 'h1') for g in range(3)):
            missing.append(n); continue
        EE[n] = _events(base20, n, h, grid); hs[n] = h; grids[n] = grid
    res = {'missing': missing, 'models': {}}
    names = list(EE)
    nD = None if not names else EE[names[0]]['c0'][0].shape[2]
    assert all(EE[n]['c0'][0].shape[2] == nD and EE[n]['h1'][0].shape[2] == nD for n in names)
    one = np.ones(nD) if nD else None
    uses = {}
    for n in names:
        r, u, D, Pp = rho_model(EE[n], ('h1', hs[n]), grids[n], one)
        uses[n] = u
        res['models'][n] = dict(rho=r, use=u, h=hs[n], D=None if D is None else D.tolist(), P=None if Pp is None else Pp.tolist())
    if len(names) != 2:
        res['XQ7'] = dict(word='はっきりしない', note=f'第 2 段がそろったモデルが {len(names)} 本')
        return res, EE, hs, grids
    pt = [res['models'][n]['rho'] for n in names]
    rng = np.random.default_rng(seed)
    Bn = {n: [] for n in names}; Bm = []
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        v = [rho_model(EE[n], ('h1', hs[n]), grids[n], w, uses[n])[0] if len(uses[n]) >= A18_MIN_T else float('nan') for n in names]
        for n, x in zip(names, v):
            Bn[n].append(x)
        Bm.append(float(np.mean(v)) if np.all(np.isfinite(v)) else float('nan'))
    for n in names:
        v = np.array(Bn[n], float); fl = float(np.mean(~np.isfinite(v)))
        ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
        res['models'][n].update(ci=ci, fail=fl, word=word18(ci, fl))
    v = np.array(Bm, float); fl = float(np.mean(~np.isfinite(v)))
    ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
    res['XQ7'] = dict(mean=float(np.mean(pt)) if np.all(np.isfinite(pt)) else float('nan'), ci=ci, fail=fl, word=word18(ci, fl),
                      per_model={n: res['models'][n]['word'] for n in names},
                      both_points_in_band=bool(all(np.isfinite(x) and A18_BAND[0] <= x <= A18_BAND[1] for x in pt)))   # 記録：平均が片方の外れを隠していないか
    res['nD'] = nD
    if base19 is not None and jobs19 is not None:                       # 記録：A2-19 の 7 本と合わせた 9 本
        _, EE19, hs19, grids19 = analyze19(base19, jobs19, boot=2)
        EA = {**{n: E for n, E in EE19.items()}, **{n: {'c0': EE[n]['c0'], 'h1': EE[n]['h1'], 'h2': EE[n]['h1']} for n in names}}
        HA = {**hs19, **{n: {'h1': hs[n], 'h2': hs[n]} for n in names}}
        GA = {**grids19, **grids}
        r9 = analyze_all(EA, HA, GA, nD, seed, boot)
        res['record_9models'] = dict(median=r9['verdict']['h1']['median'], ci=r9['verdict']['h1']['ci'], word=r9['verdict']['h1']['word'])
    return res, EE, hs, grids


def self_test20(boot=40):
    """作り物の npz（形だけ本物と同じ）で、読み込み → 2 本の平均 → 判定、欠けたモデル・予言できないモデルの扱いまで動くこと。"""
    import tempfile
    rng = np.random.default_rng(7)
    tmp = tempfile.mkdtemp()
    grid = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    P = {'models': {'m0': dict(ok=True, target=dict(hstar=-0.4, grid=grid)), 'm1': dict(ok=True, target=dict(hstar=-0.5, grid=grid)),
                    'm2': dict(ok=False, why='テスト')}}
    json.dump(P, open(f'{tmp}/h.json', 'w'))
    for m, h in (('m0', -0.4), ('m1', -0.5)):
        for a, hh in (('c0', 0.0), ('h1', h)):
            for g in range(3):
                pk = rng.random((7, 2, 4, 512)) < 0.5
                np.savez_compressed(f'{tmp}/{m}@{a}_g{g}.npz', temps=np.array(grid), h=hh, arm=a, group=g, use=np.arange(4),
                                    toks=rng.integers(0, 50, (7, 2, 4, 512)).astype(np.int32), cand=pk, pick=pk,
                                    q=np.zeros((7, 2, 4, 512), np.float32), q0=np.zeros((7, 2, 4, 512), np.float32))
    res, EE, _, _ = analyze20(tmp, f'{tmp}/h.json', boot=boot)
    assert res['nD'] == 12 and set(EE) == {'m0', 'm1'} and res['missing'] == ['m2'] and 'word' in res['XQ7']
    os.remove(f'{tmp}/m1@h1_g2.npz')
    res2, _, _, _ = analyze20(tmp, f'{tmp}/h.json', boot=boot)
    assert res2['XQ7']['word'] == 'はっきりしない' and set(res2['missing']) == {'m1', 'm2'}
    return 'OK'
