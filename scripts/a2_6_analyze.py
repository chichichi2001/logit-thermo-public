"""A2-6（ゲート docs/A2_6_GATE.md）の分析：「乳の種類」テスト。データを開く前に固定する。CPU。
scripts/a2_2_analyze.py（per_doc_feedback・feedback_stats・crossing_mono・pav）と scripts/a2_5_analyze.py（spearman）を
実行したのと同じ名前空間で実行する。記事の本文は使わない（npz の tf と gen だけ）。

    analyze6(a2_5 のフォルダ, a2_6 のフォルダ) -> dict
"""
import math
import numpy as np

A6_FAMS = {  # 家族 → [(名前, 名目のパラメータの数)]。Pythia の 6 回は A2-5 のデータ（同じ記事・同じ生成の種）
    'Pythia': [('pythia-70m', 70e6), ('pythia-160m', 160e6), ('pythia-410m', 410e6), ('pythia-1b', 1.0e9),
               ('pythia-1.4b', 1.4e9), ('pythia-2.8b', 2.8e9)],
    'Qwen2.5': [('qwen2.5-0.5b', 0.5e9), ('qwen2.5-1.5b', 1.5e9), ('qwen2.5-3b', 3.0e9)],
    'SmolLM2': [('smollm2-135m', 135e6), ('smollm2-360m', 360e6), ('smollm2-1.7b', 1.7e9)],
    'Qwen3': [('qwen3-0.6b', 0.6e9), ('qwen3-1.7b', 1.7e9), ('qwen3-4b', 4.0e9)],
}
A6_NEW = ['Qwen2.5', 'SmolLM2', 'Qwen3']          # Q9・Q10 で数える家族（Pythia は A2-5 で見たので数えない）
A6_BOOT, A6_BOOT_SEED = 1000, 20261103
A6_PERM, A6_PERM_SEED = 20000, 20261104
A6_ETA_LINE = 0.5
A6_T07 = 0.7
A6_REL = 0.83                                     # Q10 の二つめの温度：T_0 × 0.83（温度 0.7 ÷ A2-5 の T_0 のだいたい）
A6_EQ_PT, A6_EQ_BOOT = 1.05, 1.08                 # Q9「動かない」：家族の中の T_0 の最大 ÷ 最小（点の値・ブートストラップの 95% 点）


def eta2(y, labels):
    """全体のばらつきのうち、家族の平均の違いで説明できる割合（η²）。"""
    y = np.asarray(y, float); labels = np.asarray(labels)
    tot = ((y - y.mean()) ** 2).sum()
    if tot == 0:
        return float('nan')
    btw = sum((labels == g).sum() * (y[labels == g].mean() - y.mean()) ** 2 for g in np.unique(labels))
    return float(btw / tot)


def perm_p(y, labels, n, seed):
    rng = np.random.default_rng(seed)
    obs = eta2(y, labels)
    lab = np.asarray(labels)
    hit = sum(eta2(y, rng.permutation(lab)) >= obs - 1e-12 for _ in range(n))
    return float((1 + hit) / (1 + n))


def fam_dir(xs, yp, yb, pos, neg):
    """3 点（以上）の家族の向き：ρ が ±1 ちょうどで、ブートストラップの 95% 以上が同じ符号のときだけ向きを言う。"""
    r = spearman(xs, yp)
    ok = np.isfinite(yb).all(1)
    rb = np.array([spearman(xs, b) for b in yb[ok]]) if ok.any() else np.array([])
    fp = float((rb > 0).sum() / len(yb)); fn = float((rb < 0).sum() / len(yb))
    w = pos if (r >= 1 - 1e-9 and fp >= 0.95) else neg if (r <= -1 + 1e-9 and fn >= 0.95) else 'はっきりしない'
    return w, dict(rho=r, frac_pos=fp, frac_neg=fn, boot_ok=float(ok.mean()))


def q8_word(names, T0pt, T0b, fam_of):
    """log T_0 の η²・並べ替えの p・ブートストラップの区間から Q8 の言葉を出す。"""
    T0 = np.array([T0pt[n] for n in names]); lab = [fam_of[n] for n in names]
    if not np.isfinite(T0).all():
        return 'はっきりしない', dict(missing_T0=[n for n in names if not np.isfinite(T0pt[n])])
    y = np.log(T0)
    e = eta2(y, lab); p = perm_p(y, lab, A6_PERM, A6_PERM_SEED)
    Yb = np.log(np.stack([T0b[n] for n in names], 1))
    ok = np.isfinite(Yb).all(1)
    eb = np.array([eta2(Yb[b], lab) for b in np.nonzero(ok)[0]])
    lo, hi = (float(np.percentile(eb, 2.5)), float(np.percentile(eb, 97.5))) if ok.mean() >= 0.8 else (float('nan'), float('nan'))
    if ok.mean() < 0.8:
        w = 'はっきりしない'
    elif p <= 0.05 and lo >= A6_ETA_LINE:
        w = '家族で決まる'
    elif p > 0.2 and hi < A6_ETA_LINE:
        w = '家族の効き目は見えない'
    else:
        w = 'はっきりしない'
    return w, dict(eta2=e, perm_p=p, boot_ci=[lo, hi], boot_ok=float(ok.mean()),
                   chance=(len({fam_of[n] for n in names}) - 1) / (len(names) - 1))


def F_rel(F, temps, T0):
    """T_0 × A6_REL の温度での F（対数温度で直線補間）。"""
    if not np.isfinite(T0):
        return float('nan')
    Ts = A6_REL * T0
    if not (temps[0] <= Ts <= temps[-1]):
        return float('nan')
    return float(np.interp(math.log(Ts), np.log(temps), F))


def analyze6_from_fd(fds, temps):
    """fds：{名前: (nD, nT)}。全モデルで記事の並びが同じであること（共通の記事にそろえてから渡す）。"""
    nD = {v.shape[0] for v in fds.values()}
    assert len(nD) == 1
    nD = nD.pop()
    k07 = temps.index(A6_T07)
    names = [n for f in A6_FAMS.values() for n, _ in f]
    new9 = [n for g in A6_NEW for n, _ in A6_FAMS[g]]
    fam_of = {n: g for g, f in A6_FAMS.items() for n, _ in f}
    res = {'models': {}, 'n_docs': nD}
    eps = {}
    for n in names:
        assert np.isfinite(fds[n]).all(), ('f に有限でない値があります', n)
        fs = feedback_stats(fds[n], temps, np.random.default_rng(0))
        eps[n] = fs['eps']
        F = np.median(fds[n], 0)
        res['models'][n] = dict(fam=fam_of[n], T0=fs['T0'], T0_ci=fs['T0_ci'], eps=fs['eps'], F07=float(F[k07]),
                                Frel=F_rel(F, temps, fs['T0']),
                                s=float((F[temps.index(0.9)] - F[temps.index(0.8)]) / 0.1), F=F.tolist())
    eps_c = float(np.median(list(eps.values())))
    for n in names:
        res['models'][n]['T0_common_eps'] = crossing_mono(np.median(fds[n], 0), temps, eps_c)
    rng = np.random.default_rng(A6_BOOT_SEED)
    idx = rng.integers(0, nD, size=(A6_BOOT, nD))
    Fb = {n: np.stack([np.median(fds[n][idx[b]], 0) for b in range(A6_BOOT)]) for n in names}   # (B, nT)
    T0b = {n: np.array([crossing_mono(Fb[n][b], temps, eps[n]) for b in range(A6_BOOT)]) for n in names}
    T0bc = {n: np.array([crossing_mono(Fb[n][b], temps, eps_c) for b in range(A6_BOOT)]) for n in names}
    F07b = {n: Fb[n][:, k07] for n in names}
    Frelb = {n: np.array([F_rel(Fb[n][b], temps, T0b[n][b]) for b in range(A6_BOOT)]) for n in names}
    # Q8：T_0 は家族で決まるか。3 つの見方（15 モデル・新しい 9 モデル・共通の ε）がそろったときだけ言う
    T0pt = {n: res['models'][n]['T0'] for n in names}
    T0ptc = {n: res['models'][n]['T0_common_eps'] for n in names}
    views = {'all15': q8_word(names, T0pt, T0b, fam_of), 'new9': q8_word(new9, T0pt, T0b, fam_of),
             'all15_common_eps': q8_word(names, T0ptc, T0bc, fam_of)}
    ws = {v[0] for v in views.values()}
    res['Q8'] = ws.pop() if len(ws) == 1 else 'はっきりしない'
    res['Q8_detail'] = {k: dict(word=v[0], **v[1]) for k, v in views.items()}
    res['eps_common'] = eps_c
    # Q9：家族の中で T_0 が大きさで動くか（動かない家族の数）／ Q10：大きいほど浅く固まる家族の数
    res['Q9_by_fam'], res['Q10_by_fam'] = {}, {}
    for g, mem in A6_FAMS.items():
        xs = [math.log(p) for _, p in mem]; ns_ = [n for n, _ in mem]
        yt = np.array([T0pt[n] for n in ns_])
        Yb = np.stack([T0b[n] for n in ns_], 1)
        if np.isfinite(yt).all():
            w, d = fam_dir(xs, np.log(yt), np.log(Yb), '大きいほど高い', '大きいほど低い')
            ratio = float(yt.max() / yt.min())
            okb = np.isfinite(Yb).all(1)
            rb = Yb[okb].max(1) / Yb[okb].min(1)
            r95 = float(np.percentile(rb, 95)) if okb.mean() >= 0.8 else float('nan')
            if w == 'はっきりしない' and ratio <= A6_EQ_PT and r95 <= A6_EQ_BOOT:
                w = '動かない'
            d.update(ratio=ratio, ratio_boot95=r95)
        else:
            w, d = 'はっきりしない', dict(missing=True)
        res['Q9_by_fam'][g] = dict(word=w, **d)
        w1, d1 = fam_dir(xs, [res['models'][n]['F07'] for n in ns_], np.stack([F07b[n] for n in ns_], 1), '大きいほど浅い', '大きいほど深い')
        w2, d2 = fam_dir(xs, [res['models'][n]['Frel'] for n in ns_], np.stack([Frelb[n] for n in ns_], 1), '大きいほど浅い', '大きいほど深い')
        res['Q10_by_fam'][g] = dict(word=w1 if w1 == w2 else 'はっきりしない', F07=dict(word=w1, **d1), Frel=dict(word=w2, **d2))
    res['Q9'] = sum(res['Q9_by_fam'][g]['word'] == '動かない' for g in A6_NEW)
    res['Q9_counts'] = {w: sum(res['Q9_by_fam'][g]['word'] == w for g in A6_NEW) for w in ('動かない', '大きいほど高い', '大きいほど低い', 'はっきりしない')}
    res['Q10'] = sum(res['Q10_by_fam'][g]['word'] == '大きいほど浅い' for g in A6_NEW)
    # 記録：家族ごとの T_0 の平均（対数で平均して戻す）と幅
    res['fam_T0'] = {g: dict(geo_mean=float(np.exp(np.mean(np.log([T0pt[n] for n, _ in mem])))),
                             range=[float(min(T0pt[n] for n, _ in mem)), float(max(T0pt[n] for n, _ in mem))])
                     for g, mem in A6_FAMS.items() if all(np.isfinite(T0pt[n]) for n, _ in mem)}
    return res


def analyze6(d5, d6):
    fams_dir = {n: (d5 if g == 'Pythia' else d6) for g, f in A6_FAMS.items() for n, _ in f}
    raw, temps, uses = {}, None, {}
    for n, d in fams_dir.items():
        Z = np.load(f'{d}/{n}.npz')
        t = [float(x) for x in Z['temps']]
        assert temps is None or t == temps, '温度の並びが違います'
        temps = t
        uses[n] = [int(u) for u in Z['use']]
        raw[n] = per_doc_feedback(Z['tf'][0], Z['gen'][0])
    common = sorted(set.intersection(*[set(u) for u in uses.values()]))
    fds = {n: raw[n][[uses[n].index(i) for i in common]] for n in raw}
    r = analyze6_from_fd(fds, temps)
    r['common_docs'] = common
    r['dropped_docs'] = {n: sorted(set(u) - set(common)) for n, u in uses.items() if set(u) - set(common)}
    return r


def self_test_a26_analyze():
    assert abs(eta2([0, 0, 1, 1], ['a', 'a', 'b', 'b']) - 1) < 1e-12
    assert abs(eta2([0, 1, 0, 1], ['a', 'a', 'b', 'b'])) < 1e-12
    temps = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
    rng = np.random.default_rng(3)
    T0fam = {'Pythia': 0.84, 'Qwen2.5': 0.91, 'SmolLM2': 0.82, 'Qwen3': 0.87}   # 交点の両どなりに温度 0.7 が入らないように
    fds = {}
    for g, mem in A6_FAMS.items():
        for j, (n, _) in enumerate(mem):
            depth = 3.0 - 0.6 * j                         # 大きいほど浅い（F07 が上がる）
            F = np.array([(t - T0fam[g]) * (8 if t < T0fam[g] else 4) for t in temps])
            F[1] = -depth if temps[1] < T0fam[g] else F[1]
            fds[n] = F[None, :] + rng.normal(0, 0.05, (40, len(temps)))
    r = analyze6_from_fd(fds, temps)
    assert r['Q8'] == '家族で決まる', r['Q8_detail']
    assert r['Q10'] == 3, r['Q10_by_fam']
    assert r['Q9'] == 3, r['Q9_by_fam']                # 家族の中で T_0 は同じ → 3 家族とも「動かない」
    assert r['Q8_detail']['new9']['word'] == '家族で決まる', r['Q8_detail']
    # 家族の効き目がないとき
    fds2 = {n: (np.array([(t - 0.85) * 6 for t in temps])[None, :] + rng.normal(0, 0.05, (40, len(temps)))) for n in fds}
    r2 = analyze6_from_fd(fds2, temps)
    assert r2['Q8'] in ('家族の効き目は見えない', 'はっきりしない'), r2['Q8_detail']
    # 雑音が大きく、家族の効き目がないとき：「家族で決まる」は出ない
    fds3 = {n: (np.array([(t - 0.85) * 6 for t in temps])[None, :] + rng.normal(0, 0.6, (40, len(temps)))) for n in fds}
    r3 = analyze6_from_fd(fds3, temps)
    assert r3['Q8'] != '家族で決まる', r3['Q8_detail']
    # 有限でない値は黙って通さない
    bad = dict(fds); bad['qwen3-4b'] = bad['qwen3-4b'].copy(); bad['qwen3-4b'][0, 0] = np.nan
    try:
        analyze6_from_fd(bad, temps); raise RuntimeError('通ってしまった')
    except AssertionError:
        pass
    return 'OK'
