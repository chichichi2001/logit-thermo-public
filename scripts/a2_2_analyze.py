"""A2-2（ゲート v0.3、凍結 2026-10-27 21:34）の分析。データを開く前に書いて固定する。

    python3 scripts/a2_2_analyze.py <A2_2 のフォルダ> <出力 json>

ゲートに書いていない細部（データを開く前にここで決めた。結果ノートにも写す）：
- F(T)：書き出しごとに、生成 2 本の ℓ_gen を平均してから ℓ_tf を引き、t = 2〜512 で平均。書き出しどうしの中央値。
- ε：格子の各温度で、書き出し単位のブートストラップ（2,000 回、種 0）による F の 95% 区間の半分の幅。その中央値。
  T_0 のブートストラップでは ε を全データの値に固定する（入れ子のブートストラップはしない）。
  交差しない回は区間の計算から外し、その割合を記録する。
- T_cal：全書き出しの Σ(⟨z⟩_T − z_y) を足し、符号が + から − に変わるところを対数温度で直線補間。
  有限でない値を含む書き出し（人のトークンが外した番号だった場合）は T_cal から外して記録する。
- a：a_loc の、全位置・全書き出しの中央値。
- 対照 B は「ほかの家族の T_0 の平均（算術）」、C は「a × ほかの家族の T_0 ÷ a の平均」。どちらも交差のあるモデルだけで作る。
- E2 の「誤差 10% 以内」は |log(T_0 ÷ T_cal)| ≤ log(1.10)。「区間の幅 15% 以下」は 区間の上 ÷ 下 ≤ 1.15。
- E1：どちらかの組で T_0 がない（交差なし）ときは「交差なしで判定できない」と書く。
- P2：横軸 x1 = log T、x3 = ℓ_tf(T)（全位置・全書き出しの中央値）。全 7 モデルの 15 点を使う。
  3 次の最小二乗スプライン（scipy LSQUnivariateSpline、k = 3）。内側の結び目は 5 つで、学習に使う x の分位点
  （1/6, …, 5/6）に置く。x1 と x3 で柔らかさをそろえるため。同じ x は平均してまとめる。
  学習の範囲の外は端の値で止める（ext = 'const'）。範囲の外の点の割合を記録する。
  （最初は平滑化スプライン s = 点の数 × σ² にしていたが、作り物のデータの確かめで、x3 の近い点が並ぶと
   スプラインが暴れて x3 だけが不公平に負けることが分かったので、データを開く前に替えた。）
  誤差は「抜いた家族の各モデルの二乗誤差の平均」の、全モデルでの平均。
- Q6（記録）：自分の文への特別な反応 A = |ℓ_gen − ℓ_cross|、変な文だから B = |ℓ_cross − ℓ_tf(文字の窓)|。
  ℓ_gen は読み手自身の生成の t = 2〜512 の平均、ℓ_cross と ℓ_tf(文字の窓) は対照のファイルの値。
  4 つの向き × 15 温度で平均した A と B を比べ、|A − B| ≤ 0.2 × max(A, B) なら「同じくらい」。
"""
import json, math, os, sys
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

MODELS = [('pythia-1b', 'Pythia', 'base'), ('qwen2.5-0.5b', 'Qwen', 'base'), ('qwen2.5-0.5b-inst', 'Qwen', 'inst'),
          ('qwen2.5-1.5b', 'Qwen', 'base'), ('olmo2-1b', 'OLMo', 'base'), ('smollm2-1.7b', 'SmolLM', 'base'),
          ('smollm2-1.7b-inst', 'SmolLM', 'inst')]
PAIRS = [('qwen2.5-0.5b', 'qwen2.5-0.5b-inst'), ('smollm2-1.7b', 'smollm2-1.7b-inst')]
CROSS = [('pythia-1b', 'olmo2-1b'), ('olmo2-1b', 'pythia-1b'), ('qwen2.5-1.5b', 'smollm2-1.7b'), ('smollm2-1.7b', 'qwen2.5-1.5b')]
NBOOT, SEED = 2000, 0
LOG110 = math.log(1.10)


def pav(y):
    blocks = [[float(v), 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, n in blocks:
        out += [m] * n
    return np.array(out)


def crossing_mono(F, temps, eps):
    g = pav(np.asarray(F, float))
    neg = np.nonzero(g <= -eps)[0]; pos = np.nonzero(g >= eps)[0]
    if len(neg) == 0 or len(pos) == 0 or pos[0] < neg[-1]:
        return float('nan')
    i, j = neg[-1], pos[0]
    lt0, lt1 = math.log(temps[i]), math.log(temps[j])
    return float(math.exp(lt0 + (0 - g[i]) / (g[j] - g[i]) * (lt1 - lt0)))


def per_doc_feedback(tf_q, gen_q):
    """tf_q：(nD, nT, L)、gen_q：(nT, S, nD, L) → (nD, nT)：t = 2〜L の f の平均。"""
    g = gen_q.astype(np.float64).mean(1).transpose(1, 0, 2)        # (nD, nT, L)
    f = g - tf_q.astype(np.float64)
    return f[:, :, 1:].mean(-1)


def feedback_stats(fd, temps, rng):
    nD = fd.shape[0]
    F = np.median(fd, 0)
    idx = rng.integers(0, nD, size=(NBOOT, nD))
    Fb = np.median(fd[idx], 1)                                       # (NBOOT, nT)
    half = (np.percentile(Fb, 97.5, 0) - np.percentile(Fb, 2.5, 0)) / 2
    eps = float(np.median(half))
    T0 = crossing_mono(F, temps, eps)
    T0b = np.array([crossing_mono(Fb[b], temps, eps) for b in range(NBOOT)])
    ok = np.isfinite(T0b)
    ci = [float(np.percentile(T0b[ok], 2.5)), float(np.percentile(T0b[ok], 97.5))] if ok.sum() >= 20 else [float('nan')] * 2
    return dict(F=F.tolist(), F_ci_half=half.tolist(), eps=eps, F_pav=pav(F).tolist(), T0=T0, T0_ci=ci,
                boot_cross_frac=float(ok.mean()))


def t_cal(tcal, tcal_T):
    fin = np.isfinite(tcal).all(1)
    S = tcal[fin].sum(0)
    lt = np.log(tcal_T)
    i = np.nonzero((S[:-1] > 0) & (S[1:] <= 0))[0]
    if len(i) == 0:
        return float('nan'), int((~fin).sum())
    i = i[0]
    return float(math.exp(lt[i] + S[i] / (S[i] - S[i + 1]) * (lt[i + 1] - lt[i]))), int((~fin).sum())


def distinct4(toks):
    """toks：(nT, S, nD, L) → (nT,) 後半 128 トークンの distinct-4 の平均。"""
    nT, S, nD, L = toks.shape
    out = np.zeros(nT)
    for k in range(nT):
        v = []
        for s in range(S):
            for d in range(nD):
                x = toks[k, s, d, L - 128:]
                grams = {tuple(x[i:i + 4]) for i in range(len(x) - 3)}
                v.append(len(grams) / (len(x) - 3))
        out[k] = np.mean(v)
    return out.tolist()


def spline_error(xs, Fs, fams, sigma):
    errs, outside = [], []
    for f in sorted(set(fams)):
        tr = [i for i, g in enumerate(fams) if g != f]
        te = [i for i, g in enumerate(fams) if g == f]
        x = np.concatenate([xs[i] for i in tr]); y = np.concatenate([Fs[i] for i in tr])
        ux = np.unique(x)
        uy = np.array([y[x == v].mean() for v in ux])
        knots = np.quantile(ux, np.arange(1, 6) / 6)
        sp = LSQUnivariateSpline(ux, uy, knots, k=3, ext='const')
        for i in te:
            errs.append(float(np.mean((sp(xs[i]) - Fs[i]) ** 2)))
            outside.append(float(np.mean((xs[i] < ux.min()) | (xs[i] > ux.max()))))
    return float(np.mean(errs)), float(np.mean(outside))


def judge(models, temps, q_tag):
    """models：name → dict(fam, kind, fb, Tcal, a, ell_tf)。E1・E2・P2 と判定不能の条件。"""
    out = {}
    have = {n: m for n, m in models.items() if np.isfinite(m['fb']['T0'])}
    T0s = {n: m['fb']['T0'] for n, m in have.items()}
    # 判定不能の条件
    if len(T0s) >= 2:
        rng_ratio = max(T0s.values()) / min(T0s.values())
        ci_ratios = [m['fb']['T0_ci'][1] / m['fb']['T0_ci'][0] for m in have.values() if np.isfinite(m['fb']['T0_ci'][0])]
        typ = float(np.median(ci_ratios)) if ci_ratios else float('nan')
        undec_range = bool(np.isfinite(typ) and rng_ratio < typ)
    else:
        rng_ratio, typ, undec_range = float('nan'), float('nan'), True
    tc = [m['Tcal'] for m in models.values() if np.isfinite(m['Tcal'])]
    undec_D = bool(max(tc) / min(tc) < 1.10)
    out['undecidable'] = dict(T0_range_ratio=rng_ratio, T0_ci_ratio_typical=typ, E2_P2_undecidable=undec_range,
                              Tcal_range_ratio=max(tc) / min(tc), E_vs_D_undecidable=undec_D)
    # E1
    e1 = []
    for b, i in PAIRS:
        dcal = math.log(models[i]['Tcal'] / models[b]['Tcal'])
        if b in have and i in have:
            d0 = math.log(T0s[i] / T0s[b])
            ok = bool(np.sign(d0) == np.sign(dcal) and abs(d0 - dcal) <= 0.5 * abs(dcal))
        else:
            d0, ok = float('nan'), None
        e1.append(dict(pair=[b, i], d_cal=dcal, d_0=d0, ok=ok))
    if any(p['ok'] is None for p in e1):
        v = '交差なしで判定できない'
    elif all(abs(p['d_cal']) < LOG110 for p in e1):
        v = '判定不能'
    else:
        v = '当たる' if all(p['ok'] for p in e1) else '外れる'
    out['E1'] = dict(pairs=e1, verdict=v)
    # E2
    rows = {}
    for n, m in have.items():
        others = [k for k in have if models[k]['fam'] != m['fam']]
        B = float(np.mean([T0s[k] for k in others])) if others else float('nan')
        C = m['a'] * float(np.mean([T0s[k] / models[k]['a'] for k in others])) if others else float('nan')
        T0 = T0s[n]; lo, hi = m['fb']['T0_ci']
        narrow = bool(np.isfinite(lo) and hi / lo <= 1.15)
        rows[n] = dict(T0=T0, Tcal=m['Tcal'], err_E=abs(math.log(T0 / m['Tcal'])), err_B=abs(math.log(T0 / B)),
                       err_C=abs(math.log(T0 / C)), err_D=abs(math.log(T0)), ci=[lo, hi], narrow_ci=narrow,
                       within=bool(abs(math.log(T0 / m['Tcal'])) <= LOG110 or (narrow and lo <= m['Tcal'] <= hi)))
    if rows:
        mean = {k: float(np.mean([r['err_' + k] for r in rows.values()])) for k in 'EBCD'}
        compare = ['B', 'C'] + ([] if undec_D else ['D'])
        beats = all(mean['E'] < mean[k] for k in compare)
        all_within = all(r['within'] for r in rows.values())
        v = '判定不能' if undec_range else ('当たる' if (beats and all_within) else '外れる')
        out['E2'] = dict(rows=rows, mean_err=mean, compared_with=compare, beats_baselines=beats,
                         all_within=all_within, n_models=len(rows), verdict=v,
                         excluded_no_crossing=[n for n in models if n not in have])
    else:
        out['E2'] = dict(verdict='交差のあるモデルがない', excluded_no_crossing=list(models))
    # P2
    names = list(models)
    fams = [models[n]['fam'] for n in names]
    Fs = [np.array(models[n]['fb']['F']) for n in names]
    x1 = [np.log(np.array(temps)) for n in names]
    x3 = [np.array(models[n]['ell_tf']) for n in names]
    sigma = float(np.median([models[n]['fb']['eps'] / 1.96 for n in names]))
    e1_, o1 = spline_error(x1, Fs, fams, sigma)
    e3_, o3 = spline_error(x3, Fs, fams, sigma)
    v = '判定不能' if undec_range else ('成り立つ' if e3_ < 0.7 * e1_ else '成り立たない')
    out['P2'] = dict(err_x1=e1_, err_x3=e3_, ratio=e3_ / e1_ if e1_ > 0 else float('nan'), sigma=sigma,
                     outside_x1=o1, outside_x3=o3, verdict=v)
    out['Q1_crossing'] = {n: bool(n in have) for n in models}
    base_T0 = [T0s[n] for n in have if models[n]['kind'] == 'base']
    out['Q5_base_T0_median'] = float(np.median(base_T0)) if base_T0 else float('nan')
    out['tag'] = q_tag
    return out


def analyze(d):
    rng = np.random.default_rng(SEED)
    res = {'models': {}}
    raw = {}
    for name, fam, kind in MODELS:
        Z = np.load(f'{d}/{name}.npz')
        temps = [float(t) for t in Z['temps']]
        tf, gen = Z['tf'], Z['gen']
        Tcal, n_bad = t_cal(Z['tcal'], Z['tcal_T'])
        a = float(np.median(Z['a'])); h = float(np.median(Z['h']))
        m = dict(fam=fam, kind=kind, Tcal=Tcal, tcal_bad_docs=n_bad, a=a, h=h, n_docs=int(tf.shape[1]),
                 H2=int(Z['H2']), tok_len=int(Z['tok_len']))
        for q, key in [(0, 'H'), (3, 'H2')]:
            fd = per_doc_feedback(tf[q], gen[q])
            m['fb_' + key] = feedback_stats(fd, temps, rng)
            m['ell_tf_' + key] = np.median(tf[q].astype(np.float64).reshape(-1, len(temps), tf.shape[-1])
                                           .transpose(1, 0, 2).reshape(len(temps), -1), 1).tolist()
        # 記録：1 位の log-odds とエントロピーの F、f(t) の曲線、distinct-4
        for q, key in [(1, 'top1'), (2, 'ent')]:
            fd = per_doc_feedback(tf[q], gen[q])
            m['F_' + key] = np.median(fd, 0).tolist()
        g = gen[0].astype(np.float64).mean(1).transpose(1, 0, 2) - tf[0].astype(np.float64)   # (nD, nT, L)
        bins = [(1, 8), (8, 32), (32, 128), (128, 512)]
        m['f_by_position'] = {f'{a_}-{b_}': np.median(g[:, :, a_:b_].mean(-1), 0).tolist() for a_, b_ in bins}
        m['distinct4_last128'] = distinct4(Z['toks'])
        m['ell_gen_mean'] = gen[0].astype(np.float64)[:, :, :, 1:].mean((1, 2, 3)).tolist()   # (nT,)
        res['models'][name] = m
        raw[name] = m
    res['temps'] = temps
    for key in ['H', 'H2']:
        mm = {n: dict(fam=m['fam'], kind=m['kind'], fb=m['fb_' + key], Tcal=m['Tcal'], a=m['a'], ell_tf=m['ell_tf_' + key])
              for n, m in raw.items()}
        res['judge_' + key] = judge(mm, temps, key)
    # Q6：別のモデルの文の対照
    A_all, B_all, rows = [], [], {}
    for reader, writer in CROSS:
        C = np.load(f'{d}/cross_{reader}_reads_{writer}.npz')['out']           # (2, nT, S, nD)
        cross = np.nanmean(C[0], (1, 2)); human = np.nanmean(C[1], (1, 2))
        self_ = np.array(raw[reader]['ell_gen_mean'])
        A = self_ - cross; B = cross - human
        rows[f'{reader}<-{writer}'] = dict(self_minus_cross=A.tolist(), cross_minus_human=B.tolist())
        A_all.append(np.abs(A)); B_all.append(np.abs(B))
    A_m, B_m = float(np.mean(A_all)), float(np.mean(B_all))
    if abs(A_m - B_m) <= 0.2 * max(A_m, B_m):
        v = '同じくらい'
    else:
        v = '自分の文のほうが大きい' if A_m > B_m else '変な文だからのほうが大きい'
    res['Q6'] = dict(rows=rows, mean_abs_self=A_m, mean_abs_odd=B_m, verdict=v)
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f'{float(o):.6g}')
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == '__main__':
    r = analyze(sys.argv[1])
    json.dump(_clean(r), open(sys.argv[2], 'w'), ensure_ascii=False, separators=(',', ':'))
    print('wrote', sys.argv[2])
