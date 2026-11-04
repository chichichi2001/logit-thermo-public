"""A2-5（ゲート docs/A2_5_GATE.md v0.2）の分析。データを開く前に固定する。CPU。
scripts/a2_2_collect.py・scripts/a2_2_analyze.py・scripts/a2_explore_copy.py を実行したのと同じ名前空間で実行する
（split_doc・per_doc_feedback・feedback_stats・crossing_mono・distinct4・copy_flags・decompose を変えずに使う）。

    analyze5(a2_5 のフォルダ, a2_2 のフォルダ, docs) -> dict
"""
import json, math
import numpy as np

A5_SIZE = [('pythia-70m', 70e6), ('pythia-160m', 160e6), ('pythia-410m', 410e6), ('pythia-1b', 1.0e9),
           ('pythia-1.4b', 1.4e9), ('pythia-2.8b', 2.8e9)]                       # 名目のパラメータの数（名前のとおり）
A5_STEP = [('pythia-1b@step8000', 8000), ('pythia-1b@step24000', 24000), ('pythia-1b@step64000', 64000), ('pythia-1b', 143000)]
A5_A22 = [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B'), ('qwen2.5-0.5b-inst', 'Qwen/Qwen2.5-0.5B-Instruct'),
          ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'),
          ('smollm2-1.7b-inst', 'HuggingFaceTB/SmolLM2-1.7B-Instruct')]
A5_BAND = (128, 512)
A5_FMAX, A5_MIN_DOCS = -0.2, 30
A5_BOOT, A5_BOOT_SEED = 1000, 20261103
A5_TSTAR = 0.8


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def doc_parts(Z, prompts, humans):
    """→ glass, crystal, f：(nD, nT)。decompose が外した記事・温度は nan。帯は A5_BAND。"""
    toks, gen, tf = Z['toks'], Z['gen'][0], Z['tf'][0]
    nT, S, nD, L = toks.shape
    a, b = A5_BAND
    hum = [copy_flags(prompts[j], humans[j]) for j in range(nD)]
    out = np.full((3, nD, nT), np.nan)
    for k in range(nT):
        for j in range(nD):
            gfl = np.stack([copy_flags(prompts[j], toks[k, s, j]) for s in range(S)])
            r = decompose(gen[k, :, j, a:b], gfl[:, a:b], tf[j, k, a:b], hum[j][a:b])
            if r is not None:
                out[:, j, k] = r['glass'], r['crystal'], r['f']
    return out


def G_at(parts, k, idx=None):
    """温度の番号 k での G。読めなければ nan。→ (G, ガラスの平均, f の平均, 使えた記事の数)"""
    g, f = parts[0][:, k], parts[2][:, k]
    if idx is not None:
        g, f = g[idx], f[idx]
    ok = np.isfinite(f)
    n = int(ok.sum())
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    mg, mf = float(g[ok].mean()), float(f[ok].mean())
    G = mg / mf if (n >= A5_MIN_DOCS and mf <= A5_FMAX) else float('nan')
    return G, mg, mf, n


def G_rel_at(parts, temps, T0, idx=None):
    if not np.isfinite(T0):
        return float('nan')
    Ts = A5_TSTAR * T0
    lt = np.log(temps)
    if not (temps[0] <= Ts <= temps[-1]):
        return float('nan')
    i = int(np.searchsorted(lt, math.log(Ts), side='right') - 1); i = min(i, len(temps) - 2)
    w = (math.log(Ts) - lt[i]) / (lt[i + 1] - lt[i])
    _, g0, f0, n0 = G_at(parts, i, idx); _, g1, f1, n1 = G_at(parts, i + 1, idx)
    if min(n0, n1) < A5_MIN_DOCS:
        return float('nan')
    mg, mf = (1 - w) * g0 + w * g1, (1 - w) * f0 + w * f1
    return mg / mf if mf <= A5_FMAX else float('nan')


def axis_word(xs, ys_point, ys_boot, need_all, strict_one, pos, neg):
    """xs：軸の値、ys_point：点の値（nan は読めない）、ys_boot：(B, 点の数)。→ (記号, 詳細)"""
    xs = np.asarray(xs, float); yp = np.asarray(ys_point, float)
    ok = np.isfinite(yp)
    if (need_all and not ok.all()) or ok.sum() < (len(xs) if need_all else 4):
        return 'なし', dict(rho=float('nan'), n=int(ok.sum()), frac_pos=float('nan'), frac_neg=float('nan'))
    rho = spearman(xs[ok], yp[ok])
    sp, sn = 0, 0
    for yb in ys_boot:
        okb = np.isfinite(yb)
        if (need_all and not okb.all()) or okb.sum() < (len(xs) if need_all else 4):
            continue
        r = spearman(xs[okb], yb[okb])
        sp += int(r > 0); sn += int(r < 0)
    B = len(ys_boot)
    fp, fn = sp / B, sn / B
    th = (1.0 if strict_one else 0.9) - 1e-9                  # 浮動小数の誤差で線引きの上の値が外れないように
    sym = '＋' if (rho >= th and fp >= 0.95) else '−' if (rho <= -th and fn >= 0.95) else 'なし'
    return sym, dict(rho=rho, n=int(ok.sum()), frac_pos=fp, frac_neg=fn)


def words(sym_a, sym_b, pos, neg):
    if sym_a == sym_b == '＋':
        return pos
    if sym_a == sym_b == '−':
        return neg
    return 'はっきりしない'


def model_block(Z, prompts, humans, temps):
    parts = doc_parts(Z, prompts, humans)
    fd = per_doc_feedback(Z['tf'][0], Z['gen'][0])
    fs = feedback_stats(fd, temps, np.random.default_rng(0))
    return parts, fd, fs


def analyze5(d5, d2, docs):
    from transformers import AutoTokenizer
    tokp = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
    pairs = [split_doc(tokp, d['text']) for d in docs]
    names = sorted({n for n, _ in A5_SIZE} | {n for n, _ in A5_STEP})
    res = {'models': {}, 'missing': []}
    blocks = {}
    for n in names:
        try:
            Z = (lambda z: {k: z[k] for k in z.files})(np.load(f'{d5}/{n}.npz'))
        except FileNotFoundError:
            res['missing'].append(n); continue
        use = [int(u) for u in Z['use']]
        temps = [float(t) for t in Z['temps']]
        prompts = [pairs[i][0] for i in use]; humans = [pairs[i][1] for i in use]
        parts, fd, fs = model_block(Z, prompts, humans, temps)
        blocks[n] = (parts, fd, fs, use)
        k07, k08 = temps.index(0.7), temps.index(0.8)
        G07, g07, f07, n07 = G_at(parts, k07)
        G08, _, _, _ = G_at(parts, k08)
        res['models'][n] = dict(T0=fs['T0'], T0_ci=fs['T0_ci'], eps=fs['eps'], G07=G07, G08=G08, G_rel=G_rel_at(parts, temps, fs['T0']),
                                glass07=g07, f07=f07, crystal07=float(np.nanmean(parts[1][:, k07])), n_docs07=n07,
                                copy4_07=float(np.mean([copy_flags(prompts[j], Z['toks'][k07, s, j])[A5_BAND[0]:].mean()
                                                        for s in range(Z['toks'].shape[1]) for j in range(len(use))])),
                                distinct4=distinct4(Z['toks']), ent_T1_human=float(np.mean(Z['tf'][2][:, temps.index(1.0), :])))
    uses = {tuple(b[3]) for b in blocks.values()}
    assert len(uses) == 1, '使った記事が回によって違う'
    nD = len(next(iter(uses)))
    rng = np.random.default_rng(A5_BOOT_SEED)
    boots = [rng.integers(0, nD, nD) for _ in range(A5_BOOT)]

    def boot_vals(n):
        parts, fd, fs, use = blocks[n]
        out = np.full((A5_BOOT, 3), np.nan)                       # G07, G_rel, log T0
        k07 = temps.index(0.7)
        for b, idx in enumerate(boots):
            T0b = crossing_mono(np.median(fd[idx], 0), temps, fs['eps'])
            out[b] = G_at(parts, k07, idx)[0], G_rel_at(parts, temps, T0b, idx), math.log(T0b) if np.isfinite(T0b) else np.nan
        return out
    BV = {n: boot_vals(n) for n in blocks}
    # 確かめ：ブートストラップの元の値の統計
    res['boot_ci'] = {n: {k: [float(np.nanpercentile(BV[n][:, q], 2.5)), float(np.nanpercentile(BV[n][:, q], 97.5))]
                          for q, k in enumerate(['G07', 'G_rel', 'logT0'])} for n in BV}

    def axis(spec, need_all, strict_one):
        ns = [n for n, _ in spec if n in blocks]
        xs = [math.log(v) if not strict_one else v for n, v in spec if n in blocks]
        out = {}
        for q, key in enumerate(['G07', 'G_rel', 'logT0']):
            yp = [res['models'][n][key] if key != 'logT0' else (math.log(res['models'][n]['T0']) if np.isfinite(res['models'][n]['T0']) else np.nan) for n in ns]
            yb = np.stack([BV[n][:, q] for n in ns], 1)
            out[key] = axis_word(xs, yp, yb, need_all, strict_one, None, None)
        return ns, out
    ns1, a1 = axis(A5_SIZE, need_all=False, strict_one=False)
    ns2, a2 = axis(A5_STEP, need_all=True, strict_one=True)
    if len(ns2) < len(A5_STEP):
        a2 = {k: ('なし', dict(note='学びの軸の点が欠けている')) for k in a2}
    res['axes'] = {'size': {'models': ns1, **{k: dict(sym=v[0], **v[1]) for k, v in a1.items()}},
                   'step': {'models': ns2, **{k: dict(sym=v[0], **v[1]) for k, v in a2.items()}}}
    res['Q1'] = words(a1['G07'][0], a1['G_rel'][0], '大きいほどガラス寄り', '大きいほど結晶寄り')
    res['Q2'] = words(a2['G07'][0], a2['G_rel'][0], '学ぶほどガラス寄り', '学ぶほど結晶寄り')
    s3 = a1['logT0'][0]; s4 = a2['logT0'][0]
    res['Q3'] = '大きいほど T_0 が高い' if s3 == '＋' else '大きいほど T_0 が低い' if s3 == '−' else 'はっきりしない'
    res['Q4'] = '学ぶほど T_0 が高い' if s4 == '＋' else '学ぶほど T_0 が低い' if s4 == '−' else 'はっきりしない'
    # 記録：共通の ε
    eps_c = float(np.median([b[2]['eps'] for b in blocks.values()]))
    res['T0_common_eps'] = {n: crossing_mono(np.median(b[1], 0), temps, eps_c) for n, b in blocks.items()}
    res['eps_common'] = eps_c
    # 記録：A2-2 の 7 モデル（種だけのぶれ・古さとの切り分けの手がかり）
    res['a22'] = {}
    for n, hf in A5_A22:
        tok = AutoTokenizer.from_pretrained(hf)
        pr = [split_doc(tok, d['text']) for d in docs]
        Z = (lambda z: {k: z[k] for k in z.files})(np.load(f'{d2}/{n}.npz'))
        use = [int(u) for u in Z['use']]
        parts, fd, fs = model_block(Z, [pr[i][0] for i in use], [pr[i][1] for i in use], temps)
        res['a22'][n] = dict(T0=fs['T0'], G07=G_at(parts, temps.index(0.7))[0], G_rel=G_rel_at(parts, temps, fs['T0']),
                             ent_T1_human=float(np.mean(Z['tf'][2][:, temps.index(1.0), :])))
    if 'pythia-1b' in res['models']:
        a, b = res['models']['pythia-1b'], res['a22']['pythia-1b']
        res['seed_spread_pythia1b'] = {k: [a[k], b[k]] for k in ['T0', 'G07', 'G_rel']}
    res['temps'] = temps
    return res


def self_test_a25_analyze():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9 and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9
    assert abs(spearman([1, 2, 3], [1, 1, 2]) - 0.8660254) < 1e-6          # 同じ値は平均の順位
    # 軸の判定：点がきれいに並び、ブートストラップでも向きが同じなら「＋」
    xs = np.log([70e6, 160e6, 410e6, 1e9, 1.4e9])
    yb = np.array([[0.2, 0.3, 0.4, 0.5, 0.6]] * 100) + np.random.default_rng(0).normal(0, 0.01, (100, 5))
    s, d = axis_word(xs, [0.2, 0.3, 0.4, 0.5, 0.6], yb, False, False, None, None)
    assert s == '＋' and abs(d['rho'] - 1) < 1e-9, (s, d)
    s, d = axis_word(xs, [0.6, 0.5, 0.4, 0.3, 0.2], -yb, False, False, None, None)
    assert s == '−'
    s, d = axis_word(xs, [0.2, 0.3, np.nan, 0.5, 0.6], yb, False, False, None, None)
    assert s == '＋' and d['n'] == 4
    s, d = axis_word(xs, [0.2, np.nan, np.nan, 0.5, 0.6], yb, False, False, None, None)
    assert s == 'なし'
    noisy = np.random.default_rng(1).normal(0, 1, (100, 5))
    s, d = axis_word(xs, [0.2, 0.3, 0.4, 0.5, 0.6], noisy, False, False, None, None)
    assert s == 'なし', d                                                  # 点はそろっても、向きが不確かなら言葉を付けない
    st = [8000, 24000, 64000, 143000]
    s, d = axis_word(st, [0.1, 0.2, 0.3, 0.4], np.array([[0.1, 0.2, 0.3, 0.4]] * 50), True, True, None, None)
    assert s == '＋'
    s, d = axis_word(st, [0.1, 0.3, 0.2, 0.4], np.array([[0.1, 0.2, 0.3, 0.4]] * 50), True, True, None, None)
    assert s == 'なし'                                                     # 学びの軸は完全にそろった並びだけ
    s, d = axis_word(st, [0.1, np.nan, 0.3, 0.4], np.array([[0.1, 0.2, 0.3, 0.4]] * 50), True, True, None, None)
    assert s == 'なし'
    xs5 = np.log([70e6, 160e6, 410e6, 1e9, 1.4e9])                       # ρ ＝ −0.9 ちょうど（1 つだけ入れ替わり）は「−」に入る
    yb5 = np.array([[0.5, 0.4, 0.3, 0.1, 0.2]] * 50)
    s, d = axis_word(xs5, [0.5, 0.4, 0.3, 0.1, 0.2], yb5, False, False, None, None)
    assert s == '−' and abs(d['rho'] + 0.9) < 1e-9, (s, d)
    assert words('＋', '＋', 'P', 'N') == 'P' and words('＋', 'なし', 'P', 'N') == 'はっきりしない'
    # G と G_rel：作り物の parts で、補間と読まない条件
    temps = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
    parts = np.zeros((3, 40, 15))
    parts[2] = -1.0; parts[0] = -0.4; parts[1] = -0.6                      # G ＝ 0.4 どこでも
    assert abs(G_at(parts, 1)[0] - 0.4) < 1e-12 and abs(G_rel_at(parts, temps, 1.0) - 0.4) < 1e-12
    p2 = parts.copy(); p2[2, :, 1] = -0.1; p2[0, :, 1] = -0.04             # f が −0.2 より上 → 読まない
    assert np.isnan(G_at(p2, 1)[0])
    p3 = parts.copy(); p3[:, 15:, 1] = np.nan                               # 使える記事 25 本 → 読まない
    assert np.isnan(G_at(p3, 1)[0])
    assert np.isnan(G_rel_at(parts, temps, float('nan'))) and np.isnan(G_rel_at(parts, temps, 0.5))   # T* ＝ 0.4 は格子の外
    return 'OK'
