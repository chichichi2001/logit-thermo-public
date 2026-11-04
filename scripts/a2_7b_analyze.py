"""A2-7b（ゲート docs/A2_7B_GATE.md）の分析：本滴定。データを開く前に固定する。CPU。
scripts/a2_2_analyze.py（pav・crossing_mono・feedback_stats・per_doc_feedback・NBOOT）、scripts/a2_4_analyze.py（stats_fixed・paired・
verdict_inf・copy_frac）を実行したのと同じ名前空間で実行する。
"""
import math
import numpy as np

B7A_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
B7A_TEMPS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
B7A_ARMS = ['none', 'none_b', 'cq33', 'cq67', 'cq100', 'pl025', 'pl050', 'pl075']
B7A_COPY, B7A_PL = ['cq33', 'cq67', 'cq100'], ['pl025', 'pl050', 'pl075']
B7A_MASS_T = [0.8, 0.85, 0.9]
B7A_BOOT, B7A_BOOT_SEED = 1000, 20261111
B7A_Q = 1.5
DOWN_WORDS = ('大きく下がる', '大きく下がる（格子の外）', '少し下がる')


def load_7b(d, m):
    Zt = np.load(f'{d}/{m}@tf.npz')
    assert [float(t) for t in Zt['temps']] == B7A_TEMPS
    tf = Zt['tf']                                                   # (nD, 8, L)
    use = [int(u) for u in Zt['use']]
    fds, mass_doc, extra = {}, {}, {}
    kk = [B7A_TEMPS.index(t) for t in B7A_MASS_T]
    for a in B7A_ARMS:
        Z = np.load(f'{d}/{m}@{a}.npz')
        assert [float(t) for t in Z['temps']] == B7A_TEMPS and [int(u) for u in Z['use']] == use, (m, a)
        fds[a] = per_doc_feedback(tf, Z['g_ell'])
        bm = Z['g_ban_mass'].astype(np.float64)                     # (8, S, nD, L)
        mass_doc[a] = bm[kk].mean(axis=(0, 1, 3))                   # (nD,)：記事ごとの量（温度 0.8・0.85・0.9、サンプル、歩で平均）
        extra[a] = dict(toks=Z['toks'], prompts=Z['prompts'], top1=Z['g_banned_top1'].astype(np.float64).mean(axis=(1, 2, 3)).tolist())
    return fds, mass_doc, extra


def slope0(x, y):
    """原点を通る直線 y ＝ −K x の K（最小二乗）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(-(x @ y) / (x @ x)) if (x @ x) > 0 else float('nan')


def log_q(Kc, Kp):
    if not (np.isfinite(Kc) and np.isfinite(Kp)):
        return float('nan')
    if Kc > 0 and Kp > 0:
        return math.log(Kc / Kp)
    if Kc > 0 >= Kp:
        return float('inf')
    if Kp > 0 >= Kc:
        return float('-inf')
    return float('nan')


def fit_once(fds, mass_doc, eps, idx, arms_c, arms_p):
    """idx の記事で、腕ごとの T_0 と量 → (K_写し, K_偽薬, 使った腕で交点が全部出たか)。"""
    T0n = crossing_mono(np.median(fds['none'][idx], 0), B7A_TEMPS, eps)
    if not np.isfinite(T0n):
        return float('nan'), float('nan'), False
    out, ok = {}, True
    for a in arms_c + arms_p:
        T0 = crossing_mono(np.median(fds[a][idx], 0), B7A_TEMPS, eps)
        if not np.isfinite(T0):
            ok = False
            continue
        out[a] = (float(mass_doc[a][idx].mean()), math.log(T0 / T0n))
    Kc = slope0([out[a][0] for a in arms_c if a in out], [out[a][1] for a in arms_c if a in out])
    Kp = slope0([out[a][0] for a in arms_p if a in out], [out[a][1] for a in arms_p if a in out])
    return Kc, Kp, ok


def bq1_word(lq_point, ci, fail_frac, enough):
    if not enough or fail_frac > 0.1 or not (np.isfinite(ci[0]) or np.isinf(ci[0])):
        return 'はっきりしない'
    lo, hi = ci
    if lo > 0 and lq_point >= math.log(B7A_Q) - 1e-12:
        return '写しの禁止のほうがよく効く'
    if lo >= -math.log(B7A_Q) and hi <= math.log(B7A_Q):
        return '同じくらい（束一的）'
    if hi < 0 and lq_point <= -math.log(B7A_Q) + 1e-12:
        return '偽薬のほうがよく効く'
    return 'はっきりしない'


def analyze7b(d):
    res = {'models': {}}
    for m in B7A_MODELS:
        fds, mass_doc, extra = load_7b(d, m)
        eps = feedback_stats(fds['none'], B7A_TEMPS, np.random.default_rng(0))['eps']
        s0 = stats_fixed(fds['none'], B7A_TEMPS, eps)
        row = {'eps': eps, 'arms': {}}
        k07 = B7A_TEMPS.index(0.7)
        for a in B7A_ARMS:
            s = stats_fixed(fds[a], B7A_TEMPS, eps)
            toks, prm = extra[a]['toks'][k07], extra[a]['prompts']
            dd = dict(T0=s['T0'], cls=s['cls'], F=s['F'], mass=float(mass_doc[a].mean()), top1_rate_by_T=extra[a]['top1'],
                      y=(math.log(s['T0'] / s0['T0']) if np.isfinite(s['T0']) and np.isfinite(s0['T0']) else float('nan')),
                      copy4_07=float(np.mean([copy_frac(prm[j], toks[q, j], 4) for q in range(toks.shape[0]) for j in range(toks.shape[1])])),
                      copy3_07=float(np.mean([copy_frac(prm[j], toks[q, j], 3) for q in range(toks.shape[0]) for j in range(toks.shape[1])])))
            if a != 'none':
                p = paired(fds[a], fds['none'], B7A_TEMPS, eps, seed=1)
                dd.update(d_vs_none=p, word=verdict_inf(p, s['cls'], np.isfinite(s0['T0'])))
            row['arms'][a] = dd
        # BQ1：点の当てはめに使う腕 ＝ 交点が出た腕
        arms_c = [a for a in B7A_COPY if np.isfinite(row['arms'][a]['T0'])]
        arms_p = [a for a in B7A_PL if np.isfinite(row['arms'][a]['T0'])]
        enough = len(arms_c) >= 2 and len(arms_p) >= 2 and np.isfinite(s0['T0'])
        nD = fds['none'].shape[0]
        Kc, Kp, _ = fit_once(fds, mass_doc, eps, np.arange(nD), arms_c, arms_p) if enough else (float('nan'),) * 3
        lq = log_q(Kc, Kp)
        rng = np.random.default_rng(B7A_BOOT_SEED)
        lqs, fail = [], 0
        for _ in range(B7A_BOOT):
            idx = rng.integers(0, nD, nD)
            if not enough:
                break
            kc, kp, ok = fit_once(fds, mass_doc, eps, idx, arms_c, arms_p)
            if not ok:
                fail += 1
                continue
            v = log_q(kc, kp)
            if not np.isnan(v):
                lqs.append(v)
        lqs = np.array(lqs)
        ci = ([float(np.percentile(lqs, 2.5, method='lower')), float(np.percentile(lqs, 97.5, method='higher'))]
              if len(lqs) >= 20 else [float('nan'), float('nan')])
        ff = fail / B7A_BOOT if enough else 1.0
        row['BQ1'] = dict(arms_copy=arms_c, arms_placebo=arms_p, K_copy=Kc, K_placebo=Kp, logQ=lq, Q=(math.exp(lq) if np.isfinite(lq) else lq),
                          ci_logQ=ci, boot_fail=ff, n_boot_used=int(len(lqs)), word=bq1_word(lq, ci, ff, enough))
        row['BQ2_word'] = row['arms']['pl075']['word']
        res['models'][m] = row
    res['BQ1'] = sum(res['models'][m]['BQ1']['word'] == '写しの禁止のほうがよく効く' for m in B7A_MODELS)
    res['BQ2'] = sum(res['models'][m]['BQ2_word'] in DOWN_WORDS for m in B7A_MODELS)
    return res


def self_test_a27b_analyze():
    """作り物：記事ごとの f（8 温度）を、T_0 ＝ T0n × exp(y) で作る。写しは y ＝ −5 × 量、偽薬は y ＝ −1.5 × 量 → Q ≈ 3.3 で「写し」。
    偽薬も −5 × 量 → 「同じくらい」。"""
    import tempfile, os
    t = np.array(B7A_TEMPS)
    rng = np.random.default_rng(0)
    nD, S, L = 40, 2, 24

    def write(dirn, m, Kc, Kp, T0n=0.86):
        tf = np.zeros((nD, 8, L), np.float32)
        np.savez(f'{dirn}/{m}@tf.npz', tf=tf, temps=np.array(B7A_TEMPS), use=np.arange(nD))
        masses = dict(none=0.0, none_b=0.0, cq33=0.014, cq67=0.028, cq100=0.042, pl025=0.014, pl050=0.029, pl075=0.043)
        for a, mass in masses.items():
            K = Kc if a.startswith('cq') else Kp if a.startswith('pl') else 0.0
            T0 = T0n * math.exp(-K * mass)
            F = np.where(t < T0, (t - T0) * 12, (t - T0) * 5)
            g = np.broadcast_to(F[:, None, None, None], (8, S, nD, L)) + rng.normal(0, 0.15, (8, S, nD, L))
            bm = np.full((8, S, nD, L), mass, np.float32) * (1 + rng.normal(0, 0.05, (8, S, nD, L)))
            np.savez(f'{dirn}/{m}@{a}.npz', g_ell=g.astype(np.float32), g_ban_mass=bm.astype(np.float32),
                     g_banned_top1=np.zeros((8, S, nD, L), bool), toks=rng.integers(0, 1000, (8, S, nD, L)).astype(np.int32),
                     prompts=rng.integers(0, 1000, (nD, 8)).astype(np.int32), temps=np.array(B7A_TEMPS), use=np.arange(nD))

    with tempfile.TemporaryDirectory() as dirn:
        write(dirn, 'pythia-1b', 5.0, 1.5)
        write(dirn, 'olmo2-1b', 5.0, 5.0)
        write(dirn, 'qwen2.5-1.5b', 5.0, 0.0)
        r = analyze7b(dirn)
    w = {m: r['models'][m]['BQ1']['word'] for m in B7A_MODELS}
    assert w['pythia-1b'] == '写しの禁止のほうがよく効く', r['models']['pythia-1b']['BQ1']
    assert w['olmo2-1b'] == '同じくらい（束一的）', r['models']['olmo2-1b']['BQ1']
    assert w['qwen2.5-1.5b'] == '写しの禁止のほうがよく効く', r['models']['qwen2.5-1.5b']['BQ1']
    assert r['BQ1'] == 2
    assert r['models']['olmo2-1b']['BQ2_word'] in DOWN_WORDS and r['models']['qwen2.5-1.5b']['BQ2_word'] not in DOWN_WORDS
    assert log_q(1.0, -0.1) == float('inf') and np.isnan(log_q(-1.0, -1.0))
    return 'OK'
