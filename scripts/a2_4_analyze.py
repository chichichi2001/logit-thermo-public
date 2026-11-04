"""A2-4（ゲート v0.2、凍結 2026-10-28 18:28）の分析。scripts/a2_2_analyze.py を実行したのと同じ名前空間で実行する
（pav・crossing_mono・feedback_stats・per_doc_feedback・distinct4・NBOOT を変えずに使う）。データを開く前に固定する。

    analyze4(a2_4 のフォルダ, a2_2 のフォルダ) -> dict

細部（データを開く前に決めた。ゲートに書いていないもの）：
- f は per_doc_feedback（t ＝ 2〜512 の平均、記事ごと）。ε はモデルごとに 1 つ：'none' の全データで feedback_stats が出す ε。
- 組にしたブートストラップは種 1、2,000 回。d_∞ は (block, none)、d_偽 は (placebo, none)、d_差 は (block, placebo) の組。
  各回で両方が交差した回だけを使う。その割合が 0.9 未満、または使える回が 20 回未満なら「はっきりしない」。
- d_差 の判定で、どちらかの T_0 が交差しない（格子の外）ときは「はっきりしない」。
- 「10% 以上は動かない」：区間の下の端 ≥ −0.10、上の端 ≤ +0.10、区間が 0 を含む。
- Q4：3 モデルのうち |d_偽| < 0.5 × |d_∞|（点）の数。どちらかの点が出ないモデルは数えない（記録する）。
- Q5：温度 0.7、128〜512 トークン目の f（記事ごとに、生成 2 本の平均 − 人の文脈、その帯で平均）の記事の平均を、
  3 モデルで足したもの同士の比（block ÷ none）。r < 0.25 →「ほぼ全部消える」、0.25〜0.75 →「半分くらい残る」、
  0.75 より大きく 1 以下 →「ほとんど残る」、1 より大きい →「増える」。
- 写しかけ歩（人の文）：続きの各位置で、直前の 3 トークンがそれより前（書き出しを含む）に出ていて、続いたトークンがあるか。
- 結晶の分（写しかけ歩の印で）：記事ごとに、t ＝ 2〜512（または帯）の歩を、生成 2 本まとめて分ける。
  結晶 ＝ c_gen ×（ℓ_gen（写しかけ）− ℓ_gen（ほか））− c_hum ×（ℓ_tf（写しかけ）− ℓ_tf（ほか））。片方の歩がないときはその項は 0。
- d_機：'none' の記事ごとの f から結晶の分を引いて、同じ ε で T_0 を出し直す。|d_機| < 0.01 なら r は読まない。
- 確かめ 1 が通らなければ、全体（QB1）だけでなく、モデルごとの判定（d_∞・d_差）も「はっきりしない」にする（ゲート §6 の 1 の「全体」をこう読む）。
- 確かめ 4（偽薬の割合）は、温度ごとに |割合 − p(T)| ≤ 0.2 × p(T)。ただし禁止の数の差が 2 回以内なら通る（数が少ないときの揺れ）。
  通らなくても判定は変えず、記録して結果ノートに書く（ゲートは確かめ 4 が外れたときの扱いを決めていない）。
- 判定が「大きく下がる（格子の外）」「上がる（格子の外）」のときは、採点では「大きく下がる」「上がる」と同じに扱う。
- Q5 の比が出ない（足した f が 0）ときは「読まない」。
"""
import math
import numpy as np

MODELS4 = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
MODES4 = ['none', 'block', 'placebo']
BAND10 = 0.10


def human_cand(prompt, human):
    seq = [int(t) for t in prompt]
    nxt, out = {}, np.zeros(len(human), bool)
    for i in range(3, len(seq)):
        nxt.setdefault(tuple(seq[i - 3:i]), set()).add(seq[i])
    for t, y in enumerate(human):
        if len(seq) >= 3:
            out[t] = tuple(seq[-3:]) in nxt
            nxt.setdefault(tuple(seq[-3:]), set()).add(int(y))
        seq.append(int(y))
    return out


def copy_frac(prompt, cont, n):
    seq = [int(t) for t in prompt] + [int(t) for t in cont]
    P, seen, c = len(prompt), set(), 0
    for i in range(n - 1, len(seq)):
        g = tuple(seq[i - n + 1:i + 1])
        if i >= P and g in seen:
            c += 1
        seen.add(g)
    return c / len(cont)


def stats_fixed(fd, temps, eps):
    F = np.median(fd, 0)
    g = pav(F)
    T0 = crossing_mono(F, temps, eps)
    cls = 'cross' if np.isfinite(T0) else ('below' if g[0] >= eps else 'above' if g[-1] <= -eps else 'flat')
    return dict(F=F.tolist(), F_pav=g.tolist(), T0=T0, cls=cls)


def paired(fdA, fdB, temps, eps, seed=1):
    """log(T_0(A) ÷ T_0(B)) の点と、記事を組にしたブートストラップの 95% 区間。"""
    a = crossing_mono(np.median(fdA, 0), temps, eps); b = crossing_mono(np.median(fdB, 0), temps, eps)
    point = math.log(a / b) if np.isfinite(a) and np.isfinite(b) else float('nan')
    rng = np.random.default_rng(seed)
    nD = fdA.shape[0]
    lr = []
    for _ in range(NBOOT):
        idx = rng.integers(0, nD, nD)
        x = crossing_mono(np.median(fdA[idx], 0), temps, eps); y = crossing_mono(np.median(fdB[idx], 0), temps, eps)
        if np.isfinite(x) and np.isfinite(y):
            lr.append(math.log(x / y))
    frac = len(lr) / NBOOT
    ci = [float(np.percentile(lr, 2.5)), float(np.percentile(lr, 97.5))] if len(lr) >= 20 else [float('nan')] * 2
    return dict(point=point, ci=ci, both_cross=frac)


def verdict_inf(p, cls_new, base_ok):
    if not base_ok:
        return 'はっきりしない'
    if cls_new == 'below':
        return '大きく下がる（格子の外）'
    if cls_new == 'above':
        return '上がる（格子の外）'
    if cls_new == 'flat':
        return 'はっきりしない'
    lo, hi = p['ci']
    if p['both_cross'] < 0.9 or not np.isfinite(lo):
        return 'はっきりしない'
    if hi < 0:
        return '大きく下がる' if p['point'] <= -BAND10 else '少し下がる'
    if lo > 0:
        return '上がる'
    if lo >= -BAND10 and hi <= BAND10 and lo <= 0 <= hi:
        return '10% 以上は動かない'
    return 'はっきりしない'


def verdict_diff(p, ok):
    lo, hi = p['ci']
    if not ok or p['both_cross'] < 0.9 or not np.isfinite(lo):
        return 'はっきりしない'
    if hi < 0:
        return '写しを止めた分がある'
    if lo > 0:
        return '偽薬のほうが下げた'
    return '偽薬と区別できない'


def gen_mean(Z, key):
    return Z['g_' + key].astype(np.float64).mean(1)            # (nT, nD, L)、生成 2 本の平均


def crystal_part(Z, Hc, tf_ell, k, a, b):
    """記事ごとの結晶の分（写しかけ歩の印で）。Z：生成の npz、Hc：(nD, L) 人の写しかけ、tf_ell：(nD, nT, L)。→ (nD,)"""
    ge = Z['g_ell'][k][:, :, a:b].astype(np.float64)            # (S, nD, L')
    gc = Z['g_cand'][k][:, :, a:b]
    nD = ge.shape[1]
    out = np.zeros(nD)
    for j in range(nD):
        x, m = ge[:, j].ravel(), gc[:, j].ravel()
        y, h = tf_ell[j, k, a:b].astype(np.float64), Hc[j, a:b]
        cg, ch = m.mean(), h.mean()
        part_g = cg * (x[m].mean() - x[~m].mean()) if (m.any() and (~m).any()) else 0.0
        part_h = ch * (y[h].mean() - y[~h].mean()) if (h.any() and (~h).any()) else 0.0
        out[j] = part_g - part_h
    return out


def analyze4(d4, d2):
    temps = None
    res = {'models': {}, 'checks': {}}
    f07 = {m: {} for m in MODES4}
    for name in MODELS4:
        Z = {m: (lambda z: {k: z[k] for k in z.files})(np.load(f'{d4}/{name}@{m}.npz')) for m in MODES4}
        Z2 = (lambda z: {k: z[k] for k in z.files})(np.load(f'{d2}/{name}.npz'))
        temps = [float(t) for t in Z['none']['temps']]
        k07 = temps.index(0.7)
        tf_ell = Z['none']['tf'][0]
        # 確かめ 1：A2-2 と同じ測り方
        assert np.array_equal(Z['none']['use'], Z2['use']) and Z['none']['tf'].shape == Z2['tf'].shape, '記事の組が A2-2 と違う'
        c1 = {'tf_max_abs_diff': float(np.max(np.abs(Z['none']['tf'][0].astype(np.float64) - Z2['tf'][0].astype(np.float64))))}
        s_new = feedback_stats(per_doc_feedback(tf_ell, Z['none']['g_ell']), temps, np.random.default_rng(0))
        s_old = feedback_stats(per_doc_feedback(Z2['tf'][0], Z2['gen'][0]), temps, np.random.default_rng(0))
        c1['T0_new_own_eps'], c1['T0_a22'] = s_new['T0'], s_old['T0']
        c1['log_ratio'] = math.log(s_new['T0'] / s_old['T0']) if np.isfinite(s_new['T0']) and np.isfinite(s_old['T0']) else float('nan')
        c1['ok'] = bool(c1['tf_max_abs_diff'] <= 1e-3 and np.isfinite(c1['log_ratio']) and abs(c1['log_ratio']) <= 0.05)
        # 確かめ 3：block で写し歩が 0
        P, Hm = Z['none']['prompts'], Z['none']['humans']
        nT, S, nD, L = Z['block']['toks'].shape
        n_copy = sum(copy_frac(P[j], Z['block']['toks'][k, s, j], 4) > 0 for k in range(nT) for s in range(S) for j in range(nD))
        # 確かめ 4：偽薬の割合
        pp = Z['placebo']['p_placebo']
        rate = Z['placebo']['g_banned_top1'].reshape(nT, -1).mean(1)
        Nst = Z['placebo']['g_banned_top1'][0].size
        rel = [0.0 if abs(r - p) * Nst <= 2 else (abs(r / p - 1) if p > 0 else float('inf')) for r, p in zip(rate, pp)]
        # 確かめ 5：1 歩目のつまみ前の ℓ が同じ
        e0 = [Z[m]['g_ell'][:, :, :, 0] for m in MODES4]
        c5 = bool(np.allclose(e0[0], e0[1], atol=1e-4) and np.allclose(e0[0], e0[2], atol=1e-4))
        # 確かめ 2（データの側）：block で禁止した歩（禁止した確率 > 1e-3）では、つまみ後の ℓ がつまみ前と違う
        mb = Z['block']['g_banned'] & (Z['block']['g_ban_mass'] > 1e-3)
        c2 = float(np.mean(np.abs(Z['block']['g_ell_pen'][mb] - Z['block']['g_ell'][mb]) > 1e-6)) if mb.any() else float('nan')
        res['checks'][name] = dict(check1=c1, check3_block_seqs_with_copy=int(n_copy), check4_rate=rate.tolist(), check4_p=pp.tolist(),
                                   check4_ok=bool(all(x <= 0.2 for x in rel)), check5_first_step_same=c5,
                                   check2_frac_pen_differs=c2, check2_ok=bool(np.isnan(c2) or c2 >= 0.99))
        # 判定
        fd = {m: per_doc_feedback(tf_ell, Z[m]['g_ell']) for m in MODES4}
        eps = s_new['eps']
        st = {m: stats_fixed(fd[m], temps, eps) for m in MODES4}
        base_ok = st['none']['cls'] == 'cross' and c1['ok']          # 確かめ 1 が通らなければ、モデルごとの判定も「はっきりしない」
        p_inf = paired(fd['block'], fd['none'], temps, eps)
        p_pl = paired(fd['placebo'], fd['none'], temps, eps)
        p_df = paired(fd['block'], fd['placebo'], temps, eps)
        m = dict(eps=eps, stats=st, d_inf=p_inf, d_placebo=p_pl, d_diff=p_df,
                 verdict_inf=verdict_inf(p_inf, st['block']['cls'], base_ok),
                 verdict_diff=verdict_diff(p_df, base_ok and st['block']['cls'] == 'cross' and st['placebo']['cls'] == 'cross'))
        # d_機
        Hc = np.stack([human_cand(P[j], Hm[j]) for j in range(nD)])
        cr = np.stack([crystal_part(Z['none'], Hc, tf_ell, k, 1, L) for k in range(nT)], 1)      # (nD, nT)
        T0m = crossing_mono(np.median(fd['none'] - cr, 0), temps, eps)
        m['T0_mech'] = T0m
        m['d_mech'] = math.log(T0m / st['none']['T0']) if np.isfinite(T0m) and base_ok else float('nan')
        dm, dd = m['d_mech'], p_df['point']
        if np.isfinite(dm) and abs(dm) >= 0.01 and np.isfinite(dd):
            r = dd / dm
            m['r'] = r
            m['r_reading'] = ('フィードバックで増えた' if r > 1.5 else 'ほぼ見積もりどおり' if r >= 0.67 else
                              '打ち消された' if r >= 0 else '逆向き')
        else:
            m['r'], m['r_reading'] = float('nan'), '読まない'
        # 記録
        for mo in MODES4:
            g = gen_mean(Z[mo], 'ell') - tf_ell.transpose(1, 0, 2)                                  # (nT, nD, L)
            m.setdefault('f_by_position', {})[mo] = {f'{a}-{b}': np.median(g[:, :, a:b].mean(-1), 1).tolist()
                                                     for a, b in [(1, 8), (8, 32), (32, 128), (128, 512)]}
            f07[mo][name] = float(g[k07, :, 128:512].mean(-1).mean())
            for q, key in [(1, 'top1'), (2, 'ent')]:
                m.setdefault('F_' + key, {})[mo] = np.median(per_doc_feedback(Z['none']['tf'][q], Z[mo]['g_' + key]), 0).tolist()
            m.setdefault('distinct4', {})[mo] = distinct4(Z[mo]['toks'])
            m.setdefault('cand_rate', {})[mo] = Z[mo]['g_cand'].reshape(nT, -1).mean(1).tolist()
            m.setdefault('banned_rate', {})[mo] = Z[mo]['g_banned'].reshape(nT, -1).mean(1).tolist()
            m.setdefault('ban_mass_mean', {})[mo] = Z[mo]['g_ban_mass'].reshape(nT, -1).mean(1).tolist()
            m.setdefault('crystal_part_07_08_late', {})[mo] = {
                str(T): float(crystal_part(Z[mo], Hc, tf_ell, temps.index(T), 128, 512).mean()) for T in [0.7, 0.8]}
            m.setdefault('copy_frac_n', {})[mo] = {
                f'n{n}': {str(T): float(np.mean([copy_frac(P[j], Z[mo]['toks'][temps.index(T), s, j], n)
                                                for s in range(S) for j in range(nD)])) for T in [0.7, 0.8]} for n in [3, 4, 8]}
            m.setdefault('stats_raw_pen_T0', {})[mo] = crossing_mono(np.median(per_doc_feedback(tf_ell, Z[mo]['g_ell_pen']), 0), temps, eps)
        m['copy_frac_n_human'] = {f'n{n}': float(np.mean([copy_frac(P[j], Hm[j], n) for j in range(nD)])) for n in [3, 4, 8]}
        res['models'][name] = m
    res['temps'] = temps
    # 全体
    ok_all = all(c['check1']['ok'] for c in res['checks'].values())
    A = [n for n in MODELS4 if res['models'][n]['verdict_diff'] == '写しを止めた分がある']
    Bm = [n for n in MODELS4 if res['models'][n]['verdict_diff'] == '偽薬と区別できない' and res['models'][n]['verdict_inf'] == '10% 以上は動かない']
    if not ok_all:
        qb1 = 'はっきりしない（確かめ 1 が通らなかった）'
    elif len(A) == 3:
        qb1 = '4 トークンの完全な写しは T_0 を支えている'
    elif len(Bm) == 3:
        qb1 = 'T_0 は 4 トークンの完全な写しでは決まっていない'
    elif A and Bm:
        qb1 = 'モデルによる'
    else:
        qb1 = 'はっきりしない'
    res['QB1'] = qb1
    res['check1_all_ok'] = ok_all
    cnt, skipped = 0, []
    for n in MODELS4:
        a, b = res['models'][n]['d_placebo']['point'], res['models'][n]['d_inf']['point']
        if np.isfinite(a) and np.isfinite(b):
            cnt += int(abs(a) < 0.5 * abs(b))
        else:
            skipped.append(n)
    res['Q4'] = dict(count=cnt, skipped=skipped)
    sn, sb = sum(f07['none'].values()), sum(f07['block'].values())
    r5 = sb / sn if sn != 0 else float('nan')
    res['Q5'] = dict(ratio=r5, f07=f07, answer=('読まない（比が出ない）' if not np.isfinite(r5) else '増える' if r5 > 1 else 'ほとんど残る' if r5 > 0.75 else
                                               '半分くらい残る' if r5 >= 0.25 else 'ほぼ全部消える'))
    dm = {n: res['models'][n]['d_mech'] for n in MODELS4}
    dd = {n: res['models'][n]['d_diff']['point'] for n in MODELS4}
    res['order_record'] = dict(d_mech=dm, d_diff=dd)
    res['Q1_3'] = {n: res['models'][n]['verdict_inf'] for n in MODELS4}
    return res


def retok_record(d4, hf_ids, T=0.7):
    """記録（ゲート §7）：生成した続きをデコードしてエンコードし直したとき、トークン列が変わる文の割合（温度 T）。"""
    from transformers import AutoTokenizer
    out = {}
    for name in MODELS4:
        tok = AutoTokenizer.from_pretrained(hf_ids[name])
        out[name] = {}
        for mo in MODES4:
            Z = np.load(f'{d4}/{name}@{mo}.npz')
            k = [float(t) for t in Z['temps']].index(T)
            toks = Z['toks'][k]
            diff = [list(tok(tok.decode(toks[s, j]), add_special_tokens=False).input_ids) != [int(t) for t in toks[s, j]]
                    for s in range(toks.shape[0]) for j in range(toks.shape[1])]
            out[name][mo] = float(np.mean(diff))
    return out
