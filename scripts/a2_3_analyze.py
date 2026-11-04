"""A2-3（ゲート v0.2、凍結 2026-10-28 14:15）の分析。scripts/a2_2_analyze.py を実行したのと同じ名前空間で実行する
（pav・crossing_mono・feedback_stats・per_doc_feedback・t_cal・distinct4 を変えずに使う）。データを開く前に固定する。

    analyze3(a2_3 のフォルダ, A2-2 の qwen2.5-1.5b.npz のパス) -> dict

細部（データを開く前に決めた）：
- 乱数：feedback_stats の種は条件ごとに np.random.default_rng(0) から作り直す（条件の順番で結果が変わらないように）。
- 参考の判定で、交差がなく F が ±ε にとどまるときは、当てはめた F の絶対値がいちばん小さい格子の温度を点とする。
- A100 の確かめ (b)：記事の番号を共有したブートストラップ 2,000 回（種 1）。各回で、両方の F を同じ記事の組で作り、
  それぞれの全データの ε で T_0 を求め、log 比をとる。両方とも交差した回だけで 95% 区間を作る。
- ε を固定した T_0（記録）：ε ＝ 0.194（A2-2 の 7 モデルの ε の中央値）。
- コードの探索：記事ごとの NLL がいちばん低い 4 分の 1（10 本）を除いた 30 本で T_0（同じ規則）。
"""
import math
import numpy as np

BAND = (0.78, 0.93)
CI_RATIO_MAX, CROSS_FRAC_MIN = 1.20, 0.9
EPS_FIXED = 0.194
A22_BBC_T0 = {'qwen2.5-1.5b': 0.910086, 'smollm2-1.7b': 0.800452, 'pythia-1b': 0.852045}   # results/A2_2/a2_2_summary.json
JUDGED = ['qwen2.5-7b@bbc', 'olmo2-7b@bbc', 'qwen2.5-1.5b@code', 'smollm2-1.7b@code', 'pythia-1b@code']


def main_verdict(fb, temps):
    T0, (lo, hi), frac, eps = fb['T0'], fb['T0_ci'], fb['boot_cross_frac'], fb['eps']
    g = np.asarray(fb['F_pav'])
    if np.isfinite(T0):
        inside = BAND[0] <= T0 <= BAND[1]
        ci_ok = np.isfinite(lo) and np.isfinite(hi)
        if inside and ci_ok and hi / lo <= CI_RATIO_MAX and frac >= CROSS_FRAC_MIN:
            return '帯に入る'
        if not inside and ci_ok and (hi < BAND[0] or lo > BAND[1]):
            return '帯の外（上）' if T0 > BAND[1] else '帯の外（下）'
        return 'はっきりしない'
    if g[0] >= eps:
        return '帯の外（下）'
    if g[-1] <= -eps:
        return '帯の外（上）'
    return 'はっきりしない'


def ref_point(fb, temps):
    T0, eps = fb['T0'], fb['eps']
    g = np.asarray(fb['F_pav'])
    if np.isfinite(T0):
        return T0, 'T0'
    if g[0] >= eps:
        return float('-inf'), 'below_grid'
    if g[-1] <= -eps:
        return float('inf'), 'above_grid'
    return float(temps[int(np.argmin(np.abs(g)))]), 'closest_to_zero'


def ref_verdict(p):
    if BAND[0] <= p <= BAND[1]:
        return '帯に入る'
    return '帯の外（上）' if p > BAND[1] else '帯の外（下）'


def a100_check(Z_new, Z_old, temps):
    assert np.array_equal(Z_new['use'], Z_old['use']) and Z_new['tf'].shape == Z_old['tf'].shape, 'A100 の確かめ：記事の組が A2-2 と違う'
    out = {'revision_same': str(Z_new['revision']) == str(Z_old['revision'])}
    out['tf_max_abs_diff'] = float(np.max(np.abs(Z_new['tf'][0].astype(np.float64) - Z_old['tf'][0].astype(np.float64))))
    Tn, _ = t_cal(Z_new['tcal'], Z_new['tcal_T']); To, _ = t_cal(Z_old['tcal'], Z_old['tcal_T'])
    out['Tcal_new'], out['Tcal_old'], out['Tcal_abs_diff'] = Tn, To, abs(Tn - To)
    fn = per_doc_feedback(Z_new['tf'][0], Z_new['gen'][0]); fo = per_doc_feedback(Z_old['tf'][0], Z_old['gen'][0])
    sn = feedback_stats(fn, temps, np.random.default_rng(0)); so = feedback_stats(fo, temps, np.random.default_rng(0))
    out['T0_new'], out['T0_old'] = sn['T0'], so['T0']
    rng = np.random.default_rng(1)
    nD = fn.shape[0]
    lr = []
    for _ in range(NBOOT):
        idx = rng.integers(0, nD, nD)
        a = crossing_mono(np.median(fn[idx], 0), temps, sn['eps'])
        b = crossing_mono(np.median(fo[idx], 0), temps, so['eps'])
        if np.isfinite(a) and np.isfinite(b):
            lr.append(math.log(a / b))
    lr = np.array(lr)
    point = math.log(sn['T0'] / so['T0']) if np.isfinite(sn['T0']) and np.isfinite(so['T0']) else float('nan')
    ci = [float(np.percentile(lr, 2.5)), float(np.percentile(lr, 97.5))] if len(lr) >= 20 else [float('nan')] * 2
    out.update(log_ratio=point, log_ratio_ci=ci, boot_both_cross=len(lr) / NBOOT)
    ok_a = out['tf_max_abs_diff'] <= 1e-3 and out['Tcal_abs_diff'] <= 1e-3
    ok_b = (np.isfinite(ci[0]) and ci[0] <= 0 <= ci[1]) or (np.isfinite(point) and abs(point) <= 0.03)
    out.update(ok_a=bool(ok_a), ok_b=bool(ok_b), ok=bool(ok_a and ok_b))
    return out


def analyze3(d, a22_qwen15_npz):
    res = {'conditions': {}}
    temps = None
    for name in ['qwen2.5-1.5b@bbc-a100'] + JUDGED:
        Z = np.load(f'{d}/{name}.npz')
        temps = [float(t) for t in Z['temps']]
        Tcal, nbad = t_cal(Z['tcal'], Z['tcal_T'])
        c = {'Tcal': Tcal, 'tcal_bad_docs': nbad, 'n_docs': int(Z['tf'].shape[1]), 'chunk': int(Z['chunk'])}
        for q, key in [(0, 'H'), (3, 'H2')]:
            fd = per_doc_feedback(Z['tf'][q], Z['gen'][q])
            fb = feedback_stats(fd, temps, np.random.default_rng(0))
            c['fb_' + key] = fb
            if key == 'H':
                c['T0_eps_fixed'] = crossing_mono(fb['F'], temps, EPS_FIXED)
                if name.endswith('@code'):
                    keep = np.argsort(Z['nll'])[len(Z['nll']) // 4:]
                    c['T0_without_lowest_nll_quarter'] = feedback_stats(fd[np.sort(keep)], temps, np.random.default_rng(0))['T0']
        for q, key in [(1, 'top1'), (2, 'ent')]:
            c['F_' + key] = np.median(per_doc_feedback(Z['tf'][q], Z['gen'][q]), 0).tolist()
        c['distinct4_last128'] = distinct4(Z['toks'])
        c['nll'] = Z['nll'].tolist()
        c['ext'] = [str(x) for x in Z['ext']]
        c['main'] = main_verdict(c['fb_H'], temps)
        p, how = ref_point(c['fb_H'], temps)
        c['ref_point'], c['ref_point_how'], c['ref'] = p, how, ref_verdict(p)
        res['conditions'][name] = c
    res['temps'] = temps
    res['a100_check'] = a100_check(np.load(f'{d}/qwen2.5-1.5b@bbc-a100.npz'), np.load(a22_qwen15_npz), temps)
    mains = [res['conditions'][n]['main'] for n in JUDGED]
    if not res['a100_check']['ok']:
        overall = 'はっきりしない（A100 の確かめが通らなかった）'
    elif all(m == '帯に入る' for m in mains):
        overall = '一般的'
    elif any(m.startswith('帯の外') for m in mains):
        overall = '一般的ではない'
    else:
        overall = 'はっきりしない'
    refs = [res['conditions'][n]['ref'] for n in JUDGED]
    res['overall_main'] = overall
    res['overall_ref'] = '一般的' if all(r == '帯に入る' for r in refs) else '一般的ではない'
    # 軸ごとの読み方（ゲート §2）
    size = [res['conditions'][n]['main'] for n in JUDGED[:2]]
    outs = [m for m in size if m.startswith('帯の外')]
    res['axis_size'] = ('大きさによる（' + outs[0] + '）' if len(outs) == 2 and outs[0] == outs[1]
                        else '2 つとも外れる（向きが逆）' if len(outs) == 2
                        else 'その家族では外れる' if outs else '外れなし')
    code = [res['conditions'][n]['main'] for n in JUDGED[2:]]
    outc = [m for m in code if m.startswith('帯の外')]
    res['axis_text'] = ('文章の種類による（' + outc[0] + '）' if len(outc) == 3 and len(set(outc)) == 1
                        else 'モデルと文章の組み合わせによる' if outc else '外れなし')
    # 採点用
    res['Q3_code_in_band'] = sum(m == '帯に入る' for m in code)
    moves = {}
    for n in JUDGED[2:]:
        base = A22_BBC_T0[n.split('@')[0]]
        p = res['conditions'][n]['ref_point']
        r = p / base if np.isfinite(p) else (float('inf') if p > 0 else 0.0)
        moves[n] = '上がる' if r > 1.03 else '下がる' if r < 0.97 else 'ほぼ同じ'
    vals = list(moves.values())
    res['Q4_moves'] = moves
    res['Q4'] = max(set(vals), key=vals.count) if max(vals.count(v) for v in set(vals)) >= 2 else 'ばらばら'
    return res
