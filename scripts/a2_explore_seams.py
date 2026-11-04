"""探索 Y の続き：「戻れなさ σ は、壊れた文の境目の数に比例するか」（2026-10-28 夜）。CPU だけ。確かめには使わない。
scripts/a2_2_collect.py と scripts/a2_explore_xy.py を実行したのと同じ名前空間で実行する
（load_docs・y_window・y_orders を使う）。GPU の計算はしない。探索 X・Y で保存した xy_{モデル}.npz の y_logp を読み直すだけ。

壊れた境目の数（元の文章で隣り合っていた文の組 (i, i+1) のうち、並べ替えたあと「同じ向きで隣」に残らなかった組の数）：
  - sent_rev：文の数 − 1（全部の組が逆向きになる）。
  - para_rev：段落の数 − 1（段落の中の組は残り、段落の境目の組だけが壊れる）。
  - shuf1・shuf2：文の数 − 1 − 「同じ向きで隣に残った組の数」（y_orders と同じ種で並べ替えを作り直して数える）。
境目 1 つあたりの値段 c ＝ σ ÷ 壊れた境目の数（ナット）。

読み方（回す前に決める。探索の目安）：
  - 主：記事ごとの c(para_rev) ÷ c(sent_rev) の中央値を、モデルごとに出す。
      4 モデルとも 0.67〜1.5 →「境目の数でだいたい説明できる」
      4 モデルとも 0.67 未満 →「段落の境目は、文の境目より安い」
      4 モデルとも 1.5 より大きい →「段落の境目は、文の境目より高い」
      それ以外 →「はっきりしない」
  - 参考：c(shuf) ÷ c(sent_rev) の中央値（2 通りのばらばらをまとめて）。同じ線引きで言葉を付ける。
  - 参考：4 つの並べ替えをまとめて σ ＝ a ＋ c × 壊れた境目の数 を当てはめた a（境目と関係ない決まった損）と c、
    原点を通る σ ＝ c × 壊れた境目の数 の当てはまり（R²）。
  - 参考：記事ごとの σ(para_rev) ÷ σ(sent_rev) と、（段落の数 − 1）÷（文の数 − 1）の順位相関。
保存するのは数だけ（記事ごとの文の数・段落の数を含む。中身は保存しない）。
"""
import json
import numpy as np

SEAM_MODELS = ['pythia-1b', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b']
SEAM_ORDERS = ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2']   # xy の y_logp の列の順番
C_LOW, C_HIGH = 0.67, 1.5


def seam_counts(win):
    """win：段落ごとの文のリスト。→ 文の数・段落の数と、並べ替えごとの壊れた境目の数。"""
    ns, npar = sum(len(p) for p in win), len(win)
    out = {'ns': ns, 'np': npar, 'sent_rev': ns - 1, 'para_rev': npar - 1}
    for k in [1, 2]:
        r = np.random.default_rng(k).permutation(ns)          # y_orders と同じ作り方
        kept = int(np.sum(r[1:] == r[:-1] + 1))
        out[f'shuf{k}'] = ns - 1 - kept
        out[f'shuf{k}_reversed_adjacent'] = int(np.sum(r[1:] == r[:-1] - 1))
    return out


def word(m):
    if all(C_LOW <= v <= C_HIGH for v in m):
        return '境目の数でだいたい説明できる'
    if all(v < C_LOW for v in m):
        return '段落（またはばらばら）の境目は、文の境目より安い'
    if all(v > C_HIGH for v in m):
        return '段落（またはばらばら）の境目は、文の境目より高い'
    return 'はっきりしない'


def rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):                                    # 同じ値は順位の平均
        m = x == v; r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze_model(lp, counts):
    """lp：(記事, 5) の log p（SEAM_ORDERS の順）。counts：seam_counts のリスト。"""
    sig = {k: lp[:, 0] - lp[:, j] for j, k in enumerate(SEAM_ORDERS) if j > 0}
    b = {k: np.array([c[k] for c in counts], float) for k in sig}
    cst = {k: sig[k] / b[k] for k in sig}
    r_para = cst['para_rev'] / cst['sent_rev']
    r_shuf = np.concatenate([cst['shuf1'], cst['shuf2']]) / np.concatenate([cst['sent_rev'], cst['sent_rev']])
    S = np.concatenate([sig[k] for k in sig]); B = np.concatenate([b[k] for k in sig])
    c0 = float((S * B).sum() / (B * B).sum())
    r2_0 = float(1 - ((S - c0 * B) ** 2).sum() / ((S - S.mean()) ** 2).sum())
    A = np.vstack([np.ones_like(B), B]).T
    (a1, c1), *_ = np.linalg.lstsq(A, S, rcond=None)
    r2_1 = float(1 - ((S - a1 - c1 * B) ** 2).sum() / ((S - S.mean()) ** 2).sum())
    q = lambda v: [float(np.percentile(v, p)) for p in (25, 50, 75)]
    return dict(
        cost_median={k: float(np.median(v)) for k, v in cst.items()},
        para_over_sent_cost_quartiles=q(r_para), shuf_over_sent_cost_quartiles=q(r_shuf),
        sigma_ratio_para_over_sent_of_means=float(sig['para_rev'].mean() / sig['sent_rev'].mean()),
        seam_ratio_para_over_sent_of_means=float(b['para_rev'].mean() / b['sent_rev'].mean()),
        spearman_doc_sigma_ratio_vs_seam_ratio=spearman(sig['para_rev'] / sig['sent_rev'], b['para_rev'] / b['sent_rev']),
        fit_origin=dict(c=c0, r2=r2_0), fit_intercept=dict(a=float(a1), c=float(c1), r2=r2_1))


def analyze_seams(docs, xydir):
    wins = [y_window(d['text']) for d in docs]
    keep = [w for w in wins if w is not None]
    counts = [seam_counts(w) for w in keep]
    for w, c in zip(keep, counts):                           # y_orders の並べ替えと数え方が合っているか
        od = y_orders(w); sents = [s for p in w for s in p]
        assert od['sent_rev'] == ' '.join(sents[::-1]) and len(sents) == c['ns']
    res = {'n_docs': len(keep),
           'ns': [c['ns'] for c in counts], 'np': [c['np'] for c in counts],
           'broken': {k: [c[k] for c in counts] for k in ['sent_rev', 'para_rev', 'shuf1', 'shuf2']},
           'shuf_reversed_adjacent': [c['shuf1_reversed_adjacent'] + c['shuf2_reversed_adjacent'] for c in counts],
           'models': {}}
    for m in SEAM_MODELS:
        z = np.load(f'{xydir}/xy_{m}.npz')
        lp = z['y_logp'].astype(np.float64)
        assert lp.shape == (len(keep), 5), (m, lp.shape, len(keep))
        res['models'][m] = analyze_model(lp, counts)
    med_p = [res['models'][m]['para_over_sent_cost_quartiles'][1] for m in SEAM_MODELS]
    med_s = [res['models'][m]['shuf_over_sent_cost_quartiles'][1] for m in SEAM_MODELS]
    res['reading_main'] = word(med_p)
    res['reading_shuf'] = word(med_s)
    return res


def self_test_seams():
    # 数え方：段落 [3, 2, 1] 文 → 文の逆順 5、段落の逆順 2
    win = [['a1 aaa.', 'a2 aaa.', 'a3 aaa.'], ['b1 bbb.', 'b2 bbb.'], ['c1 ccc.']]
    c = seam_counts(win)
    assert (c['ns'], c['np'], c['sent_rev'], c['para_rev']) == (6, 3, 5, 2), c
    # ばらばらの数え方を、y_orders の文字列から直接数えたものと比べる
    sents = [s for p in win for s in p]
    od = y_orders(win)
    for k in [1, 2]:
        got = od[f'shuf{k}'].split(' ')
        toks = [' '.join(got[i:i + 2]) for i in range(0, len(got), 2)]   # 文は 2 語ずつ
        pos = [sents.index(t) for t in toks]
        kept = sum(1 for i in range(len(pos) - 1) if pos[i + 1] == pos[i] + 1)
        assert c[f'shuf{k}'] == len(sents) - 1 - kept, (k, c, pos)
    # 解析：σ ＝ 4 × 境目 なら、値段の比は 1、当てはめの c は 4、a は 0
    rng = np.random.default_rng(0)
    counts = []
    for _ in range(30):
        w = [['x'] * int(rng.integers(1, 5)) for _ in range(int(rng.integers(2, 7)))]
        counts.append(seam_counts(w))
    lp = np.zeros((30, 5))
    for i, cc in enumerate(counts):
        lp[i, 0] = -100.0
        for j, k in enumerate(SEAM_ORDERS[1:], 1):
            lp[i, j] = -100.0 - 4.0 * cc[k]
    lp[:, 3:] -= 1e-9                                          # 境目 0 のばらばらで 0/0 にならないように（実データでは起きにくい）
    ok = [i for i, cc in enumerate(counts) if cc['shuf1'] > 0 and cc['shuf2'] > 0]
    r = analyze_model(lp[ok], [counts[i] for i in ok])
    assert abs(r['para_over_sent_cost_quartiles'][1] - 1) < 1e-6 and abs(r['fit_origin']['c'] - 4) < 1e-6, r
    assert abs(r['fit_intercept']['a']) < 1e-6 and abs(r['fit_intercept']['c'] - 4) < 1e-6
    assert word([1.0, 0.9, 1.2, 0.7]) == '境目の数でだいたい説明できる' and word([0.3, 0.5, 0.6, 0.2]).startswith('段落')
    assert word([0.3, 1.0, 0.6, 0.2]) == 'はっきりしない'
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    return 'OK'
