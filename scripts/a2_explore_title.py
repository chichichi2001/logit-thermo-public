"""探索 Y の続き 2：タイトルは「前提」を担っているか、離れた文どうしのつながりはあるか（2026-10-28 夜）。確かめには使わない。
GPU（L4）のノートブックで、scripts/a2_2_collect.py と scripts/a2_explore_xy.py を実行したのと同じ名前空間で実行する
（load_docs・load_model・valid_mask・y_window・y_orders・BBC_MONTHS を使う）。生成はしない（人の文章を読むだけ）。

材料：探索 X・Y と同じ BBC の 40 本の窓（38 本）、同じ 4 モデル。
タイトルの 3 条件（本文の前に置く書き出し。書き出しと本文は別々にトークンに区切ってつなぐ。測るのは本文のトークンの log p の合計）：
  - none：「Article:\\n」（探索 Y と同じ）
  - match：「Title: <その記事のタイトル>\\nArticle:\\n」
  - mismatch：「Title: <別の記事のタイトル>\\nArticle:\\n」（38 本の中で 1 本ずつずらす。タイトルの書き方は同じ BBC のまま、中身だけ合わない）
並べ替え 7 つ（文の番号の並び）：
  - original、sent_rev、para_rev、shuf1、shuf2：探索 Y と同じ。
  - lead_rev：最初の文だけ先頭に残し、残りの文を逆順にする（壊れる隣の組の数は sent_rev と同じ「文の数 − 1」）。
  - para_lead：最初の段落だけ先頭に残し、残りの段落を逆順にする（段落の中の順番はそのまま。段落が 3 つ以上の記事だけ）。
各並べ替えについて数えるもの：
  - b：壊れた隣の組の数（元で隣り合っていた (i, i+1) が、同じ向きで隣に残らなかった数）。
  - K_long：逆向きになった「離れた組」の数（i < j かつ j − i ≥ 2 で、j が i より前に来た組の数）。
  - lead：最初の文が先頭にいないなら 1、いるなら 0。
戻れなさ σ ＝ log p（original）− log p（並べ替え）（同じタイトル条件の中で引く）。

読み方（回す前に決める。探索の目安。95% 区間は記事を選び直すブートストラップ 2,000 回、種 20261031）：
  R1 合うタイトルは書き出しの損を減らすか：書き出しの損 L ＝ σ(sent_rev) − σ(lead_rev)（記事ごと。最初の文を先頭からどかした損で、
     最初の文と、離れた後ろの文とのつながりの分も含む）。
     L(none) − L(match) の平均の 95% 区間が 4 モデルとも 0 より上 →「合うタイトルで書き出しの損が減る」
     4 モデルとも 0 を含むか下 →「合うタイトルでも書き出しの損は減らない」　それ以外 →「はっきりしない」
  R2 無関係なタイトルはどう効くか：L(mismatch) − L(none) の平均の 95% 区間で、
     4 モデルとも 0 を含む →「無関係なタイトルは書き出しの損を変えない」
     4 モデルとも 0 より下 →「無関係なタイトルでも書き出しの損が減る（タイトルがあるという形だけで効く）」
     4 モデルとも 0 より上 →「無関係なタイトルで書き出しの損が増える」　それ以外 →「はっきりしない」
  R3 書き出しを残しても、段落の境目は高いか（none）：記事ごとの [σ(para_lead) ÷ 段落の数 − 1] ÷ [σ(lead_rev) ÷ 文の数 − 1]
     の中央値（段落が 3 つ以上の記事）。前回と同じ線引き：4 モデルとも 0.67〜1.5 →「書き出しを残すと、境目の数でだいたい説明できる」、
     4 モデルとも 1.5 より大きい →「書き出しを残しても、段落の境目は高い」、4 モデルとも 0.67 未満 →「書き出しを残すと、段落の境目は安い」、
     それ以外 →「はっきりしない」
  R4 離れた文どうしのつながりはあるか（none）：6 つの並べ替えをまとめて σ ＝ A × lead ＋ c_adj × b ＋ c_long × K_long を当てはめる
     （切片なし）。c_long の 95% 区間が 4 モデルとも 0 より上 →「離れたつながりがある」、4 モデルとも 0 を含む →「離れたつながりは見えない」、
     それ以外 →「はっきりしない」。A・c_adj も記録する。
  参考 R5：前回の当てはめ σ ＝ a ＋ c × b（none、sent_rev・para_rev・shuf1・shuf2）の a の 95% 区間。OLMo の a（前回ほぼ 0）が、
     ほかの 3 モデルの a の点の値を区間に含めば「OLMo の a はぶれで説明できる」。
  参考：none の log p が探索 Y の値（xy_{モデル}.npz）とどれだけずれるか（同じ計算の再現の確認。記録だけ）。
タイトルの中身は表示しない・保存しない（トークンの数だけ記録する）。
"""
import gc, json, math, time
import numpy as np
import torch

T_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'),
            ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B')]
T_CONDS = ['none', 'match', 'mismatch']
T_ORDERS = ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2', 'lead_rev', 'para_lead']
T_BOOT, T_BOOT_SEED = 2000, 20261031
T_LOW, T_HIGH = 0.67, 1.5


def load_titles(docs):
    from datasets import load_dataset
    want = {d['link'] for d in docs}
    got = {}
    for m in BBC_MONTHS:
        ds = load_dataset('RealTimeData/bbc_news_alltime', m, split='train')
        assert 'title' in ds.column_names, ds.column_names
        for row in ds:
            if row['link'] in want and row['link'] not in got:
                got[row['link']] = (row['title'] or '').strip()
    titles = [got.get(d['link'], '') for d in docs]
    assert all(titles), 'タイトルが見つからない記事があります'
    return titles


def order_indices(win):
    """段落ごとの文のリスト → 並べ替えごとの文の番号の並び（para_lead は段落が 3 つ未満なら None）。"""
    sizes = [len(p) for p in win]
    ns = sum(sizes)
    starts = np.cumsum([0] + sizes[:-1])
    paras = [list(range(s, s + n)) for s, n in zip(starts, sizes)]
    out = {'original': list(range(ns)), 'sent_rev': list(range(ns))[::-1],
           'para_rev': [i for p in paras[::-1] for i in p]}
    for k in [1, 2]:
        out[f'shuf{k}'] = [int(i) for i in np.random.default_rng(k).permutation(ns)]   # y_orders と同じ作り方
    out['lead_rev'] = [0] + list(range(1, ns))[::-1]
    out['para_lead'] = paras[0] + [i for p in paras[1:][::-1] for i in p] if len(paras) >= 3 else None
    return out


def order_stats(perm):
    """文の番号の並び → (b、K_long、lead)。"""
    pos = np.empty(len(perm), int); pos[np.asarray(perm)] = np.arange(len(perm))
    n = len(perm)
    b = int(sum(1 for i in range(n - 1) if pos[i + 1] != pos[i] + 1))
    K_long = int(sum(1 for i in range(n) for j in range(i + 2, n) if pos[j] < pos[i]))
    return b, K_long, int(perm[0] != 0)


@torch.no_grad()
def body_logprob(model, tok, prefix, text, valid, device):
    pre = tok(prefix, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
    ids = pre + tok(text, add_special_tokens=False).input_ids
    x = torch.tensor([bos + ids], device=device)
    z = model(x).logits[0].double().masked_fill(~valid, float('-inf'))
    lp = torch.log_softmax(z, -1)
    start = len(bos) + len(pre)
    y = x[0, start:]
    tgt = lp[start - 1:x.shape[1] - 1].gather(-1, y[:, None])[:, 0]
    return float(tgt.sum()), int(len(y)), int(len(pre))


def prefixes(titles, j):
    n = len(titles)
    return {'none': 'Article:\n', 'match': f'Title: {titles[j]}\nArticle:\n', 'mismatch': f'Title: {titles[(j + 1) % n]}\nArticle:\n'}


def boot_ci(v, rng_seed=T_BOOT_SEED, B=T_BOOT):
    v = np.asarray(v, float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def fit_lsq(rows, cols):
    """rows：記事ごとの [(σ, 説明変数...)] のリスト。cols：使う説明変数の数。→ 係数（切片なし）。"""
    X = np.array([r[1:1 + cols] for doc in rows for r in doc], float)
    y = np.array([r[0] for doc in rows for r in doc], float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def boot_fit(rows, cols, B=T_BOOT, seed=T_BOOT_SEED):
    est = fit_lsq(rows, cols)
    rng = np.random.default_rng(seed)
    bs = np.array([fit_lsq([rows[i] for i in rng.integers(0, len(rows), len(rows))], cols) for _ in range(B)])
    return [[float(est[k]), float(np.percentile(bs[:, k], 2.5)), float(np.percentile(bs[:, k], 97.5))] for k in range(cols)]


def ci_word(cis, pos, zero, neg, other='はっきりしない'):
    if all(lo > 0 for _, lo, _ in cis):
        return pos
    if all(lo <= 0 <= hi for _, lo, hi in cis):
        return zero
    if neg is not None and all(hi < 0 for _, _, hi in cis):
        return neg
    return other


def ratio_word(meds):
    if all(T_LOW <= v <= T_HIGH for v in meds):
        return '書き出しを残すと、境目の数でだいたい説明できる'
    if all(v > T_HIGH for v in meds):
        return '書き出しを残しても、段落の境目は高い'
    if all(v < T_LOW for v in meds):
        return '書き出しを残すと、段落の境目は安い'
    return 'はっきりしない'


def analyze_title(LP, wins):
    """LP：{モデル: (記事, 3 条件, 7 並べ替え) の log p（para_lead がない記事は nan）}。"""
    idx = [order_indices(w) for w in wins]
    st = [{k: (order_stats(v) if v is not None else None) for k, v in o.items()} for o in idx]
    ns = np.array([len(o['original']) for o in idx]); npar = np.array([len(w) for w in wins])
    has_pl = npar >= 3
    oi = {k: i for i, k in enumerate(T_ORDERS)}
    res = {'n_docs': len(wins), 'n_docs_para_lead': int(has_pl.sum()), 'models': {}}
    for m, lp in LP.items():
        sig = lp[:, :, [0]] - lp                                    # (記事, 条件, 並べ替え)
        r = {}
        L = {c: sig[:, ci, oi['sent_rev']] - sig[:, ci, oi['lead_rev']] for ci, c in enumerate(T_CONDS)}
        r['lead_cost'] = {c: boot_ci(L[c]) for c in T_CONDS}
        r['lead_cost_none_minus_match'] = boot_ci(L['none'] - L['match'])
        r['lead_cost_mismatch_minus_none'] = boot_ci(L['mismatch'] - L['none'])
        r['sigma_mean'] = {c: {k: float(np.nanmean(sig[:, ci, oi[k]])) for k in T_ORDERS[1:]} for ci, c in enumerate(T_CONDS)}
        cp = sig[has_pl, 0, oi['para_lead']] / (npar[has_pl] - 1)
        cs = sig[has_pl, 0, oi['lead_rev']] / (ns[has_pl] - 1)
        r['lead_kept_para_over_sent_quartiles'] = [float(np.percentile(cp / cs, q)) for q in (25, 50, 75)]
        rows = []
        for j in range(len(wins)):
            doc = []
            for k in T_ORDERS[1:]:
                if st[j][k] is None:
                    continue
                b, K, ld = st[j][k]
                doc.append((sig[j, 0, oi[k]], ld, b, K))
            rows.append(doc)
        A, c_adj, c_long = boot_fit(rows, 3)
        r['fit_lead_adj_long'] = {'A': A, 'c_adj': c_adj, 'c_long': c_long}
        rows5 = [[(sig[j, 0, oi[k]], 1.0, st[j][k][0]) for k in ['sent_rev', 'para_rev', 'shuf1', 'shuf2']] for j in range(len(wins))]
        a5, c5 = boot_fit(rows5, 2)
        r['fit_a_plus_cb'] = {'a': a5, 'c': c5}
        res['models'][m] = r
    M = list(LP)
    res['R1'] = ci_word([res['models'][m]['lead_cost_none_minus_match'] for m in M],
                        '合うタイトルで書き出しの損が減る', '合うタイトルでも書き出しの損は減らない', '合うタイトルでも書き出しの損は減らない')
    res['R2'] = ci_word([res['models'][m]['lead_cost_mismatch_minus_none'] for m in M],
                        '無関係なタイトルで書き出しの損が増える', '無関係なタイトルは書き出しの損を変えない',
                        '無関係なタイトルでも書き出しの損が減る（タイトルがあるという形だけで効く）')
    res['R3'] = ratio_word([res['models'][m]['lead_kept_para_over_sent_quartiles'][1] for m in M])
    res['R4'] = ci_word([res['models'][m]['fit_lead_adj_long']['c_long'] for m in M],
                        '離れたつながりがある', '離れたつながりは見えない', None)
    if 'olmo2-1b' in LP:
        lo, hi = res['models']['olmo2-1b']['fit_a_plus_cb']['a'][1:]
        others = [res['models'][m]['fit_a_plus_cb']['a'][0] for m in M if m != 'olmo2-1b']
        res['R5'] = 'OLMo の a はぶれで説明できる' if all(lo <= v <= hi for v in others) else 'OLMo の a はぶれだけでは説明できない'
    res['counts'] = {'ns': ns.tolist(), 'np': npar.tolist(), 'para_sizes': [[len(p) for p in w] for w in wins],
                     'stats': [{k: v for k, v in s.items()} for s in st]}
    return res


def run_title(docs, outdir, device, notify=print):
    wins_all = [y_window(d['text']) for d in docs]
    keep = [i for i, w in enumerate(wins_all) if w is not None]
    wins = [wins_all[i] for i in keep]
    titles_all = load_titles(docs)
    titles = [titles_all[i] for i in keep]
    assert len(set(titles)) == len(titles), 'タイトルが重なっています（ずらしても合ってしまう）'
    LP, info = {}, {}
    for name, hf_id in T_MODELS:
        t0 = time.time()
        tok, model = load_model(hf_id, device)
        valid, _ = valid_mask(tok, model, device)
        lp = np.full((len(wins), len(T_CONDS), len(T_ORDERS)), np.nan)
        nt = np.zeros_like(lp); npre = np.zeros((len(wins), len(T_CONDS)))
        for j, w in enumerate(wins):
            sents = [s for p in w for s in p]
            oi = order_indices(w)
            yo = y_orders(w)
            for k in ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2']:
                assert ' '.join(sents[i] for i in oi[k]) == yo[k], ('並べ替えが探索 Y と違います', k)
            pre = prefixes(titles, j)
            for ci, c in enumerate(T_CONDS):
                for ki, k in enumerate(T_ORDERS):
                    if oi[k] is None:
                        continue
                    lp[j, ci, ki], nt[j, ci, ki], npre[j, ci] = body_logprob(model, tok, pre[c], ' '.join(sents[i] for i in oi[k]), valid, device)
        try:
            old = np.load(f'{outdir}/xy_{name}.npz')['y_logp']
            info[name] = {'max_abs_diff_vs_xy_none': float(np.max(np.abs(old - lp[:, 0, :5])))}
        except Exception as e:
            info[name] = {'max_abs_diff_vs_xy_none': None, 'note': type(e).__name__}
        info[name]['title_tokens_mean'] = float(npre[:, 1].mean() - npre[:, 0].mean())
        info[name]['seconds'] = round(time.time() - t0, 1)
        np.savez_compressed(f'{outdir}/title_{name}.npz', logp=lp, ntok=nt, npre=npre, conds=np.array(T_CONDS), orders=np.array(T_ORDERS))
        LP[name] = lp
        del model; gc.collect(); torch.cuda.empty_cache()
        notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')
    res = analyze_title(LP, wins)
    res['run_info'] = info
    return res


@torch.no_grad()
def self_test_title(device):
    # 並べ替えと数え方
    win = [['s0 aa.', 's1 aa.'], ['s2 aa.', 's3 aa.', 's4 aa.'], ['s5 aa.'], ['s6 aa.', 's7 aa.']]
    o = order_indices(win)
    assert o['lead_rev'] == [0, 7, 6, 5, 4, 3, 2, 1] and o['para_lead'] == [0, 1, 6, 7, 5, 2, 3, 4], o
    assert order_stats(o['original']) == (0, 0, 0)
    assert order_stats(o['sent_rev']) == (7, 21, 1)                  # 8 文：隣 7 組すべて、離れた組 28 − 7 ＝ 21
    assert order_stats(o['lead_rev'])[0] == 7 and order_stats(o['lead_rev'])[2] == 0
    b, K, ld = order_stats(o['para_rev'])
    assert b == 3 and ld == 1, (b, K, ld)
    sents = [s for p in win for s in p]; yo = y_orders(win)
    for k in ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2']:
        assert ' '.join(sents[i] for i in o[k]) == yo[k], k
    assert order_indices([['a'], ['b'], ['c', 'd']])['para_lead'] == [0, 2, 3, 1]
    assert order_indices([['a', 'b'], ['c', 'd']])['para_lead'] is None
    # K_long を素直な数え方と比べる
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = [int(i) for i in rng.permutation(9)]
        pos = {v: i for i, v in enumerate(p)}
        Kref = sum(1 for i in range(9) for j in range(i + 2, 9) if pos[j] < pos[i])
        assert order_stats(p)[1] == Kref
    # 当てはめ：σ ＝ 5 × lead ＋ 6 × b ＋ 0.3 × K_long を作って戻す
    wins = []
    for _ in range(25):
        wins.append([['x'] * int(rng.integers(1, 5)) for _ in range(int(rng.integers(3, 6)))])
    LP = {}
    for m in ['m1', 'm2']:
        lp = np.full((25, 3, 7), np.nan)
        for j, w in enumerate(wins):
            oi = order_indices(w)
            for ci in range(3):
                lp[j, ci, 0] = -500.0
                for ki, k in enumerate(T_ORDERS[1:], 1):
                    if oi[k] is None:
                        continue
                    b, K, ld = order_stats(oi[k])
                    lead_scale = {0: 1.0, 1: 0.4, 2: 1.0}[ci]                # match だけ書き出しの損が小さい
                    lp[j, ci, ki] = -500.0 - (5.0 * lead_scale * ld + 6.0 * b + 0.3 * K)
        LP[m] = lp
    r = analyze_title(LP, wins)
    f = r['models']['m1']['fit_lead_adj_long']
    assert abs(f['A'][0] - 5) < 1e-6 and abs(f['c_adj'][0] - 6) < 1e-6 and abs(f['c_long'][0] - 0.3) < 1e-6, f
    ns_ = np.array([sum(len(p) for p in w) for w in wins])
    # L は「最初の文を先頭からどかした損」：A ＋ 最初の文と離れた文の組（文の数 − 2 組）の分
    assert abs(r['models']['m1']['lead_cost']['none'][0] - (5 + 0.3 * (ns_ - 2).mean())) < 1e-6
    assert abs(r['models']['m1']['lead_cost_none_minus_match'][0] - 3) < 1e-6
    assert r['R1'] == '合うタイトルで書き出しの損が減る' and r['R4'] == '離れたつながりがある', (r['R1'], r['R4'])
    assert r['R2'] == '無関係なタイトルは書き出しの損を変えない', r['R2']
    assert ci_word([(1, -1, 2), (1, 0.5, 2)], 'P', 'Z', 'N') == 'はっきりしない'
    # body_logprob を素直な計算と比べる（小さなランダムのモデル）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    pre, text = 'Title: Something here\nArticle:\n', 'One sentence. Another one here.'
    lp1, n1, np1 = body_logprob(model, tok, pre, text, valid, device)
    ids = tok(pre, add_special_tokens=False).input_ids + tok(text, add_special_tokens=False).input_ids
    z = model(torch.tensor([ids], device=device)).logits[0].double().masked_fill(~valid, float('-inf'))
    ls = torch.log_softmax(z, -1)
    ref = sum(float(ls[i - 1, ids[i]]) for i in range(np1, len(ids)))
    assert abs(lp1 - ref) < 1e-6 * max(1, abs(ref)) and n1 == len(ids) - np1
    lp0, _, _ = body_logprob(model, tok, 'Article:\n', text, valid, device)
    lpy, _ = seq_logprob(model, tok, text, valid, device)            # 探索 Y の関数と同じ値になるか
    assert abs(lp0 - lpy) < 1e-9
    p = prefixes(['A', 'B', 'C'], 2)
    assert p['match'] == 'Title: C\nArticle:\n' and p['mismatch'] == 'Title: A\nArticle:\n'
    return 'OK'
