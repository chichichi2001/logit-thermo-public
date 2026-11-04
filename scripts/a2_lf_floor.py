"""A2-LF（ゲート docs/A2_LF_GATE.md）：格子の「遠い床」は本物か。データを開く前に固定する。GPU（L4）。
scripts/a2_2_collect.py・scripts/a2_explore_xy.py・scripts/a2_explore_lattice.py を実行したのと同じ名前空間で実行する
（load_docs・load_model・y_window・Y_PREFIX・token_logp・interaction_map・rep_tokens・sent_index を変えずに使う）。生成はしない。

探索 8.8 の格子と同じ材料（BBC の窓 38 本、本文は先頭から最大 384 トークン、同じ 4 モデル）で、差し替え先を 2 通りにする：
  よそ者：ほかの記事の本文に出てくるトークン（探索 8.8 と同じ選び方）
  身内：同じ記事の本文に出てくるトークン（差し替える位置のトークンそのものは除く）。話題も「記事らしさ」も保ったまま、
        その位置の中身だけを入れ替える。
差し替える位置 i は 4 つおき（計算を探索 8.8 の半分にするため）。j はすべての位置。
遠い床 FF ＝ 文の距離 k が 3〜12 の組の Δ の平均。近く S0 ＝ 同じ文（k ＝ 0）の組の Δ の平均。
分けられるもの：遠い床が「その位置のトークンの中身とのペアごとの結合」（身内に替えても残る）か、「場違いな言葉が入ったことへの
文章全体の反応」（身内に替えると消える。話題の乱れと記事らしさの乱れは、この実験では分けない）か。
"""
import gc, json, math, time
import numpy as np
import torch

LF_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'),
             ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B')]
LF_STRIDE, LF_SEED, LF_BOOT, LF_BOOT_SEED = 4, 20261105, 1000, 20261106
LF_KFAR = (3, 12)
LF_TYPES = ('foreign', 'same')


def rep_same(body, rng):
    """同じ記事の本文のトークンから、出てくる回数に比例して選ぶ。差し替える位置のトークンそのものは選ばない。"""
    body = np.asarray(body)
    out = np.empty(len(body), dtype=body.dtype)
    for i, t in enumerate(body):
        if np.all(body == t):
            out[i] = t                        # 全部同じトークンの記事（実際にはない）。そのまま
            continue
        while True:
            c = body[rng.integers(0, len(body))]
            if c != t:
                out[i] = c; break
    return out


def lf_profiles(D, sidx, body):
    """→ 文の距離 k（0..12）ごとの和と数、遠い床（k ＝ 3〜12）の和と数、そのうち「同じ言葉の繰り返し」（x_j ＝ 元の x_i）の和と数。"""
    n = D.shape[0]
    I, J = np.triu_indices(n, 1)
    v = D[I, J].astype(np.float64)
    ok = np.isfinite(v)
    k = sidx[J] - sidx[I]
    ksum, kcnt = np.zeros(13), np.zeros(13)
    m = ok & (k <= 12)
    np.add.at(ksum, k[m], v[m]); np.add.at(kcnt, k[m], 1)
    far = ok & (k >= LF_KFAR[0]) & (k <= LF_KFAR[1])
    body = np.asarray(body)
    rep = far & (body[J] == body[I])
    return ksum, kcnt, np.array([v[far].sum(), far.sum(), v[rep].sum(), rep.sum()])


def lf_stats(ksum, kcnt, farv, idx=None):
    """記事ごとの和と数 → FF・S0・繰り返しの割合。idx は記事の選び直し。"""
    if idx is not None:
        ksum, kcnt, farv = ksum[idx], kcnt[idx], farv[idx]
    ks, kc, fv = ksum.sum(0), kcnt.sum(0), farv.sum(0)
    FF = fv[0] / fv[1] if fv[1] > 0 else float('nan')
    S0 = ks[0] / kc[0] if kc[0] > 0 else float('nan')
    share = fv[2] / fv[0] if fv[0] != 0 else float('nan')
    return FF, S0, share


def lf_readings(per_model):
    """per_model：{名前: dict(R=..., R_ci=[lo, hi], share_f=..., share_f_ci=[lo, hi])}"""
    cis = [v['R_ci'] for v in per_model.values()]
    pts = [v['R'] for v in per_model.values()]
    if all(np.isfinite(p) and p <= 0.5 and hi < 0.7 for p, (lo, hi) in zip(pts, cis)):
        q1 = '遠くの床は場違いな言葉のときだけ出る（背景の場寄り）'
    elif all(lo >= 0.7 for lo, hi in cis):
        q1 = '遠くの床は身内の言葉でも残る（ペアごとの結合寄り）'
    else:
        q1 = 'はっきりしない'
    sh = [v['share_f_ci'] for v in per_model.values()]
    if all(lo > 0.5 for lo, hi in sh):
        q2 = '半分を超える'
    elif all(hi < 0.5 for lo, hi in sh):
        q2 = '半分より少ない'
    else:
        q2 = 'はっきりしない'
    return q1, q2


def lf_model_result(acc):
    """acc：{型: (ksum, kcnt, farv)}（それぞれ (記事, …)）→ 1 モデルの結果。R ＝ (FF 身内 ÷ FF よそ者) ÷ (S0 身内 ÷ S0 よそ者)。"""
    out = {}
    pt = {t: lf_stats(*acc[t]) for t in LF_TYPES}
    for t in LF_TYPES:
        out[t] = dict(FF=pt[t][0], S0=pt[t][1], share_repeat=pt[t][2], S_k=(acc[t][0].sum(0) / np.maximum(acc[t][1].sum(0), 1)).tolist())
    def ratio(p):
        qf = p['same'][0] / p['foreign'][0] if p['foreign'][0] != 0 else float('nan')
        qn = p['same'][1] / p['foreign'][1] if p['foreign'][1] != 0 else float('nan')
        return qf, qn, (qf / qn if qn not in (0, float('nan')) and np.isfinite(qn) else float('nan'))
    qf, qn, R = ratio(pt)
    nD = acc['foreign'][0].shape[0]
    rng = np.random.default_rng(LF_BOOT_SEED)
    Rb, Sb, FFsb = [], [], []
    for _ in range(LF_BOOT):
        idx = rng.integers(0, nD, nD)
        pb = {t: lf_stats(*acc[t], idx=idx) for t in LF_TYPES}
        Rb.append(ratio(pb)[2]); Sb.append(pb['foreign'][2]); FFsb.append(pb['same'][0])
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out.update(Q_far=qf, Q_near=qn, R=R, R_ci=q(Rb), share_f=pt['foreign'][2], share_f_ci=q(Sb), FF_same_ci=q(FFsb))
    return out


def run_lf(docs, outdir, device, notify=print):
    wins = [w for w in (y_window(d['text']) for d in docs) if w is not None]
    texts = [' '.join(s for p in w for s in p) for w in wins]
    sents_all = [[s for p in w for s in p] for w in wins]
    res = {'n_docs': len(wins), 'models': {}}
    for name, hf_id in LF_MODELS:
        t0 = time.time()
        tok, model = load_model(hf_id, device)
        pre = tok(Y_PREFIX, add_special_tokens=False).input_ids
        bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
        encs = [tok(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
        bodies = [e['input_ids'][:L_MAXTOK] for e in encs]
        sidx = [sent_index(t, s, e['offset_mapping'][:L_MAXTOK]) for t, s, e in zip(texts, sents_all, encs)]
        acc = {t: [np.zeros((len(wins), 13)), np.zeros((len(wins), 13)), np.zeros((len(wins), 4))] for t in LF_TYPES}
        for j, b in enumerate(bodies):
            for ti, t in enumerate(LF_TYPES):
                rng = np.random.default_rng(LF_SEED + 1000 * ti + j)
                rep = rep_tokens(bodies, j, len(b), rng) if t == 'foreign' else rep_same(b, rng)
                D = interaction_map(model, bos + pre + b, len(bos) + len(pre), rep, device, stride=LF_STRIDE)
                ks, kc, fv = lf_profiles(D, sidx[j], b)
                acc[t][0][j], acc[t][1][j], acc[t][2][j] = ks, kc, fv
        r = lf_model_result({t: tuple(acc[t]) for t in LF_TYPES})
        r['seconds'] = round(time.time() - t0, 1)
        res['models'][name] = r
        np.savez_compressed(f'{outdir}/lf_{name}.npz', **{f'{t}_{q}': acc[t][i] for t in LF_TYPES for i, q in enumerate(['ksum', 'kcnt', 'farv'])})
        del model; gc.collect(); torch.cuda.empty_cache()
        notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')
    res['LQ1'], res['LQ2'] = lf_readings(res['models'])
    return res


def self_test_lf():
    rng = np.random.default_rng(0)
    body = [5, 6, 7, 5, 8]
    r = rep_same(body, rng)
    assert all(a != b for a, b in zip(r, body)) and set(r.tolist()) <= set(body)
    # 文の距離と繰り返しの数え方
    D = np.full((5, 5), np.nan, np.float32)
    D[0, 1], D[0, 3], D[0, 4], D[1, 4] = 1.0, 2.0, 4.0, 8.0
    sidx = np.array([0, 0, 1, 3, 4])
    ks, kc, fv = lf_profiles(D, sidx, body)
    # (0,1) k=0, (0,3) k=3（body[3]=5=body[0] → 繰り返し）, (0,4) k=4, (1,4) k=4
    assert ks[0] == 1 and kc[0] == 1 and ks[3] == 2 and ks[4] == 12 and kc[4] == 2
    assert fv.tolist() == [14.0, 3.0, 2.0, 1.0]
    # 判定：身内で遠くだけ消える → 偽物寄り。遠くも近くと同じ割合で弱まる → 本物寄り
    def acc_of(FF_f, FF_s, S0_f, S0_s, n=30, noise=0.0):
        out = {}
        for t, FF, S0 in (('foreign', FF_f, S0_f), ('same', FF_s, S0_s)):
            ks = np.zeros((n, 13)); kc = np.ones((n, 13)); fv = np.zeros((n, 4))
            ks[:, 0] = S0 + rng.normal(0, noise, n); fv[:, 0] = FF + rng.normal(0, noise * 0.01, n); fv[:, 1] = 1
            fv[:, 2] = 0.2 * fv[:, 0]; fv[:, 3] = 1
            out[t] = (ks, kc, fv)
        return out
    fake = lf_model_result(acc_of(0.003, 0.0003, 0.6, 0.5, noise=0.01))
    real = lf_model_result(acc_of(0.003, 0.0025, 0.6, 0.5, noise=0.01))
    assert lf_readings({m: fake for m in 'abcd'})[0] == '遠くの床は場違いな言葉のときだけ出る（背景の場寄り）', fake
    assert lf_readings({m: real for m in 'abcd'})[0] == '遠くの床は身内の言葉でも残る（ペアごとの結合寄り）', real
    assert lf_readings({'a': fake, 'b': real, 'c': real, 'd': real})[0] == 'はっきりしない'
    assert lf_readings({m: real for m in 'abcd'})[1] == '半分より少ない'
    return 'OK'
