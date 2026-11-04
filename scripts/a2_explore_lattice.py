"""探索 Y の続き 3：文章の「相互作用の地図」と、結合の距離による弱まり方（マーデルング定数の下見）。2026-10-28 夜。確かめには使わない。
GPU（L4）のノートブックで、scripts/a2_2_collect.py と scripts/a2_explore_xy.py を実行したのと同じ名前空間で実行する
（load_docs・load_model・y_window・Y_PREFIX を使う）。生成はしない（人の文章を読むだけ）。発想はオーナー（2026-10-28 22:31）。

材料：探索 Y と同じ BBC の窓（38 本）を元の順番のまま、「Article:\\n」のあとに置く。本文は先頭から最大 384 トークン。同じ 4 モデル。
差し替える位置 i は 1 つおき（偶数番目）にする（計算を半分にするため。j はすべての位置）。
相互作用の地図：本文の位置 i のトークンを、関係ないトークンに差し替えて読ませ直し、後ろの位置 j（j > i）の log p の下がり方
  Δ(i, j) ＝ log p(x_j | 元の文) − log p(x_j | i を差し替えた文) を測る（温度 1。差を取るので、外す番号の処理はしない。float32）。
  差し替え先：同じモデルのトークンで、ほかの記事の本文に出てくるトークン（その記事の本文に出てくるものは除く）から、出てくる回数に
  比例して選ぶ（ふつうの言葉のまま、中身だけ関係なくする）。選び方を 2 通り（種）用意し、主には 2 つの平均を使う。
距離の帯：トークンの距離 d ＝ j − i。帯（2、3、4〜5、6〜7、8〜11、12〜15、16〜23、24〜31、32〜47、48〜63、64〜95、96〜127）ごとに、
  全記事の Δ をまとめた平均 D(帯) を出す。文の距離 k（j の文の番号 − i の文の番号、0〜12）ごとの平均 S(k) も出す。

読み方（回す前に決める。探索の目安。95% 区間は記事を選び直すブートストラップ 1,000 回、種 20261101）：
  R1 弱まり方の形：帯の D が正のところで、べき（log D ＝ a − α log d）と指数（log D ＝ b − d ÷ ξ）を当てはめ、log D の残差の二乗和を比べる。
     4 モデルともべきのほうが小さい →「べき型（遠くまでゆっくり効く。イオン結晶型）」
     4 モデルとも指数のほうが小さい →「指数型（すぐ消える。分子性結晶型）」　それ以外 →「はっきりしない」
     D が正でない帯が 3 つ以上あるモデルがあれば →「はっきりしない（遠くで符号が定まらない）」
     α も記録する（1 以下なら、1 次元の並びでは足し算が収束しない形）。
  R2 近くと遠くのどちらが多いか：近く N ＝ Σ_{d=1..16} D(d)、遠く F ＝ Σ_{d=17..256} D(d)（d ごとの平均を足す）。F ÷ N の 95% 区間が
     4 モデルとも 1 より上 →「遠くの寄与が近くを上回る」、4 モデルとも 1 より下 →「近くが大半」、それ以外 →「はっきりしない」。
  参考：文の距離ごとの S(k)。S(1) ÷ S(0)（隣の文と同じ文の比）、S(2) ÷ S(1)。差し替えの 2 通りで距離の形がどれだけ似ているか（相関）。
保存するのは数だけ（記事ごとの距離の和と数、文の距離の和と数、地図そのものは float16 で）。中身は表示しない・保存しない。
"""
import gc, json, math, time
import numpy as np
import torch

L_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'),
            ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B')]
L_MAXTOK, L_BATCH, L_DRAWS, L_SEED, L_STRIDE = 384, 8, 2, 20261101, 2
L_BOOT_SEED_OFFSET = 0
L_BINS = [(2, 2), (3, 3), (4, 5), (6, 7), (8, 11), (12, 15), (16, 23), (24, 31), (32, 47), (48, 63), (64, 95), (96, 127)]
L_KMAX = 12
L_BOOT = 1000


@torch.no_grad()
def token_logp(model, X, start):
    """X：(B, L) のトークン。→ 位置 start..L−1 のトークンの log p、(B, L − start)、float32。"""
    logits = model(X).logits[:, start - 1:X.shape[1] - 1].float()
    lse = torch.logsumexp(logits, -1)
    tgt = logits.gather(-1, X[:, start:, None])[:, :, 0]
    out = (tgt - lse).cpu().numpy()
    del logits, lse, tgt
    return out


@torch.no_grad()
def interaction_map(model, ids, start, rep, device, bs=L_BATCH, stride=L_STRIDE):
    """ids：bos ＋ 書き出し ＋ 本文 のトークン列。本文は位置 start から n 個。rep：(n,) の差し替え先。→ Δ (n, n)、j ≤ i は nan。
    差し替える位置 i は stride おき（計算を軽くするため。測らない行は nan のまま）。"""
    X0 = torch.tensor([ids], device=device)
    n = len(ids) - start
    base = token_logp(model, X0, start)[0]
    D = np.full((n, n), np.nan, np.float32)
    pos = list(range(0, n, stride))
    for a in range(0, len(pos), bs):
        rows = pos[a:a + bs]
        X = X0.repeat(len(rows), 1)
        for r, i in enumerate(rows):
            X[r, start + i] = int(rep[i])
        lp = token_logp(model, X, start)
        for r, i in enumerate(rows):
            d = base - lp[r]
            d[:i + 1] = np.nan
            D[i] = d
    return D


def rep_tokens(body_ids_all, j, n, rng):
    own = set(body_ids_all[j])
    pool = [t for k, b in enumerate(body_ids_all) if k != j for t in b if t not in own]
    return np.array(rng.choice(pool, size=n, replace=True))


def sent_index(body_text, sents, offsets):
    starts, pos = [], 0
    for s in sents:
        starts.append(pos); pos += len(s) + 1
    starts = np.array(starts)
    return np.array([int(np.searchsorted(starts, a, side='right') - 1) for a, _ in offsets])


def doc_profiles(D, sidx):
    """→ 距離 d（0..n−1）ごとの Δ の和と数、文の距離 k（0..L_KMAX）ごとの和と数。"""
    n = D.shape[0]
    dsum, dcnt = np.zeros(n), np.zeros(n)
    for d in range(1, n):
        v = np.diagonal(D, offset=d).astype(np.float64)
        v = v[np.isfinite(v)]
        dsum[d], dcnt[d] = v.sum(), len(v)
    ksum, kcnt = np.zeros(L_KMAX + 1), np.zeros(L_KMAX + 1)
    I, J = np.triu_indices(n, 1)
    k = sidx[J] - sidx[I]
    ok = (k <= L_KMAX) & np.isfinite(D[I, J])
    np.add.at(ksum, k[ok], D[I[ok], J[ok]].astype(np.float64))
    np.add.at(kcnt, k[ok], 1)
    return dsum, dcnt, ksum, kcnt


def pooled(dsum, dcnt, docs=None):
    """記事ごとの和と数（(記事, 長さ)）→ d ごとの平均（長さ）。"""
    if docs is not None:
        dsum, dcnt = dsum[docs], dcnt[docs]
    s, c = dsum.sum(0), dcnt.sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(c > 0, s / c, np.nan)


def bin_profile(Dd):
    out = []
    for a, b in L_BINS:
        v = Dd[a:b + 1]
        out.append(float(np.nanmean(v)) if np.any(np.isfinite(v)) else float('nan'))
    return np.array(out)


def fit_forms(Db):
    """帯の平均 → べきと指数の当てはめ。D が正の帯だけ使う。"""
    centers = np.array([math.sqrt(a * b) for a, b in L_BINS])
    ok = np.isfinite(Db) & (Db > 0)
    res = {'n_nonpos_bins': int((~ok).sum())}
    if ok.sum() < 4:
        res.update(alpha=float('nan'), xi=float('nan'), rss_power=float('nan'), rss_exp=float('nan'))
        return res
    y = np.log(Db[ok]); x1 = np.log(centers[ok]); x2 = centers[ok]
    A1 = np.vstack([np.ones_like(x1), x1]).T; A2 = np.vstack([np.ones_like(x2), x2]).T
    c1, r1, *_ = np.linalg.lstsq(A1, y, rcond=None); c2, r2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    rss1 = float(((y - A1 @ c1) ** 2).sum()); rss2 = float(((y - A2 @ c2) ** 2).sum())
    res.update(alpha=float(-c1[1]), xi=float(-1 / c2[1]) if c2[1] < 0 else float('inf'), rss_power=rss1, rss_exp=rss2)
    return res


def near_far(Dd):
    N = float(np.nansum(Dd[1:17])); F = float(np.nansum(Dd[17:257]))
    return N, F


def analyze_model_lattice(dsum, dcnt, ksum, kcnt):
    nD = dsum.shape[0]
    Dd = pooled(dsum, dcnt)
    Db = bin_profile(Dd)
    fit = fit_forms(Db)
    N, F = near_far(Dd)
    rng = np.random.default_rng(L_BOOT_SEED_OFFSET + L_SEED)
    ratios, alphas, pw = [], [], []
    for _ in range(L_BOOT):
        idx = rng.integers(0, nD, nD)
        Dd_b = pooled(dsum, dcnt, idx)
        n_b, f_b = near_far(Dd_b)
        ratios.append(f_b / n_b if n_b != 0 else float('nan'))
        fb = fit_forms(bin_profile(Dd_b))
        alphas.append(fb['alpha']); pw.append(fb['rss_power'] < fb['rss_exp'] if np.isfinite(fb['rss_power']) else np.nan)
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    S = np.where(kcnt.sum(0) > 0, ksum.sum(0) / np.maximum(kcnt.sum(0), 1), np.nan)
    return dict(D_bins=Db.tolist(), D_d1=float(Dd[1]), fit=fit, alpha_ci=q(alphas), power_wins_frac=float(np.nanmean(pw)),
                near=N, far=F, far_over_near=F / N if N != 0 else float('nan'), far_over_near_ci=q(ratios),
                S_k=S.tolist(), S1_over_S0=float(S[1] / S[0]), S2_over_S1=float(S[2] / S[1]))


def readings(models):
    M = list(models)
    fits = [models[m]['fit'] for m in M]
    if any(f['n_nonpos_bins'] >= 3 for f in fits):
        r1 = 'はっきりしない（遠くで符号が定まらない）'
    elif all(f['rss_power'] < f['rss_exp'] for f in fits):
        r1 = 'べき型（遠くまでゆっくり効く。イオン結晶型）'
    elif all(f['rss_exp'] < f['rss_power'] for f in fits):
        r1 = '指数型（すぐ消える。分子性結晶型）'
    else:
        r1 = 'はっきりしない'
    cis = [models[m]['far_over_near_ci'] for m in M]
    if all(lo > 1 for lo, _ in cis):
        r2 = '遠くの寄与が近くを上回る'
    elif all(hi < 1 for _, hi in cis):
        r2 = '近くが大半'
    else:
        r2 = 'はっきりしない'
    return r1, r2


def run_lattice(docs, outdir, device, notify=print):
    wins = [w for w in (y_window(d['text']) for d in docs) if w is not None]
    texts = [' '.join(s for p in w for s in p) for w in wins]
    sents_all = [[s for p in w for s in p] for w in wins]
    res = {'n_docs': len(wins), 'models': {}, 'draw_corr': {}}
    for name, hf_id in L_MODELS:
        t0 = time.time()
        tok, model = load_model(hf_id, device)
        pre = tok(Y_PREFIX, add_special_tokens=False).input_ids
        bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
        encs = [tok(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
        bodies = [e['input_ids'][:L_MAXTOK] for e in encs]
        sidx = [sent_index(t, s, e['offset_mapping'][:L_MAXTOK]) for t, s, e in zip(texts, sents_all, encs)]
        n_max = max(len(b) for b in bodies)
        dsum = np.zeros((L_DRAWS, len(wins), n_max)); dcnt = np.zeros_like(dsum)
        ksum = np.zeros((L_DRAWS, len(wins), L_KMAX + 1)); kcnt = np.zeros_like(ksum)
        maps = []
        for j, b in enumerate(bodies):
            Ds = []
            for g in range(L_DRAWS):
                rng = np.random.default_rng(L_SEED + 1000 * g + j)
                rep = rep_tokens(bodies, j, len(b), rng)
                D = interaction_map(model, bos + pre + b, len(bos) + len(pre), rep, device)
                a1, a2, a3, a4 = doc_profiles(D, sidx[j])
                dsum[g, j, :len(a1)], dcnt[g, j, :len(a2)], ksum[g, j], kcnt[g, j] = a1, a2, a3, a4
                Ds.append(D)
            maps.append(np.nanmean(np.stack(Ds), 0).astype(np.float16))
        # 主：2 通りの平均（和と数を足し合わせる）
        r = analyze_model_lattice(dsum.sum(0), dcnt.sum(0), ksum.sum(0), kcnt.sum(0))
        p0, p1 = bin_profile(pooled(dsum[0], dcnt[0])), bin_profile(pooled(dsum[1], dcnt[1]))
        res['draw_corr'][name] = float(np.corrcoef(p0, p1)[0, 1])
        r['seconds'] = round(time.time() - t0, 1)
        res['models'][name] = r
        np.savez_compressed(f'{outdir}/lattice_{name}.npz', dsum=dsum, dcnt=dcnt, ksum=ksum, kcnt=kcnt,
                            ntok=np.array([len(b) for b in bodies]), **{f'map{j}': m for j, m in enumerate(maps)})
        del model; gc.collect(); torch.cuda.empty_cache()
        notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')
    res['R1'], res['R2'] = readings(res['models'])
    return res


@torch.no_grad()
def self_test_lattice(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    # token_logp を素直な計算と比べる
    ids = list(range(10, 40)); start = 5
    lp = token_logp(model, torch.tensor([ids], device=device), start)[0]
    z = model(torch.tensor([ids], device=device)).logits[0].double()
    ls = torch.log_softmax(z, -1)
    ref = np.array([float(ls[t - 1, ids[t]]) for t in range(start, len(ids))])
    assert np.allclose(lp, ref, atol=1e-4), (lp[:3], ref[:3])
    # 地図：差し替えた位置より前は変わらない（向きの確認）。そのため一時的に nan を入れない版を計算する
    rep = np.array([3] * (len(ids) - start))
    X0 = torch.tensor([ids], device=device); base = token_logp(model, X0, start)[0]
    for i in [0, 7, 20]:
        X = X0.clone(); X[0, start + i] = 3
        d = base - token_logp(model, X, start)[0]
        assert np.all(np.abs(d[:i]) < 1e-5), ('前の位置が変わった', i, d[:i])
    D = interaction_map(model, ids, start, rep, device, bs=4, stride=1)
    assert np.all(np.isnan(D[np.tril_indices(len(rep))])), '対角とその下は nan のはず'
    X = X0.clone(); X[0, start + 7] = 3
    assert abs(D[7, 12] - (base[12] - token_logp(model, X, start)[0][12])) < 1e-5
    D2 = interaction_map(model, ids, start, rep, device, bs=4, stride=2)
    assert np.all(np.isnan(D2[1])) and abs(D2[6, 12] - D[6, 12]) < 1e-5            # 1 つおきの行だけ測る
    s2, c2, _, _ = doc_profiles(D2, np.zeros(len(rep), int))
    assert np.all(np.isfinite(s2)) and c2[1] == len(range(0, len(rep) - 1, 2))
    # 文の番号
    sents = ['Aa bb.', 'Cc dd ee.', 'Ff.']
    text = ' '.join(sents)
    offs = [(0, 2), (3, 6), (7, 9), (10, 12), (13, 16), (17, 20)]
    assert sent_index(text, sents, offs).tolist() == [0, 0, 1, 1, 1, 2]
    # 距離の和と文の距離の和
    Dm = np.full((4, 4), np.nan, np.float32)
    Dm[0, 1], Dm[0, 2], Dm[0, 3], Dm[1, 2], Dm[1, 3], Dm[2, 3] = 1, 2, 3, 4, 5, 6
    s, c, ks, kc = doc_profiles(Dm, np.array([0, 0, 1, 1]))
    assert s.tolist() == [0, 1 + 4 + 6, 2 + 5, 3] and c.tolist() == [0, 3, 2, 1]
    assert ks[0] == 1 + 6 and kc[0] == 2 and ks[1] == 2 + 3 + 4 + 5 and kc[1] == 4
    # 当てはめ：べきの形と指数の形を見分ける
    d = np.arange(0, 300, dtype=float)
    pw = np.where(d > 0, 2.0 * np.maximum(d, 1) ** -0.8, np.nan)
    ex = np.where(d > 0, 2.0 * np.exp(-d / 6.0), np.nan)
    fp, fe = fit_forms(bin_profile(pw)), fit_forms(bin_profile(ex))
    assert fp['rss_power'] < fp['rss_exp'] and abs(fp['alpha'] - 0.8) < 0.05, fp
    assert fe['rss_exp'] < fe['rss_power'], fe
    # 判定の言葉
    mk = lambda f, ci: {'fit': f, 'far_over_near_ci': ci}
    m = {k: mk(fp, [1.2, 2.0]) for k in 'abcd'}
    assert readings(m) == ('べき型（遠くまでゆっくり効く。イオン結晶型）', '遠くの寄与が近くを上回る')
    m = {k: mk(fe, [0.1, 0.5]) for k in 'abcd'}
    assert readings(m) == ('指数型（すぐ消える。分子性結晶型）', '近くが大半')
    # 差し替え先は、その記事に出てこないトークン
    bodies = [[1, 2, 3], [3, 4, 5], [6, 7]]
    r = rep_tokens(bodies, 0, 50, np.random.default_rng(0))
    assert not (set(r.tolist()) & {1, 2, 3}) and set(r.tolist()) <= {4, 5, 6, 7}
    return 'OK'
