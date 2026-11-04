"""A2-19 の材料集め：A2-18 の XQ1（h*）を、新しい BBC の記事 120 本で、もう一度だけ確かめる（2026-11-01 20:xx JST、GPU（L4））。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py・a2_gpre_collect.py・
a2_18_collect.py を実行したあとの同じ名前空間で実行する（clean・BBC_MONTHS・load_model・valid_mask・split_doc・CONT・generate_dsc2・GP_S・GP_CHUNK・gp_seed・GP_STALE・
A18_HF・A18_JOBS を使う）。

文章：A2-2 と同じ月（2025-04〜06）の BBC の記事で、A2-2 の 40 本（G-pre・A2-16・A2-18 で使った記事）を除いたものから、120 本を種で選ぶ（clean をかけて 4,000 字以上、
選び方は A2-2 の load_docs と同じ形：リンクの sha256 で並べて、種 20261191 で選ぶ）。40 本ずつの 3 つの組に分ける。
腕：c0（h ＝ 0 の対照）と h1（h ＝ h*）。h* と温度は A2-18 と同じ（A18_JOBS をそのまま使う）。2h* の腕は回さない。
仕事：7 モデル × 2 つの腕 × 3 つの組 ＝ 42 個。乱数の種は gp_seed に 100000 ×（組の番号 ＋ 1）を足して、A2-18 と重ならないようにする。
"""
import gc, os, time, json, hashlib, socket, uuid
import numpy as np
import torch

A19_N, A19_GROUPS, A19_SEED = 120, 3, 20261191
A19_ARMS = ('c0', 'h1')


def a19_jobs():
    J = {(n, a): (h, t) for n, a, h, t in A18_JOBS}
    return [(n, a, J[(n, a)][0], J[(n, a)][1], g) for n in A18_HF for a in A19_ARMS for g in range(A19_GROUPS)]


def load_docs19(old_meta_path):
    """A2-2 の 40 本を除いた同じ月の記事から 120 本 → 3 つの組（40 本ずつ）と目録。"""
    from datasets import load_dataset
    old = {d['link_sha256'] for d in json.load(open(old_meta_path))['docs']}
    assert len(old) == 40
    pool = []
    for m in BBC_MONTHS:
        ds = load_dataset('RealTimeData/bbc_news_alltime', m, split='train')
        for row in ds:
            c = clean(row['content'] or '')
            if len(c) >= 4000:
                pool.append({'month': m, 'link': row['link'], 'text': c})
    n_all = len(pool)
    seen_l, seen_t, uniq = set(), set(), []                              # 同じリンク・同じ本文は 1 本にする（最初のものを残す）
    for d in pool:
        ls, ts = hashlib.sha256(d['link'].encode()).hexdigest(), hashlib.sha256(d['text'].encode()).hexdigest()
        if ls in seen_l or ts in seen_t:
            continue
        seen_l.add(ls); seen_t.add(ts); uniq.append(d)
    n_uniq = len(uniq)
    pool = [d for d in uniq if hashlib.sha256(d['link'].encode()).hexdigest() not in old]
    assert n_uniq - len(pool) == 40, ('A2-2 の 40 本が見つからない', n_all, n_uniq, len(pool))
    pool.sort(key=lambda d: hashlib.sha256(d['link'].encode()).hexdigest())
    rng = np.random.default_rng(A19_SEED)
    idx = sorted(rng.choice(len(pool), size=A19_N, replace=False))
    docs = [pool[i] for i in idx]
    assert len({d['link'] for d in docs}) == A19_N and len({d['text'] for d in docs}) == A19_N
    groups = [docs[g * 40:(g + 1) * 40] for g in range(A19_GROUPS)]
    meta = {'pool_size_all': n_all, 'pool_size_unique': n_uniq, 'pool_size_new': len(pool), 'groups': [[{'month': d['month'], 'link_sha256': hashlib.sha256(d['link'].encode()).hexdigest(),
                                                                               'text_sha256': hashlib.sha256(d['text'].encode()).hexdigest(), 'chars': len(d['text'])}
                                                                              for d in grp] for grp in groups]}
    return groups, meta


A19_TOKEN = uuid.uuid4().hex                                         # このランタイム（プロセス）だけの印


def a19_claim(outdir, name, tag):
    """gp_claim と同じ決まり（終わっていない・失敗の印がない・誰も取っていないか 2 時間より古い → 取る）。
    取った印の照合に、ランタイムの番号ではなく、このプロセスだけの印（uuid）を使う（同じノートブックを 2 つのランタイムで開いても重ならないように）。"""
    if os.path.exists(f'{outdir}/{name}@{tag}.npz') or os.path.exists(f'{outdir}/locks/{name}@{tag}.failed'):
        return False
    lk = f'{outdir}/locks/{name}@{tag}.lock'
    if os.path.exists(lk):
        try:
            age = time.time() - json.load(open(lk))['t']
        except Exception:
            age = 0.0
        if age < GP_STALE:
            return False
    os.makedirs(f'{outdir}/locks', exist_ok=True)
    json.dump(dict(token=A19_TOKEN, host=socket.gethostname(), t=time.time()), open(lk, 'w'))
    time.sleep(20)
    try:
        return json.load(open(lk)).get('token') == A19_TOKEN
    except Exception:
        return False


def a19_seed(k, s, c, g):
    return gp_seed(k, s, c) + 100000 * (g + 1)


def run_a19_job(name, arm, h, temps, g, groups, outdir, device, cache):
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(A18_HF[name], device)
        assert next(model.parameters()).dtype == torch.float32
        valid, eos = valid_mask(tok, model, device)
        cache.update(name=name, tok=tok, model=model, valid=valid)
    tok, model, valid = cache['tok'], cache['model'], cache['valid']
    pairs = [split_doc(tok, d['text']) for d in groups[g]]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert len(use) == 40 and len({len(pairs[i][0]) for i in use}) == 1, (name, g, '使える記事が 40 本ない', len(use))
    maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
    assert maxpos is None or len(pairs[use[0]][0]) + CONT <= maxpos, ('文が長すぎます', maxpos)
    prompts = [pairs[i][0] for i in use]
    nD = len(use)
    chunks = [list(range(i, min(nD, i + GP_CHUNK))) for i in range(0, nD, GP_CHUNK)]
    ta = time.time()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    R = {k: np.zeros((len(temps), GP_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(temps):
        Ts, Hs = np.full(CONT, T), np.full(CONT, h)
        for s in range(GP_S):
            for c, ii in enumerate(chunks):
                gen = torch.Generator(device=device).manual_seed(a19_seed(k, s, c, g))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, gen)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    path = f'{outdir}/{name}@{arm}_g{g}.npz'
    np.savez_compressed(path + '.part.npz', temps=np.array(temps), h=h, arm=arm, group=g, use=np.array(use), seconds=time.time() - ta, chunk=GP_CHUNK, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_a19(worker, nworkers, groups, outdir, device, notify=print):
    os.makedirs(outdir, exist_ok=True)
    jobs = a19_jobs()
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for name, arm, h, temps, g in order:
            tag = f'{arm}_g{g}'
            if a19_claim(outdir, name, tag):
                took = True
                try:
                    sec = run_a19_job(name, arm, h, temps, g, groups, outdir, device, cache)
                    n += 1
                    notify(f'{name}@{tag} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{name}@{tag}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {name}@{tag} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = [f'{nm}@{a}_g{g}' + ('（失敗）' if os.path.exists(f'{outdir}/locks/{nm}@{a}_g{g}.failed') else '')
            for nm, a, _, _, g in jobs if not os.path.exists(f'{outdir}/{nm}@{a}_g{g}.npz')]
    return n, left


def self_test_a19():
    jobs = a19_jobs()
    assert len(jobs) == 42 and len({(n, a, g) for n, a, _, _, g in jobs}) == 42
    J = {(n, a): (h, t) for n, a, h, t in A18_JOBS}
    for n, a, h, t, g in jobs:
        assert (h, t) == J[(n, a)] and (h == 0.0) == (a == 'c0')
    seeds = {a19_seed(k, s, c, g) for k in range(7) for s in range(GP_S) for c in range(2) for g in range(A19_GROUPS)}
    old = {gp_seed(k, s, c) for k in range(7) for s in range(GP_S) for c in range(2)}
    assert len(seeds) == 7 * GP_S * 2 * A19_GROUPS and not (seeds & old)
    return 'OK'
