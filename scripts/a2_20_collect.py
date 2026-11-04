"""A2-20 の材料集め：7B 級の 2 本（Qwen2.5-7B・Mistral-7B-v0.3）で、A2-18・A2-19 の予言（罰をかけていない出力だけから h* を決める）を確かめる
（2026-11-02 JST、GPU（A100 40GB 以上、fp32、TF32 なし））。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py・a2_gpre_collect.py・
a2_18_collect.py・a2_19_collect.py を実行したあとの同じ名前空間で実行する（load_model・valid_mask・split_doc・CONT・generate_dsc2・GP_S・GP_CHUNK・gp_seed・
GP_STALE・load_docs19・a19_claim・a19_seed を使う）。

第 1 段（s1）：A2-2 の BBC の記事 40 本（G-pre と同じ記事）、h ＝ 0、温度 0.20〜0.90 の 15 個（0.05 おき）、サンプル 4、512 歩、q つき。
  → scripts/a2_20_predict.py が、ここから h* と第 2 段の温度を決める（CPU、ゲート §1）。
第 2 段（s2）：A2-19 の新しい記事 120 本（3 つの組）、腕 c0（h ＝ 0）と h1（h ＝ h*）、温度は予言した T* のまわりの 7 つ。
  仕事の表は results/A2_20/hstar7b.json（予言の出力）から作る。本命が予言できないときだけ、results/A2_20/hstar7b_backup.json の予備で枠を埋める。
  種は A2-19 と同じ a19_seed（組ごと）。生成で落ちた仕事（メモリ切れなど）は予備に替えず、同じ仕事を回し直す。
記事が前の実験と同じかを、目録（本文の sha256 の並び）で確かめる（check_docs20、ノートブックで呼ぶ）。npz に GPU・版・モデルのコミットを残す。
モデルは fp32 で読む（load_model のまま）。読む前に GPU のメモリが 39GB 以上あるかを確かめ、読んだあとに重みが fp32 かを確かめる。
"""
import gc, os, time, json
import numpy as np
import torch

A20_HF = {'qwen2.5-7b': 'Qwen/Qwen2.5-7B', 'mistral-7b-v0.3': 'mistralai/Mistral-7B-v0.3'}
A20_BACKUP = {'granite-3.1-8b': 'ibm-granite/granite-3.1-8b-base'}     # 予言できないモデルが出たときだけ（ゲート §2）
A20_T1 = [round(0.20 + 0.05 * i, 2) for i in range(15)]
A20_SEED1 = 20261201
A20_MIN_GPU = 39e9


def a20_seed1(k, s, c):
    """第 1 段の種（温度の番号 k、サンプル s、記事のかたまり c）。G-pre・A2-18・A2-19 の種とは重ならない。"""
    return A20_SEED1 + 1000 * k + 10 * s + c


def a20_jobs1(names):
    return [(n, 's1', 0.0, A20_T1, None) for n in names]


def a20_models2(paths):
    """予言の出力（本命のファイルと、あれば予備のファイル）→ 第 2 段に使うモデル {名前: 予言}。
    予備（Granite）は、本命の 2 本のうち「予言できない」ものがあるときだけ、その枠に入る（ゲート §1）。"""
    P = json.load(open(paths[0]))['models']
    use = {n: r for n, r in P.items() if r.get('ok')}
    if len(paths) > 1 and len(use) < len(P):
        B = json.load(open(paths[1]))['models']
        for n, r in B.items():
            if r.get('ok') and len(use) < len(P):
                use[n] = r
    return use


def a20_jobs2(paths):
    """予言の出力 → 第 2 段の仕事（モデル × 腕 c0・h1 × 組 3）。予言できなかったモデルは入れない。"""
    paths = [paths] if isinstance(paths, str) else list(paths)
    jobs = []
    for n, r in a20_models2(paths).items():
        if not r.get('ok'):
            continue
        h, grid = round(float(r['target']['hstar']), 4), [float(t) for t in r['target']['grid']]
        for arm, hh in (('c0', 0.0), ('h1', h)):
            for g in range(3):
                jobs.append((n, arm, hh, grid, g))
    return jobs


def check_docs20(meta_now, meta_ref, key):
    """記事の目録（text_sha256 の並び）が、前の実験の目録と完全に同じかを確かめる。key は 'docs'（A2-2）か 'groups'（A2-19）。"""
    f = (lambda m: [d['text_sha256'] for d in m['docs']]) if key == 'docs' else (lambda m: [[d['text_sha256'] for d in g] for g in m['groups']])
    assert f(meta_now) == f(meta_ref), ('記事が前の実験と違います', key)
    return True


def _load20(name, device, cache):
    if cache.get('name') == name:
        return cache['tok'], cache['model'], cache['valid']
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    assert torch.cuda.get_device_properties(0).total_memory >= A20_MIN_GPU, ('GPU のメモリが 40GB に足りません', torch.cuda.get_device_properties(0).total_memory)
    assert 'A100' in torch.cuda.get_device_name(0), ('A100 ではありません', torch.cuda.get_device_name(0))
    hf = {**A20_HF, **A20_BACKUP}[name]
    tok, model = load_model(hf, device)
    assert all(p.dtype == torch.float32 for p in model.parameters()), (name, '重みが fp32 ではない')
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == 'highest'
    valid, eos = valid_mask(tok, model, device)
    cache.update(name=name, tok=tok, model=model, valid=valid)
    return tok, model, valid


def run_a20_job(job, docs, outdir, device, cache, chunk):
    """job ＝ (名前, 腕, h, 温度, 組)。docs は第 1 段なら 40 本、第 2 段なら組ごとの 40 本。"""
    name, arm, h, temps, g = job
    tok, model, valid = _load20(name, device, cache)
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert len(use) == 40 and len({len(pairs[i][0]) for i in use}) == 1, (name, arm, g, '使える記事が 40 本ない', len(use))
    full = tok(docs[use[0]]['text']).input_ids                              # 頭の記号（BOS）は tokenizer の既定に合わせる（Mistral は付く、Qwen は付かない）
    bos = tok.bos_token_id if (tok.bos_token_id is not None and full[0] == tok.bos_token_id) else None
    prompts = [([bos] + pairs[i][0]) if (bos is not None and pairs[i][0][0] != bos) else pairs[i][0] for i in use]   # split_doc（凍結）が付けなかったときだけ足す
    assert prompts[0][0] == full[0] and len({len(p) for p in prompts}) == 1, (name, 'BOS の扱いが tokenizer の既定と違う', prompts[0][:3], full[:3])
    maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
    assert maxpos is None or len(prompts[0]) + CONT <= maxpos, ('文が長すぎます', maxpos)
    nD = len(use)
    chunks = [list(range(i, min(nD, i + chunk))) for i in range(0, nD, chunk)]
    seed = (lambda k, s, c: a20_seed1(k, s, c)) if g is None else (lambda k, s, c: a19_seed(k, s, c, g))
    ta = time.time()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    R = {k: np.zeros((len(temps), GP_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(temps):
        Ts, Hs = np.full(CONT, T), np.full(CONT, h)
        for s in range(GP_S):
            for c, ii in enumerate(chunks):
                gen = torch.Generator(device=device).manual_seed(seed(k, s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, gen)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    tag = arm if g is None else f'{arm}_g{g}'
    path = f'{outdir}/{name}@{tag}.npz'
    import transformers
    meta = json.dumps(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
                           commit=getattr(model.config, '_commit_hash', None), hf={**A20_HF, **A20_BACKUP}[name], bos=bos, prompt_len=len(prompts[0])))
    np.savez_compressed(path + '.part.npz', temps=np.array(temps), h=h, arm=arm, group=-1 if g is None else g, use=np.array(use), seconds=time.time() - ta,
                        chunk=chunk, meta=np.array(meta), **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_a20(jobs, worker, nworkers, docs_of, outdir, device, chunk, notify=print):
    """docs_of(job) → その仕事の記事 40 本。仕事を「まだ終わっていない・誰も取っていない」ものから取りに行く（取った印は a19_claim）。"""
    os.makedirs(outdir, exist_ok=True)
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    tag = lambda j: j[1] if j[4] is None else f'{j[1]}_g{j[4]}'
    while True:
        took = False
        for job in order:
            if a19_claim(outdir, job[0], tag(job)):
                took = True
                try:
                    sec = run_a20_job(job, docs_of(job), outdir, device, cache, chunk)
                    n += 1
                    notify(f'{job[0]}@{tag(job)} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{job[0]}@{tag(job)}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {job[0]}@{tag(job)} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = [f'{j[0]}@{tag(j)}' + ('（失敗）' if os.path.exists(f'{outdir}/locks/{j[0]}@{tag(j)}.failed') else '')
            for j in jobs if not os.path.exists(f'{outdir}/{j[0]}@{tag(j)}.npz')]
    return n, left


def self_test_a20(tmp=None):
    import tempfile
    j1 = a20_jobs1(list(A20_HF))
    assert len(j1) == 2 and all(len(j[3]) == 15 and j[3][0] == 0.2 and j[3][-1] == 0.9 and j[2] == 0.0 for j in j1)
    s1 = {a20_seed1(k, s, c) for k in range(15) for s in range(GP_S) for c in range(2)}
    old = {gp_seed(k, s, c) for k in range(7) for s in range(GP_S) for c in range(2)} | {a19_seed(k, s, c, g) for k in range(7) for s in range(GP_S) for c in range(2) for g in range(3)}
    assert len(s1) == 15 * GP_S * 2 and not (s1 & old)
    d = tmp or tempfile.mkdtemp()
    P = dict(models={'qwen2.5-7b': dict(ok=True, target=dict(hstar=-0.41234, grid=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])),
                     'mistral-7b-v0.3': dict(ok=False, why='テスト')})
    json.dump(P, open(f'{d}/h.json', 'w'))
    j2 = a20_jobs2(f'{d}/h.json')
    assert len(j2) == 6 and {j[1] for j in j2} == {'c0', 'h1'} and all(j[0] == 'qwen2.5-7b' for j in j2)
    assert {j[2] for j in j2} == {0.0, -0.4123} and {j[4] for j in j2} == {0, 1, 2}
    json.dump(dict(models={'granite-3.1-8b': dict(ok=True, target=dict(hstar=-0.3, grid=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]))}), open(f'{d}/b.json', 'w'))
    j3 = a20_jobs2([f'{d}/h.json', f'{d}/b.json'])
    assert len(j3) == 12 and {j[0] for j in j3} == {'qwen2.5-7b', 'granite-3.1-8b'}
    P['models']['mistral-7b-v0.3'] = dict(ok=True, target=dict(hstar=-0.5, grid=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])); json.dump(P, open(f'{d}/h2.json', 'w'))
    assert {j[0] for j in a20_jobs2([f'{d}/h2.json', f'{d}/b.json'])} == {'qwen2.5-7b', 'mistral-7b-v0.3'}   # 本命がそろえば予備は入らない
    m = dict(docs=[dict(text_sha256='a'), dict(text_sha256='b')]); assert check_docs20(m, m, 'docs')
    try:
        check_docs20(m, dict(docs=[dict(text_sha256='a')]), 'docs'); raise RuntimeError('違いを見逃した')
    except AssertionError:
        pass
    return 'OK'
