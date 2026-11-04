"""ゲートの前の材料集め（G-pre）：8.30 の場の下見と同じ作り（温度 0.4〜0.7 の 7 個、h ＝ −0.3・0・＋0.3、記事 40 × サンプル 4、512 歩）を、
場をかけたことのない 7 本のモデルで回し、各歩の q も記録する（2026-10-31 20:1x、GPU（L4）を 3 つ、判定しない）。
データは、ゲート（G0 ものさし・G-エンタルピーの整合）を凍結するまで開けない。

scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py を
実行したあとの同じ名前空間で実行する（load_model・valid_mask・split_doc・CONT・D_CHUNK・generate_dsc2 を変えずに使う）。
生成は generate_dsc2 に一定の温度と一定の場を渡す（generate_field と同じトークンになることは self_test_dsc2 で確かめてある）。
記録：トークン、候補があったか、候補を選んだか、U（温度 1・場なし）、q0（場の前）、q（場の後）。形は (温度 7, サンプル 4, 記事, 512)。

落ちた仕事：エラーの全文を ⚠ で通知し、印（locks/…failed）をつけて飛ばし、ほかの仕事を続ける（ひと晩回すため）。
仕事の割り振り：21 個の仕事（7 モデル × 3 つの h）を、3 つのランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行く。
取ったしるしは Drive のロックのファイル（2 時間たったら古いとみなして取り直してよい）。種は仕事だけで決まる（どのランタイムが回しても同じ結果）ので、
2 つのランタイムが同じ仕事をしてしまっても、同じ中身を書くだけで、結果は変わらない。
"""
import gc, os, time, json, socket
import numpy as np
import torch

GP_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
GP_H = [('hm03', -0.3), ('h0', 0.0), ('hp03', 0.3)]
GP_S = 4
GP_SEED = 20261141
GP_MODELS = [('pythia-410m', 'EleutherAI/pythia-410m'), ('pythia-1.4b', 'EleutherAI/pythia-1.4b'), ('gpt-neo-1.3b', 'EleutherAI/gpt-neo-1.3B'),
             ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('gpt2-1.5b', 'openai-community/gpt2-xl'),
             ('bloom-1b1', 'bigscience/bloom-1b1')]
GP_CHUNK = 20
GP_STALE = 2 * 3600


def gp_seed(k, s, c):
    """温度の番号 k、サンプル s、記事のかたまり c → 種（3 つの h で同じ ＝ 共通の乱数）。"""
    return GP_SEED + 1000 * k + 10 * s + c


def gp_jobs():
    return [(name, hf, arm, h) for name, hf in GP_MODELS for arm, h in GP_H]


def gp_claim(outdir, name, arm, worker):
    """まだ終わっていなくて、誰も取っていない（または古い）なら取る → True。"""
    if os.path.exists(f'{outdir}/{name}@{arm}.npz') or os.path.exists(f'{outdir}/locks/{name}@{arm}.failed'):
        return False
    lk = f'{outdir}/locks/{name}@{arm}.lock'
    if os.path.exists(lk):
        try:
            age = time.time() - json.load(open(lk))['t']
        except Exception:
            age = 0.0
        if age < GP_STALE:
            return False
    os.makedirs(f'{outdir}/locks', exist_ok=True)
    json.dump(dict(worker=worker, host=socket.gethostname(), t=time.time()), open(lk, 'w'))
    time.sleep(20)   # Drive の同期を少し待ってから、自分のしるしのままか確かめる
    try:
        return json.load(open(lk)).get('worker') == worker
    except Exception:
        return False


def run_gpre_job(name, hf_id, arm, h, docs, outdir, device, cache):
    """1 つの仕事（モデル × h）。cache：同じモデルを続けて使うときに読み直さないための入れ物。"""
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(hf_id, device)
        assert next(model.parameters()).dtype == torch.float32
        valid, eos = valid_mask(tok, model, device)
        pairs = [split_doc(tok, d['text']) for d in docs]
        use = [i for i, p in enumerate(pairs) if p is not None]
        P = {len(pairs[i][0]) for i in use}
        assert len(P) == 1
        maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
        assert maxpos is None or list(P)[0] + CONT <= maxpos, ('文が長すぎます', maxpos)
        cache.update(name=name, model=model, valid=valid, use=use, prompts=[pairs[i][0] for i in use])
    model, valid, use, prompts = cache['model'], cache['valid'], cache['use'], cache['prompts']
    nD = len(use)
    chunks = [list(range(i, min(nD, i + GP_CHUNK))) for i in range(0, nD, GP_CHUNK)]
    ta = time.time()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    R = {k: np.zeros((len(GP_TEMPS), GP_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(GP_TEMPS):
        Ts, Hs = np.full(CONT, T), np.full(CONT, h)
        for s in range(GP_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(gp_seed(k, s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, g)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    path = f'{outdir}/{name}@{arm}.npz'
    np.savez_compressed(path + '.part.npz', temps=np.array(GP_TEMPS), h=h, arm=arm, use=np.array(use), seconds=time.time() - ta, chunk=GP_CHUNK, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_gpre(worker, docs, outdir, device, notify=print):
    """worker（0・1・2）：自分の番号ぶんずらした順に仕事を見て、取れるものを回す。取れる仕事がなくなったら終わる。"""
    os.makedirs(outdir, exist_ok=True)
    jobs = gp_jobs()
    order = jobs[7 * worker:] + jobs[:7 * worker]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for name, hf, arm, h in order:
            if gp_claim(outdir, name, arm, worker):
                took = True
                try:
                    sec = run_gpre_job(name, hf, arm, h, docs, outdir, device, cache)
                    n += 1
                    notify(f'{name}@{arm} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    # 1 つの仕事が落ちても、ほかの仕事は続ける。落ちた仕事には印をつけて、ほかのランタイムも取らないようにする
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{name}@{arm}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {name}@{arm} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = [f'{nm}@{a}' + ('（失敗）' if os.path.exists(f'{outdir}/locks/{nm}@{a}.failed') else '') for nm, _, a, _ in jobs if not os.path.exists(f'{outdir}/{nm}@{a}.npz')]
    return n, left


def self_test_gpre():
    assert len(gp_jobs()) == 21 and len({(a, b) for a, _, b, _ in gp_jobs()}) == 21
    seeds = {gp_seed(k, s, c) for k in range(7) for s in range(GP_S) for c in range(2)}
    assert len(seeds) == 7 * GP_S * 2
    # ずらした順はどれも 21 個の並べかえ
    jobs = gp_jobs()
    for w in range(3):
        o = jobs[7 * w:] + jobs[:7 * w]
        assert sorted(o) == sorted(jobs)
    return 'OK'
