"""A2-18 の材料集め：予言した罰の強さ h* で生成して、融点が予言どおり下がるかを見る（2026-11-01 08:xx JST、GPU（L4））。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py・a2_gpre_collect.py を
実行したあとの同じ名前空間で実行する（load_model・valid_mask・split_doc・CONT・generate_dsc2・GP_S・GP_CHUNK・gp_seed・gp_claim を使う）。

h* と温度の格子は scripts/a2_18_predict.py が h ＝ 0 のデータ（G-pre）だけから出したもの（results/A2_18/hstar.json）。ここに書き写して固定する。
作り：G-pre と同じ（BBC の記事 40 × サンプル 4、512 歩、一定の温度・一定の場、q つき）。温度は予言した融点 T*（G-pre の融点 − 0.10）のまわりの 7 つ（0.05 おき）。
腕（v0.2）：c0（h ＝ 0 の対照、同じ温度の格子）・h1（h ＝ h*）・h2（h ＝ 2h*）。
仕事：7 モデル × 3 つの腕 ＝ 21 個。ランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行く。
"""
import gc, os, time, json
import numpy as np
import torch

A18_HF = {'pythia-410m': 'EleutherAI/pythia-410m', 'pythia-1.4b': 'EleutherAI/pythia-1.4b', 'gpt-neo-1.3b': 'EleutherAI/gpt-neo-1.3B',
          'smollm2-1.7b': 'HuggingFaceTB/SmolLM2-1.7B', 'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B', 'gpt2-1.5b': 'openai-community/gpt2-xl',
          'bloom-1b1': 'bigscience/bloom-1b1'}
A18_JOBS = [
    ('pythia-410m', 'c0', 0.0, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('pythia-410m', 'h1', -0.3955, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('pythia-410m', 'h2', -0.791, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('pythia-1.4b', 'c0', 0.0, [0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62]),
    ('pythia-1.4b', 'h1', -0.4975, [0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62]),
    ('pythia-1.4b', 'h2', -0.995, [0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62]),
    ('gpt-neo-1.3b', 'c0', 0.0, [0.38, 0.43, 0.48, 0.53, 0.58, 0.63, 0.68]),
    ('gpt-neo-1.3b', 'h1', -0.3878, [0.38, 0.43, 0.48, 0.53, 0.58, 0.63, 0.68]),
    ('gpt-neo-1.3b', 'h2', -0.7756, [0.38, 0.43, 0.48, 0.53, 0.58, 0.63, 0.68]),
    ('smollm2-1.7b', 'c0', 0.0, [0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56]),
    ('smollm2-1.7b', 'h1', -0.5286, [0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56]),
    ('smollm2-1.7b', 'h2', -1.0572, [0.26, 0.31, 0.36, 0.41, 0.46, 0.51, 0.56]),
    ('qwen2.5-1.5b', 'c0', 0.0, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('qwen2.5-1.5b', 'h1', -0.4686, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('qwen2.5-1.5b', 'h2', -0.9372, [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]),
    ('gpt2-1.5b', 'c0', 0.0, [0.23, 0.28, 0.33, 0.38, 0.43, 0.48, 0.53]),
    ('gpt2-1.5b', 'h1', -0.7853, [0.23, 0.28, 0.33, 0.38, 0.43, 0.48, 0.53]),
    ('gpt2-1.5b', 'h2', -1.5706, [0.23, 0.28, 0.33, 0.38, 0.43, 0.48, 0.53]),
    ('bloom-1b1', 'c0', 0.0, [0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.71]),
    ('bloom-1b1', 'h1', -0.3745, [0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.71]),
    ('bloom-1b1', 'h2', -0.749, [0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.71]),
]

def run_a18_job(name, arm, h, temps, docs, outdir, device, cache):
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(A18_HF[name], device)
        assert next(model.parameters()).dtype == torch.float32
        valid, eos = valid_mask(tok, model, device)
        pairs = [split_doc(tok, d['text']) for d in docs]
        use = [i for i, p in enumerate(pairs) if p is not None]
        assert len({len(pairs[i][0]) for i in use}) == 1
        maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
        assert maxpos is None or len(pairs[use[0]][0]) + CONT <= maxpos, ('文が長すぎます', maxpos)
        cache.update(name=name, model=model, valid=valid, use=use, prompts=[pairs[i][0] for i in use])
    model, valid, use, prompts = cache['model'], cache['valid'], cache['use'], cache['prompts']
    nD = len(use)
    chunks = [list(range(i, min(nD, i + GP_CHUNK))) for i in range(0, nD, GP_CHUNK)]
    ta = time.time()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    R = {k: np.zeros((len(temps), GP_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(temps):
        Ts, Hs = np.full(CONT, T), np.full(CONT, h)
        for s in range(GP_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(gp_seed(k, s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, g)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    path = f'{outdir}/{name}@{arm}.npz'
    np.savez_compressed(path + '.part.npz', temps=np.array(temps), h=h, arm=arm, use=np.array(use), seconds=time.time() - ta, chunk=GP_CHUNK, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_a18(worker, nworkers, docs, outdir, device, notify=print):
    os.makedirs(outdir, exist_ok=True)
    sh = (len(A18_JOBS) * worker) // nworkers
    order = A18_JOBS[sh:] + A18_JOBS[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for name, arm, h, temps in order:
            if gp_claim(outdir, name, arm, worker):
                took = True
                try:
                    sec = run_a18_job(name, arm, h, temps, docs, outdir, device, cache)
                    n += 1
                    notify(f'{name}@{arm} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
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
    left = [f'{nm}@{a}' + ('（失敗）' if os.path.exists(f'{outdir}/locks/{nm}@{a}.failed') else '') for nm, a, _, _ in A18_JOBS if not os.path.exists(f'{outdir}/{nm}@{a}.npz')]
    return n, left


def self_test_a18():
    assert len(A18_JOBS) == 21 and len({(n, a) for n, a, _, _ in A18_JOBS}) == 21
    assert all((h < 0) == (a != 'c0') and len(t) == 7 and min(t) > 0.1 for _, a, h, t in A18_JOBS)
    for n in A18_HF:                                                     # 3 つの腕は同じ温度の格子、h2 ＝ 2 × h1
        J = {a: (h, t) for m, a, h, t in A18_JOBS if m == n}
        assert J['c0'][1] == J['h1'][1] == J['h2'][1] and abs(J['h2'][0] - 2 * J['h1'][0]) < 1e-3
    assert set(n for n, _, _, _ in A18_JOBS) == set(A18_HF)
    return 'OK'
