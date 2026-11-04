"""探索 8.39（下見）：文章の DSC その 3 ——高温高圧（強い ＋h）で、凍る・溶けるの境目の線がどうなるか（臨界点はあるか）（2026-10-31 18:1x、GPU（L4）、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py を実行したあとの
同じ名前空間で実行する（load_model・valid_mask・split_doc・D_MODELS・D_CHUNK・generate_dsc2 を変えずに使う）。

h は圧力にあたる（＋h で融点が上がる、8.30）。腕は h ＝ 0・＋0.5・＋1.0・＋2.0（はじめから終わりまで同じ h）。
温度：2.0 → 0.35 を 800 歩で冷やし、0.35 → 2.0 を 800 歩で温める（1 本 1,600 歩）。上の端を 2.0 にしたのは、h ＝ ＋2 で溶ける温度が 1.5 を超えうるため
（8.38 の 1.13 ＋ 傾き 0.27 × 2 ≈ 1.67）。温める速さ（1 歩 0.0021）は 8.38 の base（0.0018）に近い。
種は腕によらず同じ（D3_SEED ＋ 10 s ＋ c、共通の乱数。場がはじめからかかるので、トークンは腕ごとに違う）。
記録は 8.38 と同じ：トークン、候補があったか、候補を選んだか、U（温度 1・場なし）、q0（場の前）、q（場の後）。
"""
import gc, os, time
import numpy as np
import torch

D3_TOP, D3_BOT, D3_N = 2.0, 0.35, 800
D3_ARMS = {'h0': 0.0, 'hp05': 0.5, 'hp10': 1.0, 'hp20': 2.0}
D3_S = 4
D3_SEED = 20261139


def d3_sched(arm):
    Ts = np.r_[np.linspace(D3_TOP, D3_BOT, D3_N), np.linspace(D3_BOT, D3_TOP, D3_N)]
    return Ts, np.full_like(Ts, D3_ARMS[arm])


def d3_seed(s, c):
    return D3_SEED + 10 * s + c


def run_dsc3(name, hf_id, docs, outdir, device, notify=print):
    todo = [a for a in D3_ARMS if not os.path.exists(f'{outdir}/{name}@{a}.npz')]
    if not todo:
        print(name, 'はもう全部終わっています（飛ばします）')
        return
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, eos = valid_mask(tok, model, device)
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    nD = len(use)
    P = {len(pairs[i][0]) for i in use}
    assert len(P) == 1
    prompts = [pairs[i][0] for i in use]
    maxpos = getattr(model.config, 'max_position_embeddings', None)
    assert maxpos is None or P.pop() + 2 * D3_N <= maxpos, ('文が長すぎます', maxpos)
    chunks = [list(range(i, min(nD, i + D_CHUNK))) for i in range(0, nD, D_CHUNK)]
    for arm in todo:
        ta = time.time()
        Ts, Hs = d3_sched(arm)
        L = len(Ts)
        dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
        R = {k: np.zeros((D3_S, nD, L), v) for k, v in dt.items()}
        for s in range(D3_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(d3_seed(s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, g)
                for k in R:
                    R[k][s, ii] = r[k]
                torch.cuda.empty_cache()
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', Ts=Ts, Hs=Hs, arm=arm, use=np.array(use), seconds=time.time() - ta, chunk=D_CHUNK, **R)
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


def self_test_dsc3(device):
    """予定（長さ・温度の端・場）と、小さなランダムのモデルで h ＝ ＋2 の q ≥ q0（候補がある歩）。"""
    for arm, h in D3_ARMS.items():
        Ts, Hs = d3_sched(arm)
        assert len(Ts) == 1600 and abs(Ts[0] - 2.0) < 1e-12 and abs(Ts[799] - 0.35) < 1e-12 and abs(Ts[800] - 0.35) < 1e-12 and abs(Ts[-1] - 2.0) < 1e-12
        assert (Hs == h).all()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4]
    Ts = np.linspace(2.0, 0.35, 60)
    b = generate_dsc2(model, prompts, Ts, np.full(60, 2.0), valid, torch.Generator(device=device).manual_seed(9))
    m = b['cand']
    assert (b['q'][m] >= b['q0'][m] - 1e-6).all() and (b['q'][~m] == 0).all()
    return 'OK'
