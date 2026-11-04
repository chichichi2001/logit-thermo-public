"""A2-13（ゲート docs/A2_13_GATE.md）のデータ集め：塩（偽薬 30%）の効き方は、Pythia の大きさで変わるか・SmolLM2 でも同じか。GPU（L4）。
scripts/a2_2_collect.py・scripts/a2_4_collect.py・scripts/a2_7_collect.py・scripts/a2_7b_collect.py を実行したあとの同じ名前空間で実行する。
generate_7b（偽薬の禁止のしかた）は A2-7b・A2-10 のまま使う。温度の並び・腕・種は C13_TEMPS・C13_ARMS・C13_SEED として別の名前で持ち、
A2-7b の B7_TEMPS・B7_ARMS・SEED7B は置きかえない（2026-10-30 18:49 の Colab で、置きかえたせいで A2-7b の自己テストが止まったため）。人の続きの ℓ（teacher_7b）は出さない
（T_0 は使わないため。2.8B で人の続きをまとめて通すとメモリが足りないため）。2.8B だけ、20 本ずつ 2 回に分けて生成する（A2-5 と同じ）。
A → L の回数が少なく区間が広い（A2-10 の Pythia-1B で ±0.25）ので、「なし」と「偽薬 30%」をそれぞれ別の種で 2 本ずつ（合わせてサンプル 4）にする。
種は gen_seed13・coin_seed13（名前が _b で終わる腕だけ ＋500000）。A2-7b の gen_seed・coin_seed は使わない。
"""
import gc, os, time
import numpy as np
import torch

C13_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
C13_ARMS = [('none', 'none', 0.0), ('none_b', 'none', 0.0), ('pl300', 'placebo', 0.30), ('pl300_b', 'placebo', 0.30)]
C13_SEED = 20261123
C13_GROUPS = {
    'small': [('pythia-70m', 'EleutherAI/pythia-70m', None), ('pythia-160m', 'EleutherAI/pythia-160m', None),
              ('pythia-410m', 'EleutherAI/pythia-410m', None), ('smollm2-135m', 'HuggingFaceTB/SmolLM2-135M', None),
              ('smollm2-360m', 'HuggingFaceTB/SmolLM2-360M', None), ('pythia-1b', 'EleutherAI/pythia-1b', None)],
    'mid': [('pythia-1.4b', 'EleutherAI/pythia-1.4b', None), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B', None)],
    'large': [('pythia-2.8b', 'EleutherAI/pythia-2.8b', 20)],
}


def gen_seed13(k, s, arm):
    return C13_SEED + 1000 * k + s + (500000 if arm.endswith('_b') else 0)


def coin_seed13(k, s, arm):
    return C13_SEED + 900000 + 1000 * k + s + (500000 if arm.endswith('_b') else 0)


def gen_chunked(model, prompts, T, valid, gen, H2, kind, dose, rng, chunk):
    """chunk 本ずつ generate_7b を呼んでつなぐ（同じ gen・rng を続けて使う。決まった順なので結果は決まる）。"""
    if not chunk or chunk >= len(prompts):
        return generate_7b(model, prompts, T, valid, gen, H2, kind, dose, rng)
    parts = [generate_7b(model, prompts[i:i + chunk], T, valid, gen, H2, kind, dose, rng) for i in range(0, len(prompts), chunk)]
    return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}


def run_13(name, hf_id, chunk, docs, outdir, device, notify=print):
    todo = [a for a in C13_ARMS if not os.path.exists(f'{outdir}/{name}@{a[0]}.npz')]
    if not todo:
        print(name, 'はもう全部終わっています（飛ばします）')
        return
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, eos = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    nD = len(use)
    assert len({len(pairs[i][0]) for i in use}) == 1
    prompts = [pairs[i][0] for i in use]
    keys_f, keys_b = ['ell', 'ban_mass'], ['cand', 'banned', 'banned_top1']
    for arm, kind, dose in todo:
        ta = time.time()
        G = {k: np.zeros((len(C13_TEMPS), SAMPLES, nD, CONT), np.float32) for k in keys_f}
        G.update({k: np.zeros((len(C13_TEMPS), SAMPLES, nD, CONT), np.bool_) for k in keys_b})
        toks = np.zeros((len(C13_TEMPS), SAMPLES, nD, CONT), np.int32)
        for k, T in enumerate(C13_TEMPS):
            for s in range(SAMPLES):
                g = torch.Generator(device=device).manual_seed(gen_seed13(k, s, arm))
                rng = np.random.default_rng(coin_seed13(k, s, arm))
                r = gen_chunked(model, prompts, T, valid, g, H2, kind, dose, rng, chunk)
                for kk in keys_f + keys_b:
                    G[kk][k, s] = r[kk]
                toks[k, s] = r['toks']
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', toks=toks, temps=np.array(C13_TEMPS), use=np.array(use), eos=np.array(eos),
                            prompts=np.array(prompts, np.int32), arm=arm, kind=kind, dose=dose, chunk=chunk or 0,
                            seconds=time.time() - ta, **{f'g_{k}': v for k, v in G.items()})
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


def self_test_13(device):
    """設定の確かめと、分けて生成したものが決まった結果になり、禁止の割合が 30% に近いこと（小さなランダムのモデル）。"""
    assert len(C13_TEMPS) == 7 and C13_TEMPS[0] == 0.4 and C13_TEMPS[-1] == 0.7
    assert [a[0] for a in C13_ARMS] == ['none', 'none_b', 'pl300', 'pl300_b'] and C13_SEED == 20261123
    arms = [a[0] for a in C13_ARMS]
    assert len({gen_seed13(k, s, a) for k in range(7) for s in range(2) for a in arms}) == 28        # なし と 偽薬 は同じ種（A2-10 と同じ）、_b は別
    assert len({coin_seed13(k, s, a) for k in range(7) for s in range(2) for a in arms}) == 28
    assert gen_seed13(0, 0, 'none_b') != gen_seed13(0, 0, 'none') and gen_seed13(0, 0, 'pl300_b') != gen_seed13(0, 0, 'pl300')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global CONT
    keep = CONT
    CONT = 100
    try:
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4, [20, 21, 22, 23] * 4] * 4
        # 分けたときと分けないときは、multinomial が束ごとに乱数を引くので同じトークンにはならない。ここでは「分けたもの」が
        # 決まった結果になること（2 回で同じ）と、形と、禁止の割合を確かめる。
        a = gen_chunked(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(3), 33, 'placebo', 0.3,
                        np.random.default_rng(0), 8)
        b = gen_chunked(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(3), 33, 'placebo', 0.3,
                        np.random.default_rng(0), 8)
        assert (a['toks'] == b['toks']).all() and a['toks'].shape == (16, 100)
        assert abs(a['banned'].mean() - 0.3) < 0.05, a['banned'].mean()
    finally:
        CONT = keep
    return 'OK'
