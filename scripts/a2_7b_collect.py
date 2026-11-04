"""A2-7b（ゲート docs/A2_7B_GATE.md）のデータ集め：本滴定（写しの禁止の小数の濃さと、偽薬の濃さ 3 段）。GPU（L4）。
scripts/a2_2_collect.py・scripts/a2_4_collect.py・scripts/a2_7_collect.py を実行したのと同じ名前空間で実行する
（step_stats・split_doc・load_model・valid_mask・h2_of・CONT・SAMPLES・CopyIndexN・generate_ngram を変えずに使う）。
"""
import gc, math, os, time
import numpy as np
import torch

B7_TEMPS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
SEED7B = 20261110
B7_ARMS = [('none', 'none', 0.0), ('none_b', 'none', 0.0),
           ('cq33', 'copy', 1 / 3), ('cq67', 'copy', 2 / 3), ('cq100', 'copy', 1.0),
           ('pl025', 'placebo', 0.025), ('pl050', 'placebo', 0.05), ('pl075', 'placebo', 0.075)]
B7_GROUPS = {'pythia': ('pythia-1b', 'EleutherAI/pythia-1b'), 'olmo': ('olmo2-1b', 'allenai/OLMo-2-0425-1B'),
             'qwen': ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')}
B7_N = 3


def gen_seed(k, s, arm):
    return SEED7B + 1000 * k + s + (500000 if arm == 'none_b' else 0)


def coin_seed(k, s):
    return SEED7B + 900000 + 1000 * k + s


@torch.no_grad()
def generate_7b(model, prompts, T, valid, gen, H2, kind, dose, rng):
    """kind：'none'・'copy'（3 トークンの写しの候補を、各歩・各記事で確率 dose で全部禁止）・'placebo'（1 位が写しの候補でない歩で、
    全部の歩 × 記事の割合 dose だけ 1 位を禁止。A2-4 の偽薬と同じ「残りの借り」）。手順は A2-7 の generate_ngram と同じ並び。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    owed = 0.0
    rec = {k: [] for k in ['ell', 'cand', 'banned', 'banned_top1', 'ban_mass', 'toks']}
    for t in range(CONT):
        ell = step_stats(logits, T, valid, H2)[0]                         # つまみ前の logit（本命）
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp = torch.log_softmax(z / T, dim=-1)
        argmax = z.argmax(-1)
        cands = [idx[b].candidates() for b in range(B)]
        cand = torch.tensor([len(c) > 0 for c in cands], device=device)
        rows, cols = [], []
        if kind == 'copy':
            coin = rng.random(B) < dose                                   # くじは毎歩 B 枚（候補がなくても引く）
            for b, c in enumerate(cands):
                if c and coin[b]:
                    rows += [b] * len(c); cols += list(c)
        elif kind == 'placebo':
            am = argmax.cpu().numpy()
            elig = np.array([int(am[b]) not in set(cands[b]) for b in range(B)])
            owed += dose * B
            q = min(1.0, owed / elig.sum()) if elig.any() else 0.0
            pick = (rng.random(B) < q) & elig
            owed -= int(pick.sum())
            for b in np.nonzero(pick)[0]:
                rows.append(int(b)); cols.append(int(am[b]))
        banned = torch.zeros(B, dtype=torch.bool, device=device)
        banned_top1 = torch.zeros(B, dtype=torch.bool, device=device)
        ban_mass = torch.zeros(B, dtype=torch.float64, device=device)
        zp = z.clone()
        if rows:
            r = torch.tensor(rows, device=device); c = torch.tensor(cols, device=device)
            zp[r, c] = float('-inf')
            banned[r] = True
            banned_top1[r[c == argmax[r]]] = True
            ban_mass.index_add_(0, r, torch.exp(lp[r, c]))
        p = torch.softmax(zp / T, dim=-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        for k, v in [('ell', ell), ('cand', cand), ('banned', banned), ('banned_top1', banned_top1), ('ban_mass', ban_mass),
                     ('toks', nxt[:, 0])]:
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    res = {}
    for k, v in rec.items():
        a = torch.stack(v, 1)
        res[k] = a.cpu().numpy().astype(np.int32 if k == 'toks' else (np.bool_ if a.dtype == torch.bool else np.float32))
    return res


@torch.no_grad()
def teacher_7b(model, prompts, humans, valid, H2, device):
    """→ (nD, nT, L)：人の続きでの ℓ（順位 H 以降）を B7_TEMPS で。"""
    out = np.zeros((len(prompts), len(B7_TEMPS), CONT), np.float32)
    for j, (pr, hu) in enumerate(zip(prompts, humans)):
        x = torch.tensor([list(pr) + list(hu)], device=device)
        lg = model(x).logits[0, len(pr) - 1: len(pr) - 1 + CONT].float()
        for k, T in enumerate(B7_TEMPS):
            out[j, k] = step_stats(lg, T, valid, H2)[0].float().cpu().numpy()
    return out


def run_7b(name, hf_id, docs, outdir, device, notify=print):
    todo = [a for a in B7_ARMS if not os.path.exists(f'{outdir}/{name}@{a[0]}.npz')]
    if not todo and os.path.exists(f'{outdir}/{name}@tf.npz'):
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
    plen = {len(pairs[i][0]) for i in use}
    assert len(plen) == 1, plen
    prompts = [pairs[i][0] for i in use]
    humans = [pairs[i][1] for i in use]
    if not os.path.exists(f'{outdir}/{name}@tf.npz'):
        tf = teacher_7b(model, prompts, humans, valid, H2, device)
        np.savez_compressed(f'{outdir}/{name}@tf.npz.part.npz', tf=tf, temps=np.array(B7_TEMPS), use=np.array(use))
        os.replace(f'{outdir}/{name}@tf.npz.part.npz', f'{outdir}/{name}@tf.npz')
        print(f'  {name} 人の続きの ℓ おわり（{time.time() - t0:.0f} 秒）')
    keys_f, keys_b = ['ell', 'ban_mass'], ['cand', 'banned', 'banned_top1']
    for arm, kind, dose in todo:
        ta = time.time()
        G = {k: np.zeros((len(B7_TEMPS), SAMPLES, nD, CONT), np.float32) for k in keys_f}
        G.update({k: np.zeros((len(B7_TEMPS), SAMPLES, nD, CONT), np.bool_) for k in keys_b})
        toks = np.zeros((len(B7_TEMPS), SAMPLES, nD, CONT), np.int32)
        for k, T in enumerate(B7_TEMPS):
            for s in range(SAMPLES):
                g = torch.Generator(device=device).manual_seed(gen_seed(k, s, arm))
                rng = np.random.default_rng(coin_seed(k, s))
                r = generate_7b(model, prompts, T, valid, g, H2, kind, dose, rng)
                for kk in keys_f + keys_b:
                    G[kk][k, s] = r[kk]
                toks[k, s] = r['toks']
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', toks=toks, temps=np.array(B7_TEMPS), use=np.array(use), eos=np.array(eos),
                            prompts=np.array(prompts, np.int32), arm=arm, kind=kind, dose=dose, seconds=time.time() - ta,
                            **{f'g_{k}': v for k, v in G.items()})
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_7b(device):
    """小さなランダムのモデルで：copy q ＝ 1 が A2-7 の n ＝ 3 と同じトークン・ℓ。copy q ＝ 0 と placebo r ＝ 0 が「なし」と同じ。
    placebo の禁止の割合が r に近い。copy q ＝ 1/3 で、候補のある歩のうち禁止した割合が 1/3 に近い。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global CONT
    keep = CONT
    CONT = 60
    try:
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4, [20, 21, 22, 23] * 4]
        for T in [0.4, 1.0]:
            g1 = torch.Generator(device=device).manual_seed(3)
            a = generate_ngram(model, prompts, T, valid, g1, 33, 3)
            g2 = torch.Generator(device=device).manual_seed(3)
            b = generate_7b(model, prompts, T, valid, g2, 33, 'copy', 1.0, np.random.default_rng(0))
            assert (a['toks'] == b['toks']).all() and np.allclose(a['ell'], b['ell']), ('q ＝ 1 が n ＝ 3 と違う', T)
            outs = []
            for kind in ('none', 'copy', 'placebo'):
                g = torch.Generator(device=device).manual_seed(4)
                outs.append(generate_7b(model, prompts, T, valid, g, 33, kind, 0.0, np.random.default_rng(1))['toks'])
            assert (outs[0] == outs[1]).all() and (outs[0] == outs[2]).all(), ('濃さ 0 が「なし」と違う', T)
        CONT = 300
        g = torch.Generator(device=device).manual_seed(5)
        r = generate_7b(model, prompts * 4, 1.0, valid, g, 33, 'placebo', 0.075, np.random.default_rng(2))
        rate = r['banned'].mean()
        assert abs(rate - 0.075) < 0.01 and (r['banned'] == r['banned_top1']).all(), rate
        g = torch.Generator(device=device).manual_seed(6)
        r = generate_7b(model, prompts * 16, 0.1, valid, g, 33, 'copy', 1 / 3, np.random.default_rng(3))   # ほぼ貪欲・64 本にして写しの候補を増やす
        nc = int(r['cand'].sum())
        fr = r['banned'][r['cand']].mean()
        # 2026-10-29 22:50 の Colab で、16 本 × 300 歩では候補が 54 歩しかなく止まった（禁止の割合 0.407 は 1/3 から標準誤差 1.2 個ぶん）。
        # 本数を増やし、許す幅を標準誤差の 4 倍にした
        assert nc >= 50 and abs(fr - 1 / 3) < 4 * math.sqrt((1 / 3) * (2 / 3) / nc), (fr, nc)
        assert not (r['banned'] & ~r['cand']).any()
        assert all(T in B7_TEMPS for T in [0.8, 0.85, 0.9]) and len({gen_seed(k, s, a) for k in range(8) for s in range(2) for a in ('none', 'none_b')}) == 32
    finally:
        CONT = keep
    return 'OK'
