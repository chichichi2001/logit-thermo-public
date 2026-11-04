"""探索 8.33（下見）：文章の DSC ——温度をゆっくり下げて上げ、凍る温度と溶ける温度のずれ（ヒステリシス）を測る（2026-10-31 13:3x、GPU（L4）、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py を実行したあとの同じ名前空間で実行する
（load_model・valid_mask・split_doc・load_docs・CopyIndexN・B7_N・generate_field を変えずに使う）。

1 本の生成の中で、温度を歩ごとに動かす（片道でまっすぐ 0.85 ↔ 0.35）：
  'ch'：冷やす → 温める（0.85 → 0.35 → 0.85）、'hc'：温める → 冷やす（0.35 → 0.85 → 0.35）。
  2 つの順番は、文章が長くなるほど写せる過去が増える「位置の効き」を、ヒステリシスと切り分けるため。
場 h（写しの候補の logit に足す、8.30 と同じ）：−0.3・0・＋0.3。片道 640 歩（'slow'）。h ＝ 0 だけ片道 320 歩（'fast'）も。
記録：トークン、写しの候補があったか、候補を選んだか、1 歩ごとのエネルギー U ＝ −ln p（温度 1・場なし。選んだトークン）、歩ごとの温度。
記事は 20 本ずつに分けて生成する（長い文の KV キャッシュのメモリのため）。種は腕・サンプル・分けた組ごとに決める。
"""
import gc, os, time
import numpy as np
import torch

D_TH, D_TL = 0.85, 0.35
D_LEG = {'slow': 640, 'fast': 320}
D_HS = [('hm03', -0.3), ('h0', 0.0), ('hp03', 0.3)]
D_ORDERS = ['ch', 'hc']
D_ARMS = [(o, ha, h, 'slow') for ha, h in D_HS for o in D_ORDERS] + [(o, 'h0', 0.0, 'fast') for o in D_ORDERS]
D_S = 4
D_CHUNK = 20
D_SEED = 20261133
D_MODELS = {'pythia': ('pythia-1b', 'EleutherAI/pythia-1b'), 'olmo': ('olmo2-1b', 'allenai/OLMo-2-0425-1B')}


def d_sched(order, rate):
    n = D_LEG[rate]
    down, up = np.linspace(D_TH, D_TL, n), np.linspace(D_TL, D_TH, n)
    return np.concatenate([down, up] if order == 'ch' else [up, down])


def d_arm_name(order, harm, rate):
    return f'{order}_{harm}_{rate}'


def d_seed(order, harm, rate, s, c):
    return (D_SEED + 100000 * ['slow', 'fast'].index(rate) + 10000 * D_ORDERS.index(order)
            + 1000 * [a for a, _ in D_HS].index(harm) + 10 * s + c)


@torch.no_grad()
def generate_dsc(model, prompts, Ts, h, valid, gen):
    """Ts：歩ごとの温度。→ toks・cand・pick・U（(B, len(Ts))）。手順は generate_field と同じ（温度だけ歩ごとに変わる）。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    toks, cand, pick, U = [], [], [], []
    for t in range(len(Ts)):
        T = float(Ts[t])
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp1 = torch.log_softmax(z, dim=-1)                     # 温度 1・場なしの確率（エネルギーのものさし）
        cands = [idx[b].candidates() for b in range(B)]
        has = torch.tensor([len(c) > 0 for c in cands], device=device)
        if h != 0.0:
            rows, cols = [], []
            for b, c in enumerate(cands):
                rows += [b] * len(c); cols += list(c)
            if rows:
                z[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] += h
        p = torch.softmax(z / T, dim=-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        u = -lp1.gather(-1, nxt)[:, 0]
        pk = torch.tensor([int(nxt[b, 0]) in set(cands[b]) for b in range(B)], device=device)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        toks.append(nxt[:, 0]); cand.append(has); pick.append(pk); U.append(u)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    return dict(toks=st(toks).astype(np.int32), cand=st(cand).astype(np.bool_), pick=st(pick).astype(np.bool_),
                U=st(U).astype(np.float32))


def run_dsc(name, hf_id, docs, outdir, device, notify=print):
    todo = [a for a in D_ARMS if not os.path.exists(f'{outdir}/{name}@{d_arm_name(a[0], a[1], a[3])}.npz')]
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
    assert maxpos is None or P.pop() + 2 * D_LEG['slow'] <= maxpos, ('文が長すぎます', maxpos)
    chunks = [list(range(i, min(nD, i + D_CHUNK))) for i in range(0, nD, D_CHUNK)]
    for order, harm, h, rate in todo:
        ta = time.time()
        Ts = d_sched(order, rate)
        L = len(Ts)
        R = {k: np.zeros((D_S, nD, L), dt) for k, dt in (('toks', np.int32), ('cand', np.bool_), ('pick', np.bool_), ('U', np.float32))}
        for s in range(D_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(d_seed(order, harm, rate, s, c))
                r = generate_dsc(model, [prompts[i] for i in ii], Ts, h, valid, g)
                for k in R:
                    R[k][s, ii] = r[k]
                torch.cuda.empty_cache()
        arm = d_arm_name(order, harm, rate)
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', Ts=Ts, h=h, order=order, rate=rate, use=np.array(use), seconds=time.time() - ta,
                            chunk=D_CHUNK, **R)
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_dsc(device):
    """小さなランダムのモデルで：温度が一定なら generate_field と同じトークン（h ＝ 0・＋0.3）。U は −ln p（温度 1）と合う。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global CONT
    keep = CONT
    CONT = 60
    try:
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4]
        for h in (0.0, 0.3):
            a = generate_field(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(7), h)
            b = generate_dsc(model, prompts, np.full(60, 0.7), h, valid, torch.Generator(device=device).manual_seed(7))
            assert (a['toks'] == b['toks']).all() and (a['pick'] == b['pick']).all(), ('温度一定で generate_field と違う', h)
        # U の確かめ：全部の列を読み直して −ln p を出す
        x = torch.tensor([prompts[0] + b['toks'][0].tolist()], device=device)
        lg = model(x).logits[0, len(prompts[0]) - 1: len(prompts[0]) - 1 + 60].double().masked_fill(~valid, float('-inf'))
        u = -torch.log_softmax(lg, -1).gather(-1, torch.tensor(b['toks'][0], device=device, dtype=torch.long)[:, None])[:, 0]
        assert np.allclose(u.cpu().numpy(), b['U'][0], atol=2e-3), np.abs(u.cpu().numpy() - b['U'][0]).max()
    finally:
        CONT = keep
    for o in D_ORDERS:
        for r in ('slow', 'fast'):
            Ts = d_sched(o, r)
            assert len(Ts) == 2 * D_LEG[r] and abs(Ts[0] - (D_TH if o == 'ch' else D_TL)) < 1e-12 and abs(Ts[-1] - Ts[0]) < 1e-12
            assert abs(Ts[D_LEG[r] - 1] - (D_TL if o == 'ch' else D_TH)) < 1e-12
    seeds = {d_seed(o, ha, r, s, c) for o, ha, _, r in D_ARMS for s in range(D_S) for c in range(2)}
    assert len(seeds) == len(D_ARMS) * D_S * 2
    assert len(D_ARMS) == 8
    return 'OK'
