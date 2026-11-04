"""探索 8.35：場の下見（8.30）の h ＝ 0 を同じ種でもう一度生成し、各歩の「候補の確率の和 q」を記録する（2026-10-31 14:3x、GPU（L4）、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py を実行したあとの同じ名前空間で実行する
（load_model・valid_mask・split_doc・CopyIndexN・B7_N・CONT・F_TEMPS・F_S・f_seed・generate_field を変えずに使う）。

使い道：厳密な重みづけ直しと一次の応答（docs/A2_EQUATION_CATALOG.md の式 24・31）。1 歩ずつ割り直す生成では
  d⟨X⟩/dh（h ＝ 0）＝ Cov(X, Σ_t（候補を選んだか_t − q_t）) ÷ T
が厳密に成り立つので、h ＝ 0 の生成だけから場の効きを予言し、8.30 の h ＝ ±0.3 の生成とくらべる。
記録：トークン、候補があったか、候補を選んだか、q（その歩の、場をかけた後の候補の確率の和）、q0（場をかける前の、同じ温度での候補の確率の和）。
  h ＝ 0 では q ＝ q0。確率の計算は倍精度、保存は float32（1 の近くの刻み 6 × 10^−8）。
トークンが 8.30 の h ＝ 0（Drive の A2_explore/field/{名前}@h0.npz）と 1 つ残らず同じことを確かめてから保存する。
"""
import gc, os, time
import numpy as np
import torch

Q_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B')]


@torch.no_grad()
def generate_fq(model, prompts, T, valid, gen, h):
    """generate_field と同じ手順（同じ乱数の使い方）で、q・q0 も記録する。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    toks, cand, pick, Q, Q0 = [], [], [], [], []
    for t in range(CONT):
        z = logits.double().masked_fill(~valid, float('-inf'))
        cands = [idx[b].candidates() for b in range(B)]
        has = torch.tensor([len(c) > 0 for c in cands], device=device)
        rows, cols = [], []
        for b, c in enumerate(cands):
            rows += [b] * len(c); cols += list(c)
        rt = torch.tensor(rows, device=device, dtype=torch.long); ct = torch.tensor(cols, device=device, dtype=torch.long)
        q0 = torch.zeros(B, dtype=torch.float64, device=device)
        if rows:
            q0.index_add_(0, rt, torch.softmax(z / T, dim=-1)[rt, ct])
        if h != 0.0 and rows:
            z[rt, ct] += h
        p = torch.softmax(z / T, dim=-1)
        q = torch.zeros(B, dtype=torch.float64, device=device)
        if rows:
            q.index_add_(0, rt, p[rt, ct])
        nxt = torch.multinomial(p, 1, generator=gen)
        pk = torch.tensor([int(nxt[b, 0]) in set(cands[b]) for b in range(B)], device=device)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        toks.append(nxt[:, 0]); cand.append(has); pick.append(pk); Q.append(q); Q0.append(q0)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    return dict(toks=st(toks).astype(np.int32), cand=st(cand).astype(np.bool_), pick=st(pick).astype(np.bool_),
                q=st(Q).astype(np.float32), q0=st(Q0).astype(np.float32))


def run_fq(name, hf_id, docs, outdir, fielddir, device, notify=print):
    path = f'{outdir}/{name}@h0q.npz'
    if os.path.exists(path):
        print(name, 'はもう終わっています（飛ばします）')
        return
    ref = np.load(f'{fielddir}/{name}@h0.npz')
    assert float(ref['h']) == 0.0
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, eos = valid_mask(tok, model, device)
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert use == np.asarray(ref['use']).tolist(), '記事の組が 8.30 と違います'
    nD = len(use)
    prompts = [pairs[i][0] for i in use]
    R = {k: np.zeros((len(F_TEMPS), F_S, nD, CONT), dt)
         for k, dt in (('toks', np.int32), ('cand', np.bool_), ('pick', np.bool_), ('q', np.float32), ('q0', np.float32))}
    for k, T in enumerate(F_TEMPS):
        for s in range(F_S):
            g = torch.Generator(device=device).manual_seed(f_seed(k, s))
            r = generate_fq(model, prompts, T, valid, g, 0.0)
            for kk in R:
                R[kk][k, s] = r[kk]
        same = bool((R['toks'][k] == ref['toks'][k]).all() and (R['pick'][k] == ref['pick'][k]).all())
        assert same, (name, '温度', T, 'のトークンが 8.30 の h ＝ 0 と違います')
    np.savez_compressed(path + '.part.npz', temps=np.array(F_TEMPS), h=0.0, use=np.array(use), seconds=time.time() - t0, **R)
    os.replace(path + '.part.npz', path)
    notify(f'{name}@h0q おわり（8.30 とトークンが一致、{time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_fq(device):
    """小さなランダムのモデルで：トークンは generate_field と同じ（h ＝ 0・＋0.3）。q は全部を読み直して出した値と合う。h ＝ 0 では q ＝ q0。"""
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
            a = generate_field(model, prompts, 0.6, valid, torch.Generator(device=device).manual_seed(11), h)
            b = generate_fq(model, prompts, 0.6, valid, torch.Generator(device=device).manual_seed(11), h)
            assert (a['toks'] == b['toks']).all() and (a['pick'] == b['pick']).all(), ('generate_field と違う', h)
            assert ((b['q'] == 0) | b['cand']).all() and ((b['q0'] == 0) | b['cand']).all()
            assert (b['q'] >= 0).all() and (b['q'] <= 1 + 1e-6).all()
            if h == 0.0:
                assert np.array_equal(b['q'], b['q0'])
            else:
                assert (b['q'][b['cand']] >= b['q0'][b['cand']] - 1e-6).all()
            # 読み直し：1 本目の文で、各歩の候補を CopyIndexN でなぞり、q を出し直す
            seq = b['toks'][0].tolist()
            x = torch.tensor([prompts[0] + seq], device=device)
            lg = model(x).logits[0, len(prompts[0]) - 1: len(prompts[0]) - 1 + 60].double().masked_fill(~valid, float('-inf'))
            ci = CopyIndexN(prompts[0], B7_N)
            for t in range(60):
                c = list(ci.candidates())
                zz = lg[t].clone()
                if c and h != 0.0:
                    zz[c] += h
                pp = torch.softmax(zz / 0.6, -1)
                qq = float(pp[c].sum()) if c else 0.0
                assert abs(qq - float(b['q'][0, t])) < 2e-3, (h, t, qq, float(b['q'][0, t]))
                ci.push(seq[t])
    finally:
        CONT = keep
    return 'OK'
