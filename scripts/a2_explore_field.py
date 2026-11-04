"""探索 8.30（下見）：「場」を足すと融点はどう動くか ——文章のクラウジウス・クラペイロン（2026-10-31 09:4x、GPU（L4）、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py を実行したあとの同じ名前空間で実行する（load_model・valid_mask・
split_doc・load_docs・CopyIndexN・B7_N・CONT・SAMPLES を変えずに使う）。

場 h：各歩で、3 トークンの写しの候補（直前 2 トークンのあとに、それまでに続いたトークン）の logit に h を足す（T で割る前）。
h > 0 は結晶寄り、h < 0 は液体寄り。熱力学では、エネルギーに −h × M（M ＝ 写しの候補を選んだ歩の数）を足すことにあたる。
熱力学どおりなら、融点の動き dT_m/dh ＝ ΔM ÷ ΔS（ΔM：1 回の相転移で変わる M、ΔS：1 回の相転移のエントロピーの差）。
ΔM は、液体と結晶で「写しの候補を選んだ割合」の差 × 協同性の単位 n（A2-8）からも見積もれるので、2 通りで比べられる。

腕：h ＝ −0.3・0・＋0.3（logit の単位、ナット）。温度 0.4〜0.7 の 7 個。サンプル 4（種は h によらず同じ：同じ乱数で比べる）。
保存：トークン、写しの候補があったか、選んだトークンが写しの候補だったか。
"""
import gc, os, time
import numpy as np
import torch

F_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
F_H = [('hm03', -0.3), ('h0', 0.0), ('hp03', 0.3)]
F_S = 4
F_SEED = 20261127
F_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B')]


def f_seed(k, s):
    return F_SEED + 1000 * k + s


@torch.no_grad()
def generate_field(model, prompts, T, valid, gen, h):
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    toks, cand, pick = [], [], []
    for t in range(CONT):
        z = logits.double().masked_fill(~valid, float('-inf'))
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
        pk = torch.tensor([int(nxt[b, 0]) in set(cands[b]) for b in range(B)], device=device)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        toks.append(nxt[:, 0]); cand.append(has); pick.append(pk)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    return dict(toks=st(toks).astype(np.int32), cand=st(cand).astype(np.bool_), pick=st(pick).astype(np.bool_))


def run_field(name, hf_id, docs, outdir, device, notify=print):
    todo = [(a, h) for a, h in F_H if not os.path.exists(f'{outdir}/{name}@{a}.npz')]
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
    assert len({len(pairs[i][0]) for i in use}) == 1
    prompts = [pairs[i][0] for i in use]
    for arm, h in todo:
        ta = time.time()
        R = {k: np.zeros((len(F_TEMPS), F_S, nD, CONT), dt) for k, dt in (('toks', np.int32), ('cand', np.bool_), ('pick', np.bool_))}
        for k, T in enumerate(F_TEMPS):
            for s in range(F_S):
                g = torch.Generator(device=device).manual_seed(f_seed(k, s))
                r = generate_field(model, prompts, T, valid, g, h)
                for kk in R:
                    R[kk][k, s] = r[kk]
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', temps=np.array(F_TEMPS), h=h, use=np.array(use), seconds=time.time() - ta, **R)
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_field(device):
    """小さなランダムのモデルで：h ＝ 0 は A2-7b の「なし」と同じトークン。h が大きいと、候補がある歩ではほぼいつも候補を選ぶ。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global CONT
    keep = CONT
    CONT = 80
    try:
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4, [20, 21, 22, 23] * 4]
        a = generate_field(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(3), 0.0)
        b = generate_7b(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(3), 33, 'none', 0.0, np.random.default_rng(0))
        assert (a['toks'] == b['toks']).all(), 'h ＝ 0 が「なし」と違う'
        c = generate_field(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(4), 30.0)
        assert c['pick'][c['cand']].mean() > 0.99, c['pick'][c['cand']].mean()
        d = generate_field(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(4), -30.0)
        assert d['pick'].mean() < 0.01, d['pick'].mean()
    finally:
        CONT = keep
    assert len(F_TEMPS) == 7 and [a for a, _ in F_H] == ['hm03', 'h0', 'hp03'] and F_S == 4
    return 'OK'
