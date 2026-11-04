"""A2-2（ゲート v0.3）のデータ集め。GPU のノートブックにこのファイルの中身をそのまま埋め込む。

何も判定しない（分析は別の CPU ノートブック）。各モデルについて：
- 人の文脈（teacher forcing）：各位置・各温度の ℓ（裾の取り分の log-odds）、1 位の log-odds、エントロピー。
- 温度計の材料：β ＝ 1/T の格子ごとに Σ(⟨z⟩_β − z_y)（人のトークンの平均エネルギーとの差）。
- a_loc・h（T ＝ 1、人の文脈の各位置）。
- 生成：各温度・各サンプルで 512 歩、各一歩の ℓ・1 位の log-odds・エントロピーと、選んだトークン。
すべて float64 で計算し、float32 で保存する。使わない番号（tokenizer の長さ以上）と EOS は −∞ にして外す。
"""
import gc, hashlib, json, math, os, re, time
import numpy as np
import torch

H = 100
PROMPT, CONT, N_DOCS, SAMPLES = 64, 512, 40, 2
TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
TCAL_T = np.exp(np.linspace(np.log(0.4), np.log(2.5), 61))   # 温度計の格子（T）
SEED = 20261027
MODELS = [
    ('pythia-1b', 'EleutherAI/pythia-1b', 'Pythia'),
    ('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B', 'Qwen'),
    ('qwen2.5-0.5b-inst', 'Qwen/Qwen2.5-0.5B-Instruct', 'Qwen'),
    ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B', 'Qwen'),
    ('olmo2-1b', 'allenai/OLMo-2-0425-1B', 'OLMo'),
    ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B', 'SmolLM'),
    ('smollm2-1.7b-inst', 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'SmolLM'),
]
CROSS = [('pythia-1b', 'olmo2-1b'), ('qwen2.5-1.5b', 'smollm2-1.7b')]
BBC_MONTHS = ['2025-04', '2025-05', '2025-06']
BOILER = re.compile(r'(Getty Images|Image source|Image caption|Related topics|Follow BBC|^Watch:|^Listen:|To play this video you need to enable JavaScript)')
H_FRAC = 0.00066   # H を語彙の割合にした版（ゲート §2.1）


def h2_of(n_vocab):
    return max(2, int(round(H_FRAC * n_vocab)))


# ---------- 一歩の量（torch、float64） ----------

def step_stats(z, T, valid, H2):
    """z：(B, V) の logit、valid：(V,) の bool。戻り値：ℓ（順位 H 以降）、1 位の log-odds、エントロピー、ℓ（順位 H2 以降）。
    どれも (B,)、float64。1 位の log-odds は「1 位を除いた logsumexp」から直接計算する（p1 → 1 でも有限）。"""
    z = z.double().masked_fill(~valid, float('-inf')) / T
    lse = torch.logsumexp(z, dim=-1)
    top = torch.topk(z, max(H, H2) - 1, dim=-1)

    def ell_of(h):
        head = torch.logsumexp(top.values[:, :h - 1], dim=-1)
        return torch.logsumexp(z.scatter(-1, top.indices[:, :h - 1], float('-inf')), dim=-1) - head
    rest = torch.logsumexp(z.scatter(-1, top.indices[:, :1], float('-inf')), dim=-1)
    top1 = top.values[:, 0] - rest
    lp = z - lse[:, None]
    p = torch.exp(lp)
    ent = -torch.where(p > 0, p * lp, torch.zeros_like(p)).sum(-1)
    return ell_of(H), top1, ent, ell_of(H2)


def np_step_stats(z, T, valid, H2):
    """自己テスト用の素直な numpy 版（1 行）。"""
    z = np.where(valid, z.astype(np.float64), -np.inf) / T
    zs = np.sort(z[np.isfinite(z)])[::-1]
    def lse(x):
        m = x.max(); return m + np.log(np.exp(x - m).sum())
    ell = lse(zs[H - 1:]) - lse(zs[:H - 1])
    ell2 = lse(zs[H2 - 1:]) - lse(zs[:H2 - 1])
    lp = zs - lse(zs)
    return ell, float(zs[0] - lse(zs[1:])), float(-(np.exp(lp) * lp).sum()), ell2


def shape_stats(z, valid):
    """T ＝ 1 の a_loc（順位 300〜3,000 の傾き）と h（頭の確率 ÷ 裾の確率の log10）。z：(N, V)。"""
    z = z.double().masked_fill(~valid, float('-inf'))
    zs = torch.sort(z, dim=-1, descending=True).values
    r = torch.arange(300, 3001, device=z.device, dtype=torch.float64)
    x = torch.log(r); x = x - x.mean()
    y = zs[:, 299:3000]
    a = -((y - y.mean(-1, keepdim=True)) * x).sum(-1) / (x * x).sum()
    h = (torch.logsumexp(zs[:, :H - 1], -1) - torch.logsumexp(zs[:, H - 1:], -1)) / math.log(10)
    return a, h


def tcal_terms(z, y, valid):
    """温度計の材料：各 T（TCAL_T）で Σ_t(⟨z⟩_T − z_y)。z：(N, V)、y：(N,)。"""
    z = z.double().masked_fill(~valid, float('-inf'))
    zy = z.gather(-1, y[:, None])[:, 0]
    out = []
    for T in TCAL_T:
        lp = torch.log_softmax(z / T, dim=-1)
        p = torch.exp(lp)
        mean_z = torch.where(p > 0, p * z, torch.zeros_like(p)).sum(-1)
        out.append(float((mean_z - zy).sum()))
    return np.array(out)


# ---------- 書き出しの文章 ----------

def clean(text):
    paras = [p.strip() for p in text.split('\n')]
    keep = [p for p in paras if len(p) >= 60 and not BOILER.search(p)]
    return '\n\n'.join(keep)


def load_docs():
    from datasets import load_dataset
    pool = []
    for m in BBC_MONTHS:
        ds = load_dataset('RealTimeData/bbc_news_alltime', m, split='train')
        for row in ds:
            c = clean(row['content'] or '')
            if len(c) >= 4000:
                pool.append({'month': m, 'link': row['link'], 'text': c})
    pool.sort(key=lambda d: hashlib.sha256(d['link'].encode()).hexdigest())
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(pool), size=N_DOCS, replace=False)
    docs = [pool[i] for i in sorted(idx)]
    meta = {'pool_size': len(pool), 'docs': [{'month': d['month'], 'link_sha256': hashlib.sha256(d['link'].encode()).hexdigest(),
                                               'text_sha256': hashlib.sha256(d['text'].encode()).hexdigest(),
                                               'chars': len(d['text'])} for d in docs]}
    return docs, meta


# ---------- モデルごと ----------

def load_model(hf_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    return tok, model


def valid_mask(tok, model, device):
    V = model.get_output_embeddings().weight.shape[0]
    valid = torch.zeros(V, dtype=torch.bool, device=device)
    valid[:len(tok)] = True
    eos = set()
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    ge = getattr(model, 'generation_config', None)
    if ge is not None and ge.eos_token_id is not None:
        eos |= set([ge.eos_token_id] if isinstance(ge.eos_token_id, int) else ge.eos_token_id)
    eot = tok.convert_tokens_to_ids('<|endoftext|>') if '<|endoftext|>' in tok.get_vocab() else None
    if eot is not None:
        eos.add(int(eot))   # 文書の終わりの記号も外す（SmolLM2-Instruct は EOS が <|im_end|> なので、ここで足す）
    for e in eos:
        if e < V:
            valid[e] = False
    return valid, sorted(eos)


def split_doc(tok, text):
    ids = tok(text, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
    if len(ids) < PROMPT + CONT:
        return None
    return bos + ids[:PROMPT], ids[PROMPT:PROMPT + CONT]


@torch.no_grad()
def teacher_forced(model, prompt, human, valid, device, H2):
    x = torch.tensor([prompt + human], device=device)
    logits = model(x).logits[0, len(prompt) - 1: len(prompt) - 1 + CONT].float()
    y = torch.tensor(human, device=device)
    res = {}
    for k, T in enumerate(TEMPS):
        res[k] = [v.float().cpu().numpy() for v in step_stats(logits, T, valid, H2)]
    a, h = shape_stats(logits, valid)
    return res, a.float().cpu().numpy(), h.float().cpu().numpy(), tcal_terms(logits, y, valid)


@torch.no_grad()
def generate(model, prompts, T, valid, gen, H2):
    """prompts：同じ長さのトークン列のリスト。512 歩を KV キャッシュで書く。"""
    x = torch.tensor(prompts, device=valid.device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    L, T1, E, L2, toks = [], [], [], [], []
    for t in range(CONT):
        ell, top1, ent, ell2 = step_stats(logits, T, valid, H2)
        L.append(ell); T1.append(top1); E.append(ent); L2.append(ell2)
        z = logits.double().masked_fill(~valid, float('-inf')) / T
        p = torch.softmax(z, dim=-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        toks.append(nxt[:, 0])
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda xs: torch.stack(xs, 1).float().cpu().numpy()
    return st(L), st(T1), st(E), st(L2), torch.stack(toks, 1).int().cpu().numpy()


def run_model(name, hf_id, docs, outdir, device, notify=print):
    path = f'{outdir}/{name}.npz'
    if os.path.exists(path):
        print(name, 'はもう終わっています（飛ばします）')
        return
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    valid, eos = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    nT, nD = len(TEMPS), len(use)
    tf = np.zeros((4, nD, nT, CONT), np.float32)   # [ℓ(H), 1 位の log-odds, エントロピー, ℓ(H2)]
    a_all = np.zeros((nD, CONT), np.float32); h_all = np.zeros((nD, CONT), np.float32)
    tcal = np.zeros((nD, len(TCAL_T)))
    for j, i in enumerate(use):
        res, a, h, tc = teacher_forced(model, *pairs[i], valid, device, H2)
        for k in range(nT):
            for q in range(4):
                tf[q, j, k] = res[k][q]
        a_all[j], h_all[j], tcal[j] = a, h, tc
    plen = {len(pairs[i][0]) for i in use}
    assert len(plen) == 1, plen
    prompts = [pairs[i][0] for i in use]
    gen = np.zeros((4, nT, SAMPLES, nD, CONT), np.float32)
    toks = np.zeros((nT, SAMPLES, nD, CONT), np.int32)
    for k, T in enumerate(TEMPS):
        for s in range(SAMPLES):
            g = torch.Generator(device=device).manual_seed(SEED + 1000 * k + s)
            l, t1, e, l2, tk = generate(model, prompts, T, valid, g, H2)
            gen[0, k, s], gen[1, k, s], gen[2, k, s], gen[3, k, s], toks[k, s] = l, t1, e, l2, tk
        print(f'  {name} T={T} おわり（{time.time() - t0:.0f} 秒）')
    np.savez_compressed(path + '.part.npz', tf=tf, a=a_all, h=h_all, tcal=tcal, tcal_T=TCAL_T, gen=gen, toks=toks,
                        temps=np.array(TEMPS), use=np.array(use), eos=np.array(eos), tok_len=len(tok),
                        prompt_len=list(plen)[0], revision=str(getattr(model.config, '_commit_hash', '')),
                        seconds=time.time() - t0, H2=H2)
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()
    notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')


@torch.no_grad()
def cross_model(reader, writer, docs, outdir, device):
    """reader が writer の書いた文を読む。文字の窓で、人の続きを読んだときと比べる（記録用）。"""
    path = f'{outdir}/cross_{reader}_reads_{writer}.npz'
    if os.path.exists(path):
        return
    from transformers import AutoTokenizer
    hf = dict((n, h) for n, h, _ in MODELS)
    wtok = AutoTokenizer.from_pretrained(hf[writer])
    W = np.load(f'{outdir}/{writer}.npz')
    tok, model = load_model(hf[reader], device)
    valid, _ = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    use = list(W['use']); Wt = W['toks']
    out = np.full((2, len(TEMPS), SAMPLES, len(use)), np.nan)   # [cross, human] の ℓ の平均
    for j, i in enumerate(use):
        text = docs[i]['text']
        enc = wtok(text, add_special_tokens=False, return_offsets_mapping=True)
        pend = enc['offset_mapping'][PROMPT - 1][1]                  # 書き手の書き出しの終わりの文字位置
        pre = text[:pend]
        for k, T in enumerate(TEMPS):
            for s in range(SAMPLES):
                gtxt = wtok.decode(Wt[k, s, j])
                C = min(len(gtxt), len(text) - pend)
                for q, cont in enumerate([gtxt[:C], text[pend:pend + C]]):
                    e2 = tok(pre + cont, add_special_tokens=False, return_offsets_mapping=True)
                    ids = e2['input_ids']; offs = e2['offset_mapping']
                    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
                    x = torch.tensor([bos + ids], device=device)
                    logits = model(x).logits[0].float()
                    pos = [n for n, (a0, _) in enumerate(offs) if pend <= a0 < pend + C and n > 0]
                    if not pos:
                        continue
                    pred = torch.tensor([n - 1 + len(bos) for n in pos], device=device)
                    ell = step_stats(logits[pred], T, valid, H2)[0]
                    out[q, k, s, j] = float(ell.mean())
    np.savez_compressed(path + '.part.npz', out=out, temps=np.array(TEMPS))
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()


# ---------- 自己テスト（本番の前に） ----------

@torch.no_grad()
def self_test(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    H2 = 33
    # ① 一歩の量が numpy の素直な計算と一致する（1 位が飛び抜けた行も入れる：1 位の log-odds が有限か）
    z = torch.randn(3, valid.shape[0], device=device) * 3
    z[2, 7] += 80.0
    for T in [0.6, 1.0, 1.5]:
        got_all = step_stats(z, T, valid, H2)
        assert all(bool(torch.isfinite(v).all()) for v in got_all), ('有限でない値', T)
        for b in range(3):
            ref = np_step_stats(z[b].cpu().numpy(), T, valid.cpu().numpy(), H2)
            got = tuple(float(v[b]) for v in got_all)
            assert all(abs(g - r) < 1e-6 * max(1, abs(r)) for g, r in zip(got, ref)), (T, got, ref)
    # ② KV キャッシュの生成と、まとめて計算した logit が一致する
    g = torch.Generator(device=device).manual_seed(0)
    prompts = [list(range(5, 5 + 16)), list(range(30, 30 + 16))]
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True); past, lg = out.past_key_values, out.logits[:, -1, :]
    seq, steps = x, []
    for t in range(8):
        nxt = torch.multinomial(torch.softmax(lg.masked_fill(~valid, float('-inf')), -1), 1, generator=g)
        steps.append(lg.clone()); seq = torch.cat([seq, nxt], 1)
        out = model(nxt, past_key_values=past, use_cache=True); past, lg = out.past_key_values, out.logits[:, -1, :]
    full = model(seq).logits
    for t in range(8):
        assert torch.allclose(full[:, 15 + t, :], steps[t], atol=1e-4), ('KV キャッシュの不一致', t)
    # ③ 1 歩目は生成と人の文脈で同じ分布
    human = list(range(100, 100 + 64))   # 小さなモデルは位置 512 までなので短くする（使うのは位置 15 だけ）
    tf_logits = model(torch.tensor([prompts[0] + human], device=device)).logits[0, 15]
    gen_first = model(torch.tensor([prompts[0]], device=device)).logits[0, -1]
    assert torch.allclose(tf_logits, gen_first, atol=1e-5), '1 歩目が一致しない'
    return 'OK'
