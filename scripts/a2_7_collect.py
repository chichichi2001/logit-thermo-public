"""A2-7（ゲート docs/A2_7_GATE.md）のデータ集め：滴定（写しの禁止の濃さを変える）。GPU（L4）のノートブックで、
scripts/a2_2_collect.py と scripts/a2_4_collect.py を実行したのと同じ名前空間で実行する（step_stats・split_doc・load_model・
valid_mask・h2_of・SEED4・TEMPS・CONT・SAMPLES を変えずに使う）。

濃さ ＝ 禁止する写しの長さ n（n トークンの完全な写しを禁止する ＝ no_repeat_ngram_size ＝ n）。n が小さいほど濃い。
A2-4 の block は n ＝ 4。ここでは n ＝ 8・6・3・2 を新しく回す（なし と n ＝ 4 は A2-4 のデータを使う）。
温度は A7_TEMPS の 8 つだけ（T_0 のまわり）。生成の種は A2-4 と同じく SEED4 ＋ 1000 × k ＋ s（k は A2-2 の 15 の温度での番号）。
"""
import gc, os, time
import numpy as np
import torch

A7_NS = [8, 6, 3, 2]
A7_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
A7_GROUPS = {'pythia': ('pythia-1b', 'EleutherAI/pythia-1b'), 'olmo': ('olmo2-1b', 'allenai/OLMo-2-0425-1B'),
             'qwen': ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')}


class CopyIndexN:
    """1 本の文の「直前の n − 1 トークン → それまでに続いたトークンの集合」。n ＝ 4 で A2-4 の CopyIndex と同じ。"""

    def __init__(self, hist, n):
        self.n, self.seq, self.nxt = n, [int(t) for t in hist], {}
        m = n - 1
        for i in range(m, len(self.seq)):
            self.nxt.setdefault(tuple(self.seq[i - m:i]), set()).add(self.seq[i])

    def candidates(self):
        m = self.n - 1
        if len(self.seq) < m:
            return ()
        return tuple(self.nxt.get(tuple(self.seq[-m:]), ()))

    def push(self, t):
        t, m = int(t), self.n - 1
        if len(self.seq) >= m:
            self.nxt.setdefault(tuple(self.seq[-m:]), set()).add(t)
        self.seq.append(t)


@torch.no_grad()
def generate_ngram(model, prompts, T, valid, gen, H2, n):
    """A2-4 の generate_pen の 'block' と同じ手順で、禁止する写しの長さだけ n にしたもの。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, n) for p in prompts]
    rec = {k: [] for k in ['ell', 'top1', 'ent', 'ell2', 'cand', 'banned_top1', 'ban_mass', 'toks']}
    for t in range(CONT):
        ell, top1, ent, ell2 = step_stats(logits, T, valid, H2)          # つまみ前の logit（本命）
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp = torch.log_softmax(z / T, dim=-1)
        argmax = z.argmax(-1)
        cands = [idx[b].candidates() for b in range(B)]
        cand = torch.tensor([len(c) > 0 for c in cands], device=device)
        rows, cols = [], []
        for b, c in enumerate(cands):
            if c:
                rows += [b] * len(c); cols += list(c)
        banned_top1 = torch.zeros(B, dtype=torch.bool, device=device)
        ban_mass = torch.zeros(B, dtype=torch.float64, device=device)
        zp = z.clone()
        if rows:
            r = torch.tensor(rows, device=device); c = torch.tensor(cols, device=device)
            zp[r, c] = float('-inf')
            banned_top1[r[c == argmax[r]]] = True
            ban_mass.index_add_(0, r, torch.exp(lp[r, c]))
        p = torch.softmax(zp / T, dim=-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        for k, v in [('ell', ell), ('top1', top1), ('ent', ent), ('ell2', ell2), ('cand', cand), ('banned_top1', banned_top1),
                     ('ban_mass', ban_mass), ('toks', nxt[:, 0])]:
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    res = {}
    for k, v in rec.items():
        a = torch.stack(v, 1)
        res[k] = a.cpu().numpy().astype(np.int32 if k == 'toks' else (np.bool_ if a.dtype == torch.bool else np.float32))
    return res


def run_ngram(name, hf_id, docs, outdir, device, n, notify=print):
    path = f'{outdir}/{name}@n{n}.npz'
    if os.path.exists(path):
        print(name, n, 'はもう終わっています（飛ばします）')
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
    ks = [TEMPS.index(T) for T in A7_TEMPS]
    keys_f, keys_b = ['ell', 'top1', 'ent', 'ell2', 'ban_mass'], ['cand', 'banned_top1']
    G = {k: np.zeros((len(ks), SAMPLES, nD, CONT), np.float32) for k in keys_f}
    G.update({k: np.zeros((len(ks), SAMPLES, nD, CONT), np.bool_) for k in keys_b})
    toks = np.zeros((len(ks), SAMPLES, nD, CONT), np.int32)
    for q, k in enumerate(ks):
        T = TEMPS[k]
        for s in range(SAMPLES):
            g = torch.Generator(device=device).manual_seed(SEED4 + 1000 * k + s)
            r = generate_ngram(model, prompts, T, valid, g, H2, n)
            for kk in keys_f + keys_b:
                G[kk][q, s] = r[kk]
            toks[q, s] = r['toks']
        print(f'  {name}@n{n} T={T} おわり（{time.time() - t0:.0f} 秒）')
    np.savez_compressed(path + '.part.npz', toks=toks, temps=np.array(A7_TEMPS), use=np.array(use), eos=np.array(eos),
                        tok_len=len(tok), prompt_len=list(plen)[0], prompts=np.array(prompts, np.int32),
                        humans=np.array(humans, np.int32), seconds=time.time() - t0, H2=H2, n=n,
                        **{f'g_{k}': v for k, v in G.items()})
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()
    notify(f'{name}@n{n} おわり（{time.time() - t0:.0f} 秒）')


@torch.no_grad()
def self_test_a27(device):
    """小さなランダムのモデルで：n ＝ 4 が A2-4 の block と同じトークン・同じ ℓ を出すか。n ＝ 2 で 2 トークンの写しが 0 か。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global CONT
    keep = CONT
    CONT = 40
    try:
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4]
        for T in [0.5, 1.0]:
            g1 = torch.Generator(device=device).manual_seed(3)
            a = generate_pen(model, prompts, T, valid, g1, 33, 'block')
            g2 = torch.Generator(device=device).manual_seed(3)
            b = generate_ngram(model, prompts, T, valid, g2, 33, 4)
            assert (a['toks'] == b['toks']).all(), ('n ＝ 4 が A2-4 の block と違うトークンを出した', T)
            assert np.allclose(a['ell'], b['ell']) and (a['cand'] == b['cand']).all()
            for n in (2, 3, 6):
                g = torch.Generator(device=device).manual_seed(0)
                r = generate_ngram(model, prompts, T, valid, g, 33, n)
                for bb in range(2):
                    seq = list(prompts[bb]) + list(r['toks'][bb])
                    seen = set()
                    for i in range(n - 1, len(seq)):
                        gram = tuple(seq[i - n + 1:i + 1])
                        if i >= len(prompts[bb]):
                            assert gram not in seen, ('写しが残っている', n, T, bb, i)
                        seen.add(gram)
        ci = CopyIndexN([1, 2, 3, 1, 2], 3)
        assert set(ci.candidates()) == {3}
        assert all(T in TEMPS for T in A7_TEMPS)
    finally:
        CONT = keep
    return 'OK'
