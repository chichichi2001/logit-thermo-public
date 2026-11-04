"""探索 8.40（下見）：PROJECT_G ——くり返しの罰を「どこにかけるか」で、同じ副作用あたりの効き目は変わるか（2026-11-01 06:3x JST、GPU（L4）を 3 つ、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py を実行したあとの同じ名前空間で実行する
（load_model・valid_mask・split_doc・CONT・CopyIndexN・B7_N を変えずに使う）。

罰 ＝ 写しの候補（3 つ組の続き）の logit に h（負）を足す（8.30 の場と同じ）。1 本の文ごとに、続けて候補を選んだ歩の数 k（8.37 のはしご）を見て、かけるかを決める：
  'none'   ：かけない
  'all'    ：いつも（k によらず、候補があればかける）
  'grow'   ：芽を摘む（1 ≤ k ≤ 7：写しを続けはじめた途中だけ。最初の 1 回の写しにはかけない）
  'melt'   ：溶かす（k ≥ 8：結晶になってから）
強さ h ＝ −0.5・−1.0・−2.0。温度 0.5・0.6。記事 40 × サンプル 4、512 歩。種は罰の組によらず同じ（共通の乱数）。
副作用：罰をかけた歩の KL（罰の後の分布 ‖ 罰の前の分布、その温度で）。かけない歩は 0。
記録：トークン、候補があったか、候補を選んだか、U（温度 1・罰なしの −ln p）、q0・q（罰の前・後の候補の確率の和）、KL、罰をかけたか。

仕事の割り振り：30 個の仕事（3 モデル × 10 組）を、ランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行く（G-pre と同じしくみ）。
種は仕事だけで決まるので、2 つのランタイムが同じ仕事をしても結果は同じ。
"""
import gc, os, time, json, socket
import numpy as np
import torch

PK_TEMPS = [0.5, 0.6]
PK_MODES = ['all', 'grow', 'melt']
PK_H = [-0.5, -1.0, -2.0]
PK_S = 4
PK_SEED = 20261144
PK_CHUNK = 20
PK_STALE = 2 * 3600
PK_KGROW = (1, 7)
PK_KMELT = 8
PK_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')]


def pk_configs():
    return [('none', 0.0)] + [(m, h) for m in PK_MODES for h in PK_H]


def pk_tag(mode, h):
    return 'none' if mode == 'none' else f'{mode}_h{int(round(-h * 10)):02d}'


def pk_seed(k, s, c):
    return PK_SEED + 1000 * k + 10 * s + c


def pk_mask(mode, kk):
    """kk：(B,) の続けて写した歩の数 → 罰をかけるか (B,)。"""
    if mode == 'none':
        return torch.zeros_like(kk, dtype=torch.bool)
    if mode == 'all':
        return torch.ones_like(kk, dtype=torch.bool)
    if mode == 'grow':
        return (kk >= PK_KGROW[0]) & (kk <= PK_KGROW[1])
    if mode == 'melt':
        return kk >= PK_KMELT
    raise ValueError(mode)


@torch.no_grad()
def generate_pork(model, prompts, T, h, mode, valid, gen, L=None):
    """一定の温度。罰は 1 本ごとの k で決めた歩だけにかける。手順と乱数の使い方は generate_field・generate_dsc2 と同じ（罰がなければ同じトークン）。"""
    L = CONT if L is None else L
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    kk = torch.zeros(B, dtype=torch.long, device=device)
    rec = {k: [] for k in ('toks', 'cand', 'pick', 'U', 'q0', 'q', 'kl', 'app')}
    for t in range(L):
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp1 = torch.log_softmax(z, dim=-1)
        cands = [idx[b].candidates() for b in range(B)]
        has = torch.tensor([len(c) > 0 for c in cands], device=device)
        rows, cols = [], []
        for b, c in enumerate(cands):
            rows += [b] * len(c); cols += list(c)
        rt = torch.tensor(rows, device=device, dtype=torch.long); ct = torch.tensor(cols, device=device, dtype=torch.long)
        lp0 = torch.log_softmax(z / T, dim=-1)
        q0 = torch.zeros(B, dtype=torch.float64, device=device)
        if rows:
            q0.index_add_(0, rt, lp0.exp()[rt, ct])
        app = pk_mask(mode, kk) & has if (h != 0.0) else torch.zeros(B, dtype=torch.bool, device=device)
        if rows and bool(app.any()):
            sel = app[rt]
            z[rt[sel], ct[sel]] += h
        lp = torch.log_softmax(z / T, dim=-1)
        p = lp.exp()
        q = torch.zeros(B, dtype=torch.float64, device=device)
        if rows:
            q.index_add_(0, rt, p[rt, ct])
        kl = torch.where(app, (p * (lp - lp0)).nan_to_num(0.0).sum(-1), torch.zeros(B, dtype=torch.float64, device=device))
        nxt = torch.multinomial(p, 1, generator=gen)
        pk = torch.tensor([int(nxt[b, 0]) in set(cands[b]) for b in range(B)], device=device)
        kk = torch.where(pk, kk + 1, torch.zeros_like(kk))
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        for k, v in (('toks', nxt[:, 0]), ('cand', has), ('pick', pk), ('U', -lp1.gather(-1, nxt)[:, 0]), ('q0', q0), ('q', q), ('kl', kl), ('app', app)):
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32, kl=np.float32, app=np.bool_)
    return {k: st(v).astype(dt[k]) for k, v in rec.items()}


def pk_jobs():
    return [(name, hf, mode, h) for name, hf in PK_MODELS for mode, h in pk_configs()]


def pk_claim(outdir, tag, worker):
    if os.path.exists(f'{outdir}/{tag}.npz') or os.path.exists(f'{outdir}/locks/{tag}.failed'):
        return False
    lk = f'{outdir}/locks/{tag}.lock'
    if os.path.exists(lk):
        try:
            age = time.time() - json.load(open(lk))['t']
        except Exception:
            age = 0.0
        if age < PK_STALE:
            return False
    os.makedirs(f'{outdir}/locks', exist_ok=True)
    json.dump(dict(worker=worker, host=socket.gethostname(), t=time.time()), open(lk, 'w'))
    time.sleep(20)
    try:
        return json.load(open(lk)).get('worker') == worker
    except Exception:
        return False


def run_pork_job(name, hf_id, mode, h, docs, outdir, device, cache):
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(hf_id, device)
        assert next(model.parameters()).dtype == torch.float32
        valid, eos = valid_mask(tok, model, device)
        pairs = [split_doc(tok, d['text']) for d in docs]
        use = [i for i, p in enumerate(pairs) if p is not None]
        assert len({len(pairs[i][0]) for i in use}) == 1
        cache.update(name=name, model=model, valid=valid, use=use, prompts=[pairs[i][0] for i in use])
    model, valid, use, prompts = cache['model'], cache['valid'], cache['use'], cache['prompts']
    nD = len(use)
    chunks = [list(range(i, min(nD, i + PK_CHUNK))) for i in range(0, nD, PK_CHUNK)]
    ta = time.time()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32, kl=np.float32, app=np.bool_)
    R = {k: np.zeros((len(PK_TEMPS), PK_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(PK_TEMPS):
        for s in range(PK_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(pk_seed(k, s, c))
                r = generate_pork(model, [prompts[i] for i in ii], T, h, mode, valid, g)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    tag = f'{name}@{pk_tag(mode, h)}'
    path = f'{outdir}/{tag}.npz'
    np.savez_compressed(path + '.part.npz', temps=np.array(PK_TEMPS), h=h, mode=mode, use=np.array(use), seconds=time.time() - ta, chunk=PK_CHUNK, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_pork(worker, nworkers, docs, outdir, device, notify=print):
    os.makedirs(outdir, exist_ok=True)
    jobs = pk_jobs()
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for name, hf, mode, h in order:
            tag = f'{name}@{pk_tag(mode, h)}'
            if pk_claim(outdir, tag, worker):
                took = True
                try:
                    sec = run_pork_job(name, hf, mode, h, docs, outdir, device, cache)
                    n += 1
                    notify(f'{tag} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{tag}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {tag} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = []
    for nm, _, mode, h in jobs:
        tag = f'{nm}@{pk_tag(mode, h)}'
        if not os.path.exists(f'{outdir}/{tag}.npz'):
            left.append(tag + ('（失敗）' if os.path.exists(f'{outdir}/locks/{tag}.failed') else ''))
    return n, left


@torch.no_grad()
def self_test_pork(device):
    """小さなランダムのモデルで：(1) 罰なし・'all' は generate_dsc2（一定の温度・一定の場）と同じトークン・q・q0。
    (2) 'grow'・'melt' は、記録した k の条件を満たす歩だけに罰がかかっている。(3) KL は罰をかけた歩だけ正、手で計算し直した値と合う。(4) 仕事の数と名前。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4]
    L = 60
    for mode, h in (('none', 0.0), ('all', -1.0)):
        a = generate_pork(model, prompts, 0.6, h, mode, valid, torch.Generator(device=device).manual_seed(5), L)
        b = generate_dsc2(model, prompts, np.full(L, 0.6), np.full(L, h), valid, torch.Generator(device=device).manual_seed(5))
        assert (a['toks'] == b['toks']).all() and np.allclose(a['q'], b['q'], atol=1e-6) and np.allclose(a['q0'], b['q0'], atol=1e-6), mode
    for mode in ('grow', 'melt'):
        a = generate_pork(model, prompts, 0.6, -2.0, mode, valid, torch.Generator(device=device).manual_seed(6), L)
        k = np.zeros(a['pick'].shape, int); run = np.zeros(a['pick'].shape[0], int)
        for t in range(L):
            k[:, t] = run; run = np.where(a['pick'][:, t], run + 1, 0)
        want = a['cand'] & (((k >= PK_KGROW[0]) & (k <= PK_KGROW[1])) if mode == 'grow' else (k >= PK_KMELT))
        assert (a['app'] == want).all(), mode
        assert (a['kl'][~a['app']] == 0).all() and (a['kl'][a['app']] >= -1e-7).all()
        assert (a['q'][a['app']] <= a['q0'][a['app']] + 1e-6).all()
    # KL を手で
    a = generate_pork(model, prompts, 0.6, -1.0, 'all', valid, torch.Generator(device=device).manual_seed(7), 20)
    seq = a['toks'][0].tolist()
    lg = model(torch.tensor([prompts[0] + seq], device=device)).logits[0, len(prompts[0]) - 1: len(prompts[0]) - 1 + 20].double().masked_fill(~valid, float('-inf'))
    ci = CopyIndexN(prompts[0], B7_N)
    for t in range(20):
        c = list(ci.candidates())
        z0 = lg[t].clone(); z1 = z0.clone()
        if c:
            z1[c] += -1.0
        l0 = torch.log_softmax(z0 / 0.6, -1); l1 = torch.log_softmax(z1 / 0.6, -1)
        klh = float((l1.exp() * (l1 - l0)).nan_to_num(0.0).sum()) if c else 0.0
        assert abs(klh - float(a['kl'][0, t])) < 2e-3, (t, klh, float(a['kl'][0, t]))
        ci.push(seq[t])
    assert len(pk_jobs()) == 30 and len({(n, pk_tag(m, h)) for n, _, m, h in pk_jobs()}) == 30
    return 'OK'
