"""探索 8.38（下見）：文章の DSC その 2 ——凍らせた結晶を 1.5 まで温めて、本当に溶ける温度と溶けるときの潜熱を測る（2026-10-31 16:5x、GPU（L4）、判定しない）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py を実行したあとの同じ名前空間で実行する
（load_model・valid_mask・split_doc・CopyIndexN・B7_N・generate_field・generate_dsc・D_MODELS・D_CHUNK を変えずに使う）。

どの腕も、はじめの 640 歩は同じ（0.85 → 0.35 で冷やす、場なし、同じ種）。なので、どの腕も「同じように凍った結晶」から温めはじめる（対になった比べ）。
腕：
  'base'     ：冷やす 640 歩 → 温める 0.35 → 1.5 を 640 歩（場なし）
  'hold'     ：冷やす 640 歩 → 0.35 で 320 歩待つ（熟成）→ 温める 640 歩（場なし）
  'fastheat' ：冷やす 640 歩 → 温める 0.35 → 1.5 を 320 歩（場なし）
  'penheat'  ：冷やす 640 歩（場なし）→ 温める 640 歩のあいだだけ h ＝ −1.0（強いくり返しの罰）
記録：トークン、候補があったか、候補を選んだか、U ＝ −ln p（温度 1・場なし、選んだトークン）、q0（場をかける前の、その温度での候補の確率の和）、
  q（場をかけた後の候補の確率の和）、歩ごとの温度と場。確率の計算は倍精度、保存は float32。
"""
import gc, os, time
import numpy as np
import torch

D2_COOL = (0.85, 0.35, 640)
D2_HEAT_TOP = 1.5
D2_ARMS = ['base', 'hold', 'fastheat', 'penheat']
D2_S = 4
D2_SEED = 20261137
D2_PEN = -1.0


def d2_sched(arm):
    """→ (温度の列, 場の列)。"""
    a, b, n = D2_COOL
    cool = np.linspace(a, b, n)
    heat = np.linspace(b, D2_HEAT_TOP, 320 if arm == 'fastheat' else 640)
    parts = [cool] + ([np.full(320, b)] if arm == 'hold' else []) + [heat]
    Ts = np.concatenate(parts)
    Hs = np.zeros_like(Ts)
    if arm == 'penheat':
        Hs[n:] = D2_PEN
    return Ts, Hs


def d2_seed(s, c):
    return D2_SEED + 10 * s + c


@torch.no_grad()
def generate_dsc2(model, prompts, Ts, Hs, valid, gen):
    """温度と場が歩ごとに変わる。手順と乱数の使い方は generate_field・generate_dsc と同じ。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    rec = {k: [] for k in ('toks', 'cand', 'pick', 'U', 'q0', 'q')}
    for t in range(len(Ts)):
        T, h = float(Ts[t]), float(Hs[t])
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp1 = torch.log_softmax(z, dim=-1)
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
        for k, v in (('toks', nxt[:, 0]), ('cand', has), ('pick', pk), ('U', -lp1.gather(-1, nxt)[:, 0]), ('q0', q0), ('q', q)):
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    return {k: st(v).astype(dt[k]) for k, v in rec.items()}


def run_dsc2(name, hf_id, docs, outdir, device, notify=print):
    todo = [a for a in D2_ARMS if not os.path.exists(f'{outdir}/{name}@{a}.npz')]
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
    assert maxpos is None or P.pop() + max(len(d2_sched(a)[0]) for a in D2_ARMS) <= maxpos, ('文が長すぎます', maxpos)
    chunks = [list(range(i, min(nD, i + D_CHUNK))) for i in range(0, nD, D_CHUNK)]
    n_cool = D2_COOL[2]
    cool_ref = None
    for arm in todo:
        ta = time.time()
        Ts, Hs = d2_sched(arm)
        L = len(Ts)
        dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
        R = {k: np.zeros((D2_S, nD, L), v) for k, v in dt.items()}
        for s in range(D2_S):
            for c, ii in enumerate(chunks):
                g = torch.Generator(device=device).manual_seed(d2_seed(s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, g)
                for k in R:
                    R[k][s, ii] = r[k]
                torch.cuda.empty_cache()
        if cool_ref is None:
            cool_ref = R['toks'][:, :, :n_cool].copy()
        else:
            assert (R['toks'][:, :, :n_cool] == cool_ref).all(), (name, arm, 'はじめの冷やす 640 歩が、ほかの腕と違います')
        path = f'{outdir}/{name}@{arm}.npz'
        np.savez_compressed(path + '.part.npz', Ts=Ts, Hs=Hs, arm=arm, use=np.array(use), seconds=time.time() - ta, chunk=D_CHUNK, **R)
        os.replace(path + '.part.npz', path)
        notify(f'{name}@{arm} おわり（{time.time() - ta:.0f} 秒、通算 {time.time() - t0:.0f} 秒）')
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_dsc2(device):
    """小さなランダムのモデルで：(1) 温度・場が一定なら generate_field と同じトークン。(2) 場が 0 なら generate_dsc と同じトークンと U。
    (3) q を読み直しで確かめる。(4) 腕の予定（長さ・温度・場）。"""
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
            a = generate_field(model, prompts, 0.7, valid, torch.Generator(device=device).manual_seed(5), h)
            b = generate_dsc2(model, prompts, np.full(60, 0.7), np.full(60, h), valid, torch.Generator(device=device).manual_seed(5))
            assert (a['toks'] == b['toks']).all() and (a['pick'] == b['pick']).all(), ('generate_field と違う', h)
        Ts = np.linspace(0.9, 0.4, 60)
        a = generate_dsc(model, prompts, Ts, 0.0, valid, torch.Generator(device=device).manual_seed(6))
        b = generate_dsc2(model, prompts, Ts, np.zeros(60), valid, torch.Generator(device=device).manual_seed(6))
        assert (a['toks'] == b['toks']).all() and np.allclose(a['U'], b['U']), 'generate_dsc と違う'
        assert np.array_equal(b['q'], b['q0'])
        # 場が途中から入る：q ≥ q0（候補がある歩）、読み直しの q と合う
        Hs = np.r_[np.zeros(30), np.full(30, -1.0)]
        b = generate_dsc2(model, prompts, Ts, Hs, valid, torch.Generator(device=device).manual_seed(7))
        assert (b['q'][:, 30:][b['cand'][:, 30:]] <= b['q0'][:, 30:][b['cand'][:, 30:]] + 1e-6).all()
        seq = b['toks'][0].tolist()
        x = torch.tensor([prompts[0] + seq], device=device)
        lg = model(x).logits[0, len(prompts[0]) - 1: len(prompts[0]) - 1 + 60].double().masked_fill(~valid, float('-inf'))
        ci = CopyIndexN(prompts[0], B7_N)
        for t in range(60):
            c = list(ci.candidates())
            zz = lg[t].clone()
            if c and Hs[t] != 0.0:
                zz[c] += Hs[t]
            pp = torch.softmax(zz / Ts[t], -1)
            qq = float(pp[c].sum()) if c else 0.0
            assert abs(qq - float(b['q'][0, t])) < 2e-3, (t, qq, float(b['q'][0, t]))
            ci.push(seq[t])
    finally:
        CONT = keep
    for arm in D2_ARMS:
        Ts, Hs = d2_sched(arm)
        assert abs(Ts[0] - 0.85) < 1e-12 and abs(Ts[639] - 0.35) < 1e-12 and abs(Ts[-1] - D2_HEAT_TOP) < 1e-12
        assert (Hs[:640] == 0).all() and ((Hs[640:] == D2_PEN).all() if arm == 'penheat' else (Hs == 0).all())
    assert [len(d2_sched(a)[0]) for a in D2_ARMS] == [1280, 1600, 960, 1280]
    return 'OK'
