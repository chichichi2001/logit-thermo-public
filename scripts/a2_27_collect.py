"""A2-27 の採点し直し（GPU、L4）：A2-19 の罰なしの腕 c0 の文を、同じモデルにもう一度通し、歩ごとに語彙全体からエネルギーの見込みなどを出して保存する。
ゲート：docs/A2_27_GATE.md。生成はしない。
scripts/a2_2_collect.py …… a2_19_collect.py を実行したあとの同じ名前空間で実行する（load_model・valid_mask・split_doc・CopyIndexN・generate_dsc2・
A18_HF・A19_TOKEN・GP_STALE・CONT・load_docs19 を使う）。名前は rs27_ で始める（a2_7_collect.py の self_test_a27 と重ねない）。

保存する量（どれも (7 温度, 4 サンプル, 40 記事, 512 歩)、float32。温度 T はその文を作ったときの温度）：
  Ux ＝ −log p_1(x_t)（温度 1、使えないトークンを外した softmax。保存してある U と同じ定義）
  Uc ＝ Σ_v p_T(v)(−log p_1(v))、zx ＝ z_t(x_t)、zc ＝ Σ_v p_T(v) z_t(v)、lseT ＝ log Σ_v exp(z_t(v)/T)、q0r ＝ Σ_{v∈C_t} p_T(v)
"""
import gc, os, time, json, socket
import numpy as np
import torch

RS27_B = 8  # 前向きの計算を一度に通す文の数
RS27_TOL_U_MAX, RS27_TOL_U_MEAN, RS27_TOL_Q = 0.01, 1e-4, 1e-3
RS27_KEYS = ('Ux', 'Uc', 'zx', 'zc', 'lseT', 'q0r')


@torch.no_grad()
def rs27_step_stats(logits_f32, xs, T, valid, prompt, cn=3):
    """1 本の文：logits (L, V)（歩 t の次のトークンを決める logit）、xs (L) の選んだトークン → 歩ごとの 6 つの量 (L,)。
    生成（generate_dsc2）と同じく、logit は double にしてから使えないトークンを −∞ にする。"""
    z = logits_f32.double().masked_fill(~valid, float('-inf'))
    lp1 = torch.log_softmax(z, dim=-1)
    zT = z / T
    lseT = torch.logsumexp(zT, dim=-1)
    pT = torch.exp(zT - lseT[:, None])
    xi = xs[:, None]
    del zT
    Ux = -lp1.gather(-1, xi)[:, 0]
    U1 = torch.where(valid, -lp1, torch.zeros_like(lp1))  # 使えないトークンは pT ＝ 0 なので、0 を入れて 0 × ∞ を避ける
    del lp1
    Uc = (pT * U1).sum(-1)
    del U1
    zv = torch.where(valid, z, torch.zeros_like(z))
    zx = z.gather(-1, xi)[:, 0]
    zc = (pT * zv).sum(-1)
    del zv
    idx = CopyIndexN(prompt, cn)  # 生成と同じ候補の集合（直前の 2 トークン → それまで続いたトークン）
    rows, cols = [], []
    for t, x in enumerate(xs.tolist()):
        c = idx.candidates()
        rows += [t] * len(c); cols += list(c)
        idx.push(x)
    q0r = torch.zeros(len(xs), dtype=torch.float64, device=z.device)
    if rows:
        rt = torch.tensor(rows, device=z.device); ct = torch.tensor(cols, device=z.device)
        q0r.index_add_(0, rt, pT[rt, ct])
    return dict(Ux=Ux, Uc=Uc, zx=zx, zc=zc, lseT=lseT, q0r=q0r)


@torch.no_grad()
def rs27_rescore(model, prompts, toks, T, valid):
    """prompts：長さのそろった書き出しのリスト、toks (B, L)：生成した続き → 量ごとに (B, L) の numpy（float32）。
    書き出し ＋ 続きの最後の 1 つを除いたものを 1 回で通し、位置 P − 1 … P ＋ L − 2 の logit を使う（生成で歩 t の次を決めた logit と同じ位置）。"""
    device = valid.device
    P = len(prompts[0]); L = toks.shape[1]
    x = torch.tensor([list(p) + list(map(int, tt[:-1])) for p, tt in zip(prompts, toks)], device=device)
    logits = model(x, use_cache=False).logits[:, P - 1:P - 1 + L, :].float()
    out = {k: np.zeros((len(prompts), L), np.float32) for k in RS27_KEYS}
    for b in range(len(prompts)):
        r = rs27_step_stats(logits[b], torch.tensor(toks[b], device=device, dtype=torch.long), T, valid, prompts[b])
        for k in RS27_KEYS:
            out[k][b] = r[k].float().cpu().numpy()
    del logits
    return out


def rs27_jobs(names):
    """BLOOM（語彙がいちばん大きく、メモリがいちばん要る）を先に回す。"""
    names = sorted(names, key=lambda n: n != 'bloom-1b1')
    return [(n, g) for n in names for g in range(3)]


def rs27_release(outdir, name, tag):
    """自分の印の lock だけを消す（落ちたとき、ほかのランタイムが 2 時間待たずに取り直せるように）。"""
    lk = f'{outdir}/locks/{name}@{tag}.lock'
    try:
        if json.load(open(lk)).get('token') == A19_TOKEN:
            os.remove(lk)
    except Exception:
        pass


def rs27_claim(outdir, name, tag):
    """a19_claim と同じ決まり（終わっていない・失敗の印がない・誰も取っていないか 2 時間より古い → 取る）。"""
    if os.path.exists(f'{outdir}/{name}@{tag}.npz') or os.path.exists(f'{outdir}/locks/{name}@{tag}.failed'):
        return False
    lk = f'{outdir}/locks/{name}@{tag}.lock'
    if os.path.exists(lk):
        try:
            age = time.time() - json.load(open(lk))['t']
        except Exception:
            age = 0.0
        if age < GP_STALE:
            return False
    json.dump(dict(token=A19_TOKEN, host=socket.gethostname(), t=time.time()), open(lk, 'w'))
    time.sleep(20)
    try:
        return json.load(open(lk)).get('token') == A19_TOKEN
    except Exception:
        return False


def rs27_job(name, hf_id, g, groups, srcdir, outdir, device, cache):
    """1 つの仕事（モデル 1 本 × 記事の組 1 つ）。→ (秒, 確かめの要約)。確かめに通らなければ AssertionError（中身の数は文面に入れない）。"""
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(hf_id, device)
        assert next(model.parameters()).dtype == torch.float32
        valid, _ = valid_mask(tok, model, device)
        cache.update(name=name, tok=tok, model=model, valid=valid)
    tok, model, valid = cache['tok'], cache['model'], cache['valid']
    Z = np.load(f'{srcdir}/{name}@c0_g{g}.npz')
    assert abs(float(Z['h'])) < 1e-12 and int(Z['group']) == g and str(Z['arm']) == 'c0', (name, g, '罰なしの腕ではありません')
    temps = [float(t) for t in Z['temps']]
    toks, U0, q00 = Z['toks'], Z['U'], Z['q0']
    nT, S, nD, L = toks.shape
    assert L == CONT and nD == 40 and S == 4, (name, g, toks.shape)
    assert bool(valid[torch.tensor(toks.reshape(-1).astype(np.int64), device=device)].all()), (name, g, '使えないトークンが入っています')
    pairs = [split_doc(tok, d['text']) for d in groups[g]]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert use == [int(u) for u in Z['use']], (name, g, '使う記事が生成のときと違います')
    prompts = [pairs[i][0] for i in use]
    ta = time.time()
    R = {k: np.zeros((nT, S, nD, L), np.float32) for k in RS27_KEYS}
    for k, T in enumerate(temps):
        for s in range(S):
            for i0 in range(0, nD, RS27_B):
                ii = list(range(i0, min(nD, i0 + RS27_B)))
                r = rs27_rescore(model, [prompts[i] for i in ii], toks[k, s, ii], T, valid)
                for kk in RS27_KEYS:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    dU = np.abs(R['Ux'].astype(np.float64) - U0.astype(np.float64)); dq = np.abs(R['q0r'].astype(np.float64) - q00.astype(np.float64))
    chk = dict(U_max=float(dU.max()), U_mean=float(dU.mean()), q_max=float(dq.max()))
    ok = chk['U_max'] <= RS27_TOL_U_MAX and chk['U_mean'] <= RS27_TOL_U_MEAN and chk['q_max'] <= RS27_TOL_Q
    path = f'{outdir}/{name}@c0_g{g}.npz'
    if not ok:
        json.dump(dict(chk, name=name, group=g), open(f'{outdir}/locks/{name}@c0_g{g}.failed', 'w'))
        raise AssertionError((name, g, '前提の確かめ：出し直した U か q0 が保存してあるものと合いません（ずれの要約は locks の .failed に書きました）'))
    np.savez_compressed(path + '.part.npz', temps=np.array(temps), group=g, use=np.array(use), seconds=time.time() - ta, batch=RS27_B, **chk, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta, chk


def rs27_run(names, hf, worker, nworkers, groups, srcdir, outdir, device, notify=print):
    """終わっていない仕事を取りに行く。→ (このランタイムで終えた数, まだ残っている仕事)。"""
    jobs = rs27_jobs(names)
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for name, g in order:
            tag = f'c0_g{g}'
            if rs27_claim(outdir, name, tag):
                took = True
                try:
                    sec, chk = rs27_job(name, hf[name], g, groups, srcdir, outdir, device, cache)
                except Exception:
                    if not os.path.exists(f'{outdir}/locks/{name}@{tag}.failed'):
                        rs27_release(outdir, name, tag)  # 確かめに通らなかった仕事は .failed を残し、ほかの落ち方では lock を外す
                    raise
                n += 1
                notify(f'{name}@{tag} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
        if not took:
            break
    left = [f'{nm}@c0_g{g}' for nm, g in jobs if not os.path.exists(f'{outdir}/{nm}@c0_g{g}.npz')]
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    return n, left


@torch.no_grad()
def self_test_rs27(device):
    """小さなランダムのモデルで：generate_dsc2（h ＝ 0、温度一定）で作った文を採点し直すと、U と q0 が生成のときと合い、
    Uc・zc・lseT が手で出したものと合う。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert B7_N == 3, ('生成の候補の集合の長さが 3 ではありません', B7_N)
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    g = torch.Generator(device=device).manual_seed(7)
    prompts = [torch.randint(0, len(tok), (16,), generator=torch.Generator().manual_seed(i)).tolist() for i in range(3)]
    L, T = 40, 0.7
    r = generate_dsc2(model, prompts, np.full(L, T), np.zeros(L), valid, g)
    out = rs27_rescore(model, prompts, r['toks'], T, valid)
    assert np.abs(out['Ux'] - r['U']).max() < 1e-4, np.abs(out['Ux'] - r['U']).max()
    assert np.abs(out['q0r'] - r['q0']).max() < 1e-5, np.abs(out['q0r'] - r['q0']).max()
    # 手で：1 本目の歩 5
    x = torch.tensor([prompts[0] + list(map(int, r['toks'][0, :-1]))], device=device)
    z = model(x).logits[0, len(prompts[0]) - 1 + 5].float().double().masked_fill(~valid, float('-inf'))
    pT = torch.softmax(z / T, -1); lp1 = torch.log_softmax(z, -1)
    m = valid
    assert abs(float((pT[m] * -lp1[m]).sum()) - out['Uc'][0, 5]) < 1e-4
    assert abs(float((pT[m] * z[m]).sum()) - out['zc'][0, 5]) < 1e-4
    assert abs(float(torch.logsumexp(z / T, -1)) - out['lseT'][0, 5]) < 1e-4
    # −log p_T(x) ＝ −z_x/T ＋ lseT
    lpT = torch.log_softmax(z / T, -1)[int(r['toks'][0, 5])]
    assert abs(float(-lpT) - (-out['zx'][0, 5] / T + out['lseT'][0, 5])) < 1e-4
    del model; gc.collect(); torch.cuda.empty_cache()
    return 'OK'
