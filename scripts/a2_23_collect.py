"""A2-23 の材料集め：前もっての予言を、(A) 実際に使われる割る型の罰（Hugging Face の repetition penalty）と、(D) 指示調整済みのモデルで確かめる
（2026-11-02 JST、GPU（L4）、fp32、TF32 なし）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py・a2_gpre_collect.py・
a2_18_collect.py・a2_19_collect.py・a2_20_collect.py を実行したあとの同じ名前空間で実行する
（load_model・valid_mask・split_doc・CONT・PROMPT・CopyIndexN・B7_N・generate_dsc2・GP_S・GP_CHUNK・gp_seed・A18_HF・A18_JOBS・a19_claim・a19_seed・A20_T1・a20_seed1 を使う）。

(A) 割る型の罰（Hugging Face の RepetitionPenaltyLogitsProcessor と同じ）：それまでに出たトークン（書き出しも含む）の logit を、正なら θ で割り、負なら θ を掛ける。
    そのあと温度 T で割ってサンプリングする。θ ＝ e^λ と書くと、λ ＝ 0 での logit の微分は −|z_v|（それまでに出たトークン v だけ）。
    恒等式の重み：1 歩ごとに A_t ＝ a(x_t) − Σ_v p_T(v) a(v)、a(v) ＝ −|z_v|・[v はそれまでに出た]。d E[Y]/dλ ＝ Cov(Y, Σ_t A_t) ÷ T。
  第 1 段（rs）：罰なしの文（G-pre の h ＝ 0 の腕、A2-19 の c0 の腕）を読み直して A_t を出す（生成しない）。読み直しの U・q0 を保存してある値と比べて、ずれを記録する。
  第 2 段（m1）：予言した θ* で、A2-19 の記事 120 本・A2-18 の温度の格子で生成（対照は A2-19 の c0 をそのまま使う）。
(D) 指示調整済みのモデル（Qwen2.5-1.5B-Instruct・SmolLM2-1.7B-Instruct）、今までの足す型の罰 h：
  書き出しはチャットの形：user「Continue this news article:\\n\\n」＋ 記事の最初の 64 トークン（そのまま差し込む）、そのあとにアシスタントの返事の始まり。
  第 1 段（s1）：A2-2 の記事 40 本、h ＝ 0、温度 0.20〜0.90 の 15 個（A2-20 の第 1 段と同じ）。→ a2_20_predict の予言の手順で h* と格子。
  第 2 段（c0・h1）：A2-19 の記事 120 本、格子は予言のもの。
種：(A) の m1 は A2-19 の c0 と同じ種（共通の乱数）。(D) はこれまでの種に A23_SEED_OFF を足して重ならないようにする（c0 と h1 は同じ種）。
"""
import gc, os, time, json
import numpy as np
import torch

A23_A_HF = dict(A18_HF)                                                    # 小さい 7 本（基本モデル）
A23_D_HF = {'qwen2.5-1.5b-it': 'Qwen/Qwen2.5-1.5B-Instruct', 'smollm2-1.7b-it': 'HuggingFaceTB/SmolLM2-1.7B-Instruct'}
A23_SEED_OFF = 7_000_000
A23_CHAT_TEXT = 'Continue this news article:\n\n'
A23_MARK = '@@ARTICLE_SLOT_7319@@'                                       # テンプレートの中で記事を差し込む位置の目印（ふつうの文には出てこない文字列）
A23_MED_DU, A23_MAX_DU = 1e-3, 0.05                                        # 読み直しの U のずれ（中央値・最大）がこれを超えたら止まる（同じトークン列を読ませているかの確かめ）


def a23_mult(z, seen, theta):
    """Hugging Face の repetition penalty と同じ：seen（B, V の真偽）のトークンの logit を、正なら θ で割り、負なら θ を掛ける。−inf はそのまま。"""
    pen = torch.where(z > 0, z / theta, z * theta)
    return torch.where(seen, pen, z)


def chat_prompt(tok, art_ids):
    """チャットの形の書き出し：テンプレートを目印で 2 つに割り、記事のトークン列をそのまま差し込む（長さが記事によらずそろう）。"""
    s = tok.apply_chat_template([{'role': 'user', 'content': A23_CHAT_TEXT + A23_MARK}], add_generation_prompt=True, tokenize=False)
    pre, suf = s.split(A23_MARK)
    pre_ids = tok(pre, add_special_tokens=False).input_ids
    suf_ids = tok(suf, add_special_tokens=False).input_ids
    if tok.bos_token_id is not None and art_ids and art_ids[0] == tok.bos_token_id:
        art_ids = art_ids[1:]
    return pre_ids + list(art_ids) + suf_ids


@torch.no_grad()
def generate_mult(model, prompts, T, theta, valid, gen):
    """一定の温度 T・割る型の罰 θ で生成。手順と乱数の使い方は generate_dsc2 と同じ（罰のかけ方だけが違う）。
    記録：toks・cand（写しの候補があるか）・pick（候補を選んだか）・U（罰なし・温度 1 の −ln p）・q0（罰なし・温度 T での候補の確率）・q（罰ありの候補の確率）。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    V = logits.shape[-1]
    seen = torch.zeros(B, V, dtype=torch.bool, device=device)
    seen.scatter_(1, x, True)
    rec = {k: [] for k in ('toks', 'cand', 'pick', 'U', 'q0', 'q')}
    for t in range(CONT):
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
        zp = a23_mult(z, seen, theta) if theta != 1.0 else z
        p = torch.softmax(zp / T, dim=-1)
        q = torch.zeros(B, dtype=torch.float64, device=device)
        if rows:
            q.index_add_(0, rt, p[rt, ct])
        nxt = torch.multinomial(p, 1, generator=gen)
        pk = torch.tensor([int(nxt[b, 0]) in set(cands[b]) for b in range(B)], device=device)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        seen.scatter_(1, nxt, True)
        for k, v in (('toks', nxt[:, 0]), ('cand', has), ('pick', pk), ('U', -lp1.gather(-1, nxt)[:, 0]), ('q0', q0), ('q', q)):
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda v: torch.stack(v, 1).cpu().numpy()
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    return {k: st(v).astype(dt[k]) for k, v in rec.items()}


@torch.no_grad()
def rescore_mult(model, prompts, conts, T, valid):
    """罰なしの文（書き出し ＋ 生成した 512 トークン）を一度に読み直し、各歩の A_t ＝ a(x_t) − Σ p_T(v) a(v)（a(v) ＝ −|z_v|・[それまでに出た]）と、
    確かめ用の U（温度 1 の −ln p）・q0（温度 T での写しの候補の確率）を返す。生成のときの logit（キャッシュで 1 歩ずつ）と、ここでの logit（一度に読む）は fp32 の丸めの差だけ違う。"""
    device = valid.device
    B, P = len(prompts), len(prompts[0])
    x = torch.tensor([[int(v) for v in list(p) + list(c)] for p, c in zip(prompts, conts)], device=device, dtype=torch.long)
    logits = model(x, use_cache=False).logits[:, P - 1:P - 1 + CONT, :].float()               # 歩 t のトークンを予言する logit
    V = logits.shape[-1]
    seen = torch.zeros(B, V, dtype=torch.bool, device=device)
    seen.scatter_(1, x[:, :P], True)
    idx = [CopyIndexN(p, B7_N) for p in prompts]
    A, U, Q0 = (np.zeros((B, CONT)) for _ in range(3))
    for t in range(CONT):
        z = logits[:, t, :].double().masked_fill(~valid, float('-inf'))
        y = x[:, P + t]
        lp1 = torch.log_softmax(z, dim=-1)
        p = torch.softmax(z / T, dim=-1)
        a = torch.where(seen & valid[None, :], -z.abs(), torch.zeros_like(z))   # 外したトークン（−inf）は 0（p も 0）
        A[:, t] = (a.gather(1, y[:, None])[:, 0] - (p * a).sum(-1)).cpu().numpy()
        U[:, t] = (-lp1.gather(1, y[:, None])[:, 0]).cpu().numpy()
        for b in range(B):
            c = idx[b].candidates()
            if c:
                Q0[b, t] = float(p[b, list(c)].sum())
            idx[b].push(int(y[b]))
        seen.scatter_(1, y[:, None], True)
    return A, U, Q0


def a23_loader(name, device, cache):
    if cache.get('name') == name:
        return cache['tok'], cache['model'], cache['valid']
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    hf = {**A23_A_HF, **A23_D_HF}[name]
    tok, model = load_model(hf, device)
    assert all(p.dtype == torch.float32 for p in model.parameters()), (name, '重みが fp32 ではない')
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.get_float32_matmul_precision() == 'highest'
    valid, eos = valid_mask(tok, model, device)
    cache.update(name=name, tok=tok, model=model, valid=valid)
    return tok, model, valid


def a23_prompts(name, tok, docs):
    """記事 40 本 → 書き出しのトークン列 40 本（長さがそろう）と、使った記事の番号。基本モデルは A2-19 と同じ（split_doc のまま）、指示調整版はチャットの形。"""
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert len(use) == 40, (name, '使える記事が 40 本ない', len(use))
    prompts = [chat_prompt(tok, pairs[i][0]) for i in use] if name in A23_D_HF else [pairs[i][0] for i in use]
    assert len({len(p) for p in prompts}) == 1, (name, '書き出しの長さがそろわない')
    return prompts, use


def a23_seed(k, s, c, g, kind='gen_add'):
    """(A) の m1 は A2-19 の c0 と同じ種（共通の乱数。θ ＝ 1 の割る型の生成は足す型 h ＝ 0 とビット単位で同じなので、対照と同じ乱数の流れになる）。
    (D) の仕事は前の種に A23_SEED_OFF を足す（c0 と h1 は同じ種 ＝ 共通の乱数）。"""
    if kind == 'gen_mult':
        return a19_seed(k, s, c, g)
    return (a20_seed1(k, s, c) if g is None else a19_seed(k, s, c, g)) + A23_SEED_OFF


def a23_tag(job):
    kind, name, arm, g = job[0], job[1], job[2], job[5]
    return arm if g is None else f'{arm}_g{g}'


def run_a23_job(job, docs, srcdir, outdir, device, cache, chunk):
    """job ＝ (種類, 名前, 腕, 強さ, 温度, 組)。種類：'rs'（読み直し）、'gen_mult'（割る型の罰で生成）、'gen_add'（足す型の罰で生成）。
    'rs' の srcdir は読み直す npz のフォルダ（G-pre なら A2_gpre、A2-19 の c0 なら A2_19）。"""
    kind, name, arm, strength, temps, g = job
    tok, model, valid = a23_loader(name, device, cache)
    prompts, use = a23_prompts(name, tok, docs)
    maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
    assert maxpos is None or len(prompts[0]) + CONT <= maxpos, ('文が長すぎます', maxpos)
    nD = len(use)
    chunks = [list(range(i, min(nD, i + chunk))) for i in range(0, nD, chunk)]
    ta = time.time()
    tag = a23_tag(job)
    import transformers
    meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
                commit=getattr(model.config, '_commit_hash', None), hf={**A23_A_HF, **A23_D_HF}[name], prompt_len=len(prompts[0]), kind=kind)
    if kind == 'rs':
        src = f'{srcdir}/{name}@{"h0" if g is None else f"c0_g{g}"}.npz'
        Z = np.load(src)
        assert list(Z['use']) == use and abs(float(Z['h'])) < 1e-12, (name, '読み直す文の記事が合わない')
        assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in temps], (name, '温度が合わない')
        toks0, U00, q00 = Z['toks'], Z['U'], Z['q0']                      # npz から 1 回だけ読む
        R = {k: np.zeros((len(temps), GP_S, nD, CONT), np.float32) for k in ('A', 'dU', 'dq0')}
        Asum = np.zeros((len(temps), GP_S, nD))
        rchunks = [list(range(i, min(nD, i + 5))) for i in range(0, nD, 5)]    # 一度に読むので、logit（文の長さ × 語彙）が大きい。5 本ずつ
        first = True
        for k, T in enumerate(temps):
            for s in range(GP_S):
                for ii in rchunks:
                    A, U, Q0 = rescore_mult(model, [prompts[i] for i in ii], [toks0[k, s, i] for i in ii], T, valid)
                    R['A'][k, s, ii] = A; Asum[k, s, ii] = A.sum(-1); R['dU'][k, s, ii] = U - U00[k, s, ii]; R['dq0'][k, s, ii] = Q0 - q00[k, s, ii]
                    if first:                                              # 書き出しがずれていれば、最初のかたまりで大きくずれる。早めに止める
                        assert float(np.median(np.abs(U - U00[k, s, ii]))) < A23_MED_DU, (name, tag, '読み直しの U が最初のかたまりから合わない', float(np.median(np.abs(U - U00[k, s, ii]))))
                        first = False
                    torch.cuda.empty_cache()
        med, mx = float(np.median(np.abs(R['dU']))), float(np.abs(R['dU']).max())
        assert med < A23_MED_DU and mx < A23_MAX_DU, (name, tag, '読み直しの U が保存した値とずれすぎ', med, mx)
        meta.update(src=os.path.basename(src), median_dU=med, max_dU=mx, max_dq0=float(np.abs(R['dq0']).max()))
        out = dict(A_sum=Asum, A=R['A'], dU_max=np.abs(R['dU']).max(-1), dq0_max=np.abs(R['dq0']).max(-1))
    else:
        dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
        out = {k: np.zeros((len(temps), GP_S, nD, CONT), v) for k, v in dt.items()}
        for k, T in enumerate(temps):
            for s in range(GP_S):
                for c, ii in enumerate(chunks):
                    gen = torch.Generator(device=device).manual_seed(a23_seed(k, s, c, g, kind))
                    if kind == 'gen_mult':
                        r = generate_mult(model, [prompts[i] for i in ii], T, strength, valid, gen)
                    else:
                        r = generate_dsc2(model, [prompts[i] for i in ii], np.full(CONT, T), np.full(CONT, strength), valid, gen)
                    for kk in out:
                        out[kk][k, s, ii] = r[kk]
                    torch.cuda.empty_cache()
    path = f'{outdir}/{name}@{tag}.npz'
    np.savez_compressed(path + '.part.npz', temps=np.array(temps), strength=strength, h=strength if kind == 'gen_add' else (0.0 if kind == 'rs' else np.nan), kind=kind, arm=arm, group=-1 if g is None else g, use=np.array(use),
                        seconds=time.time() - ta, chunk=chunk, meta=np.array(json.dumps(meta)), **out)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def a23_jobs1():
    """第 1 段：(A) 読み直し 7 本 ×（G-pre 1 ＋ A2-19 の c0 3 組）＝ 28、(D) 罰なしの融点さがし 2。"""
    J = {(n, a): (h, t) for n, a, h, t in A18_JOBS}
    rs = []
    for n in A23_A_HF:
        rs.append(('rs', n, 'rs_gpre', 1.0, [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7], None))
        for g in range(3):
            rs.append(('rs', n, 'rs_c0', 1.0, J[(n, 'c0')][1], g))
    d = [('gen_add', n, 's1', 0.0, A20_T1, None) for n in A23_D_HF]
    return [d[0]] + rs[:14] + [d[1]] + rs[14:]                              # 長い 2 つを表の 0 番と 15 番に分ける（ランタイムのずらし方で別々に取られる）


def a23_jobs2(pred_mult_path, pred_it_path):
    """第 2 段：(A) 予言した θ* で 7 本 × 3 組、(D) 予言した h* で 2 本 × 腕 c0・h1 × 3 組。予言できなかったモデルは入れない。"""
    jobs = []
    PA = json.load(open(pred_mult_path))['models']
    for n, r in PA.items():
        if r.get('ok'):
            for g in range(3):
                jobs.append(('gen_mult', n, 'm1', round(float(r['theta']), 4), [float(t) for t in r['grid']], g))
    PD = json.load(open(pred_it_path))['models']
    for n, r in PD.items():
        if r.get('ok'):
            h, grid = round(float(r['target']['hstar']), 4), [float(t) for t in r['target']['grid']]
            for arm, hh in (('c0', 0.0), ('h1', h)):
                for g in range(3):
                    jobs.append(('gen_add', n, arm, hh, grid, g))
    return jobs


def run_a23(jobs, worker, nworkers, docs_of, srcdir_of, outdir, device, chunk, notify=print):
    """仕事を「まだ終わっていない・誰も取っていない」ものから取りに行く（取った印は a19_claim）。docs_of(job) → 記事 40 本、srcdir_of(job) → 読み直す npz のフォルダ。"""
    os.makedirs(outdir, exist_ok=True)
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for job in order:
            if a19_claim(outdir, job[1], a23_tag(job)):
                took = True
                try:
                    sec = run_a23_job(job, docs_of(job), srcdir_of(job), outdir, device, cache, chunk)
                    n += 1
                    notify(f'{job[1]}@{a23_tag(job)} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{job[1]}@{a23_tag(job)}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {job[1]}@{a23_tag(job)} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = [f'{j[1]}@{a23_tag(j)}' + ('（失敗）' if os.path.exists(f'{outdir}/locks/{j[1]}@{a23_tag(j)}.failed') else '')
            for j in jobs if not os.path.exists(f'{outdir}/{j[1]}@{a23_tag(j)}.npz')]
    return n, left


def self_test_a23(device='cuda'):
    """小さな作り物のモデルで：(1) θ ＝ 1 の割る型の生成が generate_dsc2（h ＝ 0）とビット単位で同じ、(2) 割る型の罰が Hugging Face の RepetitionPenaltyLogitsProcessor と同じ、
    (3) 読み直しの U・q0 が生成のときの値と一致し、A_t の和が「θ を少し動かした文の対数確率の微分」の数値微分と一致、(4) 仕事の表と種、(5) チャットの書き出しの長さがそろう（トークナイザーがあれば）。"""
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=50, n_positions=CONT + 64, n_embd=32, n_layer=2, n_head=2)
    m = GPT2LMHeadModel(cfg).to(device).eval().float()
    valid = torch.ones(50, dtype=torch.bool, device=device); valid[0] = False
    prompts = [list(np.random.default_rng(i).integers(1, 50, 16)) for i in range(3)]
    g1 = torch.Generator(device=device).manual_seed(5); g2 = torch.Generator(device=device).manual_seed(5)
    r1 = generate_mult(m, prompts, 0.6, 1.0, valid, g1); r2 = generate_dsc2(m, prompts, np.full(CONT, 0.6), np.zeros(CONT), valid, g2)
    assert all(np.array_equal(r1[k], r2[k]) for k in r1), '割る型の生成（θ ＝ 1）が足す型（h ＝ 0）と違う'
    # (2) Hugging Face の処理と同じ
    z = torch.randn(2, 50, device=device, dtype=torch.float64); ids = torch.tensor([[1, 2, 3], [4, 5, 5]], device=device)
    seen = torch.zeros(2, 50, dtype=torch.bool, device=device); seen.scatter_(1, ids, True)
    hf = RepetitionPenaltyLogitsProcessor(1.3)(ids, z.clone())
    assert torch.allclose(a23_mult(z, seen, 1.3), hf), 'Hugging Face の repetition penalty と違う'
    # (3) 読み直し：U・q0 の一致と、A の和 ＝ d/dλ log P_θ(文)（θ ＝ e^λ、λ ＝ 0）の数値微分 × T
    g3 = torch.Generator(device=device).manual_seed(7)
    T = 0.6; r = generate_mult(m, prompts, T, 1.0, valid, g3)
    A, U, Q0 = rescore_mult(m, prompts, [r['toks'][b] for b in range(3)], T, valid)
    assert np.abs(U - r['U']).max() < 1e-4 and np.abs(Q0 - r['q0']).max() < 1e-4, (np.abs(U - r['U']).max(), np.abs(Q0 - r['q0']).max())
    def hf_logits(toks):
        x = torch.tensor([[int(v) for v in list(p) + list(c)] for p, c in zip(prompts, toks)], device=device, dtype=torch.long); P = len(prompts[0])
        return x, P, m(x, use_cache=False).logits[:, P - 1:P - 1 + CONT, :].double()
    def logP(lam, toks):                                                    # 罰は Hugging Face の処理そのもの（書き出しを含むそれまでの入力に）
        x, P, lg = hf_logits(toks); tot = torch.zeros(3, dtype=torch.float64, device=device)
        proc = RepetitionPenaltyLogitsProcessor(float(np.exp(lam)))
        for t in range(CONT):
            zz = proc(x[:, :P + t], lg[:, t].masked_fill(~valid, float('-inf')).clone())
            tot += torch.log_softmax(zz / T, -1).gather(1, x[:, P + t:P + t + 1])[:, 0]
        return tot.cpu().numpy()
    with torch.no_grad():
        num = (logP(1e-5, r['toks']) - logP(-1e-5, r['toks'])) / 2e-5
    assert np.allclose(num * T, A.sum(1), rtol=1e-4, atol=1e-4), (num * T, A.sum(1))
    # (3') θ ＝ 1.5 で割る型の生成を回し、記録した q（罰ありの候補の確率）が、全体を読んだ logit に Hugging Face の処理をかけて出した q と合う
    g4 = torch.Generator(device=device).manual_seed(9)
    r15 = generate_mult(m, prompts, T, 1.5, valid, g4)
    with torch.no_grad():
        x, P, lg = hf_logits(r15['toks']); proc = RepetitionPenaltyLogitsProcessor(1.5); idx = [CopyIndexN(p, B7_N) for p in prompts]; qq = np.zeros((3, CONT))
        for t in range(CONT):
            p = torch.softmax(proc(x[:, :P + t], lg[:, t].masked_fill(~valid, float('-inf')).clone()) / T, -1)
            for b in range(3):
                c = idx[b].candidates()
                if c:
                    qq[b, t] = float(p[b, list(c)].sum())
                idx[b].push(int(x[b, P + t]))
    assert np.abs(qq - r15['q']).max() < 1e-4, np.abs(qq - r15['q']).max()
    # (4) 仕事の表と種
    j1 = a23_jobs1()
    assert len(j1) == 30 and sum(j[0] == 'rs' for j in j1) == 28 and {j[1] for j in j1 if j[0] == 'gen_add'} == set(A23_D_HF)
    assert j1[0][0] == 'gen_add' and j1[15][0] == 'gen_add'
    assert a23_seed(1, 2, 0, 1, 'gen_mult') == a19_seed(1, 2, 0, 1)                      # (A) は A2-19 の c0 と共通の乱数
    s_new = {a23_seed(k, s, c, g, 'gen_add') for k in range(15) for s in range(GP_S) for c in range(2) for g in (None, 0, 1, 2)}
    s_old = {gp_seed(k, s, c) for k in range(7) for s in range(GP_S) for c in range(2)} | {a19_seed(k, s, c, g) for k in range(7) for s in range(GP_S) for c in range(2) for g in range(3)} \
        | {a20_seed1(k, s, c) for k in range(15) for s in range(GP_S) for c in range(2)}
    assert not (s_new & s_old)
    import tempfile
    d = tempfile.mkdtemp()
    json.dump(dict(models={'pythia-410m': dict(ok=True, theta=1.21234, grid=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]), 'gpt2-1.5b': dict(ok=False)}), open(f'{d}/a.json', 'w'))
    json.dump(dict(models={'qwen2.5-1.5b-it': dict(ok=True, target=dict(hstar=-0.5, grid=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]))}), open(f'{d}/d.json', 'w'))
    j2 = a23_jobs2(f'{d}/a.json', f'{d}/d.json')
    assert len(j2) == 3 + 6 and {a23_tag(j) for j in j2 if j[0] == 'gen_mult'} == {'m1_g0', 'm1_g1', 'm1_g2'} and j2[0][3] == 1.2123
    return 'OK'


def self_test_chat(names=None):
    """（ネットにつながるときだけ）指示調整版のトークナイザーで、チャットの書き出しの長さが記事によらずそろい、記事のトークン列がそのまま入っていること。"""
    from transformers import AutoTokenizer
    for n in (names or A23_D_HF):
        tok = AutoTokenizer.from_pretrained(A23_D_HF[n])
        a, b = list(range(1000, 1064)), list(range(2000, 2064))
        pa, pb = chat_prompt(tok, a), chat_prompt(tok, b)
        assert len(pa) == len(pb), (n, len(pa), len(pb))
        i = next(k for k in range(len(pa)) if pa[k] != pb[k])                  # 記事が始まる位置（テンプレートの前半に 1000 があってもまちがえない）
        assert pa[i:i + 64] == a and pb[i:i + 64] == b and pa[:i] == pb[:i] and pa[i + 64:] == pb[i + 64:], n
    return 'OK'
