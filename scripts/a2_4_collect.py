"""A2-4（ゲート v0.2、凍結 2026-10-28 18:28）のデータ集め。GPU（L4）のノートブックで、scripts/a2_2_collect.py を
実行したのと同じ名前空間でこのファイルを実行する（A2-2 の関数をそのまま使う）。

つまみ（ゲート §2）：
- 'none'：つまみなし（λ ＝ 0）。
- 'block'：写しの禁止（λ ＝ ∞）。直前の 3 トークン (a, b, c) がそれまでの文（書き出しを含む）に出ていたら、
  そのとき続いたトークン d を禁止する（logit を −∞）。no_repeat_ngram_size ＝ 4 と同じ。
- 'placebo'：偽薬。写しかけ歩（禁止の候補がある歩）でない歩から歩を選び、その歩のつまみ前の 1 位を禁止する。
  全体の禁止の割合が p(T) になるように、各歩で「ここまでに禁止しておきたい数の残り」（owed：毎歩 p(T) × 行の数を足し、
  実際に禁止した数を引く）を、写しかけ歩でない行それぞれに確率 min(1, owed ÷ 写しかけ歩でない行の数) で割り当てる
  （足りなかった分は次の歩に持ち越す。データを開く前に決めた細部。ゲート §6 の 4 の確かめはこの割合で行う）。
  p(T) は 'block' の生成で「禁止したトークンがつまみ前の 1 位だった歩」の割合（温度ごと、全記事・全サンプル・全歩）。
  歩の選び方の乱数は、生成の乱数と別の種（numpy、PLACEBO_SEED）で決める。
つまみは温度で割る前の logit にかける。

物差し（ゲート §3）：ℓ・1 位の log-odds・エントロピー・ℓ(H2) は、つまみをかける前の logit から計算する（本命）。
記録として、つまみをかけたあとの logit から ℓ（H ＝ 100）も計算する。
歩ごとの印：写しかけ歩か（cand）、禁止の候補のつまみ前の確率の合計（cand_mass）、実際に禁止したか（banned）、
禁止したトークンにつまみ前の 1 位が入っていたか（banned_top1）、実際に禁止した確率の合計（ban_mass、つまみ前）。
確率はその温度の分布（logit ÷ T の softmax、外した番号を除く）で測る。
人の文脈は A2-2 と同じ（つまみはかからない）。人の文のトークン（書き出しと続き）も保存する（分析で写しかけの印を作るため）。
判定はしない（分析は scripts/a2_4_analyze.py）。
"""
import gc, json, math, os, time
import numpy as np
import torch

SEED4 = 20261029
PLACEBO_SEED = 20261030
MODES = ['none', 'block', 'placebo']
CONDITIONS = [  # 名前, Hugging Face の ID
    ('pythia-1b', 'EleutherAI/pythia-1b'),
    ('olmo2-1b', 'allenai/OLMo-2-0425-1B'),
    ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'),
]


class CopyIndex:
    """1 本の文の「3 トークン → それまでに続いたトークンの集合」。"""

    def __init__(self, hist):
        self.seq = [int(t) for t in hist]
        self.nxt = {}
        for i in range(3, len(self.seq)):
            self.nxt.setdefault(tuple(self.seq[i - 3:i]), set()).add(self.seq[i])

    def candidates(self):
        if len(self.seq) < 3:
            return ()
        return tuple(self.nxt.get(tuple(self.seq[-3:]), ()))

    def push(self, t):
        t = int(t)
        if len(self.seq) >= 3:
            self.nxt.setdefault(tuple(self.seq[-3:]), set()).add(t)
        self.seq.append(t)


@torch.no_grad()
def generate_pen(model, prompts, T, valid, gen, H2, mode, p_placebo=0.0, rng=None):
    """prompts：同じ長さのトークン列のリスト。512 歩を KV キャッシュで書く。戻り値は dict（numpy）。"""
    device = valid.device
    B = len(prompts)
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    idx = [CopyIndex(p) for p in prompts]
    owed = 0.0
    rec = {k: [] for k in ['ell', 'top1', 'ent', 'ell2', 'ell_pen', 'cand_mass', 'ban_mass', 'cand', 'banned', 'banned_top1', 'toks']}
    for t in range(CONT):
        ell, top1, ent, ell2 = step_stats(logits, T, valid, H2)          # つまみ前の logit（本命）
        z = logits.double().masked_fill(~valid, float('-inf'))
        lp = torch.log_softmax(z / T, dim=-1)                             # その温度の分布（つまみ前）
        argmax = z.argmax(-1)
        cands = [idx[b].candidates() for b in range(B)]
        cand = torch.tensor([len(c) > 0 for c in cands], device=device)
        cm = torch.zeros(B, dtype=torch.float64, device=device)
        rows, cols = [], []
        for b, c in enumerate(cands):
            if c:
                cc = torch.tensor(c, device=device)
                cm[b] = torch.exp(torch.logsumexp(lp[b, cc], 0))
                if mode == 'block':
                    rows += [b] * len(c); cols += list(c)
        if mode == 'placebo':
            nc = ~cand.cpu().numpy()
            owed += p_placebo * B                                         # この歩までに禁止しておきたい数の残り
            q = min(1.0, owed / nc.sum()) if nc.any() else 0.0
            pick = (rng.random(B) < q) & nc
            owed -= int(pick.sum())
            for b in np.nonzero(pick)[0]:
                rows.append(int(b)); cols.append(int(argmax[b]))
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
        ell_pen = step_stats(zp.float(), T, valid, H2)[0]                 # 記録：つまみ後の logit の ℓ
        p = torch.softmax(zp / T, dim=-1)
        nxt = torch.multinomial(p, 1, generator=gen)
        for b in range(B):
            idx[b].push(int(nxt[b, 0]))
        for k, v in [('ell', ell), ('top1', top1), ('ent', ent), ('ell2', ell2), ('ell_pen', ell_pen), ('cand_mass', cm),
                     ('ban_mass', ban_mass), ('cand', cand), ('banned', banned), ('banned_top1', banned_top1), ('toks', nxt[:, 0])]:
            rec[k].append(v)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    res = {}
    for k, v in rec.items():
        a = torch.stack(v, 1)
        res[k] = a.cpu().numpy().astype(np.int32 if k == 'toks' else (np.bool_ if a.dtype == torch.bool else np.float32))
    return res


def run_mode(name, hf_id, docs, outdir, device, mode, notify=print):
    path = f'{outdir}/{name}@{mode}.npz'
    if os.path.exists(path):
        print(name, mode, 'はもう終わっています（飛ばします）')
        return
    p_pl = None
    if mode == 'placebo':
        B = np.load(f'{outdir}/{name}@block.npz')
        p_pl = B['g_banned_top1'].reshape(B['g_banned_top1'].shape[0], -1).mean(1)     # (nT,)
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, eos = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    nT, nD = len(TEMPS), len(use)
    plen = {len(pairs[i][0]) for i in use}
    assert len(plen) == 1, plen
    prompts = [pairs[i][0] for i in use]
    humans = [pairs[i][1] for i in use]
    tf = np.zeros((4, nD, nT, CONT), np.float32)
    if mode == 'none':   # 人の文脈（A2-2 と同じ。つまみはかからない）は 'none' でだけ計算する
        for j, i in enumerate(use):
            res, a, h, tc = teacher_forced(model, *pairs[i], valid, device, H2)
            for k in range(nT):
                for q in range(4):
                    tf[q, j, k] = res[k][q]
    keys_f = ['ell', 'top1', 'ent', 'ell2', 'ell_pen', 'cand_mass', 'ban_mass']
    keys_b = ['cand', 'banned', 'banned_top1']
    G = {k: np.zeros((nT, SAMPLES, nD, CONT), np.float32) for k in keys_f}
    G.update({k: np.zeros((nT, SAMPLES, nD, CONT), np.bool_) for k in keys_b})
    toks = np.zeros((nT, SAMPLES, nD, CONT), np.int32)
    for k, T in enumerate(TEMPS):
        for s in range(SAMPLES):
            g = torch.Generator(device=device).manual_seed(SEED4 + 1000 * k + s)
            rng = np.random.default_rng(PLACEBO_SEED + 1000 * k + s)
            r = generate_pen(model, prompts, T, valid, g, H2, mode, 0.0 if p_pl is None else float(p_pl[k]), rng)
            for kk in keys_f + keys_b:
                G[kk][k, s] = r[kk]
            toks[k, s] = r['toks']
        print(f'  {name}@{mode} T={T} おわり（{time.time() - t0:.0f} 秒）')
    extra = {} if p_pl is None else {'p_placebo': p_pl}
    np.savez_compressed(path + '.part.npz', tf=tf, toks=toks, temps=np.array(TEMPS), use=np.array(use), eos=np.array(eos),
                        tok_len=len(tok), prompt_len=list(plen)[0], prompts=np.array(prompts, np.int32),
                        humans=np.array(humans, np.int32), revision=str(getattr(model.config, '_commit_hash', '')),
                        seconds=time.time() - t0, H2=H2, mode=mode, **{f'g_{k}': v for k, v in G.items()}, **extra)
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()
    notify(f'{name}@{mode} おわり（{time.time() - t0:.0f} 秒）')


@torch.no_grad()
def self_test_pen(device):
    """小さなランダムのモデルで：つまみ前の logit を記録しているか、禁止が効いているか、偽薬が写しかけ歩を避けるか。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    H2 = 33
    global CONT
    keep = CONT
    CONT = 40
    try:
        # くり返しの多い書き出し（写しかけ歩が出やすい）
        prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4]
        for T in [0.5, 1.0]:
            g = torch.Generator(device=device).manual_seed(0)
            r = generate_pen(model, prompts, T, valid, g, H2, 'block')
            # ① 写し歩が 0（4 トークンの並びがそれまでに出ていない）
            for b in range(2):
                seq = list(prompts[b]) + list(r['toks'][b])
                seen = set()
                for i in range(3, len(seq)):
                    gram = tuple(seq[i - 3:i + 1])
                    if i >= len(prompts[b]):
                        assert gram not in seen, ('写しが残っている', T, b, i)
                    seen.add(gram)
            assert r['cand'].any(), '写しかけ歩が 1 つもない（確かめにならない）'
            # ② 記録した ℓ は、つまみ前の logit と一致する（同じバッチの形で KV キャッシュで流し直し、相対 1e-6。ゲート §6 の 2）。
            #    まとめて計算し直した値とも近い（絶対 1e-3）。禁止した歩ではつまみ後と違う
            xx = torch.tensor(prompts, device=device)
            o = model(xx, use_cache=True); pk, lg = o.past_key_values, o.logits[:, -1, :].float()
            for t in range(CONT):
                e = step_stats(lg, T, valid, H2)[0].cpu().numpy()
                assert np.allclose(e, r['ell'][:, t], rtol=1e-6, atol=1e-9), ('流し直した ℓ と一致しない', T, t)
                nx = torch.tensor(r['toks'][:, t:t + 1], device=device, dtype=torch.long)
                o = model(nx, past_key_values=pk, use_cache=True); pk, lg = o.past_key_values, o.logits[:, -1, :].float()
            for b in range(2):
                seq = torch.tensor([list(prompts[b]) + list(r['toks'][b])], device=device)
                full = model(seq).logits[0, len(prompts[b]) - 1: len(prompts[b]) - 1 + CONT].float()
                ref = step_stats(full, T, valid, H2)[0].cpu().numpy()
                assert np.allclose(ref, r['ell'][b], atol=1e-3), ('つまみ前の ℓ と一致しない', T, b)
                m = r['banned'][b] & (r['ban_mass'][b] > 1e-3)
                if m.any():
                    assert np.all(np.abs(r['ell_pen'][b][m] - r['ell'][b][m]) > 1e-6), ('禁止した歩でつまみ後の ℓ が同じ', T, b)
            # ③ 偽薬：写しかけ歩では禁止しない。p ＝ 1 なら、写しかけ歩でない歩は全部 1 位を禁止する
            g = torch.Generator(device=device).manual_seed(0)
            rp = generate_pen(model, prompts, T, valid, g, H2, 'placebo', 1.0, np.random.default_rng(0))
            assert not (rp['banned'] & rp['cand']).any(), '偽薬が写しかけ歩で禁止した'
            assert (rp['banned'] == ~rp['cand']).all() and (rp['banned_top1'] == rp['banned']).all(), '偽薬の禁止が狙いどおりでない'
            # ④ つまみなし：何も禁止しない。1 歩目のつまみ前の ℓ は 3 つで同じ
            g = torch.Generator(device=device).manual_seed(0)
            rn = generate_pen(model, prompts, T, valid, g, H2, 'none')
            assert not rn['banned'].any()
            assert np.allclose(rn['ell'][:, 0], r['ell'][:, 0]) and np.allclose(rn['ell'][:, 0], rp['ell'][:, 0])
    finally:
        CONT = keep
    return 'OK'


@torch.no_grad()
def self_test_run_mode(device, tmpdir='/tmp/a2_4_selftest'):
    """小さなランダムのモデルで run_mode を 3 つのつまみで通しで回し、分析が読むキーがそろっているか確かめる。"""
    import shutil
    global PROMPT, CONT, TEMPS, shape_stats
    keep = (PROMPT, CONT, TEMPS, shape_stats)
    PROMPT, CONT, TEMPS = 8, 20, [0.7, 1.0]
    shape_stats = lambda z, valid: (torch.zeros(z.shape[0]), torch.zeros(z.shape[0]))   # 小さな語彙では順位 300〜3,000 がない
    shutil.rmtree(tmpdir, ignore_errors=True); os.makedirs(tmpdir)
    try:
        rng = np.random.default_rng(0)
        words = ['alpha', 'beta', 'gamma', 'delta', 'eps', 'zeta', 'eta', 'theta']
        docs = [{'text': ' '.join(rng.choice(words, 200))} for _ in range(3)]
        hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
        for mode in MODES:
            run_mode('tiny', hf_id, docs, tmpdir, device, mode)
        need = {'tf', 'toks', 'temps', 'use', 'prompts', 'humans', 'g_ell', 'g_top1', 'g_ent', 'g_ell2', 'g_ell_pen',
                'g_cand', 'g_cand_mass', 'g_banned', 'g_banned_top1', 'g_ban_mass'}
        for mode in MODES:
            Z = np.load(f'{tmpdir}/tiny@{mode}.npz')
            miss = need - set(Z.files)
            assert not miss, (mode, miss)
            assert Z['g_ell'].shape == (2, SAMPLES, Z['use'].shape[0], 20), Z['g_ell'].shape
        assert 'p_placebo' in np.load(f'{tmpdir}/tiny@placebo.npz').files
        assert np.abs(np.load(f'{tmpdir}/tiny@none.npz')['tf']).sum() > 0
    finally:
        PROMPT, CONT, TEMPS, shape_stats = keep
        shutil.rmtree(tmpdir, ignore_errors=True)
    return 'OK'
