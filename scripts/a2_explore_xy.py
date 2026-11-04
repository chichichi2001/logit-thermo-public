"""探索 X（型破り）と Y（文の順番の戻れなさ）の下見。2026-10-28 夜。確かめには使わない。
GPU（L4）のノートブックで、scripts/a2_2_collect.py を実行したのと同じ名前空間で実行する
（load_docs・split_doc・load_model・valid_mask・step_stats・h2_of・PROMPT・CONT を使う）。生成はしない（人の文章を読むだけ）。

モデル：A2-2 の素のモデル 4 つ（Pythia-1B、Qwen2.5-1.5B、OLMo-2-1B、SmolLM2-1.7B）。記事：A2-2 と同じ BBC の 40 本。

X（型破り）：A2-2 と同じ書き出し 64 ＋ 人の続き 512 トークンで、温度 1 の分布について、各歩の
  エントロピー H（選ぶ前の迷い、ナット）、サプライザル s ＝ −log p（実際の一歩の意外さ、ナット）、ℓ（H ＝ 100）、1 位の log-odds、
  人のトークンの順位、を記録する（xy_{モデル}.npz に保存）。
  読み方（回す前に決める。探索の目安）：
  - 型破り歩：H < 1.0 かつ s > H ＋ 3.0（迷いは小さいのに、意外さが期待値より 3 ナット以上大きい）。
  - 分かれ道の歩：H > 3.0。
  - 記録する量：型破り歩の割合、型破り歩が NLL（s の合計）に占める割合、分かれ道の歩の割合と NLL に占める割合、
    位置の帯（1〜8・8〜32・32〜128・128〜512）ごとの型破り歩の割合。

Y（文の順番の戻れなさ）：記事の頭から段落を足していき、合わせて 2,500 字を超えないところまでを「窓」にする
  （少なくとも 2 段落・6 文）。窓の文を、次の並びにして、固定の書き出し "Article:\\n" のあとに置いた文章の
  対数確率（書き出しのあとの全トークンの log p の合計、温度 1、外した番号は除く）を測る。文はすべて半角スペース 1 つでつなぐ。
  書き出しと本文は別々にトークンに区切ってからつなぐ。
  - original：元の順番。
  - sent_rev：文の順番を逆にする。
  - para_rev：段落の順番を逆にする（段落の中の文の順番はそのまま）。
  - shuf1・shuf2：文の順番をばらばらにする（種 1・2）。
  - 戻れなさ σ ＝ log p（original）− log p（並べ替え）。記事ごとの値（ナット）と、トークンあたりの値。
  読み方（回す前に決める。探索の目安）：
  - 検量線：sent_rev と shuf で σ > 0 の記事が 9 割以上なら「ものさしは使える」。そうでなければ、この探索の言葉は使わない。
  - 粗く見ると小さくなるか：para_rev の σ が sent_rev の σ 以下の記事の割合（確率的な熱力学では、粗く見たときの
    エントロピー生成は下限になる。ここでは「段落の逆順」を「文の逆順」より粗い並べ替えとみなす）。
  - 逆順とばらばらの比べ：σ(sent_rev) と σ(shuf) の平均の比（逆順は、ばらばらより「戻れない」か）。
  文の分け方：段落の中を、. ! ? のあとの空白で分ける（20 字未満の断片は前の文につなぐ）。中身は表示しない・保存しない。
"""
import gc, json, math, re, time
import numpy as np
import torch

XY_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'),
             ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B')]
X_LOW_H, X_JUMP, FORK_H = 1.0, 3.0, 3.0
Y_CHARS, Y_MIN_PARA, Y_MIN_SENT = 2500, 2, 6
Y_PREFIX = 'Article:\n'
X_BINS = [(1, 8), (8, 32), (32, 128), (128, 512)]


def split_sentences(para):
    parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', para) if p.strip()]
    out = []
    for p in parts:
        if out and len(p) < 20:
            out[-1] = out[-1] + ' ' + p
        else:
            out.append(p)
    return out


def y_window(text):
    """→ 段落ごとの文のリスト（list of list）。条件を満たさなければ None。"""
    paras = [p for p in text.split('\n\n') if p.strip()]
    win, n = [], 0
    for p in paras:
        if n + len(p) > Y_CHARS and len(win) >= Y_MIN_PARA:
            break
        win.append(split_sentences(p)); n += len(p)
    if len(win) < Y_MIN_PARA or sum(len(s) for s in win) < Y_MIN_SENT:
        return None
    return win


def y_orders(win):
    sents = [s for p in win for s in p]
    out = {'original': sents, 'sent_rev': sents[::-1], 'para_rev': [s for p in win[::-1] for s in p]}
    for k in [1, 2]:
        r = np.random.default_rng(k).permutation(len(sents))
        out[f'shuf{k}'] = [sents[i] for i in r]
    return {k: ' '.join(v) for k, v in out.items()}


@torch.no_grad()
def seq_logprob(model, tok, text, valid, device):
    pre = tok(Y_PREFIX, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
    ids = pre + tok(text, add_special_tokens=False).input_ids      # 書き出しと本文は別々に区切ってつなぐ（境目で区切りが変わらないように）
    x = torch.tensor([bos + ids], device=device)
    z = model(x).logits[0].double().masked_fill(~valid, float('-inf'))
    lp = torch.log_softmax(z, -1)
    start = len(bos) + len(pre)
    y = x[0, start:]
    tgt = lp[start - 1:x.shape[1] - 1].gather(-1, y[:, None])[:, 0]
    return float(tgt.sum()), int(len(y))


@torch.no_grad()
def x_steps(model, prompt, human, valid, device, H2):
    x = torch.tensor([prompt + human], device=device)
    logits = model(x).logits[0, len(prompt) - 1: len(prompt) - 1 + CONT].float()
    z = logits.double().masked_fill(~valid, float('-inf'))
    lp = torch.log_softmax(z, -1)
    y = torch.tensor(human, device=device)
    s = -lp.gather(-1, y[:, None])[:, 0]
    p = torch.exp(lp)
    Hh = -torch.where(p > 0, p * lp, torch.zeros_like(p)).sum(-1)
    ell, top1, _, _ = step_stats(logits, 1.0, valid, H2)
    rank = (z > z.gather(-1, y[:, None])).sum(-1)
    f = lambda v: v.float().cpu().numpy()
    return dict(H=f(Hh), s=f(s), ell=f(ell), top1=f(top1), rank=rank.int().cpu().numpy())


def x_summary(X):
    H, s = X['H'].astype(np.float64), X['s'].astype(np.float64)      # (nD, L)
    jump = (H < X_LOW_H) & (s > H + X_JUMP)
    fork = H > FORK_H
    tot = s.sum()
    return dict(jump_rate=float(jump.mean()), jump_share_nll=float(s[jump].sum() / tot),
                fork_rate=float(fork.mean()), fork_share_nll=float(s[fork].sum() / tot),
                jump_rate_by_bin={f'{a}-{b}': float(jump[:, a:b].mean()) for a, b in X_BINS},
                nll_per_token=float(s.mean()), mean_H=float(H.mean()))


def y_summary(Y):
    """Y：記事ごとの {order: (logp, ntok)} のリスト。"""
    out = {}
    lp = {k: np.array([d[k][0] for d in Y]) for k in Y[0]}
    nt = {k: np.array([d[k][1] for d in Y]) for k in Y[0]}
    for k in ['sent_rev', 'para_rev', 'shuf1', 'shuf2']:
        sig = lp['original'] - lp[k]
        out[k] = dict(sigma_mean=float(sig.mean()), sigma_median=float(np.median(sig)),
                      sigma_per_token_mean=float((sig / nt['original']).mean()), frac_positive=float((sig > 0).mean()),
                      ntok_diff_mean=float((nt[k] - nt['original']).mean()))
    shuf_frac = min(out['shuf1']['frac_positive'], out['shuf2']['frac_positive'])
    out['calibration_ok'] = bool(out['sent_rev']['frac_positive'] >= 0.9 and shuf_frac >= 0.9)
    s_sent, s_para = lp['original'] - lp['sent_rev'], lp['original'] - lp['para_rev']
    out['frac_para_le_sent'] = float((s_para <= s_sent).mean())
    s_shuf = ((lp['original'] - lp['shuf1']) + (lp['original'] - lp['shuf2'])) / 2
    out['ratio_rev_over_shuf'] = float(s_sent.mean() / s_shuf.mean()) if s_shuf.mean() != 0 else float('nan')
    out['n_docs'] = len(Y)
    return out


def run_xy(docs, outdir, device, notify=print):
    res = {}
    wins = [y_window(d['text']) for d in docs]
    res['y_docs_used'] = int(sum(w is not None for w in wins))
    for name, hf_id in XY_MODELS:
        path = f'{outdir}/xy_{name}.npz'
        t0 = time.time()
        tok, model = load_model(hf_id, device)
        valid, _ = valid_mask(tok, model, device)
        H2 = h2_of(len(tok))
        pairs = [split_doc(tok, d['text']) for d in docs]
        use = [i for i, p in enumerate(pairs) if p is not None]
        Xs = [x_steps(model, *pairs[i], valid, device, H2) for i in use]
        X = {k: np.stack([x[k] for x in Xs]) for k in Xs[0]}
        Y = []
        for w in wins:
            if w is None:
                continue
            Y.append({k: seq_logprob(model, tok, t, valid, device) for k, t in y_orders(w).items()})
        np.savez_compressed(path, use=np.array(use), **X,
                            y_logp=np.array([[d[k][0] for k in ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2']] for d in Y]),
                            y_ntok=np.array([[d[k][1] for k in ['original', 'sent_rev', 'para_rev', 'shuf1', 'shuf2']] for d in Y]))
        res[name] = dict(x=x_summary(X), y=y_summary(Y), seconds=round(time.time() - t0, 1))
        del model; gc.collect(); torch.cuda.empty_cache()
        notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')
    return res


@torch.no_grad()
def self_test_xy(device):
    """小さなランダムのモデルで：文の分け方と並べ替え、seq_logprob と x_steps が素直な計算と合うか。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # 文の分け方・窓・並べ替え
    para = 'One sentence here. Two is here! Three? ok. Four is long enough here.'
    ss = split_sentences(para)
    assert ss[-1].endswith('here.') and all(len(s) >= 9 for s in ss), ss
    text = '\n\n'.join(['Alpha one is a sentence. Alpha two is another one. Alpha three ends here.'] * 2 +
                       ['Beta one is a sentence. Beta two is another one.'])
    w = y_window(text)
    assert w is not None and len(w) == 3
    od = y_orders(w)
    assert sorted(od['original'].split()) == sorted(od['sent_rev'].split()) == sorted(od['shuf1'].split())
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    lp, n = seq_logprob(model, tok, od['original'], valid, device)
    pre_ids = tok(Y_PREFIX, add_special_tokens=False).input_ids
    ids = pre_ids + tok(od['original'], add_special_tokens=False).input_ids
    z = model(torch.tensor([ids], device=device)).logits[0].double().masked_fill(~valid, float('-inf'))
    ls = torch.log_softmax(z, -1)
    pre = len(pre_ids)
    ref = sum(float(ls[i - 1, ids[i]]) for i in range(pre, len(ids)))
    assert abs(lp - ref) < 1e-6 * max(1, abs(ref)) and n == len(ids) - pre, (lp, ref)
    global CONT
    keep = CONT
    CONT = 20
    try:
        prompt, human = list(range(5, 13)), list(range(20, 40))
        X = x_steps(model, prompt, human, valid, device, 33)
        full = model(torch.tensor([prompt + human], device=device)).logits[0].double().masked_fill(~valid, float('-inf'))
        ls = torch.log_softmax(full, -1)
        for t in [0, 7, 19]:
            assert abs(float(X['s'][t]) + float(ls[len(prompt) - 1 + t, human[t]])) < 1e-4
        assert np.all(X['H'] >= 0) and np.all(X['s'] >= 0)
        sm = x_summary({k: v[None] for k, v in X.items()})
        assert 0 <= sm['jump_rate'] <= 1
    finally:
        CONT = keep
    return 'OK'
