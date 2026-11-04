"""A2-12（ゲート docs/A2_12_GATE.md）：1 回の禁止で乱れるトークンの数 k を測り、凝固点降下と熱量計の ΔH のずれから先に決めた k の予想と比べる。
GPU（L4）で生成、そのあと同じノートブックで分析（CPU の計算）。
scripts/a2_2_collect.py（load_model・valid_mask・split_doc・load_docs・SEED）と scripts/a2_7_collect.py（CopyIndexN）を実行したのと同じ名前空間で実行する。
分析の部分はデータを開く前に固定する。

やり方：
- 書き出し（A2-2 と同じ 40 本の記事の先頭 64 トークン）から、温度 T でガンベルの最大（ノイズ g を足して一番大きいトークンを選ぶ。
  ふつうのサンプリングと同じ分布）で 256 歩の「下書き」を作る。
- 分かれ目 t0 ＝ 64・128・192・256 のそれぞれで、下書きの t0 歩目までを共通にして 2 本に分ける。
  a：何もしない。b：t0 歩目だけ、1 位のトークンを禁止する（A2-10 の偽薬と同じく、1 位が 3 トークンの写しの候補でないときだけ。
  候補なら、その分かれ目は数えない）。
- 分かれ目から 64 歩、2 本で同じノイズ g（歩ごと・記事ごと）を使う（共通の乱数）。分布がほとんど同じならトークンも同じになる。
- k ＝ その 64 歩で 2 本のトークンが食い違った歩の数（分かれ目の歩を含む）。a が分かれ目で 1 位を選ばなかったときは、b も同じトークンを
  選ぶので、その後もずっと同じ（k ＝ 0）。これも A2-10 の偽薬の 1 回の禁止と同じ（濃さは禁止の回数で数えている）。
- 記録：歩ごとの 2 本の分布の全変動距離（TV、0〜1）とその和、歩ごとの食い違いの割合（曲線）、分かれ目での 1 位の確率。
"""
import gc, math, os, time
import numpy as np
import torch

K12_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')]
K12_TEMPS = [0.5, 0.55, 0.6]
K12_T_MAIN = 0.55
K12_FORKS = [64, 128, 192, 256]
K12_W = 64
K12_PREFIX = 256
K12_SEED = 20261119
# 予想の k（A2-10 §4 と A2-8 の数から、データを見る前に決まっている）：k_pred ＝ 熱量計の ΔH ÷ 凝固点降下の ΔH（x ＝ 濃さ）
K12_DH_CAL = {'pythia-1b': 1.2785, 'olmo2-1b': 1.3645, 'qwen2.5-1.5b': 1.4053}      # A2-8（ナット／トークン）
K12_DH_COL = {'pythia-1b': 1.5301, 'olmo2-1b': 0.764344, 'qwen2.5-1.5b': 0.721599}  # A2-10（ナット、x ＝ 濃さ）
K12_PRED = {m: K12_DH_CAL[m] / K12_DH_COL[m] for m in K12_DH_CAL}
K12_BAND = 1.5
K12_BOOT, K12_BOOT_SEED = 1000, 20261120
K12_Q2_BINS = [(1.0, '1 未満'), (2.0, '1〜2'), (4.0, '2〜4'), (float('inf'), '4 以上')]


def prefill(model, x):
    """最後の位置の logit だけ作る（全部の位置の logit を作ると、Qwen では数十 GB になる）。"""
    try:
        return model(x, use_cache=True, logits_to_keep=1)
    except TypeError:
        return model(x, use_cache=True)


def gumbel(shape, seed, device):
    g = torch.Generator(device=device).manual_seed(int(seed))
    u = torch.rand(shape, generator=g, device=device, dtype=torch.float64).clamp_(1e-12, 1 - 1e-12)
    return -torch.log(-torch.log(u))


def choose(z, T, valid, g):
    """z：(B, V) の logit。→ ガンベルの最大で選んだトークン (B,)。z/T ＋ g の最大 ＝ softmax(z/T) からのサンプル。"""
    s = (z.double() / T).masked_fill(~valid, float('-inf')) + g
    return s.argmax(-1)


@torch.no_grad()
def make_prefix(model, prompts, T, valid, seed):
    """→ (B, K12_PREFIX) の下書き（何も禁止しない）。"""
    device = valid.device
    x = torch.tensor(prompts, device=device)
    out = prefill(model, x)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    toks = []
    for t in range(K12_PREFIX):
        nxt = choose(logits, T, valid, gumbel(logits.shape, seed + t, device))
        toks.append(nxt)
        out = model(nxt[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    return torch.stack(toks, 1).cpu().numpy()


@torch.no_grad()
def fork_once(model, prompts, prefix, t0, T, valid, seed, ban=True):
    """t0 で分ける。→ dict(elig (B,), diff (B, W), tv (B, W), p1 (B,), chose1 (B,))。ban＝False なら b も禁止しない（自己テスト用）。"""
    device = valid.device
    B = len(prompts)
    hist = [list(prompts[i]) + list(map(int, prefix[i, :t0])) for i in range(B)]
    x = torch.tensor(hist + hist, device=device)                        # 2B 本：前半が a、後半が b
    out = prefill(model, x)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    z0 = logits[:B].double().masked_fill(~valid, float('-inf'))
    top1 = z0.argmax(-1)
    cand = [CopyIndexN(h, 3).candidates() for h in hist]
    elig = np.array([int(top1[i]) not in set(cand[i]) for i in range(B)])
    p1 = torch.softmax(z0 / T, -1).gather(-1, top1[:, None])[:, 0].cpu().numpy()
    diff = np.zeros((B, K12_W), np.int8); tv = np.zeros((B, K12_W), np.float32); chose1 = np.zeros(B, bool)
    for t in range(K12_W):
        za = logits[:B].double().masked_fill(~valid, float('-inf'))
        zb = logits[B:].double().masked_fill(~valid, float('-inf'))
        if t == 0 and ban:
            rows = torch.tensor(np.flatnonzero(elig), device=device, dtype=torch.long)
            if len(rows):
                zb[rows, top1[rows]] = float('-inf')
        pa, pb = torch.softmax(za / T, -1), torch.softmax(zb / T, -1)
        tv[:, t] = (0.5 * (pa - pb).abs().sum(-1)).float().cpu().numpy()
        g = gumbel((B, za.shape[-1]), seed + t, device)
        ta = (za / T + g).argmax(-1); tb = (zb / T + g).argmax(-1)
        if t == 0:
            chose1 = (ta == top1).cpu().numpy()
        diff[:, t] = (ta != tb).cpu().numpy()
        nxt = torch.cat([ta, tb])
        out = model(nxt[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    return dict(elig=elig, diff=diff, tv=tv, p1=p1, chose1=chose1)


def run_k12(name, hf_id, docs, outdir, device, notify=print):
    path = f'{outdir}/{name}.npz'
    if os.path.exists(path):
        print(name, 'はもう終わっています（飛ばします）')
        return
    t0_ = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, _ = valid_mask(tok, model, device)
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    prompts = [pairs[i][0] for i in use]
    assert len({len(p) for p in prompts}) == 1
    nT, nF, B = len(K12_TEMPS), len(K12_FORKS), len(prompts)
    save = {k: np.zeros((nT, nF, B) + s, dt) for k, s, dt in [('elig', (), bool), ('chose1', (), bool), ('p1', (), np.float32),
                                                               ('diff', (K12_W,), np.int8), ('tv', (K12_W,), np.float32)]}
    prefixes = np.zeros((nT, B, K12_PREFIX), np.int32)
    for k, T in enumerate(K12_TEMPS):
        pre = make_prefix(model, prompts, T, valid, K12_SEED + 100000 * k)
        prefixes[k] = pre
        for f, t0 in enumerate(K12_FORKS):
            r = fork_once(model, prompts, pre, t0, T, valid, K12_SEED + 100000 * k + 10000 * (f + 1))
            for kk in save:
                save[kk][k, f] = r[kk]
        notify(f'{name} 温度 {T} おわり（{time.time() - t0_:.0f} 秒）')
    np.savez_compressed(path + '.part.npz', prefixes=prefixes, use=np.array(use), temps=np.array(K12_TEMPS), forks=np.array(K12_FORKS), **save)
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def self_test_k12(device):
    """小さなランダムのモデルで：(1) 禁止しなければ 2 本はずっと同じ。(2) a が分かれ目で 1 位を選ばなかった行は、禁止しても 2 本はずっと同じ。
    (3) a が 1 位を選んだ禁止できる行は、分かれ目で必ず食い違う。(4) ガンベルの最大の選び方の頻度が softmax に近い。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    global K12_W, K12_PREFIX
    keep = (K12_W, K12_PREFIX)
    K12_W, K12_PREFIX = 12, 20
    try:
        prompts = [[5 + i, 6, 7, 8] * 4 for i in range(16)]
        pre = make_prefix(model, prompts, 0.3, valid, 1)
        r0 = fork_once(model, prompts, pre, 10, 0.3, valid, 2, ban=False)
        assert r0['diff'].sum() == 0 and np.abs(r0['tv']).max() < 1e-4
        r = fork_once(model, prompts, pre, 10, 0.3, valid, 2, ban=True)
        same = ~(r['elig'] & r['chose1'])
        assert r['diff'][same].sum() == 0
        r = fork_once(model, prompts, pre, 10, 0.001, valid, 2, ban=True)        # ほぼ貪欲：a は必ず 1 位を選ぶ
        same = ~(r['elig'] & r['chose1'])
        assert r['diff'][same].sum() == 0
        hit = r['elig'] & r['chose1']
        assert hit.any() and (r['diff'][hit, 0] == 1).all()
        # ガンベルの最大：小さな分布で頻度を確かめる
        z = torch.tensor([[2.0, 1.0, 0.0, -1.0]], device=device).repeat(20000, 1)
        v = torch.ones(4, dtype=torch.bool, device=device)
        c = choose(z, 1.0, v, gumbel(z.shape, 3, device)).cpu().numpy()
        p = torch.softmax(z[0].double(), -1).cpu().numpy()
        assert np.abs(np.bincount(c, minlength=4) / len(c) - p).max() < 0.015
    finally:
        K12_W, K12_PREFIX = keep
    return 'OK'


# ---------- 分析（CPU） ----------

def k_stats(Z, T, boot_seed=K12_BOOT_SEED):
    """→ 温度 T での k（禁止できた分かれ目の食い違いの数の平均）と、記事を選び直した 95% 区間。"""
    k = list(Z['temps']).index(T)
    elig = Z['elig'][k]                                   # (nF, B)
    dsum = Z['diff'][k].astype(np.float64).sum(-1)        # (nF, B)
    tvs = Z['tv'][k].astype(np.float64).sum(-1)
    B = elig.shape[1]
    num = (dsum * elig).sum(0); den = elig.sum(0).astype(float)
    numt = (tvs * elig).sum(0)
    kp = float(num.sum() / den.sum()) if den.sum() else float('nan')
    rng = np.random.default_rng(boot_seed)
    bs = []
    for _ in range(K12_BOOT):
        ii = rng.integers(0, B, B)
        if den[ii].sum():
            bs.append(num[ii].sum() / den[ii].sum())
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    curve = ((Z['diff'][k].astype(np.float64) * elig[..., None]).sum((0, 1)) / den.sum()).tolist()
    return dict(k=kp, ci=ci, k_tv=float(numt.sum() / den.sum()), n_forks=int(den.sum()), elig_frac=float(elig.mean()),
                chose1_frac=float((Z['chose1'][k] & elig).sum() / den.sum()), p1_mean=float(Z['p1'][k][elig].mean()),
                curve=curve, late16=float(np.mean(curve[-16:])))


def analyze12(Zs):
    res = {'models': {}, 'pred': K12_PRED}
    for name, _ in K12_MODELS:
        Z = Zs[name]
        row = {str(T): k_stats(Z, T) for T in K12_TEMPS}
        m = row[str(K12_T_MAIN)]
        pr = K12_PRED[name]
        m['within'] = bool(pr / K12_BAND <= m['k'] <= pr * K12_BAND)
        res['models'][name] = row
    res['KQ1'] = sum(res['models'][n][str(K12_T_MAIN)]['within'] for n, _ in K12_MODELS)
    med = float(np.median([res['models'][n][str(K12_T_MAIN)]['k'] for n, _ in K12_MODELS]))
    res['k_median'] = med
    res['KQ2'] = next(w for e, w in K12_Q2_BINS if med < e)
    return res


def self_test_a12():
    """作り物：分かれ目で 1 位を選ぶ確率 0.6、選んだら平均 3 歩食い違う → k ≈ 1.8。予想の帯の判定。"""
    rng = np.random.default_rng(0)
    nT, nF, B = len(K12_TEMPS), len(K12_FORKS), 40
    Zs = {}
    for name, _ in K12_MODELS:
        elig = rng.random((nT, nF, B)) < 0.8
        chose = rng.random((nT, nF, B)) < 0.6
        diff = np.zeros((nT, nF, B, K12_W), np.int8)
        L = rng.geometric(1 / 3, (nT, nF, B))
        for idx in np.ndindex(nT, nF, B):
            if elig[idx] and chose[idx]:
                diff[idx][:min(L[idx], K12_W)] = 1
        Zs[name] = dict(temps=np.array(K12_TEMPS), elig=elig, chose1=chose & elig, p1=np.full((nT, nF, B), 0.6, np.float32),
                        diff=diff, tv=diff.astype(np.float32) * 0.5)
    r = analyze12(Zs)
    k = r['models']['pythia-1b'][str(K12_T_MAIN)]['k']
    assert 1.4 < k < 2.2, k
    assert r['KQ1'] == 2 and r['KQ2'] == '1〜2', (r['KQ1'], r['KQ2'], {n: r['models'][n][str(K12_T_MAIN)]['k'] for n, _ in K12_MODELS})
    assert abs(K12_PRED['pythia-1b'] - 0.8356) < 1e-3 and abs(K12_PRED['olmo2-1b'] - 1.7852) < 1e-3 and abs(K12_PRED['qwen2.5-1.5b'] - 1.9475) < 1e-3
    return 'OK'
