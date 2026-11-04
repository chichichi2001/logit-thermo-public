"""A2-3（ゲート v0.2、凍結 2026-10-28 14:15）のデータ集め。GPU のノートブックで、scripts/a2_2_collect.py を
実行したのと同じ名前空間でこのファイルを実行する（A2-2 の関数をそのまま使う）。

足すもの：コードの文章の選び方（ゲート §3）、かたまりに分けた生成（7B のメモリ）、読み込む前のメモリの見積もり、
記事ごとの NLL。判定はしない（分析は scripts/a2_3_analyze.py）。
"""
import gc, hashlib, json, math, os, re, time
import numpy as np
import torch

SEED3 = 20261028
CODE_MONTHS = ['2025-04', '2025-05', '2025-06']
# 凍結後の修正（2026-10-28 14:3x、データを見る前。分かっていたのは「3 か月では候補が 16 本」だけ）：
# 足りなければ 2025-03 から過去へ 1 か月ずつ、いちばん古くて 2024-12 まで足す。それでも足りなければ 1 リポジトリ 2 ファイルまで。
# それでも足りなければ、30 本以上ならその本数で回す。30 本未満なら止める。
CODE_MONTHS_MORE = ['2025-03', '2025-02', '2025-01', '2024-12']
CODE_MIN = 30
CODE_EXT = ('.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.go', '.c', '.cc', '.cpp', '.h', '.hpp', '.rs', '.rb', '.php',
            '.cs', '.kt', '.swift', '.scala')
BAD_PATH = ('vendor/', 'node_modules/', 'dist/', 'build/', 'third_party/', '.min.')
LICENSE_RE = re.compile(r'(copyright|license|licensed|spdx)', re.I)
# 名前, Hugging Face の ID, 文章, 一度に生成する本数
CONDITIONS = [
    ('qwen2.5-1.5b@bbc-a100', 'Qwen/Qwen2.5-1.5B', 'bbc', 40),     # A100 の確かめ（A2-2 と同じ記事・同じ種）
    ('pythia-1b@code', 'EleutherAI/pythia-1b', 'code', 40),
    ('smollm2-1.7b@code', 'HuggingFaceTB/SmolLM2-1.7B', 'code', 40),
    ('qwen2.5-1.5b@code', 'Qwen/Qwen2.5-1.5B', 'code', 40),
    ('qwen2.5-7b@bbc', 'Qwen/Qwen2.5-7B', 'bbc', 20),
    ('olmo2-7b@bbc', 'allenai/OLMo-2-1124-7B', 'bbc', 8),
]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def code_ok(path, code):
    """ゲート §3 の残すファイルの条件。"""
    ext = os.path.splitext(path)[1].lower()
    if ext not in CODE_EXT:
        return False
    p = path.lower()
    if any(b in p for b in BAD_PATH):
        return False
    lines = code.split('\n')
    if max(len(l) for l in lines) > 300 or sum(len(l) for l in lines) / len(lines) > 100:
        return False
    if LICENSE_RE.search(code[:2000]):
        return False
    return len(code) >= 4000


def filter_counts(rows):
    """条件ごとに何本残るか（記録用）。"""
    c = {'rows': len(rows), 'has_code': 0, 'ext': 0, 'path': 0, 'lines': 0, 'license': 0, 'length': 0}
    for r in rows:
        code = (r['code'] or '').replace('\r\n', '\n'); path = r['path'] or ''
        if not code:
            continue
        c['has_code'] += 1
        if os.path.splitext(path)[1].lower() not in CODE_EXT:
            continue
        c['ext'] += 1
        if any(b in path.lower() for b in BAD_PATH):
            continue
        c['path'] += 1
        lines = code.split('\n')
        if max(len(l) for l in lines) > 300 or sum(len(l) for l in lines) / len(lines) > 100:
            continue
        c['lines'] += 1
        if LICENSE_RE.search(code[:2000]):
            continue
        c['license'] += 1
        if len(code) >= 4000:
            c['length'] += 1
    return c


def unique_pool(rows, per_repo):
    pool = []
    for r in rows:
        code = (r['code'] or '').replace('\r\n', '\n')
        path, repo = r['path'] or '', r['repo'] or ''
        if code and code_ok(path, code):
            pool.append({'month': r['month'], 'repo': repo, 'path': path, 'text': code, 'key': sha(repo + '\n' + path)})
    pool.sort(key=lambda d: d['key'])
    uniq, seen_c, per = [], set(), {}
    for d in pool:
        c = sha(d['text'])
        if c in seen_c or per.get(d['repo'], 0) >= per_repo:
            continue
        seen_c.add(c); per[d['repo']] = per.get(d['repo'], 0) + 1; uniq.append(d)
    return pool, uniq


def select_code(rows, per_repo=1, allow_fewer=False):
    """rows：{'month','repo','path','code'} の並び → 選んだ記事（最大 40 本）と目録。中身は返すが、目録にはハッシュだけ。"""
    pool, uniq = unique_pool(rows, per_repo)
    if len(uniq) < N_DOCS and not (allow_fewer and len(uniq) >= CODE_MIN):
        raise RuntimeError(f'コードの候補が {len(uniq)} 本しかありません（{N_DOCS} 本必要）')
    n = min(N_DOCS, len(uniq))
    rng = np.random.default_rng(SEED3)
    idx = rng.choice(len(uniq), size=n, replace=False)
    docs = [uniq[i] for i in sorted(idx)]
    meta = {'pool_after_filter': len(pool), 'pool_unique': len(uniq), 'per_repo': per_repo, 'n_docs': n,
            'docs': [{'month': d['month'], 'key_sha256': d['key'], 'text_sha256': sha(d['text']),
                      'ext': os.path.splitext(d['path'])[1].lower(), 'chars': len(d['text'])} for d in docs]}
    return docs, meta


def load_code_docs():
    """凍結後の修正の順番どおりに、月を広げ、それでも足りなければ 1 リポジトリ 2 ファイルまで、最後に 30 本以上で回す。"""
    from datasets import load_dataset
    rows, months, steps = [], [], []
    def add(m):
        for r in load_dataset('RealTimeData/code_alltime', m, split='train'):
            rows.append({'month': m, 'repo': r['repo_name'], 'path': r['file_path'], 'code': r['code']})
        months.append(m)
    for m in CODE_MONTHS:
        add(m)
    for m in [None] + CODE_MONTHS_MORE:
        if m is not None:
            add(m)
        n1 = len(unique_pool(rows, 1)[1])
        steps.append({'months': list(months), 'unique_per_repo1': n1})
        if n1 >= N_DOCS:
            docs, meta = select_code(rows, 1)
            break
    else:
        n2 = len(unique_pool(rows, 2)[1])
        steps.append({'months': list(months), 'unique_per_repo2': n2})
        docs, meta = select_code(rows, 2, allow_fewer=True)
    meta['months'] = list(months)
    meta['steps'] = steps
    meta['filter_counts'] = filter_counts(rows)
    return docs, meta


def memory_needed(hf_id, chunk):
    """重み（fp32）＋ KV（fp32、chunk 本 × 577 トークン）＋ 余裕 3GB の見積もり（バイト）。"""
    from transformers import AutoConfig
    from huggingface_hub import HfApi
    cfg = AutoConfig.from_pretrained(hf_id)
    n_params = None
    try:
        info = HfApi().model_info(hf_id, files_metadata=False)
        if getattr(info, 'safetensors', None) is not None:
            n_params = int(info.safetensors.total)
    except Exception:
        pass
    if n_params is None:
        h, L, V = cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size
        n_params = 12 * L * h * h + 2 * V * h
    heads = cfg.num_attention_heads
    kvh = getattr(cfg, 'num_key_value_heads', None) or heads
    hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // heads
    kv_tok = 2 * cfg.num_hidden_layers * kvh * hd * 4
    V = cfg.vocab_size
    tf_peak = 6 * CONT * V * 8 + (PROMPT + CONT + 1) * V * 4          # 人の文脈の計算の float64 の一時テンソル
    gen_peak = kv_tok * chunk * (PROMPT + CONT + 1) + chunk * PROMPT * V * 4
    return int(n_params * 4 + max(tf_peak, gen_peak) + 3e9), int(n_params), int(kv_tok)


@torch.no_grad()
def teacher_forced3(model, prompt, human, valid, device, H2):
    """A2-2 の teacher_forced に、記事ごとの NLL（T = 1、外した番号を除いた正規化）を足したもの。"""
    res, a, h, tc = teacher_forced(model, prompt, human, valid, device, H2)
    x = torch.tensor([prompt + human], device=device)
    z = model(x).logits[0, len(prompt) - 1: len(prompt) - 1 + CONT].double().masked_fill(~valid, float('-inf'))
    y = torch.tensor(human, device=device)
    nll = float((torch.logsumexp(z, -1) - z.gather(-1, y[:, None])[:, 0]).mean())
    return res, a, h, tc, nll


def run_condition(name, hf_id, docs, outdir, device, chunk, notify=print):
    """A2-2 の run_model と同じものを、かたまりに分けた生成・メモリの見積もり・NLL つきで。"""
    path = f'{outdir}/{name}.npz'
    if os.path.exists(path):
        print(name, 'はもう終わっています（飛ばします）')
        return
    need, n_params, kv_tok = memory_needed(hf_id, chunk)
    have = torch.cuda.get_device_properties(0).total_memory
    if need > have:
        raise RuntimeError(f'{name}：GPU のメモリが足りません（見積もり {need / 1e9:.1f} GB、GPU {have / 1e9:.1f} GB）')
    t0 = time.time()
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, eos = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    pairs = [split_doc(tok, d['text']) for d in docs]
    use = [i for i, p in enumerate(pairs) if p is not None]
    nT, nD = len(TEMPS), len(use)
    tf = np.zeros((4, nD, nT, CONT), np.float32)
    a_all = np.zeros((nD, CONT), np.float32); h_all = np.zeros((nD, CONT), np.float32)
    tcal = np.zeros((nD, len(TCAL_T))); nll = np.zeros(nD)
    for j, i in enumerate(use):
        res, a, h, tc, nl = teacher_forced3(model, *pairs[i], valid, device, H2)
        for k in range(nT):
            for q in range(4):
                tf[q, j, k] = res[k][q]
        a_all[j], h_all[j], tcal[j], nll[j] = a, h, tc, nl
    plen = {len(pairs[i][0]) for i in use}
    assert len(plen) == 1, plen
    prompts = [pairs[i][0] for i in use]
    gen = np.zeros((4, nT, SAMPLES, nD, CONT), np.float32)
    toks = np.zeros((nT, SAMPLES, nD, CONT), np.int32)
    for k, T in enumerate(TEMPS):
        for s in range(SAMPLES):
            for c, lo in enumerate(range(0, nD, chunk)):
                hi = min(nD, lo + chunk)
                g = torch.Generator(device=device).manual_seed(SEED + 1000 * k + s + 100000 * c)
                l, t1, e, l2, tk = generate(model, prompts[lo:hi], T, valid, g, H2)
                gen[0, k, s, lo:hi], gen[1, k, s, lo:hi], gen[2, k, s, lo:hi], gen[3, k, s, lo:hi] = l, t1, e, l2
                toks[k, s, lo:hi] = tk
        print(f'  {name} T={T} おわり（{time.time() - t0:.0f} 秒）')
    ext = np.array([os.path.splitext(docs[i].get('path', ''))[1].lower() for i in use])
    np.savez_compressed(path + '.part.npz', tf=tf, a=a_all, h=h_all, tcal=tcal, tcal_T=TCAL_T, gen=gen, toks=toks, nll=nll,
                        ext=ext, temps=np.array(TEMPS), use=np.array(use), eos=np.array(eos), tok_len=len(tok),
                        prompt_len=list(plen)[0], revision=str(getattr(model.config, '_commit_hash', '')),
                        seconds=time.time() - t0, H2=H2, chunk=chunk, mem_need=need, n_params=n_params, kv_tok=kv_tok)
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()
    notify(f'{name} おわり（{time.time() - t0:.0f} 秒）')
