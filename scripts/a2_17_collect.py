"""A2-17 の材料集め v0.2：文章の種類を変えても同じ形が出るか（A2-16 と同じ作りを、BBC 以外の 3 種類の英語の文章で）（2026-11-01 07:xx JST、GPU（L4）を 3 つ）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_explore_field.py・a2_explore_dsc.py・a2_explore_dsc2.py・a2_gpre_collect.py を
実行したあとの同じ名前空間で実行する（load_model・valid_mask・split_doc・clean・BOILER・CONT・generate_dsc2・GP_TEMPS・GP_H・GP_S・GP_CHUNK・gp_seed・GP_STALE を使う）。
コードの文章は A2-3 の選び方（scripts/a2_3_collect.py の load_code_docs）をそのまま使う：ノートブックで別の名前空間に読み込み、A3_LOAD_CODE として渡す。
データは、ゲート（A2-17）を凍結するまで開けない。

文章（種で決めて選ぶ。BBC は A2-16 の材料をそのまま使う）：
  'arxiv'   ：RealTimeData/arxiv_alltime の 2025-04・2025-05（論文の本文。行の折り返しはなく、1 行が 1 段落）。clean（60 字以上の段落）をかけ、6,000 字以上。40 本。
  'code'    ：A2-3 と同じ 35 本（RealTimeData/code_alltime の 2024-12〜2025-06、ふつうの言語の拡張子、ライセンス文を含むものを除く、1 リポジトリ 2 ファイルまで）。
              A2-3 の code_meta.json と文章のハッシュが一致することを確かめる。
  'fiction' ：sedthh/gutenberg_english の最後のファイル（train-00036）から。足りなければ 1 つ前のファイルも。題目に fiction を含み、本の番号 > 30000（有名な古典を避ける）、
              言語が en、本文 40,000 字以上。本ごとに、行の 3 割より多くが 55〜80 字なら「折り返しの本」とみなして空行で段落に分け、行をつなぎ直す。
              そうでなければ 1 行を 1 段落にする。段落の平均が 3,000 字を超える本（段落に分けられない）は除く。
              本文の 30% の位置の段落から、段落を短いものも含めて続けてつなぎ（[Illustration…] などの印の段落だけ除く）、6,000 字以上。40 本。
モデル（4 つ、A2-16 と同じ材料がある）：Pythia-1.4B・SmolLM2-1.7B・Qwen2.5-1.5B・BLOOM-1.1B。
作り：A2-16（G-pre）と同じ（温度 0.4〜0.7 の 7 個、h ＝ −0.3・0・＋0.3、サンプル 4、512 歩、q つき、3 つの h で同じ種）。
覚えているかの記録（仕事 'nll'、4 種類とも同じ形）：人の書いた続き（512 トークン）を温度 1 で読ませたときの −log p の平均（文章ごと）。
仕事：3 種類 × 4 モデル × 3 つの h ＝ 36 個と、4 種類（BBC を含む）× 4 モデルの読ませ 16 個。ランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行く。
文章の読み込みは、仕事を取りに行く前に 1 回だけ（失敗したら、そのランタイムは止まる。仕事に失敗の印はつかない）。
"""
import gc, os, time, json, hashlib, re, socket
import numpy as np
import torch

A17_GENRES = ['arxiv', 'code', 'fiction']
A17_MODELS = [('pythia-1.4b', 'EleutherAI/pythia-1.4b'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'),
              ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('bloom-1b1', 'bigscience/bloom-1b1')]
A17_N = 40
A17_MINCH = 6000
A17_PG_SHARDS = ['train-00036-of-00037', 'train-00035-of-00037', 'train-00034-of-00037']
A17_PG_MINID = 30000
A17_PG_MINCH = 40000
A17_PG_POOL = 2 * A17_N          # 小説の候補がこれ未満なら、1 つ前のファイルも読む
A17_SEEDS = {'arxiv': 20261151, 'fiction': 20261153}
A17_MARK = re.compile(r'^\[(Illustration|Footnote|Transcriber)', re.I)


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def pg_paras(text):
    """グーテンベルクの本文 → (段落のリスト, 折り返しの本か)。段落に分けられない本は (None, …)。"""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    ne = [l.strip() for l in lines if l.strip()]
    if not ne:
        return None, False
    wrapped = np.mean([55 <= len(l) <= 80 for l in ne]) > 0.3
    if wrapped:
        paras = [' '.join(l.strip() for l in b.split('\n') if l.strip()) for b in re.split(r'\n\s*\n', text)]
        paras = [p for p in paras if p]
    else:
        paras = ne
    if sum(len(p) for p in paras) / len(paras) > 3000:
        return None, wrapped
    return paras, wrapped


def pg_passage(paras):
    """30% の位置の段落から続けてつなぐ（印の段落だけ除く）。"""
    cum = np.cumsum([len(p) + 1 for p in paras])
    start = int(np.searchsorted(cum, 0.3 * cum[-1])) + 1
    acc = []
    for p in paras[start:]:
        if A17_MARK.search(p):
            continue
        acc.append(p)
        if sum(len(a) for a in acc) >= A17_MINCH:
            break
    body = '\n\n'.join(acc)
    return body if len(body) >= A17_MINCH else None


def _pick(pool, genre):
    pool.sort(key=lambda d: _sha(d['id']))
    rng = np.random.default_rng(A17_SEEDS[genre])
    idx = rng.choice(len(pool), size=A17_N, replace=False)
    return [pool[i] for i in sorted(idx)]


def _retry(f, tries=3, wait=60):
    for i in range(tries):
        try:
            return f()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(wait)


def load_arxiv():
    from datasets import load_dataset
    pool, n_rows, n_lines, n_kept = [], 0, 0, 0
    for m in ('2025-04', '2025-05'):
        for row in _retry(lambda: load_dataset('RealTimeData/arxiv_alltime', m, split='train')):
            n_rows += 1
            raw = row['text'] or ''
            ls = [l for l in raw.split('\n') if l.strip()]
            c = clean(raw)
            n_lines += len(ls); n_kept += c.count('\n\n') + 1 if c else 0
            if len(c) >= A17_MINCH:
                pool.append({'id': str(row['entry_id']), 'text': c})
    assert len(pool) >= A17_N, ('論文の候補が少なすぎます', len(pool))
    docs = _pick(pool, 'arxiv')
    return docs, dict(rows=n_rows, pool_size=len(pool), kept_line_frac=n_kept / max(n_lines, 1))


def load_fiction():
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    files = _retry(lambda: list_repo_files('sedthh/gutenberg_english', repo_type='dataset'))
    pool, used, stat = [], [], dict(books=0, fiction_ok=0, wrapped=0, unsplittable=0, short_passage=0)
    for pre in A17_PG_SHARDS:
        fn = [f for f in files if f.endswith('.parquet') and os.path.basename(f).startswith(pre)]
        assert len(fn) == 1, ('グーテンベルクのファイルが見つかりません', pre, len(fn))
        path = _retry(lambda: hf_hub_download('sedthh/gutenberg_english', fn[0], repo_type='dataset'))
        used.append(dict(file=fn[0], sha256=hashlib.sha256(open(path, 'rb').read()).hexdigest()))
        df = pd.read_parquet(path)
        for text, meta in zip(df['TEXT'], df['METADATA']):
            stat['books'] += 1
            try:
                md = json.loads(meta) if isinstance(meta, str) else dict(meta)
                tid = int(md.get('text_id') or 0)
            except Exception:
                continue
            subj = md.get('subjects')
            subj = ' '.join(map(str, subj)) if isinstance(subj, (list, tuple)) else str(subj or '')
            lang = md.get('language')
            lang = ' '.join(map(str, lang)) if isinstance(lang, (list, tuple)) else str(lang or '')
            if 'fiction' not in subj.lower() or tid <= A17_PG_MINID or lang.strip("[]' ").lower() != 'en' or not isinstance(text, str) or len(text) < A17_PG_MINCH:
                continue
            stat['fiction_ok'] += 1
            paras, wrapped = pg_paras(text)
            stat['wrapped'] += int(wrapped)
            if paras is None:
                stat['unsplittable'] += 1
                continue
            body = pg_passage(paras)
            if body is None:
                stat['short_passage'] += 1
                continue
            pool.append({'id': str(tid), 'text': body})
        if len(pool) >= A17_PG_POOL:
            break
    assert len(pool) >= A17_N, ('小説の候補が少なすぎます', len(pool), stat)
    docs = _pick(pool, 'fiction')
    return docs, dict(pool_size=len(pool), files=used, **stat)


def load_code(code_meta_path):
    docs, meta = A3_LOAD_CODE()
    old = json.load(open(code_meta_path))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'コードの文章が A2-3 と違います'
    return docs, dict(pool_size=meta['pool_unique'], per_repo=meta['per_repo'], months=meta['months'], n_docs=meta['n_docs'])


def prepare_docs(outdir, bbc_docs, code_meta_path, notify=print):
    """4 種類の文章を 1 回だけ読み込み、ほかのランタイムと同じかを docs_meta.json で確かめる → {種類: 文章}。"""
    D = {'bbc': bbc_docs}
    for g, f in (('arxiv', load_arxiv), ('code', lambda: load_code(code_meta_path)), ('fiction', load_fiction)):
        docs, info = f()
        meta = dict(info, docs=[dict(text_sha256=_sha(d['text']), chars=len(d['text'])) for d in docs])
        os.makedirs(f'{outdir}/{g}', exist_ok=True)
        mp = f'{outdir}/{g}/docs_meta.json'
        if os.path.exists(mp):
            old = json.load(open(mp))
            assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], (g, '文章がほかのランタイムと違います')
        else:
            json.dump(meta, open(mp, 'w'), ensure_ascii=False, indent=1)
        D[g] = docs
        notify(f'{g}：{len(docs)} 本を読み込みました（候補 {info["pool_size"]} 本）')
    os.makedirs(f'{outdir}/bbc', exist_ok=True)
    return D


@torch.no_grad()
def human_nll(model, pairs, valid, device):
    """pairs：[(書き出し, 人の続き)] → 1 本ごとの、人の続きの −log p の平均（温度 1、使えないトークンを除いた正規化）と、
    使えないトークン（確率 0）で平均から外した歩の数。"""
    out, bad = [], []
    for prompt, cont in pairs:
        x = torch.tensor([prompt + cont], device=device)
        lg = model(x).logits[0, len(prompt) - 1: len(prompt) - 1 + len(cont)].double().masked_fill(~valid, float('-inf'))
        lp = torch.log_softmax(lg, -1).gather(-1, torch.tensor(cont, device=device)[:, None])[:, 0]
        ok = torch.isfinite(lp)
        out.append(float(-lp[ok].mean())); bad.append(int((~ok).sum()))
    return np.array(out), np.array(bad)


def a17_jobs():
    jobs = []
    for g in A17_GENRES:
        for name, hf in A17_MODELS:
            for arm, h in GP_H:
                jobs.append((g, name, hf, arm, h))
    for g in ['bbc'] + A17_GENRES:
        for name, hf in A17_MODELS:
            jobs.append((g, name, hf, 'nll', None))
    return jobs


def a17_tag(g, name, arm):
    return f'{g}/{name}@{arm}'


def a17_claim(outdir, g, name, arm, worker):
    """gp_claim と同じ決まり（終わっていない・失敗の印がない・誰も取っていないか 2 時間より古い → 取る）。ファイルの名前だけ文章の種類つき。"""
    tag = f'{g}__{name}@{arm}'
    if os.path.exists(f'{outdir}/{a17_tag(g, name, arm)}.npz') or os.path.exists(f'{outdir}/locks/{tag}.failed'):
        return False
    lk = f'{outdir}/locks/{tag}.lock'
    if os.path.exists(lk):
        try:
            age = time.time() - json.load(open(lk))['t']
        except Exception:
            age = 0.0
        if age < GP_STALE:
            return False
    os.makedirs(f'{outdir}/locks', exist_ok=True)
    json.dump(dict(worker=worker, host=socket.gethostname(), t=time.time()), open(lk, 'w'))
    time.sleep(20)
    try:
        return json.load(open(lk)).get('worker') == worker
    except Exception:
        return False


def run_a17_job(g, name, hf_id, arm, h, D, outdir, device, cache):
    if cache.get('name') != name:
        cache.clear(); gc.collect(); torch.cuda.empty_cache()
        tok, model = load_model(hf_id, device)
        assert next(model.parameters()).dtype == torch.float32
        valid, eos = valid_mask(tok, model, device)
        cache.update(name=name, tok=tok, model=model, valid=valid)
    tok, model, valid = cache['tok'], cache['model'], cache['valid']
    pairs = [split_doc(tok, d['text']) for d in D[g]]
    use = [i for i, p in enumerate(pairs) if p is not None]
    assert len(use) >= 30, (g, name, '使える文章が少なすぎます', len(use))
    assert len({len(pairs[i][0]) for i in use}) == 1
    maxpos = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_positions', None)
    assert maxpos is None or len(pairs[use[0]][0]) + CONT <= maxpos, ('文が長すぎます', maxpos)
    ta = time.time()
    path = f'{outdir}/{a17_tag(g, name, arm)}.npz'
    if arm == 'nll':
        nll, bad = human_nll(model, [pairs[i] for i in use], valid, device)
        np.savez_compressed(path + '.part.npz', genre=g, use=np.array(use), human_nll=nll, n_invalid=bad, seconds=time.time() - ta)
        os.replace(path + '.part.npz', path)
        return time.time() - ta
    prompts = [pairs[i][0] for i in use]
    nD = len(use)
    chunks = [list(range(i, min(nD, i + GP_CHUNK))) for i in range(0, nD, GP_CHUNK)]
    dt = dict(toks=np.int32, cand=np.bool_, pick=np.bool_, U=np.float32, q0=np.float32, q=np.float32)
    R = {k: np.zeros((len(GP_TEMPS), GP_S, nD, CONT), v) for k, v in dt.items()}
    for k, T in enumerate(GP_TEMPS):
        Ts, Hs = np.full(CONT, T), np.full(CONT, h)
        for s in range(GP_S):
            for c, ii in enumerate(chunks):
                gg = torch.Generator(device=device).manual_seed(gp_seed(k, s, c))
                r = generate_dsc2(model, [prompts[i] for i in ii], Ts, Hs, valid, gg)
                for kk in R:
                    R[kk][k, s, ii] = r[kk]
                torch.cuda.empty_cache()
    np.savez_compressed(path + '.part.npz', temps=np.array(GP_TEMPS), h=h, arm=arm, genre=g, use=np.array(use), seconds=time.time() - ta, chunk=GP_CHUNK, **R)
    os.replace(path + '.part.npz', path)
    return time.time() - ta


def run_a17(worker, nworkers, D, outdir, device, notify=print):
    """D：prepare_docs の返り値。worker の番号ぶんずらした順に仕事を見て、取れるものを回す。"""
    os.makedirs(outdir, exist_ok=True)
    jobs = a17_jobs()
    sh = (len(jobs) * worker) // nworkers
    order = jobs[sh:] + jobs[:sh]
    cache, n, t0 = {}, 0, time.time()
    while True:
        took = False
        for g, name, hf, arm, h in order:
            if a17_claim(outdir, g, name, arm, worker):
                took = True
                tag = f'{g}__{name}@{arm}'
                try:
                    sec = run_a17_job(g, name, hf, arm, h, D, outdir, device, cache)
                    n += 1
                    notify(f'{g}/{name}@{arm} おわり（{sec:.0f} 秒、このランタイムで {n} 個目、通算 {time.time() - t0:.0f} 秒）')
                except Exception as e:
                    import traceback
                    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    v = os.environ.get('HF_TOKEN')
                    if v:
                        txt = txt.replace(v, '***')
                    txt = txt.encode('utf-8')[-3000:].decode('utf-8', 'ignore')
                    json.dump(dict(worker=worker, t=time.time(), error=txt), open(f'{outdir}/locks/{tag}.failed', 'w'), ensure_ascii=False)
                    notify(f'⚠️ {g}/{name}@{arm} で止まりました（この仕事は飛ばして、ほかを続けます）：\n' + txt)
                    cache.clear(); gc.collect(); torch.cuda.empty_cache()
                break
        if not took:
            break
    cache.clear(); gc.collect(); torch.cuda.empty_cache()
    left = []
    for g, nm, _, arm, _ in jobs:
        tag = a17_tag(g, nm, arm)
        if not os.path.exists(f'{outdir}/{tag}.npz'):
            left.append(tag + ('（失敗）' if os.path.exists(f"{outdir}/locks/{tag.replace('/', '__')}.failed") else ''))
    return n, left


@torch.no_grad()
def self_test_a17(device):
    """(1) 仕事の数と名前。(2) 小さなランダムのモデルで、human_nll が手で計算した −log p の平均と合う。(3) 文章の選び方が決まる。(4) 段落の分け方。"""
    jobs = a17_jobs()
    assert len(jobs) == 52 and len({a17_tag(g, n, a) for g, n, _, a, _ in jobs}) == 52
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    prompt, cont = [5, 6, 7, 8] * 4, [9, 10, 11, 12, 13, 14]
    a, bad = human_nll(model, [(prompt, cont)], valid, device)
    lg = model(torch.tensor([prompt + cont], device=device)).logits[0].double().masked_fill(~valid, float('-inf'))
    lps = [float(torch.log_softmax(lg[len(prompt) - 1 + i], -1)[cont[i]]) for i in range(len(cont))]
    assert abs(a[0] + np.mean(lps)) < 1e-6 and bad[0] == 0, (a, np.mean(lps))
    pool = [{'id': str(i), 'text': 'x' * 7000} for i in range(100)]
    d1 = _pick(list(pool), 'arxiv'); d2 = _pick(list(reversed(pool)), 'arxiv')
    assert [d['id'] for d in d1] == [d['id'] for d in d2] and len(d1) == 40
    _self_test_paras()
    return 'OK'


def _self_test_paras():
    line = lambda c: (c * 9 + ' ') * 7                 # 70 字の行
    wrapped = '\r\n'.join([line('a'), line('b'), 'cc end.', '', line('d'), 'short.', '', '"Yes," he said.', '', '']) * 200
    paras, wr = pg_paras(wrapped)
    assert wr and paras[0] == ' '.join([line('a').strip(), line('b').strip(), 'cc end.']) and paras[2] == '"Yes," he said.', paras[:3]
    flat = '\n'.join(['x' * 300, '"No."', 'y' * 250, '']) * 50
    paras, wr = pg_paras(flat)
    assert not wr and paras[:3] == ['x' * 300, '"No."', 'y' * 250]
    one_block = '\n'.join([line('q')] * 2000)          # 折り返しで、空行がない本 → 段落に分けられない
    assert pg_paras(one_block)[0] is None
    body = pg_passage(['p' * 100] * 30 + ['[Illustration: x]'] + ['"Hi."'] + ['r' * 500] * 30)
    assert body is not None and '[Illustration' not in body and len(body) >= A17_MINCH
