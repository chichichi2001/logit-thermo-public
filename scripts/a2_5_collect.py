"""A2-5（ゲート docs/A2_5_GATE.md v0.2）のデータ集め。GPU（L4）のノートブックで、scripts/a2_2_collect.py を実行したのと
同じ名前空間で実行する。A2-2 の run_model と generate は変えずに使い、名前で呼ばれる部品だけを差し替える：
- load_model を、revision（学習途中のチェックポイント）を受け取る版に差し替える。
- 生成のあいだだけ SEED を A5_SEED にする（記事は先に A2-2 の種で選んでおく。load_docs も SEED を使うため）。
- 2.8B だけ、generate を「20 本ずつ 2 回に分けて呼び、つなぐ」版に差し替える（メモリのため）。
"""
import hashlib, json, time
import numpy as np
import torch

A5_SEED = 20261102
A5_RUNS = [  # (名前, Hugging Face の ID, revision, 生成を分ける本数（None なら分けない）)
    ('pythia-70m', 'EleutherAI/pythia-70m', 'main', None),
    ('pythia-160m', 'EleutherAI/pythia-160m', 'main', None),
    ('pythia-410m', 'EleutherAI/pythia-410m', 'main', None),
    ('pythia-1b', 'EleutherAI/pythia-1b', 'main', None),
    ('pythia-1.4b', 'EleutherAI/pythia-1.4b', 'main', None),
    ('pythia-1b@step8000', 'EleutherAI/pythia-1b', 'step8000', None),
    ('pythia-1b@step24000', 'EleutherAI/pythia-1b', 'step24000', None),
    ('pythia-1b@step64000', 'EleutherAI/pythia-1b', 'step64000', None),
    ('pythia-2.8b', 'EleutherAI/pythia-2.8b', 'main', 20),
]
_A22_SEED = None
_ORIG_GENERATE = None


def split_spec(spec):
    return tuple(spec.split('@', 1)) if '@' in spec else (spec, None)


def load_model_rev(spec, device):
    """spec：'Hugging Face の ID' または 'ID@revision'。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id, rev = split_spec(spec)
    tok = AutoTokenizer.from_pretrained(hf_id, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(hf_id, revision=rev, torch_dtype=torch.float32).to(device).eval()
    return tok, model


def make_chunked(orig, size):
    def generate_chunked(model, prompts, T, valid, gen, H2):
        outs = [orig(model, prompts[a:a + size], T, valid, gen, H2) for a in range(0, len(prompts), size)]
        return tuple(np.concatenate([o[q] for o in outs], 0) for q in range(len(outs[0])))
    return generate_chunked


def pairs_hash(tok, docs):
    pairs = [split_doc(tok, d['text']) for d in docs]
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps(None if p is None else [list(map(int, p[0])), list(map(int, p[1]))]).encode())
    return h.hexdigest()


def run_a25(docs, outdir, device, notify=print):
    """docs は先に load_docs()（A2-2 の種）で読んでおくこと。"""
    global SEED, load_model, generate, _A22_SEED, _ORIG_GENERATE
    from transformers import AutoTokenizer
    _A22_SEED, _ORIG_GENERATE = SEED, generate
    # 確かめ 4：トークナイザーの区切り方が全部の回で同じ
    hs = {name: pairs_hash(AutoTokenizer.from_pretrained(hf_id, revision=rev), docs) for name, hf_id, rev, _ in A5_RUNS}
    assert len(set(hs.values())) == 1, ('トークナイザーの区切り方が違う回があります', hs)
    meta = {'gen_seed': A5_SEED, 'doc_seed': _A22_SEED, 'pairs_sha256': list(hs.values())[0], 'runs': {}}
    load_model = load_model_rev
    try:
        for name, hf_id, rev, chunk in A5_RUNS:
            t0 = time.time()
            SEED = A5_SEED
            generate = make_chunked(_ORIG_GENERATE, chunk) if chunk else _ORIG_GENERATE
            assert run_model.__globals__['SEED'] == A5_SEED and run_model.__globals__['load_model'] is load_model_rev
            try:
                run_model(name, f'{hf_id}@{rev}', docs, outdir, device, notify=notify)
            finally:
                SEED, generate = _A22_SEED, _ORIG_GENERATE
            Z = np.load(f'{outdir}/{name}.npz')
            meta['runs'][name] = {'hf_id': hf_id, 'revision_requested': rev, 'commit': str(Z['revision']), 'chunk': chunk,
                                  'use': [int(u) for u in Z['use']], 'seconds': round(time.time() - t0, 1)}
            json.dump(meta, open(f'{outdir}/a2_5_runs.json', 'w'), ensure_ascii=False, indent=1)
    finally:
        SEED, generate = _A22_SEED, _ORIG_GENERATE
    uses = {tuple(r['use']) for r in meta['runs'].values()}
    assert len(uses) == 1, '使った記事が回によって違います'
    return meta


def self_test_a25(device):
    assert split_spec('EleutherAI/pythia-1b@step8000') == ('EleutherAI/pythia-1b', 'step8000')
    assert split_spec('EleutherAI/pythia-1b') == ('EleutherAI/pythia-1b', None)
    tok, model = load_model_rev('hf-internal-testing/tiny-random-GPT2LMHeadModel@main', device)
    valid, _ = valid_mask(tok, model, device)
    # 分けて生成しても、形と値の範囲が同じ（同じ生成器を順に使う）
    prompts = [list(range(5 + i, 5 + i + 8)) for i in range(5)]
    global CONT
    keep = CONT; CONT = 6
    try:
        g1 = torch.Generator(device=device).manual_seed(0)
        full = generate(model, prompts, 1.0, valid, g1, 33)
        g2 = torch.Generator(device=device).manual_seed(0)
        part = make_chunked(generate, 2)(model, prompts, 1.0, valid, g2, 33)
    finally:
        CONT = keep
    assert all(a.shape == b.shape for a, b in zip(full, part)), [(a.shape, b.shape) for a, b in zip(full, part)]
    assert np.allclose(full[0][:2, 0], part[0][:2, 0], atol=1e-4)      # 1 歩目は乱数に関係なく同じ
    assert len({n for n, _, _, _ in A5_RUNS}) == len(A5_RUNS)
    assert 'SEED' in run_model.__globals__ and 'generate' in run_model.__globals__
    return 'OK'
