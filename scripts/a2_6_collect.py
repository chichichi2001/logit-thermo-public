"""A2-6（ゲート docs/A2_6_GATE.md）のデータ集め：「乳の種類」テスト。GPU（L4）のノートブックで、
scripts/a2_2_collect.py と scripts/a2_5_collect.py を実行したのと同じ名前空間で実行する。
A2-2 の run_model・generate と、A2-5 の make_chunked・pairs_hash をそのまま使う。
- 生成のあいだだけ SEED を A2-5 と同じ 20261102 にする（Pythia の 6 回は A2-5 のデータをそのまま使うため）。
- 家族（グループ）ごとに 1 つのノートブックで回す。家族の中でトークナイザーの区切り方が同じかを確かめる。
"""
import json, time
import numpy as np

A6_SEED = 20261102
A6_GROUPS = {  # (名前, Hugging Face の ID, 生成を分ける本数（None なら分けない）)
    'qwen25': [('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B', None), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B', None),
               ('qwen2.5-3b', 'Qwen/Qwen2.5-3B', 20)],
    'smol2': [('smollm2-135m', 'HuggingFaceTB/SmolLM2-135M', None), ('smollm2-360m', 'HuggingFaceTB/SmolLM2-360M', None),
              ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B', None)],
    'qwen3': [('qwen3-0.6b', 'Qwen/Qwen3-0.6B-Base', None), ('qwen3-1.7b', 'Qwen/Qwen3-1.7B-Base', None),
              ('qwen3-4b', 'Qwen/Qwen3-4B-Base', 20)],
}


def hub_sha(hf_id):
    """いまの main の版（コミット）を記録する。transformers 5 では config の _commit_hash が空になるため。"""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(hf_id).sha
    except Exception as e:
        return f'取れませんでした：{type(e).__name__}'


def run_a26(group, docs, outdir, device, notify=print):
    """docs は先に load_docs()（A2-2 の種 20261027）で読んでおくこと。"""
    global SEED, generate
    from transformers import AutoTokenizer
    keep_seed, keep_gen = SEED, generate
    runs = A6_GROUPS[group]
    hs = {name: pairs_hash(AutoTokenizer.from_pretrained(hf_id), docs) for name, hf_id, _ in runs}
    assert len(set(hs.values())) == 1, ('家族の中でトークナイザーの区切り方が違います', hs)
    meta = {'group': group, 'gen_seed': A6_SEED, 'doc_seed': keep_seed, 'pairs_sha256': list(hs.values())[0], 'runs': {}}
    try:
        for name, hf_id, chunk in runs:
            t0 = time.time()
            SEED = A6_SEED
            generate = make_chunked(keep_gen, chunk) if chunk else keep_gen
            assert run_model.__globals__['SEED'] == A6_SEED and run_model.__globals__['generate'] is generate
            try:
                run_model(name, hf_id, docs, outdir, device, notify=notify)
            finally:
                SEED, generate = keep_seed, keep_gen
            Z = np.load(f'{outdir}/{name}.npz')
            meta['runs'][name] = {'hf_id': hf_id, 'chunk': chunk, 'use': [int(u) for u in Z['use']],
                                  'hub_sha': hub_sha(hf_id), 'seconds': round(time.time() - t0, 1)}
            json.dump(meta, open(f'{outdir}/a2_6_runs_{group}.json', 'w'), ensure_ascii=False, indent=1)
    finally:
        SEED, generate = keep_seed, keep_gen
    return meta


def self_test_a26():
    names = [n for g in A6_GROUPS.values() for n, _, _ in g]
    assert len(set(names)) == len(names)
    assert not any(n.startswith('pythia') for n in names)
    assert 'SEED' in run_model.__globals__ and 'generate' in run_model.__globals__
    return 'OK'
