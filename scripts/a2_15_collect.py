"""A2-15（ゲート docs/A2_15_GATE.md）のデータ集め：Pythia・GPT-Neo の「塩で A が溶けにくい癖」は、Pile の中身のせいか、LayerNorm のせいか。GPU（L4）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_13_collect.py を実行したあとの同じ名前空間で実行する。
手順・温度・腕・種は A2-13 のまま（run_13・C13_TEMPS・C13_ARMS・C13_SEED を変えずに使う）。モデルの組だけを C15_GROUPS で足す。
どれも LayerNorm で、Pile 以外のデータで学習したモデル：GPT-2（WebText）、RedPajama-INCITE（Pythia と同じ GPT-NeoX の作り、RedPajama-Data-1T）、BLOOM（多言語の ROOTS）。
"""
C15_GROUPS = {
    'gpt2': [('gpt2-124m', 'openai-community/gpt2', None), ('gpt2-355m', 'openai-community/gpt2-medium', None),
             ('gpt2-774m', 'openai-community/gpt2-large', None)],
    'xl_bloom': [('gpt2-1.5b', 'openai-community/gpt2-xl', None), ('bloom-560m', 'bigscience/bloom-560m', None),
                 ('bloom-1b1', 'bigscience/bloom-1b1', None)],
    'rp': [('redpajama-3b', 'togethercomputer/RedPajama-INCITE-Base-3B-v1', 20)],
}


def self_test_15():
    names = [n for g in C15_GROUPS.values() for n, _, _ in g]
    assert len(names) == len(set(names)) == 7
    assert C13_SEED == 20261123 and len(C13_TEMPS) == 7 and [a[0] for a in C13_ARMS] == ['none', 'none_b', 'pl300', 'pl300_b']
    return 'OK'
