"""A2-14（ゲート docs/A2_14_GATE.md）のデータ集め：Pythia の「塩で A が溶けにくい癖」は学習データ（Pile）のせいか。GPU（L4）。
scripts/a2_2_collect.py・a2_4_collect.py・a2_7_collect.py・a2_7b_collect.py・a2_13_collect.py を実行したあとの同じ名前空間で実行する。
手順・温度・腕・種は A2-13 のまま（run_13・C13_TEMPS・C13_ARMS・C13_SEED を変えずに使う）。モデルの組だけを C14_GROUPS で足す。
"""
C14_GROUPS = {
    'dedup': [('pythia-160m-deduped', 'EleutherAI/pythia-160m-deduped', None), ('pythia-410m-deduped', 'EleutherAI/pythia-410m-deduped', None),
              ('pythia-1b-deduped', 'EleutherAI/pythia-1b-deduped', None)],
    'neo': [('gpt-neo-125m', 'EleutherAI/gpt-neo-125m', None), ('gpt-neo-1.3b', 'EleutherAI/gpt-neo-1.3B', None),
            ('gpt-neo-2.7b', 'EleutherAI/gpt-neo-2.7B', 20)],
    'ref': [('olmo2-1b', 'allenai/OLMo-2-0425-1B', None), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B', None)],
}


def self_test_14():
    names = [n for g in C14_GROUPS.values() for n, _, _ in g]
    assert len(names) == len(set(names)) == 8
    assert C13_SEED == 20261123 and len(C13_TEMPS) == 7 and [a[0] for a in C13_ARMS] == ['none', 'none_b', 'pl300', 'pl300_b']
    return 'OK'
