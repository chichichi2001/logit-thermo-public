"""A2-10（ゲート docs/A2_10_GATE.md）のデータ集め：凝固点降下（偽薬を濃くして、本当の融点 T_m と T_0 の両方の下がり方を測る）。GPU（L4）。
scripts/a2_7b_collect.py を実行したあとの同じ名前空間で実行し、温度の並び・腕・種だけを置きかえる（generate_7b・run_7b・teacher_7b は
実行するときに B7_TEMPS・B7_ARMS・SEED7B を読むので、禁止のしかたは A2-7b のまま）。ノートブックごとに C10_GROUP を決める。
"""
B7_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
B7_ARMS = [('none', 'none', 0.0), ('none_b', 'none', 0.0),
           ('pl075', 'placebo', 0.075), ('pl150', 'placebo', 0.15), ('pl300', 'placebo', 0.30)]
SEED7B = 20261116


def self_test_10():
    assert len(B7_TEMPS) == 14 and all(T in B7_TEMPS for T in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05])
    assert [a[0] for a in B7_ARMS] == ['none', 'none_b', 'pl075', 'pl150', 'pl300']
    seeds = {gen_seed(k, s, a) for k in range(14) for s in range(2) for a in ('none', 'none_b')}
    assert len(seeds) == 56 and SEED7B == 20261116
    return 'OK'
