"""A2-7c（ゲート docs/A2_7C_GATE.md）：A2-7b の Qwen だけのやり直し。温度の格子を A2-7 と同じ 0.6〜1.05 に戻す。GPU（L4）。
scripts/a2_7b_collect.py を実行したあとの同じ名前空間で実行し、温度の並びだけを置きかえる（関数は実行するときに B7_TEMPS を読むので、
腕・種の決め方・禁止のしかたは A2-7b のまま）。生成の種の k は、この新しい 8 つの温度での番号。
"""
B7_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
C7_GROUP = 'qwen'


def self_test_7c():
    assert B7_TEMPS == [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
    assert all(T in B7_TEMPS for T in [0.8, 0.85, 0.9])
    assert [a[0] for a in B7_ARMS] == ['none', 'none_b', 'cq33', 'cq67', 'cq100', 'pl025', 'pl050', 'pl075']
    assert B7_GROUPS[C7_GROUP] == ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')
    return 'OK'
