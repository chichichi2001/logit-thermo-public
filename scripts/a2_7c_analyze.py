"""A2-7c（ゲート docs/A2_7C_GATE.md）の分析。scripts/a2_7b_analyze.py を実行し、その自己テストを通したあとの同じ名前空間で実行して、
温度の並びとモデルだけを置きかえる（判定の決め方は A2-7b のまま。データを開く前に固定する）。
"""
B7A_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
B7A_MODELS = ['qwen2.5-1.5b']


def analyze7c(d):
    r = analyze7b(d)
    m = r['models']['qwen2.5-1.5b']
    r['CQ1'] = m['BQ1']['word']
    r['CQ2'] = m['BQ2_word']
    return r
