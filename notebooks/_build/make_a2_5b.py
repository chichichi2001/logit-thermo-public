"""A2-5b（T_0 での f の傾き）の分析ノートブック（CPU）を組み立てる。
A2-2 の分析と A2-5 の分析のコードは Drive の前のノートブックから取り出し、ハッシュで確かめる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A5, B5 = rd('scripts/a2_5_analyze.py'), rd('scripts/a2_5b_slope.py')
src5 = rd('notebooks/_build/make_a2_5.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
EXTRACT = ns['EXTRACT']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_5'
LABEL = 'logit-thermo A2-5b 分析（T_0 での f の傾き、CPU）'
HEAD = f"""# logit-thermo A2-5b：T_0 での f の傾き（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 10〜15 分。
- ゲート：`docs/A2_5B_GATE.md`。A2-5 の 9 回（`A2_5/*.npz`）から、s ＝（F(0.9) − F(0.8)）÷ 0.1 を出し、Q5〜Q7 を判定して `A2_5/a2_5b_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_analyze.py`・`scripts/a2_5_analyze.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_5b_slope.py`（`{h(B5)[:16]}…`）は埋め込み（データを開く前に固定）。記事の本文は使いません。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：分析のコード\n" + EXTRACT +
        f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRC5A': {h(A5)!r}}})\n"
        f"SRCA22, SRC5A = _s['SRCA22'], _s['SRC5A']\nSRC5B = {B5!r}\n"
        f"assert hashlib.sha256(SRC5B.encode()).hexdigest() == {h(B5)!r}\n"
        "exec(SRCA22)\nexec(SRC5A)\nexec(SRC5B)\nprint('自己テスト：', self_test_a25b())\n")
RUN = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze5b({OUT!r})
    r['a2_5b_slope_sha256'] = {h(B5)!r}
    json.dump(_clean(r), open({OUT!r} + '/a2_5b_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-5b の分析')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-5b slope', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '判定と要約を A2_5/a2_5b_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_5b_slope_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size); print('a2_5b_slope', h(B5)); print('a2_5_analyze', h(A5)); print('a2_2_analyze(if False)', h(A22))
