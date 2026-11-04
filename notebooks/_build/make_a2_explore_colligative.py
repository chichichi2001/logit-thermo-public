"""探索：凝固点降下の下見のノートブック（CPU）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A4, CL = rd('scripts/a2_4_analyze.py'), rd('scripts/a2_explore_colligative.py')
src5 = rd('notebooks/_build/make_a2_5.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
notify_cell, DONE, FAIL, EXTRACT = ns['notify_cell'], ns['DONE'], ns['FAIL'], ns['EXTRACT']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 凝固点降下（邪魔の量と融点、CPU）'
HEAD = f"""# logit-thermo 探索：凝固点降下の下見（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜15 分。
- 探索（予想なし）。A2-4（なし・写しの禁止・偽薬）と A2-7（n ＝ 8・6・3・2）のデータから、「邪魔した確率の量」と「融点の下がり方」を並べて `A2_explore/colligative_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-4 の分析のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_colligative.py`（`{h(CL)[:16]}…`）は埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRCA24': {h(A4)!r}}})\nSRCA22, SRCA24 = _s['SRCA22'], _s['SRCA24']\n"
        f"SRCCL = {CL!r}\nassert hashlib.sha256(SRCCL.encode()).hexdigest() == {h(CL)!r}\n"
        "exec(SRCA22)\nexec(SRCA24)\nexec(SRCCL)\nprint('自己テスト：', self_test_colligative())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
import json
try:
    r = run_colligative({BASE!r})
    r['a2_explore_colligative_sha256'] = {h(CL)!r}
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/colligative_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 凝固点降下')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore colligative', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '邪魔の量と融点を A2_explore/colligative_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_colligative_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(CL))
