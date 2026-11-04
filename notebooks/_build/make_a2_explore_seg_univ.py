"""探索 8.29：3 つの谷を新しいものさしで 25 本（CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON = rd('scripts/a2_explore_onset.py')
A8, SG, SU = rd('scripts/a2_8_vh.py'), rd('scripts/a2_explore_seg.py'), rd('scripts/a2_explore_seg_univ.py')
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
LABEL = 'logit-thermo 探索 3 つの谷を新しいものさしで 25 本（CPU）'
HEAD = f"""# logit-thermo 探索：3 つの谷を新しいものさし（かたまりで切り出す）で 25 本数え直す（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 10〜20 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 の 25 本の生成（温度 0.6・0.7）で、探索 8.25 の新しいものさしで L・A・C を数え、割合・入り方・出方を
  `A2_explore/seg_univ_summary.json` に書きます（8.24 の数え直し）。**結果は表示しません。**
- 計算のコード：結晶の始まりの探索と A2-8 のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_seg.py`・`a2_explore_seg_univ.py`（`{h(SU)[:16]}…`）は埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport math\n" + EXTRACT +
        f"_s = take_src({{'SRCON': {h(ON)!r}, 'SRC8': {h(A8)!r}}})\nSRCON, SRC8 = _s['SRCON'], _s['SRC8']\n"
        f"SRCSG = {SG!r}\nSRCSU = {SU!r}\n"
        f"assert hashlib.sha256(SRCSG.encode()).hexdigest() == {h(SG)!r} and hashlib.sha256(SRCSU.encode()).hexdigest() == {h(SU)!r}\n"
        "exec(SRCON)\nexec(SRC8)\nexec(SRCSG)\nexec(SRCSU)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset(), self_test_seg() is not None, self_test_su())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
try:
    r = seg_univ({BASE!r})
    r = dict(runs=r, onset_sha256={h(ON)!r}, a2_8_sha256={h(A8)!r}, seg_sha256={h(SG)!r}, seg_univ_sha256={h(SU)!r})
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/seg_univ_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 3 つの谷を新しいものさしで 25 本')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore seg univ', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '新しいものさしの 25 本の要約を A2_explore/seg_univ_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_seg_univ_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size)
