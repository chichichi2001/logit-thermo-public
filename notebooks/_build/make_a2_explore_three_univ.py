"""探索：3 つの谷が 25 本のモデルでも同じか（CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON = rd('scripts/a2_explore_onset.py')
HD, TH, TU = rd('scripts/a2_explore_hidden.py'), rd('scripts/a2_explore_three.py'), rd('scripts/a2_explore_three_univ.py')
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
LABEL = 'logit-thermo 探索 3 つの谷は 25 本のモデルでも同じか（CPU）'
HEAD = f"""# logit-thermo 探索：3 つの谷（液体 ⇄ 長い周期の結晶 ⇄ 短い周期の結晶）は 25 本のモデルでも同じか（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 10〜20 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 の 25 本の生成で、窓ごとに L・A・C のラベルを付け、割合・移り変わり・段ごとの ΔH を出して
  `A2_explore/three_univ_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：結晶の始まりの探索のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_hidden.py`・`a2_explore_three.py`・`a2_explore_three_univ.py`（`{h(TU)[:16]}…`）は埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport math\n" + EXTRACT +
        f"_s = take_src({{'SRCON': {h(ON)!r}}})\nSRCON = _s['SRCON']\n"
        f"SRCHD = {HD!r}\nSRCTH = {TH!r}\nSRCTU = {TU!r}\n"
        f"assert hashlib.sha256(SRCHD.encode()).hexdigest() == {h(HD)!r} and hashlib.sha256(SRCTH.encode()).hexdigest() == {h(TH)!r} and hashlib.sha256(SRCTU.encode()).hexdigest() == {h(TU)!r}\n"
        "exec(SRCON)\nexec(SRCHD)\nexec(SRCTH)\nexec(SRCTU)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset(), self_test_tu())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
try:
    r = three_univ({BASE!r})
    r = dict(runs=r, onset_sha256={h(ON)!r}, hidden_sha256={h(HD)!r}, three_sha256={h(TH)!r}, three_univ_sha256={h(TU)!r})
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/three_univ_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 3 つの谷は 25 本のモデルでも同じか')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore three states universality', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '3 つの谷の 25 本の要約を A2_explore/three_univ_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_three_univ_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size)
