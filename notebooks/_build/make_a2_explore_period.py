"""探索：結晶の格子定数（周期）のノートブック（CPU）を組み立てる。make_a2_explore_onset.py から作った。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON = rd('scripts/a2_explore_onset.py')
PD = rd('scripts/a2_explore_period.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 格子定数（結晶の周期・助走の位置補正、CPU）'
HEAD = f"""# logit-thermo 探索：結晶の格子定数（くり返しの周期）（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜10 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 で保存した生成のトークン（25 本）で、結晶が始まった本ごとに「何トークンごとに同じ並びがくり返すか」（周期）と
  そろい具合を測り、結晶にならなかった本の位置ごとの写しの割合（助走の「位置のせい」の分）も数えて、
  `A2_explore/period_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_explore_onset.py`（`{h(ON)[:16]}…`）と `scripts/a2_explore_period.py`（`{h(PD)[:16]}…`）を埋め込み。生成もモデルの読み込みもしません。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = (f"# セル2：計算のコード\nimport hashlib, json, math\nSRCON = {ON!r}\n"
        f"assert hashlib.sha256(SRCON.encode()).hexdigest() == {h(ON)!r}\nexec(SRCON)\n"
        f"SRCPD = {PD!r}\nassert hashlib.sha256(SRCPD.encode()).hexdigest() == {h(PD)!r}\nexec(SRCPD)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset(), self_test_period())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
try:
    r = run_period({BASE!r})
    r['a2_explore_onset_sha256'] = {h(ON)!r}
    r['a2_explore_period_sha256'] = {h(PD)!r}
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/period_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 格子定数')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore period', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '結晶の周期を A2_explore/period_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_period_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(ON), h(PD))
