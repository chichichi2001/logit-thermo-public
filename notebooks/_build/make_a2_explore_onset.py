"""探索：結晶が始まる歩のノートブック（CPU）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON = rd('scripts/a2_explore_onset.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 結晶の始まり（始まる歩・滴定か核か、CPU）'
HEAD = f"""# logit-thermo 探索：結晶が始まる歩（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜10 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 で保存した生成のトークン（25 本）から、1 本ずつ「結晶が始まった歩」を拾い、
  同じ記事の 2 サンプルでそろうか（滴定寄りか核寄りか）、始まる手前に助走があるか、を数えて
  `A2_explore/onset_summary.json` と `A2_explore/onset_detail.npz` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_explore_onset.py`（`{h(ON)[:16]}…`）を埋め込み。生成もモデルの読み込みもしません。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = (f"# セル2：計算のコード\nimport hashlib, json, math\nSRCON = {ON!r}\n"
        f"assert hashlib.sha256(SRCON.encode()).hexdigest() == {h(ON)!r}\nexec(SRCON)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
import numpy as np
try:
    r, allon = run_onset({BASE!r})
    r['a2_explore_onset_sha256'] = {h(ON)!r}
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/onset_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    np.savez_compressed({BASE!r} + '/A2_explore/onset_detail.npz', **{{k.replace('/', '__'): v for k, v in allon.items()}})
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 結晶の始まり')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore onset', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '結晶の始まる歩を A2_explore/onset_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_onset_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(ON))
