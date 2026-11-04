"""A2-11（印の大きさの確かめ、T_m ÷ T_0 のそろい方）のノートブック（CPU）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON, A8, A11 = rd('scripts/a2_explore_onset.py'), rd('scripts/a2_8_vh.py'), rd('scripts/a2_11_probe_univ.py')
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
OUT = BASE + '/A2_11'
LABEL = 'logit-thermo A2-11 印の大きさの確かめと T_m ÷ T_0 のそろい方（CPU）'
HEAD = f"""# logit-thermo A2-11：印の大きさの確かめと、T_m ÷ T_0 のそろい方（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 15〜30 分。
- ゲート：`docs/A2_11_GATE.md`。A2-8 の読み直しで写しの印を 3・4・6 トークンに変えて n と T_m を出し直し（C）、A2-2・A2-5・A2-6 の 25 本で T_m⁺ ÷ T_0 を出して（D）判定し、`A2_11/a2_11_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：結晶の始まりの探索と A2-8 のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_11_probe_univ.py`（`{h(A11)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport math\n" + EXTRACT +
        f"_s = take_src({{'SRCON': {h(ON)!r}, 'SRC8': {h(A8)!r}}})\nSRCON, SRC8 = _s['SRCON'], _s['SRC8']\n"
        f"SRC11 = {A11!r}\nassert hashlib.sha256(SRC11.encode()).hexdigest() == {h(A11)!r}\n"
        "exec(SRCON)\nexec(SRC8)\nexec(SRC11)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset(), self_test_a8(), self_test_a11())\n")
RUN = f"""# セル3：数えて判定して保存（結果は表示しない）
try:
    r = analyze11({BASE!r})
    r.update(gate='docs/A2_11_GATE.md', onset_sha256={h(ON)!r}, a2_8_sha256={h(A8)!r}, a2_11_sha256={h(A11)!r})
    os.makedirs({OUT!r}, exist_ok=True)
    json.dump(_clean(r), open({OUT!r} + '/a2_11_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-11')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-11 probe and universality', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'A2-11 の判定を A2_11/a2_11_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_11_probe_univ_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(A11))
