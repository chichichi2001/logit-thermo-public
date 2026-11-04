"""探索：偽薬で本当の融点 T_m は下がるか（A2-7b のトークンの下見、CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
ON, A8, TS = rd('scripts/a2_explore_onset.py'), rd('scripts/a2_8_vh.py'), rd('scripts/a2_explore_tm_salt.py')
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
LABEL = 'logit-thermo 探索 偽薬で本当の融点 T_m は下がるか（A2-7b の下見、CPU）'
HEAD = f"""# logit-thermo 探索：偽薬（塩）で本当の融点 T_m は下がるか（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜10 分。
- 探索（予想なし）。A2-7b の「なし」「なし（別の種）」「偽薬 2.5%・5%・7.5%」のトークンから、A2-8 と同じ決め方で凍る速さ・溶ける速さを数えて T_m を出し、
  `A2_explore/tm_salt_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：結晶の始まりの探索と A2-8 のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_tm_salt.py`（`{h(TS)[:16]}…`）は埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport math\n" + EXTRACT +
        f"_s = take_src({{'SRCON': {h(ON)!r}, 'SRC8': {h(A8)!r}}})\nSRCON, SRC8 = _s['SRCON'], _s['SRC8']\n"
        f"SRCTS = {TS!r}\nassert hashlib.sha256(SRCTS.encode()).hexdigest() == {h(TS)!r}\n"
        "exec(SRCON)\nexec(SRC8)\nexec(SRCTS)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_onset(), self_test_tm_salt())\n")
RUN = f"""# セル3：数えて保存（結果は表示しない）
try:
    r = tm_salt({BASE!r})
    r.update(onset_sha256={h(ON)!r}, a2_8_sha256={h(A8)!r}, tm_salt_sha256={h(TS)!r})
    os.makedirs({BASE!r} + '/A2_explore', exist_ok=True)
    json.dump(_clean(r), open({BASE!r} + '/A2_explore/tm_salt_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 偽薬で T_m は下がるか')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore Tm salt', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '偽薬の T_m を A2_explore/tm_salt_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_tm_salt_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(TS))
