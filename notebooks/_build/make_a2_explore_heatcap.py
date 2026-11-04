"""探索：比熱の山（A2-2・A2-5・A2-6 の温度計の材料 tcal を取り出す。CPU）のノートブックを組み立てる。"""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
BASE = '/content/drive/MyDrive/logit-thermo'
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 比熱の山（温度計の材料を取り出す、CPU）'
HEAD = f"""# logit-thermo 探索：比熱の山と T_0（温度計の材料を取り出す、CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 2〜5 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 の 25 本の保存ファイルから、人の文章を読ませたときの「温度計の材料」`tcal`（記事ごと・61 の温度ごとの Σ(⟨z⟩_T − z_y)）だけを取り出して、
  小さなファイル `A2_explore/tcal_all.npz` にまとめます（Claude が手元で比熱を計算するため。元のファイルは 10MB を超えて Claude が直接取り込めない）。**結果は表示しません。**
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
RUN = f"""# セル2：取り出して保存（結果は表示しない）
import numpy as np, os
RUNS = {{'A2_2': ['pythia-1b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-inst', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b', 'smollm2-1.7b-inst'],
        'A2_5': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
                 'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000'],
        'A2_6': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
                 'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b']}}
try:
    out, missing = {{}}, []
    for folder, names in RUNS.items():
        for n in names:
            p = {BASE!r} + f'/{{folder}}/{{n}}.npz'
            if not os.path.exists(p):
                missing.append(p); continue
            Z = np.load(p)
            if 'tcal' not in Z.files:
                missing.append(p + ' (tcal なし)'); continue
            out[f'{{folder}}/{{n}}|tcal'] = Z['tcal'].astype(np.float64)
            out[f'{{folder}}/{{n}}|tcal_T'] = Z['tcal_T'].astype(np.float64)
    np.savez_compressed({BASE!r} + '/A2_explore/tcal_all.npz', missing=np.array(missing), **out)
    print('保存しました：', len(out) // 2, '本。見つからなかったもの：', len(missing))
except Exception as e:
{FAIL.replace('{where}', '探索 比熱の山（温度計の材料を取り出す）')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore heat capacity extract', "")), code(RUN),
                code(DONE.replace('{what}', '温度計の材料を A2_explore/tcal_all.npz に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_heatcap_extract_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size)
