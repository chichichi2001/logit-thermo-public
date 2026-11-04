"""探索 T-scan（温度ごとの崩れ方）のノートブック（CPU）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, TS = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_tscan.py')
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
LABEL = 'logit-thermo 探索 T-scan（温度ごとの崩れ方、25 回、CPU）'
HEAD = f"""# logit-thermo 探索 T-scan：温度ごとの崩れ方（CPU）

- **必要な secret：なし**（`HF_TOKEN` があればトークナイザーの取り込みが少し速くなるだけ）。**ランタイム：CPU（標準）**。見込みは 15〜30 分。
- 探索（予想なし）。A2-2・A2-5・A2-6 で保存した生成のトークンをデコードして、温度ごとに「結晶（写し）・電子不足（代名詞）・具体物（固有名詞）・共有結合（登場人物のつながり）・気体（ほかの記事に出ない単語）」の割合を数え、`A2_explore/tscan_summary.json` に書きます。**文章の中身は表示も保存もしません。**
- 計算のコード：`scripts/a2_2_collect.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_tscan.py`（`{h(TS)[:16]}…`）は埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}}})\nSRC22 = _s['SRC22']\nSRCTS = {TS!r}\n"
        f"assert hashlib.sha256(SRCTS.encode()).hexdigest() == {h(TS)!r}\n"
        "exec(SRC22)\nexec(SRCTS)\nprint('自己テスト：', self_test_tscan())\n")
RUN = f"""# セル3：記事（A2-2 と同じか確かめる）→ 25 回を数える → 保存（結果は表示しない）
import json
try:
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = run_tscan({BASE!r}, docs)
    r['a2_explore_tscan_sha256'] = {h(TS)!r}
    json.dump(r, open({BASE!r} + '/A2_explore/tscan_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 T-scan')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore T-scan', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '温度ごとの崩れ方を A2_explore/tscan_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_tscan_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(TS))
