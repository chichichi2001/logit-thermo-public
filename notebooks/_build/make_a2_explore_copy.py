"""探索（候補 C の手）：写し歩と新しい歩に分けるノートブック（CPU）を組み立てる。コードはそのまま埋め込む。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22, CX = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_copy.py')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
src = rd('notebooks/_build/make_a2_3.py')
# 通知のセルと最後のセルは A2-3 と同じものを使う（関数と文字列だけを取り出して実行する）
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src, re.S).group(0), ns)
notify_cell, DONE = ns['notify_cell'], ns['DONE']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
BASE = '/content/drive/MyDrive/logit-thermo'
OUT = BASE + '/A2_explore'
LABEL = 'logit-thermo 探索 C（写し歩とガラス、A2-2 と 7B の BBC、CPU）'

HEAD = f"""# logit-thermo 探索 C：裾が細るのは「結晶」か「ガラス」か（CPU）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：CPU（標準）**。見込みは 10〜20 分。
- **探索**です。A2-2 と A2-3（7B の BBC）の生のファイルを見直して、生成の 1 歩ずつを「写し歩」（それまでに出た 4 トークンの並びの続きをそのまま写した歩）と「新しい歩」に分け、書き進めたあとに裾が細る分が、どちらから来るかを計算します。確かめには使いません（`docs/A2_EXPLORE_T0_FEATURES.md` §5 の候補 C）。
- 最初に検量線：人の続きの前半 256 トークンを後半に写した「人工の結晶」を Qwen2.5-0.5B に CPU で読ませて（記事 10 本）、手が結晶を結晶と言えるかを確かめます。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）と `scripts/a2_explore_copy.py`（`{h(CX)[:16]}…`）をそのまま埋め込み。
- 結果は `MyDrive/logit-thermo/A2_explore/copy_summary.json` に保存します。**結果は表示しません。**
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったらランタイムを切ります。
"""
SECRET = """try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
"""
CORE = ("# セル2：計算のコード（a2_2_collect.py と a2_explore_copy.py をそのまま。同じ名前空間で実行）\nimport hashlib\n"
        f"SRC22 = {C22!r}\nSRCX = {CX.replace(chr(10) + 'if __name__ == ' + repr('__main__') + ':', chr(10) + 'if False:')!r}\n"
        f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
        f"assert hashlib.sha256(SRCX.encode()).hexdigest() == {h(CX.replace(chr(10) + 'if __name__ == ' + repr('__main__') + ':', chr(10) + 'if False:'))!r}\n"
        "exec(SRC22)\nexec(SRCX)\nprint('自己テスト：', self_test())\n")
RUN = f"""# セル3：記事を読み直して（A2-2 と同じか確かめる）、計算して保存（結果は表示しない）
import json, os
os.makedirs({OUT!r}, exist_ok=True)
docs, meta = load_docs()
old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
r = run_all({BASE!r}, docs, TEMPS)
r['a2_2_collect_sha256'] = {h(C22)!r}
r['a2_explore_copy_sha256'] = {h(CX)!r}
json.dump(r, open({OUT!r} + '/copy_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('保存しました')
"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore C', SECRET)), code(CORE), code(RUN),
                code(DONE.replace('{what}', '写し歩とガラスの分け方を copy_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_copy_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size, 'a2_explore_copy', h(CX))
