"""A2-6F（床と天井）の分析ノートブック（CPU）を組み立てる。A2-2・A2-5・探索 C のコードは Drive の前のノートブックから取り出す。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22 = rd('scripts/a2_2_collect.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
CX = rd('scripts/a2_explore_copy.py').replace("\nif __name__ == '__main__':", "\nif False:")
A5, F6 = rd('scripts/a2_5_analyze.py'), rd('scripts/a2_6f_floor.py')
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
LABEL = 'logit-thermo A2-6F 分析（しらふ同士の床と天井、CPU）'
HEAD = f"""# logit-thermo A2-6F：しらふ同士の床と、家族の差の天井（CPU）

- **必要な secret：なし**（`HF_TOKEN` があればトークナイザーの取り込みが少し速くなるだけ）。**ランタイム：CPU（標準）**。見込みは 20〜30 分。
- **A2-6 のデータ集め 3 本と、A2-6 の分析が終わってから**回してください（A2-6 の要約を読みます）。
- ゲート：`docs/A2_6F_GATE.md`。床を 3 種類（種だけ違う 4 組／1 回の中の 2 本のサンプルを分けた 25 組／記事を 20 本ずつに分けた 25 組）出し、`A2_6/a2_6f_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・探索 C・A2-5 のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_6f_floor.py`（`{h(F6)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：分析のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRCA22': {h(A22)!r}, 'SRCX': {h(CX)!r}, 'SRC5A': {h(A5)!r}}})\n"
        "SRC22, SRCA22, SRCX, SRC5A = _s['SRC22'], _s['SRCA22'], _s['SRCX'], _s['SRC5A']\n"
        f"SRC6F = {F6!r}\nassert hashlib.sha256(SRC6F.encode()).hexdigest() == {h(F6)!r}\n"
        "exec(SRC22)\nexec(SRCA22)\nexec(SRCX)\nexec(SRC5A)\nexec(SRC6F)\nprint('自己テスト：', self_test_a26f())\n")
RUN = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    # A2-2 の分析のコード（SRCA22）が SEED を上書きするので、記事を選ぶあいだだけ A2-2 の種（20261027）にする（A2-5 の分析と同じ）
    _seed_keep = SEED
    SEED = 20261027
    docs, meta = load_docs()
    SEED = _seed_keep
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = analyze6f({BASE!r}, docs)
    r['a2_6f_floor_sha256'] = {h(F6)!r}
    json.dump(_clean(r), open({BASE!r} + '/A2_6/a2_6f_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-6F の分析')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-6F floor', "")), code(CORE), code(RUN),
                code(DONE.replace('{what}', '床と天井を A2_6/a2_6f_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_6F_floor_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size); print('a2_6f_floor', h(F6)); print('copy', h(CX)); print('a22', h(A22))
