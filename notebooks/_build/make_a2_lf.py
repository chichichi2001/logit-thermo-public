"""A2-LF（格子の遠い床）のノートブック（L4）を組み立てる。A2-2・探索 X/Y・探索 8.8 のコードは Drive の前のノートブックから取り出す。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, CXY, CT, LF = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_xy.py'), rd('scripts/a2_explore_lattice.py'), rd('scripts/a2_lf_floor.py')
src5 = rd('notebooks/_build/make_a2_5.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
notify_cell, DONE, FAIL, EXTRACT, GPU_CHECK = ns['notify_cell'], ns['DONE'], ns['FAIL'], ns['EXTRACT'], ns['GPU_CHECK']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_explore'
LABEL = 'logit-thermo A2-LF 格子の遠い床（よそ者と身内の差し替え、4 モデル、L4）'
HEAD = f"""# logit-thermo A2-LF：格子の「遠い床」は本物か（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 1〜1.5 時間（探索 8.8 の半分の計算）。終わったら自動で切れます。
- ゲート：`docs/A2_LF_GATE.md`。探索 8.8 の格子と同じ 38 本・4 モデルで、差し替え先を「よそ者（ほかの記事の言葉）」と「身内（同じ記事の言葉）」の 2 通りにして、遠い床をくらべます。判定もこのノートブックの中で計算して `A2_explore/lf_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_collect.py`・`scripts/a2_explore_xy.py`・`scripts/a2_explore_lattice.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_lf_floor.py`（`{h(LF)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・モデルごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRCXY': {h(CXY)!r}, 'SRCT': {h(CT)!r}}})\n"
        "SRC22, SRCXY, SRCT = _s['SRC22'], _s['SRCXY'], _s['SRCT']\n"
        f"SRCLF = {LF!r}\nassert hashlib.sha256(SRCLF.encode()).hexdigest() == {h(LF)!r}\n"
        "exec(SRC22)\nexec(SRCXY)\nexec(SRCT)\nexec(SRCLF)\nprint('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 4 モデル × 2 通りの差し替え → 判定 → 保存（結果は表示しない）
import json
try:
    print('自己テスト：', self_test('cuda'), self_test_xy('cuda'), self_test_lattice('cuda'), self_test_lf())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = run_lf(docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    r['a2_lf_floor_sha256'] = {h(LF)!r}
    json.dump(r, open({OUT!r} + '/lf_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-LF')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-LF far floor', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', '判定と要約を A2_explore/lf_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_LF_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size); print('lf', h(LF)); print('22', h(C22)); print('xy', h(CXY)); print('lattice', h(CT))
