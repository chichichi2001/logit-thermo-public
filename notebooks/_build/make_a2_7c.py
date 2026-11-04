"""A2-7c（Qwen のやり直し）のデータ集め（L4、1 本）と分析（CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_4_collect.py'), rd('scripts/a2_7_collect.py')
C7B, A7B = rd('scripts/a2_7b_collect.py'), rd('scripts/a2_7b_analyze.py')
C7C, A7C = rd('scripts/a2_7c_collect.py'), rd('scripts/a2_7c_analyze.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A4 = rd('scripts/a2_4_analyze.py')
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
OUT = BASE + '/A2_7C'
label = 'logit-thermo A2-7c 本滴定やり直し データ集め（Qwen2.5-1.5B、温度 0.6〜1.05、L4）'
head = f"""# logit-thermo A2-7c：本滴定のやり直し（Qwen2.5-1.5B、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは約 1.5〜2 時間。終わったら自動で切れます。
- ゲート：`docs/A2_7C_GATE.md`。A2-7b の Qwen だけを、温度 0.6〜1.05（A2-7 と同じ 8 つ）でやり直す。腕は A2-7b と同じ 8 本。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7c_collect.py`（`{h(C7C)[:16]}…`）は埋め込み。
- 保存先 `A2_7C/`。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。
- **通知**：🚀・腕ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
core = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}}})\n"
        "SRC22, SRC24, SRC7C, SRC7BC = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC']\n"
        f"SRC7CC = {C7C!r}\nassert hashlib.sha256(SRC7CC.encode()).hexdigest() == {h(C7C)!r}\n"
        "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRC7CC)\nprint('コードを置きました')\n")
run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 8 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_7c())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = B7_GROUPS[C7_GROUP]
    run_7b(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(gate='docs/A2_7C_GATE.md', group=C7_GROUP, temps=B7_TEMPS, a2_7b_collect_sha256={h(C7B)!r},
                   a2_7c_collect_sha256={h(C7C)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_qwen.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-7c のデータ集め（Qwen2.5-1.5B）')}"""
nb = {"cells": [md(head), code(notify_cell(label, 'logit-thermo - A2-7c titration qwen redo', GPU_CHECK)), code(core), code(run),
                code(DONE.replace('{what}', 'A2-7c の Qwen2.5-1.5B を A2_7C に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
LABEL_A = 'logit-thermo A2-7c 本滴定やり直し 分析（Qwen の判定、CPU）'
HEAD_A = f"""# logit-thermo A2-7c：本滴定のやり直しの分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜10 分。**データ集めが終わってから**回してください。
- `A2_7C/` の Qwen の 8 本の腕を読み、A2-7b と同じ決め方で判定して `A2_7C/a2_7c_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-4・A2-7b の分析のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7c_analyze.py`（`{h(A7C)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRCA24': {h(A4)!r}, 'SRC7BA': {h(A7B)!r}}})\n"
          "SRCA22, SRCA24, SRC7BA = _s['SRCA22'], _s['SRCA24'], _s['SRC7BA']\n"
          f"SRC7CA = {A7C!r}\nassert hashlib.sha256(SRC7CA.encode()).hexdigest() == {h(A7C)!r}\n"
          "exec(SRCA22)\nexec(SRCA24)\nexec(SRC7BA)\nprint('自己テスト（A2-7b の分析）：', self_test_a27b_analyze())\nexec(SRC7CA)\n"
          "assert B7A_TEMPS == [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] and B7A_MODELS == ['qwen2.5-1.5b']\nprint('温度とモデルを置きかえました')\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze7c({OUT!r})
    r['a2_7c_analyze_sha256'] = {h(A7C)!r}
    json.dump(_clean(r), open({OUT!r} + '/a2_7c_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-7c の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-7c analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定を A2_7C/a2_7c_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
for n, fn in [(nb, 'A2_7C_collect_qwen_L4.ipynb'), (nba, 'A2_7C_analyze_CPU.ipynb')]:
    p = ROOT / 'notebooks' / fn
    p.write_text(json.dumps(n, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
