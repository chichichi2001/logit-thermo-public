"""A2-7（滴定）のデータ集め（L4、モデルごとに 3 本）と分析（CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_4_collect.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A4 = rd('scripts/a2_4_analyze.py')
A5 = rd('scripts/a2_5_analyze.py')
C7, A7 = rd('scripts/a2_7_collect.py'), rd('scripts/a2_7_analyze.py')
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
OUT7, OUT4 = BASE + '/A2_7', BASE + '/A2_4'
GROUPS = {'pythia': ('Pythia-1B', '約 1 時間'), 'olmo': ('OLMo-2-1B', '約 1 時間'), 'qwen': ('Qwen2.5-1.5B', '約 1.5 時間')}
out = []
for g, (title, eta) in GROUPS.items():
    label = f'logit-thermo A2-7 滴定 データ集め（{title}、n ＝ 8・6・3・2、L4）'
    head = f"""# logit-thermo A2-7：滴定のデータ集め（{title}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは{eta}。終わったら自動で切れます。
- ゲート：`docs/A2_7_GATE.md`。{title} で、禁止する写しの長さ n ＝ 8・6・3・2 の 4 回（温度は T_0 のまわりの 8 つ）。A2-7 のデータ集めはモデルごとに 3 本あり、同時に回して大丈夫です。
- 計算のコード：`scripts/a2_2_collect.py`・`scripts/a2_4_collect.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7_collect.py`（`{h(C7)[:16]}…`）は{'埋め込み' if g == 'pythia' else 'Pythia のノートブックから取り出してハッシュで確かめる'}。
- 保存先 `A2_7/`。途中で切れても、もう一度「すべて実行」すれば終わった回は飛ばします。
- **通知**：🚀・回ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    if g == 'pythia':
        core = ("# セル2：計算のコード\n" + EXTRACT +
                f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}}})\nSRC22, SRC24 = _s['SRC22'], _s['SRC24']\nSRC7C = {C7!r}\n"
                f"assert hashlib.sha256(SRC7C.encode()).hexdigest() == {h(C7)!r}\n"
                "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nprint('コードを置きました')\n")
    else:
        core = ("# セル2：計算のコード（A2-7 のコードは Pythia のノートブックに埋め込んだものを取り出す）\n" + EXTRACT +
                f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}}})\n"
                "SRC22, SRC24, SRC7C = _s['SRC22'], _s['SRC24'], _s['SRC7C']\n"
                "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ n ＝ 8・6・3・2 → 保存（結果は表示しない）
import json, sys, transformers
GROUP = {g!r}
try:
    print('自己テスト：', self_test('cuda'), self_test_pen('cuda'), self_test_a27('cuda'))
    os.makedirs({OUT7!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = A7_GROUPS[GROUP]
    for n in A7_NS:
        run_ngram(name, hf_id, docs, {OUT7!r}, 'cuda', n, notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(gate='docs/A2_7_GATE.md', group=GROUP, a2_2_collect_sha256={h(C22)!r}, a2_4_collect_sha256={h(C4)!r},
                   a2_7_collect_sha256={h(C7)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0)),
              open({OUT7!r} + f'/run_meta_{{GROUP}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-7 のデータ集め（{title}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-7 titration {g}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'A2-7 の {title} を A2_7 に保存しました'))],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    out.append((nb, f'A2_7_collect_{g}_L4.ipynb'))
LABEL_A = 'logit-thermo A2-7 滴定 分析（判定と要約、CPU）'
HEAD_A = f"""# logit-thermo A2-7：滴定の分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 15〜30 分。**データ集めの 3 本が全部終わってから**回してください。
- `A2_4/` の「なし」と n ＝ 4（block）、`A2_7/` の n ＝ 8・6・3・2 を読み、判定を計算して `A2_7/a2_7_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-4・A2-5 の分析のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7_analyze.py`（`{h(A7)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRCA24': {h(A4)!r}, 'SRC5A': {h(A5)!r}}})\n"
          "SRCA22, SRCA24, SRC5A = _s['SRCA22'], _s['SRCA24'], _s['SRC5A']\n"
          f"SRC7A = {A7!r}\nassert hashlib.sha256(SRC7A.encode()).hexdigest() == {h(A7)!r}\n"
          "exec(SRCA22)\nexec(SRCA24)\nexec(SRC5A)\nexec(SRC7A)\nprint('自己テスト：', self_test_a27_analyze())\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze7({OUT4!r}, {OUT7!r})
    r['a2_7_analyze_sha256'] = {h(A7)!r}
    json.dump(_clean(r), open({OUT7!r} + '/a2_7_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-7 の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-7 analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定と要約を A2_7/a2_7_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
out.append((nba, 'A2_7_analyze_CPU.ipynb'))
for nb, fn in out:
    p = ROOT / 'notebooks' / fn
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
print('c7', h(C7)); print('a7', h(A7)); print('c4', h(C4)); print('a4', h(A4))
