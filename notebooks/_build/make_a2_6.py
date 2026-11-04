"""A2-6（乳の種類）のデータ集め（L4、家族ごとに 3 本）と分析（CPU）のノートブックを組み立てる。
A2-2・A2-5 のコードは Drive の前のノートブックから取り出し、ハッシュで確かめる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22 = rd('scripts/a2_2_collect.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
C5, A5 = rd('scripts/a2_5_collect.py'), rd('scripts/a2_5_analyze.py')
C6, A6 = rd('scripts/a2_6_collect.py'), rd('scripts/a2_6_analyze.py')
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
OUT6, OUT5 = BASE + '/A2_6', BASE + '/A2_5'
GROUPS = {'qwen25': ('Qwen2.5（0.5B・1.5B・3B）', '約 2 時間'), 'smol2': ('SmolLM2（135M・360M・1.7B）', '約 1 時間'),
          'qwen3': ('Qwen3（0.6B・1.7B・4B）', '約 2.5 時間')}
gpu_meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}
out = []
for g, (title, eta) in GROUPS.items():
    label = f'logit-thermo A2-6 データ集め（乳の種類、{title}、L4）'
    head = f"""# logit-thermo A2-6：データ集め（乳の種類、{title}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは{eta}。終わったら自動で切れます。
- ゲート：`docs/A2_6_GATE.md`。この 1 本は {title} の 3 回。A2-6 のデータ集めは家族ごとに 3 本あり、同時に回して大丈夫です。
- 記事は A2-2 と同じ 40 本（A2-2 の種で選ぶ）。生成の種は A2-5 と同じ（20261102）。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）と `scripts/a2_5_collect.py`（`{h(C5)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_6_collect.py`（`{h(C6)[:16]}…`）は{'埋め込み' if g == 'qwen25' else 'Qwen2.5 のノートブックから取り出してハッシュで確かめる'}。**logit-thermo フォルダの前のノートブックは動かさないでください。**
- 保存先 `A2_6/`。途中で切れても、もう一度「すべて実行」すれば終わった回は飛ばします。
- **通知**：🚀・回ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    if g == 'qwen25':   # 埋め込むのは 1 本だけ。ほかの 2 本は、この 1 本から取り出してハッシュで確かめる
        core = ("# セル2：計算のコード\n" + EXTRACT +
                f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC5C': {h(C5)!r}}})\nSRC22, SRC5C = _s['SRC22'], _s['SRC5C']\nSRC6C = {C6!r}\n"
                f"assert hashlib.sha256(SRC6C.encode()).hexdigest() == {h(C6)!r}\n"
                "exec(SRC22)\nexec(SRC5C)\nexec(SRC6C)\nprint('コードを置きました')\n")
    else:
        core = ("# セル2：計算のコード（A2-6 のコードは Qwen2.5 のノートブックに埋め込んだものを取り出す）\n" + EXTRACT +
                f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC5C': {h(C5)!r}, 'SRC6C': {h(C6)!r}}})\n"
                "SRC22, SRC5C, SRC6C = _s['SRC22'], _s['SRC5C'], _s['SRC6C']\n"
                "exec(SRC22)\nexec(SRC5C)\nexec(SRC6C)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 の種で選び、A2-2 と同じか確かめる）→ 3 回 → 保存（結果は表示しない）
import json, sys, transformers
GROUP = {g!r}
try:
    print('自己テスト：', self_test('cuda'), self_test_a26())
    os.makedirs({OUT6!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    m = run_a26(GROUP, docs, {OUT6!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    assert SEED == 20261027, '種が戻っていません'
    m.update(gate='docs/A2_6_GATE.md', a2_2_collect_sha256={h(C22)!r}, a2_5_collect_sha256={h(C5)!r}, a2_6_collect_sha256={h(C6)!r},
             torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
             gpu=torch.cuda.get_device_name(0), float32_matmul_precision=torch.get_float32_matmul_precision(),
             max_memory_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    json.dump(m, open({OUT6!r} + f'/run_meta_{{GROUP}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-6 のデータ集め（{title}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-6 collect {g}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'A2-6 の {title} を A2_6 に保存しました'))],
          "metadata": gpu_meta, "nbformat": 4, "nbformat_minor": 0}
    out.append((nb, f'A2_6_collect_{g}_L4.ipynb'))
# ---------- 分析（CPU） ----------
LABEL_A = 'logit-thermo A2-6 分析（乳の種類、判定と要約、CPU）'
HEAD_A = f"""# logit-thermo A2-6：分析（乳の種類、CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 15〜30 分。**データ集めの 3 本が全部終わってから**回してください。
- `A2_5/` の Pythia 6 回と `A2_6/` の 9 回を読み、ゲートの判定を計算して `A2_6/a2_6_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_analyze.py`・`scripts/a2_5_analyze.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_6_analyze.py`（`{h(A6)[:16]}…`）は埋め込み（データを開く前に固定）。記事の本文は使いません。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRC5A': {h(A5)!r}}})\nSRCA22, SRC5A = _s['SRCA22'], _s['SRC5A']\nSRC6A = {A6!r}\n"
          f"assert hashlib.sha256(SRC6A.encode()).hexdigest() == {h(A6)!r}\n"
          "exec(SRCA22)\nexec(SRC5A)\nexec(SRC6A)\nprint('自己テスト：', self_test_a26_analyze())\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze6({OUT5!r}, {OUT6!r})
    r['a2_6_analyze_sha256'] = {h(A6)!r}
    json.dump(_clean(r), open({OUT6!r} + '/a2_6_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-6 の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-6 analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定と要約を A2_6/a2_6_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
out.append((nba, 'A2_6_analyze_CPU.ipynb'))
for nb, fn in out:
    p = ROOT / 'notebooks' / fn
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
print('a2_6_collect', h(C6)); print('a2_6_analyze', h(A6)); print('a2_5_collect', h(C5))
