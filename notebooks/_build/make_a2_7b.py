"""A2-7b（本滴定）のデータ集め（L4、モデルごとに 3 本）と分析（CPU）のノートブックを組み立てる。make_a2_7.py から作った。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_4_collect.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A4 = rd('scripts/a2_4_analyze.py')
A5 = rd('scripts/a2_5_analyze.py')
C7 = rd('scripts/a2_7_collect.py')
C7B, A7B = rd('scripts/a2_7b_collect.py'), rd('scripts/a2_7b_analyze.py')
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
OUT7B = BASE + '/A2_7B'
GROUPS = {'pythia': ('Pythia-1B', '約 1.5〜2 時間'), 'olmo': ('OLMo-2-1B', '約 1.5〜2 時間'), 'qwen': ('Qwen2.5-1.5B', '約 2〜2.5 時間')}
out = []
for g, (title, eta) in GROUPS.items():
    label = f'logit-thermo A2-7b 本滴定 データ集め（{title}、写し q・偽薬 r の 8 本、L4）'
    head = f"""# logit-thermo A2-7b：本滴定のデータ集め（{title}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは{eta}。終わったら自動で切れます。
- ゲート：`docs/A2_7B_GATE.md`。{title} で 8 本の腕（なし・なし（別の種）・写しの禁止 q ＝ 1/3・2/3・1・偽薬 r ＝ 2.5%・5%・7.5%）を、温度 8 つ（0.4〜1.0）で。データ集めはモデルごとに 3 本あり、同時に回して大丈夫です。
- 計算のコード：`scripts/a2_2_collect.py`・`scripts/a2_4_collect.py`・`scripts/a2_7_collect.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7b_collect.py`（`{h(C7B)[:16]}…`）は埋め込み。
- 保存先 `A2_7B/`。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。
- **通知**：🚀・腕ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    core = ("# セル2：計算のコード\n" + EXTRACT +
            f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}}})\n"
            "SRC22, SRC24, SRC7C = _s['SRC22'], _s['SRC24'], _s['SRC7C']\n"
            f"SRC7BC = {C7B!r}\nassert hashlib.sha256(SRC7BC.encode()).hexdigest() == {h(C7B)!r}\n"
            "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 8 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
GROUP = {g!r}
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'))
    os.makedirs({OUT7B!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = B7_GROUPS[GROUP]
    run_7b(name, hf_id, docs, {OUT7B!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(gate='docs/A2_7B_GATE.md', group=GROUP, a2_2_collect_sha256={h(C22)!r}, a2_4_collect_sha256={h(C4)!r},
                   a2_7_collect_sha256={h(C7)!r}, a2_7b_collect_sha256={h(C7B)!r}, torch=torch.__version__,
                   transformers=transformers.__version__, python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                   matmul=torch.get_float32_matmul_precision()),
              open({OUT7B!r} + f'/run_meta_{{GROUP}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-7b のデータ集め（{title}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-7b titration {g}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'A2-7b の {title} を A2_7B に保存しました'))],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    out.append((nb, f'A2_7B_collect_{g}_L4.ipynb'))
LABEL_A = 'logit-thermo A2-7b 本滴定 分析（判定と要約、CPU）'
HEAD_A = f"""# logit-thermo A2-7b：本滴定の分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 10〜20 分。**データ集めの 3 本が全部終わってから**回してください。
- `A2_7B/` の 3 モデル × 8 本の腕を読み、判定を計算して `A2_7B/a2_7b_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-4 の分析のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_7b_analyze.py`（`{h(A7B)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRCA24': {h(A4)!r}}})\n"
          "SRCA22, SRCA24 = _s['SRCA22'], _s['SRCA24']\n"
          f"SRC7BA = {A7B!r}\nassert hashlib.sha256(SRC7BA.encode()).hexdigest() == {h(A7B)!r}\n"
          "exec(SRCA22)\nexec(SRCA24)\nexec(SRC7BA)\nprint('自己テスト：', self_test_a27b_analyze())\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze7b({OUT7B!r})
    r['a2_7b_analyze_sha256'] = {h(A7B)!r}
    json.dump(_clean(r), open({OUT7B!r} + '/a2_7b_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-7b の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-7b analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定と要約を A2_7B/a2_7b_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
out.append((nba, 'A2_7B_analyze_CPU.ipynb'))
for nb, fn in out:
    p = ROOT / 'notebooks' / fn
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
print('c7b', h(C7B)); print('a7b', h(A7B)); print('c7', h(C7)); print('c22', h(C22)); print('c4', h(C4)); print('a22', h(A22)); print('a4', h(A4))
