"""A2-10（凝固点降下：偽薬を濃くして T_m と T_0 を測る）のデータ集め（L4、モデルごとに 1 本）と分析（CPU）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B = rd('scripts/a2_2_collect.py'), rd('scripts/a2_4_collect.py'), rd('scripts/a2_7_collect.py'), rd('scripts/a2_7b_collect.py')
C10, A10 = rd('scripts/a2_10_collect.py'), rd('scripts/a2_10_analyze.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
A4, A7B, ON, A8 = rd('scripts/a2_4_analyze.py'), rd('scripts/a2_7b_analyze.py'), rd('scripts/a2_explore_onset.py'), rd('scripts/a2_8_vh.py')
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
OUT = BASE + '/A2_10'
META_GPU = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}
NAMES = {'pythia': 'Pythia-1B', 'olmo': 'OLMo-2-1B', 'qwen': 'Qwen2.5-1.5B'}
for grp, nice in NAMES.items():
    label = f'logit-thermo A2-10 凝固点降下 データ集め（{nice}、偽薬 7.5・15・30%、温度 0.4〜1.05、L4）'
    head = f"""# logit-thermo A2-10：凝固点降下（{nice}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは約 2.5〜3 時間。終わったら自動で切れます。
- ゲート：`docs/A2_10_GATE.md`。腕は「なし」「なし（別の種）」「偽薬 7.5%・15%・30%」の 5 本、温度は 0.4〜1.05 の 14 個。モデルごとに 1 本のノートブック（3 枠で並べて回す）。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_10_collect.py`（`{h(C10)[:16]}…`）は埋め込み。
- 保存先 `A2_10/`。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。
- **通知**：🚀・腕ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    core = ("# セル2：計算のコード\n" + EXTRACT +
            f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC']\n"
            f"SRC10C = {C10!r}\nassert hashlib.sha256(SRC10C.encode()).hexdigest() == {h(C10)!r}\n"
            f"exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRC10C)\nC10_GROUP = {grp!r}\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 5 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_10())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = B7_GROUPS[C10_GROUP]
    run_7b(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(gate='docs/A2_10_GATE.md', group=C10_GROUP, temps=B7_TEMPS, arms=[a[0] for a in B7_ARMS], seed=SEED7B,
                   a2_7b_collect_sha256={h(C7B)!r}, a2_10_collect_sha256={h(C10)!r}, torch=torch.__version__,
                   transformers=transformers.__version__, python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                   matmul=torch.get_float32_matmul_precision()), open({OUT!r} + f'/run_meta_{{C10_GROUP}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-10 のデータ集め（{nice}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-10 freezing point {grp}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'A2-10 の {nice} を A2_10 に保存しました'))],
          "metadata": META_GPU, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_10_collect_{grp}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
LABEL_A = 'logit-thermo A2-10 凝固点降下 分析（T_m と T_0 の判定、CPU）'
HEAD_A = f"""# logit-thermo A2-10：凝固点降下の分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 10〜20 分。**3 本のデータ集めが全部終わってから**回してください。
- `A2_10/` の 3 モデル × 5 本の腕を読み、T_m（凍る速さと溶ける速さ）と T_0（ℓ の交点）を出して判定し、`A2_10/a2_10_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-4・A2-7b の分析、結晶の始まりの探索、A2-8 のコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_10_analyze.py`（`{h(A10)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRCA22': {h(A22)!r}, 'SRCA24': {h(A4)!r}, 'SRC7BA': {h(A7B)!r}, 'SRCON': {h(ON)!r}, 'SRC8': {h(A8)!r}}})\n"
          "SRCA22, SRCA24, SRC7BA, SRCON, SRC8 = _s['SRCA22'], _s['SRCA24'], _s['SRC7BA'], _s['SRCON'], _s['SRC8']\n"
          f"SRC10A = {A10!r}\nassert hashlib.sha256(SRC10A.encode()).hexdigest() == {h(A10)!r}\n"
          "exec(SRCA22)\nexec(SRCA24)\nexec(SRC7BA)\nexec(SRCON)\nexec(SRC8)\n"
          "print('自己テスト（前の部品）：', self_test_a27b_analyze(), self_test_onset(), self_test_a8())\n"
          "exec(SRC10A)\nprint('自己テスト（A2-10）：', self_test_a10())\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    r = analyze10({OUT!r})
    r['a2_10_analyze_sha256'] = {h(A10)!r}
    json.dump(_clean(r), open({OUT!r} + '/a2_10_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-10 の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-10 analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定を A2_10/a2_10_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_10_analyze_CPU.ipynb'
p.write_text(json.dumps(nba, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, 'c10', h(C10)[:16], 'a10', h(A10)[:16])
