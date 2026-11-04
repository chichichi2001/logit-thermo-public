"""A2-8（ファントホッフと熱量計）のノートブック（L4、1 本：読み直し → 分析）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, ON, A8 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_onset.py'), rd('scripts/a2_8_vh.py')
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
OUT = BASE + '/A2_8'
LABEL = 'logit-thermo A2-8 ファントホッフと熱量計（3 モデルの読み直し → 分析、L4）'
HEAD = f"""# logit-thermo A2-8：ファントホッフの ΔH と熱量計の ΔH（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 30〜50 分。終わったら自動で切れます。
- ゲート：`docs/A2_8_GATE.md`。新しく生成はしません。A2-2 と A2-7b（なし・なし（別の種））で保存した生成のトークンを、
  Pythia-1B・OLMo-2-1B・Qwen2.5-1.5B に読み直させて、1 トークンごとのエネルギー（温度 1 の −log p）を出し、
  凍る速さ・溶ける速さと合わせて判定して `A2_8/a2_8_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2 のデータ集めのコードと結晶の始まりの探索のコードは Drive の前のノートブックから取り出してハッシュで確かめ、
  `scripts/a2_8_vh.py`（`{h(A8)[:16]}…`）は埋め込み（分析の部分はデータを開く前に固定）。
- 保存先 `A2_8/`。途中で切れても、もう一度「すべて実行」すれば読み直しの終わったモデルは飛ばします。
- **通知**：🚀・材料ごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport gc, math\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRCON': {h(ON)!r}}})\nSRC22, SRCON = _s['SRC22'], _s['SRCON']\n"
        f"SRC8 = {A8!r}\nassert hashlib.sha256(SRC8.encode()).hexdigest() == {h(A8)!r}\n"
        "exec(SRC22)\nexec(SRCON)\nexec(SRC8)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 読み直し（モデルごと）→ 分析 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_onset(), self_test_a8(), self_test_score('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    for name, hf_id in A8_MODELS:
        run_score(name, hf_id, {BASE!r}, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    Zs = {{name: dict(np.load({OUT!r} + f'/{{name}}.npz')) for name, _ in A8_MODELS}}
    r = analyze8(Zs)
    r.update(gate='docs/A2_8_GATE.md', a2_8_sha256={h(A8)!r}, a2_2_collect_sha256={h(C22)!r}, onset_sha256={h(ON)!r},
             torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
             gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision())
    json.dump(_clean(r), open({OUT!r} + '/a2_8_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-8（ファントホッフと熱量計）')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-8 van t Hoff vs calorimetric', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'A2-8 の判定を A2_8/a2_8_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_8_vanthoff_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, 'a2_8', h(A8), 'a22', h(C22)[:16], 'onset', h(ON)[:16])
