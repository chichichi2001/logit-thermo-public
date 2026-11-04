"""A2-9（文章の結合エネルギー）のノートブック（L4、1 本：読ませる → 分析）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, A9 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_9_bond.py')
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
OUT = BASE + '/A2_9'
LABEL = 'logit-thermo A2-9 文章の結合エネルギー（代名詞・名前の再登場・足し算、3 モデル、L4）'
HEAD = f"""# logit-thermo A2-9：文章の結合エネルギー（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 10〜20 分。終わったら自動で切れます。
- ゲート：`docs/A2_9_GATE.md`。型を決めて作った短い文を Pythia-1B・OLMo-2-1B・Qwen2.5-1.5B に読ませ、名前を入れ替えて「結合」を切ったときに、
  代名詞や 2 回目の名字の −log p がどれだけ上がるかを測って判定し、`A2_9/a2_9_summary.json` に書きます。**結果も文も表示しません。**
- 計算のコード：A2-2 のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_9_bond.py`（`{h(A9)[:16]}…`）は埋め込み（分析の部分はデータを開く前に固定）。
- 保存先 `A2_9/`。途中で切れても、もう一度「すべて実行」すれば終わったモデルは飛ばします。
- **通知**：🚀・モデルごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport gc, math\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}}})\nSRC22 = _s['SRC22']\n"
        f"SRC9 = {A9!r}\nassert hashlib.sha256(SRC9.encode()).hexdigest() == {h(A9)!r}\n"
        "exec(SRC22)\nexec(SRC9)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 読ませる（モデルごと）→ 分析 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a9(), self_test_score9('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    items = build_items()
    allres = {{}}
    for name, hf_id in A9_MODELS:
        p = {OUT!r} + f'/raw_{{name}}.json'
        if not os.path.exists(p):
            t0 = time.time()
            r = run_bond(name, hf_id, items, 'cuda'); gc.collect(); torch.cuda.empty_cache()
            json.dump(r, open(p + '.part', 'w')); os.replace(p + '.part', p)
            notify(f'✅ {{name}} おわり（{{time.time() - t0:.0f}} 秒）', tags='white_check_mark')
        allres[name] = json.load(open(p))
    r = analyze9(allres)
    r.update(gate='docs/A2_9_GATE.md', a2_9_sha256={h(A9)!r}, a2_2_collect_sha256={h(C22)!r},
             torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
             gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision())
    json.dump(_clean(r), open({OUT!r} + '/a2_9_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-9（文章の結合エネルギー）')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-9 bond energy', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'A2-9 の判定を A2_9/a2_9_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_9_bond_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, 'a2_9', h(A9))
