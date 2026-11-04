"""A2-12（1 回の禁止で乱れるトークンの数 k）のノートブック（L4、1 本：生成 → 分析）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C7, A12 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_7_collect.py'), rd('scripts/a2_12_kdist.py')
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
OUT = BASE + '/A2_12'
LABEL = 'logit-thermo A2-12 1 回の禁止で乱れるトークンの数 k（3 モデル、L4）'
HEAD = f"""# logit-thermo A2-12：1 回の禁止で乱れるトークンの数 k（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 20〜30 分。終わったら自動で切れます。
- ゲート：`docs/A2_12_GATE.md`。同じ途中の状態から「何もしない」と「1 位を 1 回だけ禁止」の 2 本を、同じノイズで 64 歩生成して、食い違うトークンの数を数え、
  凝固点降下のずれから先に決めた予想の k と比べて `A2_12/a2_12_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：A2-2・A2-7 のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_12_kdist.py`（`{h(A12)[:16]}…`）は埋め込み（分析の部分はデータを開く前に固定）。
- 保存先 `A2_12/`。途中で切れても、もう一度「すべて実行」すれば終わったモデルは飛ばします。
- **通知**：🚀・温度ごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport gc, math\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC7C': {h(C7)!r}}})\nSRC22, SRC7C = _s['SRC22'], _s['SRC7C']\n"
        f"SRC12 = {A12!r}\nassert hashlib.sha256(SRC12.encode()).hexdigest() == {h(A12)!r}\n"
        "exec(SRC22)\nexec(SRC7C)\nexec(SRC12)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, (np.floating, np.integer)): o = o.item()\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 生成（モデルごと）→ 分析 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a12(), self_test_k12('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    for name, hf_id in K12_MODELS:
        run_k12(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    Zs = {{name: dict(np.load({OUT!r} + f'/{{name}}.npz')) for name, _ in K12_MODELS}}
    r = analyze12(Zs)
    r.update(gate='docs/A2_12_GATE.md', a2_12_sha256={h(A12)!r}, a2_2_collect_sha256={h(C22)!r}, a2_7_collect_sha256={h(C7)!r},
             torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
             gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision())
    json.dump(_clean(r), open({OUT!r} + '/a2_12_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-12（1 回の禁止で乱れるトークンの数）')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - A2-12 tokens disturbed per ban', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'A2-12 の判定を A2_12/a2_12_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_12_kdist_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(A12))
