"""探索 8.30（下見）：場を足すと融点はどう動くか（L4、1 本）のノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field'))
src5, src3, src4 = rd('notebooks/_build/make_a2_5.py'), rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
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
OUT = BASE + '/A2_explore/field'
label = 'logit-thermo 探索 場と融点（クラペイロンの下見、Pythia-1B・OLMo-1B、h ＝ −0.3・0・＋0.3、L4）'
head = f"""# logit-thermo 探索：場を足すと融点はどう動くか ——文章のクラウジウス・クラペイロンの下見（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは約 2 時間。終わったら自動で切れます。
- 探索（下見、予想なし）。写しの候補のトークンの logit に h（−0.3・0・＋0.3）を足して生成し、融点がどれだけ動くかを測る材料を集めます。モデルは Pythia-1B と OLMo-2-1B、温度 0.4〜0.7 の 7 個、サンプル 4。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_field.py`（`{h(CF)[:16]}…`）は埋め込み。
- 保存先 `A2_explore/field/`。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。**結果は表示しません。**
- **通知**：🚀・腕ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
core = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}}})\n"
        "SRC22, SRC24, SRC7C, SRC7BC = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC']\n"
        f"SRCF = {CF!r}\nassert hashlib.sha256(SRCF.encode()).hexdigest() == {h(CF)!r}\n"
        "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nprint('コードを置きました')\n")
run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ モデルごとに 3 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    for name, hf_id in F_MODELS:
        run_field(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(models=F_MODELS, temps=F_TEMPS, h=F_H, samples=F_S, seed=F_SEED, a2_7b_collect_sha256={h(C7B)!r},
                   field_sha256={h(CF)!r}, torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
                   gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 場と融点（クラペイロンの下見）')}"""
nb = {"cells": [md(head), code(notify_cell(label, 'logit-thermo - explore field Clapeyron', GPU_CHECK)), code(core), code(run),
                code(DONE.replace('{what}', '場と融点の下見の材料を A2_explore/field に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_field_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(CF)[:16])
