"""探索 8.33（下見）：文章の DSC（ヒステリシス）のノートブックを、モデルごとに 1 本ずつ組み立てる（L4、2 本）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc'))
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
OUT = BASE + '/A2_explore/dsc'
for key, (name, hf_id, short) in {'pythia': ('pythia-1b', 'EleutherAI/pythia-1b', 'Pythia-1B'), 'olmo': ('olmo2-1b', 'allenai/OLMo-2-0425-1B', 'OLMo-2-1B')}.items():
    label = f'logit-thermo 探索 文章の DSC（ヒステリシスの下見、{short}、温度 0.85↔0.35 の往復 × 場 3 本、L4）'
    head = f"""# logit-thermo 探索：文章の DSC ——温度をゆっくり下げて上げ、凍る温度と溶ける温度のずれ（ヒステリシス）を測る下見（{short}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 1〜1.5 時間。終わったら自動で切れます。
- もう 1 本（{'OLMo-2-1B' if key == 'pythia' else 'Pythia-1B'}）のノートブックと、別のランタイムで同時に回せます（保存するファイルの名前が違うので、ぶつかりません）。
- 探索（下見、予想なし）。1 本の生成の中で温度を 0.85 → 0.35 → 0.85（と、その逆の順）にゆっくり動かし、写しの候補への場 h（−0.3・0・＋0.3）ごとに、凍る温度と溶ける温度のずれを測る材料を集めます。h ＝ 0 だけ、2 倍の速さの往復も。記事 40 × サンプル 4。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードと `scripts/a2_explore_field.py`（`{h(CF)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_explore_dsc.py`（`{h(CD)[:16]}…`）は{'このノートブックに埋め込み' if key == 'pythia' else ' Pythia-1B のノートブック（同じ Drive のフォルダ）から取り出してハッシュで確かめる'}。
- 保存先 `A2_explore/dsc/`。腕は 8 本。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。**結果は表示しません。**
- **通知**：🚀・腕ごとの ✅（8 回）・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    core = ("# セル2：計算のコード\n" + EXTRACT +
            f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}"
            + ("" if key == 'pythia' else f", 'SRCD': {h(CD)!r}") + "})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF']\n"
            + (f"SRCD = {CD!r}\nassert hashlib.sha256(SRCD.encode()).hexdigest() == {h(CD)!r}\n" if key == 'pythia' else "SRCF, SRCD = _s['SRCF'], _s['SRCD']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n")
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 8 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = D_MODELS[{key!r}]
    run_dsc(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(model=[name, hf_id], arms=D_ARMS, legs=D_LEG, T=[D_TH, D_TL], samples=D_S, chunk=D_CHUNK, seed=D_SEED,
                   field_sha256={h(CF)!r}, dsc_sha256={h(CD)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + f'/run_meta_{{name}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'探索 文章の DSC（{short}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - explore DSC hysteresis {short}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'文章の DSC（{short}）の材料を A2_explore/dsc に保存しました'))],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_explore_dsc_{key}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(CD)[:16])
