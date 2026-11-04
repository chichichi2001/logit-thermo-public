"""探索 8.38（下見）：文章の DSC その 2（1.5 まで温める）のノートブックを、モデルごとに 1 本ずつ組み立てる（L4、2 本）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2'))
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
for key, (name, hf_id, short) in {'pythia': ('pythia-1b', 'EleutherAI/pythia-1b', 'Pythia-1B'), 'olmo': ('olmo2-1b', 'allenai/OLMo-2-0425-1B', 'OLMo-2-1B')}.items():
    OUT = BASE + f'/A2_explore/dsc2_{key}'
    label = f'logit-thermo 探索 文章の DSC その 2（1.5 まで温める・熟成・速く温める・強い罰、{short}、L4）'
    head = f"""# logit-thermo 探索：文章の DSC その 2 ——凍らせた結晶を 1.5 まで温めて、本当に溶ける温度と溶けるときの潜熱を測る下見（{short}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 50 分〜1 時間。終わったら自動で切れます。
- もう 1 本（{'OLMo-2-1B' if key == 'pythia' else 'Pythia-1B'}）のノートブックと、別のランタイムで同時に回せます（保存先のフォルダがモデルごとに別です）。
- 探索（下見、予想なし）。どの腕も、はじめの 640 歩は同じ（0.85 → 0.35 で冷やして凍らせる、同じ種）。そこから：基本（0.35 → 1.5 を 640 歩で温める）・熟成（0.35 で 320 歩待ってから温める）・速く温める（320 歩）・強い罰（温めるあいだだけ h ＝ −1.0）の 4 本。記事 40 × サンプル 4。各歩の q（候補の確率の和）とエネルギー U も記録します。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコード、`scripts/a2_explore_field.py`（`{h(CF)[:16]}…`）、`scripts/a2_explore_dsc.py`（`{h(CD)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_explore_dsc2.py`（`{h(CD2)[:16]}…`）は{'このノートブックに埋め込み' if key == 'pythia' else ' Pythia-1B のノートブック（同じフォルダ）から取り出してハッシュで確かめる'}。
- 保存先 `A2_explore/dsc2_{key}/`。腕は 4 本。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。**結果は表示しません。**
- **通知**：🚀・腕ごとの ✅（4 回）・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    want = f"'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}, 'SRCD': {h(CD)!r}" + ("" if key == 'pythia' else f", 'SRCD2': {h(CD2)!r}")
    core = ("# セル2：計算のコード\n" + EXTRACT + f"_s = take_src({{{want}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF, SRCD = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF'], _s['SRCD']\n"
            + (f"SRCD2 = {CD2!r}\nassert hashlib.sha256(SRCD2.encode()).hexdigest() == {h(CD2)!r}\n" if key == 'pythia'
               else "SRCD, SRCD2 = _s['SRCD'], _s['SRCD2']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n")
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 4 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = D_MODELS[{key!r}]
    run_dsc2(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(model=[name, hf_id], arms=D2_ARMS, cool=D2_COOL, heat_top=D2_HEAT_TOP, pen=D2_PEN, samples=D2_S, chunk=D_CHUNK, seed=D2_SEED,
                   dsc2_sha256={h(CD2)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + f'/run_meta_{{name}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'探索 文章の DSC その 2（{short}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - explore DSC2 melting {short}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'文章の DSC その 2（{short}）の材料を A2_explore/dsc2_{key} に保存しました'))],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_explore_dsc2_{key}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(CD2)[:16])
