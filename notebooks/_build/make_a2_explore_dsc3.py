"""探索 8.39（下見）：文章の DSC その 3（高温高圧、h ＝ 0・＋0.5・＋1・＋2、2.0 ⇄ 0.35）のノートブックを、モデルごとに 1 本ずつ組み立てる（L4、2 本）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CD3 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_explore_dsc3'))
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
    OUT = BASE + f'/A2_explore/dsc3_{key}'
    label = f'logit-thermo 探索 文章の DSC その 3（高温高圧 h 0〜＋2、2.0 まで、{short}、L4）'
    head = f"""# logit-thermo 探索：文章の DSC その 3 ——高温高圧（強い ＋h）で、凍る・溶けるの境目の線はどうなるか（臨界点はあるか）の下見（{short}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 1 時間 10〜20 分。終わったら自動で切れます。
- もう 1 本（{'OLMo-2-1B' if key == 'pythia' else 'Pythia-1B'}）のノートブックと、別のランタイムで同時に回せます（保存先のフォルダがモデルごとに別です）。{'' if key == 'pythia' else '**ただし、Pythia-1B のノートブックが Drive の同じフォルダにあることが必要です**（計算のコードをそこから取り出します）。'}
- 探索（下見、予想なし）。場 h（圧力にあたる）を h ＝ 0・＋0.5・＋1.0・＋2.0 の 4 本の腕にして、どの腕も 2.0 → 0.35 を 800 歩で冷やし、0.35 → 2.0 を 800 歩で温めます。記事 40 × サンプル 4。各歩の q（候補の確率の和）とエネルギー U も記録します。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコード、`scripts/a2_explore_field.py`（`{h(CF)[:16]}…`）、`scripts/a2_explore_dsc.py`（`{h(CD)[:16]}…`）、`scripts/a2_explore_dsc2.py`（`{h(CD2)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_explore_dsc3.py`（`{h(CD3)[:16]}…`）は{'このノートブックに埋め込み' if key == 'pythia' else ' Pythia-1B のノートブック（同じフォルダ）から取り出してハッシュで確かめる'}。
- 保存先 `A2_explore/dsc3_{key}/`。腕は 4 本。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。**結果は表示しません。**
- **通知**：🚀・腕ごとの ✅（4 回）・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    want = f"'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}, 'SRCD': {h(CD)!r}, 'SRCD2': {h(CD2)!r}" + ("" if key == 'pythia' else f", 'SRCD3': {h(CD3)!r}")
    core = ("# セル2：計算のコード\n" + EXTRACT + f"_s = take_src({{{want}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF, SRCD = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF'], _s['SRCD']\n"
            "SRCD, SRCD2 = _s['SRCD'], _s['SRCD2']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + (f"SRCD3 = {CD3!r}\nassert hashlib.sha256(SRCD3.encode()).hexdigest() == {h(CD3)!r}\n" if key == 'pythia'
               else "SRCD2, SRCD3 = _s['SRCD2'], _s['SRCD3']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n")
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCD3)\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 4 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_dsc3('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    name, hf_id = D_MODELS[{key!r}]
    run_dsc3(name, hf_id, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(model=[name, hf_id], arms=D3_ARMS, top=D3_TOP, bot=D3_BOT, n=D3_N, samples=D3_S, chunk=D_CHUNK, seed=D3_SEED,
                   dsc2_sha256={h(CD2)!r}, dsc3_sha256={h(CD3)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + f'/run_meta_{{name}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'探索 文章の DSC その 3（{short}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - explore DSC3 high T high h {short}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'文章の DSC その 3（{short}）の材料を A2_explore/dsc3_{key} に保存しました'))],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_explore_dsc3_{key}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(CD3)[:16])
