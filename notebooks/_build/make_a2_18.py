"""A2-18（罰の強さの予言の確かめ）の材料集めのノートブックを、ランタイム 3 つぶん組み立てる（L4、中身は同じで、仕事を取りに行く順だけが違う）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_18_collect'))
src5, src3, src4 = rd('notebooks/_build/make_a2_5.py'), rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
notify_cell, FAIL, EXTRACT, GPU_CHECK = ns['notify_cell'], ns['FAIL'], ns['EXTRACT'], ns['GPU_CHECK']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_18'
for w in range(3):
    label = f'logit-thermo A2-18 罰の強さの予言の確かめ（7 モデル × 対照・h*・2h*、ランタイム {w + 1}/3、L4）'
    head = f"""# logit-thermo A2-18：h ＝ 0 のデータだけから予言した罰の強さ h* で生成して、融点が予言どおり下がるか——ランタイム {w + 1}/3（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。
- **3 本のノートブック（ランタイム 1/3・2/3・3/3）を、別々のランタイムで同時に回してください。** 21 個の仕事（7 モデル × 3 つの腕：罰なしの対照・予言した罰 h*・その 2 倍）を、3 つのランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行きます。3 つ同時なら 4 時間くらいの見込みです。1 つだけでも全部回せます。
- 作り：G-pre と同じ（BBC の記事 40 × サンプル 4、512 歩、q つき）。罰の強さ h* と温度（予言した融点のまわりの 7 つ。3 つの腕で同じ）は、生成の前に固定した予言（`results/A2_18/hstar.json`）。モデル：Pythia-410M・Pythia-1.4B・GPT-Neo-1.3B・SmolLM2-1.7B・Qwen2.5-1.5B・GPT-2-XL・BLOOM-1.1B。
- **データは、ゲートを凍結するまで開けません。** 結果は表示しません。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコード、`scripts/a2_explore_field.py`・`a2_explore_dsc.py`・`a2_explore_dsc2.py`（`{h(CD2)[:16]}…`）、`scripts/a2_gpre_collect.py`（`{h(CG)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_18_collect.py`（`{h(C17)[:16]}…`）は、Drive の `A2_explore/a2_18_collect.py` から読んでハッシュで確かめる。
- 保存先 `A2_18/`。途中で切れても、もう一度「すべて実行」すれば、終わった仕事は飛ばします（取りかけの仕事は 2 時間たつとほかのランタイムが取り直せます）。
- **通知**：🚀・仕事ごとの ✅・🎉（このランタイムで回した数と、まだ残っている仕事）・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    want = f"'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}, 'SRCD': {h(CD)!r}, 'SRCD2': {h(CD2)!r}, 'SRCG': {h(CG)!r}" + ""
    core = ("# セル2：計算のコード\n" + EXTRACT + f"_s = take_src({{{want}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF, SRCD = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF'], _s['SRCD']\n"
            "SRCD, SRCD2 = _s['SRCD'], _s['SRCD2']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + "SRCD2, SRCG = _s['SRCD2'], _s['SRCG']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + "def take_file(name, want):\n"
              "    \"\"\"Drive の logit-thermo のフォルダから、名前とハッシュが合う .py のファイルを読む。\"\"\"\n"
              "    for p in sorted(glob.glob(" + repr(BASE) + " + '/**/' + name, recursive=True)):\n"
              "        s = open(p, encoding='utf-8').read()\n"
              "        if hashlib.sha256(s.encode()).hexdigest() == want:\n"
              "            return s\n"
              "    raise AssertionError(('Drive に ' + name + ' （ハッシュが合うもの）が見つかりません', want))\n"
            + f"SRC17 = take_file('a2_18_collect.py', {h(C17)!r})\n"
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\n"
            "print('コードを置きました')\n")
    run = f"""# セル3：自己テスト → BBC の記事（A2-2 と同じか確かめる）→ 仕事を取りに行って回す → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a18())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    N_DONE, LEFT = run_a18({w}, 3, docs, {OUT!r}, 'cuda', notify=lambda s: notify(s, tags='warning') if s.startswith('⚠') else notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(worker={w}, done=N_DONE, left=LEFT, jobs=A18_JOBS, samples=GP_S, chunk=GP_CHUNK, seed=GP_SEED,
                   a18_sha256={h(C17)!r}, gpre_sha256={h(CG)!r}, dsc2_sha256={h(CD2)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_worker{w}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-18 罰の強さの予言の確かめ（ランタイム {w + 1}/3）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-18 罰の強さの予言の確かめ（ランタイム {w + 1}/3）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-18 penalty prediction worker {w + 1}of3', GPU_CHECK)), code(core), code(run), code(done)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_18_w{w + 1}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C17)[:16])
