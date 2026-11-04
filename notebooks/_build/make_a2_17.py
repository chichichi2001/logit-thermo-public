"""A2-17（文章の種類の確かめ）の材料集めのノートブックを、ランタイム 3 つぶん組み立てる（L4、中身は同じで、仕事を取りに行く順だけが違う）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17, C23 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_17_collect', 'a2_3_collect'))
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
OUT = BASE + '/A2_17'
for w in range(3):
    label = f'logit-thermo A2-17 文章の種類の確かめ（論文・コード・小説 × 4 モデル × 場 ±0.3、ランタイム {w + 1}/3、L4）'
    head = f"""# logit-thermo A2-17：文章の種類（論文・コード・小説）を変えて、A2-16 と同じ場の生成（q つき）——ランタイム {w + 1}/3（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。
- **3 本のノートブック（ランタイム 1/3・2/3・3/3）を、別々のランタイムで同時に回してください。** 52 個の仕事（3 種類 × 4 モデル × h ＝ −0.3・0・＋0.3 の 36 個と、4 種類 × 4 モデルの「人の続きを読ませる」短い仕事 16 個）を、3 つのランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行きます。3 つ同時なら 7〜8 時間くらいの見込みです。1 つだけでも、時間はかかりますが全部回せます。
- 文章：arXiv の論文（2025-04・05、40 本）、GitHub のコード（A2-3 と同じ 35 本。A2-3 の目録と一致するか確かめます）、グーテンベルクの小説（本の番号 30000 より後、本文の 30% の位置から、40 本）。種で決めて選びます（3 つのランタイムで同じになることを確かめます）。最初に文章を全部読み込み、失敗したらそこで止まって ⚠ を送ります。
- 作り：A2-16（G-pre）と同じ（温度 0.4〜0.7 の 7 個、サンプル 4、512 歩、q つき）。モデル：Pythia-1.4B・SmolLM2-1.7B・Qwen2.5-1.5B・BLOOM-1.1B。人の書いた続きを読ませたときの −log p も、短い仕事で記録します（覚えているかの記録）。
- **データは、ゲートを凍結するまで開けません。** 結果も文章の中身も表示しません。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコード、`scripts/a2_explore_field.py`・`a2_explore_dsc.py`・`a2_explore_dsc2.py`（`{h(CD2)[:16]}…`）、`scripts/a2_gpre_collect.py`（`{h(CG)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_3_collect.py`（`{h(C23)[:16]}…`、コードの選び方）は A2-3 のノートブックから取り出してハッシュで確かめる。`scripts/a2_17_collect.py`（`{h(C17)[:16]}…`）は、Drive の `A2_explore/a2_17_collect.py` から読んでハッシュで確かめる。
- 保存先 `A2_17/`（種類ごとのフォルダ）。途中で切れても、もう一度「すべて実行」すれば、終わった仕事は飛ばします（取りかけの仕事は 2 時間たつとほかのランタイムが取り直せます）。
- **通知**：🚀・仕事ごとの ✅・🎉（このランタイムで回した数と、まだ残っている仕事）・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    want = f"'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}, 'SRCD': {h(CD)!r}, 'SRCD2': {h(CD2)!r}, 'SRCG': {h(CG)!r}" + f", 'SRC23': {h(C23)!r}"
    core = ("# セル2：計算のコード\n" + EXTRACT + f"_s = take_src({{{want}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF, SRCD = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF'], _s['SRCD']\n"
            "SRCD, SRCD2 = _s['SRCD'], _s['SRCD2']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + "SRCD2, SRCG = _s['SRCD2'], _s['SRCG']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + "SRCG, SRC23 = _s['SRCG'], _s['SRC23']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            + "def take_file(name, want):\n"
              "    \"\"\"Drive の logit-thermo のフォルダから、名前とハッシュが合う .py のファイルを読む。\"\"\"\n"
              "    for p in sorted(glob.glob(" + repr(BASE) + " + '/**/' + name, recursive=True)):\n"
              "        s = open(p, encoding='utf-8').read()\n"
              "        if hashlib.sha256(s.encode()).hexdigest() == want:\n"
              "            return s\n"
              "    raise AssertionError(('Drive に ' + name + ' （ハッシュが合うもの）が見つかりません', want))\n"
            + f"SRC17 = take_file('a2_17_collect.py', {h(C17)!r})\n"
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\n"
            "# A2-3 のコードの選び方は、名前がぶつからないように別の名前空間で読み込み、関数だけを渡す\n"
            "_ns3 = dict(globals()); exec(SRC23, _ns3); A3_LOAD_CODE = _ns3['load_code_docs']\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → BBC の記事（A2-2 と同じか確かめる）→ 仕事を取りに行って回す → 保存（結果も文章も表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a17('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    D = prepare_docs({OUT!r}, docs, {BASE!r} + '/A2_3/code_meta.json', notify=print)   # 文章の読み込み（失敗したら、ここで止まって ⚠）
    N_DONE, LEFT = run_a17({w}, 3, D, {OUT!r}, 'cuda', notify=lambda s: notify(s, tags='warning') if s.startswith('⚠') else notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(worker={w}, done=N_DONE, left=LEFT, models=A17_MODELS, genres=A17_GENRES, h=GP_H, temps=GP_TEMPS, samples=GP_S, chunk=GP_CHUNK, seed=GP_SEED,
                   a17_sha256={h(C17)!r}, a23_sha256={h(C23)!r}, gpre_sha256={h(CG)!r}, dsc2_sha256={h(CD2)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_worker{w}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-17 文章の種類の確かめ（ランタイム {w + 1}/3）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-17 文章の種類の確かめ（ランタイム {w + 1}/3）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-17 genre collect worker {w + 1}of3', GPU_CHECK)), code(core), code(run), code(done)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_17_w{w + 1}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C17)[:16])
