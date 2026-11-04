"""A2-19（A2-18 の XQ1 を新しい記事 120 本でもう一度だけ）の材料集めのノートブックを、ランタイム 3 つぶん組み立てる（L4、仕事を取りに行く順だけが違う）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17, C19 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_18_collect', 'a2_19_collect'))
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
OUT = BASE + '/A2_19'
for w in range(3):
    label = f'logit-thermo A2-19 罰の強さの予言・もう一度だけ（新しい記事 120 本 × 7 モデル × 対照・h*、ランタイム {w + 1}/3、L4）'
    head = f"""# logit-thermo A2-19：A2-18 の XQ1（予言した罰 h*）を、新しい BBC の記事 120 本でもう一度だけ確かめる——ランタイム {w + 1}/3（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。
- **3 本のノートブック（ランタイム 1/3・2/3・3/3）を、別々のランタイムで同時に回してください。** 42 個の仕事（7 モデル × 2 つの腕（罰なしの対照・予言した罰 h*）× 記事の 3 つの組）を、3 つのランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行きます。3 つ同時なら 7 時間くらいの見込みです（CU は 35 くらい）。4 本目以降のランタイムで同じノートブックを開いても、仕事を取った印はランタイムごとに別なので、ふつうは重なりません（ほぼ同時に始めると、まれに同じ仕事を 2 回回すことがあります。中身は同じになるので、結果は変わりません）。
- 作り：A2-18 と同じ（罰の強さ h* と温度は A2-18 のまま、サンプル 4、512 歩、q つき）。記事だけが新しい：A2-2 と同じ月（2025-04〜06）の BBC の記事から、これまでの 40 本を除いて 120 本を種で選びます。
- **データは、ゲートを凍結するまで開けません。** 結果も記事の中身も表示しません。
- 計算のコード：これまでのデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_18_collect.py`（`{h(C17)[:16]}…`）と `scripts/a2_19_collect.py`（`{h(C19)[:16]}…`）は、Drive の `A2_explore/` の .py から読んでハッシュで確かめる。
- 保存先 `A2_19/`。途中で切れても、もう一度「すべて実行」すれば、終わった仕事は飛ばします。
- **通知**：🚀・仕事ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
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
            + f"SRC19 = take_file('a2_19_collect.py', {h(C19)!r})\n"
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\nexec(SRC19)\n"
            "print('コードを置きました')\n")
    run = f"""# セル3：自己テスト → BBC の記事（A2-2 と同じか確かめる）→ 新しい 120 本 → 仕事を取りに行って回す → 保存（結果も記事も表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a18(), self_test_a19())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    GROUPS, META19 = load_docs19({BASE!r} + '/A2_2/docs_meta.json')
    import glob as _glob
    for mp in sorted(_glob.glob({OUT!r} + '/docs_meta_w*.json')):
        prev = json.load(open(mp))
        assert [[d['text_sha256'] for d in g] for g in prev['groups']] == [[d['text_sha256'] for d in g] for g in META19['groups']], ('新しい記事がほかのランタイムと違います', mp)
    json.dump(META19, open({OUT!r} + '/docs_meta_w{w}.json', 'w'), ensure_ascii=False, indent=1)
    N_DONE, LEFT = run_a19({w}, 3, GROUPS, {OUT!r}, 'cuda', notify=lambda s: notify(s, tags='warning') if s.startswith('⚠') else notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(worker={w}, done=N_DONE, left=LEFT, samples=GP_S, chunk=GP_CHUNK,
                   a19_sha256={h(C19)!r}, a18_sha256={h(C17)!r}, gpre_sha256={h(CG)!r}, dsc2_sha256={h(CD2)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_worker{w}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-19 罰の強さの予言・もう一度だけ（ランタイム {w + 1}/3）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-19 罰の強さの予言・もう一度だけ（ランタイム {w + 1}/3）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-19 penalty replication worker {w + 1}of3', GPU_CHECK)), code(core), code(run), code(done)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_19_w{w + 1}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C19)[:16])
