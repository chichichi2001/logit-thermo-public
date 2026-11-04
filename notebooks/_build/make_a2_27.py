"""A2-27（A2-19 の罰なしの腕 c0 の文を、小さい 7 本でもう一度通して採点し直す。仕事 21 個）のノートブックを、L4 のランタイム 1 つぶん組み立てる。
採点し直しのコード（scripts/a2_27_collect.py）は Drive の A2_27/ に置き、ハッシュで確かめて読む。
2 つ目のセルで、ノートブック自身（Drive の A2_27_L4.ipynb）の各セルの中身がハッシュと合うかを確かめる（Drive に上げるときの写しまちがいを、回す前に見つけるため）。同じノートブックを 2 つ目のランタイムで開いても、仕事は重ならない。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17, C19, C20 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_18_collect', 'a2_19_collect', 'a2_20_collect'))
src5, src3, src4, src20 = rd('notebooks/_build/make_a2_5.py'), rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py'), rd('notebooks/_build/make_a2_20s1.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
notify_cell, FAIL, EXTRACT = ns['notify_cell'], ns['FAIL'], ns['EXTRACT']
CUT = re.search(r"(def _cut_on_error\(result\):.*?_CUT_ON = True\n)", src20, re.S).group(1)
GPU_CHECK = ns['GPU_CHECK'] + CUT
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_27'
SRCDIR = BASE + '/A2_19'
C27 = rd('scripts/a2_27_collect.py')
NW = 1
for w in range(NW):
    label = 'logit-thermo A2-27（A2-19 の罰なしの文を小さい 7 本で採点し直す、仕事 21 個、L4）'
    head = f"""# logit-thermo A2-27：A2-19 の罰なしの腕の文を、同じモデルにもう一度通して採点し直す（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。
- **ランタイム：L4**（ハイメモリはいりません）。GPU のメモリが 20GB に足りなければ、⚠ を送ってランタイムを切ります。
- **生成はしません。** A2-19 の罰なしの文（Drive の `A2_19/`）をモデルに 1 回通して、歩ごとの量（エネルギーの見込みなど）を保存するだけです。仕事は 21 個（小さい 7 本 × 記事の 3 組）。1 つ 3〜5 分くらいで、1〜1.5 時間、CU 2〜3 の見込みです。
- 早く終わらせたいときは、このノートブックを 2 つ目のランタイムでも「すべて実行」してください。仕事は取りに行く方式なので重なりません。
- 保存先 `A2_27/` と `A2_27/locks/` は、Claude が先に Drive に作ってあります。
- **最初に、出し直した U と q0 が、生成のときに保存したものと合うかを仕事ごとに確かめます。** 合わなければ ⚠ を送って止まります。
- **データは、分析のコードを回すまで開けません。** 結果も記事の中身も表示しません。途中で切れても、もう一度「すべて実行」すれば、終わった仕事は飛ばします。
- **通知**：🚀・仕事ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはどのセルでもエラーで止まったら、ランタイムを切ります（手で止めたときも切れます）。
"""
    want = f"'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}, 'SRCD': {h(CD)!r}, 'SRCD2': {h(CD2)!r}, 'SRCG': {h(CG)!r}"
    core = ("# セル2：計算のコード\n" + EXTRACT + f"_s = take_src({{{want}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC, SRCF, SRCD = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF'], _s['SRCD']\n"
            "SRCD, SRCD2 = _s['SRCD'], _s['SRCD2']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            "SRCD2, SRCG = _s['SRCD2'], _s['SRCG']   # 名前を 2 つ並べて置く（take_src が読み飛ばす形）\n"
            "def take_file(name, want):\n"
            "    \"\"\"Drive の logit-thermo のフォルダから、名前とハッシュが合う .py のファイルを読む。\"\"\"\n"
            "    for p in sorted(glob.glob(" + repr(BASE) + " + '/**/' + name, recursive=True)):\n"
            "        s = open(p, encoding='utf-8').read()\n"
            "        if hashlib.sha256(s.encode()).hexdigest() == want:\n"
            "            return s\n"
            "    raise AssertionError(('Drive に ' + name + ' （ハッシュが合うもの）が見つかりません', want))\n"
            + f"SRC17 = take_file('a2_18_collect.py', {h(C17)!r})\n"
            + f"SRC19 = take_file('a2_19_collect.py', {h(C19)!r})\n"
            + f"SRC20 = take_file('a2_20_collect.py', {h(C20)!r})\n"
            + f"SRC27 = take_file('a2_27_collect.py', {h(C27)!r})\n"
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\nexec(SRC19)\nexec(SRC20)\nexec(SRC27)\n"
            "print('コードを置きました')\n")
    run = f"""# セル3：自己テスト → BBC の記事（A2-2・A2-19 と同じか目録で確かめる）→ 仕事を取りに行って採点し直す → 保存（結果も記事も表示しない）
import json, sys, transformers
try:
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    print('自己テスト：', self_test_rs27('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a18(), self_test_a19(), self_test_a20())
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    GROUPS, META19 = load_docs19({BASE!r} + '/A2_2/docs_meta.json')
    check_docs20(META19, json.load(open({BASE!r} + '/A2_19/docs_meta_w0.json')), 'groups')
    assert os.path.isdir({OUT!r}) and os.path.isdir({OUT!r} + '/locks'), ('Claude が先に作った保存先のフォルダが見つかりません', {OUT!r})
    NAMES = list(A18_HF)
    assert len(NAMES) == 7, NAMES
    N_DONE, LEFT = rs27_run(NAMES, A18_HF, {w}, {NW}, GROUPS, {SRCDIR!r}, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(done=N_DONE, left=LEFT, batch=RS27_B, a27_sha256={h(C27)!r}, a19_sha256={h(C19)!r},
                   torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                   matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_' + A19_TOKEN[:8] + '.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-27（採点し直し）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-27（採点し直し）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    cells = [md(head), code(notify_cell(label, 'logit-thermo - A2-27 rescoring of A2-19 c0 arm', GPU_CHECK)), code(core), code(run), code(done)]
    srcs = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
    want_cells = {i: h(srcs(c)) for i, c in enumerate(cells)}
    verify = f"""# セル1b：このノートブックの中身が、ゲート（docs/A2_27_GATE.md）に書いたハッシュと合うかを確かめる（Drive に上げるときの写しまちがいを見つけるため）
import glob, hashlib, json
_hits = sorted(glob.glob({BASE!r} + '/**/A2_27_L4.ipynb', recursive=True))
assert len(_hits) == 1, ('Drive に A2_27_L4.ipynb が 1 つだけ見つかりません', _hits)
_cells = json.load(open(_hits[0]))['cells']
_src = lambda c: c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
_want = {want_cells!r}
_got = {{i: hashlib.sha256(_src(_cells[i if i < 2 else i + 1]).encode()).hexdigest() for i in _want}}
_bad = [i for i in _want if _got[i] != _want[i]]
assert not _bad, ('ノートブックの中身がゲートのハッシュと違います（Drive に上げるときの写しまちがい）。回さずに Claude に知らせてください', _bad)
print('ノートブックの中身はゲートのハッシュと一致しました')
"""
    cells.insert(2, code(verify))
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / 'A2_27_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C27)[:16])
