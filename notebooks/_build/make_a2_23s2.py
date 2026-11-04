"""A2-23 第 2 段（(A) 予言した θ* での生成 21 個、(D) 予言した h* での生成 6 個）のノートブックを、L4 のランタイム 3 つぶん組み立てる。予言の JSON をハッシュつきで埋め込む。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17, C19, C20, C23 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_18_collect', 'a2_19_collect', 'a2_20_collect', 'a2_23_collect'))
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
OUT = BASE + '/A2_23_s2'
PM, PI = rd('results/A2_23/pred_mult.json'), rd('results/A2_23/pred_it.json')
# 封をした予言から、仕事の表に要る値だけを取り出した短い JSON（a23_jobs2 が読む形）。元のファイルのハッシュはゲート §10・§11 に記録
_pm, _pi = json.loads(PM)['models'], json.loads(PI)['models']
PMC = json.dumps({'models': {n: ({'ok': True, 'theta': r['theta'], 'grid': r['grid']} if r['ok'] else {'ok': False}) for n, r in _pm.items()}}, sort_keys=True, separators=(',', ':'))
PIC = json.dumps({'models': {n: ({'ok': True, 'target': {'hstar': r['target']['hstar'], 'grid': r['target']['grid']}} if r['ok'] else {'ok': False}) for n, r in _pi.items()}}, sort_keys=True, separators=(',', ':'))
NW = 3
for w in range(NW):
    label = f'logit-thermo A2-23 第2段（割る型の罰 θ* で 7 本・指示調整版 Qwen で h*、生成 27 個、ランタイム {w + 1}/{NW}、L4）'
    head = f"""# logit-thermo A2-23 第 2 段：予言した罰で生成する——(A) 割る型の罰 θ*（小さい 7 本）と (D) 指示調整版（Qwen2.5-1.5B-Instruct）の h*——ランタイム {w + 1}/{NW}（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。
- **ランタイム：L4**（ハイメモリはいりません）。GPU のメモリが 20GB に足りなければ、⚠ を送ってランタイムを切ります。
- **3 本のノートブック（ランタイム 1/3・2/3・3/3）を、別々のランタイムで同時に回してください。** 仕事は 27 個（(A) 7 本 × 記事の 3 組、(D) 1 本 × 罰なし・罰あり × 3 組）。1 つ 35〜40 分くらいで、3 つ同時なら 6〜7 時間、CU は合わせて 30 くらいの見込みです。
- 保存先 `A2_23_s2/` と `A2_23_s2/locks/` は、Claude が先に Drive に 1 つずつ作ってあります（第 1 段で、同時に始めたランタイムが同じ名前のフォルダを別々に作ってしまったため）。
- 予言（θ* と h*）は、封をした予言から仕事の表に要る値だけを取り出してこのノートブックに埋め込み、ハッシュで確かめます。
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
            + f"SRC23 = take_file('a2_23_collect.py', {h(C23)!r})\n"
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\nexec(SRC19)\nexec(SRC20)\nexec(SRC23)\n"
            "print('コードを置きました')\n")
    run = f"""# セル3：自己テスト → BBC の記事（A2-2・A2-19 と同じか目録で確かめる）→ 仕事を取りに行って回す → 保存（結果も記事も表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a18(), self_test_a19(), self_test_a20(), self_test_a23('cuda'), self_test_chat())
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    check_docs20(meta, json.load(open({BASE!r} + '/A2_2/docs_meta.json')), 'docs')
    GROUPS, META19 = load_docs19({BASE!r} + '/A2_2/docs_meta.json')
    check_docs20(META19, json.load(open({BASE!r} + '/A2_19/docs_meta_w0.json')), 'groups')
    PRED_MULT, PRED_IT = {PMC!r}, {PIC!r}   # 封をした予言（pred_mult.json {h(PM)[:16]}…・pred_it.json {h(PI)[:16]}…）から仕事の表に要る値だけを取り出したもの
    assert hashlib.sha256(PRED_MULT.encode()).hexdigest() == {h(PMC)!r} and hashlib.sha256(PRED_IT.encode()).hexdigest() == {h(PIC)!r}, '封をした予言と違います'
    open('/content/pred_mult.json', 'w', encoding='utf-8').write(PRED_MULT); open('/content/pred_it.json', 'w', encoding='utf-8').write(PRED_IT)
    assert os.path.isdir({OUT!r}) and os.path.isdir({OUT!r} + '/locks'), ('Claude が先に作った保存先のフォルダが見つかりません', {OUT!r})
    JOBS = a23_jobs2('/content/pred_mult.json', '/content/pred_it.json')
    assert len(JOBS) == 27, len(JOBS)
    docs_of = lambda j: GROUPS[j[5]]
    srcdir_of = lambda j: None
    N_DONE, LEFT = run_a23(JOBS, {w}, {NW}, docs_of, srcdir_of, {OUT!r}, 'cuda', 20, notify=lambda s: notify(s, tags='warning') if s.startswith('⚠') else notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(worker={w}, done=N_DONE, left=LEFT, samples=GP_S, chunk=20, a23_sha256={h(C23)!r}, pred_mult_sha256={h(PM)!r}, pred_it_sha256={h(PI)!r}, pred_compact_sha256=[{h(PMC)!r}, {h(PIC)!r}], a20_sha256={h(C20)!r}, a19_sha256={h(C19)!r},
                   torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                   matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_worker{w}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-23 第2段（ランタイム {w + 1}/{NW}）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-23 第2段（ランタイム {w + 1}/{NW}）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-23 stage2 mult penalty and instruct generation worker {w + 1}of{NW}', GPU_CHECK)), code(core), code(run), code(done)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_23s2_w{w + 1}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C23)[:16])
