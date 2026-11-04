"""A2-20 第 2 段（予言した罰 h* と対照、A2-19 の記事 120 本、仕事 12 個）のノートブックを、A100 のランタイム 3 つぶん組み立てる（仕事を取りに行く順だけが違う）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD, CD2, CG, C17, C19, C20 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_dsc', 'a2_explore_dsc2', 'a2_gpre_collect', 'a2_18_collect', 'a2_19_collect', 'a2_20_collect'))
src5, src3, src4 = rd('notebooks/_build/make_a2_5.py'), rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
notify_cell, FAIL, EXTRACT = ns['notify_cell'], ns['FAIL'], ns['EXTRACT']
GPU_CHECK = """os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch, psutil
_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU なし'
_gmem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
_ram = psutil.virtual_memory().total
if 'A100' not in _gpu or _gmem < 39e9 or _ram < 48e9:
    notify(f'⚠️ ランタイムが合いません（GPU {_gpu}、GPU のメモリ {_gmem / 1e9:.0f}GB、CPU のメモリ {_ram / 1e9:.0f}GB）。A100（40GB 以上）を選び、CPU のメモリが 48GB 以上あることを確かめてください。ランタイムを切ります', tags='warning')
    from google.colab import runtime; runtime.unassign()
    raise SystemExit
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
def _cut_on_error(result):
    # どのセルでエラーになっても（A2-19 ではセル 2 で止まったときに切れなかった）、⚠ を送ったあとでランタイムを切る。手で止めたときも切れます
    if result.error_in_exec is not None and globals().get('AUTO_DISCONNECT', True):
        time.sleep(10)
        try:
            from google.colab import drive as _d; _d.flush_and_unmount()
        except Exception:
            pass
        from google.colab import runtime as _r; _r.unassign()
if not globals().get('_CUT_ON'):
    get_ipython().events.register('post_run_cell', _cut_on_error); _CUT_ON = True
"""
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_20_s2'
NW = 3
HJ = rd('results/A2_20/hstar7b.json')
for w in range(NW):
    label = f'logit-thermo A2-20 第2段 7B に予言した罰（Qwen2.5-7B・Mistral-7B、対照・h*、記事 120 本、ランタイム {w + 1}/{NW}、A100）'
    head = f"""# logit-thermo A2-20 第 2 段：7B 級 2 本（Qwen2.5-7B・Mistral-7B-v0.3）に、予言した罰 h* をかける——ランタイム {w + 1}/{NW}（GPU、A100）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。
- **ランタイム：A100（40GB 以上）**。ハイメモリにしなくて大丈夫です（CPU のメモリが 48GB 以上あれば動きます。足りなければ ⚠ を送ってランタイムを切ります）。A100 でなければ、⚠ を送ってランタイムを切ります。
- **3 本のノートブック（ランタイム 1/3・2/3・3/3）を、別々のランタイムで同時に回してください。** 仕事は 12 個（モデル 2 本 × 腕 2 つ（罰なしの対照・予言した罰 h*）× 記事の 3 つの組）で、3 つのランタイムが「まだ終わっていない・誰も取っていない」ものから取りに行きます。1 つの仕事は 25〜45 分くらい、3 つ同時なら 2〜3 時間の見込みです（CU は 3 つ合わせて 30〜40 くらい）。
- 作り：A2-19 の新しい BBC の記事 120 本（目録で照らし合わせます）、サンプル 4、512 歩、q つき。h* と温度は第 1 段から予言して封をしたもの（`results/A2_20/hstar7b.json`、ハッシュで確かめます）：Qwen2.5-7B は h* ＝ −0.675・温度 0.30〜0.60、Mistral-7B は h* ＝ −0.345・温度 0.19〜0.49。モデルは fp32、TF32 なし。
- **データは、第 2 段が終わるまで開けません。** 結果も記事の中身も表示しません。
- 計算のコード：これまでのデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_18_collect.py`（`{h(C17)[:16]}…`）・`a2_19_collect.py`（`{h(C19)[:16]}…`）・`a2_20_collect.py`（`{h(C20)[:16]}…`）は、Drive の `A2_explore/` の .py から読んでハッシュで確かめる。
- 保存先 `A2_20_s2/`。途中で切れても、もう一度「すべて実行」すれば、終わった仕事は飛ばします。
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
            + "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCD)\nexec(SRCD2)\nexec(SRCG)\nexec(SRC17)\nexec(SRC19)\nexec(SRC20)\n"
            "print('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2・A2-19 と同じか目録で確かめる）→ 封をした予言（ハッシュで確かめる）→ 仕事を取りに行って回す → 保存（結果も記事も表示しない）
import json, sys, transformers
HSTAR_JSON = {HJ!r}
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_dsc('cuda'), self_test_dsc2('cuda'), self_test_gpre(), self_test_a18(), self_test_a19(), self_test_a20())
    assert hashlib.sha256(HSTAR_JSON.encode()).hexdigest() == {h(HJ)!r}, '封をした予言と違います'
    open('/content/hstar7b.json', 'w', encoding='utf-8').write(HSTAR_JSON)
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    check_docs20(meta, json.load(open({BASE!r} + '/A2_2/docs_meta.json')), 'docs')
    GROUPS, META19 = load_docs19({BASE!r} + '/A2_2/docs_meta.json')
    check_docs20(META19, json.load(open({BASE!r} + '/A2_19/docs_meta_w0.json')), 'groups')
    JOBS = a20_jobs2(['/content/hstar7b.json'])
    assert len(JOBS) == 12 and sorted({{j[0] for j in JOBS}}) == ['mistral-7b-v0.3', 'qwen2.5-7b'], JOBS
    N_DONE, LEFT = run_a20(JOBS, {w}, {NW}, lambda j: GROUPS[j[4]], {OUT!r}, 'cuda', 20, notify=lambda s: notify(s, tags='warning') if s.startswith('⚠') else notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(worker={w}, done=N_DONE, left=LEFT, samples=GP_S, chunk=20, hstar_sha256={h(HJ)!r},
                   a20_sha256={h(C20)!r}, a19_sha256={h(C19)!r}, a18_sha256={h(C17)!r}, dsc2_sha256={h(CD2)!r}, torch=torch.__version__, transformers=transformers.__version__,
                   python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta_worker{w}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-20 第2段 7B に予言した罰（ランタイム {w + 1}/{NW}）')}"""
    done = f"""# 最後：終わりの通知と、ランタイムを切る
notify(f'🎉 終わりました：A2-20 第2段 7B に予言した罰（ランタイム {w + 1}/{NW}）。このランタイムで {{N_DONE}} 個。まだ残っている仕事：{{len(LEFT)}} 個' + (f'（{{", ".join(LEFT)}}）' if LEFT else '（全部そろいました）'), tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-20 stage2 7B predicted penalty worker {w + 1}of{NW}', GPU_CHECK)), code(core), code(run), code(done)],
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_20s2_w{w + 1}_A100.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size, h(C20)[:16])
