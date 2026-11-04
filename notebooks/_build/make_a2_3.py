"""A2-3 の GPU（データ集め）と CPU（分析）のノートブックを組み立てる。コードはそのまま埋め込む。"""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22, C23 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_3_collect.py')
A22, A23 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:"), rd('scripts/a2_3_analyze.py')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
TOPIC = "YOUR_NTFY_TOPIC"
OUT = "/content/drive/MyDrive/logit-thermo/A2_3"
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}

def notify_cell(label, title, gpu_check):
    return f"""# セル1：通知、GPU、Drive
import os, time, urllib.request
NTFY_TOPIC = '{TOPIC}'
NB_LABEL = {label!r}
def notify(msg, tags='pig'):
    try:
        req = urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\\n[' + NB_LABEL + ']').encode('utf-8'),
                                     headers={{'Title': {title!r}, 'Tags': tags}})
        urllib.request.urlopen(req, timeout=10).read()
        print('通知を送りました：', msg)
    except Exception as e:
        print('通知は送れませんでした（実験には影響しません）：', type(e).__name__)
_NOTIFIED = {{'error': False}}
def _on_error(result):
    if result.error_in_exec is not None and not _NOTIFIED['error']:
        _NOTIFIED['error'] = True
        import traceback
        e = result.error_in_exec
        txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        v = os.environ.get('HF_TOKEN')
        if v: txt = txt.replace(v, '***')
        txt = txt.encode('utf-8')[-3300:].decode('utf-8', 'ignore')
        notify('⚠️ 止まりました（エラー）：\\n' + txt, tags='warning')
if not globals().get('_HOOK_ON'):
    get_ipython().events.register('post_run_cell', _on_error); _HOOK_ON = True
notify('🚀 スタートしました（これが届けば通知の設定は OK です）', tags='rocket')
{gpu_check}
from google.colab import drive
drive.mount('/content/drive')
"""

GPU_CHECK = """os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 38e9:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU なし'
    notify(f'⚠️ GPU のメモリが足りません（{name}）。A100 を選んでください。ランタイムを切ります', tags='warning')
    from google.colab import runtime; runtime.unassign()
    raise SystemExit
import psutil
if psutil.virtual_memory().total < 50e9:
    notify(f'⚠️ システムの RAM が足りません（{psutil.virtual_memory().total/1e9:.0f} GB）。7B を fp32 で読むのに約 31GB 要ります。ランタイムを切ります', tags='warning')
    from google.colab import runtime; runtime.unassign()
    raise SystemExit
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
print('GPU：', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass"""

# ---------- GPU ----------
LABEL_G = "logit-thermo A2-3 データ集め（7B 2つ＋コード 3つ＋A100 の確かめ、A100）"
HEAD_G = f"""# logit-thermo A2-3：T_0 ≈ 0.85 は一般的か — データ集め（GPU、A100）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）
- **ランタイム：A100（メモリ 40GB 以上）**。7B を fp32 で載せるためです。足りない GPU だと ⚠ で止まってランタイムを切ります。見込みは **1〜2 時間**。
- **ゲート `docs/A2_3_GATE.md` v0.2（凍結 2026-10-28 14:15）**。数字を集めて保存するだけで、何も判定しません。結果も表示しません。
- 順番：①自己テスト ②BBC の 40 本（A2-2 と同じことを確かめる）とコードの 40 本を選ぶ（中身は表示しない）③A100 の確かめ（Qwen2.5-1.5B × BBC）→ コード 3 つ → Qwen2.5-7B → OLMo-2-7B。
- 計算のコード：`scripts/a2_2_collect.py`（sha256 `{h(C22)[:16]}…`）と `scripts/a2_3_collect.py`（`{h(C23)[:16]}…`）をそのまま埋め込み。TF32 は切ります。
- 結果の置き場所：`MyDrive/logit-thermo/A2_3/`。途中で切れても「すべてのセルを実行」で、終わった条件は飛ばして続きから回ります。
- **通知**：ntfy の `{TOPIC}` に 🚀・条件ごとの ✅・🎉・⚠️（エラーの文面まるごと）。名札は「{LABEL_G}」。
- **終わったら自動でランタイムを切ります**（止めたいときは最初に `AUTO_DISCONNECT = False`）。
"""
CORE_G = ("# セル2：計算のコード（a2_2_collect.py と a2_3_collect.py をそのまま。同じ名前空間で実行）\nimport hashlib\n"
          f"SRC22 = {C22!r}\nSRC23 = {C23!r}\n"
          f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
          f"assert hashlib.sha256(SRC23.encode()).hexdigest() == {h(C23)!r}\n"
          "exec(SRC22)\nexec(SRC23)\nprint('コードを置きました')\n")
TEST_G = "# セル3：自己テスト（小さなランダムのモデル）\nprint('自己テスト：', self_test('cuda'))\n"
DATA_G = f"""# セル4：文章を選ぶ（中身は表示しない。ここで止まっても、通知してからランタイムを切る）
import json
OUT = {OUT!r}
os.makedirs(OUT, exist_ok=True)
try:
    bbc, bbc_meta = load_docs()
    old = json.load(open('/content/drive/MyDrive/logit-thermo/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in bbc_meta['docs']], 'BBC の記事が A2-2 と違います'
    code_docs, code_meta = load_code_docs()
    mp = OUT + '/code_meta.json'
    if os.path.exists(mp):
        assert [d['text_sha256'] for d in json.load(open(mp))['docs']] == [d['text_sha256'] for d in code_meta['docs']], '前の回とコードが違います'
    else:
        json.dump(code_meta, open(mp, 'w'), indent=1)
except Exception as e:
    import traceback
    _NOTIFIED['error'] = True
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__)).encode('utf-8')[-3300:].decode('utf-8', 'ignore')
    notify('⚠️ 文章を選ぶところで止まりました。ランタイムを切ります：\\n' + txt, tags='warning')
    if globals().get('AUTO_DISCONNECT', True):
        from google.colab import drive as _d, runtime as _r
        _d.flush_and_unmount(); time.sleep(3); _r.unassign()
    raise
print('BBC：A2-2 と同じ 40 本。コード：候補', code_meta['pool_unique'], '本から', code_meta['n_docs'], '本（月', code_meta['months'], '、1 リポジトリ', code_meta['per_repo'], 'ファイルまで）')
"""
MAIN_G = f"""# セル5：条件を順に回す（終わった条件は飛ばす）
import transformers, platform
docs_of = {{'bbc': bbc, 'code': code_docs}}
t_all = time.time()
try:
    for name, hf_id, text, chunk in CONDITIONS:
        run_condition(name, hf_id, docs_of[text], OUT, 'cuda', chunk, notify=lambda m: notify('✅ ' + m, tags='white_check_mark'))
except Exception as e:
    # A100 は高いので、ここでのエラーは通知してからランタイムを切る（終わった条件は保存済みで、続きから回せる）
    import traceback
    _NOTIFIED['error'] = True
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__)).encode('utf-8')[-3300:].decode('utf-8', 'ignore')
    notify('⚠️ 止まりました。ランタイムを切ります：\\n' + txt, tags='warning')
    if globals().get('AUTO_DISCONNECT', True):
        from google.colab import drive as _d, runtime as _r
        _d.flush_and_unmount(); time.sleep(3); _r.unassign()
    raise
try:
    tf32_rec = torch.get_float32_matmul_precision()
except Exception as e:
    tf32_rec = 'unknown: ' + type(e).__name__
json.dump({{'gate': 'docs/A2_3_GATE.md v0.2', 'a2_2_collect_sha256': {h(C22)!r}, 'a2_3_collect_sha256': {h(C23)!r},
           'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(),
           'gpu': torch.cuda.get_device_name(0), 'float32_matmul_precision': tf32_rec,
           'seconds_this_session': round(time.time() - t_all, 1)}}, open(OUT + '/run_meta.json', 'w'), indent=1)
print('全部おわり')
"""
DONE = """# 最後：終わりの通知と、ランタイムを切る
notify('🎉 終わりました：{what}', tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
nbg = {"cells": [md(HEAD_G), code(notify_cell(LABEL_G, 'logit-thermo - A2-3 collect', GPU_CHECK)), code(CORE_G), code(TEST_G),
                 code(DATA_G), code(MAIN_G), code(DONE.replace('{what}', '6 つの条件を Drive に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}}, "nbformat": 4, "nbformat_minor": 0}
(ROOT / 'notebooks' / 'A2_3_collect_A100.ipynb').write_text(json.dumps(nbg, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

# ---------- CPU ----------
LABEL_C = "logit-thermo A2-3 分析（判定と要約、CPU）"
HEAD_C = f"""# logit-thermo A2-3：分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 2〜5 分。
- `MyDrive/logit-thermo/A2_3/` の 6 つの条件と、A2-2 の `qwen2.5-1.5b.npz`（A100 の確かめ用）を読み、ゲート v0.2 の判定（本命・参考・軸ごとの読み方）を計算して `A2_3/a2_3_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_analyze.py`（`{h(rd('scripts/a2_2_analyze.py'))[:16]}…`）と `scripts/a2_3_analyze.py`（`{h(A23)[:16]}…`）をそのまま埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_C}」。終わったらランタイムを切ります。
"""
CORE_C = ("# セル2：分析のコード\nimport hashlib\n" f"SRCA22 = {A22!r}\nSRCA23 = {A23!r}\n"
          f"assert hashlib.sha256(SRCA22.encode()).hexdigest() == {h(A22)!r}\n"
          f"assert hashlib.sha256(SRCA23.encode()).hexdigest() == {h(A23)!r}\n"
          "exec(SRCA22)\nexec(SRCA23)\nprint('コードを置きました')\n")
RUN_C = f"""# セル3：分析して保存（結果は表示しない）
import json
r = analyze3({OUT!r}, '/content/drive/MyDrive/logit-thermo/A2_2/qwen2.5-1.5b.npz')
r['a2_3_analyze_sha256'] = {h(A23)!r}
r['a2_2_analyze_sha256'] = {h(rd('scripts/a2_2_analyze.py'))!r}
json.dump(_clean(r), open({OUT!r} + '/a2_3_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('保存しました')
"""
nbc = {"cells": [md(HEAD_C), code(notify_cell(LABEL_C, 'logit-thermo - A2-3 analyze', '')), code(CORE_C), code(RUN_C),
                 code(DONE.replace('{what}', '判定と要約を a2_3_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
(ROOT / 'notebooks' / 'A2_3_analyze_CPU.ipynb').write_text(json.dumps(nbc, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print('collect', h(C23), 'analyze', h(A23))
