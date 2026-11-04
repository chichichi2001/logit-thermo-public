"""A1（ゲート v0.3）のノートブックを組み立てる。計算は scripts/a1_core.py をそのまま埋め込む。

    python3 notebooks/_build/make_a1.py PQW7   ->  notebooks/A1_explore_PQW7_CPU.ipynb
    python3 notebooks/_build/make_a1.py PQC7   ->  notebooks/A1_confirm_PQC7_CPU.ipynb
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = sys.argv[1]
ROLE = {'PQW7': ('探索', 'explore'), 'PQC7': ('確認', 'confirm')}[MODEL]
CORE = (ROOT / 'scripts' / 'a1_core.py').read_text()
CORE_SHA = hashlib.sha256(CORE.encode()).hexdigest()
NTFY_TOPIC = "YOUR_NTFY_TOPIC"
LABEL = f"logit-thermo A1 {ROLE[0]}（{MODEL}）裾の型と壁の圧力"
TITLE = f"logit-thermo - A1 {ROLE[1]} {MODEL}"
TITLE.encode("ascii")
OUTDIR = f"/content/drive/MyDrive/logit-thermo/A1/{MODEL}"

HEAD = f"""# logit-thermo A1 {ROLE[0]}（{MODEL}）：裾の型と、壁の圧力

- **必要な secret：なし**（Drive をつなぐ許可だけ聞かれます）
- **ランタイム：CPU（標準）でOK**。GPU・ハイメモリはいりません。見込みは 10〜20 分くらい。
- 何をするか：ゲート `docs/A1_WALL_GATE.md` v0.3 の手順 {'2' if MODEL == 'PQW7' else '3'}。`PROJECT_A/_LOGITS/PARENT/{MODEL}` の logit（1,536 行）を sha256 で確かめてから、行ごとに裾の傾き・目印 g と c・壁の圧縮因子・比熱のピークを計算し、J1・J2（裾の型）と R1〜R5（記録）をまとめます。
- 計算のコードは logit-thermo の `scripts/a1_core.py`（sha256 `{CORE_SHA[:16]}…`）をそのまま埋め込んであります。
- 結果の置き場所：`MyDrive/logit-thermo/A1/{MODEL}/`（まとめ `a1_summary.json`、行ごとの表 `a1_rows.csv`）。
- **通知**：ntfy の `{NTFY_TOPIC}` に、🚀 始まったとき・🎉 終わったとき・⚠️ エラーで止まったとき（エラーの文面まるごと）に届きます。名札は「{LABEL}」。ランタイムごと切れたときは届かないので、そのときは Colab を見てください。
- **終わったら自動でランタイムを切ります**（Drive への保存を確定してから）。エラーで止まったときは切らずに残します。切りたくないときは、最初に `AUTO_DISCONNECT = False` を実行してください。
- 使い方：「ランタイム」→「すべてのセルを実行」。
"""

SETUP = f"""# セル1：通知のしくみ（ntfy）と、Drive をつなぐ
import urllib.request, time
NTFY_TOPIC = '{NTFY_TOPIC}'
NB_LABEL = {LABEL!r}   # どのノートブックからの通知かの名札
def notify(msg, tags='pig'):
    try:
        req = urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\\n[' + NB_LABEL + ']').encode('utf-8'),
                                     headers={{'Title': {TITLE!r}, 'Tags': tags}})
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
        notify('⚠️ 止まりました（エラー）：\\n' + txt[-3500:], tags='warning')
if not globals().get('_HOOK_ON'):
    get_ipython().events.register('post_run_cell', _on_error); _HOOK_ON = True
notify('🚀 スタートしました（これが届けば通知の設定は OK です）', tags='rocket')

from google.colab import drive
drive.mount('/content/drive')
"""

CORE_CELL = "# セル2：計算のコード（logit-thermo の scripts/a1_core.py をそのまま埋め込み）\n" + CORE

MAIN = f'''# セル3：sha256 を確かめてから、1,536 行を計算してまとめる
import hashlib, io, json, os, platform
MODEL = {MODEL!r}
BASE = '/content/drive/MyDrive/PROJECT_A/_LOGITS/PARENT'
OUT = {OUTDIR!r}
os.makedirs(OUT, exist_ok=True)
d = f'{{BASE}}/{{MODEL}}/logits'
meta = json.load(open(f'{{d}}/metadata.json'))
tok_len = int(meta['tokenizer_len'])
assert tok_len == 151665, tok_len
theta_full, theta_tail = thresholds(tok_len)
assert abs(theta_full - 0.133450) < 1e-6 and abs(theta_tail - 0.432732) < 1e-6, (theta_full, theta_tail)
items = sorted(meta['precisions']['fp32']['files'].items(), key=lambda kv: kv[1]['file'])
rows, prompts, ids = [], [], []
t0 = time.time()
for pi, (sid, info) in enumerate(items):
    raw = open(f"{{d}}/{{info['file']}}", 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == info['sha256'], f"sha256 が合いません：{{info['file']}}"
    a = np.load(io.BytesIO(raw)).astype(np.float64)
    assert a.shape == (info['rows'], info['vocab']), a.shape
    for r in range(a.shape[0]):
        rows.append(row_metrics(a[r, :tok_len], theta_full, theta_tail)); prompts.append(pi); ids.append((info['file'], r))
    del a, raw
    print(f"{{pi + 1}}/{{len(items)}} ファイルおわり（{{time.time() - t0:.0f}} 秒）")
summary = summarize(rows, np.array(prompts))
summary.update({{'model': MODEL, 'model_id': meta.get('model_id'), 'gate': 'docs/A1_WALL_GATE.md v0.3',
                'a1_core_sha256': {CORE_SHA!r}, 'files_sha256_ok': len(items), 'numpy': np.__version__,
                'python': platform.python_version(), 'elapsed_seconds': round(time.time() - t0, 1)}})
keys = ['a_win', 'r2_win', 'a_loc', 's', 'c', 'sigma_rob', 'g', 'h', 't_on_full', 'edge_full', 't_on_tail', 'edge_tail', 't_c', 'edge_c']
with open(f'{{OUT}}/a1_rows.csv', 'w') as f:
    f.write('file,row,prompt,' + ','.join(keys) + '\\n')
    for (fn, r), p, m in zip(ids, prompts, rows):
        vals = [('' if m[k] is None else (f'{{m[k]:.6f}}' if isinstance(m[k], float) else str(m[k]))) for k in keys]
        f.write(f'{{fn}},{{r}},{{p}},' + ','.join(vals) + '\\n')
summary['rows_csv_sha256'] = hashlib.sha256(open(f'{{OUT}}/a1_rows.csv', 'rb').read()).hexdigest()
json.dump(summary, open(f'{{OUT}}/a1_summary.json', 'w'), ensure_ascii=False, indent=1)
print('まとめの大きさ：', os.path.getsize(f'{{OUT}}/a1_summary.json'), 'バイト')
print(json.dumps(summary, ensure_ascii=False, indent=1))
'''

DONE = """# セル4：終わりの通知と、ランタイムを切る
notify('🎉 終わりました：' + str(summary['rows']) + ' 行', tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    print('Drive への書き込みを確定してから、ランタイムを切ります（CU の節約。止めたいときは AUTO_DISCONNECT = False）')
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount()
    time.sleep(3)
    _runtime.unassign()
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


nb = {"cells": [md(HEAD), code(SETUP), code(CORE_CELL), code(MAIN), code(DONE)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
name = f"A1_{ROLE[1]}_{MODEL}_CPU.ipynb"
out = ROOT / "notebooks" / name
out.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))
print("wrote", out, "core sha256", CORE_SHA)
