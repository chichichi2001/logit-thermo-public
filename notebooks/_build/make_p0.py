"""A1 予備計測 P0 のノートブックを組み立てる（σ だけ。順位ごとの値・傾き・圧縮因子は計算しない）。

    python3 notebooks/_build/make_p0.py   ->  notebooks/A1_P0_sigma_CPU.ipynb

通知のしくみは PROJECT_A の notebooks/_build/notify.py（SETUP・setup・done_and_disconnect）をもとにした。
変えたところ：題名を「logit-thermo - ...」に、本文の先頭に絵文字（🚀・⚠️・🎉）を付けた（オーナーの希望、2026-10-15）。
"""
import json
from pathlib import Path

NTFY_TOPIC = "YOUR_NTFY_TOPIC"
LABEL = "logit-thermo A1 予備計測P0（σだけ、PQW7・PQC7）"
TITLE = "logit-thermo - A1 P0 sigma"
TITLE.encode("ascii")

HEAD = f"""# logit-thermo A1 予備計測 P0（σ だけ）

- **必要な secret：なし**（GitHub も Hugging Face も使いません。Drive をつなぐ許可だけ聞かれます）
- **ランタイム：CPU（標準）でOK**。GPU・ハイメモリはいりません。見込みは 5〜15 分くらい（ほとんどが Drive からの読み込み）。
- 何をするか：PROJECT_A の親子の実験で取った logit（`PROJECT_A/_LOGITS/PARENT/` の `PQW7`・`PQC7`、各 1,536 行 × 語彙 152,064）を sha256 で確かめてから、行ごとに **σ_rob・ふつうの標準偏差・平均・中央値・z_max** だけを計算します（ゲート `docs/A1_WALL_GATE.md` §6 の手順 0）。順位ごとの値・傾き・圧縮因子は計算しません。
- 結果の置き場所：`MyDrive/logit-thermo/A1_P0/`（まとめ `p0_summary.json` と、行ごとの表 `p0_rows.csv`）。
- **通知**：スマホの ntfy アプリで `{NTFY_TOPIC}` を購読しておくと、🚀 始まったとき・🎉 終わったとき・⚠️ エラーで止まったとき（エラーの文面まるごと）に届きます。ランタイムごと切れたときは届かないので、そのときは Colab を見てください。
- **終わったら自動でランタイムを切ります**（CU の節約）。Drive への保存を確定してから切るので、結果は残ります。エラーで止まったときは切らずに残します。切りたくないときは、最初に `AUTO_DISCONNECT = False` を実行してください。
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

MAIN = r'''# セル2：sha256 を確かめてから、行ごとに σ だけを計算する
import hashlib, io, json, os, platform
import numpy as np

BASE = '/content/drive/MyDrive/PROJECT_A/_LOGITS/PARENT'
OUT = '/content/drive/MyDrive/logit-thermo/A1_P0'
MODELS = ['PQW7', 'PQC7']
EXPECT_VOCAB, EXPECT_ROWS, EXPECT_TOKLEN = 152064, 128, 151665
os.makedirs(OUT, exist_ok=True)

def q(x, p):
    return float(np.percentile(x, p))

rows_out = []
summary = {'gate': 'docs/A1_WALL_GATE.md v0.2, step 0 (P0)', 'numpy': np.__version__, 'python': platform.python_version(),
           'models': {}}
t0 = time.time()
for m in MODELS:
    d = f'{BASE}/{m}/logits'
    meta = json.load(open(f'{d}/metadata.json'))
    tok_len = int(meta['tokenizer_len'])
    files = meta['precisions']['fp32']['files']
    items = sorted(files.items(), key=lambda kv: kv[1]['file'])
    ok_sha = 0
    for sid, info in items:
        raw = open(f"{d}/{info['file']}", 'rb').read()
        got = hashlib.sha256(raw).hexdigest()
        assert got == info['sha256'], f"sha256 が合いません：{m}/{info['file']}"
        ok_sha += 1
        a = np.load(io.BytesIO(raw)).astype(np.float64)
        assert a.shape == (info['rows'], info['vocab']), (m, info['file'], a.shape)
        z = a[:, :tok_len]                      # 使われていない番号（tokenizer_len 以上）を除く
        q25, q50, q75 = np.percentile(z, [25, 50, 75], axis=1)
        s_rob = (q75 - q25) / 1.349
        mean, std, zmax = z.mean(axis=1), z.std(axis=1), z.max(axis=1)
        lang = 'JA' if '-JA-' in sid else 'EN'
        for r in range(z.shape[0]):
            rows_out.append((m, info['file'], lang, r, mean[r], std[r], q50[r], s_rob[r], (zmax[r] - q50[r]) / s_rob[r]))
        del a, z, raw
    sub = [x for x in rows_out if x[0] == m]
    col = lambda i: np.array([x[i] for x in sub])
    def stats(v):
        return {'p10': q(v, 10), 'p25': q(v, 25), 'median': q(v, 50), 'p75': q(v, 75), 'p90': q(v, 90)}
    summary['models'][m] = {
        'model_id': meta.get('model_id'), 'tokenizer_len': tok_len, 'vocab_in_file': items[0][1]['vocab'],
        'tokenizer_len_as_expected': tok_len == EXPECT_TOKLEN, 'files_sha256_ok': ok_sha, 'rows': len(sub),
        'sigma_rob': stats(col(7)), 'std': stats(col(5)), 'mean': stats(col(4)), 'median_logit': stats(col(6)),
        'z_max': stats(col(8)),
        'sigma_rob_median_by_lang': {L: q(np.array([x[7] for x in sub if x[2] == L]), 50) for L in ('EN', 'JA')},
    }
    print(m, 'おわり：', len(sub), '行、sha256 OK', ok_sha, 'ファイル')

summary['elapsed_seconds'] = round(time.time() - t0, 1)
with open(f'{OUT}/p0_rows.csv', 'w') as f:
    f.write('model,file,lang,row,mean,std,median,sigma_rob,z_max\n')
    for x in rows_out:
        f.write(f'{x[0]},{x[1]},{x[2]},{x[3]},' + ','.join(f'{v:.6f}' for v in x[4:]) + '\n')
summary['rows_csv_sha256'] = hashlib.sha256(open(f'{OUT}/p0_rows.csv', 'rb').read()).hexdigest()
json.dump(summary, open(f'{OUT}/p0_summary.json', 'w'), ensure_ascii=False, indent=1)
print('まとめの大きさ：', os.path.getsize(f'{OUT}/p0_summary.json'), 'バイト')
print(json.dumps(summary, ensure_ascii=False, indent=1))
'''

DONE = """# セル3：終わりの通知と、ランタイムを切る
notify('🎉 終わりました：' + str(sum(v['rows'] for v in summary['models'].values())) + ' 行', tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    print('Drive への書き込みを確定してから、ランタイムを切ります（CU の節約。止めたいときは AUTO_DISCONNECT = False）')
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount()
    time.sleep(3)
    _runtime.unassign()
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


nb = {"cells": [md(HEAD), code(SETUP), code(MAIN), code(DONE)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parents[1] / "A1_P0_sigma_CPU.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("wrote", out)
