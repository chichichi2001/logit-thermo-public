"""A2-2 の分析の CPU ノートブックを組み立てる。計算は scripts/a2_2_analyze.py をそのまま埋め込む。"""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CORE = (ROOT / 'scripts' / 'a2_2_analyze.py').read_text()
SHA = hashlib.sha256(CORE.encode()).hexdigest()
TOPIC = "YOUR_NTFY_TOPIC"
LABEL = "logit-thermo A2-2 分析（判定と要約、CPU）"
D = "/content/drive/MyDrive/logit-thermo/A2_2"
HEAD = f"""# logit-thermo A2-2：分析（CPU）

- **必要な secret：なし**
- **ランタイム：CPU（標準）で大丈夫です。** 見込みは 1〜3 分。
- 何をするか：`MyDrive/logit-thermo/A2_2/` の 7 モデルと対照のデータを読み、ゲート v0.3 の判定（E1・E2・P2・判定不能の条件）と記録の数字を計算して、`A2_2/a2_2_summary.json` に書きます。**結果はここには表示しません**（Claude が読みに行きます）。
- 計算のコードは logit-thermo の `scripts/a2_2_analyze.py`（sha256 `{SHA[:16]}…`、データを開く前に固定）をそのまま埋め込んであります。
- **通知**：ntfy の `{TOPIC}` に、🚀 始まったとき・🎉 終わったとき・⚠️ エラーで止まったとき（エラーの文面まるごと）に届きます。名札は「{LABEL}」。
- **終わったら自動でランタイムを切ります**（止めたいときは、最初に `AUTO_DISCONNECT = False` を実行）。
- 使い方：「すべてのセルを実行」。
"""
SETUP = f"""# セル1：通知と Drive
import urllib.request, time, os
NTFY_TOPIC = '{TOPIC}'
NB_LABEL = {LABEL!r}
def notify(msg, tags='pig'):
    try:
        req = urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\\n[' + NB_LABEL + ']').encode('utf-8'),
                                     headers={{'Title': 'logit-thermo - A2-2 analyze', 'Tags': tags}})
        urllib.request.urlopen(req, timeout=10).read()
        print('通知を送りました：', msg)
    except Exception as e:
        print('通知は送れませんでした（分析には影響しません）：', type(e).__name__)
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
CORE_CELL = "# セル2：分析のコード（scripts/a2_2_analyze.py をそのまま埋め込み）\n" + CORE.replace("if __name__ == '__main__':", "if False:")
RUN = f"""# セル3：分析して保存（結果は表示しない）
import json
D = {D!r}
r = analyze(D)
r['analyze_sha256'] = {SHA!r}
json.dump(_clean(r), open(D + '/a2_2_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('保存しました：', D + '/a2_2_summary.json', os.path.getsize(D + '/a2_2_summary.json'), 'バイト')
"""
DONE = """# セル4：終わりの通知と、ランタイムを切る
notify('🎉 終わりました：判定と要約を a2_2_summary.json に保存しました', tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount(); time.sleep(3); _runtime.unassign()
"""
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
nb = {"cells": [md(HEAD), code(SETUP), code(CORE_CELL), code(RUN), code(DONE)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "A2_2_analyze_CPU.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))
print("wrote", out, SHA)
