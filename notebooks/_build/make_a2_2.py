"""A2-2（ゲート v0.3）のデータ集めの GPU ノートブックを組み立てる。計算は scripts/a2_2_collect.py をそのまま埋め込む。

    python3 notebooks/_build/make_a2_2.py  ->  notebooks/A2_2_collect_GPU.ipynb
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE = (ROOT / 'scripts' / 'a2_2_collect.py').read_text()
CORE_SHA = hashlib.sha256(CORE.encode()).hexdigest()
NTFY_TOPIC = "YOUR_NTFY_TOPIC"
LABEL = "logit-thermo A2-2 データ集め（7モデル、人の文脈・温度計・生成）"
TITLE = "logit-thermo - A2-2 collect"
OUT = "/content/drive/MyDrive/logit-thermo/A2_2"

HEAD = f"""# logit-thermo A2-2：データ集め（GPU）

- **必要な secret：なし**（`HF_TOKEN` があればモデルの取り込みが少し速くなるだけ。なくても動きます）
- **ランタイム：GPU（L4）**。ハイメモリはいりません。見込みは **1〜2 時間**（7 モデル × 15 温度 × 2 本 × 40 記事 × 512 トークンの生成と、別のモデルの文の対照）。
- **ゲート `docs/A2_2_GATE.md` v0.3 が凍結されてから回してください。** このノートブックは数字を集めて保存するだけで、何も判定しません（分析は別の CPU ノートブック）。結果の中身も表示しません。
- 何をするか：①小さなランダムのモデルで自己テスト（一歩の量の計算・KV キャッシュ・1 歩目の一致）。だめなら ⚠ で止まります。②BBC のニュース（2025-04〜06）から定型文を除いた記事を 40 本選ぶ（中身は表示せず、ハッシュだけ保存）。③7 モデルそれぞれで、人の文脈の一歩の量・温度計の材料・a と h・生成を計算して保存。④別のモデルの文の対照（Pythia-1B ↔ OLMo-2-1B、Qwen2.5-1.5B ↔ SmolLM2-1.7B）。
- 計算のコードは logit-thermo の `scripts/a2_2_collect.py`（sha256 `{CORE_SHA[:16]}…`）をそのまま埋め込んであります。
- 結果の置き場所：`MyDrive/logit-thermo/A2_2/`（モデルごとの `<名前>.npz`、`cross_*.npz`、`docs_meta.json`、`run_meta.json`）。**途中で切れても、もう一度「すべてのセルを実行」すれば、終わったモデルは飛ばして続きから回ります。**
- **通知**：ntfy の `{NTFY_TOPIC}` に、🚀 始まったとき・モデルごとの区切り・🎉 終わったとき・⚠️ エラーで止まったとき（エラーの文面まるごと）に届きます。名札は「{LABEL}」。
- **終わったら自動でランタイムを切ります**（Drive への保存を確定してから）。エラーで止まったときは切らずに残します。切りたくないときは、最初に `AUTO_DISCONNECT = False` を実行してください。
- 使い方：「ランタイム」→「ランタイムのタイプを変更」で L4 を選んでから、「すべてのセルを実行」。
"""

SETUP = f"""# セル1：通知のしくみ（ntfy）、GPU の確認、Drive をつなぐ
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
        import traceback, os
        e = result.error_in_exec
        txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        v = os.environ.get('HF_TOKEN')
        if v: txt = txt.replace(v, '***')
        notify('⚠️ 止まりました（エラー）：\\n' + txt[-3500:], tags='warning')
if not globals().get('_HOOK_ON'):
    get_ipython().events.register('post_run_cell', _on_error); _HOOK_ON = True
notify('🚀 スタートしました（これが届けば通知の設定は OK です）', tags='rocket')

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # モデルを入れ替えるときのメモリの断片化を防ぐ（torch を読む前に）
import torch
if not torch.cuda.is_available():
    raise RuntimeError('GPU がありません。「ランタイムのタイプを変更」で L4 を選んでください')
print('GPU：', torch.cuda.get_device_name(0))
try:
    from google.colab import userdata
    import os
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
from google.colab import drive
drive.mount('/content/drive')
"""

CORE_CELL = "# セル2：計算のコード（logit-thermo の scripts/a2_2_collect.py をそのまま埋め込み）\n" + CORE

TEST = """# セル3：自己テスト（小さなランダムのモデル。1 分くらい）
print('自己テスト：', self_test('cuda'))
"""

DATA = f"""# セル4：記事を選ぶ（中身は表示しない。ハッシュだけ保存）
import json, os, platform
OUT = {OUT!r}
os.makedirs(OUT, exist_ok=True)
docs, docs_meta = load_docs()
mp = f'{{OUT}}/docs_meta.json'
if os.path.exists(mp):
    old = json.load(open(mp))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in docs_meta['docs']], '前の回と記事が違います'
else:
    json.dump(docs_meta, open(mp, 'w'), indent=1)
print('記事の候補：', docs_meta['pool_size'], '本、選んだ記事：', len(docs), '本')
"""

MAIN = f"""# セル5：7 モデルを順に回す（終わったモデルは飛ばす）→ 別のモデルの文の対照
import transformers, datasets
t_all = time.time()
for name, hf_id, fam in MODELS:
    run_model(name, hf_id, docs, OUT, 'cuda', notify=lambda m: notify('✅ ' + m, tags='white_check_mark'))
for a, b in CROSS:
    cross_model(a, b, docs, OUT, 'cuda')
    cross_model(b, a, docs, OUT, 'cuda')
json.dump({{'gate': 'docs/A2_2_GATE.md v0.3', 'a2_2_collect_sha256': {CORE_SHA!r}, 'torch': torch.__version__,
           'transformers': transformers.__version__, 'datasets': datasets.__version__, 'python': platform.python_version(),
           'gpu': torch.cuda.get_device_name(0), 'seconds_this_session': round(time.time() - t_all, 1)}},
          open(f'{{OUT}}/run_meta.json', 'w'), indent=1)
print('全部おわり')
"""

DONE = """# セル6：終わりの通知と、ランタイムを切る
notify('🎉 終わりました：7 モデルと対照のデータを Drive に保存しました', tags='tada')
if globals().get('AUTO_DISCONNECT', True):
    print('Drive への書き込みを確定してから、ランタイムを切ります（止めたいときは AUTO_DISCONNECT = False）')
    from google.colab import drive as _drive, runtime as _runtime
    _drive.flush_and_unmount()
    time.sleep(3)
    _runtime.unassign()
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


nb = {"cells": [md(HEAD), code(SETUP), code(CORE_CELL), code(TEST), code(DATA), code(MAIN), code(DONE)],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}},
      "nbformat": 4, "nbformat_minor": 0}
out = ROOT / "notebooks" / "A2_2_collect_GPU.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')))
print("wrote", out, "core sha256", CORE_SHA)
