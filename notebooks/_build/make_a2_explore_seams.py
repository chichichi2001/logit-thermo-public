"""探索 Y の続き（壊れた文の境目の数）の CPU ノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22 = rd('scripts/a2_2_collect.py')
CXY = rd('scripts/a2_explore_xy.py')
CS = rd('scripts/a2_explore_seams.py')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
src = rd('notebooks/_build/make_a2_3.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src, re.S).group(0), ns)
notify_cell, DONE = ns['notify_cell'], ns['DONE']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
BASE = '/content/drive/MyDrive/logit-thermo'
OUT = BASE + '/A2_explore'
LABEL = 'logit-thermo 探索 Y の続き（戻れなさは壊れた文の境目の数に比例するか、CPU）'
HEAD = f"""# logit-thermo 探索 Y の続き：戻れなさは「壊れた文の境目の数」に比例するか（CPU）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：CPU（標準）**。見込みは 5〜10 分。
- **探索**です。確かめには使いません。GPU の計算はしません。探索 X・Y で保存した `A2_explore/xy_{{モデル}}.npz` を読み直し、記事ごとに文の数と段落の数を数えて、境目 1 つあたりの「戻れなさ」を出します。
- 読み方は回す前に決めて、`scripts/a2_explore_seams.py` の説明文に書いてあります。
- 中身は表示も保存もしません（数だけ）。保存先 `A2_explore/seams_summary.json`。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）、`scripts/a2_explore_xy.py`（`{h(CXY)[:16]}…`）、`scripts/a2_explore_seams.py`（`{h(CS)[:16]}…`）をそのまま埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport hashlib\n"
        f"SRC22 = {C22!r}\nSRCXY = {CXY!r}\nSRCS = {CS!r}\n"
        f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
        f"assert hashlib.sha256(SRCXY.encode()).hexdigest() == {h(CXY)!r}\n"
        f"assert hashlib.sha256(SRCS.encode()).hexdigest() == {h(CS)!r}\n"
        "exec(SRC22)\nexec(SRCXY)\nexec(SRCS)\nprint('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 数えて計算 → 保存（結果は表示しない）
import json
try:
    print('自己テスト：', self_test_seams())
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = analyze_seams(docs, {OUT!r})
    r['a2_explore_seams_sha256'] = {h(CS)!r}
    json.dump(r, open({OUT!r} + '/seams_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
    import traceback
    _NOTIFIED['error'] = True
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    v = os.environ.get('HF_TOKEN')
    if v: txt = txt.replace(v, '***')
    txt = txt.encode('utf-8')[-3300:].decode('utf-8', 'ignore')
    notify('⚠️ 探索 Y の続きで止まりました。ランタイムを切ります：\\n' + txt, tags='warning')
    if globals().get('AUTO_DISCONNECT', True):
        from google.colab import drive as _d, runtime as _r
        _d.flush_and_unmount(); time.sleep(3); _r.unassign()
    raise
"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore seams', """try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
""")), code(CORE), code(RUN), code(DONE.replace('{what}', '境目の数の計算を A2_explore/seams_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_seams_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size, h(CS), h(CXY))
