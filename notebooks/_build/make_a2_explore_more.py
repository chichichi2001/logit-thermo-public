"""探索の続き（C の手の確かめ直し ＋ 候補 A の下見）の CPU ノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22 = rd('scripts/a2_2_collect.py')
CX = rd('scripts/a2_explore_copy.py').replace("\nif __name__ == '__main__':", "\nif False:")
CM = rd('scripts/a2_explore_more.py')
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
LABEL = 'logit-thermo 探索の続き（C の手の確かめ直しと、長い文章の下見、CPU）'
HEAD = f"""# logit-thermo 探索の続き：C の手の確かめ直し ＋ 候補 A（長さ）の下見（CPU）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：CPU（標準）**。見込みは 20〜40 分。
- **探索**です。確かめには使いません。
- 前半（案 1）：探索 C の「写し歩」を決める並びの長さを 3 と 8 に変えて分け直す（ニュースの 9 条件）。A2-3 の 6 条件で、位置ごとのフィードバックを出す。保存先 `A2_explore/more_part1.json`。
- 後半（案 2）：2025-01〜06 の文章で、書き出し 64 ＋ 続き 2,048 トークン以上ある文書の数を、データセット・月ごとに数える。**中身は表示も保存もしません**（数だけ）。保存先 `A2_explore/more_part2.json`。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）、`scripts/a2_explore_copy.py`、`scripts/a2_explore_more.py`（`{h(CM)[:16]}…`）をそのまま埋め込み。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL}」。終わったらランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport hashlib\n"
        f"SRC22 = {C22!r}\nSRCX = {CX!r}\nSRCM = {CM!r}\n"
        f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
        f"assert hashlib.sha256(SRCX.encode()).hexdigest() == {h(CX)!r}\n"
        f"assert hashlib.sha256(SRCM.encode()).hexdigest() == {h(CM)!r}\n"
        "exec(SRC22)\nexec(SRCX)\nexec(SRCM)\nprint('自己テスト（写しの判定と分け方）：', self_test())\n")
RUN1 = f"""# セル3：前半（案 1）。記事を読み直して（A2-2 と同じか確かめる）、計算して保存
import json, os
os.makedirs({OUT!r}, exist_ok=True)
docs, meta = load_docs()
old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
r1 = part1({BASE!r}, docs)
r1['a2_explore_more_sha256'] = {h(CM)!r}
json.dump(r1, open({OUT!r} + '/more_part1.json', 'w'), ensure_ascii=False, separators=(',', ':'))
notify('✅ 前半（C の手の確かめ直し）を保存しました', tags='white_check_mark')
"""
RUN2 = f"""# セル4：後半（案 2）。長い文章の数を数えて保存（中身は保存しない）
r2 = part2()
r2_meta = {{'a2_explore_more_sha256': {h(CM)!r}, 'need_tokens': LONG_NEED, 'tokenizer': LONG_TOKENIZER}}
json.dump({{'counts': r2, 'meta': r2_meta}}, open({OUT!r} + '/more_part2.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('保存しました')
"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore more', """try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
""")), code(CORE), code(RUN1), code(RUN2), code(DONE.replace('{what}', '前半と後半を A2_explore に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_more_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size, h(CM))
