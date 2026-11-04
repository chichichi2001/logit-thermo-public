"""探索 Y の続き 2（タイトルと離れたつながり）の GPU（L4）ノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22, CXY, CT = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_xy.py'), rd('scripts/a2_explore_title.py')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
src = rd('notebooks/_build/make_a2_4.py')
src3 = rd('notebooks/_build/make_a2_3.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src, re.S).group(0), ns)
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
BASE = '/content/drive/MyDrive/logit-thermo'
OUT = BASE + '/A2_explore'
LABEL = 'logit-thermo 探索 Y の続き 2（タイトルは前提を担うか・離れた文のつながり、4 モデル、L4）'
HEAD = f"""# logit-thermo 探索 Y の続き 2：タイトルは「前提」を担っているか、離れた文どうしのつながりはあるか（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 20〜30 分。
- **探索**です。確かめには使いません。生成はせず、探索 Y と同じ BBC の記事の窓（38 本）を 4 つのモデルに読ませるだけです。
- タイトルの 3 条件（なし・その記事のタイトル・別の記事のタイトル）× 並べ替え 7 つ（探索 Y の 5 つ ＋「最初の文だけ残して残りを逆順」「最初の段落だけ残して残りの段落を逆順」）で、本文の確率を測ります。
- 読み方（R1〜R4、参考 R5）は回す前に決めて、`scripts/a2_explore_title.py` の説明文に書いてあります。
- タイトルも本文も、中身は表示も保存もしません。保存先 `A2_explore/title_summary.json` と `A2_explore/title_{{モデル}}.npz`（数字だけ）。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）、`scripts/a2_explore_xy.py`（`{h(CXY)[:16]}…`）、`scripts/a2_explore_title.py`（`{h(CT)[:16]}…`）を使います（はじめの 2 つは、Drive にある探索 X・Y のノートブックから取り出して、ハッシュで確かめます。**`A2_explore_xy_L4.ipynb` を logit-thermo フォルダから動かさないでください**）。
- **通知**：🚀・モデルごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
GPU_CHECK = """os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14e9:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU なし'
    notify(f'⚠️ GPU が足りません（{name}）。L4 を選んでください。ランタイムを切ります', tags='warning')
    from google.colab import runtime; runtime.unassign()
    raise SystemExit
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
"""
CORE = ("# セル2：計算のコード\n"
        "# a2_2_collect.py と a2_explore_xy.py は、Drive にある探索 X・Y のノートブック（A2_explore_xy_L4.ipynb）に埋め込んだものを\n"
        "# 取り出して使う（ハッシュで中身が同じことを確かめる）。このノートブックに埋め込むのは a2_explore_title.py だけ。\n"
        "import ast, glob, hashlib, json\n"
        f"_cands = sorted(glob.glob({BASE!r} + '/**/A2_explore_xy_L4*.ipynb', recursive=True))\n"
        "assert _cands, 'Drive の logit-thermo フォルダに A2_explore_xy_L4.ipynb が見つかりません'\n"
        "_got = {}\n"
        "for _p in _cands:\n"
        "    for _c in json.load(open(_p))['cells']:\n"
        "        _s = _c['source'] if isinstance(_c['source'], str) else ''.join(_c['source'])\n"
        "        if 'SRC22 = ' not in _s:\n"
        "            continue\n"
        "        for _n in ast.parse(_s).body:\n"
        "            if isinstance(_n, ast.Assign) and isinstance(_n.targets[0], ast.Name) and _n.targets[0].id in ('SRC22', 'SRCXY'):\n"
        "                _got[_n.targets[0].id] = ast.literal_eval(_n.value)\n"
        "    if len(_got) == 2:\n"
        "        break\n"
        "SRC22, SRCXY = _got['SRC22'], _got['SRCXY']\n"
        f"SRCT = {CT!r}\n"
        f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
        f"assert hashlib.sha256(SRCXY.encode()).hexdigest() == {h(CXY)!r}\n"
        f"assert hashlib.sha256(SRCT.encode()).hexdigest() == {h(CT)!r}\n"
        "exec(SRC22)\nexec(SRCXY)\nexec(SRCT)\nprint('コードを置きました（', _p, 'から 2 つ取り出しました）')\n")
RUN = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 4 モデルで 3 条件 × 7 並べ替え → 保存（結果は表示しない）
import json
try:
    print('自己テスト：', self_test('cuda'), self_test_xy('cuda'), self_test_title('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = run_title(docs, {OUT!r}, 'cuda', notify=lambda m: notify('✅ ' + m, tags='white_check_mark'))
    r['a2_explore_title_sha256'] = {h(CT)!r}
    json.dump(r, open({OUT!r} + '/title_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 Y の続き 2')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore title', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'タイトルと離れたつながりの計算を A2_explore/title_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_title_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size, h(CT))
