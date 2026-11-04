"""探索 X（型破り）と Y（文の順番の戻れなさ）の下見の GPU（L4）ノートブックを組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22, CXY = rd('scripts/a2_2_collect.py'), rd('scripts/a2_explore_xy.py')
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
LABEL = 'logit-thermo 探索 X・Y（型破りと文の順番の戻れなさ、4 モデル、L4）'
HEAD = f"""# logit-thermo 探索 X・Y：型破りと、文の順番の戻れなさ（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 20〜40 分。
- **探索**です。確かめには使いません。生成はせず、人の文章（A2-2 と同じ BBC の 40 本）を 4 つのモデル（Pythia-1B・Qwen2.5-1.5B・OLMo-2-1B・SmolLM2-1.7B）に読ませるだけです。
- X：人の文の 1 歩ずつの「選ぶ前の迷い（エントロピー）」と「実際の一歩の意外さ（サプライザル）」を記録します。
- Y：記事の頭の数段落を、元の順番・文の逆順・段落の逆順・ばらばら（2 通り）に並べて読ませ、文章全体の確率がどれだけ下がるかを測ります。
- 中身は表示も保存もしません。保存先 `A2_explore/xy_summary.json` と `A2_explore/xy_{{モデル}}.npz`（数字だけ）。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）と `scripts/a2_explore_xy.py`（`{h(CXY)[:16]}…`）をそのまま埋め込み。
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
CORE = ("# セル2：計算のコード\nimport hashlib\n"
        f"SRC22 = {C22!r}\nSRCXY = {CXY!r}\n"
        f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
        f"assert hashlib.sha256(SRCXY.encode()).hexdigest() == {h(CXY)!r}\n"
        "exec(SRC22)\nexec(SRCXY)\nprint('コードを置きました')\n")
RUN = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 4 モデルで X と Y → 保存（結果は表示しない）
import json
try:
    print('自己テスト：', self_test('cuda'), self_test_xy('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = run_xy(docs, {OUT!r}, 'cuda', notify=lambda m: notify('✅ ' + m, tags='white_check_mark'))
    r['a2_explore_xy_sha256'] = {h(CXY)!r}
    json.dump(r, open({OUT!r} + '/xy_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 X・Y')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore XY', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', '型破りと文の順番の戻れなさを A2_explore に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_xy_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p, p.stat().st_size, h(CXY))
