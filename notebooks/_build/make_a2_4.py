"""A2-4 の GPU（データ集め、L4）と CPU（分析）のノートブックを組み立てる。コードはそのまま埋め込む。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
C22, C24 = rd('scripts/a2_2_collect.py'), rd('scripts/a2_4_collect.py')
A22, A24 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:"), rd('scripts/a2_4_analyze.py')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
src = rd('notebooks/_build/make_a2_3.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src, re.S).group(0), ns)
notify_cell, DONE = ns['notify_cell'], ns['DONE']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
BASE = '/content/drive/MyDrive/logit-thermo'
OUT = BASE + '/A2_4'
META = lambda d: {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, **d}

# ---------- GPU ----------
LABEL_G = 'logit-thermo A2-4 データ集め（写しの禁止と偽薬、3 モデル × 3、L4）'
HEAD_G = f"""# logit-thermo A2-4：結晶を止めると T_0 は動くか — データ集め（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。
- **ランタイム：L4**（ハイメモリはいりません）。見込みは **3〜5 時間**（3 モデル × 3 つまみ × 15 温度 × 2 本 × 40 記事 × 512 トークン）。
- **ゲート `docs/A2_4_GATE.md` v0.2（凍結 2026-10-28 18:28）**。数字を集めて保存するだけで、何も判定しません。結果も表示しません。
- 順番：①自己テスト（A2-2 のものと、つまみの確かめ）②BBC の 40 本（A2-2 と同じことを確かめる）③モデルごとに「つまみなし → 写しの禁止 → 偽薬」（Pythia-1B → OLMo-2-1B → Qwen2.5-1.5B）。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）と `scripts/a2_4_collect.py`（`{h(C24)[:16]}…`）をそのまま埋め込み。TF32 は切ります。
- 結果の置き場所：`MyDrive/logit-thermo/A2_4/`。途中で切れても「すべてのセルを実行」で、終わったものは飛ばして続きから回ります。
- **通知**：ntfy の `YOUR_NTFY_TOPIC` に 🚀・回ごとの ✅・🎉・⚠️（エラーの文面まるごと）。名札は「{LABEL_G}」。
- **終わったら自動でランタイムを切ります**。エラーで止まったときも、通知してから切ります（止めたいときは最初に `AUTO_DISCONNECT = False`）。
"""
GPU_CHECK = """os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20e9:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU なし'
    notify(f'⚠️ GPU が足りません（{name}）。L4 を選んでください。ランタイムを切ります', tags='warning')
    from google.colab import runtime; runtime.unassign()
    raise SystemExit
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
print('GPU：', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception:
    pass
"""
CORE_G = ("# セル2：計算のコード（a2_2_collect.py と a2_4_collect.py をそのまま。同じ名前空間で実行）\nimport hashlib\n"
          f"SRC22 = {C22!r}\nSRC24 = {C24!r}\n"
          f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
          f"assert hashlib.sha256(SRC24.encode()).hexdigest() == {h(C24)!r}\n"
          "exec(SRC22)\nexec(SRC24)\nprint('コードを置きました')\n")
TEST_G = None   # FAIL のあとで作る
FAIL = """    import traceback
    _NOTIFIED['error'] = True
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    v = os.environ.get('HF_TOKEN')
    if v: txt = txt.replace(v, '***')
    txt = txt.encode('utf-8')[-3300:].decode('utf-8', 'ignore')
    notify('⚠️ {where}で止まりました。ランタイムを切ります：\\n' + txt, tags='warning')
    if globals().get('AUTO_DISCONNECT', True):
        from google.colab import drive as _d, runtime as _r
        _d.flush_and_unmount(); time.sleep(3); _r.unassign()
    raise
"""
TEST_G = f"""# セル3：自己テスト（小さなランダムのモデル。ここで止まっても、通知してからランタイムを切る）
try:
    print('A2-2 の自己テスト：', self_test('cuda'))
    print('つまみの自己テスト：', self_test_pen('cuda'))
    print('run_mode の通しの自己テスト：', self_test_run_mode('cuda'))
except Exception as e:
{FAIL.replace('{where}', '自己テスト')}"""
DATA_G = f"""# セル4：記事を選ぶ（A2-2 と同じことを確かめる。中身は表示しない）
import json
OUT = {OUT!r}
os.makedirs(OUT, exist_ok=True)
try:
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
except Exception as e:
{FAIL.replace('{where}', '記事を選ぶところ')}print('BBC：A2-2 と同じ 40 本')
"""
MAIN_G = f"""# セル5：モデルごとに「つまみなし → 写しの禁止 → 偽薬」（終わったものは飛ばす）
import transformers, platform
t_all = time.time()
try:
    for name, hf_id in CONDITIONS:
        for mode in MODES:
            run_mode(name, hf_id, docs, OUT, 'cuda', mode, notify=lambda m: notify('✅ ' + m, tags='white_check_mark'))
except Exception as e:
{FAIL.replace('{where}', 'データ集め')}json.dump({{'gate': 'docs/A2_4_GATE.md v0.2', 'a2_2_collect_sha256': {h(C22)!r}, 'a2_4_collect_sha256': {h(C24)!r},
           'torch': torch.__version__, 'transformers': transformers.__version__, 'python': platform.python_version(),
           'gpu': torch.cuda.get_device_name(0), 'float32_matmul_precision': torch.get_float32_matmul_precision(),
           'seconds_this_session': round(time.time() - t_all, 1)}}, open(OUT + '/run_meta.json', 'w'), indent=1)
print('全部おわり')
"""
nbg = {"cells": [md(HEAD_G), code(notify_cell(LABEL_G, 'logit-thermo - A2-4 collect', GPU_CHECK)), code(CORE_G), code(TEST_G),
                 code(DATA_G), code(MAIN_G), code(DONE.replace('{what}', '9 回分を Drive に保存しました'))],
       "metadata": META({"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}), "nbformat": 4, "nbformat_minor": 0}
(ROOT / 'notebooks' / 'A2_4_collect_L4.ipynb').write_text(json.dumps(nbg, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')

# ---------- CPU ----------
LABEL_C = 'logit-thermo A2-4 分析（判定と要約、CPU）'
HEAD_C = f"""# logit-thermo A2-4：分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 5〜15 分。
- `MyDrive/logit-thermo/A2_4/` の 9 回分と、A2-2 の 3 モデル（確かめ用）を読み、ゲート v0.2 の判定を計算して `A2_4/a2_4_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_analyze.py`（`{h(rd('scripts/a2_2_analyze.py'))[:16]}…`）と `scripts/a2_4_analyze.py`（`{h(A24)[:16]}…`）をそのまま埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_C}」。終わったらランタイムを切ります。
"""
CORE_C = ("# セル2：分析のコード\nimport hashlib\n"
          f"SRCA22 = {A22!r}\nSRCA24 = {A24!r}\n"
          f"assert hashlib.sha256(SRCA22.encode()).hexdigest() == {h(A22)!r}\n"
          f"assert hashlib.sha256(SRCA24.encode()).hexdigest() == {h(A24)!r}\n"
          "exec(SRCA22)\nexec(SRCA24)\nprint('コードを置きました')\n")
RUN_C = f"""# セル3：分析して保存（結果は表示しない）
import json
r = analyze4({OUT!r}, {BASE!r} + '/A2_2')
r['retok'] = retok_record({OUT!r}, {{'pythia-1b': 'EleutherAI/pythia-1b', 'olmo2-1b': 'allenai/OLMo-2-0425-1B', 'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B'}})
r['a2_4_analyze_sha256'] = {h(A24)!r}
r['a2_2_analyze_sha256'] = {h(rd('scripts/a2_2_analyze.py'))!r}
json.dump(_clean(r), open({OUT!r} + '/a2_4_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
print('保存しました')
"""
nbc = {"cells": [md(HEAD_C), code(notify_cell(LABEL_C, 'logit-thermo - A2-4 analyze', '')), code(CORE_C), code(RUN_C),
                 code(DONE.replace('{what}', '判定と要約を a2_4_summary.json に保存しました'))],
       "metadata": META({"colab": {"provenance": []}}), "nbformat": 4, "nbformat_minor": 0}
(ROOT / 'notebooks' / 'A2_4_analyze_CPU.ipynb').write_text(json.dumps(nbc, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
for p in ['A2_4_collect_L4.ipynb', 'A2_4_analyze_CPU.ipynb']:
    print(p, (ROOT / 'notebooks' / p).stat().st_size)
print('a2_4_collect', h(C24)); print('a2_4_analyze', h(A24))
