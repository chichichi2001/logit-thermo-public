"""A2-5 のデータ集め（L4）と分析（CPU）のノートブックを組み立てる。
大きなコード（A2-2 のデータ集め・分析、探索 C）は Drive にある前のノートブックから取り出し、ハッシュで確かめる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22 = rd('scripts/a2_2_collect.py')
A22 = rd('scripts/a2_2_analyze.py').replace("if __name__ == '__main__':", "if False:")
CX = rd('scripts/a2_explore_copy.py').replace("\nif __name__ == '__main__':", "\nif False:")
C5, A5 = rd('scripts/a2_5_collect.py'), rd('scripts/a2_5_analyze.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
notify_cell, DONE, FAIL = ns['notify_cell'], ns['DONE'], ns['FAIL']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
BASE = '/content/drive/MyDrive/logit-thermo'
OUT = BASE + '/A2_5'
EXTRACT = f"""import ast, glob, hashlib, json
def take_src(want):
    \"\"\"want：{{変数の名前: sha256}}。Drive の logit-thermo のノートブックから、埋め込んだコードを取り出してハッシュで確かめる。\"\"\"
    got = {{}}
    for p in sorted(glob.glob({BASE!r} + '/**/*.ipynb', recursive=True)):
        try:
            cells = json.load(open(p))['cells']
        except Exception:
            continue
        for c in cells:
            s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            if c.get('cell_type') != 'code' or 'SRC' not in s:
                continue
            try:
                body = ast.parse(s).body
            except SyntaxError:
                continue
            for n in body:
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in want and n.targets[0].id not in got:
                    v = ast.literal_eval(n.value)
                    if hashlib.sha256(v.encode()).hexdigest() == want[n.targets[0].id]:
                        got[n.targets[0].id] = v
        if len(got) == len(want):
            break
    miss = [k for k in want if k not in got]
    assert not miss, ('Drive のノートブックからコードを取り出せませんでした（前のノートブックを動かしていませんか）：', miss)
    return got
"""
GPU_CHECK = """os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 20e9:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU なし'
    notify(f'⚠️ GPU のメモリが足りません（{name}）。L4 を選んでください。ランタイムを切ります', tags='warning')
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
# ---------- データ集め（L4） ----------
LABEL_C = 'logit-thermo A2-5 データ集め（Pythia の大きさと学びの軸、9 回、L4）'
HEAD_C = f"""# logit-thermo A2-5：データ集め（Pythia の大きさと学びの軸、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 3.5〜4 時間（寝ているあいだに終わって、自動で切れます）。
- ゲート：`docs/A2_5_GATE.md` v0.2。Pythia の 70M・160M・410M・1B・1.4B・2.8B と、1B の学習途中（step 8000・24000・64000）の 9 回。いちばん最後が 2.8B（メモリのため 20 本ずつ 2 回に分けて生成）。
- 記事は A2-2 と同じ 40 本（A2-2 の種で選ぶ）。生成の種だけ新しい（20261102）。
- 計算のコード：`scripts/a2_2_collect.py`（`{h(C22)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_5_collect.py`（`{h(C5)[:16]}…`）は埋め込み。**logit-thermo フォルダの前のノートブックは動かさないでください。**
- 保存先 `A2_5/`。途中で切れても、もう一度「すべて実行」すれば終わった回は飛ばします。
- **通知**：🚀・回ごとの ✅・🎉・⚠️。名札は「{LABEL_C}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_C = ("# セル2：計算のコード\n" + EXTRACT +
          f"_s = take_src({{'SRC22': {h(C22)!r}}})\nSRC22 = _s['SRC22']\nSRC5C = {C5!r}\n"
          f"assert hashlib.sha256(SRC22.encode()).hexdigest() == {h(C22)!r}\n"
          f"assert hashlib.sha256(SRC5C.encode()).hexdigest() == {h(C5)!r}\n"
          "exec(SRC22)\nexec(SRC5C)\nprint('コードを置きました')\n")
RUN_C = f"""# セル3：自己テスト → 記事（A2-2 の種で選び、A2-2 と同じか確かめる）→ 9 回 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test('cuda'), self_test_a25('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    m = run_a25(docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    assert SEED == 20261027, '種が戻っていません'
    m.update(gate='docs/A2_5_GATE.md v0.2', a2_2_collect_sha256={h(C22)!r}, a2_5_collect_sha256={h(C5)!r},
             torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
             gpu=torch.cuda.get_device_name(0), float32_matmul_precision=torch.get_float32_matmul_precision())
    json.dump(m, open({OUT!r} + '/run_meta.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-5 のデータ集め')}"""
nbc = {"cells": [md(HEAD_C), code(notify_cell(LABEL_C, 'logit-thermo - A2-5 collect', GPU_CHECK)), code(CORE_C), code(RUN_C),
                 code(DONE.replace('{what}', 'A2-5 の 9 回を A2_5 に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
# ---------- 分析（CPU） ----------
LABEL_A = 'logit-thermo A2-5 分析（判定と要約、CPU）'
HEAD_A = f"""# logit-thermo A2-5：分析（CPU）

- **必要な secret：なし**。**ランタイム：CPU（標準）**。見込みは 20〜40 分。
- `A2_5/` の 9 回と、A2-2 の 7 モデル（記録用）を読み、ゲート v0.2 の判定を計算して `A2_5/a2_5_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_2_collect.py`・`scripts/a2_2_analyze.py`・`scripts/a2_explore_copy.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_5_analyze.py`（`{h(A5)[:16]}…`）は埋め込み（データを開く前に固定）。
- **通知**：🚀・🎉・⚠️。名札は「{LABEL_A}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE_A = ("# セル2：分析のコード\n" + EXTRACT +
          f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRCA22': {h(A22)!r}, 'SRCX': {h(CX)!r}}})\n"
          f"SRC22, SRCA22, SRCX = _s['SRC22'], _s['SRCA22'], _s['SRCX']\nSRC5A = {A5!r}\n"
          f"assert hashlib.sha256(SRC5A.encode()).hexdigest() == {h(A5)!r}\n"
          "exec(SRC22)\nexec(SRCA22)\nexec(SRCX)\nexec(SRC5A)\nprint('自己テスト：', self_test_a25_analyze())\n")
RUN_A = f"""# セル3：分析して保存（結果は表示しない）
import json
try:
    # A2-2 の分析のコード（SRCA22）が、同じ名前の SEED（ブートストラップの種 0）で記事を選ぶ種を上書きするので、
    # 記事を選ぶあいだだけ A2-2 の種（20261027）に戻す（2026-10-29 11:4x に直した。分析のコード a2_5_analyze.py は変えていない）
    _seed_keep = SEED
    SEED = 20261027
    docs, meta = load_docs()
    SEED = _seed_keep
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    r = analyze5({OUT!r}, {BASE!r} + '/A2_2', docs)
    r['a2_5_analyze_sha256'] = {h(A5)!r}
    json.dump(_clean(r), open({OUT!r} + '/a2_5_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', 'A2-5 の分析')}"""
nba = {"cells": [md(HEAD_A), code(notify_cell(LABEL_A, 'logit-thermo - A2-5 analyze', "")), code(CORE_A), code(RUN_A),
                 code(DONE.replace('{what}', '判定と要約を A2_5/a2_5_summary.json に保存しました'))],
       "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 0}
for nb, fn in [(nbc, 'A2_5_collect_L4.ipynb'), (nba, 'A2_5_analyze_CPU.ipynb')]:
    p = ROOT / 'notebooks' / fn
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p, p.stat().st_size)
print('a2_5_collect', h(C5)); print('a2_5_analyze', h(A5)); print('a2_2_analyze(if False)', h(A22)); print('copy(if False)', h(CX))
