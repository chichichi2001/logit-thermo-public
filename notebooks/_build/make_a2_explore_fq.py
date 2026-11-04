"""探索 8.35：場の下見の h ＝ 0 を同じ種で回し直して q を記録するノートブックを組み立てる（L4、1 本）。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, CF, CD = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_explore_field', 'a2_explore_fq'))
src5, src3, src4 = rd('notebooks/_build/make_a2_5.py'), rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
notify_cell, DONE, FAIL, EXTRACT, GPU_CHECK = ns['notify_cell'], ns['DONE'], ns['FAIL'], ns['EXTRACT'], ns['GPU_CHECK']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
OUT = BASE + '/A2_explore/fq'
FIELD = BASE + '/A2_explore/field'
label = 'logit-thermo 探索 q の記録（場の下見の h ＝ 0 を同じ種で回し直し、Pythia-1B・OLMo-1B、L4）'
head = f"""# logit-thermo 探索：場の下見の h ＝ 0 を同じ種で回し直し、各歩の「候補の確率の和 q」を記録する（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 35〜40 分。終わったら自動で切れます。
- 探索（予想なし）。8.30 の場の下見の h ＝ 0（Pythia-1B・OLMo-2-1B、温度 0.4〜0.7、サンプル 4、記事 40）と**同じ種**で生成し、トークンが 8.30 と 1 つ残らず同じことを確かめてから、各歩の q（候補の確率の和、float32）も保存します。q があれば、h ＝ 0 の生成だけから場の効きを厳密に予言でき、8.30 の h ＝ ±0.3 とくらべられます。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードと `scripts/a2_explore_field.py`（`{h(CF)[:16]}…`）は Drive の前のノートブックから取り出してハッシュで確かめる。`scripts/a2_explore_fq.py`（`{h(CD)[:16]}…`）は埋め込み。
- 保存先 `A2_explore/fq/`。8.30 のデータ（`A2_explore/field/`）を読みます。途中で切れても、もう一度「すべて実行」すれば終わったモデルは飛ばします。**結果は表示しません。**
- **通知**：🚀・モデルごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
core = ("# セル2：計算のコード\n" + EXTRACT +
        f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}, 'SRCF': {h(CF)!r}}})\n"
        "SRC22, SRC24, SRC7C, SRC7BC, SRCF = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC'], _s['SRCF']\n"
        f"SRCQ = {CD!r}\nassert hashlib.sha256(SRCQ.encode()).hexdigest() == {h(CD)!r}\n"
        "exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRCF)\nexec(SRCQ)\nprint('コードを置きました')\n")
run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ 2 つのモデル → 8.30 とトークンが同じか確かめて保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_field('cuda'), self_test_fq('cuda'))
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    for name, hf_id in Q_MODELS:
        run_fq(name, hf_id, docs, {OUT!r}, {FIELD!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(models=Q_MODELS, temps=F_TEMPS, samples=F_S, seed=F_SEED, field_sha256={h(CF)!r}, fq_sha256={h(CD)!r},
                   torch=torch.__version__, transformers=transformers.__version__, python=sys.version.split()[0],
                   gpu=torch.cuda.get_device_name(0), matmul=torch.get_float32_matmul_precision()),
              open({OUT!r} + '/run_meta.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 q の記録')}"""
nb = {"cells": [md(head), code(notify_cell(label, 'logit-thermo - explore q record', GPU_CHECK)), code(core), code(run),
                code(DONE.replace('{what}', 'q の記録（8.30 の h ＝ 0 と同じトークン）を A2_explore/fq に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_fq_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(CD)[:16])
