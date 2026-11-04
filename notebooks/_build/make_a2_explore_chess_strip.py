"""探索：チェスの記号を外した取り直しのノートブック（L4）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
CH, ST = rd('scripts/a2_explore_chess.py'), rd('scripts/a2_explore_chess_strip.py')
src5 = rd('notebooks/_build/make_a2_5.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
ns['BASE'] = BASE
exec(re.search(r'EXTRACT = f""".*?\n"""\n', src5, re.S).group(0), ns)
notify_cell, DONE, FAIL, GPU_CHECK, EXTRACT = ns['notify_cell'], ns['DONE'], ns['FAIL'], ns['GPU_CHECK'], ns['EXTRACT']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 チェスの記号を外した取り直し（メイトの # のせいか、4 モデル、L4）'
HEAD = f"""# logit-thermo 探索：チェスの記号を外した取り直し（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**。見込みは 20〜30 分。終わったら自動で切れます。
- 探索（予想なし）。前の下見の 250 局面（`A2_explore/chess_positions.json`）で、手の文字列から「+」と「#」を外した確率を 4 モデルで取り直し、メイトの手の確率が低かったのが記号のせいか手のせいかを比べて `A2_explore/chess_strip_summary.json` に書きます。**結果は表示しません。**
- 計算のコード：`scripts/a2_explore_chess.py` は Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_explore_chess_strip.py`（`{h(ST)[:16]}…`）は埋め込み。
- **通知**：🚀・モデルごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = ("# セル2：計算のコード\nimport gc, math\n" + EXTRACT +
        f"_s = take_src({{'SRCCH': {h(CH)!r}}})\nSRCCH = _s['SRCCH']\n"
        f"SRCST = {ST!r}\nassert hashlib.sha256(SRCST.encode()).hexdigest() == {h(ST)!r}\n"
        "exec(SRCCH)\nexec(SRCST)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('自己テスト：', self_test_strip())\n")
RUN = f"""# セル3：取り直し → まとめ（結果は表示しない）
try:
    OUTD = {BASE!r} + '/A2_explore'
    P = json.load(open(OUTD + '/chess_positions.json'))['positions']
    assert len(P) == 250, len(P)
    orig, strip = {{}}, {{}}
    for name, hf_id, dt in CH_MODELS:
        orig[name] = json.load(open(OUTD + f'/chess_scores_{{name}}.json'))['scores']
        sp = OUTD + f'/chess_scores_strip_{{name}}.json'
        if not os.path.exists(sp):
            t0 = time.time()
            tok, model = load_any(hf_id, dt, 'cuda')
            S = [score_moves(tok, model, pz['prompt'], [strip_san(s) for s in pz['moves_san']], 'cuda').tolist() for pz in P]
            json.dump(dict(model=hf_id, dtype=dt, scores=S), open(sp + '.part', 'w')); os.replace(sp + '.part', sp)
            del model; gc.collect(); torch.cuda.empty_cache()
            notify(f'✅ {{name}} おわり（{{time.time() - t0:.0f}} 秒）', tags='white_check_mark')
        strip[name] = json.load(open(sp))['scores']
    summ = strip_summary(P, orig, strip)
    summ = dict(models=summ, a2_explore_chess_sha256={h(CH)!r}, a2_explore_chess_strip_sha256={h(ST)!r})
    json.dump(_clean(summ), open(OUTD + '/chess_strip_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 チェスの記号を外した取り直し')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore chess strip', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', '記号を外した取り直しを A2_explore/chess_strip_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_chess_strip_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(CH), h(ST))
