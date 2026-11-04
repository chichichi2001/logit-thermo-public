"""探索：チェスの盤の上のエネルギーのノートブック（L4）を組み立てる。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
CH = rd('scripts/a2_explore_chess.py')
src5 = rd('notebooks/_build/make_a2_5.py')
src3, src4 = rd('notebooks/_build/make_a2_3.py'), rd('notebooks/_build/make_a2_4.py')
ns = {'TOPIC': 'YOUR_NTFY_TOPIC'}
exec(re.search(r'def notify_cell\(.*?\n"""\n', src3, re.S).group(0), ns)
exec(re.search(r'DONE = """.*?"""\n', src3, re.S).group(0), ns)
exec(re.search(r'FAIL = """.*?"""\n', src4, re.S).group(0), ns)
exec(re.search(r'GPU_CHECK = """.*?"""\n', src5, re.S).group(0), ns)
BASE = '/content/drive/MyDrive/logit-thermo'
notify_cell, DONE, FAIL, GPU_CHECK = ns['notify_cell'], ns['DONE'], ns['FAIL'], ns['GPU_CHECK']
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
LABEL = 'logit-thermo 探索 チェスの盤のエネルギー（Stockfish・4 モデル、L4）'
HEAD = f"""# logit-thermo 探索：チェスの盤の上のエネルギー（GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは 40〜70 分。終わったら自動で切れます。
- 探索（予想なし）。Lichess のパズル（CC0）から 5 つのテーマ × 50 局面を選び、局面を棋譜の形（FEN の見出し＋相手の 1 手）で書き、Stockfish（Stockfish）で全部の合法手のエネルギー（最善手からの勝率の下がり方）を出し、4 つのモデル（Pythia-1B・OLMo-2-1B・Qwen2.5-1.5B・Qwen2.5-7B）が各手に付ける確率と比べます。**結果は表示しません。**
- 保存先 `A2_explore/`：`chess_positions.json`（局面とエネルギー）、`chess_scores_*.json`（モデルごと）、`chess_summary.json`。途中で切れても、もう一度「すべて実行」すれば、終わったところは飛ばします。
- 計算のコード：`scripts/a2_explore_chess.py`（`{h(CH)[:16]}…`）を埋め込み。Stockfishは `apt`、チェスのライブラリ（python-chess）は `pip` で入れます。Lichess のサイトにはつなぎません。
- **通知**：🚀・区切りごとの ✅・🎉・⚠️。名札は「{LABEL}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
CORE = (f"# セル2：計算のコード\nimport hashlib, json, math, gc\nSRCCH = {CH!r}\n"
        f"assert hashlib.sha256(SRCCH.encode()).hexdigest() == {h(CH)!r}\nexec(SRCCH)\n"
        "def _clean(o):\n"
        "    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}\n"
        "    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]\n"
        "    if isinstance(o, float) and not math.isfinite(o): return None\n"
        "    return o\n"
        "print('コードを置きました')\n")
RUN = f"""# セル3：道具を入れる → 自己テスト → 局面とエネルギー → モデルの確率 → まとめ（結果は表示しない）
import subprocess, shutil, sys
try:
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'stockfish'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'chess'], check=True)
    STOCKFISH_PATH = shutil.which('stockfish') or '/usr/games/stockfish'
    print('自己テスト：', self_test_chess_math(), self_test_chess_board(), self_test_score('cuda'))
    OUTD = {BASE!r} + '/A2_explore'
    os.makedirs(OUTD, exist_ok=True)
    pp = OUTD + '/chess_positions.json'
    if os.path.exists(pp):
        P = json.load(open(pp))
    else:
        import chess.engine
        eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        eng.configure({{'Threads': 2, 'Hash': 256}})
        positions, skipped, ncand = collect_positions(eng, notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
        P = dict(positions=positions, skipped=skipped, n_cand=ncand, engine=eng.id.get('name'), depth=CH_DEPTH)
        eng.quit()
        json.dump(P, open(pp + '.part', 'w'), ensure_ascii=False); os.replace(pp + '.part', pp)
    for name, hf_id, dt in CH_MODELS:
        sp = OUTD + f'/chess_scores_{{name}}.json'
        if os.path.exists(sp):
            continue
        t0 = time.time()
        tok, model = load_any(hf_id, dt, 'cuda')
        S = [score_moves(tok, model, pz['prompt'], pz['moves_san'], 'cuda').tolist() for pz in P['positions']]
        json.dump(dict(model=hf_id, dtype=dt, scores=S), open(sp + '.part', 'w')); os.replace(sp + '.part', sp)
        del model; gc.collect(); torch.cuda.empty_cache()
        notify(f'✅ {{name}} おわり（{{time.time() - t0:.0f}} 秒）', tags='white_check_mark')
    scores = {{name: json.load(open(OUTD + f'/chess_scores_{{name}}.json'))['scores'] for name, _, _ in CH_MODELS}}
    summ = summarize(P['positions'], scores)
    summ.update(engine=P['engine'], depth=P['depth'], skipped=P['skipped'], n_cand=P['n_cand'],
                both_castle=int(sum(pz['both_castle'] for pz in P['positions'])),
                dtypes={{name: dt for name, _, dt in CH_MODELS}}, a2_explore_chess_sha256={h(CH)!r})
    json.dump(_clean(summ), open(OUTD + '/chess_summary.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', '探索 チェスの盤のエネルギー')}"""
nb = {"cells": [md(HEAD), code(notify_cell(LABEL, 'logit-thermo - explore chess energy', GPU_CHECK)), code(CORE), code(RUN),
                code(DONE.replace('{what}', 'チェスの盤のエネルギーを A2_explore/chess_summary.json に保存しました'))],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}, "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_explore_chess_L4.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size, h(CH))
