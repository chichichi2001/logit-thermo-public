"""A2-15（塩の癖は Pile の中身か LayerNorm か）のデータ集めのノートブック（L4、3 本）を組み立てる。分析は手元の CPU。"""
import hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
rd = lambda p: (ROOT / p).read_text(encoding='utf-8')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()
C22, C4, C7, C7B, C13, C14 = (rd(f'scripts/{n}.py') for n in ('a2_2_collect', 'a2_4_collect', 'a2_7_collect', 'a2_7b_collect', 'a2_13_collect', 'a2_15_collect'))
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
OUT = BASE + '/A2_15'
META_GPU = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"}}
GROUPS = {'gpt2': ('GPT-2 124M・355M・774M', '約 1〜1.5 時間'),
          'xl_bloom': ('GPT-2 1.5B、BLOOM-560M・1.1B', '約 2〜2.5 時間'),
          'rp': ('RedPajama-INCITE-3B', '約 1.5〜2 時間')}
for grp, (nice, eta) in GROUPS.items():
    label = f'logit-thermo A2-15 塩の癖は Pile か LayerNorm か データ集め（{grp}：{nice}、偽薬 30%、温度 0.4〜0.7、L4）'
    head = f"""# logit-thermo A2-15：塩の癖は Pile の中身か LayerNorm か（{nice}、GPU、L4）

- **必要な secret：なし**（`HF_TOKEN` があれば取り込みが少し速くなるだけ）。**ランタイム：L4**（ハイメモリはいりません）。見込みは{eta}。終わったら自動で切れます。
- ゲート：`docs/A2_15_GATE.md`（手順は A2-13 と同じ）。腕は「なし」「なし（別の種）」「偽薬 30%」「偽薬 30%（別の種）」の 4 本、温度は 0.4〜0.7 の 7 個（0.05 刻み）。3 本のノートブック（gpt2・xl_bloom・rp）を 3 枠で並べて回せます。
- 計算のコード：A2-2・A2-4・A2-7・A2-7b のデータ集めのコードは Drive の前のノートブックから取り出してハッシュで確かめ、`scripts/a2_13_collect.py`（`{h(C13)[:16]}…`）と `scripts/a2_15_collect.py`（`{h(C14)[:16]}…`）は埋め込み。
- 保存先 `A2_15/`。途中で切れても、もう一度「すべて実行」すれば終わった腕は飛ばします。
- **通知**：🚀・腕ごとの ✅・🎉・⚠️。名札は「{label}」。終わったら、またはエラーで止まったら、ランタイムを切ります。
"""
    core = ("# セル2：計算のコード\n" + EXTRACT +
            f"_s = take_src({{'SRC22': {h(C22)!r}, 'SRC24': {h(C4)!r}, 'SRC7C': {h(C7)!r}, 'SRC7BC': {h(C7B)!r}}})\n"
            "SRC22, SRC24, SRC7C, SRC7BC = _s['SRC22'], _s['SRC24'], _s['SRC7C'], _s['SRC7BC']\n"
            f"SRC13C = {C13!r}\nassert hashlib.sha256(SRC13C.encode()).hexdigest() == {h(C13)!r}\n"
            f"SRC15C = {C14!r}\nassert hashlib.sha256(SRC15C.encode()).hexdigest() == {h(C14)!r}\n"
            f"exec(SRC22)\nexec(SRC24)\nexec(SRC7C)\nexec(SRC7BC)\nexec(SRC13C)\nexec(SRC15C)\nC15_GROUP = {grp!r}\nprint('コードを置きました')\n")
    run = f"""# セル3：自己テスト → 記事（A2-2 と同じか確かめる）→ モデルごとに 4 本の腕 → 保存（結果は表示しない）
import json, sys, transformers
try:
    print('自己テスト：', self_test_a27('cuda'), self_test_7b('cuda'), self_test_13('cuda'), self_test_15())
    os.makedirs({OUT!r}, exist_ok=True)
    assert SEED == 20261027, ('記事を選ぶ種が A2-2 と違います', SEED)
    docs, meta = load_docs()
    old = json.load(open({BASE!r} + '/A2_2/docs_meta.json'))
    assert [d['text_sha256'] for d in old['docs']] == [d['text_sha256'] for d in meta['docs']], 'BBC の記事が A2-2 と違います'
    for name, hf_id, chunk in C15_GROUPS[C15_GROUP]:
        run_13(name, hf_id, chunk, docs, {OUT!r}, 'cuda', notify=lambda s: notify('✅ ' + s, tags='white_check_mark'))
    json.dump(dict(gate='docs/A2_15_GATE.md', group=C15_GROUP, models=C15_GROUPS[C15_GROUP], temps=C13_TEMPS, arms=[a[0] for a in C13_ARMS],
                   seed=C13_SEED, a2_7b_collect_sha256={h(C7B)!r}, a2_13_collect_sha256={h(C13)!r}, a2_15_collect_sha256={h(C14)!r}, torch=torch.__version__,
                   transformers=transformers.__version__, python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                   matmul=torch.get_float32_matmul_precision()), open({OUT!r} + f'/run_meta_{{C15_GROUP}}.json', 'w'), ensure_ascii=False, indent=1)
    print('保存しました')
except Exception as e:
{FAIL.replace('{where}', f'A2-15 のデータ集め（{grp}）')}"""
    nb = {"cells": [md(head), code(notify_cell(label, f'logit-thermo - A2-15 salt Pile or LN {grp}', GPU_CHECK)), code(core), code(run),
                    code(DONE.replace('{what}', f'A2-15 の {grp}（{nice}）を A2_15 に保存しました'))],
          "metadata": META_GPU, "nbformat": 4, "nbformat_minor": 0}
    p = ROOT / 'notebooks' / f'A2_15_collect_{grp}_L4.ipynb'
    p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(p.name, p.stat().st_size)
print('c13', h(C13)[:16], 'c15', h(C14)[:16])
