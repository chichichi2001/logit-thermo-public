"""A2-23 第 1 段のあと：(1) 指示調整版の s1 の npz（9MB くらい）を 3MB ずつに切る、(2) 重複したフォルダの npz が本番と中身まで同じかを確かめる（CPU、中身は表示しない）。"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
code = r'''import os, time, json, hashlib, urllib.request, traceback
NTFY_TOPIC = 'YOUR_NTFY_TOPIC'
NB_LABEL = 'logit-thermo A2-23 第1段のあと（指示調整版のデータの小分けと、重複フォルダの中身の一致の確かめ、CPU）'
def notify(msg, tags='pig'):
    try:
        urllib.request.urlopen(urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\n[' + NB_LABEL + ']').encode('utf-8'),
                               headers={'Title': 'logit-thermo - A2-23 split and duplicate check', 'Tags': tags}), timeout=10).read()
    except Exception as e:
        print('通知は送れませんでした：', type(e).__name__)
notify('🚀 スタートしました', tags='rocket')
from google.colab import drive, runtime
try:
    import numpy as np
    drive.mount('/content/drive')
    B = '/content/drive/MyDrive/logit-thermo'
    D, DUP = B + '/A2_23_s1', B + '/A2_23_s1_dup_runtime2'
    assert os.path.isdir(D) and os.path.isdir(DUP), 'フォルダが見つかりません'
    # (1) 小分け
    os.makedirs(D + '/parts', exist_ok=True)
    man = {}
    for name in ('qwen2.5-1.5b-it@s1.npz', 'smollm2-1.7b-it@s1.npz'):
        raw = open(f'{D}/{name}', 'rb').read()
        parts = [raw[i:i + 3_000_000] for i in range(0, len(raw), 3_000_000)]
        for i, p in enumerate(parts):
            open(f'{D}/parts/{name}.part{i}', 'wb').write(p)
        back = b''.join(open(f'{D}/parts/{name}.part{i}', 'rb').read() for i in range(len(parts)))
        assert hashlib.sha256(back).hexdigest() == hashlib.sha256(raw).hexdigest(), name
        man[name] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), parts=[len(p) for p in parts])
    json.dump(man, open(D + '/parts/manifest.json', 'w'), indent=1)
    # (2) 重複の確かめ：同じ名前の npz を、かかった秒数（seconds）と記録（meta）以外の全部の配列で比べる（中身は表示しない）
    chk = {}
    for name in sorted(os.listdir(DUP)):
        if not name.endswith('.npz') or not os.path.exists(f'{D}/{name}'):
            continue
        a, b = np.load(f'{D}/{name}'), np.load(f'{DUP}/{name}')
        keys = sorted(set(a.files) - {'seconds', 'meta'})
        same = {k: bool(k in b.files and a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f')) for k in keys}
        ma, mb = json.loads(str(a['meta'])), json.loads(str(b['meta']))
        chk[name] = dict(all_same=all(same.values()), differ=[k for k, v in same.items() if not v], gpu=[ma.get('gpu'), mb.get('gpu')],
                         seconds=[float(a['seconds']), float(b['seconds'])])
    json.dump(chk, open(D + '/dup_check.json', 'w'), indent=1, ensure_ascii=False)
    n_same = sum(v['all_same'] for v in chk.values())
    drive.flush_and_unmount()
    notify(f'🎉 終わりました：小分け 2 つ。重複 {len(chk)} 個のうち、中身まで同じ {n_same} 個', tags='tada')
except Exception as e:
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
    notify('⚠️ 止まりました：\n' + txt, tags='warning')
time.sleep(3); runtime.unassign()
'''
head = """# logit-thermo A2-23：第 1 段のデータの小分けと、重複フォルダの確かめ（CPU）

- **必要な secret：なし**。**ランタイム：CPU**（GPU はいりません）。
- (1) `A2_23_s1/` の指示調整版の npz 2 つ（各 9MB くらい）を、Claude が取りに行ける大きさ（3MB ずつ）に切って `A2_23_s1/parts/` に置きます。切ったものをつなげたハッシュが元と同じかを確かめます。
- (2) ランタイム 2/3 が別のフォルダ（`A2_23_s1_dup_runtime2/`）に書いた重複の npz が、本番の npz と中身まで同じか（かかった秒数と記録を除く全部の配列）を確かめて、`A2_23_s1/dup_check.json` に「同じか・違う配列の名前」だけを書きます。
- 中身は見ません・表示しません。1〜3 分で終わります。終わったら、またはエラーで止まったら、通知（🚀・🎉・⚠）を送ってランタイムを切ります。
"""
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": head},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_23_split_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size)
