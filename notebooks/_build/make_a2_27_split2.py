"""A2-27 のあと（2 回目）：9MB の前半（part0）は Drive の道具が受け取れなかった（返事が大きすぎて接続が切れる）。
part0 だけを 3MB ずつに切り直す（CPU、中身は表示しない）。後半（part1、2MB くらい）は受け取れたのでそのまま。"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
code = r'''import os, time, json, hashlib, urllib.request, traceback
NTFY_TOPIC = 'YOUR_NTFY_TOPIC'
NB_LABEL = 'logit-thermo A2-27 のあと（データの前半を 3MB ずつに切り直し、CPU）'
def notify(msg, tags='pig'):
    try:
        urllib.request.urlopen(urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\n[' + NB_LABEL + ']').encode('utf-8'),
                               headers={'Title': 'logit-thermo - A2-27 split2', 'Tags': tags}), timeout=10).read()
    except Exception as e:
        print('通知は送れませんでした：', type(e).__name__)
notify('🚀 スタートしました', tags='rocket')
from google.colab import drive, runtime
try:
    drive.mount('/content/drive')
    D = '/content/drive/MyDrive/logit-thermo/A2_27'
    P = D + '/parts'
    man = json.load(open(P + '/manifest.json'))
    assert len(man) == 21, ('manifest が 21 個ではありません', len(man))
    os.makedirs(P + '/sub', exist_ok=True)
    sub = {}
    for name in sorted(man):
        raw = open(f'{D}/{name}', 'rb').read()
        assert hashlib.sha256(raw).hexdigest() == man[name]['sha256'], ('元の npz のハッシュが manifest と違います', name)
        p0 = raw[:man[name]['parts'][0]]
        pieces = [p0[i:i + 3_000_000] for i in range(0, len(p0), 3_000_000)]
        for j, s in enumerate(pieces):
            open(f'{P}/sub/{name}.part0.s{j}', 'wb').write(s)
        back = b''.join(open(f'{P}/sub/{name}.part0.s{j}', 'rb').read() for j in range(len(pieces)))
        assert back == p0, name
        sub[name] = dict(part0_size=len(p0), part0_sha256=hashlib.sha256(p0).hexdigest(),
                         subs=[len(s) for s in pieces], sub_sha256=[hashlib.sha256(s).hexdigest() for s in pieces])
    json.dump(sub, open(P + '/sub/manifest_sub.json', 'w'), indent=1)
    drive.flush_and_unmount()
    notify(f'🎉 終わりました：{len(sub)} 個の前半を 3MB ずつに切り直しました（元と一致を確かめ済み）', tags='tada')
except Exception as e:
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
    notify('⚠️ 止まりました：\n' + txt, tags='warning')
time.sleep(3); runtime.unassign()
'''
head = """# logit-thermo A2-27 のあと：データの前半を 3MB ずつに切り直す（CPU）

- **必要な secret：なし**。**ランタイム：CPU**（GPU はいりません）。
- 前回 9MB ずつに切った前半（`part0`）は、Claude の Drive の道具が受け取れませんでした（返事が大きすぎて接続が切れる）。後半（`part1`、2MB くらい）は受け取れました。
- そこで前半だけを 3MB ずつに切り直して `A2_27/parts/sub/` に置き、つなげたものが元と同じかを確かめて、`A2_27/parts/sub/manifest_sub.json` に大きさとハッシュを書きます。前回の部品も元の npz も消しません・変えません。
- 中身は見ません・表示しません。1〜3 分で終わります。終わったら、またはエラーで止まったら、通知（🚀・🎉・⚠）を送ってランタイムを切ります。
"""
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": head},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_27_split2_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(p.name, p.stat().st_size)
