"""A2-27 のあと：採点し直しの npz 21 個（各 11MB くらい）を、Drive の道具の上限（10MB）より小さい 9MB ずつに切る（CPU、中身は表示しない）。"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
code = r'''import os, time, json, hashlib, urllib.request, traceback
NTFY_TOPIC = 'YOUR_NTFY_TOPIC'
NB_LABEL = 'logit-thermo A2-27 のあと（採点し直しのデータ 21 個の小分け、CPU）'
def notify(msg, tags='pig'):
    try:
        urllib.request.urlopen(urllib.request.Request('https://ntfy.sh/' + NTFY_TOPIC, data=(msg + '\n[' + NB_LABEL + ']').encode('utf-8'),
                               headers={'Title': 'logit-thermo - A2-27 split', 'Tags': tags}), timeout=10).read()
    except Exception as e:
        print('通知は送れませんでした：', type(e).__name__)
notify('🚀 スタートしました', tags='rocket')
from google.colab import drive, runtime
try:
    drive.mount('/content/drive')
    B = '/content/drive/MyDrive/logit-thermo'
    D = B + '/A2_27'
    assert os.path.isdir(D), 'フォルダが見つかりません'
    names = sorted(n for n in os.listdir(D) if n.endswith('.npz'))
    assert len(names) == 21, ('npz が 21 個ではありません', len(names))
    os.makedirs(D + '/parts', exist_ok=True)
    man = {}
    for name in names:
        raw = open(f'{D}/{name}', 'rb').read()
        parts = [raw[i:i + 9_000_000] for i in range(0, len(raw), 9_000_000)]
        for i, p in enumerate(parts):
            open(f'{D}/parts/{name}.part{i}', 'wb').write(p)
        back = b''.join(open(f'{D}/parts/{name}.part{i}', 'rb').read() for i in range(len(parts)))
        assert hashlib.sha256(back).hexdigest() == hashlib.sha256(raw).hexdigest(), name
        man[name] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), parts=[len(p) for p in parts])
    json.dump(man, open(D + '/parts/manifest.json', 'w'), indent=1)
    drive.flush_and_unmount()
    notify(f'🎉 終わりました：{len(man)} 個を小分けにしました（つなげたハッシュはどれも元と一致）', tags='tada')
except Exception as e:
    txt = ''.join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:]
    notify('⚠️ 止まりました：\n' + txt, tags='warning')
time.sleep(3); runtime.unassign()
'''
head = """# logit-thermo A2-27 のあと：採点し直しのデータの小分け（CPU）

- **必要な secret：なし**。**ランタイム：CPU**（GPU はいりません）。
- `A2_27/` の npz 21 個（各 11MB くらい）は、Claude が Drive から取れる大きさの上限（10MB）を少し超えています。そこで 9MB ずつに切って `A2_27/parts/` に置き、つなげたハッシュが元と同じかを確かめて、`A2_27/parts/manifest.json` に大きさとハッシュを書きます。
- 中身は見ません・表示しません。1〜3 分で終わります。終わったら、またはエラーで止まったら、通知（🚀・🎉・⚠）を送ってランタイムを切ります。
"""
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": head},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
p = ROOT / 'notebooks' / 'A2_27_split_CPU.ipynb'
p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print(p.name, p.stat().st_size)
