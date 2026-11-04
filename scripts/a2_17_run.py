"""A2-17 の実行：A2-16 と同じ名前空間（scripts/a2_16_run.py の load16ns）に scripts/a2_17_analyze.py を足して、種類ごとに当てる。
使い方：python3 scripts/a2_17_run.py <A2_17 のフォルダ（arxiv/・code/・fiction/・bbc/ を含む）> <出力の json>
        python3 scripts/a2_17_run.py --self-test <作業用の空のフォルダ>"""
import sys, json, hashlib, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
from a2_16_run import load16ns, clean


def load17ns():
    ns = load16ns()
    exec(open('scripts/a2_17_analyze.py').read(), ns)
    return ns


if __name__ == '__main__':
    ns = load17ns()
    if sys.argv[1] == '--self-test':
        print(json.dumps(clean(ns['self_test17'](sys.argv[2])), ensure_ascii=False, indent=1)); sys.exit()
    base, out = sys.argv[1], sys.argv[2]
    shas = {f: hashlib.sha256(open(f'scripts/{f}', 'rb').read()).hexdigest() for f in
            ('a2_17_analyze.py', 'a2_16_analyze.py', 'a2_explore_onset.py', 'a2_8_vh.py', 'a2_13_analyze.py', 'a2_explore_ladder.py')}
    print({k: v[:16] for k, v in shas.items()})
    st = ns['self_test16'](); print('self_test16 OK')
    res = ns['analyze17'](base)
    res['sha256'] = shas
    full = {g: res['records'][g].pop('_full') for g in ns['A17_GEN']}
    json.dump(clean(res), open(out, 'w'), ensure_ascii=False, indent=1)
    json.dump(clean(full), open(out.replace('.json', '_full.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(clean(res['verdict']), ensure_ascii=False, indent=1))
    print(json.dumps(clean({g: {k: v[0] for k, v in res['genres'][g].items()} for g in ns['A17_GEN']}), ensure_ascii=False, indent=1))
