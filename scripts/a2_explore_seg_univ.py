"""探索 8.29：8.24（3 つの谷は 25 本でも同じか）を、8.25 の新しいものさし（かたまりで切り出す）でやり直す（2026-10-31 09:3x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）・scripts/a2_8_vh.py の phases・scripts/a2_explore_seg.py を実行したのと同じ名前空間で実行する。
材料：A2-2・A2-5・A2-6 の 25 本（温度 0.6〜1.5 の 15 個、サンプル 2 × 記事 40）。本ごとに：温度 0.6・0.7 の L・A・C の歩の割合、
温度 0.6・0.7 を足した移り変わりの回数、C から出るうちの L への割合、C に入るうちの L からの割合、C → A のあとの行き先。
"""
import numpy as np

SU_RUNS = {
    'A2_2': ['pythia-1b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-inst', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b', 'smollm2-1.7b-inst'],
    'A2_5': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
             'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000'],
    'A2_6': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
             'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'],
}
SU_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]


def one_run_seg(toks):
    keys = ['LA', 'AL', 'AC', 'CA', 'LC', 'CL']
    row, tr = {}, {k: 0 for k in keys}
    after = {'C': 0, 'L': 0, 'end': 0}
    for T in (0.6, 0.7):
        k = SU_TEMPS.index(T)
        occ = np.zeros(3)
        for s in range(toks.shape[1]):
            for j in range(toks.shape[2]):
                o, t, labs = seg_counts(toks[k, s, j])
                occ += [o['L'], o['A'], o['C']]
                for x in keys:
                    tr[x] += t.get(x, 0)
                for i in range(len(labs) - 1):
                    if labs[i] == 'C' and labs[i + 1] == 'A':
                        m = i + 1
                        while m < len(labs) and labs[m] == 'A':
                            m += 1
                        after[labs[m] if m < len(labs) else 'end'] += 1
        occ = occ / occ.sum()
        row[f'frac_{T}'] = dict(L=float(occ[0]), A=float(occ[1]), C=float(occ[2]))
    row['trans_0.6_0.7'] = tr
    row['C_exit_to_L'] = tr['CL'] / (tr['CL'] + tr['CA']) if (tr['CL'] + tr['CA']) else None
    row['C_entry_from_L'] = tr['LC'] / (tr['LC'] + tr['AC']) if (tr['LC'] + tr['AC']) else None
    row['after_CA'] = after
    return row


def seg_univ(base):
    res = {}
    for folder, names in SU_RUNS.items():
        for n in names:
            Z = np.load(f'{base}/{folder}/{n}.npz')
            assert [round(float(t), 4) for t in Z['temps']] == SU_TEMPS
            res[f'{folder}/{n}'] = one_run_seg(Z['toks'])
    return res


def self_test_su():
    rng = np.random.default_rng(0)
    toks = rng.integers(100, 30000, (15, 2, 3, 512))
    toks[0, 0, 0, 150:330] = np.tile([5, 6, 7, 8, 9, 10, 11], 60)[:180]
    r = one_run_seg(toks)
    assert r['trans_0.6_0.7']['LC'] == 1 and r['trans_0.6_0.7']['CL'] == 1 and r['C_exit_to_L'] == 1.0
    return 'OK'
