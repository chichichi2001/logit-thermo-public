"""探索：3 つの谷（液体 L ⇄ 長い周期の結晶 A ⇄ 短い周期の結晶 C）は、25 本のモデルでも同じか（2026-10-30 18:0x、CPU、判定しない）。
scripts/a2_explore_onset.py・scripts/a2_explore_hidden.py・scripts/a2_explore_three.py を実行したのと同じ名前空間で実行する。
材料：A2-2・A2-5・A2-6 の生成（探索 8.12 と同じ 25 本、温度 0.6〜1.5、サンプル 2 × 記事 40）。温度が 0.6 からしかないので、L⇄A の融点は
0.6 以上の温度だけで出した「⁺」の数（探索 A2-11 の T_m⁺ と同じく、本当の値より高めに出る）。
本ごとに：温度 0.6・0.7 での L・A・C の窓の割合、温度 0.6 の移り変わりの回数（L→A、L→C、A→C、C→A、A→L、C→L）、
C から出るうちの C→L の割合、C に入るうちの L→C の割合、L⇄A と A⇄C の ΔH と融点（⁺）。
"""
import numpy as np

TU_RUNS = {
    'A2_2': ['pythia-1b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-inst', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b', 'smollm2-1.7b-inst'],
    'A2_5': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
             'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000'],
    'A2_6': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
             'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'],
}
TU_TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]


def one_run(toks):
    cnt = counts_by_doc(toks)                     # (15, nD, 9)
    tot = cnt.sum(1)
    keys = ['LA', 'AL', 'AC', 'CA', 'LC', 'CL']
    row = {}
    for T in (0.6, 0.7):
        k = TU_TEMPS.index(T)
        occ = tot[k, :3] / tot[k, :3].sum()
        row[f'frac_{T}'] = dict(L=float(occ[0]), A=float(occ[1]), C=float(occ[2]))
    k6 = 0
    tr = {x: int(tot[k6, 3 + i]) for i, x in enumerate(keys)}
    row['trans_0.6'] = tr
    row['C_exit_to_L'] = tr['CL'] / (tr['CL'] + tr['CA']) if (tr['CL'] + tr['CA']) else None
    row['C_entry_from_L'] = tr['LC'] / (tr['LC'] + tr['AC']) if (tr['LC'] + tr['AC']) else None
    la = vh_step(tot, TU_TEMPS, 'LA', 'AL', 3, 4); ac = vh_step(tot, TU_TEMPS, 'AC', 'CA', 5, 6)
    row['LA'] = {k: la.get(k) for k in ('ok', 'dH', 'Tm')}; row['LA']['temps'] = [TU_TEMPS[i] for i in la['idx']]
    row['AC'] = {k: ac.get(k) for k in ('ok', 'dH', 'Tm')}; row['AC']['temps'] = [TU_TEMPS[i] for i in ac['idx']]
    return row


def three_univ(base):
    res = {}
    for folder, names in TU_RUNS.items():
        for n in names:
            Z = np.load(f'{base}/{folder}/{n}.npz')
            assert [float(t) for t in Z['temps']] == TU_TEMPS
            res[f'{folder}/{n}'] = one_run(Z['toks'])
    return res


def self_test_tu():
    rng = np.random.default_rng(0)
    toks = rng.integers(100, 30000, (15, 2, 4, 512))
    # 1 本目の温度 0.6：途中から 7 の周期の短い結晶
    toks[0, 0, 0, 200:] = np.tile([5, 6, 7, 8, 9, 10, 11], 60)[:312]
    r = one_run(toks)
    assert r['frac_0.6']['C'] > 0 and r['trans_0.6']['CL'] == 0
    return 'OK'
