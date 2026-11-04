"""探索：結晶の格子定数（くり返しの周期）と、助走の「位置のせい」の分（2026-10-29 18:5x）。確かめには使わない（予想なし）。CPU。
scripts/a2_explore_onset.py を実行したのと同じ名前空間で実行する（ON_RUNS・ON_W・ON_TH・copy_marks・onset_of を使う）。生成もモデルの読み込みもしない。

発想：オーナーの（オーナーの承認・発言。原文は非公開の記録にある）（2026-10-29 18:20）と、
あるチャットアプリ の 3 つの崩れ方（チロシンのつぶ・単結晶・ポリエチレン。2026-10-29 18:24）。

(1) 周期：結晶が始まった本ごとに、窓の頭（始まりの歩 − 31）から PD_SEG 歩を切り出し、ずらし幅 p（1〜PD_PMAX）ごとに
    「p 歩前と同じトークン」の割合 m(p) を数える。m の最大が PD_MMIN 以上なら、最大の 0.9 倍以上になる最小の p を周期とする
    （周期 p の結晶は 2p・3p でも合うので、最小をとる）。m(周期) を「そろい具合」とする（単結晶なら 1 に近く、
    枠の言い回しが毎回少しずつ変わるポリエチレン型なら低め）。
(2) 位置のせい：結晶にならなかった本の、位置（32 歩ずつ）ごとの写しの印の平均。結晶になった本の、始まりにそろえた印の平均と、
    同じ本の同じ絶対位置での「ならなかった本の平均」とを並べる（ならなかった本が 5 本以上ある温度だけ）。
"""
import numpy as np

PD_SEG, PD_PMAX, PD_MMIN = 128, 64, 0.5
PD_BINS = [(1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 31), (32, 64)]
PD_POSBIN = 32
PD_MIN_NEG = 5
PD_PRE, PD_POST, PD_ABIN = 160, 32, 8


def period_of(seg):
    """→ (周期 p、そろい具合 m(p))。合わなければ (−1, 最大の m)。"""
    s = np.asarray(seg)
    n = len(s)
    ms = np.full(PD_PMAX + 1, -1.0)
    for p in range(1, min(PD_PMAX, n - 8) + 1):
        ms[p] = float(np.mean(s[p:] == s[:-p]))
    best = ms.max()
    if best < PD_MMIN:
        return -1, float(best)
    p = int(np.flatnonzero(ms >= 0.9 * best)[0])
    return p, float(ms[p])


def period_stats(toks):
    """toks：(nT, S, nD, L) → 温度ごとの周期のまとめ、位置ごとの基準、そろえた印（生と基準）。"""
    nT, S, nD, L = toks.shape
    nb_pos = L // PD_POSBIN
    nb_al = (PD_PRE + PD_POST) // PD_ABIN
    per_T = []
    al_raw, al_base, al_n = np.zeros(nb_al), np.zeros(nb_al), 0
    for k in range(nT):
        ps, mm = [], []
        C, on = [], []
        for s in range(S):
            for j in range(nD):
                c = copy_marks(toks[k, s, j]); o = onset_of(c)
                C.append(c); on.append(o)
                if o >= 0:
                    a = o - ON_W + 1
                    seg = toks[k, s, j, a:a + PD_SEG]
                    if len(seg) >= 48:                    # 始まりが終わり近くで短すぎる本は数えない
                        p, m = period_of(seg)
                        ps.append(p); mm.append(m)
        C, on = np.array(C), np.array(on)
        row = dict(n_cryst=len(ps))
        if ps:
            pa = np.array(ps)
            ok = pa > 0
            row.update(n_period=int(ok.sum()),
                       hist=[int(((pa >= lo) & (pa <= hi)).sum()) for lo, hi in PD_BINS] + [int((~ok).sum())],
                       med_p=float(np.median(pa[ok])) if ok.any() else None,
                       mean_logp=round(float(np.mean(np.log(pa[ok]))), 4) if ok.any() else None,
                       med_m=round(float(np.median(np.array(mm)[ok])), 3) if ok.any() else None)
        neg = on < 0
        if neg.sum() >= PD_MIN_NEG:
            base = C[neg].astype(float)[:, :nb_pos * PD_POSBIN].reshape(neg.sum(), nb_pos, PD_POSBIN).mean(axis=(0, 2))
            row['base_pos'] = base.round(4).tolist()
            base_full = np.repeat(base, PD_POSBIN)
            for i in np.flatnonzero(~neg):
                oj = on[i]
                if oj >= PD_PRE and oj + PD_POST + 1 <= nb_pos * PD_POSBIN:
                    sl = slice(oj - PD_PRE + 1, oj + PD_POST + 1)
                    al_raw += C[i, sl].astype(float).reshape(nb_al, PD_ABIN).mean(1)
                    al_base += base_full[sl].reshape(nb_al, PD_ABIN).mean(1)
                    al_n += 1
        per_T.append(row)
    al = None
    if al_n:
        al = dict(raw=(al_raw / al_n).round(4).tolist(), base=(al_base / al_n).round(4).tolist(), n=al_n)
    return per_T, al


def run_period(base):
    res = {'bins': [list(b) for b in PD_BINS] + ['none'], 'runs': {}}
    for folder, names in ON_RUNS.items():
        for name in names:
            Z = np.load(f'{base}/{folder}/{name}.npz')
            per_T, al = period_stats(Z['toks'])
            res['runs'][f'{folder}/{name}'] = dict(per_T=per_T, aligned=al)
            res.setdefault('temps', [float(t) for t in Z['temps']])
    return res


def self_test_period():
    rng = np.random.default_rng(0)
    assert period_of(np.tile([5, 6, 7, 8, 9, 10, 11], 20))[0] == 7
    assert period_of(np.tile([5], 100))[0] == 1
    assert period_of(rng.integers(100, 30000, 128))[0] == -1
    # ポリエチレン型：単位 6 のうち 1 つが毎回変わる → 周期 6、そろい具合は 5/6 くらい
    seq = np.concatenate([[1, 2, 3, 4, 5, int(rng.integers(100, 30000))] for _ in range(22)])
    p, m = period_of(seq)
    assert p == 6 and 0.75 < m < 0.9, (p, m)
    # 全体：結晶は周期 5、始まらない本は乱数 → 周期のヒストグラムの 4〜7 に入る。位置ごとの基準は 0 に近い
    L, nD = 512, 20
    toks = np.zeros((1, 2, nD, L), np.int64)
    for j in range(nD):
        toks[0, 0, j] = np.concatenate([rng.integers(100, 30000, 200 + 5 * j), np.tile([5, 6, 7, 8, 9], L)[:L - 200 - 5 * j]])
        toks[0, 1, j] = rng.integers(100, 30000, L)
    per_T, al = period_stats(toks)
    r = per_T[0]
    assert r['n_cryst'] == nD and r['hist'][3] == nD and r['med_p'] == 5, r
    assert max(r['base_pos']) < 0.01 and al['n'] == nD and al['raw'][-1] > 0.9 and max(al['base']) < 0.01, (r, al)
    return 'OK'
