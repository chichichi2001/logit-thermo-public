"""探索：結晶の谷の中に、周期のない結晶（中間の状態）の谷が隠れていないか（2026-10-30 17:4x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）を実行したのと同じ名前空間で実行する。

窓（WIN ＝ 64 歩、16 歩おき）ごとに：
- r ＝ 写しの印（4 トークン）の割合（0〜1）
- P ＝ 周期の強さ ＝ 2〜32 歩のずれ p で、窓の中で「p 歩前と同じトークン」の割合 m(p) の最大（探索 8.13 と同じ m(p)）
ラベル：液体 L（r < 0.25）、周期のない結晶 A（r ≥ 0.5 かつ P < 0.5）、周期のある結晶 C（r ≥ 0.5 かつ P ≥ 0.5）、そのあいだは「ー」。
(r, P) の 2 次元の分布と、同じ本の中で隣り合う窓のラベルの移り変わり（L→A、L→C、A→C …）を数える。
オストワルトの段階則の形なら、C へは L から直接ではなく A を通って入ることが多く、A は C より先に出る。
"""
import numpy as np

HD_WIN, HD_STEP, HD_PMAX = 64, 16, 32


def window_feats(seq):
    c = copy_marks(seq).astype(float)
    s = np.asarray(seq)
    L = len(s)
    out = []
    for a in range(0, L - HD_WIN + 1, HD_STEP):
        w = s[a:a + HD_WIN]
        r = c[a:a + HD_WIN].mean()
        m = [np.mean(w[p:] == w[:-p]) for p in range(2, HD_PMAX + 1)]
        out.append((r, max(m)))
    return np.array(out)


def label(r, P):
    if r < 0.25:
        return 'L'
    if r >= 0.5:
        return 'C' if P >= 0.5 else 'A'
    return '-'


def hidden(toks_list):
    """toks_list：(本数, L) のトークン列の集まり → 2 次元の分布・ラベルの割合・移り変わり。"""
    feats, trans = [], {}
    for seq in toks_list:
        f = window_feats(seq)
        feats.append(f)
        labs = [label(r, P) for r, P in f]
        # '-' を飛ばして、意味のあるラベルの並びで隣り合うものを数える
        labs2 = [x for x in labs if x != '-']
        for x, y in zip(labs2[:-1], labs2[1:]):
            if x != y:
                trans[x + y] = trans.get(x + y, 0) + 1
    F = np.concatenate(feats)
    H, _, _ = np.histogram2d(F[:, 0], F[:, 1], bins=[np.linspace(0, 1, 6), np.linspace(0, 1, 6)])
    labs = [label(r, P) for r, P in F]
    frac = {k: float(np.mean([x == k for x in labs])) for k in 'LAC-'}
    cry = F[F[:, 0] >= 0.5, 1]
    Ph = np.histogram(cry, bins=np.linspace(0, 1, 11))[0]
    return dict(hist2d=(H / H.sum()).round(4).tolist(), frac=frac, trans=trans, P_hist_in_crystal=(Ph / max(1, Ph.sum())).round(3).tolist())
