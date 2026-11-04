"""探索：周期のない結晶 A の中身をモデルごとに比べる（なぜ Pythia の A は塩で溶けにくいか、2026-10-30 18:0x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）と scripts/a2_explore_hidden.py（window_feats・label）を実行したのと同じ名前空間で実行する。
A の窓（64 歩）ごとに：写しの元までの距離（同じ 4 トークンの組が最後に出た位置から何歩前か）の中央値、写しが続く長さ（連続した印の長さ）の中央値、
窓の中の違うトークンの割合、ℓ（その歩の裾の取り分、生成のときに記録したもの）の平均。A の続いた長さ（窓の数）と、A の終わり方（L へか C へか）。
偽薬 30% の腕で、A の窓の中で 1 位が禁止された歩の割合。
"""
import numpy as np


def copy_lags(seq):
    """→ 各歩の写しの元までの距離（印がなければ 0）。"""
    s = np.asarray(seq, dtype=np.int64)
    last, lag = {}, np.zeros(len(s), np.int32)
    for t in range(3, len(s)):
        key = tuple(s[t - 3:t + 1])
        if key in last:
            lag[t] = t - last[key]
        last[key] = t
    return lag


def runs(c):
    out, n = [], 0
    for v in c:
        if v:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def amorph(toks, ell=None, banned=None, win=64, step=16):
    """toks：(本数, L) → A と L の窓の特徴の要約。ell・banned：同じ形（任意）。"""
    feat = {k: {'A': [], 'L': [], 'C': []} for k in ('lag', 'run', 'uniq', 'ell', 'ban')}
    dwell, ends = [], {'L': 0, 'C': 0, 'end': 0}
    for i, seq in enumerate(toks):
        c = copy_marks(seq); lag = copy_lags(seq); f = window_feats(seq)
        labs, cur = [], 'L'
        for r, P in f:
            x = label(r, P); cur = x if x != '-' else cur; labs.append(cur)
        for w, x in enumerate(labs):
            a = w * step; b = a + win
            cc = c[a:b]; lg = lag[a:b][cc == 1]
            feat['lag'][x].append(float(np.median(lg)) if len(lg) else np.nan)
            rr = runs(cc); feat['run'][x].append(float(np.median(rr)) if rr else np.nan)
            feat['uniq'][x].append(len(set(seq[a:b])) / win)
            if ell is not None:
                feat['ell'][x].append(float(np.mean(ell[i][a:b])))
            if banned is not None:
                feat['ban'][x].append(float(np.mean(banned[i][a:b])))
        # A の続いた長さと終わり方
        w = 0
        while w < len(labs):
            if labs[w] == 'A':
                st = w
                while w < len(labs) and labs[w] == 'A':
                    w += 1
                dwell.append(w - st)
                ends['end' if w == len(labs) else labs[w]] += 1
            else:
                w += 1
    med = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(np.nanmedian(v)) else None
    summ = {k: {s: med(feat[k][s]) for s in 'ALC'} for k in feat}
    summ['A_dwell_median'] = float(np.median(dwell)) if dwell else None
    summ['A_dwell_mean'] = float(np.mean(dwell)) if dwell else None
    summ['A_ends'] = ends
    return summ
