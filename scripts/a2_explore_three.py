"""探索：3 つの谷 L ⇄ A ⇄ C の段ごとの速さ・ファントホッフの ΔH・塩の効き（2026-10-30 17:5x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）と scripts/a2_explore_hidden.py（window_feats・label）を実行したのと同じ名前空間で実行する。
窓（64 歩、16 歩おき）のラベルの列で、「ー」（どちらとも言えない）は直前のラベルのままとみなす（履歴つき）。
速さ k_XY ＝ X→Y の回数 ÷ X にいた窓の数（1 窓 ＝ 16 歩）。段ごとの平衡定数 K_LA ＝ k_LA ÷ k_AL、K_AC ＝ k_AC ÷ k_CA。
ファントホッフ：ln K を 1/T に重みつきで当てはめた傾き（ナット、1 回の移り変わりあたり）。回数が両向きそれぞれ 8 回以上の温度だけ。
"""
import numpy as np

TH_MIN = 8
TH_BOOT, TH_SEED = 300, 20261122


def seq_counts(seq):
    """→ dict：各状態にいた窓の数、各移り変わりの回数。"""
    f = window_feats(seq)
    labs, cur = [], 'L'
    for r, P in f:
        x = label(r, P)
        if x != '-':
            cur = x
        labs.append(cur)
    occ = {s: 0 for s in 'LAC'}; tr = {}
    for x, y in zip(labs[:-1], labs[1:]):
        occ[x] += 1
        if x != y:
            tr[x + y] = tr.get(x + y, 0) + 1
    return occ, tr


def counts_by_doc(toks):
    """toks：(nT, S, nD, L) → (nT, nD, 3 + 6)：[L, A, C にいた窓の数, LA, AL, AC, CA, LC, CL の回数]。"""
    nT, S, nD, _ = toks.shape
    keys = ['LA', 'AL', 'AC', 'CA', 'LC', 'CL']
    out = np.zeros((nT, nD, 9))
    for k in range(nT):
        for j in range(nD):
            for s in range(S):
                occ, tr = seq_counts(toks[k, s, j])
                out[k, j, :3] += [occ['L'], occ['A'], occ['C']]
                out[k, j, 3:] += [tr.get(x, 0) for x in keys]
    return out


def rates(tot):
    """tot：(nT, 9) → 速さ（1 窓あたり）。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return dict(LA=tot[:, 3] / tot[:, 0], AL=tot[:, 4] / tot[:, 1], AC=tot[:, 5] / tot[:, 1], CA=tot[:, 6] / tot[:, 2],
                    LC=tot[:, 7] / tot[:, 0], CL=tot[:, 8] / tot[:, 2])


def vh_step(tot, temps, fwd, bwd, ifw, ibw):
    k = rates(tot)
    ok = (tot[:, ifw] >= TH_MIN) & (tot[:, ibw] >= TH_MIN)
    idx = np.flatnonzero(ok)
    if len(idx) < 3:
        return dict(ok=False, idx=idx.tolist())
    y = np.log(k[fwd][idx] / k[bwd][idx]); x = 1 / np.asarray(temps)[idx]
    w = 1 / (1 / tot[idx, ifw] + 1 / tot[idx, ibw])
    X = np.stack([np.ones_like(x), x], 1)
    a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    return dict(ok=True, dH=float(b), dS=float(-a), Tm=float(-b / a) if (b > 0 and a < 0) else float('nan'), idx=idx.tolist(), lnK=y.tolist())


def three_state(cnt, temps, seed=TH_SEED):
    tot = cnt.sum(1)
    res = dict(LA=vh_step(tot, temps, 'LA', 'AL', 3, 4), AC=vh_step(tot, temps, 'AC', 'CA', 5, 6),
               rates={k: v.tolist() for k, v in rates(tot).items()}, counts=tot.tolist())
    nD = cnt.shape[1]
    rng = np.random.default_rng(seed)
    bs = {'LA': [], 'AC': []}
    for _ in range(TH_BOOT):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        t = np.tensordot(cnt, w, axes=([1], [0]))
        for st, (f, b, i, j) in {'LA': ('LA', 'AL', 3, 4), 'AC': ('AC', 'CA', 5, 6)}.items():
            if not res[st]['ok']:
                continue
            kk = rates(t)
            idx = res[st]['idx']
            with np.errstate(divide='ignore', invalid='ignore'):
                y = np.log(kk[f][idx] / kk[b][idx])
            if not np.all(np.isfinite(y)):
                continue
            x = 1 / np.asarray(temps)[idx]; ww = 1 / (1 / np.maximum(t[idx, i], 1) + 1 / np.maximum(t[idx, j], 1))
            X = np.stack([np.ones_like(x), x], 1)
            bs[st].append(float(np.linalg.solve(X.T @ (ww[:, None] * X), X.T @ (ww * y))[1]))
    for st in bs:
        if res[st]['ok'] and len(bs[st]) >= 20:
            res[st]['dH_ci'] = [float(np.percentile(bs[st], 2.5)), float(np.percentile(bs[st], 97.5))]
    return res


def salt_steps(cnt_none, cnt_arm, use_T):
    """use_T：温度の番号。→ 4 つの速さそれぞれの log(偽薬 ÷ なし) を、温度で平均。"""
    a, n = rates(cnt_arm.sum(1)), rates(cnt_none.sum(1))
    with np.errstate(divide='ignore', invalid='ignore'):
        return {k: float(np.nanmean(np.log(a[k][use_T] / n[k][use_T]))) for k in ('LA', 'AL', 'AC', 'CA')}
