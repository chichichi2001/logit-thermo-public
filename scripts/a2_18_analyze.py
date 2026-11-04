"""A2-18 の分析 v0.2：罰をかけていない（h ＝ 0）出力だけから、罰をかけたときの相の釣り合いの動きを予言できるか（2026-11-01 08:xx JST、データを集める前に書いた。CPU）。
A2-16 と同じ名前空間（scripts/a2_16_run.py の load16ns）に scripts/a2_18_predict.py を足して実行する。数え（窓のものさし）は A2-16 と同じ。
v0.2（新鮮な目のレビューを受けて）：融点どうしを別の温度の格子で比べるのをやめ、同じ温度で、対照（h ＝ 0）の腕と罰の腕の ln K を比べる。

量（モデルごと、3 つの腕は同じ 7 つの温度・同じ記事）：
  ln K(T, h) ＝ ln(凍る回 ÷ 液体の歩) − ln(溶ける回 ÷ 結晶の歩)（窓のものさし、記事とサンプルを足した数えから）。
  測った動き Δ(T) ＝ ln K(T, h) − ln K(T, 0)。
  予言 P(T) ＝ h × g(T)。g(T) は対照の腕だけから、恒等式（凍る・溶けるの回数と Σ（選んだか − q）の共分散、A2-16 の records_dlnK と同じ式）で出す。
  使う温度：対照と罰の腕の両方で、凍る回・溶ける回がそれぞれ 10 以上の温度（点で決め、選び直しでも同じ）。
  比 ρ ＝ Σ_T Δ(T) ÷ Σ_T P(T)（使う温度で足す）。予言どおりなら 1。
全体：7 つのモデルの ρ の中央値。区間：記事の選び直し 1,000 回（種 20261181。7 つのモデルと 3 つの腕に同じ選び方。モデルごとに ρ を出し直して中央値）。
  使う温度が 2 つ未満のモデルは、そのモデルの ρ を出さない（中央値は残りで。出ないモデルが 3 つ以上なら「はっきりしない」）。選び直しで中央値が出ない回が 1 割を超えたら「はっきりしない」。
問い：
  XQ1（h ＝ h*、|h| ＝ 0.37〜0.79）・XQ2（h ＝ 2h*、|h| ＝ 0.75〜1.57）：
    「予言どおり（2 割の内）」：中央値の区間が 0.8〜1.25 に収まる。
    「予言より大きく効く」：区間の下 ＞ 1.25。
    「予言より小さく効く」：区間の上 ＜ 0.8。
    「逆向き」：区間の上 ＜ 0。
    それ以外は「はっきりしない」。
記録：モデルごとの ρ と区間と言葉、温度ごとの Δ と P、3 つの腕の融点（同じ格子での当てはめ）、罰の腕の融点が狙った T* に来たか（ln K(T*, h*) と T_m(h*) − T*）、
  対抗の予言（平衡のクラペイロン）での r_C、±0.3 の腕から単純に伸ばした比べの相手（G-pre の測った ∂ln K/∂h を格子に重なる温度で使う）。
"""
import os, json
import numpy as np

A18_BOOT, A18_SEED = 1000, 20261181
A18_FAIL = 0.10
A18_BAND = (0.8, 1.25)
A18_MIN_EV = 10
A18_MIN_T = 2
A18_ARMS = ('c0', 'h1', 'h2')


def _fit(tot, temps, use=None):
    """温度の並びを temps にして fit_vh（fit_vh が実際に読む名前空間の A8_TU を置きかえ、終わったら戻す）。"""
    G = fit_vh.__globals__
    keep = G['A8_TU']
    try:
        G['A8_TU'] = [float(t) for t in temps]
        return fit_vh(tot[:, :10], temps_use=use)
    finally:
        G['A8_TU'] = keep


def seq_events(toks, pick, q):
    """1 つの腕 (7, S, nD, L) → 1 本の文ごとの [液体の歩, 凍る回, 結晶の歩, 溶ける回] (7, S, nD, 4) と σ ＝ Σ（選んだか − q）(7, S, nD)。"""
    nT, S, nD, L = toks.shape
    ev = np.zeros((nT, S, nD, 4))
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                st, _ = phases(copy_marks(toks[k, s, j]))
                a, b = st[:-1], st[1:]
                ev[k, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
    sig = (pick.astype(float) - q.astype(float)).sum(-1)
    return ev, sig


def lnK_from(ev, w):
    """ev：(7, S, nD, 4)、w：記事の重み (nD) → 温度ごとの ln K と、凍る回・溶ける回。"""
    c = np.einsum('ksjc,j->kc', ev, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(c[:, 1] / c[:, 0]) - np.log(c[:, 3] / c[:, 2]), c[:, 1], c[:, 3]


def g_identity(ev, sig, temps, w):
    """恒等式の ∂ln K/∂h（h ＝ 0 の腕、記事の重み w つき）→ 温度ごと。A2-16 の records_dlnK と同じ式（w ＝ 1 で一致）。"""
    out = []
    S = ev.shape[1]
    ww = np.repeat(w[None, :], S, 0).ravel()                              # (S × nD)
    for k, T in enumerate(temps):
        sg = sig[k].ravel(); W = ww.sum()
        if W <= 0:
            out.append(float('nan')); continue
        ms = (ww * sg).sum() / W
        d = []
        for i in range(4):
            e = ev[k, :, :, i].ravel(); me = (ww * e).sum() / W
            d.append(float((ww * (e - me) * (sg - ms)).sum() / W / (T * me)) if me > 0 else float('nan'))
        out.append((d[1] - d[0]) - (d[3] - d[2]))
    return np.array(out)


def rho_model(E, h, temps, w, use=None):
    """E：{'c0': (ev, sig), 'h1': …} → (ρ, 使う温度, Δ, P)。"""
    l0, f0, m0 = lnK_from(E['c0'][0], w)
    l1, f1, m1 = lnK_from(E[h[0]][0], w)
    if use is None:
        use = [k for k in range(len(temps)) if min(f0[k], m0[k], f1[k], m1[k]) >= A18_MIN_EV]
    if len(use) < A18_MIN_T:
        return float('nan'), use, None, None
    g = g_identity(E['c0'][0], E['c0'][1], temps, w)
    D = l1 - l0; P = h[1] * g
    u = np.array(use, int)
    if not (np.all(np.isfinite(D[u])) and np.all(np.isfinite(P[u]))) or P[u].sum() == 0:
        return float('nan'), use, D, P
    return float(D[u].sum() / P[u].sum()), use, D, P


def word18(ci, fail):
    if fail > A18_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if ci[1] < 0:
        return '逆向き'
    if A18_BAND[0] <= ci[0] and ci[1] <= A18_BAND[1]:
        return '予言どおり（2 割の内）'
    if ci[0] > A18_BAND[1]:
        return '予言より大きく効く'
    if ci[1] < A18_BAND[0]:
        return '予言より小さく効く'
    return 'はっきりしない'


def _median_or_nan(v):
    v = np.array(v, float)
    ok = np.isfinite(v)
    return float(np.median(v[ok])) if ok.sum() >= len(v) - 2 else float('nan')     # 出ないモデルが 3 つ以上なら出さない


def analyze_all(EE, hs, grids, nD, seed=A18_SEED, boot=A18_BOOT):
    """EE：{モデル: {腕: (ev, sig)}}、hs：{モデル: {'h1': h*, 'h2': 2h*}}、grids：{モデル: 温度}。"""
    names = list(EE)
    res = {'models': {n: {} for n in names}, 'verdict': {}}
    one = np.ones(nD)
    uses = {}
    for arm in ('h1', 'h2'):
        pts = []
        for n in names:
            r, u, D, P = rho_model(EE[n], (arm, hs[n][arm]), grids[n], one)
            uses[(n, arm)] = u
            res['models'][n][arm] = dict(rho=r, use=u, h=hs[n][arm], D=None if D is None else D.tolist(), P=None if P is None else P.tolist())
            pts.append(r)
        res['verdict'][arm] = dict(median=_median_or_nan(pts))
    rng = np.random.default_rng(seed)
    B = {(n, a): [] for n in names for a in ('h1', 'h2')}
    Bm = {a: [] for a in ('h1', 'h2')}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        for arm in ('h1', 'h2'):
            v = []
            for n in names:
                r = rho_model(EE[n], (arm, hs[n][arm]), grids[n], w, uses[(n, arm)])[0] if len(uses[(n, arm)]) >= A18_MIN_T else float('nan')
                B[(n, arm)].append(r); v.append(r)
            Bm[arm].append(_median_or_nan(v))
    for arm in ('h1', 'h2'):
        for n in names:
            v = np.array(B[(n, arm)], float)
            fl = float(np.mean(~np.isfinite(v)))
            ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
            res['models'][n][arm].update(ci=ci, fail=fl, word=word18(ci, fl))
        v = np.array(Bm[arm], float)
        fl = float(np.mean(~np.isfinite(v)))
        ci = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
        res['verdict'][arm].update(ci=ci, fail=fl, word=word18(ci, fl), per_model={n: res['models'][n][arm]['word'] for n in names})
    return res


def load_arm(path, temps_expect, h_expect):
    Z = np.load(path)
    assert [round(float(t), 4) for t in Z['temps']] == [round(float(t), 4) for t in temps_expect], (path,)
    assert abs(float(Z['h']) - h_expect) < 1e-9, (path, float(Z['h']))
    toks, pick, q, q0, U = Z['toks'], Z['pick'], Z['q'], Z['q0'], Z['U']
    assert toks.shape[0] == 7 and toks.shape[3] == 512 and not (pick & ~Z['cand']).any()
    if abs(h_expect) < 1e-12:
        assert float(np.abs(q.astype(float) - q0.astype(float)).max()) == 0.0, (path, 'h ＝ 0 で q と q0 が違う')
    return dict(toks=toks, pick=pick, q=q, U=U), np.asarray(Z['use']).tolist()


def analyze18(base18, base_gpre=None, hstar_json='results/A2_18/hstar.json', jobs=None, seed=A18_SEED, boot=A18_BOOT):
    P = json.load(open(hstar_json))
    J = {(n, a): (h, t) for n, a, h, t in (jobs if jobs is not None else A18_JOBS)}
    EE, hs, grids, recs, nD = {}, {}, {}, {}, None
    for name in P:
        if not all(os.path.exists(f'{base18}/{name}@{a}.npz') for a in A18_ARMS):
            continue
        arms, us = {}, []
        for a in A18_ARMS:
            h, t = J[(name, a)]
            arms[a], u = load_arm(f'{base18}/{name}@{a}.npz', t, h); us.append(u)
        assert us[0] == us[1] == us[2], name
        nD = len(us[0]) if nD is None else nD
        assert len(us[0]) == nD, ('記事の数がモデルで違う', name)
        EE[name] = {a: seq_events(arms[a]['toks'], arms[a]['pick'], arms[a]['q']) for a in A18_ARMS}
        hs[name] = {'h1': J[(name, 'h1')][0], 'h2': J[(name, 'h2')][0]}
        grids[name] = J[(name, 'c0')][1]
        # 記録：同じ格子での 3 つの腕の融点、罰の腕が T* に来たか
        tg = P[name]['targets']['-0.10']
        rc = dict(Tstar=tg['Tstar'], r_clapeyron=tg['r_clapeyron'], Tm0_gpre=P[name]['Tm0'])
        for a in A18_ARMS:
            cw, _ = arm_counts(arms[a]['toks'], arms[a]['U'], arms[a]['pick'])
            f = _fit(cw.sum(1), grids[name])
            rc[f'Tm_{a}'] = float(f['Tm']) if f.get('ok') and np.isfinite(f.get('Tm', np.nan)) else float('nan')
        l1, _, _ = lnK_from(EE[name]['h1'][0], np.ones(nD))
        rc['lnK_h1_at_center'] = float(l1[3])                              # 格子の真ん中 ＝ T*（0.01 に丸めた）
        rc['Tm_h1_minus_Tstar'] = rc['Tm_h1'] - tg['Tstar']
        recs[name] = rc
    names = [n for n in P]
    missing = [n for n in names if n not in EE]
    res = analyze_all(EE, hs, grids, nD, seed, boot) if EE else {'models': {}, 'verdict': {}}
    for n in missing:                                                     # 3 つの腕がそろわないモデルは「はっきりしない」（中央値の分母には入らない）
        res['models'][n] = {a: dict(rho=float('nan'), word='はっきりしない', missing=True) for a in ('h1', 'h2')}
    if len(missing) >= 3:
        for a in ('h1', 'h2'):
            res['verdict'][a] = dict(median=float('nan'), ci=[float('nan')] * 2, word='はっきりしない', note='そろったモデルが 4 つ以下')
    res['records'] = recs; res['missing'] = missing
    # 記録：±0.3 の腕から単純に伸ばした比べの相手（G-pre の測った ∂ln K/∂h を、格子に重なる温度で h 倍）
    if base_gpre and os.path.exists(f'{base_gpre}/{names[0]}@hm03.npz'):
        A16 = json.load(open('results/A2_16/a2_16_summary.json'))['models']
        for n in EE:
            bt = {round(b['T'], 2): b['dlnK_meas'] for b in A16[n]['records']['by_T']}
            res['records'][n]['baseline_dlnK_meas_by_T'] = {str(t): bt.get(round(t, 2)) for t in grids[n]}
    return res


# ---------- 作り物（自己テスト） ----------
def synth_events(temps, Tm, dH=3.2, nD=40, S=4, L=512, pl=0.15, seed=1):
    """本当の融点 Tm の 2 つの状態の鎖（A2-16 の synth16 と同じ作り）→ seq_events と同じ形 (ev, sig)。sig は 0（恒等式は自己テストでは差しかえる）。"""
    rng = np.random.default_rng(seed)
    nT = len(temps)
    ev = np.zeros((nT, S, nD, 4))
    for k, T in enumerate(temps):
        lnK = dH / T - dH / Tm
        kf = 0.01 * np.exp(lnK / 2); km = 0.01 * np.exp(-lnK / 2)
        st = np.zeros((S, nD), bool); ph = np.zeros((S, nD), int); ptr = np.zeros((S, nD), int)
        blk = rng.integers(100, 30000, (S, nD, 7))
        seq = np.zeros((S, nD, L), np.int64)
        for t in range(L):
            if t > 0:
                r = rng.random((S, nD))
                st = np.where(st, r >= km, r < kf)
            ph = np.where(st, ph + 1, 0)
            cp = (~st) & (rng.random((S, nD)) < pl) & (t > 8)
            src = np.take_along_axis(seq, np.clip(ptr, 0, L - 1)[..., None], -1)[..., 0]
            seq[..., t] = np.where(st, np.take_along_axis(blk, (ph % 7)[..., None], -1)[..., 0], np.where(cp, src, rng.integers(100, 30000, (S, nD))))
            ptr = np.where(cp, ptr + 1, rng.integers(0, max(t, 1), (S, nD)))
        for s in range(S):
            for j in range(nD):
                stt, _ = phases(copy_marks(seq[s, j]))
                a, b = stt[:-1], stt[1:]
                ev[k, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
    return ev, np.zeros((nT, S, nD))


def self_test18(boot=40):
    """作り物：ln K を温度によらず Δ ＝ ρ_真 × h × g_真 だけ動かした鎖（融点で動かす）。恒等式の g は g_真 に差しかえる。
    ρ_真 ＝ 1 → 「予言どおり」、0.5 → 「小さく効く」、1.6 → 「大きく効く」。あわせて、重みつきの恒等式が重みなしの式（A2-16）と一致すること。"""
    global g_identity
    out = {}
    temps = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    dH, Tm, gtrue, h = 3.2, 0.6, 2.0, -0.4
    keep = g_identity
    try:
        g_identity = lambda ev, sig, tt, w: np.full(len(tt), gtrue)
        for rho, want in ((1.0, '予言どおり（2 割の内）'), (0.5, '予言より小さく効く'), (1.6, '予言より大きく効く')):
            EE, hs, gr = {}, {}, {}
            for m in range(7):
                Tm_h = 1.0 / (1.0 / Tm - rho * h * gtrue / dH)                 # ln K を rho h g だけ動かす融点
                E0 = synth_events(temps, Tm, dH, seed=100 + m)
                E1 = synth_events(temps, Tm_h, dH, seed=200 + m)
                Tm_2 = 1.0 / (1.0 / Tm - rho * 2 * h * gtrue / dH)
                E2 = synth_events(temps, Tm_2, dH, seed=300 + m)
                EE[f'm{m}'] = {'c0': E0, 'h1': E1, 'h2': E2}; hs[f'm{m}'] = {'h1': h, 'h2': 2 * h}; gr[f'm{m}'] = temps
            r = analyze_all(EE, hs, gr, 40, boot=boot)
            out[f'rho{rho}'] = {a: dict(median=r['verdict'][a]['median'], ci=r['verdict'][a]['ci'], word=r['verdict'][a]['word']) for a in ('h1', 'h2')}
            assert r['verdict']['h1']['word'] == want, (rho, r['verdict']['h1'])
    finally:
        g_identity = keep
    # 重みつきの恒等式 ＝ A2-16 の式（重み 1）
    rng = np.random.default_rng(5)
    ev = rng.integers(1, 50, (7, 4, 10, 4)).astype(float); sig = rng.normal(0, 3, (7, 4, 10))
    gw = g_identity(ev, sig, G16_TEMPS, np.ones(10))
    for k, T in enumerate(G16_TEMPS):
        sg = sig[k].ravel()
        d = [float(np.mean((ev[k, :, :, i].ravel() - ev[k, :, :, i].mean()) * (sg - sg.mean())) / (T * ev[k, :, :, i].mean())) for i in range(4)]
        assert abs(gw[k] - ((d[1] - d[0]) - (d[3] - d[2]))) < 1e-12
    assert word18([0.85, 1.2], 0) == '予言どおり（2 割の内）' and word18([0.7, 1.2], 0) == 'はっきりしない' and word18([-0.5, -0.1], 0) == '逆向き'
    assert word18([1.3, 2.0], 0) == '予言より大きく効く' and word18([0.2, 0.7], 0) == '予言より小さく効く' and word18([0.85, 1.2], 0.2) == 'はっきりしない'
    assert [round(t, 4) for t in fit_vh.__globals__['A8_TU']] == G16_TEMPS
    return out
