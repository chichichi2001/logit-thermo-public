"""A2-21 の分析：平衡の読み（1 トークンあたりのクラペイロン g_C ＝ n × Δm_T ÷ T）は、写した数 N で文全体を傾けたときの応答 g_N と同じものか。
ゲート：docs/A2_21_GATE.md。データを開く前に書いた（2026-11-02 15:2x JST）。CPU。
scripts/a2_20_run.py の load20ns() の名前空間（seq_events・arm_counts・fit_vh・A8_TU・G16_MODELS など）で実行する。

1 本の文ごとに N ＝ Σ 選んだか、Q ＝ Σ q、S ＝ N − Q（seq_events の σ と同じ範囲：全部の歩）。
g_X(T) ＝ (d₁ − d₀) − (d₃ − d₂)、d_i ＝ Cov(回数_i, X) ÷ (T × E[回数_i])（g_identity と同じ組み立て、記事の重みつき）。
g_S ＝ g_N − g_Q（厳密）。g_C(T) ＝ n × Δm_T ÷ T（A2-16 の records_dlnK と同じ。n ＝ ΔH_vH ÷ 熱量計の ΔH、Δm_T ＝ 結晶で選んだ割合 − 液体で選んだ割合）。
v0.2（新鮮な目のレビューを入れた）：主な比べは「文全体どうし」。
  ΔH_全体(T) ＝ ∂ln K/∂(1/T)（記事ごとに文全体を e^{−δ ΣU} で傾けたとき、U ＝ 温度 1・場なしの −ln p）＝ −T × g_from(ΣU)。
  向きは fit_vh の ΔH_vH（ln K ＝ a ＋ b/T の b）と同じ。
  g_C全体(T) ＝ (ΔH_全体(T) ÷ 熱量計の ΔH) × Δm_T ÷ T（平衡の 2 状態の読みを、文全体の傾けで作った n で書いたもの）。
  κ全体 ＝ Σ_use g_N ÷ Σ_use g_C全体（判定）。κ混ぜ ＝ Σ_use g_N ÷ Σ_use g_C（g_C の n はファントホッフ、つまり 1 歩ずつ割り直す温度から。記録）。
  r ＝ Σ_use g_Q ÷ Σ_use g_N。use ＝ fit_vh が点で選んだ温度。
"""
import numpy as np

A21_BAND = (0.8, 1.25)
A21_BOOT, A21_SEED = 1000, 20261121
A21_FAIL = 0.10
A21_BIG = ['qwen2.5-7b', 'mistral-7b-v0.3']


def g_from(ev, X, temps, w, within=True, unbiased=True):
    """ev：(nT, S, nD, 4)、X：(nT, S, nD)（1 本ごとの量）、w：記事の重み (nD) → 温度ごとの g_X。
    within ＝ True：共分散は記事の中だけで取る（記事ごとに平均を引き、記事の重みで足す）。記事ごとに文全体を傾ける形（P_h(x | 記事) ∝ P_0(x | 記事) e^{hN/T}）の応答。
      unbiased ＝ True なら記事の中の共分散を S ÷ (S − 1) 倍する（記事あたり S 本の標本の偏りを直す）。
    within ＝ False：記事をまたいでまとめて取る（A2-18 の g_identity と同じ）。"""
    S_ = ev.shape[1]
    W = w.sum() * S_
    out = []
    for k, T in enumerate(temps):
        x = X[k]                                                             # (S, nD)
        if W <= 0:
            out.append(np.nan); continue
        xc = x - x.mean(0, keepdims=True) if within else x - (w[None, :] * x).sum() / W
        fac = S_ / (S_ - 1) if (within and unbiased) else 1.0
        d = []
        for i in range(4):
            e = ev[k, :, :, i]; me = (w[None, :] * e).sum() / W
            ec = e - e.mean(0, keepdims=True) if within else e - me
            d.append(float(fac * (w[None, :] * ec * xc).sum() / W / (T * me)) if me > 0 else np.nan)
        out.append((d[1] - d[0]) - (d[3] - d[2]))
    return np.array(out)


def g_parts(ev, N, Q, temps, w, within=True):
    gN, gQ = g_from(ev, N, temps, w, within), g_from(ev, Q, temps, w, within)
    gS = g_from(ev, N - Q, temps, w, within)
    return gS, gN, gQ


def clap(cw, temps, w, use=None):
    """cw：(nT, nD, 16)（窓の数え）→ n・Δm_T・g_C(T)・使う温度など。fit_vh は名前空間の A8_TU を温度として使うので、ここで一時的に差しかえる。"""
    g = fit_vh.__globals__; old = g['A8_TU']; g['A8_TU'] = [float(t) for t in temps]
    try:
        tot = np.einsum('kjc,j->kc', cw, w)
        f = fit_vh(tot[:, :10], temps_use=use)
    finally:
        g['A8_TU'] = old
    res = dict(ok=False, use=f.get('use_idx', []))
    if not (f.get('ok') and np.isfinite(f.get('Tm', np.nan)) and f['dH_cal'] > 0):
        return res
    with np.errstate(divide='ignore', invalid='ignore'):
        dm_T = tot[:, 11] / tot[:, 7] - tot[:, 10] / tot[:, 5]
    k = np.array(f['use_idx'], int)
    if not np.all(np.isfinite(dm_T[k])):
        return res
    n = float(f['dH_vH'] / f['dH_cal'])
    gC = n * dm_T / np.array(temps, float)
    res.update(ok=True, n=n, dm=float(np.mean(dm_T[k])), dm_T=dm_T.tolist(), gC=gC.tolist(), Tm=float(f['Tm']), dHvH=float(f['dH_vH']), dHcal=float(f['dH_cal']))
    return res


def kappa_point(ev, N, Q, E, cw, temps, w, use=None, within=True):
    """E：(nT, S, nD) 1 本ごとの ΣU。→ κ全体（判定）・κ混ぜ・r など。"""
    c = clap(cw, temps, w, use)
    if not c['ok']:
        return dict(ok=False, use=c['use'])
    k = np.array(c['use'], int)
    T = np.array(temps, float)
    gS, gN, gQ = g_parts(ev, N, Q, temps, w, within)
    dHw = -T * g_from(ev, E, temps, w, within)
    dm_T = np.array(c['dm_T'])
    gC = np.array(c['gC']); gCw = dHw / c['dHcal'] * dm_T / T
    sN, sC, sCw, sQ = gN[k].sum(), gC[k].sum(), gCw[k].sum(), gQ[k].sum()
    ok = all(np.isfinite(v) and v != 0 for v in (sN, sC, sCw))
    nan = float('nan')
    return dict(ok=bool(ok), use=c['use'], kappa=float(sN / sCw) if ok else nan, kappa_mixed=float(sN / sC) if ok else nan,
                r=float(sQ / sN) if ok else nan, n=c['n'], n_whole=float(np.mean(dHw[k]) / c['dHcal']), dm=c['dm'], Tm=c['Tm'], dHcal=c['dHcal'],
                gS=gS.tolist(), gN=gN.tolist(), gQ=gQ.tolist(), gC=gC.tolist(), gC_whole=gCw.tolist(), dH_whole=dHw.tolist())


def word21(ci, fail):
    lo, hi = A21_BAND
    if fail > A21_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if lo <= ci[0] and ci[1] <= hi:
        return 'そろう'
    if ci[1] < lo or ci[0] > hi:
        return 'そろわない'
    return 'はっきりしない'


def analyze21(data, agg, seed=A21_SEED, boot=A21_BOOT):
    """data：{モデル: dict(ev, N, Q, cw, temps)}（記事の番号はモデルをまたいで同じ並び）。agg：'median' か 'mean'。"""
    names = list(data)
    nD = data[names[0]]['ev'].shape[2]
    assert all(data[m]['ev'].shape[2] == nD for m in names)
    one = np.ones(nD)
    pts = {m: kappa_point(d['ev'], d['N'], d['Q'], d['E'], d['cw'], d['temps'], one) for m, d in data.items()}
    for m, d in data.items():                                              # 記録：記事をまたいでまとめた共分散での κ（点だけ）
        pp = kappa_point(d['ev'], d['N'], d['Q'], d['E'], d['cw'], d['temps'], one, within=False)
        pts[m]['pooled'] = {k: pp.get(k) for k in ('kappa', 'kappa_mixed', 'r', 'gS')}
    f = np.median if agg == 'median' else np.mean
    point = float(f([pts[m]['kappa'] for m in names])) if all(pts[m]['ok'] for m in names) else np.nan
    rng = np.random.default_rng(seed)
    bs, bm = [], {m: [] for m in names}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        ks = []
        for m in names:
            d = data[m]
            p = kappa_point(d['ev'], d['N'], d['Q'], d['E'], d['cw'], d['temps'], w, use=pts[m]['use']) if pts[m]['ok'] else dict(ok=False)
            v = p['kappa'] if p.get('ok') else np.nan
            ks.append(v); bm[m].append(v)
        bs.append(f(ks) if np.all(np.isfinite(ks)) else np.nan)
    bs = np.array(bs); fail = float(np.mean(~np.isfinite(bs)))
    ci = np.nanpercentile(bs, [2.5, 97.5]).tolist() if np.isfinite(bs).any() else [np.nan, np.nan]
    for m in names:
        b = np.array(bm[m]); pts[m]['ci'] = np.nanpercentile(b, [2.5, 97.5]).tolist() if np.isfinite(b).any() else [np.nan, np.nan]
        pts[m]['fail'] = float(np.mean(~np.isfinite(b)))
    return dict(models=pts, agg=agg, point=point, ci=ci, fail=fail, word=word21(ci, fail))


# ---------- 自己テスト ----------
def _toy_sample_U(S, L, V, T, beta, base, rng):
    """toy_sample と同じ鎖。あわせて U ＝ 温度 1 の −ln p（選んだトークン）を返す。"""
    pick = np.zeros((S, L), bool); q = np.zeros((S, L)); U = np.zeros((S, L))
    prev = rng.integers(0, V, S)
    for t in range(L):
        z = np.repeat(base[t][None, :], S, 0); z[np.arange(S), prev] += beta
        z1 = z - z.max(1, keepdims=True); p1 = np.exp(z1); p1 /= p1.sum(1, keepdims=True)
        zT = z / T; zT -= zT.max(1, keepdims=True); p = np.exp(zT); p /= p.sum(1, keepdims=True)
        q[:, t] = p[np.arange(S), prev]
        x = (rng.random(S)[:, None] < p.cumsum(1)).argmax(1)
        U[:, t] = -np.log(p1[np.arange(S), x]); pick[:, t] = x == prev; prev = x
    return pick, q, U


# ---------- 自己テスト ----------
def self_test21(S=200, nD=30, L=100, V=10, T=0.7, beta=1.5, seed=5):
    rng = np.random.default_rng(seed)
    ev = np.zeros((1, S, nD, 4)); Nn = np.zeros((1, S, nD)); Qq = np.zeros((1, S, nD)); Ee = np.zeros((1, S, nD))
    for j in range(nD):                                                     # 記事ごとに違う base（記事のばらつきを入れる）と違う beta
        base = rng.normal(0, 1.0, (L, V)); b = beta + rng.normal(0, 0.5)
        pk, q, U = _toy_sample_U(S, L, V, T, b, base, rng)
        ev[0, :, j] = toy_functionals(pk); Nn[0, :, j] = pk.sum(1); Qq[0, :, j] = q.sum(1); Ee[0, :, j] = U.sum(1)
    one = np.ones(nD)
    gS, gN, gQ = g_parts(ev, Nn, Qq, [T], one)
    # (1) 分解（記事の中でも、まとめても）
    assert abs(gS[0] - (gN[0] - gQ[0])) < 1e-10, (gS, gN, gQ)
    pS, pN, pQ = g_parts(ev, Nn, Qq, [T], one, within=False)
    assert abs(pS[0] - (pN[0] - pQ[0])) < 1e-10
    # (2) 記事の中の g_N（偏りの補正なし）＝ 記事ごとに e^{hN/T} で傾けた ln K の数値微分。ΔH_全体 ＝ 記事ごとに e^{−δΣU} で傾けた ∂ln K/∂δ
    e = ev[0]
    def lnK_tilt(wt):
        wt = wt / wt.sum(0, keepdims=True) / nD
        c = (wt[..., None] * e).sum((0, 1))
        return np.log(c[1] / c[0]) - np.log(c[3] / c[2])
    eps = 1e-5
    num = (lnK_tilt(np.exp(eps * Nn[0] / T)) - lnK_tilt(np.exp(-eps * Nn[0] / T))) / (2 * eps)
    gN_pop = g_from(ev, Nn, [T], one, within=True, unbiased=False)[0]
    assert abs(num - gN_pop) < 1e-5 * max(1, abs(gN_pop)), (num, gN_pop)
    assert abs(gN[0] - gN_pop * S / (S - 1)) < 1e-9 * max(1, abs(gN_pop))
    numH = (lnK_tilt(np.exp(-eps * Ee[0])) - lnK_tilt(np.exp(eps * Ee[0]))) / (2 * eps)       # ∂ln K/∂(1/T)（文全体）
    dHw_pop = -T * g_from(ev, Ee, [T], one, within=True, unbiased=False)[0]
    assert abs(numH - dHw_pop) < 1e-5 * max(1, abs(dHw_pop)), (numH, dHw_pop)
    assert dHw_pop > 0, dHw_pop                                             # 結晶はエネルギーが低いので、冷やす（1/T を上げる）と K が上がる（fit_vh の b と同じ向き）
    # (3) まとめた g_S は g_identity と同じ
    gi = g_identity(ev, (Nn - Qq), [T], one)
    assert abs(gi[0] - pS[0]) < 1e-10, (gi, pS)
    # (4) 記事の中の g_S とまとめた g_S は近い
    assert abs(gS[0] - pS[0]) < 0.1 * abs(pS[0]), (gS, pS)
    # (5) clap：温度の格子を差しかえても fit_vh が正しく当てはめ、終わったら A8_TU が元に戻る
    tg = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]; a0, b0 = -6.0, 2.4                 # ln K ＝ a0 ＋ b0/T → T_m ＝ 0.4
    cw = np.zeros((7, 1, 16))
    for i, t in enumerate(tg):
        L0 = L1 = 1e6; km = 0.01; kf = km * np.exp(a0 + b0 / t)
        cw[i, 0, :4] = [L0, L0 * kf, L1, L1 * km]
        cw[i, 0, 4:8] = [1.5 * L0, L0, 0.3 * L1, L1]                                     # 熱量計の ΔH ＝ 1.2
        cw[i, 0, 10:12] = [0.2 * L0, 0.9 * L1]                                           # Δm ＝ 0.7
    before = list(fit_vh.__globals__['A8_TU'])
    c = clap(cw, tg, np.ones(1))
    assert fit_vh.__globals__['A8_TU'] == before
    assert c['ok'] and abs(c['Tm'] - 0.4) < 1e-6 and abs(c['dHvH'] - 2.4) < 1e-6 and abs(c['dHcal'] - 1.2) < 1e-9 and abs(c['dm'] - 0.7) < 1e-9, c
    # (6) 言葉
    assert word21([0.85, 1.2], 0) == 'そろう' and word21([1.3, 1.6], 0) == 'そろわない' and word21([0.6, 0.9], 0) == 'はっきりしない'
    assert word21([0.9, 1.1], 0.2) == 'はっきりしない' and word21([0.5, 0.7], 0) == 'そろわない'
    return dict(ok=True, gS_within=float(gS[0]), gS_pooled=float(pS[0]), gN_within=float(gN[0]), gN_pooled=float(pN[0]), gQ_within=float(gQ[0]),
                dH_whole=float(dHw_pop), num_tilt=float(num))
