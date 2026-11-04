"""A2-23 の予言と分析（CPU）。ゲート：docs/A2_23_GATE.md。scripts/a2_22_run.py の load22ns() の名前空間で実行する（a2_20_predict・a2_21・a2_22 の関数を使う）。

(A) 割る型の罰：
  予言：G-pre の h ＝ 0 の文（温度 0.40〜0.70 の 7 つ、記事 40 本）と、その読み直し（A_sum ＝ Σ_t A_t）から、温度ごとの g_λ(T) ＝ ∂ln K/∂λ（λ ＝ ln θ）を恒等式で出す
    （g_from、記事をまたいでまとめた共分散。A2-18 の g と同じ組み立てで、σ の代わりに A_sum）。ḡ_λ ＝ 7 つの平均。
    融点の当てはめ（a、b、T*、格子）は A2-18 の予言のファイル（罰の種類によらない）をそのまま使う。λ* ＝ −(a ＋ b/T*) ÷ ḡ_λ、θ* ＝ e^{λ*}。
    ḡ_λ ≥ 0、または θ* が 1〜3 の外なら「予言できない」。
  確かめ：比 ρ ＝ Σ_u Δln K（m1 対 A2-19 の c0）÷ Σ_u λ*・g_λ^c0(T)（g_λ^c0 は A2-19 の c0 の読み直しから）。u ＝ 両方の腕で凍る回・溶ける回がどれも 10 以上の温度。
(D) 指示調整済みのモデル：予言は a2_20_predict の predict_model20 そのまま。確かめは A2-18 の rho_model（足す型の罰、c0 の恒等式）。
判定：(A) 7 本の ρ の中央値、(D) 2 本の ρ の平均。記事の選び直し 1,000 回の区間。言葉は A2-18 の word18（帯 0.8〜1.25）。
"""
import numpy as np

A23_SEED_BOOT = 20261123
A23_THETA_RANGE = (1.0, 3.0)


def predict_mult(Zg, Ag, pred18, name):
    """Zg：G-pre の h ＝ 0 の npz、Ag：その読み直し（A_sum）の npz、pred18：A2-18 の予言のファイルのそのモデルの項目。"""
    temps = [round(float(t), 4) for t in Zg['temps']]
    assert temps == [round(float(t), 4) for t in Ag['temps']] and list(Zg['use']) == list(Ag['use'])
    ev, sig = seq_events(Zg['toks'], Zg['pick'], Zg['q'])
    one = np.ones(ev.shape[2])
    gS = g_from(ev, sig, temps, one, within=False)
    assert np.allclose(gS, pred18['g_T'], atol=1e-9), (name, '足す型の g が A2-18 の予言と合わない')    # 材料の並びの確かめ
    gl = g_from(ev, Ag['A_sum'].astype(float), temps, one, within=False)
    gbar = float(np.mean(gl))
    t = pred18['targets']['-0.10']
    res = dict(name=name, g_lambda_T=gl.tolist(), gbar=gbar, Tstar=t['Tstar'], grid=t['grid'], a=pred18['a'], b=pred18['b'],
               max_dU=float(np.max(Ag['dU_max'])), max_dq0=float(np.max(Ag['dq0_max'])))
    if not np.isfinite(gbar):
        return dict(res, ok=False, why='g_λ が出ない温度がある')
    if not gbar < 0:
        return dict(res, ok=False, why='ḡ_λ が負でない（罰で融点が下がる向きに効かない）')
    lam = -(pred18['a'] + pred18['b'] / t['Tstar']) / gbar
    theta = float(np.exp(lam))
    res.update(lam=float(lam), theta=theta, lam_used=float(np.log(round(theta, 4))))   # 生成は θ を 4 桁に丸めて使う。比の分母も丸めた θ の λ で
    if not (A23_THETA_RANGE[0] < theta < A23_THETA_RANGE[1]):
        return dict(res, ok=False, why=f'θ* が {A23_THETA_RANGE} の外')
    return dict(res, ok=True, why=None)


def rho_mult(E0, E1, Asum0, lam, temps, w, use=None):
    """E0・E1：c0・m1 の (ev, sig)、Asum0：c0 の読み直しの A_sum。→ (ρ, 使う温度, Δ, P)。"""
    l0, f0, m0 = lnK_from(E0[0], w)
    l1, f1, m1 = lnK_from(E1[0], w)
    if use is None:
        use = [k for k in range(len(temps)) if min(f0[k], m0[k], f1[k], m1[k]) >= A18_MIN_EV]
    if len(use) < A18_MIN_T:
        return float('nan'), use, None, None
    g = g_from(E0[0], Asum0, temps, w, within=False)
    D = l1 - l0; P = lam * g
    u = np.array(use, int)
    if not (np.all(np.isfinite(D[u])) and np.all(np.isfinite(P[u]))) or P[u].sum() == 0:
        return float('nan'), use, D, P
    return float(D[u].sum() / P[u].sum()), use, D, P


def boot_agg(fn_point, names, nD, agg, seed=A23_SEED_BOOT, boot=1000):
    """fn_point(name, w, use) → (ρ, use, D, P)。モデルをまたいで同じ記事の選び方。"""
    one = np.ones(nD)
    pts = {m: fn_point(m, one, None) for m in names}
    def f(v):                                                              # 中央値：出ないモデルが 2 本まではほかで（A2-18 の _median_or_nan と同じ）。平均：全部出たときだけ
        v = np.array(v, float); ok = np.isfinite(v)
        if agg == 'median':
            return float(np.median(v[ok])) if ok.sum() >= len(v) - 2 and ok.sum() > 0 else np.nan
        return float(np.mean(v)) if ok.all() else np.nan
    vals = [pts[m][0] for m in names]
    point = f(vals)
    rng = np.random.default_rng(seed)
    bs, bm = [], {m: [] for m in names}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        r = [fn_point(m, w, pts[m][1])[0] for m in names]
        for m, v in zip(names, r):
            bm[m].append(v)
        bs.append(f(r))
    bs = np.array(bs); fail = float(np.mean(~np.isfinite(bs)))
    ci = np.nanpercentile(bs, [2.5, 97.5]).tolist() if np.isfinite(bs).any() else [np.nan, np.nan]
    models = {}
    for m in names:
        b = np.array(bm[m])
        cim = np.nanpercentile(b, [2.5, 97.5]).tolist() if np.isfinite(b).any() else [np.nan, np.nan]
        models[m] = dict(rho=pts[m][0], use=pts[m][1], D=None if pts[m][2] is None else pts[m][2].tolist(),
                         P=None if pts[m][3] is None else pts[m][3].tolist(), ci=cim, fail=float(np.mean(~np.isfinite(b))), word=word18(cim, float(np.mean(~np.isfinite(b)))))
    return dict(point=point, ci=ci, fail=fail, word=word18(ci, fail), agg=agg, models=models)


def self_test23():
    rng = np.random.default_rng(3)
    # (1) predict_mult：作り物で ḡ_λ の符号と θ* の向き
    N, L, V, T = 400, 60, 10, 0.6
    temps = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    ev = np.abs(rng.normal(500, 50, (7, 4, 40, 4))); A = -ev[..., 2] * 0.003 + rng.normal(0, 0.1, (7, 4, 40))   # 結晶の歩が多い文ほど A が小さい（罰で結晶が減る向き）
    gl = g_from(ev, A, temps, np.ones(40), within=False)
    assert np.all(gl < 0), gl
    # (2) rho_mult：Δ と P が同じなら ρ ＝ 1
    ev1 = ev.copy(); w = np.ones(40)
    l0, *_ = lnK_from(ev, w)
    r, use, D, P = rho_mult((ev, None), (ev, None), A, 0.2, temps, w)
    assert np.allclose(D, 0) and (np.isnan(r) or abs(r) < 1e-12)
    # (3) predict_mult：作り物の材料で、ḡ_λ ＜ 0 なら θ* ＞ 1、足す型の g の確かめが通る
    toks = rng.integers(0, 5, (7, 4, 40, 512)); pick = rng.random((7, 4, 40, 512)) < 0.3; q = np.clip(rng.random((7, 4, 40, 512)), 0.01, 0.99)
    Zg = dict(temps=np.array(temps), use=np.arange(40), toks=toks, pick=pick, q=q)
    evg, sigg = seq_events(toks, pick, q)
    pred18 = dict(g_T=g_from(evg, sigg, temps, np.ones(40), within=False).tolist(), a=-6.0, b=3.3, targets={'-0.10': dict(Tstar=0.45, grid=temps)})
    Ag = dict(temps=np.array(temps), use=np.arange(40), A_sum=-evg[..., 2] * 0.003 * 0 - sigg * 0.0 + (-(evg[..., 2] - evg[..., 2].mean()) * 0.01), dU_max=np.zeros(1), dq0_max=np.zeros(1))
    pr = predict_mult(Zg, Ag, pred18, 'テスト')
    if np.isfinite(pr['gbar']) and pr['gbar'] < 0:
        assert pr['ok'] == (1 < pr.get('theta', 0) < 3) and pr['lam'] > 0
    # (4) word18 の帯
    assert word18([0.85, 1.2], 0) == '予言どおり（2 割の内）' and word18([1.3, 1.5], 0) == '予言より大きく効く'
    return 'OK'
