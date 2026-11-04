"""探索 8.35 の分析：h ＝ 0 の生成（q つき）だけから場の効きを厳密に予言し、8.30 の h ＝ ±0.3 の実測とくらべる（CPU、判定しない）。
データを開く前に書いた（2026-10-31 14:4x）。scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases・fit_vh）・a2_13_analyze.py（A8_TU）
を実行したのと同じ名前空間で実行する。

1 本の文（512 歩）について、P ＝ Σ_t（候補を選んだか）、σ ＝ Σ_t（候補を選んだか − q_t）（全部の歩）。
1 歩ずつ割り直す生成では、場 h の重みは w(h) ＝ exp(h P / T − Σ_t ln(1 − q_t ＋ q_t e^(h/T)))（近似なし）。
 (a) 一次の応答（厳密）：d⟨X⟩/dh ＝ Cov(X, σ) ÷ T。X ＝ M（歩 16〜495 で候補を選んだ数）。
     予言の傾きを、8.30 の実測（(⟨M⟩(＋0.3) − ⟨M⟩(−0.3)) ÷ 0.6）とくらべる。比 R_予言 ＝ Cov(M, σ) ÷ Var(M) を、8.34 の R（0.01）とくらべる。
 (b) 速さ：d ln k/dh ＝ Cov(n_回, σ)/(T ⟨n_回⟩) − Cov(N_歩, σ)/(T ⟨N_歩⟩)（凍る k_f ＝ 凍る回 ÷ 液体の歩、溶ける k_m も同じ）。
     ΔM_速さ（予言）＝ T ×（d ln k_f/dh − d ln k_m/dh）を、8.30 の実測（T × ∂ln K/∂h、±0.3 の差）とくらべる。
 (c) 融点の傾き：h ＝ ±0.05 に厳密に重みづけし直した数えで、2 つの状態の当てはめ（温度は h ＝ 0 と同じ）→ T_m(±0.05) → 傾き。
     8.30 の実測（0.268・0.207）とくらべる。重みの有効なサンプル数（ESS）も記録。参考に ±0.3 への重みづけと ESS も。
区間：記事の選び直し 500 回（種 20261136）。
"""
import numpy as np

FQ_EPS = 0.05
FQ_BOOT, FQ_SEED = 500, 20261136


def fq_seq_stats(toks, pick, q):
    """toks・pick・q：(nT, S, nD, L) → dict（どれも (nT, S, nD)）：P、σ、M、ln(1 − q ＋ q e^x) を足すための q の列は返さない。
    2 つの状態の数え c2：(nT, S, nD, 4) ＝ [液体の歩, 凍る回, 結晶の歩, 溶ける回]。"""
    nT, S, nD, L = toks.shape
    P = pick.sum(-1).astype(float)
    sig = (pick.astype(float) - q.astype(float)).sum(-1)
    M = pick[..., 16:496].sum(-1).astype(float)
    c2 = np.zeros((nT, S, nD, 4))
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                st, _ = phases(copy_marks(toks[k, s, j]))
                a, b = st[:-1], st[1:]
                c2[k, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
    return dict(P=P, sig=sig, M=M, c2=c2)


def log_w(P, q, T, h):
    """P：(…,) 、q：(…, L) → ln w(h)（1 本の文ごと）。"""
    x = h / T
    return h * P / T - np.log1p(q.astype(float) * np.expm1(x)).sum(-1)


def lin_resp(X, sig, T):
    """X・sig：(N,) → Cov(X, σ)/T。"""
    return float(np.mean((X - X.mean()) * (sig - sig.mean())) / T)


def fq_point(st, Ts, lw, idx, use_idx):
    """st：fq_seq_stats の結果、lw：{h: (nT, S, nD) の ln w}、idx：記事の番号、use_idx：fit_vh の温度。"""
    nT = len(Ts)
    res = {'pred_dM': [], 'R_pred': [], 'dlnkf': [], 'dlnkm': [], 'dM_kin_pred': []}
    for k in range(nT):
        T = Ts[k]
        M = st['M'][k][:, idx].ravel(); sg = st['sig'][k][:, idx].ravel(); c = st['c2'][k][:, idx].reshape(-1, 4)
        res['pred_dM'].append(lin_resp(M, sg, T))
        res['R_pred'].append(float(np.mean((M - M.mean()) * (sg - sg.mean())) / M.var()))
        d = [lin_resp(c[:, i], sg, T) / c[:, i].mean() if c[:, i].mean() > 0 else np.nan for i in range(4)]
        res['dlnkf'].append(d[1] - d[0]); res['dlnkm'].append(d[3] - d[2]); res['dM_kin_pred'].append(T * ((d[1] - d[0]) - (d[3] - d[2])))
    tms, ess = {}, {}
    for h, L in lw.items():
        tot = np.zeros((nT, 10)); e = []
        for k in range(nT):
            w = np.exp(L[k][:, idx].ravel() - L[k][:, idx].max()); c = st['c2'][k][:, idx].reshape(-1, 4)
            tot[k, :4] = (w[:, None] * c).sum(0) / w.sum() * len(w)
            tot[k, 5], tot[k, 7] = tot[k, 0], tot[k, 2]          # fit_vh の熱量計の欄が 0/0 にならないように（U は 0）
            e.append(float(w.sum() ** 2 / (w ** 2).sum() / len(w)))
        f = fit_vh(tot, temps_use=use_idx)
        tms[h] = f.get('Tm', float('nan')) if f.get('ok') else float('nan'); ess[h] = e
    res['Tm_rw'] = tms; res['ess'] = ess
    res['slope_rw'] = (tms[FQ_EPS] - tms[-FQ_EPS]) / (2 * FQ_EPS)
    res['slope_rw03'] = (tms[0.3] - tms[-0.3]) / 0.6
    return res


def fq_measured(base_field, name, idx=None):
    """8.30 の ±0.3 の実測：⟨M⟩ の傾き、温度ごとの T × ∂ln K/∂h。"""
    out = {}
    Z = {a: np.load(f'{base_field}/{name}@{a}.npz') for a in ('hm03', 'hp03')}
    Ts = [float(t) for t in Z['hp03']['temps']]
    nD = Z['hp03']['toks'].shape[2]
    idx = np.arange(nD) if idx is None else idx
    Mm = {a: Z[a]['pick'][:, :, idx, 16:496].sum(-1).astype(float).mean((1, 2)) for a in Z}
    out['dM'] = ((Mm['hp03'] - Mm['hm03']) / 0.6).tolist()
    return out, Ts


def analyze_fq(base_fq, base_field, name):
    Z = np.load(f'{base_fq}/{name}@h0q.npz')
    F0 = np.load(f'{base_field}/{name}@h0.npz')
    assert (Z['toks'] == F0['toks']).all(), '8.30 の h ＝ 0 とトークンが違う'
    Ts = [float(t) for t in Z['temps']]
    q = Z['q0']
    st = fq_seq_stats(Z['toks'], Z['pick'], q)
    lw = {h: np.stack([log_w(st['P'][k], q[k], Ts[k], h) for k in range(len(Ts))]) for h in (FQ_EPS, -FQ_EPS, 0.3, -0.3)}
    tot0 = np.zeros((len(Ts), 10)); tot0[:, :4] = st['c2'].sum((1, 2)); tot0[:, 5], tot0[:, 7] = tot0[:, 0], tot0[:, 2]
    f0 = fit_vh(tot0)
    use_idx = f0['use_idx']
    nD = Z['toks'].shape[2]
    pt = fq_point(st, Ts, lw, np.arange(nD), use_idx)
    pt['Tm_h0'] = f0.get('Tm')
    meas, _ = fq_measured(base_field, name)
    pt['meas_dM'] = meas['dM']
    rng = np.random.default_rng(FQ_SEED); bs = []
    for _ in range(FQ_BOOT):
        idx = rng.integers(0, nD, nD)
        b = fq_point(st, Ts, lw, idx, use_idx)
        b['meas_dM'] = fq_measured(base_field, name, idx)[0]['dM']
        bs.append(b)
    ci = {}
    for key in ('pred_dM', 'R_pred', 'dlnkf', 'dlnkm', 'dM_kin_pred', 'meas_dM'):
        arr = np.array([b[key] for b in bs], float)
        ci[key] = [np.nanpercentile(arr, 2.5, 0).tolist(), np.nanpercentile(arr, 97.5, 0).tolist()]
    for key in ('slope_rw', 'slope_rw03'):
        arr = np.array([b[key] for b in bs], float)
        ci[key] = [float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
    ratio = np.array([np.array(b['pred_dM']) / np.array(b['meas_dM']) for b in bs], float)
    pt['ratio_dM'] = (np.array(pt['pred_dM']) / np.array(pt['meas_dM'])).tolist()
    ci['ratio_dM'] = [np.nanpercentile(ratio, 2.5, 0).tolist(), np.nanpercentile(ratio, 97.5, 0).tolist()]
    pt['ci'] = ci; pt['temps'] = Ts
    return pt


def self_test_fq_an():
    """作り物：毎歩 q ＝ 0.3 の候補を、温度 T で独立に選ぶ列。重みづけで h ＝ 0.05 の平均を出すと解析解と合う。一次の応答も解析解と合う。"""
    rng = np.random.default_rng(2)
    T, L, N, qv = 0.5, 200, 40000, 0.3
    q = np.full((N, L), qv)
    pick = rng.random((N, L)) < qv
    P = pick.sum(1).astype(float); sig = (pick - q).sum(1)
    for h in (0.05, -0.05):
        lw = log_w(P, q, T, h); w = np.exp(lw - lw.max())
        qh = qv * np.exp(h / T) / (1 - qv + qv * np.exp(h / T))
        est = (w * P).sum() / w.sum()
        assert abs(est - L * qh) < 0.3, (h, est, L * qh)
    lr = lin_resp(P, sig, T)
    assert abs(lr - L * qv * (1 - qv) / T) < 2.0, (lr, L * qv * (1 - qv) / T)
    return 'OK'
