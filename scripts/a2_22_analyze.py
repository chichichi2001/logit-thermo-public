"""A2-22 の分析：前もっての確かめのデータ（A2-19 の小さい 7 本、A2-20 の 7B の 2 本、c0 と h1 の腕）で、
XQ10 融点は狙いどおり 0.10 下がったか、XQ11 罰は入る速さと出る速さのどちらに効いたか。記録：平衡の読みの比、恒等式の f_入、状態ごとの寄与、重み付けの予言。
ゲート：docs/A2_22_GATE.md。データを開く前に書いた（2026-11-02 16:xx JST）。CPU。
scripts/a2_21_run.py の load21ns() の名前空間（seq_events・arm_counts・fit_vh・phases・copy_marks・g_from・lnK_from・A18_MIN_EV など）で実行する。
"""
import numpy as np

A22_BAND = (0.8, 1.25)
A22_ENTRY = (0.33, 0.67)
A22_BOOT, A22_SEED = 1000, 20261122
A22_FAIL = 0.10


def seq_states(toks):
    """(nT, S, nD, L) → 歩ごとの状態 (nT, S, nD, L)：0 ふつう、1 くり返し、−1 端（相がつかない歩）。"""
    nT, S, nD, L = toks.shape
    st = np.full((nT, S, nD, L), -1, np.int8)
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                p, t = phases(copy_marks(toks[k, s, j]))
                st[k, s, j, t] = p
    return st


def prep22(Z):
    """1 つの腕（toks・pick・q・U・temps）→ 分析の材料。"""
    toks, pick, q = Z['toks'], Z['pick'], Z['q'].astype(float)
    ev, sig = seq_events(toks, pick, q)
    cw, _ = arm_counts(toks, Z['U'], pick)
    st = seq_states(toks)
    dev = pick.astype(float) - q
    parts = {'liq': (dev * (st == 0)).sum(-1), 'cry': (dev * (st == 1)).sum(-1), 'edge': (dev * (st == -1)).sum(-1)}
    return dict(ev=ev, sig=sig, cw=cw, parts=parts, pick=pick.astype(float), q=q, temps=[float(t) for t in Z['temps']])


def tm_fit(cw, temps, w):
    """窓の数え → ファントホッフの当てはめの融点（fit_vh、温度は点で選ぶ）。出なければ nan。"""
    g = fit_vh.__globals__; old = g['A8_TU']; g['A8_TU'] = [float(t) for t in temps]
    try:
        f = fit_vh(np.einsum('kjc,j->kc', cw, w)[:, :10])
    finally:
        g['A8_TU'] = old
    return float(f['Tm']) if f.get('ok') and np.isfinite(f.get('Tm', np.nan)) else np.nan


def rates(ev, w):
    """ev (nT, S, nD, 4) → 温度ごとの ln k_f・ln k_m と、凍る回・溶ける回。"""
    c = np.einsum('ksjc,j->kc', ev, w)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(c[:, 1] / c[:, 0]), np.log(c[:, 3] / c[:, 2]), c[:, 1], c[:, 3]


def model_point(A0, A1, h, w, use=None):
    """1 つのモデルの c0（A0）と h1（A1）→ 比（ΔT_m ÷ −0.10）、f_入、使う温度など。"""
    T = A0['temps']
    lf0, lm0, f0, m0 = rates(A0['ev'], w)
    lf1, lm1, f1, m1 = rates(A1['ev'], w)
    if use is None:
        use = [k for k in range(len(T)) if min(f0[k], m0[k], f1[k], m1[k]) >= A18_MIN_EV]
    u = np.array(use, int)
    t0, t1 = tm_fit(A0['cw'], T, w), tm_fit(A1['cw'], T, w)
    dK = (lf1 - lm1) - (lf0 - lm0)
    ok_u = len(u) >= A18_MIN_T and np.all(np.isfinite(dK[u])) and dK[u].sum() != 0
    f_in = float((lf1 - lf0)[u].sum() / dK[u].sum()) if ok_u else np.nan
    return dict(use=use, Tm_c0=t0, Tm_h1=t1, dTm=t1 - t0, ratio=(t1 - t0) / -0.10, f_entry=f_in,
                dlnkf=(lf1 - lf0).tolist(), dlnkm=(lm1 - lm0).tolist(), dlnK=dK.tolist())


def word10(ci, fail):
    lo, hi = A22_BAND
    if fail > A22_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if lo <= ci[0] and ci[1] <= hi:
        return '狙いどおり'
    if ci[0] > hi:
        return '狙いより大きく動く'
    if ci[1] < lo:
        return '狙いより小さく動く'
    return 'はっきりしない'


def word11(ci, fail):
    lo, hi = A22_ENTRY
    if fail > A22_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if ci[0] > hi:
        return '入る速さが主'
    if ci[1] < lo:
        return '出る速さが主'
    if lo <= ci[0] and ci[1] <= hi:
        return '両方'
    return 'はっきりしない'


def identity_split(A0, w):
    """c0 の腕で、恒等式の g を「凍る速さの部分」「溶ける速さの部分」と、歩の状態ごとの寄与に分ける（まとめた共分散、g_identity と同じ）。"""
    T = A0['temps']; ev = A0['ev']
    def parts4(X):
        S_ = ev.shape[1]; ww = np.repeat(w[None, :], S_, 0).ravel(); W = ww.sum(); out = []
        for k, t in enumerate(T):
            x = X[k].ravel(); mx = (ww * x).sum() / W; d = []
            for i in range(4):
                e = ev[k, :, :, i].ravel(); me = (ww * e).sum() / W
                d.append(float((ww * (e - me) * (x - mx)).sum() / W / (t * me)) if me > 0 else np.nan)
            out.append(d)
        return np.array(out)                                               # (nT, 4)
    d = parts4(A0['sig'])
    g_in, g_out = d[:, 1] - d[:, 0], -(d[:, 3] - d[:, 2])                  # g ＝ g_in ＋ g_out
    by_state = {k: (lambda dd: (dd[:, 1] - dd[:, 0]) - (dd[:, 3] - dd[:, 2]))(parts4(v)) for k, v in A0['parts'].items()}
    return g_in, g_out, by_state


def is_predict(A0, h, w):
    """c0 の腕の文ごとの重み w_seq ＝ P_h(x)/P_0(x)（厳密）→ 記事ごとに正規化した ln K の予言、まとめて正規化した予言、有効な標本数。"""
    T = np.array(A0['temps'], float)[:, None, None, None]
    pick, q = A0['pick'], A0['q']
    logw = ((h / T) * pick - np.log1p(q * np.expm1(h / T))).sum(-1)        # (nT, S, nD)
    ev = A0['ev']
    out = {}
    for mode in ('doc', 'pooled'):
        if mode == 'doc':
            lw = logw - logw.max(1, keepdims=True); ws = np.exp(lw); ws = ws / ws.sum(1, keepdims=True)    # 記事ごとに正規化
            c = np.einsum('ksj,ksjc,j->kc', ws, ev, w) / w.sum()
            ess = float(np.mean((ws.sum(1) ** 2) / (ws ** 2).sum(1)))
        else:
            lw = logw - logw.max((1, 2), keepdims=True); ws = np.exp(lw) * w[None, None, :]
            ws = ws / ws.sum((1, 2), keepdims=True)
            c = np.einsum('ksj,ksjc->kc', ws, ev)
            ess = float(np.mean(1.0 / (ws ** 2).sum((1, 2))))
        with np.errstate(divide='ignore', invalid='ignore'):
            lnK = np.log(c[:, 1] / c[:, 0]) - np.log(c[:, 3] / c[:, 2])
        out[mode] = dict(lnK=lnK, ess=ess)
    return out


def analyze22(data, agg, seed=A22_SEED, boot=A22_BOOT):
    """data：{モデル: dict(c0=…, h1=…, h=h*, rC=r_C, Tstar=T*)}（記事の番号はモデルをまたいで同じ並び）。agg：'median' か 'mean'。"""
    names = list(data)
    nD = data[names[0]]['c0']['ev'].shape[2]
    one = np.ones(nD)
    f = np.median if agg == 'median' else np.mean
    pts = {m: model_point(d['c0'], d['h1'], d['h'], one) for m, d in data.items()}
    rng = np.random.default_rng(seed)
    bs10, bs11, bm = [], [], {m: {'ratio': [], 'f_entry': []} for m in names}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        r10, r11 = [], []
        for m in names:
            d = data[m]
            p = model_point(d['c0'], d['h1'], d['h'], w, use=pts[m]['use'])
            r10.append(p['ratio']); r11.append(p['f_entry'])
            bm[m]['ratio'].append(p['ratio']); bm[m]['f_entry'].append(p['f_entry'])
        bs10.append(f(r10) if np.all(np.isfinite(r10)) else np.nan)
        bs11.append(f(r11) if np.all(np.isfinite(r11)) else np.nan)
    res = {}
    for key, bs, wf in (('XQ10', bs10, word10), ('XQ11', bs11, word11)):
        bs = np.array(bs); fail = float(np.mean(~np.isfinite(bs)))
        ci = np.nanpercentile(bs, [2.5, 97.5]).tolist() if np.isfinite(bs).any() else [np.nan, np.nan]
        pk = 'ratio' if key == 'XQ10' else 'f_entry'
        vals = [pts[m][pk] for m in names]
        point = float(f(vals)) if np.all(np.isfinite(vals)) else np.nan
        res[key] = dict(point=point, ci=ci, fail=fail, word=wf(ci, fail))
    for m in names:
        for k in ('ratio', 'f_entry'):
            b = np.array(bm[m][k]); pts[m][k + '_ci'] = np.nanpercentile(b, [2.5, 97.5]).tolist() if np.isfinite(b).any() else [np.nan, np.nan]
        d = data[m]; p = pts[m]; u = np.array(p['use'], int)
        # 記録：平衡の読みの比、狙いに置けたか、恒等式の f_入、状態ごとの寄与、重み付けの予言
        p['ratio_over_rC'] = float(p['ratio'] / d['rC'])
        p['Tm_h1_minus_Tstar'] = float(p['Tm_h1'] - d['Tstar'])
        g_in, g_out, by_state = identity_split(d['c0'], one)
        g = g_in + g_out
        p['f_entry_identity'] = float(g_in[u].sum() / g[u].sum())
        p['state_share'] = {k: float(v[u].sum() / g[u].sum()) for k, v in by_state.items()}
        ip = is_predict(d['c0'], d['h'], one)
        l0 = (np.array(rates(d['c0']['ev'], one)[0]) - np.array(rates(d['c0']['ev'], one)[1]))
        D = np.array(p['dlnK'])
        p['is'] = {mode: dict(meas_over_is=float(D[u].sum() / (v['lnK'] - l0)[u].sum()), ess=v['ess']) for mode, v in ip.items()}
    return dict(models=pts, agg=agg, **res)


# ---------- 自己テスト ----------
def self_test22(seed=7):
    rng = np.random.default_rng(seed)
    # (1) 重み付けの式：作り物のくり返しの鎖で、罰なしの文の重み付けが、罰ありで本当に生成した文の平均に（小さい h で）一致する
    N, L, V, T, beta, h = 40000, 60, 10, 0.7, 1.5, -0.15
    base = rng.normal(0, 1.0, (L, V))
    _, pk0, q0 = toy_sample(N, L, V, T, 0.0, beta, base, rng)
    _, pkh, _ = toy_sample(N, L, V, T, h, beta, base, np.random.default_rng(seed + 1))
    logw = ((h / T) * pk0 - np.log1p(q0 * np.expm1(h / T))).sum(1)
    wv = np.exp(logw - logw.max()); wv /= wv.sum()
    assert abs(np.exp(logw).mean() - 1) < 0.05, np.exp(logw).mean()                     # E_0[w] ＝ 1
    est, act = float((wv * pk0.sum(1)).sum()), float(pkh.sum(1).mean())
    se = pkh.sum(1).std() / np.sqrt(N)
    assert abs(est - act) < 5 * se + 0.02 * act, (est, act, se)
    # (2) 融点の当てはめ：別の格子で正しく、終わったら A8_TU が戻る
    tg = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    def mk(a0, b0):
        cw = np.zeros((7, 1, 16))
        for i, t in enumerate(tg):
            L0 = L1 = 1e6; km = 0.01; kf = km * np.exp(a0 + b0 / t)
            cw[i, 0, :4] = [L0, L0 * kf, L1, L1 * km]; cw[i, 0, 4:8] = [1.5 * L0, L0, 0.3 * L1, L1]
        return cw
    before = list(fit_vh.__globals__['A8_TU'])
    t0 = tm_fit(mk(-6.0, 2.7), tg, np.ones(1)); t1 = tm_fit(mk(-6.0, 2.4), tg, np.ones(1))
    assert fit_vh.__globals__['A8_TU'] == before
    assert abs(t0 - 0.45) < 1e-6 and abs(t1 - 0.40) < 1e-6, (t0, t1)
    # (3) f_入：入る速さだけ変えた作り物で 1、出る速さだけで 0
    def evs(kf, km):
        ev = np.zeros((7, 1, 1, 4)); ev[:, 0, 0] = [1e6, 1e6 * kf, 1e6, 1e6 * km]; return ev
    A0 = dict(ev=evs(0.02, 0.01), cw=mk(-6.0, 2.7), temps=tg)
    p_in = model_point(A0, dict(ev=evs(0.01, 0.01), cw=mk(-6.0, 2.4), temps=tg), -0.3, np.ones(1))
    p_out = model_point(A0, dict(ev=evs(0.02, 0.02), cw=mk(-6.0, 2.4), temps=tg), -0.3, np.ones(1))
    assert abs(p_in['f_entry'] - 1) < 1e-9 and abs(p_out['f_entry']) < 1e-9, (p_in['f_entry'], p_out['f_entry'])
    assert abs(p_in['ratio'] - 0.5) < 1e-6, p_in['ratio']
    # (4) 言葉
    assert word10([0.85, 1.2], 0) == '狙いどおり' and word10([1.3, 1.5], 0) == '狙いより大きく動く' and word10([0.5, 0.7], 0) == '狙いより小さく動く'
    assert word10([0.7, 0.9], 0) == 'はっきりしない' and word10([0.9, 1.0], 0.2) == 'はっきりしない'
    assert word11([0.7, 0.9], 0) == '入る速さが主' and word11([0.1, 0.3], 0) == '出る速さが主' and word11([0.4, 0.6], 0) == '両方' and word11([0.5, 0.8], 0) == 'はっきりしない'
    return dict(ok=True, is_est=est, is_act=act)
