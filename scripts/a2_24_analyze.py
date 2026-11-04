"""A2-24 の分析（CPU、探索）。ゲート：docs/A2_24_GATE.md。scripts/a2_22_run.py の load22ns() の名前空間で実行する
（copy_marks・seq_events・lnK_from・g_identity・fit_vh・A18_MIN_EV・A18_MIN_T・word18 などを使う）。
XQ14 恒等式の予言を、歩の迷い度 q_t で 3 つに分ける。XQ15 融点に近い温度 1 つでの窓の割合の分布。XQ16 状態の分け方を変えたときの比 ρ。
"""
import numpy as np

A24_SEED, A24_BOOT, A24_FAIL = 20261124, 1000, 0.10
A24_QBINS = (0.1, 0.9)
A24_QBINS_ALT = ((0.05, 0.95), (0.2, 0.8))
A24_LABELS = [(w, e, x) for w in (16, 32, 64) for (e, x) in ((0.5, 0.25), (0.6, 0.3), (0.4, 0.2))]
A24_BAND = (0.8, 1.25)


# ---------- 状態の分け方（窓・しきい値を変えられる版。窓 32・(0.5, 0.25) で a2_8_vh.phases と同じ） ----------

def states_of(c, W, enter, exit_):
    """写しの印 c → (状態の列, 歩の番号)。r[k] は歩 k..k+W−1 の平均で、歩 k + W//2 に置く。最初の状態は最初の窓で決める。"""
    r = np.convolve(c.astype(np.float64), np.ones(W), 'valid') / W
    t = np.arange(len(r)) + W // 2
    s = np.zeros(len(r), np.int8)
    cur = 1 if r[0] >= enter - 1e-12 else 0
    for i, v in enumerate(r):
        if cur == 0 and v >= enter - 1e-12:
            cur = 1
        elif cur == 1 and v < exit_ - 1e-12:
            cur = 0
        s[i] = cur
    return s, t


def marks_of(toks):
    """(nT, S, nD, L) → 写しの印 (nT, S, nD, L)。"""
    nT, S, nD, L = toks.shape
    C = np.zeros(toks.shape, np.int8)
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                C[k, s, j] = copy_marks(toks[k, s, j])
    return C


def events_of(C, W, enter, exit_, half=None):
    """写しの印 → 1 本の文ごとの [ふつうの歩, 入る回, くり返しの歩, 出る回] (nT, S, nD, 4)。
    half ＝ 0/1 なら、遷移の元の歩が 256 より前／以後のものだけを数える（R2）。"""
    nT, S, nD, L = C.shape
    ev = np.zeros((nT, S, nD, 4))
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                st, t = states_of(C[k, s, j], W, enter, exit_)
                a, b = st[:-1], st[1:]
                if half is not None:
                    keep = (t[:-1] < 256) if half == 0 else (t[:-1] >= 256)
                    a, b = a[keep], b[keep]
                ev[k, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
    return ev


# ---------- 恒等式の g を、σ の部分ごとに（まとめた共分散、a2_22 の identity_split と同じ式） ----------

def g_part(ev, X, temps, w):
    """ev (nT, S, nD, 4)、X (nT, S, nD)：σ の一部 → 温度ごとの g の寄与。"""
    S_ = ev.shape[1]; ww = np.repeat(w[None, :], S_, 0).ravel(); W = ww.sum(); out = []
    for k, t in enumerate(temps):
        x = X[k].ravel(); mx = (ww * x).sum() / W; d = []
        for i in range(4):
            e = ev[k, :, :, i].ravel(); me = (ww * e).sum() / W
            d.append(float((ww * (e - me) * (x - mx)).sum() / W / (t * me)) if me > 0 else np.nan)
        out.append((d[1] - d[0]) - (d[3] - d[2]))
    return np.array(out)


def q_parts(pick, q, lo, hi):
    """σ ＝ Σ (c − q) を、q ＜ lo・lo ≤ q ≤ hi・q ＞ hi の歩の和に分ける → dict (nT, S, nD)。"""
    dev = pick.astype(float) - q
    return {'sure_no': (dev * (q < lo)).sum(-1), 'unsure': (dev * ((q >= lo) & (q <= hi))).sum(-1), 'sure_copy': (dev * (q > hi)).sum(-1)}


def q_share(ev, parts, temps, use, w):
    """→ 各範囲の割合 Σ_u g_part ÷ Σ_u g（g は 3 つの和）。"""
    u = np.array(use, int)
    gp = {k: g_part(ev, v, temps, w) for k, v in parts.items()}
    tot = sum(gp.values())
    den = tot[u].sum()
    return {k: float(v[u].sum() / den) if den != 0 and np.isfinite(den) else np.nan for k, v in gp.items()}


def var_share(pick, q, use, w, lo, hi):
    """迷っている歩（lo ≤ q ≤ hi）が、使う温度の c − q の分散 Σ q(1 − q) に占める割合（記事の重み w つき）。"""
    u = np.array(use, int)
    v = q[u] * (1 - q[u])                                                   # (nu, S, nD, L)
    tot = np.einsum('ksjl,j->', v, w)
    mid = np.einsum('ksjl,j->', v * ((q[u] >= lo) & (q[u] <= hi)), w)
    return float(mid / tot) if tot > 0 else np.nan


def word14(ci, fail):
    """迷っている歩の「効きの割合 ÷ 分散の割合」（9 本の中央値）。"""
    lo, hi = A24_BAND
    if fail > A24_FAIL or not np.all(np.isfinite(ci)):
        return 'はっきりしない'
    if lo <= ci[0] and ci[1] <= hi:
        return '迷い度（分散）に比例して効く'
    if ci[0] > hi:
        return '迷った歩で分散より多く効く'
    if ci[1] < lo:
        return '迷った歩で分散より少なく効く'
    return 'はっきりしない'


# ---------- XQ15 ----------

def window_hist(C_k, W=32, only=None):
    """1 つの温度の写しの印 (S, nD, L) → 窓の割合 k/W の分布 (W+1)。only：(S, nD) の真偽、真の文だけ数える。"""
    h = np.zeros(W + 1)
    for s in range(C_k.shape[0]):
        for j in range(C_k.shape[1]):
            if only is not None and not only[s, j]:
                continue
            r = np.convolve(C_k[s, j].astype(float), np.ones(W), 'valid')
            h += np.bincount(np.rint(r).astype(int), minlength=W + 1)
    return h / h.sum() if h.sum() > 0 else h


def seq_kinds(C_k, W=32, enter=0.5, exit_=0.25):
    """1 つの温度の文ごとに、状態の並びから：入れかわりがあったか、ずっとふつうか、ずっとくり返しか (S, nD)。"""
    S_, nD = C_k.shape[:2]
    sw = np.zeros((S_, nD), bool); allo = np.zeros((S_, nD), bool); allr = np.zeros((S_, nD), bool)
    for s in range(S_):
        for j in range(nD):
            st, _ = states_of(C_k[s, j], W, enter, exit_)
            sw[s, j] = bool(np.any(st[1:] != st[:-1])); allo[s, j] = bool(np.all(st == 0)); allr[s, j] = bool(np.all(st == 1))
    return sw, allo, allr


def bimodal(hist, W=32):
    f = np.arange(W + 1) / W
    zl, zm, zh = f < 0.25, (f >= 0.25) & (f <= 0.75), f > 0.75                # 幅は入れ物の数で（W ＝ 32 なら 8・17・8）
    lo, mid, hi = hist[zl].sum() / zl.sum(), hist[zm].sum() / zm.sum(), hist[zh].sum() / zh.sum()
    return dict(low=float(lo), mid=float(mid), high=float(hi), ratio=float(mid / min(lo, hi)) if min(lo, hi) > 0 else np.inf,
                bimodal=bool(min(lo, hi) > 0 and mid < 0.2 * min(lo, hi)))


def word15(n, N=9):
    return '9 本すべて' if n == N else ('7〜8 本' if n >= 7 else '6 本以下')


# ---------- XQ16 ----------

def rho_label(ev0, sig0, ev1, h, temps, w, use=None):
    """1 つの分け方での比 ρ（A2-18 の rho と同じ式。使う温度は両方の腕で入る回・出る回が 10 以上）。"""
    l0, f0, m0 = lnK_from(ev0, w)
    l1, f1, m1 = lnK_from(ev1, w)
    if use is None:
        use = [k for k in range(len(temps)) if min(f0[k], m0[k], f1[k], m1[k]) >= A18_MIN_EV]
    if len(use) < A18_MIN_T:
        return np.nan, use
    g = np.array(g_identity(ev0, sig0, temps, w))
    u = np.array(use, int)
    D, P = (l1 - l0)[u], (h * g)[u]
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(P))) or P.sum() == 0:
        return np.nan, use
    return float(D.sum() / P.sum()), use


def agg(vals, how, max_missing):
    v = np.array(vals, float); ok = np.isfinite(v)
    if ok.sum() < len(v) - max_missing or ok.sum() == 0:
        return np.nan
    return float(np.median(v[ok]) if how == 'median' else np.mean(v[ok]))


def boot(point_fn, names, nD, how, max_missing, seed=A24_SEED, n=A24_BOOT):
    """point_fn(name, w, use) → (値, use)。→ 点、区間、出ない回の割合、モデルごとの点と区間。"""
    one = np.ones(nD)
    pts = {m: point_fn(m, one, None) for m in names}
    point = agg([pts[m][0] for m in names], how, max_missing)
    rng = np.random.default_rng(seed)
    bs, bm = [], {m: [] for m in names}
    for _ in range(n):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        r = [point_fn(m, w, pts[m][1])[0] for m in names]
        for m, v in zip(names, r):
            bm[m].append(v)
        bs.append(agg(r, how, max_missing))
    bs = np.array(bs); fail = float(np.mean(~np.isfinite(bs)))
    ci = np.nanpercentile(bs, [2.5, 97.5]).tolist() if np.isfinite(bs).any() else [np.nan, np.nan]
    models = {m: dict(point=pts[m][0], ci=(np.nanpercentile(bm[m], [2.5, 97.5]).tolist() if np.isfinite(bm[m]).any() else [np.nan, np.nan]))
              for m in names}
    return dict(point=point, ci=ci, fail=fail, models=models)


def word16(rows):
    """rows：元以外の 8 通りの dict(point, ci)。"""
    lo, hi = A24_BAND
    if any(not np.isfinite(r['point']) or not (lo <= r['point'] <= hi) for r in rows):
        return '帯を外れる分け方がある'
    if all(np.all(np.isfinite(r['ci'])) and lo <= r['ci'][0] and r['ci'][1] <= hi for r in rows):
        return 'どの分け方でも帯の内'
    return '点はどれも帯の内'


# ---------- 自己テスト ----------

def self_test24():
    rng = np.random.default_rng(5)
    # (1) states_of（窓 32・(0.5, 0.25)）が a2_8_vh.phases と同じ
    for _ in range(200):
        c = (rng.random(512) < rng.choice([0.05, 0.5, 0.95])).astype(np.int8)
        c[rng.integers(0, 400):][:rng.integers(0, 120)] = 1
        s1, t1 = states_of(c, 32, 0.5, 0.25); s0, t0 = phases(c)
        assert np.array_equal(s1, s0) and np.array_equal(t1, t0)
    # (2) events_of が seq_events の ev と同じ（作り物のトークン列）
    toks = rng.integers(0, 6, (2, 2, 3, 512)).astype(np.int64)
    toks[0, 0, 0, 200:] = np.tile([1, 2, 3, 4], 78)[:312]
    pick = rng.random(toks.shape) < 0.3; q = rng.random(toks.shape)
    ev_ref, sig_ref = seq_events(toks, pick, q)
    assert np.array_equal(events_of(marks_of(toks), 32, 0.5, 0.25), ev_ref)
    # (3) q_parts の和が σ、g_part の和が g_identity
    parts = q_parts(pick, q, 0.1, 0.9)
    assert np.allclose(sum(parts.values()), sig_ref)
    temps = [0.5, 0.6]; w = np.ones(3)
    g_ref = np.array(g_identity(ev_ref, sig_ref, temps, w))
    assert np.allclose(sum(g_part(ev_ref, v, temps, w) for v in parts.values()), g_ref, equal_nan=True)
    # (4) σ が迷った歩だけにあれば、割合は 1
    ev = np.abs(rng.normal(300, 30, (2, 4, 40, 4))); X = ev[..., 1] * 0.01 + rng.normal(0, 0.01, (2, 4, 40))
    sh = q_share(ev, {'sure_no': np.zeros_like(X), 'unsure': X, 'sure_copy': np.zeros_like(X)}, temps, [0, 1], np.ones(40))
    assert abs(sh['unsure'] - 1) < 1e-12 and abs(sh['sure_no']) < 1e-12
    # (5) 2 つの山の決まり
    hb = np.zeros(33); hb[:3] = 0.25; hb[-3:] = 0.25; hb[10:23] = 0.0
    hb /= hb.sum(); assert bimodal(hb)['bimodal']
    hu = np.ones(33) / 33; assert not bimodal(hu)['bimodal']
    # (6) 言葉
    assert word14([0.9, 1.1], 0) == '迷い度（分散）に比例して効く' and word14([1.3, 1.6], 0) == '迷った歩で分散より多く効く' and word14([0.5, 0.7], 0) == '迷った歩で分散より少なく効く' and word14([0.7, 1.1], 0) == 'はっきりしない'
    # (7) var_share：q が全部 0.5 なら、範囲 (0.1, 0.9) の分散の割合は 1、全部 0.95 なら 0
    qq = np.full((2, 2, 3, 10), 0.5); assert abs(var_share(None, qq, [0, 1], np.ones(3), 0.1, 0.9) - 1) < 1e-12
    qq[:] = 0.95; assert var_share(None, qq, [0, 1], np.ones(3), 0.1, 0.9) == 0
    # (8) seq_kinds：ずっと 0 の文はずっとふつう、途中で写しが続けば入れかわりあり
    Ck = np.zeros((1, 2, 512), np.int8); Ck[0, 1, 200:300] = 1
    sw, ao, ar = seq_kinds(Ck); assert ao[0, 0] and not sw[0, 0] and sw[0, 1] and not ao[0, 1]
    assert word16([dict(point=1.0, ci=[0.9, 1.1])] * 8) == 'どの分け方でも帯の内'
    assert word16([dict(point=1.0, ci=[0.7, 1.1])] * 8) == '点はどれも帯の内'
    assert word16([dict(point=1.3, ci=[1.1, 1.5])] + [dict(point=1.0, ci=[0.9, 1.1])] * 7) == '帯を外れる分け方がある'
    return 'OK'
