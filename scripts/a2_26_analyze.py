"""A2-26 の分析（CPU、探索）。ゲート：docs/A2_26_GATE.md。scripts/a2_24_run.py の load24ns() の名前空間で実行する
（states_of・marks_of・events_of・g_identity・lnK_from・boot・agg・phases・copy_marks・arm_counts など）。
XQ20 文の前半・後半で平衡の読みの比、XQ21 分け方 8 通りで平衡の読みの比、XQ22 くり返しからの出やすさがとどまった長さによらないか、XQ23 融点の行きすぎはどこから来たか（設計の温度での ln K の分け）。
"""
import numpy as np

A26_SEED, A26_BOOT, A26_FAIL = 20261126, 1000, 0.10
A26_BAND = (0.8, 1.25)
A26_HALF = 256                                                          # 前半：遷移の元の歩 ＜ 256、後半：≥ 256
A26_SHORT, A26_LONG = (49, 96), 97                                      # XQ22 のとどまった長さの区切り（窓 32 とヒステリシスで、d ≲ 33 は出られないため。レビュー 1）
A26_HZ_MIN = 10                                                         # XQ22：モデルごとに、どちらの区間でも出た回が 10 以上


# ---------- 数え（状態・遷移・状態ごとの U と候補を選んだ数） ----------

def counts10(C, U, pick, W=32, en=0.5, ex=0.25, half=None):
    """写しの印 C、U、pick：(nT, S, nD, L) → (nT, S, nD, 10)：
    [N0, n01, N1, n10, U ふつう の和, ふつうの歩, U くり返し の和, くり返しの歩, 候補を選んだ数 ふつう, くり返し]。
    遷移（0〜3）は対の元の歩で、状態の和（4〜9）はラベルのある歩で数える。half ＝ 0/1 なら、元の歩・歩が前半／後半のものだけ。"""
    nT, S, nD, L = C.shape
    out = np.zeros((nT, S, nD, 10))
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                st, t = states_of(C[k, s, j], W, en, ex)
                a, b, ta = st[:-1], st[1:], t[:-1]
                if half is None:
                    keep_p, keep_s = np.ones(len(a), bool), np.ones(len(st), bool)
                else:
                    keep_p = (ta < A26_HALF) if half == 0 else (ta >= A26_HALF)
                    keep_s = (t < A26_HALF) if half == 0 else (t >= A26_HALF)
                a2, b2 = a[keep_p], b[keep_p]
                o, r = (st == 0) & keep_s, (st == 1) & keep_s
                u, p = U[k, s, j, t].astype(np.float64), pick[k, s, j, t].astype(np.float64)
                out[k, s, j] = [(a2 == 0).sum(), ((a2 == 0) & (b2 == 1)).sum(), (a2 == 1).sum(), ((a2 == 1) & (b2 == 0)).sum(),
                                u[o].sum(), o.sum(), u[r].sum(), r.sum(), p[o].sum(), p[r].sum()]
    return out


def tot(c, w):
    return np.einsum('ksjc,j->kc', c, w)


def lnK_of(t):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(t[:, 1] / t[:, 0]) - np.log(t[:, 3] / t[:, 2])


def wfit(x, y, wt, deg):
    X = np.stack([x ** i for i in range(deg + 1)], 1)
    return np.linalg.solve(X.T @ (wt[:, None] * X), X.T @ (wt * y))


def eq_pred(t0, temps, h, use):
    """罰なしの腕の数え（温度ごとの合計）→ 平衡の読みの予言 E(T) ＝ h n Δc_T / T（n ＝ b / ΔE）。"""
    u = np.array(use, int); T = np.array(temps, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dc = t0[:, 9] / t0[:, 7] - t0[:, 8] / t0[:, 5]
        dE = t0[:, 4] / t0[:, 5] - t0[:, 6] / t0[:, 7]
    lk = lnK_of(t0)
    wt = 1.0 / (1.0 / t0[u, 1] + 1.0 / t0[u, 3])
    b = wfit(1.0 / T[u], lk[u], wt, 1)[1]
    if not np.mean(dE[u]) > 0:
        return np.full(len(T), np.nan), dict(b=float(b), dE=float(np.mean(dE[u])), n=np.nan)
    n = b / float(np.mean(dE[u]))
    return h * n * dc / T, dict(b=float(b), dE=float(np.mean(dE[u])), n=float(n))


def use_of(t0, t1):
    return [k for k in range(t0.shape[0]) if min(t0[k, 1], t0[k, 3], t1[k, 1], t1[k, 3]) >= A18_MIN_EV]


def ratios(c0, c1, sig, temps, h, w, use=None):
    """→ (ρ_E, ρ, use)：平衡の読みに対する比、恒等式に対する比。"""
    t0, t1 = tot(c0, w), tot(c1, w)
    if use is None:
        use = use_of(t0, t1)
    if len(use) < A18_MIN_T:
        return np.nan, np.nan, use
    u = np.array(use, int)
    D = lnK_of(t1) - lnK_of(t0)
    E, _ = eq_pred(t0, temps, h, use)
    g = np.array(g_identity(c0[..., :4], sig, temps, w))
    P = h * g
    if not (np.all(np.isfinite(D[u])) and np.all(np.isfinite(E[u])) and np.all(np.isfinite(P[u]))):
        return np.nan, np.nan, use
    return float(D[u].sum() / E[u].sum()), float(D[u].sum() / P[u].sum()), use


def word_over(rows, fail_max=A26_FAIL):
    """rows：dict(point, ci, fail) の並び（半分ごと、または分け方ごと）。言いすぎ ＝ 区間の上 ＜ 0.8。"""
    lo = A26_BAND[0]
    if any(r.get('fail', 0) > fail_max or not np.isfinite(r['point']) or not np.all(np.isfinite(r['ci'])) for r in rows):
        return 'unclear'
    if all(r['ci'][1] < lo for r in rows):
        return 'all'
    if all(r['point'] < lo for r in rows):
        return 'points'
    return 'not_all'


# ---------- XQ22：とどまった長さとハザード ----------

def hazard_from_states(st, acc_rep, acc_ord, j):
    """状態の列 st（1 本の文）→ 記事 j の行に [短い区間の危険の歩, 出た回, 長い区間の危険の歩, 出た回, 短い区間の位置の和, 長い区間の位置の和] を足す（くり返し → acc_rep、ふつう → acc_ord）。
    最初の歩から続いている状態（いつ入ったか分からない）は数えない。文の終わりで打ち切り。"""
    n = len(st); i = 0
    while i < n:
        v = st[i]; e = i
        while e + 1 < n and st[e + 1] == v:
            e += 1
        if i > 0:
            acc = acc_rep if v == 1 else acc_ord
            for d in range(1, e - i + 2):                               # d 歩目（1 始まり）にいて、次の歩で出るか
                pos = i + d - 1
                if pos + 1 >= n:
                    break
                out = st[pos + 1] != v
                if A26_SHORT[0] <= d <= A26_SHORT[1]:
                    acc[j, 0] += 1; acc[j, 1] += out; acc[j, 4] += pos
                elif d >= A26_LONG:
                    acc[j, 2] += 1; acc[j, 3] += out; acc[j, 5] += pos
        i = e + 1


def hazard_counts(C_k, W=32, en=0.5, ex=0.25):
    """1 つの温度の写しの印 (S, nD, L) → 記事ごとの (nD, 6)：くり返しから出る、ふつうから入る。"""
    S_, nD = C_k.shape[:2]
    rep = np.zeros((nD, 6)); ordi = np.zeros((nD, 6))
    for s in range(S_):
        for j in range(nD):
            hazard_from_states(states_of(C_k[s, j], W, en, ex)[0], rep, ordi, j)
    return rep, ordi


def dwell_hist(C_k, W=32, en=0.5, ex=0.25, edges=(1, 17, 65, 129, 257, 513)):
    """とどまる長さ（入った時が分かり、文の終わりより前に出たもの）のくり返しの状態の数を、区切りごとに。"""
    h = np.zeros(len(edges) - 1, int)
    for s in range(C_k.shape[0]):
        for j in range(C_k.shape[1]):
            st = states_of(C_k[s, j], W, en, ex)[0]
            ch = np.flatnonzero(np.diff(st)) + 1
            for a, b in zip(ch[:-1], ch[1:]):
                if st[a] == 1:
                    h += np.histogram([b - a], bins=list(edges))[0]
    return h.tolist()


def hazard_ratio(cnt, w, min_ev=A26_HZ_MIN):
    """記事を足した比：長い区間のハザード ÷ 短い区間のハザード。どちらかの出た回が min_ev 未満なら出ない。"""
    c = cnt[:, :4].T @ w
    if c[1] < min_ev or c[3] < min_ev or c[0] <= 0 or c[2] <= 0:
        return np.nan
    return float((c[3] / c[2]) / (c[1] / c[0]))


def word_band(ci, fail, words):
    lo, hi = A26_BAND
    if fail > A26_FAIL or not np.all(np.isfinite(ci)):
        return words[3]
    if lo <= ci[0] and ci[1] <= hi:
        return words[0]
    if ci[1] < lo:
        return words[1]
    if ci[0] > hi:
        return words[2]
    return words[3]


# ---------- XQ23：融点の行きすぎの内訳（設計の温度での ln K の分け） ----------

def design_split(t0, t1, g, temps, k, a, b, gbar, h):
    """設計の温度に近い格子の温度 T（k 番目）で、罰ありの腕の ln K の、設計の直線（a ＋ b/T ＋ h ḡ）からのずれを 3 つに分ける。
    ずれ ＝ [ln K_c0(T) − (a ＋ b/T)]（罰なしの ln K が設計の直線からずれた分：曲がり・データの違い）
         ＋ h [g(T) − ḡ]（g が温度で変わる分）＋ [Δ(T) − h g(T)]（恒等式との差）。厳密な足し算。
    向き：b ＞ 0 なので、ln K が下にずれるほど融点は下に動く（行きすぎ）。行きすぎの向きを正にするため、どれも −1 倍して返す。"""
    if min(t0[k, 1], t0[k, 3], t1[k, 1], t1[k, 3]) < A18_MIN_EV or not np.isfinite(g[k]):
        return (np.nan,) * 4
    T = temps[k]; l0 = lnK_of(t0)[k]; l1 = lnK_of(t1)[k]
    p1 = l0 - (a + b / T); p2 = h * (g[k] - gbar); p3 = (l1 - l0) - h * g[k]
    tot_ = l1 - (a + b / T + h * gbar)
    assert abs(p1 + p2 + p3 - tot_) < 1e-9
    return -p1, -p2, -p3, -tot_


def curv_split(t0, temps, k, a, b, use):
    """記録：罰なしのずれ ln K_c0(T) − (a ＋ b/T) を、この格子の中の曲がり（測った ln K − この格子での直線の当てはめ）と、
    残り（この格子での直線 − 設計の直線：データの違い）に分ける。−1 倍（行きすぎの向きを正）。"""
    u = np.array(use, int); T = np.array(temps, float)
    lk = lnK_of(t0)
    wt = 1.0 / (1.0 / t0[u, 1] + 1.0 / t0[u, 3])
    c = wfit(1.0 / T[u], lk[u], wt, 1)
    lin = c[0] + c[1] / T[k]
    return -(lk[k] - lin), -(lin - (a + b / T[k]))


def word23(r1, r2, fail_max=A26_FAIL):
    """r1・r2：罰なしのずれ・g の温度変化（行きすぎの向きを正）の、9 本の中央値の dict(point, ci, fail)。"""
    if any(r['fail'] > fail_max or not np.all(np.isfinite(r['ci'])) for r in (r1, r2)):
        return 'unclear'
    p1, p2 = r1['ci'][0] > 0, r2['ci'][0] > 0
    return {(True, False): 'base', (False, True): 'g', (True, True): 'both', (False, False): 'neither'}[(p1, p2)]


# ---------- 自己テスト ----------

def self_test26():
    rng = np.random.default_rng(7)
    # (1) counts10 の遷移の数えが events_of と同じ、前半＋後半 ＝ 全文
    toks = rng.integers(0, 6, (2, 2, 3, 512)).astype(np.int64)
    toks[0, 0, 0, 200:] = np.tile([1, 2, 3, 4], 78)[:312]; toks[1, 1, 2, 50:300] = np.tile([5, 4, 3], 84)[:250]
    C = marks_of(toks); U = rng.random(toks.shape).astype(np.float32); pick = rng.random(toks.shape) < 0.3
    full = counts10(C, U, pick); h0 = counts10(C, U, pick, half=0); h1 = counts10(C, U, pick, half=1)
    assert np.array_equal(full[..., :4], events_of(C, 32, 0.5, 0.25))
    assert np.allclose(h0 + h1, full)
    # (2) 状態ごとの和が a2_16 の counts_window と同じ
    for k in range(2):
        for j in range(3):
            cw = counts_window(toks[k, :, j], U[k, :, j], pick[k, :, j])
            assert np.allclose(full[k, :, j].sum(0)[[4, 5, 6, 7, 8, 9]], cw[[4, 5, 6, 7, 10, 11]])
    # (3) eq_pred：作り物の合計で n ＝ b / ΔE
    temps = [0.4, 0.5, 0.6]; t0 = np.zeros((3, 10))
    for i, T in enumerate(temps):
        K = np.exp(-2 + 1.0 / T)
        t0[i] = [1000, 100 * K, 1000, 100, 2000, 1000, 500, 1000, 200, 900]
    E, info = eq_pred(t0, temps, -0.3, [0, 1, 2])
    assert abs(info['b'] - 1.0) < 1e-9 and abs(info['dE'] - 1.5) < 1e-12 and np.allclose(E, -0.3 * (1 / 1.5) * 0.7 / np.array(temps))
    # (4) hazard_from_states：一定のハザード（マルコフ鎖）なら比は 1 に近い、長さで下がるハザードなら 1 より小さい
    def chain(haz_fn, n=200000):
        st = np.zeros(n, np.int8); cur, d = 0, 0
        for i in range(n):
            d += 1
            p = haz_fn(d) if cur == 1 else 0.02
            if rng.random() < p:
                cur ^= 1; d = 0
            st[i] = cur
        return st
    for fn, lo_, hi_ in ((lambda d: 0.02, 0.85, 1.15), (lambda d: 0.04 if d < 97 else 0.01, 0.15, 0.4)):
        rep = np.zeros((1, 6)); od = np.zeros((1, 6))
        hazard_from_states(chain(fn), rep, od, 0)
        hr = hazard_ratio(rep, np.ones(1))
        assert lo_ < hr < hi_, hr
    st = np.array([1, 1, 0, 0, 1, 1, 1, 0], np.int8); rep = np.zeros((1, 6)); od = np.zeros((1, 6))
    hazard_from_states(st, rep, od, 0)
    assert rep.sum() == 0 and od.sum() == 0                             # d ＜ 49 だけ：数えない
    # (4b) 較正：トークンの列を、写しの印 → 窓の状態 → ハザード と通しても、マルコフな鎖で比が 1 に近い（レビュー 1）
    def tok_chain(n_seq, L=512, sw=0.01):
        out = np.zeros((1, n_seq, 512), np.int64)
        for i in range(n_seq):
            cur = int(rng.random() < 0.5); cyc = rng.integers(1000, 2000, 7)
            for t_ in range(L):
                if rng.random() < sw:
                    cur ^= 1
                out[0, i, t_] = cyc[t_ % 7] if cur else rng.integers(5000, 60000)
        return out
    tk = tok_chain(1500)
    Ck = np.stack([copy_marks(tk[0, i]) for i in range(tk.shape[1])])[None].astype(np.int8)
    rep, _ = hazard_counts(Ck)
    hr = hazard_ratio(rep, np.ones(rep.shape[0]))
    assert 0.85 < hr < 1.15, hr
    # 出た回が 10 未満なら出ない
    assert np.isnan(hazard_ratio(np.array([[100, 5, 100, 50, 0, 0]], float), np.ones(1)))
    # (5) design_split：作り物で 3 つの和がずれと一致し、向きが正しい
    temps = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]; Tg = np.array(temps)
    a_, b_ = -5.0, 2.5; gbar = 2.0; h_ = -(a_ + b_ / 0.5) / gbar
    lk0 = a_ + b_ / Tg - 0.2; g_ = 2.0 + 0.5 * (Tg - 0.5) + 0.3
    mk = lambda lk: np.stack([np.full(7, 1e7), 1e6 * np.exp(lk), np.full(7, 1e7), np.full(7, 1e6), *np.zeros((6, 7))], 1)
    q1, q2, q3, qt = design_split(mk(lk0), mk(lk0 + h_ * g_), g_, temps, 3, a_, b_, gbar, h_)
    assert abs(q1 - 0.2) < 1e-9 and abs(q2 - (-h_ * 0.3)) < 1e-9 and abs(q3) < 1e-9 and abs(qt - (q1 + q2 + q3)) < 1e-12
    c1_, c2_ = curv_split(mk(lk0), temps, 3, a_, b_, list(range(7)))
    assert abs(c1_) < 1e-9 and abs(c2_ - 0.2) < 1e-9                    # 曲がりなし：全部データの違い
    # (6) 言葉
    R = lambda lo, hi, f=0.0: dict(point=(lo + hi) / 2, ci=[lo, hi], fail=f)
    assert word23(R(0.1, 0.3), R(-0.1, 0.1)) == 'base' and word23(R(-0.1, 0.1), R(0.1, 0.2)) == 'g'
    assert word23(R(0.1, 0.3), R(0.1, 0.2)) == 'both' and word23(R(-0.1, 0.3), R(-0.1, 0.2)) == 'neither' and word23(R(0.1, 0.3, 0.2), R(0.1, 0.2)) == 'unclear'
    assert word_over([R(0.4, 0.6)] * 2) == 'all' and word_over([R(0.4, 0.9)] * 2) == 'points' and word_over([R(0.8, 1.0)]) == 'not_all'
    assert word_over([R(0.4, 0.6), dict(point=np.nan, ci=[np.nan, np.nan], fail=0)]) == 'unclear' and word_over([R(0.4, 0.6, 0.2)]) == 'unclear'
    assert word_band([0.9, 1.1], 0, 'abcd') == 'a' and word_band([0.5, 0.7], 0, 'abcd') == 'b' and word_band([1.3, 1.5], 0, 'abcd') == 'c' and word_band([0.7, 1.1], 0, 'abcd') == 'd'
    return 'OK'
