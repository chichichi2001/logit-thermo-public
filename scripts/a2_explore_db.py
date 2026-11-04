"""探索 8.32：詳細つり合い（コルモゴロフの閉路）を、温度ごとに・記事の選び直しの区間つきで確かめる（2026-10-31 13:0x、CPU、判定しない）。
8.31 の弱いところ（温度 0.6・0.7 の回数を足してから積を取った、区間がない）を直す。
scripts/a2_explore_onset.py・a2_8_vh.py（phases）・a2_explore_three.py・a2_explore_seg.py・a2_13_analyze.py・a2_14_analyze.py・a2_15_analyze.py を
実行したのと同じ名前空間で実行する（load14・C14_FOLDER・C15_MODELS・seg_counts・seg_counts_by_doc を使う）。

閉路の量：c ＝ ln（N_LA × N_AC × N_CL ÷ (N_AL × N_CA × N_LC)）。速さ ＝ 回数 ÷ いた時間で、時間は分子と分母で消える。
平衡（詳細つり合い）なら c ＝ 0。c > 0 は L → A → C → L の向きに回る流れ。
材料：A2-13・A2-14・A2-15 の 24 モデル、「なし」＋「なし（別の種）」（温度 0.4〜0.7 の 7 個、サンプル 2 ＋ 2、記事 40、512 歩）。
温度ごとに c を出し（6 つの回数がどれも 5 回以上の温度だけ）、モデルごとに 1/分散（＝ 1/Σ(1/N)）の重みで平均する。
区間：記事の選び直し 1,000 回（種 20261131、全部のモデルで同じ選び方）。
対照（ものさしの見かけ）：つり合っている作り物（3 つの状態を、どの組も両向き同じ確率で行き来する）と、回る作り物（L → A → C → L の向きの速さを √1.5 倍、逆を ÷√1.5、真の c ＝ 3 ln 1.5 ＝ 1.22）を、
同じラベルで数える。つり合っている作り物で c が 0 から外れれば、その分はラベルが作った見かけ。
"""
import numpy as np

DB_KEYS = ['LA', 'AL', 'AC', 'CA', 'LC', 'CL']
DB_MIN = 5
DB_BOOT, DB_SEED = 1000, 20261131


def cycle_of(cnt):
    """cnt：(..., 9)（[L, A, C, LA, AL, AC, CA, LC, CL]）→ (c, 分散)。どれかの回数が DB_MIN 未満なら nan。"""
    n = np.asarray(cnt, float)[..., 3:]
    ok = (n >= DB_MIN).all(-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.log(n[..., 0]) + np.log(n[..., 2]) + np.log(n[..., 5]) - np.log(n[..., 1]) - np.log(n[..., 3]) - np.log(n[..., 4])
        v = (1.0 / n).sum(-1)
    return np.where(ok, c, np.nan), np.where(ok, v, np.nan)


def model_cycle(cnt_T):
    """cnt_T：(nT, 9) → (重みつき平均の c, 温度ごとの c のリスト)。"""
    c, v = cycle_of(cnt_T)
    m = np.isfinite(c)
    if not m.any():
        return float('nan'), c
    w = 1.0 / v[m]
    return float((w * c[m]).sum() / w.sum()), c


def synth_counts(n_seq, drive, seed, L=512, p=1 / 120, lagA=45, lagC=23, rev=False):
    """作り物：歩ごとに状態を変える（L・A・C、どの組も 1 歩あたり p。drive 倍だけ L→A・A→C・C→L を強め、逆を弱める）。
    L はでたらめなトークン、A は 45 歩前の写し、C は 23 歩前の写し（入った歩からすぐ写す）。→ 9 個の数え（足したもの）。"""
    rng = np.random.default_rng(seed)
    st = {'L': 0, 'A': 1, 'C': 2}
    P = np.full((3, 3), p)
    for a, b in (('L', 'A'), ('A', 'C'), ('C', 'L')):
        P[st[a], st[b]] = p * np.sqrt(drive); P[st[b], st[a]] = p / np.sqrt(drive)
    np.fill_diagonal(P, 0); np.fill_diagonal(P, 1 - P.sum(1))
    tot = np.zeros(9)
    for _ in range(n_seq):
        s = np.zeros(L, np.int64); x = 0
        for t in range(L):
            if t > 0:
                x = rng.choice(3, p=P[x])
            if x == 0 or (x == 1 and t < lagA) or (x == 2 and t < lagC):
                s[t] = rng.integers(100, 30000)
            else:
                s[t] = s[t - (lagA if x == 1 else lagC)]
        occ, tr, _ = seg_counts(s[::-1] if rev else s)
        tot[:3] += [occ['L'], occ['A'], occ['C']]
        tot[3:] += [tr.get(k, 0) for k in DB_KEYS]
    return tot


def synth_check(n_seq=400):
    out = {}
    for name, d, seed in (('balanced', 1.0, 1), ('balanced_2', 1.0, 2), ('driven_x1.5', 1.5, 3)):
        cnt = synth_counts(n_seq, d, seed)
        c, v = cycle_of(cnt)
        cr, vr = cycle_of(synth_counts(n_seq, d, seed, rev=True))
        out[name] = dict(c=float(c), se=float(np.sqrt(v)), c_rev=float(cr), se_rev=float(np.sqrt(vr)), counts=cnt[3:].tolist(), true=float(3 * np.log(d)))
    return out


def db_counts(base, models, rev=False):
    """→ {モデル: (7, nD, 9)}（「なし」＋「なし（別の種）」を足す）。rev：トークンの列を後ろから読む（時間を逆にする）。"""
    res = {}
    for m in models:
        tn, un = load14(base, m, 'none'); tb, ub = load14(base, m, 'none_b')
        assert un == ub, m
        if rev:
            tn, tb = tn[..., ::-1], tb[..., ::-1]
        res[m] = seg_counts_by_doc(tn) + seg_counts_by_doc(tb)
    return res


def analyze_db(cnts):
    models = list(cnts)
    nD = {cnts[m].shape[1] for m in models}
    assert len(nD) == 1, nD
    nD = nD.pop()
    pt = {}
    for m in models:
        cm, cT = model_cycle(cnts[m].sum(1))
        pt[m] = dict(c=cm, c_T=[None if not np.isfinite(x) else float(x) for x in cT],
                     n_T=cnts[m].sum(1)[:, 3:].astype(int).tolist())
    allm = np.array([pt[m]['c'] for m in models])
    pooled_T = cycle_of(sum(cnts[m].sum(1) for m in models))[0]
    rng = np.random.default_rng(DB_SEED)
    bm, bT = [], []
    for _ in range(DB_BOOT):
        idx = rng.integers(0, nD, nD)
        cs = [model_cycle(cnts[m][:, idx].sum(1))[0] for m in models]
        bm.append(np.nanmean(cs))
        bT.append(cycle_of(sum(cnts[m][:, idx].sum(1) for m in models))[0])
    bm = np.array(bm); bT = np.array(bT)
    pc = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return dict(models=pt, mean=float(np.nanmean(allm)), mean_ci=pc(bm), n_pos=int((allm > 0).sum()), n_models=len(models),
                pooled_T=[float(x) for x in pooled_T], pooled_T_ci=[pc(bT[:, k]) for k in range(bT.shape[1])])


def self_test_db():
    c, v = cycle_of(np.array([0, 0, 0, 10, 10, 10, 10, 10, 10]))
    assert abs(c) < 1e-12 and abs(v - 0.6) < 1e-12
    c, _ = cycle_of(np.array([0, 0, 0, 20, 10, 20, 10, 10, 20]))
    assert abs(c - 3 * np.log(2)) < 1e-12
    c, _ = cycle_of(np.array([0, 0, 0, 4, 10, 10, 10, 10, 10]))
    assert np.isnan(c)
    m, _ = model_cycle(np.array([[0, 0, 0, 10, 10, 10, 10, 10, 10], [0, 0, 0, 20, 10, 20, 10, 10, 20]]))
    assert 0 < m < 3 * np.log(2)
    return 'OK'
