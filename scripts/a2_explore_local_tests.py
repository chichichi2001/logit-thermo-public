"""探索 8.34：式の一覧（docs/A2_EQUATION_CATALOG.md）のうち、手元のデータで試せるもの（2026-10-31 14:2x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases）・a2_explore_three.py・a2_explore_seg.py・a2_13_analyze.py・a2_14_analyze.py・
a2_15_analyze.py・a2_explore_db.py を実行したのと同じ名前空間で実行する。

試すもの：
 (1) へスの法則：ln K_LA ＋ ln K_AC − ln K_LC は閉路の量 c（8.32）と同じ（いた時間は消える）。よって
     ΔH_LA ＋ ΔH_AC − ΔH_LC ＝ dc/d(1/T)。へスの法則 ⇔ c が温度によらない。（c そのものの値はラベルのくせを含むが、温度による変わり方は別の問い）
 (2) 場の側（8.30 のデータ）：
     (a) ゆらぎと応答：d⟨M⟩/dh（＋0.3 と −0.3 の差 ÷ 0.6）と Var(M)/T（h ＝ 0）の比 R。全体のボルツマン分布なら 1、
         1 歩ずつ割り直す生成では R ＝ 1 − Cov(M, Q)/Var(M)（Q ＝ Σ 候補の確率）。M ＝ 1 本の文で候補を選んだ歩の数。
     (b) 場の α：α_h ＝ ∂ln k_f/∂h ÷ (∂ln k_f/∂h − ∂ln k_m/∂h)（遷移状態が「写しの数」でどこにあるか）。エネルギーの α（8.27：0.52〜0.55）と比べる。
     (c) 3 つの状態の場の効き：T × ∂ln K/∂h を L⇄A・A⇄C・L⇄C で（A⇄C のクラペイロン）。
 (3) 速さ（A2-13〜15 の 24 モデル、「なし」2 本、温度 0.4〜0.7、2 つの状態）：
     (a) アレニウス：E_a,f・E_a,m・α ＝ E_a,f ÷ (E_a,f − E_a,m)、モデルをまたいだブレンステッドの関係（E_a,f と ΔH の直線）。
     (b) アイリング：ln(k/T) ＝ ΔS‡ − ΔH‡/T（試みの回数は 1 歩に 1 回、T を前の係数に入れる）。
 (4) 結晶の大きさと育ち方（同じ 24 モデル）：
     (a) 溶ける危険率（1 歩あたり）を、結晶になってからの歩数（年齢）ごと・周期ごとに。ギブズ・トムソン（小さい結晶ほど溶けやすい）の形を見る。
     (b) アヴラミ：はじめの液体が結晶になるまでの生き残り S(t)。−ln S(t) ∝ t^n の n（1 なら芽のでき方が一定、1 より大きいなら、
         文が長くなるほど芽ができやすい ＝ エネルギーの地形が書きかわる）。
区間：記事の選び直し（種 20261135、回数は各所に書く）。
"""
import numpy as np

LT_SEED = 20261135
LT_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
LT_AGE_BINS = [0, 8, 16, 32, 64, 128, 256, 600]
LT_PER_BINS = [0, 8, 16, 32, 64, 128, 1000]


# ---------- (1) へス ----------

def hess_from_counts(cnt, temps, boot=1000, seed=LT_SEED):
    """cnt：(nT, nD, 9) → D ＝ dc/d(1/T)（c ＝ 閉路の量、温度ごと）と区間。"""
    def fit(tot):
        c, v = cycle_of(tot)
        ok = np.isfinite(c)
        if ok.sum() < 3:
            return float('nan'), c
        x = 1 / np.asarray(temps)[ok]; w = 1 / v[ok]
        X = np.stack([np.ones_like(x), x], 1)
        a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * c[ok]))
        return float(b), c
    D, c = fit(cnt.sum(1))
    rng = np.random.default_rng(seed); nD = cnt.shape[1]; bs = []
    for _ in range(boot):
        bs.append(fit(cnt[:, rng.integers(0, nD, nD)].sum(1))[0])
    bs = np.array(bs)
    return dict(D=D, ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], c_T=[None if not np.isfinite(x) else float(x) for x in c])


# ---------- (2) 場 ----------

def field_tests(base, name, boot=500, seed=LT_SEED):
    Z = {a: np.load(f'{base}/{name}@{a}.npz') for a in ('hm03', 'h0', 'hp03')}
    temps = [float(t) for t in Z['h0']['temps']]
    nT, S, nD, L = Z['h0']['toks'].shape
    # (a) M ＝ 歩 16〜495 で候補を選んだ数
    M = {a: Z[a]['pick'][..., 16:496].sum(-1).astype(float) for a in Z}          # (nT, S, nD)
    # 2 つの状態の数え（記事ごと）
    c2 = {a: counts2(Z[a]['toks']) for a in Z}                                    # (nT, nD, 10)
    s3 = {a: seg_counts_by_doc(Z[a]['toks']) for a in Z}                          # (nT, nD, 9)

    def point(idx):
        out = {}
        Mi = {a: M[a][:, :, idx] for a in M}
        slope = (Mi['hp03'].mean((1, 2)) - Mi['hm03'].mean((1, 2))) / 0.6
        var = Mi['h0'].reshape(nT, -1).var(1, ddof=1)
        out['R'] = slope / (var / np.array(temps))
        out['M_mean'] = np.stack([Mi[a].mean((1, 2)) for a in ('hm03', 'h0', 'hp03')], 1)
        tot = {a: c2[a][:, idx].sum(1) for a in c2}
        with np.errstate(divide='ignore', invalid='ignore'):
            kf = {a: tot[a][:, 1] / tot[a][:, 0] for a in tot}; km = {a: tot[a][:, 3] / tot[a][:, 2] for a in tot}
            gf = (np.log(kf['hp03']) - np.log(kf['hm03'])) / 0.6; gm = (np.log(km['hp03']) - np.log(km['hm03'])) / 0.6
        out['dlnkf_dh'] = gf; out['dlnkm_dh'] = gm; out['alpha_h'] = gf / (gf - gm)
        t3 = {a: s3[a][:, idx].sum(1) for a in s3}
        for f, b, i, j in (('LA', 'AL', 3, 4), ('AC', 'CA', 5, 6), ('LC', 'CL', 7, 8)):
            with np.errstate(divide='ignore', invalid='ignore'):
                lk = {a: np.log(rates(t3[a])[f] / rates(t3[a])[b]) for a in t3}
            ok = np.array([min(t3[a][k, i], t3[a][k, j]) >= 8 for k in range(nT) for a in ('hm03', 'hp03')]).reshape(nT, 2).all(1)
            out[f'dM_{f}'] = np.where(ok, np.array(temps) * (lk['hp03'] - lk['hm03']) / 0.6, np.nan)
        return out
    pt = point(np.arange(nD))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(boot):
        bs.append(point(rng.integers(0, nD, nD)))
    res = {'temps': temps}
    for k in pt:
        if k == 'M_mean':
            res[k] = pt[k].tolist(); continue
        arr = np.array([b[k] for b in bs])
        res[k] = dict(point=[None if not np.isfinite(x) else float(x) for x in pt[k]],
                      lo=[float(x) for x in np.nanpercentile(arr, 2.5, axis=0)], hi=[float(x) for x in np.nanpercentile(arr, 97.5, axis=0)])
        # 温度を平均した値（点と区間）
        res[k]['mean'] = float(np.nanmean(pt[k]))
        mb = np.nanmean(arr, 1)
        res[k]['mean_ci'] = [float(np.nanpercentile(mb, 2.5)), float(np.nanpercentile(mb, 97.5))]
    return res


# ---------- (3)(4) 相の列から ----------

def seq_info(seq):
    """→ st（歩 16〜495 の相）, 結晶のかたまり [(はじめ, おわり, 溶けて終わったか, 周期)]（位置は st の番号）。"""
    st, t = phases(copy_marks(seq))
    lag = copy_lags(seq)
    segs = []
    i, n = 0, len(st)
    while i < n:
        if st[i] == 1:
            j = i
            while j < n and st[j] == 1:
                j += 1
            lg = lag[t[i]:t[j - 1] + 1]; lg = lg[lg > 0]
            per = float(np.percentile(lg, 10)) if len(lg) else float('nan')
            segs.append((i, j, j < n, per))
            i = j
        else:
            i += 1
    return st, segs


def model_seqs(toks):
    """toks：(nT, S, nD, L) → infos[k][j] ＝ そのモデル・温度・記事の (st, segs) のリスト（サンプル分）。"""
    nT, S, nD, _ = toks.shape
    return [[[seq_info(toks[k, s, j]) for s in range(S)] for j in range(nD)] for k in range(nT)]


def two_state_doc(infos):
    """→ (nT, nD, 4)：[液体の歩, 凍る回, 結晶の歩, 溶ける回]（歩 k → k＋1 の数え、A2-8 と同じ）。"""
    nT, nD = len(infos), len(infos[0])
    out = np.zeros((nT, nD, 4))
    for k in range(nT):
        for j in range(nD):
            for st, _ in infos[k][j]:
                a, b = st[:-1], st[1:]
                out[k, j] += [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
    return out


def arrh(tot, temps, eyring=False):
    """tot：(nT, 4) → E_a,f・E_a,m・ln A（回数 10 以上の温度、回数の重み）。eyring なら ln(k/T) で当てはめ（ΔH‡・ΔS‡）。"""
    out = {}
    for nm, i, ii in (('f', 1, 0), ('m', 3, 2)):
        ok = tot[:, i] >= 10
        if ok.sum() < 3:
            out[nm] = (float('nan'), float('nan')); continue
        k = tot[ok, i] / tot[ok, ii]; T = np.asarray(temps)[ok]
        y = np.log(k / T) if eyring else np.log(k)
        x = 1 / T; w = tot[ok, i]
        X = np.stack([np.ones_like(x), x], 1)
        a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        out[nm] = (float(-b), float(a))          # E（または ΔH‡）、ln A（または ΔS‡）
    return out


def hazard(infos, bins, key):
    """溶ける危険率（1 歩あたり）を、年齢（key＝'age'）か周期（key＝'per'）の箱ごとに。→ (nT, nD, 箱, 2)：[溶けた回, 危険にさらされた歩]。"""
    nT, nD = len(infos), len(infos[0]); nb = len(bins) - 1
    out = np.zeros((nT, nD, nb, 2))
    for k in range(nT):
        for j in range(nD):
            for st, segs in infos[k][j]:
                for (a, b, melted, per) in segs:
                    dur = b - a
                    if key == 'age':
                        ages = np.arange(dur)
                        idx = np.clip(np.searchsorted(bins, ages, side='right') - 1, 0, nb - 1)
                        np.add.at(out[k, j, :, 1], idx, 1)
                        if melted:
                            out[k, j, idx[-1], 0] += 1
                    else:
                        if not np.isfinite(per):
                            continue
                        q = int(np.clip(np.searchsorted(bins, per, side='right') - 1, 0, nb - 1))
                        out[k, j, q, 1] += dur; out[k, j, q, 0] += int(melted)
    return out


def first_freeze(infos):
    """→ (nT, nD, S)：はじめて結晶になった位置（st の番号）。ならなければ −1。"""
    nT, nD = len(infos), len(infos[0]); S = len(infos[0][0])
    out = -np.ones((nT, nD, S), int)
    for k in range(nT):
        for j in range(nD):
            for s, (st, segs) in enumerate(infos[k][j]):
                if st[0] == 1:
                    out[k, j, s] = 0
                elif segs:
                    out[k, j, s] = segs[0][0]
    return out


def avrami_n(ff, L=480, lo=0.05, hi=1.5):
    """ff：はじめて凍った位置の配列（−1 は凍らない）→ −ln S(t) が lo〜hi のあいだで log-log の傾き n。"""
    ff = np.asarray(ff).ravel()
    ts = np.arange(1, L)
    S = np.array([np.mean((ff < 0) | (ff >= t)) for t in ts])
    with np.errstate(divide='ignore'):
        H = -np.log(np.clip(S, 1e-12, 1))
    ok = (H >= lo) & (H <= hi)
    if ok.sum() < 10:
        return float('nan'), H
    n, _ = np.polyfit(np.log(ts[ok]), np.log(H[ok]), 1)
    return float(n), H


def self_test_lt():
    # アヴラミ：一定の危険率（n ＝ 1）と、t に比例して増える危険率（n ＝ 2）
    rng = np.random.default_rng(1)
    ff1 = rng.geometric(1 / 150, 20000) - 1; ff1[ff1 >= 480] = -1
    u = rng.random(20000); ff2 = np.floor(np.sqrt(-2 * np.log(u) / 2e-5)).astype(int); ff2[ff2 >= 480] = -1
    n1, _ = avrami_n(ff1); n2, _ = avrami_n(ff2)
    assert abs(n1 - 1) < 0.1 and abs(n2 - 2) < 0.15, (n1, n2)
    # へス：c が温度によらなければ D ＝ 0、c ＝ 0.3/T なら D ＝ 0.3
    temps = [0.4, 0.5, 0.6, 0.7]
    cnt = np.zeros((4, 1, 9)); cnt[:, 0, 3:] = 1000
    assert abs(hess_from_counts(cnt, temps, boot=5)['D']) < 1e-9
    for k, T in enumerate(temps):
        cnt[k, 0, 3] = 1000 * np.exp(0.3 / T)
    assert abs(hess_from_counts(cnt, temps, boot=5)['D'] - 0.3) < 1e-6
    # かたまりの切り出し：でたらめ → 周期 23 → でたらめ
    s = list(rng.integers(100, 30000, 150)) + list(np.tile(rng.integers(100, 30000, 23), 20)[:200]) + list(rng.integers(100, 30000, 162))
    st, segs = seq_info(np.array(s))
    assert len(segs) == 1 and segs[0][2] and abs(segs[0][3] - 23) < 1e-9, segs
    return 'OK'
