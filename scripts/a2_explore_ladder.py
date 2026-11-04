"""探索 8.37：はしごを「測って」作るひな形（2026-10-31 14:5x、CPU、判定しない）。8.36 の精密版。
材料：8.30 の場の下見（Pythia-1B・OLMo-2-1B、h ＝ −0.3・0・＋0.3、温度 0.4〜0.7、サンプル 4 × 記事 40 × 512 歩）の、候補があったか・候補を選んだか。

1) ものさしをそろえる：歩 t の前の「続けて候補を選んだ歩の数」k_t（生成のはじめは 0）。k_t ≥ m を結晶とする（m ＝ 8、感度に 4・16）。
   データもひな形も、同じものさしで、凍る回・溶ける回・液体と結晶の歩を数える（率 ＝ 回の合計 ÷ 歩の合計、A2-8 と同じ作り）。
2) はしごを測る（h ＝ 0 のデータだけ、温度ごと）：
   ρ_k ＝ P(候補がある | k)、p_k ＝ P(候補を選ぶ | 候補がある, k)（k ＝ 0〜K、K ＝ 32 で頭打ち）。
   ρ_0 は文の位置による変わり方も測る（32 歩ずつの箱）。
   ひな形 ＝ この ρ・p で動くマルコフ連鎖（はじめ k ＝ 0、512 歩）。分布を 1 歩ずつ動かして、数えの期待値を厳密に出す（サンプリングなし）。
3) 予言：
   (a) h ＝ 0：ものさしをそろえた T_m・ΔH・アレニウス・⟨M⟩ を、データと比べる（はしごの中身はデータから測ったので、合えば「k だけで決まるマルコフ」で足りている）。
   (b) h ＝ ±0.3：p_k(h) ＝ p_k e^(h/T) ÷ (1 − p_k ＋ p_k e^(h/T))（h ＝ 0 の p_k から式で。ρ は変えない）で、±0.3 の数えを予言し、8.30 の実測と比べる。
       注意：本当は各歩の候補の確率 q に式を当てるべきところを、その平均 p_k に当てている（q のばらつきのぶん、ずれうる）。8.35 の q で直せる。
   (c) 記事ごとの違い：記事ごとに ρ_0 の大きさだけを変えた（位置の形は共通）ひな形を記事の数だけ解き、⟨M⟩ と Var(M) を混ぜる。
       ゆらぎと応答の比 R ＝ (d⟨M⟩/dh) ÷ (Var(M)/T) を、記事の違いなし・ありで出し、8.34 の実測（0.01）と比べる。
   (d) アヴラミ：はじめての結晶までの生き残り。ρ_0 の位置による変わり方を入れたひな形と、入れない（一定）ひな形で n を出し、データと比べる。
"""
import numpy as np

LD_K = 32
LD_M = 8
LD_POSBIN = 32
LD_L = 512


def streaks(pick):
    """pick：(…, L) → k（…, L）：歩 t の前の、続けて候補を選んだ歩の数（K で頭打ちしない生の値）。"""
    k = np.zeros(pick.shape, int)
    run = np.zeros(pick.shape[:-1], int)
    for t in range(pick.shape[-1]):
        k[..., t] = run
        run = np.where(pick[..., t], run + 1, 0)
    return k


def measure_ladder(cand, pick, K=LD_K):
    """cand・pick：(S, nD, L)（ある温度）→ ρ_k・p_k（k ＝ 0〜K）、ρ_0 の位置の箱ごとの値、記事ごとの ρ_0 の比。"""
    k = np.minimum(streaks(pick), K)
    S, nD, L = pick.shape
    n = np.bincount(k.ravel(), minlength=K + 1).astype(float)
    c = np.bincount(k.ravel(), weights=cand.ravel().astype(float), minlength=K + 1)
    y = np.bincount(k.ravel(), weights=(cand & pick).ravel().astype(float), minlength=K + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rho = c / n; p = y / c
    # 数が少ない段は、ひとつ下の段の値で埋める
    for i in range(1, K + 1):
        if not np.isfinite(rho[i]) or n[i] < 20:
            rho[i] = rho[i - 1]
        if not np.isfinite(p[i]) or c[i] < 20:
            p[i] = p[i - 1]
    nb = L // LD_POSBIN
    z = (k == 0)
    pos = np.arange(L) // LD_POSBIN
    rho0_pos = np.array([cand[..., pos == b][z[..., pos == b]].mean() if z[..., pos == b].any() else rho[0] for b in range(nb)])
    rho0_doc = np.array([cand[:, j][z[:, j]].mean() if z[:, j].any() else rho[0] for j in range(nD)]) / rho[0]
    return dict(rho=rho, p=p, n=n, rho0_pos=rho0_pos, rho0_doc=rho0_doc)


def ruler_counts(pick, m=LD_M):
    """データ：(…, L) → [液体の歩, 凍る回, 結晶の歩, 溶ける回]（歩 t → t＋1、t ＝ 0〜L−2）と M（歩 16〜495 の選んだ数）。"""
    x = streaks(pick) >= m
    a, b = x[..., :-1], x[..., 1:]
    c = np.stack([(~a).sum(-1), (~a & b).sum(-1), a.sum(-1), (a & ~b).sum(-1)], -1).astype(float)
    return c, pick[..., 16:496].sum(-1).astype(float)


def p_field(p, T, h):
    e = np.exp(h / T)
    return p * e / (1 - p + p * e)


def chain_counts(rho, p, rho0_t, m=LD_M, L=LD_L, lam=0.0):
    """マルコフ連鎖（はじめ k ＝ 0）→ 数えの期待値 [液体の歩, 凍る回, 結晶の歩, 溶ける回]、⟨M⟩（歩 16〜495）。
    lam ≠ 0 なら、M に exp(λ M) の重みをかけた（母関数用）ln E[exp(λ M)] も返す。"""
    K = len(p) - 1
    v = np.zeros(K + 1); v[0] = 1.0
    out = np.zeros(4); EM = 0.0; logs = 0.0
    for t in range(L - 1 if lam == 0.0 else L):
        r = rho.copy(); r[0] = rho0_t[t]
        go = r * p                                     # 写して 1 段上がる確率
        if lam == 0.0:
            liq = v[:m].sum(); cry = v[m:].sum()
            out[0] += liq; out[2] += cry
            out[1] += v[m - 1] * go[m - 1]
            out[3] += (v[m:] * (1 - go[m:])).sum()
            if 16 <= t < 496:
                EM += (v * go).sum()
        nv = np.zeros(K + 1)
        wcopy = np.exp(lam) if (lam != 0.0 and 16 <= t < 496) else 1.0
        nv[1:] += v[:-1] * go[:-1] * wcopy
        nv[K] += v[K] * go[K] * wcopy
        nv[0] += (v * (1 - go)).sum()
        if lam != 0.0:
            s = nv.sum(); logs += np.log(s); nv /= s
        v = nv
    return (out, EM) if lam == 0.0 else logs


def chain_M_moments(rho, p, rho0_t, eps=1e-3):
    f0 = chain_counts(rho, p, rho0_t, lam=1e-300)  # ほぼ 0：正規化だけ
    fp = chain_counts(rho, p, rho0_t, lam=eps); fm = chain_counts(rho, p, rho0_t, lam=-eps)
    mean = (fp - fm) / (2 * eps); var = (fp - 2 * f0 + fm) / eps ** 2
    return float(mean), float(var)


def vh_fit(cnt, temps, min_ev=10):
    """cnt：(nT, 4) → ln(k_f/k_m) を 1/T にまっすぐ当てはめ（両方 min_ev 回以上の温度、回数の重み）。アレニウスも。"""
    ok = (cnt[:, 1] >= min_ev) & (cnt[:, 3] >= min_ev)
    if ok.sum() < 3:
        return dict(ok=False)
    kf = cnt[ok, 1] / cnt[ok, 0]; km = cnt[ok, 3] / cnt[ok, 2]
    x = 1 / np.asarray(temps)[ok]; w = 1 / (1 / cnt[ok, 1] + 1 / cnt[ok, 3])
    X = np.stack([np.ones_like(x), x], 1)
    a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * np.log(kf / km)))
    bf = np.polyfit(x, np.log(kf), 1, w=np.sqrt(cnt[ok, 1]))[0]; bm = np.polyfit(x, np.log(km), 1, w=np.sqrt(cnt[ok, 3]))[0]
    return dict(ok=True, dH=float(b), Tm=float(-b / a) if (b > 0 and a < 0) else float('nan'), Ea_f=float(-bf), Ea_m=float(-bm),
                lnK=np.log(kf / km).tolist(), idx=np.flatnonzero(ok).tolist())


def rho0_profile(lad, L=LD_L, positional=True):
    if not positional:
        return np.full(L, lad['rho'][0])
    prof = np.repeat(lad['rho0_pos'], LD_POSBIN)
    return np.concatenate([prof, np.full(max(0, L - len(prof)), prof[-1])])[:L]


def first_freeze_data(pick, m=LD_M):
    x = streaks(pick) >= m
    hit = x.any(-1)
    return np.where(hit, x.argmax(-1), -1)


def chain_survival(rho, p, rho0_t, m=LD_M, L=LD_L):
    K = len(p) - 1
    v = np.zeros(K + 1); v[0] = 1.0; S = []
    for t in range(L):
        r = rho.copy(); r[0] = rho0_t[t]; go = r * p
        nv = np.zeros(K + 1)
        nv[1:] += v[:-1] * go[:-1]; nv[0] += (v * (1 - go)).sum()
        nv[m:] = 0.0
        v = nv; S.append(v.sum())
    return np.array(S)


def avrami(S, lo=0.05, hi=1.5):
    t = np.arange(1, len(S) + 1); H = -np.log(np.clip(S, 1e-300, 1)); ok = (H >= lo) & (H <= hi)
    return float(np.polyfit(np.log(t[ok]), np.log(H[ok]), 1)[0]) if ok.sum() >= 5 else float('nan')


def self_test_ladder():
    # streaks
    pk = np.array([[0, 1, 1, 0, 1, 1, 1, 0]], bool)
    assert streaks(pk)[0].tolist() == [0, 0, 1, 2, 0, 1, 2, 3]
    # 連鎖の数えと、同じ連鎖から引いた列の数えが合う
    rng = np.random.default_rng(4)
    K = LD_K; rho = np.full(K + 1, 0.95); rho[0] = 0.3; p = np.linspace(0.6, 0.99, K + 1)
    r0 = np.full(LD_L, 0.3)
    N = 3000; pick = np.zeros((N, LD_L), bool); cand = np.zeros((N, LD_L), bool); k = np.zeros(N, int)
    for t in range(LD_L):
        kk = np.minimum(k, K)
        c = rng.random(N) < np.where(kk == 0, r0[t], rho[kk]); y = c & (rng.random(N) < p[kk])
        cand[:, t] = c; pick[:, t] = y; k = np.where(y, k + 1, 0)
    cnt, M = ruler_counts(pick)
    ec, EM = chain_counts(rho, p, r0)
    assert np.allclose(cnt.mean(0), ec, rtol=0.05), (cnt.mean(0), ec)
    assert abs(M.mean() - EM) / EM < 0.02, (M.mean(), EM)
    lad = measure_ladder(cand[None], pick[None])
    assert abs(lad['rho'][0] - 0.3) < 0.01 and abs(lad['p'][5] - p[5]) < 0.03, (lad['rho'][0], lad['p'][5], p[5])
    mean, var = chain_M_moments(rho, p, r0)
    assert abs(mean - M.mean()) / M.mean() < 0.02 and abs(var - M.var()) / M.var() < 0.1, (mean, M.mean(), var, M.var())
    return 'OK'
