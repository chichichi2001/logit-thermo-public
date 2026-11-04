"""切り詰めつきサンプリング（Min-p・top-p）でも、スコア関数の恒等式が「切り詰めたあとの候補の確率 q^S」で成り立つかを、
小さな作り物の自己回帰モデルで、全部の文を数え上げて（厳密に）確かめる。CPU、乱数の作り物だけ（実データは使わない）。

  python3 scripts/check_truncated_identity.py

確かめること
  (1) h ＝ 0 での微分：数え上げた E[Y](h) の中心差分 と、恒等式 Cov(Y, Σ(c − q^S))/T が一致する。q を切り詰め前のまま使うと一致しない。
  (2) 有限の h：どの文のどの歩でもマージンの条件（Min-p は マージン ＞ 2|h|/T）が満たされる範囲では、E[Y](h) は h になめらか（候補集合が変わらない）。
      条件を破る h では、候補集合が変わる歩が出て、E[Y](h) が飛ぶ（1 次の予言の外）。
  (3) 割る型の罰（Hugging Face と同じ）でも (1) が成り立つ。
"""
import itertools
import numpy as np

V, L, T = 5, 5, 0.7
rng = np.random.default_rng(11)
TABLE = rng.normal(0, 2.0, (V, V, V))          # 直前 2 トークン → logit（作り物のモデル）
PROMPT = [0, 1, 2]


def cands(hist):
    """直前の 2 トークンのあとに、それまで（書き出しを含む）続いたことのあるトークン（論文の写しの候補と同じ作り）。"""
    a, b = hist[-2], hist[-1]
    return {hist[i + 2] for i in range(len(hist) - 2) if hist[i] == a and hist[i + 1] == b}


def raw_logits(hist):
    return TABLE[hist[-2], hist[-1]].copy()


def penalized(hist, h, kind):
    z = raw_logits(hist)
    if kind == 'add':
        for v in cands(hist):
            z[v] += h
    else:                                                        # 割る型：θ ＝ e^h、見たことのあるトークン
        th = np.exp(h)
        for v in set(hist):
            z[v] = z[v] / th if z[v] > 0 else z[v] * th
    return z


def truncate(zt, rule):
    """zt ＝ z/T（温度をかけたあと）。→ 残すトークンの印。"""
    p = np.exp(zt - zt.max()); p /= p.sum()
    if rule == 'none':
        return np.ones(V, bool)
    if rule == 'minp':                                           # p_i ≥ r·p_max
        return p >= 0.12 * p.max()
    if rule == 'topp':                                           # 大きい順に累積 0.8 まで（境目のトークンを含む）
        o = np.argsort(-zt, kind='stable'); c = np.cumsum(p[o])
        keep = np.zeros(V, bool); keep[o[:int(np.searchsorted(c, 0.8)) + 1]] = True
        return keep
    raise ValueError(rule)


def step_probs(hist, h, kind, rule):
    zt = penalized(hist, h, kind) / T
    keep = truncate(zt, rule)
    p = np.where(keep, np.exp(zt - zt.max()), 0.0); p /= p.sum()
    return p, keep


def margin_minp(hist, h, kind):
    """Min-p の φ_i ＝ z_i/T − max z/T − ln r の、0 からの距離の最小（h での logit について）。"""
    zt = penalized(hist, h, kind) / T
    phi = zt - zt.max() - np.log(0.12)
    return float(np.min(np.abs(phi)))


def enumerate_all(h, kind, rule, Yfn, use_trunc_q=True):
    """全部の続き（V^L 通り）を数え上げて E[Y] と、恒等式の Cov(Y, σ)/T の材料を返す。"""
    EY, EYs, Es = 0.0, 0.0, 0.0
    worst_margin = np.inf
    for seq in itertools.product(range(V), repeat=L):
        hist = list(PROMPT); logp, sig = 0.0, 0.0
        for x in seq:
            p, keep = step_probs(hist, h, kind, rule)
            if p[x] == 0:
                logp = -np.inf; break
            logp += np.log(p[x])
            C = cands(hist)
            if kind == 'add':
                pc = p if use_trunc_q else (lambda q: q / q.sum())(np.exp(penalized(hist, h, kind) / T - (penalized(hist, h, kind) / T).max()))
                qS = sum(pc[v] for v in C)
                sig += ((x in C) - qS)                          # ∂log p_t(x)/∂h × T
            else:
                z0 = raw_logits(hist); seen = set(hist)
                a = np.array([-abs(z0[v]) if v in seen else 0.0 for v in range(V)])
                pc = p if use_trunc_q else (lambda q: q / q.sum())(np.exp(z0 / T - (z0 / T).max()))
                sig += a[x] - float(np.dot(pc, a))
            if rule == 'minp':
                worst_margin = min(worst_margin, margin_minp(hist, h, kind))
            hist.append(x)
        if logp == -np.inf:
            continue
        P = np.exp(logp); Y = Yfn(hist)
        EY += P * Y; EYs += P * Y * sig; Es += P * sig
    return EY, (EYs - EY * Es) / T, worst_margin


def Y_cand(hist):                                               # 候補を選んだ回数
    return sum(hist[len(PROMPT) + i] in cands(hist[:len(PROMPT) + i]) for i in range(L))


def Y_rep(hist):                                                # 続きの中に同じ 3 つ組が 2 回出たか（0/1、なめらかでない統計）
    s = hist[len(PROMPT):]
    g = [tuple(s[i:i + 3]) for i in range(len(s) - 2)]
    return float(len(g) != len(set(g)))


if __name__ == '__main__':
    eps = 1e-5
    print('(1)(3) h ＝ 0 での微分：中心差分 対 恒等式（切り詰めた q）／（切り詰める前の q）')
    for kind in ('add', 'mult'):
        for rule in ('none', 'minp', 'topp'):
            for nm, Yf in (('Y_cand', Y_cand), ('Y_rep', Y_rep)):
                fd = (enumerate_all(eps, kind, rule, Yf)[0] - enumerate_all(-eps, kind, rule, Yf)[0]) / (2 * eps)
                _, g_tr, m0 = enumerate_all(0.0, kind, rule, Yf, True)
                _, g_un, _ = enumerate_all(0.0, kind, rule, Yf, False)
                print(f'  {kind:4s} {rule:4s} {nm:6s}  差分 {fd:+.6f}  恒等式（切り詰めた q）{g_tr:+.6f}  （切り詰める前の q）{g_un:+.6f}'
                      + (f'  h=0 の最小マージン {m0:.3f}' if rule == 'minp' else ''))
    print('(2) Min-p、足す型：有限の h での E[Y_cand] と 1 次の予言、最小マージンと 2|h|/T')
    E0, g0, _ = enumerate_all(0.0, 'add', 'minp', Y_cand)
    for h in (-0.02, -0.05, -0.1, -0.2, -0.4, -0.8):
        Eh, _, m = enumerate_all(h, 'add', 'minp', Y_cand)
        _, _, m0 = enumerate_all(0.0, 'add', 'minp', Y_cand)
        print(f'  h={h:+.2f}  ΔE={Eh - E0:+.5f}  h·g={h * g0:+.5f}  比={((Eh - E0) / (h * g0)):.3f}  h=0 の最小マージン {m0:.3f}  2|h|/T={2 * abs(h) / T:.3f}  条件 {"満たす" if m0 > 2 * abs(h) / T else "破る"}')


def set_change_stats(h, kind='add'):
    """Min-p：h ＝ 0 の文の重みで、(a) 候補集合が 0 と h で変わる歩の割合、(b) マージン ≤ 2|h|/T の歩の割合、
    (c) マージン ＞ 2|h|/T なのに集合が変わった歩の数（定理どおりなら 0）。"""
    w_change = w_low = w_tot = 0.0; violations = 0
    for seq in itertools.product(range(V), repeat=L):
        hist = list(PROMPT); logp = 0.0
        for x in seq:
            p0, k0 = step_probs(hist, 0.0, kind, 'minp')
            if p0[x] == 0:
                logp = -np.inf; break
            _, kh = step_probs(hist, h, kind, 'minp')
            m0 = margin_minp(hist, 0.0, kind)
            P = np.exp(logp)
            ch = bool(np.any(k0 != kh)); low = m0 <= 2 * abs(h) / T
            w_change += P * ch; w_low += P * low; w_tot += P
            violations += int(ch and not low)
            logp += np.log(p0[x]); hist.append(x)
    return w_change / w_tot, w_low / w_tot, violations


if __name__ == '__main__':
    print('(4) Min-p、足す型：歩ごとのマージンの条件と、候補集合が実際に変わる歩（h ＝ 0 の文の重みでの割合）')
    for h in (-0.02, -0.05, -0.1, -0.2, -0.4, -0.8):
        c, lo, viol = set_change_stats(h)
        print(f'  h={h:+.2f}  集合が変わる歩 {c:.4f}  マージン ≤ 2|h|/T の歩 {lo:.4f}  条件を満たすのに変わった歩 {viol}')


def E_frozen_sets(h, kind, rule, Yfn):
    """候補集合を h ＝ 0 のものに固定したまま（なめらかな続き）、確率だけ h で出した E[Y]。実際の E[Y](h) との差が、集合が変わる効き。"""
    EY = 0.0
    for seq in itertools.product(range(V), repeat=L):
        hist = list(PROMPT); logp = 0.0
        for x in seq:
            _, k0 = step_probs(hist, 0.0, kind, rule)
            zt = penalized(hist, h, kind) / T
            p = np.where(k0, np.exp(zt - zt.max()), 0.0); p /= p.sum()
            if p[x] == 0:
                logp = -np.inf; break
            logp += np.log(p[x]); hist.append(x)
        if logp > -np.inf:
            EY += np.exp(logp) * Yfn(hist)
    return EY


if __name__ == '__main__':
    print('(5) Min-p、足す型：実際の ΔE と、集合を固定したなめらかな ΔE（差が集合の入れかわりの効き）')
    E0 = enumerate_all(0.0, 'add', 'minp', Y_cand)[0]
    for h in (-0.1, -0.15, -0.2, -0.4, -0.8):
        Eh = enumerate_all(h, 'add', 'minp', Y_cand)[0]; Ef = E_frozen_sets(h, 'add', 'minp', Y_cand)
        print(f'  h={h:+.2f}  実際 {Eh - E0:+.5f}  集合を固定 {Ef - E0:+.5f}  入れかわりの効き {Eh - Ef:+.5f}')
