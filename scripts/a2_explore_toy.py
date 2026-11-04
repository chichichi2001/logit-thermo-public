"""探索 8.36：厳密に解けるひな形「写しのはしご」（2026-10-31 14:5x、CPU、判定しない）。
docs/A2_EQUATION_CATALOG.md の式 33。サンプリングをしないで、マルコフ連鎖の行列の計算だけで、融点・ΔH・アレニウス・クラペイロン・
ゆらぎと応答・アヴラミの n を厳密に出す。いくつかの量で実験（Pythia-1B）に合わせ、ほかの量を予言としてくらべる。

ひな形：
- 状態 k ＝ いま続けて写している歩の数（0〜K、K で頭打ち）。
- k ＝ 0（新しいトークンを出した直後）では、写しの候補がある確率 ρ(t)（文の位置 t で増えてよい：ρ(t) ＝ ρ∞ × (1 − e^(−(t＋1)/τ))、τ ＝ 0 なら一定）。
- 候補があるとき、候補の logit は z_k ＝ a ＋ b × min(k, K)（続けて写すほど写しが強くなる ＝ 自己強化）。場 h はここに足す。
- 候補以外（背景）：いちばん上のトークン 1 個が logit 0、ほかに B 個が logit −Δ。Z_b(T) ＝ 1 ＋ B e^(−Δ/T)。
- 写す確率 p_k ＝ e^((z_k＋h)/T) ÷ (e^((z_k＋h)/T) ＋ Z_b(T))。写せば k → min(k＋1, K)、写さなければ k → 0。
- 相：k ≥ m を結晶、k < m を液体（m ＝ 8）。凍る ＝ m に届く、溶ける ＝ 結晶から 0 に落ちる。
- エネルギー（熱量計）：選んだトークンの −logit（写し：−z_k、背景のいちばん上：0、ほか：Δ）。場は入れない。
"""
import numpy as np

TY_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


class Toy:
    def __init__(self, a, b, Delta, B, rho, K=32, m=8, tau=0.0):
        self.a, self.b, self.Delta, self.B, self.rho, self.K, self.m, self.tau = a, b, Delta, B, rho, K, m, tau

    def z(self):
        return self.a + self.b * np.minimum(np.arange(self.K + 1), self.K)

    def Zb(self, T):
        return 1.0 + self.B * np.exp(-self.Delta / T)

    def p(self, T, h=0.0):
        e = np.exp((self.z() + h) / T)
        return e / (e + self.Zb(T))

    def rho_t(self, t):
        return self.rho if self.tau <= 0 else self.rho * (1 - np.exp(-(t + 1) / self.tau))

    def P(self, T, h=0.0, rho=None):
        """遷移行列（行 ＝ いま、列 ＝ 次）。"""
        rho = self.rho if rho is None else rho
        K = self.K; p = self.p(T, h)
        M = np.zeros((K + 1, K + 1))
        M[0, 1] = rho * p[0]; M[0, 0] = 1 - rho * p[0]
        for k in range(1, K + 1):
            M[k, min(k + 1, K)] += p[k]; M[k, 0] += 1 - p[k]
        return M

    def stationary(self, T, h=0.0):
        M = self.P(T, h)
        w, v = np.linalg.eig(M.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = pi / pi.sum()
        return pi

    def rates(self, T, h=0.0):
        """→ k_f（液体の 1 歩あたり凍る率）、k_m（結晶の 1 歩あたり溶ける率）。"""
        pi = self.stationary(T, h); p = self.p(T, h); m = self.m
        liq, cry = pi[:m].sum(), pi[m:].sum()
        kf = pi[m - 1] * p[m - 1] / liq
        km = (pi[m:] * (1 - p[m:])).sum() / cry
        return kf, km

    def phase_means(self, T, h=0.0):
        """→ 液体・結晶それぞれの、候補を選んだ割合と、選んだトークンのエネルギーの平均（場なしのエネルギー）。"""
        pi = self.stationary(T, h); p = self.p(T, h); z = self.z(); m = self.m
        Zb = self.Zb(T); top = 1.0 / Zb; other = 1 - top                   # 背景の中の割合
        pc = p.copy(); pc[0] *= self.rho                                    # 実際に写す確率
        Ecopy = -z
        Ebg = top * 0.0 + other * self.Delta
        out = {}
        for nm, sl in (('liq', slice(0, m)), ('cry', slice(m, None))):
            w = pi[sl]
            out[f'm_{nm}'] = float((w * pc[sl]).sum() / w.sum())
            out[f'E_{nm}'] = float((w * (pc[sl] * Ecopy[sl] + (1 - pc[sl]) * Ebg)).sum() / w.sum())
        return out

    def vh(self, temps=TY_TEMPS, h=0.0):
        k = np.array([self.rates(T, h) for T in temps])
        x = 1 / np.array(temps); y = np.log(k[:, 0] / k[:, 1])
        b, a = np.polyfit(x, y, 1)
        bf, af = np.polyfit(x, np.log(k[:, 0]), 1); bm, am = np.polyfit(x, np.log(k[:, 1]), 1)
        return dict(dH=float(b), dS=float(-a), Tm=float(-b / a) if (b > 0 and a < 0) else float('nan'),
                    Ea_f=float(-bf), Ea_m=float(-bm), lnA_f=float(af), lnA_m=float(am))

    # ---- 有限の長さの文（t ＝ 0 で k ＝ 0 から）：ゆらぎと応答・アヴラミ ----
    def gen_fn(self, T, h, L, lam=0.0, mu=0.0):
        """ln E[exp(λ M ＋ μ Q)]（M ＝ 候補を選んだ歩の数、Q ＝ Σ q_t）。厳密（行列の積）。"""
        K = self.K; p = self.p(T, h)
        v = np.zeros(K + 1); v[0] = 1.0; logs = 0.0
        for t in range(L):
            rho = self.rho_t(t)
            W = np.zeros((K + 1, K + 1))
            # k ＝ 0：候補あり（確率 ρ、q ＝ p0）・なし（1 − ρ、q ＝ 0）
            W[0, 1] += rho * p[0] * np.exp(lam + mu * p[0])
            W[0, 0] += rho * (1 - p[0]) * np.exp(mu * p[0]) + (1 - rho)
            for k in range(1, K + 1):
                W[k, min(k + 1, K)] += p[k] * np.exp(lam + mu * p[k]); W[k, 0] += (1 - p[k]) * np.exp(mu * p[k])
            v = v @ W; s = v.sum(); logs += np.log(s); v = v / s
        return logs

    def moments(self, T, h, L, eps=3e-4):
        """→ ⟨M⟩、Var(M)、Cov(M, M − Q)（有限差分、行列の計算は厳密）。"""
        f = lambda l, u: self.gen_fn(T, h, L, l, u)
        f00 = f(0, 0); fp0 = f(eps, 0); fm0 = f(-eps, 0)
        mean = (fp0 - fm0) / (2 * eps); var = (fp0 - 2 * f00 + fm0) / eps ** 2
        # Cov(M, Q) ＝ ∂²/∂λ∂μ
        cmq = (f(eps, eps) - f(eps, -eps) - f(-eps, eps) + f(-eps, -eps)) / (4 * eps ** 2)
        return float(mean), float(var), float(var - cmq)

    def survival(self, T, h, L):
        """はじめの液体が、まだ一度も結晶（k ≥ m）になっていない確率 S(t)。"""
        K, m = self.K, self.m; p = self.p(T, h)
        v = np.zeros(K + 1); v[0] = 1.0; S = []
        for t in range(L):
            rho = self.rho_t(t)
            nv = np.zeros(K + 1)
            nv[1] += v[0] * rho * p[0]; nv[0] += v[0] * (1 - rho * p[0])
            for k in range(1, m):
                nv[k + 1] += v[k] * p[k]; nv[0] += v[k] * (1 - p[k])
            v = nv; v[m:] = 0.0                                      # 結晶になったら吸い込む
            S.append(v.sum())
        return np.array(S)


def avrami_from_S(S, lo=0.05, hi=1.5):
    t = np.arange(1, len(S) + 1); H = -np.log(np.clip(S, 1e-300, 1))
    ok = (H >= lo) & (H <= hi)
    if ok.sum() < 5:
        return float('nan')
    return float(np.polyfit(np.log(t[ok]), np.log(H[ok]), 1)[0])


def toy_report(toy, temps=TY_TEMPS, L=512, dh=0.05):
    r = toy.vh(temps)
    rp, rm = toy.vh(temps, dh), toy.vh(temps, -dh)
    r['dTm_dh'] = (rp['Tm'] - rm['Tm']) / (2 * dh)
    Tm = r['Tm']
    pm = toy.phase_means(Tm)
    r.update(pm)
    r['dH_cal'] = pm['E_liq'] - pm['E_cry']
    r['n'] = r['dH'] / r['dH_cal'] if r['dH_cal'] > 0 else float('nan')
    r['dm'] = pm['m_cry'] - pm['m_liq']
    r['pred_clap'] = r['n'] * r['dm'] * Tm / r['dH']
    r['clap_ratio'] = r['dTm_dh'] / r['pred_clap']
    # 場の α と ΔM_速さ（温度ごと）
    al, dMk = [], []
    for T in temps:
        kfp, kmp = toy.rates(T, dh); kfm, kmm = toy.rates(T, -dh)
        gf = (np.log(kfp) - np.log(kfm)) / (2 * dh); gm = (np.log(kmp) - np.log(kmm)) / (2 * dh)
        al.append(gf / (gf - gm)); dMk.append(T * (gf - gm))
    r['alpha_h'] = al; r['dM_kin'] = dMk
    # ゆらぎと応答（有限の長さ L、温度ごと）
    R, Rid = [], []
    for T in temps:
        mean0, var0, cid = toy.moments(T, 0.0, L)
        mp = toy.moments(T, dh, L)[0]; mm = toy.moments(T, -dh, L)[0]
        slope = (mp - mm) / (2 * dh)
        R.append(slope / (var0 / T)); Rid.append(slope / (cid / T))     # Rid は恒等式の確かめ（1 になるはず）
    r['R'] = R; r['R_identity'] = Rid
    r['avrami_n'] = [avrami_from_S(toy.survival(T, 0.0, L)) for T in temps]
    return r


def self_test_toy():
    toy = Toy(a=0.5, b=0.08, Delta=2.0, B=50, rho=0.25)
    pi = toy.stationary(0.5)
    assert abs(pi.sum() - 1) < 1e-12 and (pi >= -1e-12).all()
    # 定常分布は P で動かない
    assert np.allclose(pi @ toy.P(0.5), pi, atol=1e-12)
    # 恒等式：d⟨M⟩/dh ＝ Cov(M, M − Q)/T（ひな形の中で厳密に成り立つこと）
    L, T, dh = 120, 0.5, 1e-3
    mean0, var0, cid = toy.moments(T, 0.0, L)
    slope = (toy.moments(T, dh, L)[0] - toy.moments(T, -dh, L)[0]) / (2 * dh)
    assert abs(slope / (cid / T) - 1) < 3e-3, slope / (cid / T)   # 差分の誤差のぶん
    # 生き残り：一定の ρ・はしごなし（m ＝ 1）なら幾何分布（n ＝ 1）
    t1 = Toy(a=0.0, b=0.0, Delta=2.0, B=50, rho=0.01, m=1, K=2)
    S = t1.survival(0.5, 0.0, 2000)
    assert abs(avrami_from_S(S) - 1) < 0.02
    return 'OK'
