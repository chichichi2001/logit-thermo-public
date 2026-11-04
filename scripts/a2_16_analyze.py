"""A2-16 の分析 v0.2：場の効きはものさしの見かけではないか（HQ1）と、1 トークンあたりのクラペイロンは閉じるか（HQ3）。
データ（ゲート前の材料 G-pre、Drive `A2_gpre/`、7 モデル × h ＝ −0.3・0・＋0.3）を開く前に書いた（2026-11-01 06:xx〜07:xx JST）。CPU。
scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases・fit_vh）・a2_13_analyze.py（A8_TU ＝ 7 つの温度）・a2_explore_ladder.py（streaks・ruler_counts）を
実行したのと同じ名前空間で実行する（A8_TU が 7 つの温度でなければ止まる）。

2 つのものさし：
  窓（A2-8 と同じ）：写しの印（4 つ組のくり返し）の 32 歩の窓、入口 0.5・出口 0.25。相がつくのは歩 16〜496。
  続けて写す歩の数（8.37 と同じ）：歩 t の前に、続けて候補を選んだ歩の数 k_t ≥ 8 なら結晶。歩 0〜511。
数え（記事ごと・温度ごと、16 列）：[液体の歩, 凍る回, 結晶の歩, 溶ける回, ΣU 液体, 液体の歩, ΣU 結晶, 結晶の歩, 0, 0,
  液体で候補を選んだ歩, 結晶で候補を選んだ歩, ΣU 選んだ歩, 選んだ歩, ΣU 選ばなかった歩, 選ばなかった歩]（U は温度 1・場なしの −ln p）。

量（モデルごと）：
  腕ごと・ものさしごとに A2-8 と同じファントホッフの当てはめ（fit_vh）→ T_m・ΔH_vH・熱量計の ΔH（使う温度の単純平均）。
  場の傾き ＝ (T_m(＋0.3) − T_m(−0.3)) ÷ 0.6（ものさしごと。3 点の直線の傾きと同じ。h ＝ 0 の T_m は使わない）。
  クラペイロンの予言（h ＝ 0、窓）：Δm ＝ 使う温度ごとの（結晶で候補を選んだ割合 − 液体で選んだ割合）の単純平均（熱量計の ΔH と同じ平均のしかた）。
    予言の傾き ＝ T_m(0) × Δm ÷ 熱量計の ΔH。比 R ＝ 場の傾き（窓）÷ 予言の傾き。
    R は「場から見た協同性の単位 ÷ 温度（エネルギー）から見た協同性の単位」とも読める：平衡の 2 つの状態で、写した数 M が場の共役な量なら R ≈ 1。
区間：記事の選び直し 1,000 回（種 20261143。モデルの中で 3 つの腕と 2 つのものさしに同じ選び方。使う温度は点のときと同じ）。
  選び直しで、ある量が出ない回が 1 割を超えたら、その量の言葉は「はっきりしない」。
"""
import numpy as np

G16_MODELS = ['pythia-410m', 'pythia-1.4b', 'gpt-neo-1.3b', 'smollm2-1.7b', 'qwen2.5-1.5b', 'gpt2-1.5b', 'bloom-1b1']
G16_H = [('hm03', -0.3), ('h0', 0.0), ('hp03', 0.3)]
G16_TEMPS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
G16_M = 8
G16_BOOT, G16_SEED = 1000, 20261143
G16_FAIL = 0.10
G16_CLOSE = (0.8, 1.25)
G16_MAJ = 5
G16_NC = 16
# 空の作り物（本当の融点は場で動かず、液体の中で写す割合だけが場で増える）で、ものさしが場に直接反応して出す見かけの傾き。
# self_test16 の calib_empty で測った（液体で写す割合 0.1・0.2・0.3・0.5 で −0.017〜＋0.009）。絶対値の最大を切り上げて 0.02。HQ1 はこれより大きいかで決める。
G16_ART = {'w': 0.02, 's': 0.02}


def _check_temps():
    assert [round(float(t), 4) for t in A8_TU] == G16_TEMPS, ('A8_TU が 7 つの温度ではない', A8_TU)


# ---------- 数え ----------
def _acc(acc, st, u, p):
    """st：相（0/1）、u：U、p：候補を選んだか（どれも同じ歩の並び）。"""
    liq, cry = st == 0, st == 1
    acc[4] += u[liq].sum(); acc[5] += liq.sum(); acc[6] += u[cry].sum(); acc[7] += cry.sum()
    acc[10] += p[liq].sum(); acc[11] += p[cry].sum()
    pk = p > 0.5
    acc[12] += u[pk].sum(); acc[13] += pk.sum(); acc[14] += u[~pk].sum(); acc[15] += (~pk).sum()


def counts_window(toks, U, pick):
    """(S, L) の 1 つの記事・1 つの温度 → 長さ 16。"""
    acc = np.zeros(G16_NC)
    for s in range(toks.shape[0]):
        st, t = phases(copy_marks(toks[s]))
        a, b = st[:-1], st[1:]
        acc[0] += (a == 0).sum(); acc[1] += ((a == 0) & (b == 1)).sum(); acc[2] += (a == 1).sum(); acc[3] += ((a == 1) & (b == 0)).sum()
        _acc(acc, st, U[s, t].astype(np.float64), pick[s, t].astype(np.float64))
    return acc


def counts_streak(U, pick, m=G16_M):
    """同じ形。相は k_t ≥ m（歩 0〜L−1）。凍る・溶けるは歩 t → t＋1。"""
    acc = np.zeros(G16_NC)
    x = streaks(pick) >= m
    a, b = x[:, :-1], x[:, 1:]
    acc[0] = (~a).sum(); acc[1] = (~a & b).sum(); acc[2] = a.sum(); acc[3] = (a & ~b).sum()
    _acc(acc, x.astype(np.int8).ravel(), U.astype(np.float64).ravel(), pick.astype(np.float64).ravel())
    return acc


def arm_counts(toks, U, pick):
    """(7, S, nD, L) → 窓 (7, nD, 16)、続けて写す (7, nD, 16)。"""
    nT, S, nD, L = toks.shape
    cw = np.zeros((nT, nD, G16_NC)); cs = np.zeros((nT, nD, G16_NC))
    for k in range(nT):
        for j in range(nD):
            cw[k, j] = counts_window(toks[k, :, j], U[k, :, j], pick[k, :, j])
            cs[k, j] = counts_streak(U[k, :, j], pick[k, :, j])
    return cw, cs


def load_arm(base, name, arm, h):
    Z = np.load(f'{base}/{name}@{arm}.npz')
    assert [round(float(t), 4) for t in Z['temps']] == G16_TEMPS, (name, arm)
    assert abs(float(Z['h']) - h) < 1e-9, (name, arm, float(Z['h']))
    d = {k: Z[k] for k in ('toks', 'pick', 'U', 'q0', 'q', 'cand')}
    sh = d['toks'].shape
    assert sh[0] == 7 and sh[1] == 4 and sh[3] == 512 and all(d[k].shape == sh for k in d), (name, arm, {k: d[k].shape for k in d})
    assert not (d['pick'] & ~d['cand']).any(), (name, arm, '候補がない歩で候補を選んでいる')
    return d, np.asarray(Z['use']).tolist()


# ---------- 点 ----------
def _ok_tm(f):
    return bool(f.get('ok')) and np.isfinite(f.get('Tm', float('nan')))


def point16(C, use=None):
    """C：{(腕, ものさし): (7, 16)}（記事を足したもの）。use：{(腕, ものさし): 使う温度}（None なら点で決める）→ 量、使う温度。"""
    fits, uses = {}, {}
    for key, tot in C.items():
        f = fit_vh(tot[:, :10], temps_use=None if use is None else use[key])
        fits[key] = f; uses[key] = f['use_idx']
    res = {}
    for r in ('w', 's'):
        for a, _ in G16_H:
            f = fits[(a, r)]
            res[f'Tm_{a}_{r}'] = float(f['Tm']) if _ok_tm(f) else float('nan')
            res[f'dHvH_{a}_{r}'] = float(f['dH_vH']) if f.get('ok') else float('nan')
            res[f'dHcal_{a}_{r}'] = float(f['dH_cal']) if f.get('ok') else float('nan')
        res[f'slope_{r}'] = (res[f'Tm_hp03_{r}'] - res[f'Tm_hm03_{r}']) / 0.6
        res[f'curv_{r}'] = res[f'Tm_h0_{r}'] - (res[f'Tm_hp03_{r}'] + res[f'Tm_hm03_{r}']) / 2      # 記録：h ＝ 0 が両腕の中点からどれだけずれるか
    f0, t0 = fits[('h0', 'w')], C[('h0', 'w')]
    k = np.array(uses[('h0', 'w')], int)
    for key in ('n', 'dm', 'pred', 'R', 'U_pick', 'U_nopick', 'cal_over_split'):
        res[key] = float('nan')
    if _ok_tm(f0) and f0['dH_cal'] > 0 and len(k):
        with np.errstate(divide='ignore', invalid='ignore'):
            dm_T = t0[k, 11] / t0[k, 7] - t0[k, 10] / t0[k, 5]
        if np.all(np.isfinite(dm_T)):
            dm = float(np.mean(dm_T))
            res['n'] = float(f0['dH_vH'] / f0['dH_cal'])
            res['dm'] = dm
            res['pred'] = float(f0['Tm'] * dm / f0['dH_cal'])
            res['R'] = float(res['slope_w'] / res['pred']) if res['pred'] > 0 else float('nan')
        with np.errstate(divide='ignore', invalid='ignore'):
            up, un = t0[k, 12].sum() / t0[k, 13].sum(), t0[k, 14].sum() / t0[k, 15].sum()
        res['U_pick'], res['U_nopick'] = float(up), float(un)
        # 記録：熱量計の ΔH が「Δm × 選ばなかった歩と選んだ歩の U の差」でどれだけ説明されるか
        if np.isfinite(res['dm']) and res['dm'] > 0 and un > up:
            res['cal_over_split'] = float(f0['dH_cal'] / (res['dm'] * (un - up)))
    return res, uses


# ---------- 言葉 ----------
def word_hq1(ci_w, ci_s, fail_w, fail_s, art=None):
    art = G16_ART if art is None else art
    if fail_w > G16_FAIL or fail_s > G16_FAIL or not (np.all(np.isfinite(ci_w)) and np.all(np.isfinite(ci_s))):
        return 'はっきりしない'
    if ci_w[0] > art['w'] and ci_s[0] > art['s']:
        return '両方で、ものさしの見かけより大きく上がる'
    if ci_w[1] < 0 and ci_s[1] < 0:
        return '両方で下がる'
    if (ci_w[0] > art['w'] and ci_s[1] < 0) or (ci_w[1] < 0 and ci_s[0] > art['s']):
        return '食い違う'
    return 'はっきりしない'


def word_R(ci, fail, ci_slope):
    if fail > G16_FAIL or not np.all(np.isfinite(ci)):
        if np.all(np.isfinite(ci_slope)) and ci_slope[1] < 0:
            return '逆向き（場で融点が下がる）'
        return 'はっきりしない'
    if ci[1] < 0:
        return '逆向き（場で融点が下がる）'
    if G16_CLOSE[0] <= ci[0] and ci[1] <= G16_CLOSE[1]:
        return '閉じる（2 割の内）'
    if ci[1] < G16_CLOSE[0]:
        return '場の効きが予言より小さい'
    if ci[0] > G16_CLOSE[1]:
        return '場の効きが予言より大きい'
    if ci[1] < 1:
        return '1 より小さい向きだが、ものさしのゆがみと区別できない'
    return 'はっきりしない'


def majority(words, k=G16_MAJ):
    """words：7 つ全部（出なかったモデルも「はっきりしない」として分母に入る）。"""
    vals, cnt = np.unique(np.array(words, dtype=object), return_counts=True)
    i = int(np.argmax(cnt))
    return (str(vals[i]), int(cnt[i])) if cnt[i] >= k and vals[i] != 'はっきりしない' else ('モデルで分かれる・はっきりしない', int(cnt[i]))


# ---------- モデルひとつ ----------
def analyze_counts(CW, CS, seed=G16_SEED, boot=G16_BOOT, art=None):
    def summed(w=None):
        out = {}
        for a, _ in G16_H:
            for r, X in (('w', CW), ('s', CS)):
                out[(a, r)] = X[a].sum(1) if w is None else np.tensordot(X[a], w, axes=([1], [0]))
        return out
    pt, uses = point16(summed())
    nD = CW['h0'].shape[1]
    rng = np.random.default_rng(seed)
    keys = list(pt)
    bs = {k: [] for k in keys}
    for _ in range(boot):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        b, _ = point16(summed(w), use=uses)
        for k in keys:
            bs[k].append(b[k])
    ci, fail = {}, {}
    for k in keys:
        v = np.array(bs[k], float)
        fail[k] = float(np.mean(~np.isfinite(v)))
        ci[k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
    words = dict(HQ1=word_hq1(ci['slope_w'], ci['slope_s'], fail['slope_w'], fail['slope_s'], art),
                 HQ3=word_R(ci['R'], fail['R'], ci['slope_w']))
    return dict(point=pt, ci=ci, fail=fail, words=words, use={f'{a}|{r}': u for (a, r), u in uses.items()})


def records_dlnK(arms, CW, pt):
    """記録（判定しない、点だけ）：温度ごとの ∂ln K/∂h（窓）を 3 通りで並べる。
    測った値 ＝ (ln K(＋0.3) − ln K(−0.3)) ÷ 0.6。恒等式の予言 ＝ h ＝ 0 の q から Cov(回数, σ)/(T⟨回数⟩) で（σ ＝ Σ（選んだか − q）、1 本の文ごと）。
    M の熱力学の予言 ＝ n × Δm_T ÷ T（n ＝ ΔH_vH ÷ 熱量計の ΔH、h ＝ 0）。あわせて、h ＝ 0 の q から予言した d⟨M⟩/dh と ±0.3 の実測、h ＝ 0 の腕の |q − q0| の最大。"""
    Z0 = arms['h0']
    pick, q = Z0['pick'].astype(float), Z0['q'].astype(float)
    sig = (pick - q).sum(-1)                                              # (7, S, nD)
    M = pick[..., 16:496].sum(-1)
    out = []
    for k, T in enumerate(G16_TEMPS):
        ev = np.zeros((4,) + sig.shape[1:])                               # [液体の歩, 凍る回, 結晶の歩, 溶ける回]（1 本の文ごと）
        for s in range(sig.shape[1]):
            for j in range(sig.shape[2]):
                st, _ = phases(copy_marks(Z0['toks'][k, s, j]))
                a, b = st[:-1], st[1:]
                ev[:, s, j] = [(a == 0).sum(), ((a == 0) & (b == 1)).sum(), (a == 1).sum(), ((a == 1) & (b == 0)).sum()]
        sg = sig[k].ravel()
        d = [float(np.mean((ev[i].ravel() - ev[i].mean()) * (sg - sg.mean())) / (T * ev[i].mean())) if ev[i].mean() > 0 else float('nan') for i in range(4)]
        ident = (d[1] - d[0]) - (d[3] - d[2])
        lnK = lambda c: np.log(c[k, 1] / c[k, 0]) - np.log(c[k, 3] / c[k, 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            meas = float((lnK(CW['hp03'].sum(1)) - lnK(CW['hm03'].sum(1))) / 0.6)
            t0 = CW['h0'].sum(1)
            dm_T = t0[k, 11] / t0[k, 7] - t0[k, 10] / t0[k, 5]
        mth = float(pt['n'] * dm_T / T) if np.isfinite(pt['n']) else float('nan')
        m, s_ = M[k].ravel(), sg
        out.append(dict(T=T, dlnK_meas=meas, dlnK_identity=float(ident), dlnK_M_thermo=mth,
                        dM_pred=float(np.mean((m - m.mean()) * (s_ - s_.mean())) / T),
                        dM_meas=float((arms['hp03']['pick'][k, ..., 16:496].sum(-1).mean() - arms['hm03']['pick'][k, ..., 16:496].sum(-1).mean()) / 0.6)))
    return dict(by_T=out, h0_q_minus_q0_max=float(np.max(np.abs(Z0['q'].astype(float) - Z0['q0'].astype(float)))))


def analyze16(base, models=G16_MODELS, seed=G16_SEED, boot=G16_BOOT):
    _check_temps()
    assert G16_ART['w'] is not None and G16_ART['s'] is not None, '空の作り物の見かけの傾きが置かれていない'
    res = {'models': {}}
    for name in models:
        arms, uses = {}, []
        for a, h in G16_H:
            arms[a], u = load_arm(base, name, a, h); uses.append(u)
        assert uses[0] == uses[1] == uses[2], name
        CW, CS = {}, {}
        for a, _ in G16_H:
            CW[a], CS[a] = arm_counts(arms[a]['toks'], arms[a]['U'], arms[a]['pick'])
        r = analyze_counts(CW, CS, seed, boot)
        r['records'] = records_dlnK(arms, CW, r['point'])
        r['nD'] = len(uses[0])
        r['by_T_h0_w'] = CW['h0'].sum(1).tolist()
        res['models'][name] = r
    res['verdict'] = verdict16(res['models'])
    return res


def verdict16(M):
    names = list(M)
    v = {}
    w1 = [M[n]['words']['HQ1'] for n in names]
    n_up = sum(w.startswith('両方で、') for w in w1); n_dis = sum(w == '食い違う' for w in w1)
    v['HQ1'] = ('場の効きは、ものさしの見かけではない' if n_up >= 6 else 'ものさしで変わる' if n_dis >= 2 else 'はっきりしない', dict(zip(names, w1)))
    w3 = [M[n]['words']['HQ3'] for n in names]
    v['HQ3'] = (majority(w3)[0], dict(zip(names, w3)))
    # 記録：n（2 通りのエンタルピーの比）。ばらつきは標準偏差（ddof ＝ 0）÷ 平均
    ns = np.array([M[n]['point']['n'] if M[n]['fail']['n'] <= G16_FAIL else np.nan for n in names], float)
    ok = np.isfinite(ns)
    v['record_n'] = dict(n=dict(zip(names, ns.tolist())), median=float(np.median(ns[ok])) if ok.any() else float('nan'),
                         cv=float(np.std(ns[ok]) / np.mean(ns[ok])) if ok.sum() >= 2 else float('nan'))
    return v


# ---------- 作り物（自己テスト・較正） ----------
def synth16(seed=1, nD=40, S=4, L=512, dH=4.0, Tm=0.55, Uc=0.2, Ul=1.5, pl=0.15, dM_scale=1.0, raw=False):
    """2 つの状態を行き来する鎖。ln K ＝ ΔH/T − ΔH/T_m ＋ h ΔM/T、ΔM ＝ dM_scale × n_真 × Δm（n_真 ＝ ΔH ÷ (Ul − Uc)、Δm ＝ 1 − pl）。
    結晶：周期 7 のくり返し、候補を必ず選ぶ。液体：候補を選ぶ割合 p_l(h, T) ＝ pl e^(h/T) ÷ (1 − pl ＋ pl e^(h/T))（場で増える）。
      液体で選んだときは、それまでの文の一か所から続けて写す（続けて選ぶと本当に 4 つ組のくり返しになり、写しの印が立つ）。選ばなければでたらめ。
    dM_scale ＝ 1：1 トークンあたりのクラペイロンがちょうど閉じる。0：本当の融点は場で動かない（空の作り物：ものさしが場に直接反応する分だけが出る）。"""
    rng = np.random.default_rng(seed)
    n_true = dH / (Ul - Uc); dM = dM_scale * n_true * (1 - pl)
    arms = {}
    for a, h in G16_H:
        toks = np.zeros((7, S, nD, L), np.int64); U = np.zeros((7, S, nD, L), np.float32); pick = np.zeros((7, S, nD, L), bool)
        for k, T in enumerate(G16_TEMPS):
            lnK = dH / T - dH / Tm + h * dM / T
            kf = 0.01 * np.exp(lnK / 2); km = 0.01 * np.exp(-lnK / 2)
            e = np.exp(h / T); plh = pl * e / (1 - pl + pl * e)
            st = np.zeros((S, nD), bool); ph = np.zeros((S, nD), int); ptr = np.zeros((S, nD), int)
            blk = rng.integers(100, 30000, (S, nD, 7))
            seq = np.zeros((S, nD, L), np.int64)
            for t in range(L):
                if t > 0:
                    r = rng.random((S, nD))
                    st = np.where(st, r >= km, r < kf)
                ph = np.where(st, ph + 1, 0)
                cp = (~st) & (rng.random((S, nD)) < plh) & (t > 8)
                src = np.take_along_axis(seq, np.clip(ptr, 0, L - 1)[..., None], -1)[..., 0]
                tok = np.where(st, np.take_along_axis(blk, (ph % 7)[..., None], -1)[..., 0], np.where(cp, src, rng.integers(100, 30000, (S, nD))))
                seq[..., t] = tok
                # 次の写し元：写したら 1 つ進める。写さなければ、それまでの文の別の場所へ
                ptr = np.where(cp, ptr + 1, rng.integers(0, max(t, 1), (S, nD)))
                U[k, :, :, t] = np.where(st, Uc, Ul) + rng.normal(0, 0.05, (S, nD))
                pick[k, :, :, t] = st | cp
            toks[k] = seq
        arms[a] = dict(toks=toks, U=U, pick=pick)
    if raw:
        return arms
    CW, CS = {}, {}
    for a, _ in G16_H:
        CW[a], CS[a] = arm_counts(arms[a]['toks'], arms[a]['U'], arms[a]['pick'])
    return CW, CS, dict(slope_true=dM * Tm / dH, n_true=n_true)


def calib_empty(seeds=(101, 102, 103), pls=(0.1, 0.2, 0.3)):
    """空の作り物（dM_scale ＝ 0）で、見かけの場の傾き（点）を、液体で写す割合を変えて測る → 窓・続けて写す、それぞれの最大。"""
    out = []
    for sd, pl in zip(seeds, pls):
        CW, CS, _ = synth16(seed=sd, pl=pl, dM_scale=0.0)
        pt, _ = point16({(a, r): X[a].sum(1) for a, _ in G16_H for r, X in (('w', CW), ('s', CS))})
        out.append(dict(pl=pl, slope_w=pt['slope_w'], slope_s=pt['slope_s']))
    return out


def self_test16(boot=60):
    _check_temps()
    out = {}
    art = dict(w=0.0, s=0.0)
    # 1) 閉じる作り物
    CW, CS, tr = synth16(seed=1)
    r = analyze_counts(CW, CS, boot=boot, art=art); p = r['point']
    out['closed'] = dict(slope_true=tr['slope_true'], slope_w=p['slope_w'], slope_s=p['slope_s'], R=p['R'], ci_R=r['ci']['R'], n=p['n'], n_true=tr['n_true'],
                         Tm_h0_w=p['Tm_h0_w'], words=r['words'])
    assert p['slope_w'] > 0.15 and p['slope_s'] > 0.15, p
    assert abs(p['Tm_h0_w'] - 0.55) < 0.06, p['Tm_h0_w']
    assert 0.7 < p['R'] < 1.3, p['R']
    assert r['words']['HQ3'] in ('閉じる（2 割の内）', '1 より小さい向きだが、ものさしのゆがみと区別できない'), r['words']
    # 2) 場の効きが予言の半分
    CW2, CS2, _ = synth16(seed=2, dM_scale=0.5)
    r2 = analyze_counts(CW2, CS2, boot=boot, art=art)
    out['half'] = dict(R=r2['point']['R'], ci_R=r2['ci']['R'], words=r2['words'])
    assert 0.3 < r2['point']['R'] / p['R'] < 0.75 and r2['words']['HQ3'] == '場の効きが予言より小さい', (r2['point']['R'], r2['words'])
    # 3) 言葉
    assert word_R([0.85, 1.2], 0, [0.1, 0.3]) == '閉じる（2 割の内）' and word_R([0.4, 0.7], 0, [0.1, 0.3]) == '場の効きが予言より小さい'
    assert word_R([0.75, 0.97], 0, [0.1, 0.3]) == '1 より小さい向きだが、ものさしのゆがみと区別できない'
    assert word_R([1.3, 2.0], 0, [0.1, 0.3]) == '場の効きが予言より大きい' and word_R([0.7, 1.3], 0, [0.1, 0.3]) == 'はっきりしない'
    assert word_R([-0.6, -0.1], 0, [-0.3, -0.1]) == '逆向き（場で融点が下がる）' and word_R([0.85, 1.2], 0.2, [0.1, 0.3]) == 'はっきりしない'
    a0 = dict(w=0.05, s=0.08)
    assert word_hq1([0.1, 0.3], [0.1, 0.2], 0, 0, a0).startswith('両方で、') and word_hq1([0.03, 0.3], [0.1, 0.2], 0, 0, a0) == 'はっきりしない'
    assert word_hq1([0.1, 0.3], [-0.3, -0.1], 0, 0, a0) == '食い違う'
    assert majority(['a'] * 5 + ['b'] * 2)[0] == 'a' and majority(['a'] * 4 + ['b'] * 3)[0].startswith('モデルで')
    # 4) 続けて写す歩のものさしの数えが、ladder の ruler_counts と合う
    pk = np.random.default_rng(9).random((3, 512)) < 0.7
    c, _ = ruler_counts(pk)
    assert np.allclose(counts_streak(np.zeros((3, 512)), pk)[[0, 1, 2, 3]], c.sum(0))
    # 5) 空の作り物：見かけの傾き
    out['empty'] = calib_empty()
    return out
