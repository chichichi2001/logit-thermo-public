"""A2-2 の罠チェック：おもちゃの自己回帰モデルで、ゲートの測り方が正しく働くか。

おもちゃのモデル：語彙 V、各一歩の logit は「べきの形（傾き a、1 位を +bump）」を、一歩ごとにランダムに並べ替えたもの。
フィードバック 2 種類（強さを 0 にすると消える）：
- 凝縮（beta）：直近 32 歩に出たトークンの logit に beta × 回数を足す（繰り返しを呼ぶ）。
- ほどけ（kappa）：直近の「裾から引いた割合」u（指数移動平均）が大きいほど、logit を 1 / (1 + kappa × u) 倍に平らにする。
人の文章：同じべきの形から温度 T_h で引いたトークン（人はフィードバックなし）。

確かめること：
1. フィードバックなし（beta = kappa = 0）なら、すべての温度で F ≈ 0。
2. フィードバックありなら、F は低温で負・高温で正になり、0 を横切る（T_0 がある）。
3. 温度計 T_cal（人のトークンの NLL が最小の温度）は、フィードバックなしのとき T_h を取り戻す。
4. 対応状態の判定の道具（家族ごとに抜いたスプライン）：作り込んだ曲線では x3 が勝ち、壊した曲線では勝たない。
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

V, H, STEPS, PROMPT = 4000, 100, 256, 16
TEMPS = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0])


def shape(a, bump):
    z = -a * np.log(np.arange(1, V + 1.0))
    z[0] += bump
    return z


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def ell(z, T):
    """裾の取り分の log-odds（log を直接計算）。"""
    zs = np.sort(z)[::-1] / T
    return lse(zs[H - 1:]) - lse(zs[:H - 1])


def run(a, bump, beta, kappa, T_h, n_seeds=16, seed=0, human_a=None):
    base = shape(a, bump)
    hbase = shape(human_a if human_a is not None else a, bump)
    rng = np.random.default_rng(seed)
    F = []
    for T in TEMPS:
        diffs = []
        for s in range(n_seeds):
            r = np.random.default_rng(seed * 1000 + s)
            perms = [r.permutation(V) for _ in range(PROMPT + STEPS)]
            # 人の文章（書き出し＋続き）：各一歩、並べ替えたべきの形から温度 T_h で引く（フィードバックなし）
            human = []
            for t in range(PROMPT + STEPS):
                z = np.empty(V); z[perms[t]] = hbase
                p = np.exp((z - z.max()) / T_h); p /= p.sum()
                human.append(r.choice(V, p=p))

            def model_logits(t, hist, u):
                z = np.empty(V); z[perms[t]] = base
                if beta:
                    recent = hist[-32:]
                    if recent:
                        z += beta * np.bincount(recent, minlength=V)
                if kappa:
                    z = z / (1 + kappa * u)
                return z

            def walk(tokens_fn):
                hist, u, ls = list(human[:PROMPT]), 0.0, []
                for t in range(PROMPT, PROMPT + STEPS):
                    z = model_logits(t, hist, u)
                    ls.append(ell(z, T))
                    tok = tokens_fn(t, z)
                    rank = int((z > z[tok]).sum())
                    u = 0.9 * u + 0.1 * (rank >= H - 1)
                    hist.append(tok)
                return np.array(ls)

            gen_rng = np.random.default_rng(seed * 7919 + s)
            def sample(t, z):
                p = np.exp((z - z.max()) / T); p /= p.sum()
                return gen_rng.choice(V, p=p)
            l_gen = walk(sample)
            l_tf = walk(lambda t, z: human[t])
            diffs.append(np.mean(l_gen[1:] - l_tf[1:]))
        F.append(np.median(diffs))
    return np.array(F)


def t_cal(a, bump, T_h, n=4000, seed=1, human_a=None):
    """人のトークン（温度 T_h）の NLL が最小になる温度（フィードバックなしの一歩）。"""
    base = shape(a, bump)
    hbase = shape(human_a if human_a is not None else a, bump)
    r = np.random.default_rng(seed)
    p = np.exp((hbase - hbase.max()) / T_h); p /= p.sum()
    toks = r.choice(V, size=n, p=p)            # 並べ替えは NLL に効かないので省く
    grid = np.exp(np.linspace(np.log(0.3), np.log(3), 400))
    nll = [-(base[toks] / T).mean() + lse(base / T) for T in grid]
    return float(grid[int(np.argmin(nll))])


def crossing(F):
    i = np.nonzero(np.diff(np.sign(F)) > 0)[0]
    if len(i) == 0:
        return np.nan
    i = i[0]
    lt0, lt1 = np.log(TEMPS[i]), np.log(TEMPS[i + 1])
    return float(np.exp(lt0 + (0 - F[i]) / (F[i + 1] - F[i]) * (lt1 - lt0)))


def pav(y):
    """単調増加の当てはめ（隣り合う違反を平均でならす）。"""
    blocks = [[v, 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [m, blocks[i][1] + blocks[i + 1][1]]; del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for m, n in blocks:
        out += [m] * n
    return np.array(out)


def crossing_mono(F, eps=0.05):
    """v0.3 の規則：F を温度について単調増加に当てはめ、その当てはめが -eps 以下から +eps 以上へ移る温度。"""
    g = pav(np.asarray(F, float))
    neg = np.nonzero(g <= -eps)[0]; pos = np.nonzero(g >= eps)[0]
    if len(neg) == 0 or len(pos) == 0 or pos[0] < neg[-1]:
        return np.nan
    i, j = neg[-1], pos[0]
    lt0, lt1 = np.log(TEMPS[i]), np.log(TEMPS[j])
    return float(np.exp(lt0 + (0 - g[i]) / (g[j] - g[i]) * (lt1 - lt0)))


def family_out_error(xs, Fs, fams):
    """家族ごとに抜いて、ほかの点に当てたスプラインで抜いた家族の F を予言したときの二乗誤差の平均。"""
    errs = []
    for f in set(fams):
        tr = [i for i, g in enumerate(fams) if g != f]
        te = [i for i, g in enumerate(fams) if g == f]
        x = np.concatenate([xs[i] for i in tr]); y = np.concatenate([Fs[i] for i in tr])
        o = np.argsort(x); x, y = x[o], y[o]
        x, idx = np.unique(x, return_index=True); y = y[idx]
        sp = UnivariateSpline(x, y, k=3, s=len(x) * 0.01)
        for i in te:
            errs.append(np.mean((sp(xs[i]) - Fs[i]) ** 2))
    return float(np.mean(errs))


if __name__ == '__main__':
    print('== 3. 温度計は T_h を取り戻すか（フィードバックなし） ==')
    for T_h in [0.8, 1.0, 1.3]:
        print(f'  T_h={T_h}  T_cal={t_cal(1.4, 5.0, T_h):.3f}')

    print('\n== 1. フィードバックなし：F ≈ 0 か ==')
    F0 = run(1.4, 5.0, beta=0.0, kappa=0.0, T_h=1.0, n_seeds=6)
    print('  F =', np.round(F0, 3))

    print('\n== 2. フィードバックあり：F の符号が温度で変わるか ==')
    for name, (beta, kappa) in {'凝縮だけ': (1.5, 0.0), 'ほどけだけ': (0.0, 3.0), '両方': (1.5, 3.0)}.items():
        F = run(1.4, 5.0, beta=beta, kappa=kappa, T_h=1.0, n_seeds=6)
        print(f'  {name:6s} F = {np.round(F, 2)}  T_0 = {crossing(F):.3f}  T_0(単調の規則) = {crossing_mono(F):.3f}')

    print('\n== 2b. 人の文章の形がモデルと違う（モデル a=1.4、人 a=1.0・1.8）：温度計は外れうるか ==')
    for ha in [1.0, 1.8]:
        F = run(1.4, 5.0, beta=0.0, kappa=3.0, T_h=1.0, n_seeds=6, human_a=ha)
        print(f'  人 a={ha}: T_0 = {crossing(F):.3f}  T_0(単調の規則) = {crossing_mono(F):.3f}  T_cal = {t_cal(1.4, 5.0, 1.0, human_a=ha):.3f}')

    print('\n== 2c. E が外れるべきおもちゃ：上位を強く押し上げる写し取り型の凝縮（v0.2 のレビューで追加） ==')
    print(f'  （参考：人の文章は T_h=1.0、T_cal={t_cal(1.4, 5.0, 1.0):.3f}）')
    for name, (beta, kappa) in {'写し取りだけ β=4': (4.0, 0.0), '写し取りだけ β=6': (6.0, 0.0),
                                '写し取り β=4＋ほどけ': (4.0, 3.0), '写し取り β=6＋ほどけ': (6.0, 3.0)}.items():
        F = run(1.4, 5.0, beta=beta, kappa=kappa, T_h=1.0, n_seeds=6)
        print(f'  {name:16s} F = {np.round(F, 2)}  T_0 = {crossing(F):.3f}  T_0(単調の規則) = {crossing_mono(F):.3f}')

    print('\n== 4. 対応状態の判定の道具（作った曲線で） ==')
    rng = np.random.default_rng(3)
    fams, x1s, x3s_good, F_good, F_bad = [], [], [], [], []
    for fam in range(4):
        for m in range(2):
            sh = rng.uniform(0.7, 1.5)                         # モデルごとに一歩の状態の温度の目盛りが違う
            lt = np.log(TEMPS)
            x3 = (lt - np.log(sh)) * 3.0                        # 一歩の状態（ℓ_tf のまね）
            fams.append(fam); x1s.append(lt); x3s_good.append(x3)
            F_good.append(np.tanh(x3) + rng.normal(0, 0.05, len(lt)))          # F は一歩の状態だけで決まる
            F_bad.append(np.tanh(3 * lt) + rng.normal(0, 0.05, len(lt)))       # F は温度だけで決まる（対応状態なし）
    for label, Fs in [('作り込み（対応状態あり）', F_good), ('壊したもの（温度だけ）', F_bad)]:
        e1 = family_out_error(x1s, Fs, fams); e3 = family_out_error(x3s_good, Fs, fams)
        print(f'  {label}: 誤差 x1(log T)={e1:.4f}  x3(一歩の状態)={e3:.4f}  → x3 の勝ち? {e3 < 0.7 * e1}')
