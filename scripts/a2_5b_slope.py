"""A2-5b（ゲート docs/A2_5B_GATE.md）：T_0 での f の傾き。データを開く前に固定する。CPU。
scripts/a2_2_analyze.py（per_doc_feedback）と scripts/a2_5_analyze.py（axis_word・A5_SIZE・A5_STEP）を実行したのと
同じ名前空間で実行する。記事の本文はいらない（npz の tf と gen だけ使う）。

    analyze5b(a2_5 のフォルダ) -> dict
"""
import math
import numpy as np

B5_T = (0.8, 0.9)                      # 傾きをとる 2 つの温度
B5_BOOT, B5_SEED = 1000, 20261103      # 記事を引き直すブートストラップ（全モデルで同じ引き方＝対応あり）
B5_Q7 = (1.5, 3.0)                     # 70M ÷ 2.8B の線


def slope_of(fd, temps, idx=None):
    """fd：(nD, nT) の記事ごとの f。→ s ＝（F(0.9) − F(0.8)）÷ 0.1。F は記事の中央値（T_0 と同じ量）。"""
    k0, k1 = temps.index(B5_T[0]), temps.index(B5_T[1])
    x = fd if idx is None else fd[idx]
    F = np.median(x, 0)
    return float((F[k1] - F[k0]) / (temps[k1] - temps[k0]))


def q7_word(r):
    if not np.isfinite(r):
        return 'なし'
    return '1.5 倍より小さい' if r < B5_Q7[0] else '1.5〜3 倍' if r <= B5_Q7[1] else '3 倍より大きい'


def analyze5b_from_fd(fds, temps):
    """fds：{名前: (nD, nT)}。全モデルで記事の並びが同じであること。"""
    nD = {v.shape[0] for v in fds.values()}
    assert len(nD) == 1, '記事の数がモデルによって違います'
    nD = nD.pop()
    rng = np.random.default_rng(B5_SEED)
    idx = rng.integers(0, nD, size=(B5_BOOT, nD))
    s = {n: slope_of(fd, temps) for n, fd in fds.items()}
    sb = {n: np.array([slope_of(fd, temps, idx[b]) for b in range(B5_BOOT)]) for n, fd in fds.items()}
    ci = {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in sb.items()}
    res = {'s': s, 's_ci': ci, 'T': list(B5_T), 'boot': B5_BOOT, 'seed': B5_SEED}
    size = [n for n, _ in A5_SIZE]; step = [n for n, _ in A5_STEP]
    xs = [math.log(p) for _, p in A5_SIZE]
    sym, det = axis_word(xs, [s[n] for n in size], np.stack([sb[n] for n in size], 1), True, False, '+', '-')
    res['Q5'] = {'＋': '大きいほど急', '−': '小さいほど急'}.get(sym, 'はっきりしない'); res['Q5_detail'] = det
    xt = [math.log(t) for _, t in A5_STEP]
    sym, det = axis_word(xt, [s[n] for n in step], np.stack([sb[n] for n in step], 1), True, True, '+', '-')
    res['Q6'] = {'＋': '学ぶほど急', '−': '学ぶほど緩やか'}.get(sym, 'はっきりしない'); res['Q6_detail'] = det
    r = s['pythia-70m'] / s['pythia-2.8b'] if s['pythia-2.8b'] != 0 else float('nan')
    rb = sb['pythia-70m'] / sb['pythia-2.8b']
    res['Q7'] = q7_word(r)
    res['Q7_detail'] = {'ratio': r, 'ci': [float(np.nanpercentile(rb, 2.5)), float(np.nanpercentile(rb, 97.5))]}
    # 記録（判定しない）：ΔH 相当（T_0 × s）は T_0 を別に読むので、ここでは s だけ残す
    return res


def analyze5b(d5):
    names = [n for n, _ in A5_SIZE] + [n for n, _ in A5_STEP if n not in dict(A5_SIZE)]
    fds, temps, uses = {}, None, set()
    for n in names:
        Z = np.load(f'{d5}/{n}.npz')
        t = [float(x) for x in Z['temps']]
        assert temps is None or t == temps, '温度の並びが違います'
        temps = t
        uses.add(tuple(int(u) for u in Z['use']))
        fds[n] = per_doc_feedback(Z['tf'][0], Z['gen'][0])
    assert len(uses) == 1, '使った記事がモデルによって違います'
    return analyze5b_from_fd(fds, temps)


def self_test_a25b():
    temps = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    rng = np.random.default_rng(1)
    def fake(k):  # 傾き k（0.8→0.9 で 0.1k 上がる）＋小さな雑音
        base = np.array([-3, -1, 0, 0.5, 1, 1.5, 2.0])
        base[4] = base[2] + 0.1 * k
        return base[None, :] + rng.normal(0, 0.01, (40, len(temps)))
    sizes = [n for n, _ in A5_SIZE]; steps = [n for n, _ in A5_STEP]
    ks = dict(zip(sizes, [12, 10, 8, 6, 4, 2]))
    ks.update(zip(steps, [2, 3, 4, 6]))         # pythia-1b は両方の軸にいる：大きさの軸で 6、学びの最後でも 6
    fds = {n: fake(k) for n, k in ks.items()}
    r = analyze5b_from_fd(fds, temps)
    assert abs(r['s']['pythia-70m'] - 12) < 0.5, r['s']
    assert r['Q5'] == '小さいほど急', r
    assert r['Q6'] == '学ぶほど急', r
    assert r['Q7'] == '3 倍より大きい', r
    assert q7_word(1.49) == '1.5 倍より小さい' and q7_word(3.0) == '1.5〜3 倍' and q7_word(float('nan')) == 'なし'
    return 'OK'
