"""探索：結晶が始まる歩（2026-10-29 18:3x）。確かめには使わない（予想なし）。CPU。生成もモデルの読み込みもしない。

発想（オーナー 2026-10-29 18:27・18:28）：（オーナーの承認・発言。原文は非公開の記録にある）
問い：
  (1) 結晶の始まりは「文脈で決まる」（滴定寄り：同じ記事なら 2 サンプルとも同じころに始まる）か、「運」（核寄り：同じ記事でもばらばら）か。
  (2) 始まる手前に、写しがなだらかに増える助走があるか。
  (3) Pythia の大きさで、始まり方が変わるか。

材料：A2-2・A2-5・A2-6 で保存した生成のトークン（toks、温度 15 × サンプル 2 × 記事 40 × 512 歩）。探索 T-scan と同じ 25 本。
写しの印 c_t：続きの t 歩目で終わる 4 トークンの組が、続きの中でそれより前に出ていたら 1（書き出しは数えない。トークナイザを使わないため）。
始まりの歩：直近 W ＝ 32 歩の c の平均が初めて 0.5 以上になった歩（窓の最後の歩）。512 歩で始まらなければ「始まらない」。
"""
import json
import numpy as np

ON_RUNS = {
    'A2_2': ['pythia-1b', 'qwen2.5-0.5b', 'qwen2.5-0.5b-inst', 'qwen2.5-1.5b', 'olmo2-1b', 'smollm2-1.7b', 'smollm2-1.7b-inst'],
    'A2_5': ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
             'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000'],
    'A2_6': ['qwen2.5-0.5b', 'qwen2.5-1.5b', 'qwen2.5-3b', 'smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
             'qwen3-0.6b', 'qwen3-1.7b', 'qwen3-4b'],
}
ON_W, ON_TH = 32, 0.5
ON_PRE, ON_POST, ON_BIN = 160, 32, 8          # 始まりにそろえた平均の幅（歩）と、まとめる幅
ON_HBINS = 4                                  # 始まりやすさ（ハザード）を 128 歩ずつ 4 つで
ON_MIN_BOTH = 8                               # 同じ記事の 2 サンプルの比べ：両方始まった記事がこれ以上のときだけ


def copy_marks(seq):
    """続きのトークン列 → c（長さ L、0/1）。t 歩目で終わる 4 つ組が、それより前（続きの中）に出ていたら 1。"""
    s = np.asarray(seq, dtype=np.int64)
    L = len(s)
    c = np.zeros(L, np.int8)
    if L < 4:
        return c
    g = np.stack([s[:-3], s[1:-2], s[2:-1], s[3:]], axis=1)          # 4 つ組（語彙が 15 万あるので 1 つの整数にはまとめない）
    _, first, inv = np.unique(g, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    c[3:] = (first[inv] < np.arange(len(g))).astype(np.int8)
    return c


def onset_of(c):
    """→ 始まりの歩（0 始まり、窓の最後の歩）。なければ −1。"""
    if len(c) < ON_W:
        return -1
    r = np.convolve(c.astype(np.float64), np.ones(ON_W), 'valid') / ON_W   # r[k] は歩 k..k+W−1 の平均
    k = np.flatnonzero(r >= ON_TH - 1e-12)
    return int(k[0] + ON_W - 1) if len(k) else -1


def run_stats(toks):
    """toks：(nT, S, nD, L) → 温度ごとのまとめと、始まりにそろえた平均。"""
    nT, S, nD, L = toks.shape
    on = np.full((nT, S, nD), -1, np.int32)
    lock = np.full((nT, S, nD), np.nan)
    C = np.zeros((nT, S, nD, L), np.int8)
    for k in range(nT):
        for s in range(S):
            for j in range(nD):
                c = copy_marks(toks[k, s, j]); C[k, s, j] = c
                o = onset_of(c); on[k, s, j] = o
                if o >= 0 and o < L - 1:
                    r = np.convolve(c.astype(np.float64), np.ones(ON_W), 'valid') / ON_W
                    lock[k, s, j] = float(np.mean(r[o - ON_W + 1:] >= ON_TH - 1e-12))
    per_T = []
    nb = (ON_PRE + ON_POST) // ON_BIN
    aligned_sum, aligned_n = np.zeros(nb), 0
    for k in range(nT):
        o = on[k]                                        # (S, nD)
        hit = o >= 0
        p = float(hit.mean())
        oo = o[hit]
        row = dict(p=round(p, 4), n_on=int(hit.sum()))
        if len(oo):
            row.update(med=float(np.median(oo)), q25=float(np.percentile(oo, 25)), q75=float(np.percentile(oo, 75)),
                       cv=round(float(np.std(oo) / np.mean(oo)), 3) if np.mean(oo) > 0 else None,
                       lock=round(float(np.nanmean(lock[k][hit])), 3))
        # 片方だけ始まった記事の割合（記事が関係ないなら 2p(1−p)）
        if S == 2:
            split = float(np.mean(hit[0] != hit[1]))
            row.update(split_obs=round(split, 4), split_exp=round(2 * p * (1 - p), 4))
            both = hit[0] & hit[1]
            nboth = int(both.sum()); row['n_both'] = nboth
            if nboth >= ON_MIN_BOTH:
                a, b = o[0][both].astype(float), o[1][both].astype(float)
                within = float(np.mean(np.abs(a - b)))
                diff = np.abs(a[:, None] - b[None, :])
                between = float(diff[~np.eye(nboth, dtype=bool)].mean())
                row.update(within=round(within, 1), between=round(between, 1), R=round(within / between, 3) if between > 0 else None)
        # 始まりやすさ：128 歩ずつ、その区間の頭でまだ始まっていない本のうち、区間の中で始まった割合
        if p >= 0.1:
            hz, edges = [], np.linspace(0, L, ON_HBINS + 1).astype(int)
            flat = o.ravel()
            for b0, b1 in zip(edges[:-1], edges[1:]):
                atrisk = (flat < 0) | (flat >= b0)
                ev = (flat >= b0) & (flat < b1)
                hz.append(round(float(ev.sum() / atrisk.sum()), 4) if atrisk.sum() else None)
            row['hazard'] = hz
        # 始まらなかった本の写しの割合（基準）
        if (~hit).any():
            row['base_c'] = round(float(C[k][~hit].mean()), 4)
        # 始まりにそろえた c の平均（手前に ON_PRE 歩、後ろに ON_POST 歩とれる本だけ）
        for s in range(S):
            for j in range(nD):
                oj = o[s, j]
                if oj >= ON_PRE and oj + ON_POST + 1 <= L:          # 切り出しの終わり oj + ON_POST + 1 が L を超えないこと
                    seg = C[k, s, j, oj - ON_PRE + 1: oj + ON_POST + 1].astype(float)
                    aligned_sum += seg.reshape(nb, ON_BIN).mean(1); aligned_n += 1
        per_T.append(row)
    return per_T, (aligned_sum / aligned_n).round(4).tolist() if aligned_n else None, aligned_n, on


def run_onset(base):
    res = {'W': ON_W, 'TH': ON_TH, 'aligned_bins': [-ON_PRE + ON_BIN * i for i in range(1, (ON_PRE + ON_POST) // ON_BIN + 1)], 'runs': {}}
    allon = {}
    for folder, names in ON_RUNS.items():
        for name in names:
            Z = np.load(f'{base}/{folder}/{name}.npz')
            toks = Z['toks']
            if 'temps' in res:
                assert [float(t) for t in Z['temps']] == res['temps']
            else:
                res['temps'] = [float(t) for t in Z['temps']]
            per_T, al, aln, on = run_stats(toks)
            res['runs'][f'{folder}/{name}'] = dict(per_T=per_T, aligned=al, aligned_n=aln)
            allon[f'{folder}/{name}'] = on
    return res, allon


def self_test_onset():
    c = copy_marks([1, 2, 3, 4, 1, 2, 3, 4, 5])
    assert c.tolist() == [0, 0, 0, 0, 0, 0, 0, 1, 0], c          # 2 回目の 1 2 3 4 の 4 で 1
    # 40 歩はばらばら、そのあと 7 の周期の単結晶 → 始まりは周期が始まって少しあと
    rng = np.random.default_rng(0)
    seq = list(rng.integers(100, 30000, 200)) + [11, 12, 13, 14, 15, 16, 17] * 40
    c = copy_marks(seq)
    o = onset_of(c)
    assert 200 + 3 <= o <= 200 + 7 + 3 + ON_W, o
    assert onset_of(copy_marks(list(rng.integers(100, 30000, 512)))) == -1
    # 記事で決まる場合：同じ記事の 2 サンプルが同じ歩で始まる → R ＝ 0、片方だけは 0
    L, nD = 512, 20
    toks = np.zeros((1, 2, nD, L), np.int64)
    for j in range(nD):
        st = 100 + 15 * j
        for s in range(2):
            toks[0, s, j] = np.concatenate([rng.integers(100, 30000, st), np.tile([5, 6, 7, 8, 9], (L - st) // 5 + 1)[:L - st]])
    per_T, al, aln, on = run_stats(toks)
    r = per_T[0]
    assert r['p'] == 1.0 and r['split_obs'] == 0 and r['R'] < 0.05, r
    assert al is not None and al[-1] > 0.9 and al[0] < 0.1, al
    # 境目：始まりの歩がちょうど L − ON_POST − 1 と L − ON_POST になる本（2026-10-29 18:35 のエラーの再現）
    for target in (L - ON_POST - 1, L - ON_POST):
        for st in range(target - ON_W - 10, target + 1):
            sq = np.concatenate([rng.integers(100, 30000, st), np.tile([5, 6, 7, 8, 9], L)[:L - st]])
            if onset_of(copy_marks(sq)) == target:
                break
        assert onset_of(copy_marks(sq)) == target, target
        run_stats(np.tile(sq, (1, 2, 1, 1)))
    return 'OK'
