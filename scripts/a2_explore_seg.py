"""探索 8.25：窓の形の見かけを除いた 3 つの状態のものさし（かたまりで切り出す）で、8.22 をやり直す（2026-10-30 18:2x、CPU、判定しない）。
scripts/a2_explore_onset.py（copy_marks）・scripts/a2_8_vh.py（phases）・scripts/a2_explore_three.py（rates・vh_step）を実行したのと同じ名前空間で実行する。

8.24 で分かったこと：8.21 の窓（64 歩、16 歩おき）のラベルは、周期 17 歩以上のきれいな結晶がいきなり終わっても、
溶け方を「C → A → L」に見せてしまう（窓の中で結晶の部分が減ると、周期の強さ P が写しの割合 r より先に 0.5 を切るため）。
新しいものさし：
1) 相（液体・結晶）は A2-8 と同じ（写しの印、中心の窓 32、入口 0.5・出口 0.25 の履歴つき）。歩ごと（歩 16〜495）。
2) 結晶の歩の「形」は、写しの印が付いた歩それぞれの「写しのずれ」（その 4 つ組が直前に出た位置までの歩数）で決める。
   歩 t のずれ ＝ t を中心にした 33 歩の中の、写しの印が付いた歩のずれの下から 10% の値（印がなければ直前のまま）。
   中央値にしないのは、結晶に傷（ほかのトークン）があると、傷のあとの 4 つ組は 1 周期前ではなく 2 周期前と合い、ずれが周期の 2 倍に出るため
   （作り物で、周期 23・傷 10% のとき、中央値では C ⇄ A のちらつきが 100 回出たが、下から 10% では 8 回）。
   ずれ ≤ 32 → C（短い周期）、ずれ > 32 → A（長い周期・周期なし）。
   ずれはトークンそのものの性質なので、結晶の終わりで窓の中の結晶が減っても、ずれは変わらない（見かけが出にくい）。
速さは 1 歩あたり。段ごとの平衡定数・ファントホッフは 8.22 と同じ（rates・vh_step、両向きそれぞれ 8 回以上の温度）。
"""
import numpy as np

SG_HALF = 16          # ずれの中央値をとる幅（t−16〜t+16）
SG_SHORT = 32         # これ以下のずれを「短い周期」
SG_Q = 10             # ずれの下から何 % の値を使うか


def copy_lags(seq):
    """→ lag（長さ L、整数。写しの印がない歩は 0）。t 歩目で終わる 4 つ組が、直前に出た位置（終わりの歩）までの歩数。"""
    s = [int(x) for x in seq]
    L = len(s)
    lag = np.zeros(L, np.int64)
    last = {}
    for t in range(3, L):
        g = (s[t - 3], s[t - 2], s[t - 1], s[t])
        if g in last:
            lag[t] = t - last[g]
        last[g] = t
    return lag


def seg_labels(seq):
    """→ 歩 16〜495 のラベルの列（'L'・'A'・'C'）。"""
    c = copy_marks(seq)
    st, t = phases(c)
    lag = copy_lags(seq)
    out, prev = [], 'A'
    for ph, tt in zip(st, t):
        if ph == 0:
            out.append('L'); prev = 'A'
            continue
        a, b = max(0, tt - SG_HALF), min(len(lag), tt + SG_HALF + 1)
        w = lag[a:b]; w = w[w > 0]
        if len(w):
            prev = 'C' if np.percentile(w, SG_Q) <= SG_SHORT else 'A'
        out.append(prev)
    return out


def seg_counts(seq):
    labs = seg_labels(seq)
    occ = {s: 0 for s in 'LAC'}; tr = {}
    for x, y in zip(labs[:-1], labs[1:]):
        occ[x] += 1
        if x != y:
            tr[x + y] = tr.get(x + y, 0) + 1
    return occ, tr, labs


def seg_counts_by_doc(toks):
    """toks：(nT, S, nD, L) → (nT, nD, 9)：[L, A, C にいた歩の数, LA, AL, AC, CA, LC, CL の回数]（8.22 の counts_by_doc と同じ並び）。"""
    nT, S, nD, _ = toks.shape
    keys = ['LA', 'AL', 'AC', 'CA', 'LC', 'CL']
    out = np.zeros((nT, nD, 9))
    for k in range(nT):
        for j in range(nD):
            for s in range(S):
                occ, tr, _ = seg_counts(toks[k, s, j])
                out[k, j, :3] += [occ['L'], occ['A'], occ['C']]
                out[k, j, 3:] += [tr.get(x, 0) for x in keys]
    return out


def after_CA(toks, ks):
    """C → A のあと、A がどこへ行くか（C・L・終わり）。"""
    res = {'C': 0, 'L': 0, 'end': 0}
    for k in ks:
        for s in range(toks.shape[1]):
            for j in range(toks.shape[2]):
                labs = seg_labels(toks[k, s, j])
                for i in range(len(labs) - 1):
                    if labs[i] == 'C' and labs[i + 1] == 'A':
                        m = i + 1
                        while m < len(labs) and labs[m] == 'A':
                            m += 1
                        res[labs[m] if m < len(labs) else 'end'] += 1
    return res


def self_test_seg(seed=1):
    """作り物：きれいな結晶をいきなり入れて、いきなり終わらせる。短い周期は L → C → L、長い周期は L → A → L、
    途中で周期が 61 から 7 に縮むものは L → A → C → L とだけ数えること。"""
    rng = np.random.default_rng(seed)
    def make(parts):
        s = list(rng.integers(100, 30000, 150))
        for per, n in parts:
            s += list(np.tile(rng.integers(100, 30000, per), 600)[:n])
        s += list(rng.integers(100, 30000, 512 - len(s)))
        return np.array(s[:512])
    out = {}
    for name, parts in {'p3': [(3, 180)], 'p7': [(7, 180)], 'p23': [(23, 200)], 'p29': [(29, 200)],
                        'p45': [(45, 220)], 'p61': [(61, 240)], 'p61to7': [(61, 200), (7, 120)]}.items():
        tot = {}
        for _ in range(20):
            _, tr, _ = seg_counts(make(parts))
            for k, v in tr.items():
                tot[k] = tot.get(k, 0) + v
        out[name] = tot
    for n in ('p3', 'p7', 'p23', 'p29'):
        assert out[n].get('LC', 0) == 20 and out[n].get('CL', 0) == 20 and sum(out[n].values()) == 40, (n, out[n])
    for n in ('p45', 'p61'):
        assert out[n].get('LA', 0) == 20 and out[n].get('AL', 0) == 20 and sum(out[n].values()) == 40, (n, out[n])
    o = out['p61to7']
    assert o.get('LA', 0) == 20 and o.get('AC', 0) == 20 and o.get('CL', 0) == 20 and sum(o.values()) == 60, o
    # 傷 10% の結晶（周期 23）：ちらつき（C ⇄ A）が、入って出る回数の 2 割より少ないこと
    tot = {}
    for _ in range(30):
        s = list(rng.integers(100, 30000, 150)); body = np.tile(rng.integers(100, 30000, 23), 600)[:220].copy()
        m = rng.random(220) < 0.10; body[m] = rng.integers(100, 30000, m.sum())
        s += list(body); s += list(rng.integers(100, 30000, 512 - len(s)))
        _, tr, _ = seg_counts(np.array(s))
        for k, v in tr.items():
            tot[k] = tot.get(k, 0) + v
    out['p23_defect10'] = tot
    assert tot.get('CA', 0) + tot.get('AC', 0) < 0.2 * (tot.get('LC', 0) + tot.get('CL', 0)), tot
    return out
