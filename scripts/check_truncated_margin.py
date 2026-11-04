"""Min-p set stability under margin > osc,
# 付録 G の十分条件（マージン ＞ |h|/T）を作り物の数え上げで確かめる。2026-11-04、付録 G の書き手（経緯を知らないサブエージェント）が作り、Claude がリポジトリに移した。
# 使い方：python3 scripts/check_truncated_margin.py（リポジトリの一番上で）
with osc = |h|/T (additive) and osc = max shift of seen logits / T (multiplicative, theta = e^h)."""
import sys, itertools
import numpy as np
sys.path.insert(0, 'scripts')
import check_truncated_identity as m

V, L, T, PROMPT = m.V, m.L, m.T, m.PROMPT


def osc(hist, h, kind):
    if kind == 'add':
        return abs(h) / T
    z0 = m.raw_logits(hist)
    return float(np.max(np.abs(m.penalized(hist, h, kind) - z0))) / T


def stats(h, kind):
    w_change = w_low = w_tot = 0.0; viol = 0
    for seq in itertools.product(range(V), repeat=L):
        hist = list(PROMPT); logp = 0.0
        for x in seq:
            p0, k0 = m.step_probs(hist, 0.0, kind, 'minp')
            if p0[x] == 0:
                break
            _, kh = m.step_probs(hist, h, kind, 'minp')
            m0 = m.margin_minp(hist, 0.0, kind)
            P = np.exp(logp)
            ch = bool(np.any(k0 != kh)); low = m0 <= osc(hist, h, kind)
            w_change += P * ch; w_low += P * low; w_tot += P
            viol += int(ch and not low)
            logp += np.log(p0[x]); hist.append(x)
    return w_change / w_tot, w_low / w_tot, viol


for kind, hs in (('add', (-0.02, -0.05, -0.1, -0.15, -0.2, -0.4, -0.8)),
                 ('mult', (0.02, 0.05, 0.1, 0.2, 0.4))):
    for h in hs:
        c, lo, v = stats(h, kind)
        print(f'{kind:4s} h={h:+.2f}  changed {c:.4f}  margin<=osc {lo:.4f}  changed despite margin>osc {v}')
