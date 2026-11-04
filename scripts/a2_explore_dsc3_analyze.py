"""探索 8.39 の分析：DSC その 3（高温高圧：h ＝ 0・＋0.5・＋1・＋2、2.0 ⇄ 0.35）。データを開く前に書いた（2026-10-31 18:2x）。CPU、判定しない。
scripts/a2_explore_onset.py（copy_marks）・a2_8_vh.py（phases、A8_W）を実行したのと同じ名前空間で実行する。

相：2 つの状態（A2-8 と同じ窓のものさし、8.33・8.38 と同じ）。
道：冷やす道 ＝ 歩 0〜799（2.0 → 0.35）、温める道 ＝ 歩 800〜1599（0.35 → 2.0）。温度の箱：0.35〜2.0 を 0.05 ずつ（33 個）。
量（モデル × 腕）：
  φ_冷(T)・φ_温(T)：箱ごとの結晶の割合。φ0 ＝ 温める道のいちばん低い箱の φ。高さ ＝ φ0/2。
  T_melt：温める道で、低い温度から上へ見て、はじめて φ < 高さ になる温度（箱の真ん中の温度のあいだを直線でつなぐ）。横切らなければ打ち切り（nan）。
  T_freeze：冷やす道で、高い温度から下へ見て、はじめて φ ≥ 高さ になる温度（同じつなぎ方）。いちばん高い箱ですでに高さ以上なら打ち切り（nan）。
  幅 ＝ T_melt − T_freeze（ヒステリシスの幅。0 に縮めば、凍ると溶けるが同じ温度 ＝ 一次転移の跳びが消える向き）。
  ΔH_熱量計(T)：道ごと・箱ごとに、液体の歩と結晶の歩の U の平均の差（両方 200 歩以上）。ΔH_melt ＝ 温める道の ΔH を T_melt で、ΔH_freeze ＝ 冷やす道の ΔH を T_freeze で（出た箱のあいだを直線でつなぐ、端は端の値）。
  ΔM(T)：同じく、結晶の歩と液体の歩の「候補を選んだ割合」の差。ΔM_melt・ΔM_freeze。
  潜熱（結晶 1 トークンあたり）：凍る（冷やす道）・溶ける（温める道）、8.38 と同じ Σ ΔH × |Δφ| ÷ Σ|Δφ|。ΔH が出ない箱は、その道の出た箱からの直線のつなぎで埋める。
  中間の窓の割合 f_mid：32 歩の窓の写しの割合 r が 0.3〜0.7 の窓の割合（転移の温度 ±0.1 の箱、温める道は T_melt、冷やす道は T_freeze のまわり）。
    はっきりした一次転移なら、窓は液体（r 小）か結晶（r 大）にほぼ分かれ、f_mid は小さい。境目がぼやける（臨界点に近づく・超えて連続になる）と大きくなる。
    比べのため、作り物の一次転移と連続の変わり方の f_mid も記録する（self_test）。r の 10 個の箱のヒストグラムも記録（区間なし）。
  場の向きの傾き：T_melt・T_freeze を h にまっすぐ当てはめた傾き（打ち切りでない腕だけ、3 つ以上）。
  h ＝ 0 との差：各腕の T_melt・T_freeze・幅・ΔH_melt・ΔH_freeze・ΔM_melt・f_mid_melt・f_mid_freeze − h ＝ 0 の値。
区間：記事の選び直し 500 回（種 20261140、モデルの中で 4 本の腕に同じ選び方）。
"""
import numpy as np

D3A_ARMS = ['h0', 'hp05', 'hp10', 'hp20']
D3A_H = {'h0': 0.0, 'hp05': 0.5, 'hp10': 1.0, 'hp20': 2.0}
D3A_E = np.round(np.arange(0.35, 2.0 + 1e-9, 0.05), 4)
D3A_C = (D3A_E[:-1] + D3A_E[1:]) / 2
D3A_N = 800
D3A_MIN = 200
D3A_MID = (0.3, 0.7)
D3A_WIN = 0.1
D3A_BOOT, D3A_SEED = 500, 20261140
D3A_COLS = 9   # [結晶の歩, 歩, U 液体の和, 液体の歩, U 結晶の和, 選んだ 液体の和, 選んだ 結晶の和, 中間の窓, r の和]


def d3_binidx(T):
    return np.clip(np.searchsorted(D3A_E, T, side='right') - 1, 0, len(D3A_E) - 2)


def doc_bins3(toks, U, pick, Ts, sl):
    """toks・U・pick：(S, nD, L) → (nD, 33, 9) と r のヒストグラム用の (nD, 33, 10)。"""
    S, nD, L = toks.shape
    nb = len(D3A_E) - 1
    out = np.zeros((nD, nb, D3A_COLS)); hist = np.zeros((nD, nb, 10))
    b = d3_binidx(Ts)
    for j in range(nD):
        for s in range(S):
            c = copy_marks(toks[s, j])
            st, t = phases(c)
            r = np.convolve(c.astype(np.float64), np.ones(A8_W), 'valid') / A8_W
            r = r[:len(st)]
            m = (t >= sl.start) & (t < sl.stop)
            x = st[m].astype(float); tt = t[m]; rr = r[m]
            u = U[s, j, tt].astype(float); pk = pick[s, j, tt].astype(float); bb = b[tt]
            mid = ((rr >= D3A_MID[0] - 1e-12) & (rr <= D3A_MID[1] + 1e-12)).astype(float)
            for col, val in ((0, x), (1, np.ones_like(x)), (2, u * (1 - x)), (3, 1 - x), (4, u * x), (5, pk * (1 - x)), (6, pk * x), (7, mid), (8, rr)):
                out[j, :, col] += np.bincount(bb, weights=val, minlength=nb)
            hb = np.clip((rr * 10).astype(int), 0, 9)
            np.add.at(hist[j], (bb, hb), 1.0)
    return out, hist


def _ratio(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return a / b


def leg_curves(s):
    """s：(33, 9)（記事を足したもの）→ φ、ΔH、ΔM、f_mid（箱ごと）。"""
    phi = _ratio(s[:, 0], s[:, 1])
    ok = (s[:, 3] >= D3A_MIN) & (s[:, 0] >= D3A_MIN)
    dH = np.where(ok, _ratio(s[:, 2], s[:, 3]) - _ratio(s[:, 4], s[:, 0]), np.nan)
    dM = np.where(ok, _ratio(s[:, 6], s[:, 0]) - _ratio(s[:, 5], s[:, 3]), np.nan)
    return phi, dH, dM


def interp_at(y, T):
    ok = np.isfinite(y)
    if not ok.any() or not np.isfinite(T):
        return float('nan')
    return float(np.interp(T, D3A_C[ok], y[ok]))


def fill_line(y):
    ok = np.isfinite(y)
    if not ok.any():
        return y
    return np.interp(D3A_C, D3A_C[ok], y[ok])


def t_melt3(phi, lvl):
    ok = np.isfinite(phi)
    if not ok.any() or not np.isfinite(lvl):
        return float('nan')
    for k in range(1, len(phi)):
        if np.isfinite(phi[k]) and phi[k] < lvl:
            a = phi[k - 1]
            if not np.isfinite(a) or a == phi[k]:
                return float(D3A_C[k])
            return float(D3A_C[k - 1] + (a - lvl) / (a - phi[k]) * (D3A_C[k] - D3A_C[k - 1]))
    return float('nan')


def t_freeze3(phi, lvl):
    """冷やす道（箱は温度の低い順）：高い温度から下へ見て、はじめて φ ≥ 高さ。いちばん高い（出た）箱ですでに高さ以上なら nan。"""
    idx = np.flatnonzero(np.isfinite(phi))
    if not len(idx) or not np.isfinite(lvl):
        return float('nan')
    top = idx[-1]
    if phi[top] >= lvl:
        return float('nan')
    for k in range(top - 1, -1, -1):
        if np.isfinite(phi[k]) and phi[k] >= lvl:
            a = phi[k + 1]
            if not np.isfinite(a) or a == phi[k]:
                return float(D3A_C[k])
            return float(D3A_C[k + 1] - (lvl - a) / (phi[k] - a) * (D3A_C[k + 1] - D3A_C[k]))
    return float('nan')


def latent3(phi, dH):
    tot, frac = 0.0, 0.0
    for k in range(len(phi) - 1):
        a, b = phi[k], phi[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)) or a <= b:
            continue
        h = np.nanmean([dH[k], dH[k + 1]]) if (np.isfinite(dH[k]) or np.isfinite(dH[k + 1])) else np.nan
        if np.isfinite(h):
            tot += h * (a - b); frac += a - b
    return tot / frac if frac > 0 else float('nan'), frac


def f_mid_near(s, T):
    if not np.isfinite(T):
        return float('nan')
    k = np.abs(D3A_C - T) <= D3A_WIN + 1e-9
    n = s[k, 1].sum()
    return float(s[k, 7].sum() / n) if n > 0 else float('nan')


def arm_point(cool, heat):
    """cool・heat：(33, 9)（記事を足したもの）→ 腕の量。"""
    pc, dHc, dMc = leg_curves(cool); ph, dHh, dMh = leg_curves(heat)
    fin = np.isfinite(ph)
    p0 = float(ph[np.argmax(fin)]) if fin.any() else float('nan')
    lvl = p0 / 2
    tm, tf = t_melt3(ph, lvl), t_freeze3(pc, lvl)
    exo, fexo = latent3(pc, fill_line(dHc)); endo, fendo = latent3(ph, fill_line(dHh))
    return dict(phi_cool=pc.tolist(), phi_heat=ph.tolist(), dH_cool=dHc.tolist(), dH_heat=dHh.tolist(), dM_cool=dMc.tolist(), dM_heat=dMh.tolist(),
                fmid_cool=_ratio(cool[:, 7], cool[:, 1]).tolist(), fmid_heat=_ratio(heat[:, 7], heat[:, 1]).tolist(),
                phi0=p0, T_melt=tm, T_freeze=tf, width=tm - tf,
                dH_melt=interp_at(dHh, tm), dH_freeze=interp_at(dHc, tf), dM_melt=interp_at(dMh, tm), dM_freeze=interp_at(dMc, tf),
                exo_per=exo, exo_frac=fexo, endo_per=endo, endo_frac=fendo,
                fmid_melt=f_mid_near(heat, tm), fmid_freeze=f_mid_near(cool, tf))


D3A_KEYS = ['T_melt', 'T_freeze', 'width', 'dH_melt', 'dH_freeze', 'dM_melt', 'dM_freeze', 'exo_per', 'endo_per', 'fmid_melt', 'fmid_freeze']


def slope_h(vals):
    h = np.array([D3A_H[a] for a in D3A_ARMS]); y = np.array([vals[a] for a in D3A_ARMS], float); ok = np.isfinite(y)
    return float(np.polyfit(h[ok], y[ok], 1)[0]) if ok.sum() >= 3 else float('nan')


def point3(C, H):
    """C・H：{腕: (nD, 33, 9)} → 量。"""
    res = {a: arm_point(C[a].sum(0), H[a].sum(0)) for a in D3A_ARMS}
    for k in ('T_melt', 'T_freeze'):
        res[f'slope_{k}'] = slope_h({a: res[a][k] for a in D3A_ARMS})
    for a in D3A_ARMS[1:]:
        for k in D3A_KEYS:
            res[f'd_{k}_{a}'] = res[a][k] - res['h0'][k]
    return res


def analyze_dsc3(base_dir, name):
    Z = {a: np.load(f'{base_dir}/{name}@{a}.npz') for a in D3A_ARMS}
    uses = [np.asarray(Z[a]['use']).tolist() for a in D3A_ARMS]
    assert all(u == uses[0] for u in uses), name
    C, H, HC, HH = {}, {}, {}, {}
    for a in D3A_ARMS:
        Ts = np.asarray(Z[a]['Ts']); assert len(Ts) == 2 * D3A_N and abs(float(np.asarray(Z[a]['Hs'])[0]) - D3A_H[a]) < 1e-9
        C[a], HC[a] = doc_bins3(Z[a]['toks'], Z[a]['U'], Z[a]['pick'], Ts, slice(0, D3A_N))
        H[a], HH[a] = doc_bins3(Z[a]['toks'], Z[a]['U'], Z[a]['pick'], Ts, slice(D3A_N, 2 * D3A_N))
    pt = point3(C, H)
    # r のヒストグラム（転移の温度 ±0.1、記事を足したもの）
    for a in D3A_ARMS:
        for leg, hh, T in (('heat', HH[a], pt[a]['T_melt']), ('cool', HC[a], pt[a]['T_freeze'])):
            if np.isfinite(T):
                k = np.abs(D3A_C - T) <= D3A_WIN + 1e-9
                pt[a][f'rhist_{leg}'] = hh.sum(0)[k].sum(0).tolist()
    nD = C['h0'].shape[0]
    rng = np.random.default_rng(D3A_SEED)
    flat = lambda r: {**{f'{a}|{k}': r[a][k] for a in D3A_ARMS for k in D3A_KEYS},
                      **{k: v for k, v in r.items() if not isinstance(v, dict)}}
    bs = []
    for _ in range(D3A_BOOT):
        idx = rng.integers(0, nD, nD)
        bs.append(flat(point3({a: v[idx] for a, v in C.items()}, {a: v[idx] for a, v in H.items()})))
    pt['ci'] = {}; pt['censored_frac'] = {}
    for k in bs[0]:
        v = np.array([b[k] for b in bs], float)
        pt['ci'][k] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float('nan')] * 2
        pt['censored_frac'][k] = float(np.mean(~np.isfinite(v)))
    return pt


# ---- 作り物（自己テスト・較正）----
def _synth(kind, seed=3, n_doc=40, ta=0.6, tb=1.2):
    """kind ＝ 'first'：冷やす道で T < ta から結晶（周期 23 のくり返し）、温める道で T > tb で溶ける（記事ごとに ±0.04）。結晶の U ＝ 0.5、液体 3、
    結晶の歩は必ず候補を選ぶ（pick ＝ 1）、液体は 0.1。
    kind ＝ 'cont'：毎歩、確率 p(T) ＝ 1/(1＋e^((T − 0.9)/0.1)) で「くり返しの続き」、そうでなければでたらめ（境目のない、なだらかな変わり方）。"""
    rng = np.random.default_rng(seed)
    Ts = np.r_[np.linspace(2.0, 0.35, D3A_N), np.linspace(0.35, 2.0, D3A_N)]
    L = len(Ts)
    toks = np.zeros((4, n_doc, L), np.int64); U = np.full((4, n_doc, L), 3.0, np.float32); pick = np.zeros((4, n_doc, L), bool)
    for s in range(4):
        for j in range(n_doc):
            seq = rng.integers(100, 30000, L); blk = rng.integers(100, 30000, 23)
            pick[s, j] = rng.random(L) < 0.1
            if kind == 'first':
                a = int(np.argmax(Ts < ta + 0.02 * (j % 5 - 2))); b = D3A_N + int(np.argmax(Ts[D3A_N:] > tb + 0.02 * (j % 5 - 2)))
                seq[a:b] = np.tile(blk, (b - a) // 23 + 1)[:b - a]
                U[s, j, a + 23:b] = 0.5; pick[s, j, a + 23:b] = True
            else:
                p = 1 / (1 + np.exp((Ts - 0.9) / 0.1)); ph = 0
                for t in range(L):
                    if rng.random() < p[t]:
                        seq[t] = blk[ph % 23]; ph += 1
                    else:
                        ph = 0
            toks[s, j] = seq
    C = doc_bins3(toks, U, pick, Ts, slice(0, D3A_N))[0]; H = doc_bins3(toks, U, pick, Ts, slice(D3A_N, L))[0]
    return arm_point(C.sum(0), H.sum(0))


def self_test_dsc3_an():
    f = _synth('first')
    # 窓のものさしは、結晶を窓が埋まってから見つけるので、T_freeze は少し低く、ΔH・ΔM は真の値（2.5・0.9）より小さく出る（較正として値を記録する）
    assert abs(f['T_freeze'] - 0.6) < 0.08 and abs(f['T_melt'] - 1.2) < 0.06, (f['T_freeze'], f['T_melt'])
    assert abs(f['width'] - 0.6) < 0.12, f['width']
    assert 1.0 < f['dH_melt'] < 2.6 and 1.0 < f['dH_freeze'] < 2.6, (f['dH_melt'], f['dH_freeze'])
    assert f['dM_melt'] > 0.4 and f['dM_freeze'] > 0.4, (f['dM_melt'], f['dM_freeze'])
    c = _synth('cont')
    assert c['fmid_melt'] > 2 * f['fmid_melt'] and c['fmid_freeze'] > 2 * f['fmid_freeze'], (c['fmid_melt'], f['fmid_melt'], c['fmid_freeze'], f['fmid_freeze'])
    assert abs(c['width']) < 0.2, c['width']   # なめらかでも、窓のものさしの履歴（入口 0.5・出口 0.25）で見かけの幅が少し出る（較正）
    # 打ち切り：いちばん高い箱ですでに結晶なら T_freeze は nan、溶けなければ T_melt は nan
    assert np.isnan(t_freeze3(np.full(33, 0.9), 0.45)) and np.isnan(t_melt3(np.full(33, 0.9), 0.45))
    return dict(first=dict(T_freeze=f['T_freeze'], T_melt=f['T_melt'], width=f['width'], fmid_melt=f['fmid_melt'], fmid_freeze=f['fmid_freeze'],
                           dH_melt=f['dH_melt'], dH_freeze=f['dH_freeze'], dM_melt=f['dM_melt'], dM_freeze=f['dM_freeze'], exo_per=f['exo_per'], endo_per=f['endo_per']),
                cont=dict(T_freeze=c['T_freeze'], T_melt=c['T_melt'], width=c['width'], fmid_melt=c['fmid_melt'], fmid_freeze=c['fmid_freeze']))
