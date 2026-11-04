"""探索：偽薬（塩）で本当の融点 T_m は下がるか（A2-7b のトークンの下見、2026-10-30 13:4x）。予想なし。CPU。生成も読み直しもしない。
scripts/a2_explore_onset.py（copy_marks）と scripts/a2_8_vh.py（phases・doc_counts・fit_vh・A8_TU）を実行したのと同じ名前空間で実行する。

オーナー（2026-10-30 13:37）：（オーナーの承認・発言。原文は非公開の記録にある）
（13:38）（オーナーの承認・発言。原文は非公開の記録にある）
やること：A2-7b の「なし」「なし（別の種）」「偽薬 2.5%・5%・7.5%」の腕（温度 0.4〜1.0、サンプル 2、記事 40）で、A2-8 と同じ決め方
（写しの印・中心の窓・入口 0.5／出口 0.25・凍る回と溶ける回がそれぞれ 10 回以上の温度）で T_m を出し、「なし」との比を記事の組の
選び直し（同じ記事の選び方を全部の腕で使う）で比べる。写しの禁止の腕は、結晶を見つける印そのものを禁止しているので使わない。
"""
import numpy as np

TS_ARMS = ['none', 'none_b', 'pl025', 'pl050', 'pl075']
TS_DOSE = {'none': 0.0, 'none_b': 0.0, 'pl025': 0.025, 'pl050': 0.05, 'pl075': 0.075}
TS_TEMPS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
TS_MODELS = ['pythia-1b', 'olmo2-1b', 'qwen2.5-1.5b']
TS_BOOT, TS_SEED = 1000, 20261115


def counts_arm(toks):
    """toks：(8, S, nD, L) → (17, nD, 10)。U と HT は使わないので 0 を入れる（熱量計の ΔH は 0 になり、n は出ない）。"""
    nT, S, nD, L = toks.shape
    out = np.zeros((len(A8_TU), nD, 10))
    z = np.zeros((S, L))
    for k, T in enumerate(TS_TEMPS):
        ku = A8_TU.index(T)
        for j in range(nD):
            C = np.stack([copy_marks(toks[k, s, j]) for s in range(S)])
            out[ku, j] += doc_counts(C, z, z)
    return out


def tm_salt(base):
    res = {'models': {}, 'arms': TS_ARMS, 'temps': TS_TEMPS}
    for m in TS_MODELS:
        cnt = {}
        for a in TS_ARMS:
            Z = np.load(f'{base}/A2_7B/{m}@{a}.npz')
            assert [float(t) for t in Z['temps']] == TS_TEMPS
            cnt[a] = counts_arm(Z['toks'])
        pt = {a: fit_vh(cnt[a].sum(1)) for a in TS_ARMS}
        use = pt['none']['use_idx']
        row = {'point': {a: dict(Tm=pt[a].get('Tm'), dH_vH=pt[a].get('dH_vH'), use_idx=pt[a]['use_idx'], lnK=pt[a]['lnK'],
                                 n_freeze=cnt[a].sum(1)[:, 1].tolist(), n_melt=cnt[a].sum(1)[:, 3].tolist()) for a in TS_ARMS}}
        # 同じ温度の組（「なし」で使えた温度）で当てはめ直したもの（腕どうしを同じ物差しで比べる）
        same = {a: fit_vh(cnt[a].sum(1), temps_use=use) for a in TS_ARMS}
        row['same_temps'] = {a: dict(Tm=same[a].get('Tm'), dH_vH=same[a].get('dH_vH'), ok=same[a]['ok']) for a in TS_ARMS}
        nD = cnt['none'].shape[1]
        rng = np.random.default_rng(TS_SEED)
        y = {a: [] for a in TS_ARMS if a != 'none'}
        fail = 0
        for _ in range(TS_BOOT):
            w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
            f = {a: fit_vh(np.tensordot(cnt[a], w, axes=([1], [0])), temps_use=use) for a in TS_ARMS}
            if not all(f[a]['ok'] and np.isfinite(f[a]['Tm']) for a in TS_ARMS):
                fail += 1
                continue
            for a in y:
                y[a].append(float(np.log(f[a]['Tm'] / f['none']['Tm'])))
        row['y_logTm'] = {a: dict(point=(float(np.log(same[a]['Tm'] / same['none']['Tm'])) if same[a]['ok'] and same['none']['ok'] else None),
                                  ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) >= 20 else None)
                          for a, v in y.items()}
        row['boot_fail'] = fail / TS_BOOT
        res['models'][m] = row
    return res


def self_test_tm_salt():
    """作り物：凍る速さを濃さで下げた鎖 → 偽薬の腕の T_m が「なし」より下がることを拾う。"""
    import math
    rng = np.random.default_rng(0)
    L, nD, S = 512, 40, 2

    def make(shift):
        toks = np.zeros((len(TS_TEMPS), S, nD, L), np.int64)
        for k, T in enumerate(TS_TEMPS):
            lnK = 6.0 / T - 6.0 / (0.6 - shift)
            km = 0.01 * math.exp(-lnK / 2); kf = 0.01 * math.exp(lnK / 2)
            for s in range(S):
                for j in range(nD):
                    st, seq, per = 0, [], list(rng.integers(100, 30000, 7))
                    for t in range(L):
                        if t > 0:
                            st = (1 if rng.random() < kf else 0) if st == 0 else (0 if rng.random() < km else 1)
                        seq.append(per[t % 7] if st else int(rng.integers(100, 30000)))
                    toks[k, s, j] = seq
        return toks
    a, b = counts_arm(make(0.0)), counts_arm(make(0.06))
    ta, tb = fit_vh(a.sum(1))['Tm'], fit_vh(b.sum(1))['Tm']
    assert 0.55 < ta < 0.66 and 0.49 < tb < 0.60 and tb < ta - 0.03, (ta, tb)
    return 'OK'
