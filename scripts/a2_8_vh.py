"""A2-8（ゲート docs/A2_8_GATE.md）：ファントホッフの ΔH と熱量計の ΔH。GPU（L4）で読み直し、そのあと同じノートブックで分析（CPU の計算）。
scripts/a2_2_collect.py（load_model・valid_mask・step_stats・h2_of・CONT）と scripts/a2_explore_onset.py（copy_marks）を
実行したのと同じ名前空間で実行する。分析の部分はデータを開く前に固定する。

材料（新しく生成はしない）：
- A2-2 の生成（温度 15 × サンプル 2 × 記事 40 × 512 歩）：`A2_2/{name}.npz` の toks。書き出しは A2-7b の npz の prompts を使う
  （A2-7b は A2-2 と同じ記事・同じ切り方。use が同じことを確かめる）。
- A2-7b の「なし」と「なし（別の種）」（温度 8 × サンプル 2 × 記事 40 × 512 歩）：`A2_7B/{name}@none.npz`・`@none_b.npz`。
読み直し（teacher forcing）：書き出し ＋ 生成したトークンをモデルに通し、各歩で
- U ＝ −log p₁(選んだトークン)（温度 1、使わない番号と EOS を外す。生成のときと同じ valid）
- H1 ＝ 温度 1 の分布のエントロピー、HT ＝ 生成した温度の分布のエントロピー
- ELL ＝ 生成した温度の ℓ（順位 H 以降）。保存してある ℓ と比べて、トークンと位置がずれていないことを確かめる。
"""
import gc, math, os, time
import numpy as np
import torch

A8_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')]
A8_T22 = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
A8_T7B = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
A8_TU = sorted(set(A8_T22) | set(A8_T7B))                     # 17 の温度
A8_SOURCES = [('a22', 'A2_2/{name}.npz', A8_T22), ('b7n', 'A2_7B/{name}@none.npz', A8_T7B), ('b7b', 'A2_7B/{name}@none_b.npz', A8_T7B)]
A8_BS = 4
# T_0 のものさし：A2-2 の結果（fb_H の T0 と 95% 区間）。A2-2 の生成とは同じデータから出ているが、決め方（ℓ の交点）は別。
A8_T0 = {'pythia-1b': (0.852045, 0.810311, 0.869373), 'olmo2-1b': (0.800822, 0.775764, 0.856064),
         'qwen2.5-1.5b': (0.910086, 0.861524, 0.956799)}
# 相の決め方
A8_W = 32                 # 窓（中心をそろえる：歩 t の相は t−16〜t+15 の写しの印の平均で決める）
A8_ENTER, A8_EXIT = 0.5, 0.25
A8_SENS_EXIT = [0.5, 0.1]  # 記録用の別の出口
A8_MIN_EV = 10            # 使う温度：凍る回と溶ける回が、それぞれこれ以上
A8_MIN_TEMPS = 3
A8_BOOT, A8_BOOT_SEED = 1000, 20261112
A8_Q1_BAND = 1.10
A8_Q2_BINS = [(0.5, '1 より小さい（途中の段がある）'), (2.0, '1 くらい（ばらばらに溶ける）'), (10.0, '数トークン（2〜10）'),
              (100.0, '数十トークン（10〜100、結晶核くらい）'), (float('inf'), '100 トークン以上')]


# ---------- 読み直し（GPU） ----------

@torch.no_grad()
def score_seqs(model, prompts, toks, temps, valid, H2, device):
    """prompts：(N, P) の list、toks：(N, L)、temps：(N,)。→ U, H1, HT, ELL：(N, L) の float32。"""
    N, L = toks.shape
    out = {k: np.zeros((N, L), np.float32) for k in ('U', 'H1', 'HT', 'ELL')}
    for i0 in range(0, N, A8_BS):
        ii = list(range(i0, min(N, i0 + A8_BS)))
        P = len(prompts[ii[0]])
        x = torch.tensor([list(prompts[i]) + list(map(int, toks[i])) for i in ii], device=device)
        lg = model(x).logits[:, P - 1: P - 1 + L].float()                  # 歩 t の logit は toks[t] を選ぶ前のもの
        for b, i in enumerate(ii):
            z = lg[b]
            y = torch.tensor(toks[i], device=device, dtype=torch.long)
            z1 = z.double().masked_fill(~valid, float('-inf'))
            lp1 = torch.log_softmax(z1, dim=-1)
            p1 = torch.exp(lp1)
            out['U'][i] = (-lp1.gather(-1, y[:, None])[:, 0]).float().cpu().numpy()
            out['H1'][i] = (-torch.where(p1 > 0, p1 * lp1, torch.zeros_like(p1)).sum(-1)).float().cpu().numpy()
            ell, _, ent, _ = step_stats(z, float(temps[i]), valid, H2)
            out['ELL'][i] = ell.float().cpu().numpy(); out['HT'][i] = ent.float().cpu().numpy()
    return out


def load_sources(base, name):
    """→ {source: dict(toks=(nT, S, nD, L), temps, stored_ell=(nT, S, nD, L))}, prompts（A2-7b の npz から）, use。"""
    Zb = np.load(f'{base}/A2_7B/{name}@none.npz')
    prompts = [list(map(int, p)) for p in Zb['prompts']]
    use = [int(u) for u in Zb['use']]
    src = {}
    for key, pat, temps in A8_SOURCES:
        Z = np.load(f'{base}/' + pat.format(name=name))
        assert [float(t) for t in Z['temps']] == temps, (name, key)
        assert [int(u) for u in Z['use']] == use, ('記事の並びが違います', name, key)
        toks = Z['toks']
        ell = Z['gen'][0] if key == 'a22' else Z['g_ell']
        if key == 'a22':
            assert int(Z['prompt_len']) == len(prompts[0]), ('書き出しの長さが A2-2 と違います', name, int(Z['prompt_len']), len(prompts[0]))
        assert toks.shape == ell.shape and toks.shape[2] == len(use), (toks.shape, ell.shape)
        src[key] = dict(toks=toks, temps=temps, stored_ell=ell)
    return src, prompts, use


def run_score(name, hf_id, base, outdir, device, notify=print):
    path = f'{outdir}/{name}.npz'
    if os.path.exists(path):
        print(name, 'はもう読み直しが終わっています（飛ばします）')
        return
    t0 = time.time()
    src, prompts, use = load_sources(base, name)
    tok, model = load_model(hf_id, device)
    assert next(model.parameters()).dtype == torch.float32
    valid, _ = valid_mask(tok, model, device)
    H2 = h2_of(len(tok))
    save, chk = {}, {}
    for key, d in src.items():
        nT, S, nD, L = d['toks'].shape
        assert L == CONT
        flat = d['toks'].reshape(-1, L)
        temps = np.repeat(np.array(d['temps']), S * nD)
        prm = [prompts[j] for _ in range(nT) for _ in range(S) for j in range(nD)]
        r = score_seqs(model, prm, flat, temps, valid, H2, device)
        diff = np.abs(r['ELL'] - d['stored_ell'].reshape(-1, L))
        diff = diff[np.isfinite(diff)]
        chk[key] = dict(med=float(np.median(diff)), q99=float(np.quantile(diff, 0.99)), mx=float(diff.max()))
        # トークンと位置がずれていたら、ℓ は何単位も違う（数値のずれは 1/1000 くらい）
        assert chk[key]['med'] < 0.05, ('読み直しの ℓ が保存した ℓ と合いません（トークンか位置のずれ）', name, key, chk[key])
        for k, v in r.items():
            if k != 'ELL':
                save[f'{key}_{k}'] = v.reshape(nT, S, nD, L)
        save[f'{key}_toks'] = d['toks']
        notify(f'{name}・{key} の読み直しおわり（{time.time() - t0:.0f} 秒、ℓ のずれの中央値 {chk[key]["med"]:.1e}）')
    np.savez_compressed(path + '.part.npz', use=np.array(use), check=np.array([json_dumps(chk)]), **save)
    os.replace(path + '.part.npz', path)
    del model; gc.collect(); torch.cuda.empty_cache()


def json_dumps(o):
    import json
    return json.dumps(o, ensure_ascii=False)


@torch.no_grad()
def self_test_score(device):
    """小さなランダムのモデルで：KV キャッシュで 1 歩ずつ出した ℓ・エントロピーと、読み直しの値が合う。U が −log p₁ と合う。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    prompts = [[5, 6, 7, 8] * 4, [9, 10, 11, 12] * 4, [3, 4, 3, 4] * 4, [20, 21, 22, 23] * 4, [30, 31, 32, 33] * 4]
    L, T = 40, 0.7
    x = torch.tensor(prompts, device=device)
    out = model(x, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :].float()
    g = torch.Generator(device=device).manual_seed(0)
    E, Ent, U, Y = [], [], [], []
    for t in range(L):
        ell, _, ent, _ = step_stats(logits, T, valid, 33)
        z = logits.double().masked_fill(~valid, float('-inf'))
        nxt = torch.multinomial(torch.softmax(z / T, -1), 1, generator=g)
        U.append(-torch.log_softmax(z, -1).gather(-1, nxt)[:, 0]); E.append(ell); Ent.append(ent); Y.append(nxt[:, 0])
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
    st = lambda xs: torch.stack(xs, 1).double().cpu().numpy()
    toks = torch.stack(Y, 1).cpu().numpy()
    r = score_seqs(model, prompts, toks, np.full(len(prompts), T), valid, 33, device)
    assert np.abs(r['ELL'] - st(E)).max() < 1e-3 and np.abs(r['HT'] - st(Ent)).max() < 1e-3 and np.abs(r['U'] - st(U)).max() < 1e-3
    # ずらすと合わない（確かめの仕組みが効く）
    r2 = score_seqs(model, prompts, np.roll(toks, 1, axis=1), np.full(len(prompts), T), valid, 33, device)
    assert np.abs(r2['U'] - st(U)).max() > 1e-2
    return 'OK'


# ---------- 分析（CPU） ----------

def phases(c, exit_th=A8_EXIT):
    """写しの印 c（長さ L）→ 歩 16〜L−16 の相（0 ＝ 液体、1 ＝ 結晶）と、その歩の番号。入口 0.5、出口 exit_th の履歴つき。"""
    L = len(c)
    r = np.convolve(c.astype(np.float64), np.ones(A8_W), 'valid') / A8_W   # r[k] は歩 k..k+31 の平均 → 中心の歩 k+16
    t = np.arange(len(r)) + A8_W // 2
    s = np.zeros(len(r), np.int8)
    cur = 1 if r[0] >= A8_ENTER - 1e-12 else 0
    for i, v in enumerate(r):
        if cur == 0 and v >= A8_ENTER - 1e-12:
            cur = 1
        elif cur == 1 and v < exit_th - 1e-12:
            cur = 0
        s[i] = cur
    return s, t


def doc_counts(C, U, HT, exit_th=A8_EXIT):
    """C・U・HT：(S, L)（同じ記事・同じ温度のサンプル）→ [N0, n01, N1, n10, ΣU液, n液, ΣU結, n結, ΣHT液, ΣHT結]。"""
    acc = np.zeros(10)
    for s in range(C.shape[0]):
        st, t = phases(C[s], exit_th)
        a, b = st[:-1], st[1:]
        acc[0] += (a == 0).sum(); acc[1] += ((a == 0) & (b == 1)).sum()
        acc[2] += (a == 1).sum(); acc[3] += ((a == 1) & (b == 0)).sum()
        liq, cry = st == 0, st == 1
        acc[4] += U[s, t[liq]].sum(); acc[5] += liq.sum(); acc[6] += U[s, t[cry]].sum(); acc[7] += cry.sum()
        acc[8] += HT[s, t[liq]].sum(); acc[9] += HT[s, t[cry]].sum()
    return acc


def build_counts(Z, exit_th=A8_EXIT):
    """Z：読み直しの npz（dict でもよい）→ (17 の温度, nD, 10)。同じ温度は 3 つの材料を足す。"""
    nD = len(Z['use'])
    out = np.zeros((len(A8_TU), nD, 10))
    for key, _, temps in A8_SOURCES:
        toks, U, HT = Z[f'{key}_toks'], Z[f'{key}_U'], Z[f'{key}_HT']
        for k, T in enumerate(temps):
            ku = A8_TU.index(T)
            for j in range(nD):
                C = np.stack([copy_marks(toks[k, s, j]) for s in range(toks.shape[1])])
                out[ku, j] += doc_counts(C, U[k, :, j].astype(np.float64), HT[k, :, j].astype(np.float64), exit_th)
    return out


def fit_vh(tot, temps_use=None):
    """tot：(17, 10)（記事を足したもの）→ 使う温度・ln K・ファントホッフの当てはめ・熱量計の ΔH。"""
    n01, n10, N0, N1 = tot[:, 1], tot[:, 3], tot[:, 0], tot[:, 2]
    ok = (n01 >= A8_MIN_EV) & (n10 >= A8_MIN_EV) if temps_use is None else np.isin(np.arange(len(A8_TU)), temps_use)
    idx = np.flatnonzero(ok)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnK = np.log(n01 / N0) - np.log(n10 / N1)
        dHcal = tot[:, 4] / tot[:, 5] - tot[:, 6] / tot[:, 7]
        dScal = tot[:, 8] / tot[:, 5] - tot[:, 9] / tot[:, 7]
    res = dict(use_idx=idx.tolist(), lnK=lnK.tolist(), dHcal_T=dHcal.tolist(), dScal_T=dScal.tolist())
    if len(idx) < A8_MIN_TEMPS or not np.all(np.isfinite(lnK[idx])) or not np.all(np.isfinite(dHcal[idx])):
        res.update(ok=False)
        return res
    x = 1.0 / np.array(A8_TU)[idx]; y = lnK[idx]
    w = 1.0 / (1.0 / n01[idx] + 1.0 / n10[idx])
    X = np.stack([np.ones_like(x), x], 1)
    a, b = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
    Tm = (-b / a) if (b > 0 and a < 0) else float('nan')
    dH_cal = float(np.mean(dHcal[idx]))
    n = b / dH_cal if dH_cal > 0 else float('nan')
    res.update(ok=True, dH_vH=float(b), dS_vH=float(-a), Tm=float(Tm), dH_cal=dH_cal, n=float(n),
               dS_cal=float(np.mean(dScal[idx])), Tcal_ratio=float(dH_cal / np.mean(dScal[idx])) if np.mean(dScal[idx]) > 0 else float('nan'))
    if len(idx) >= 4:                                                      # 曲がり（記録）：2 次の係数
        X2 = np.stack([np.ones_like(x), x, x * x], 1)
        res['quad'] = np.linalg.solve(X2.T @ (w[:, None] * X2), X2.T @ (w * y)).tolist()
    return res


def q1_word(Tm_ci, T0):
    t0, lo0, hi0 = T0
    lo, hi = Tm_ci
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 'はっきりしない'
    if lo > hi0:
        return 'ずれる（高い）'
    if hi < lo0:
        return 'ずれる（低い）'
    if lo >= t0 / A8_Q1_BAND and hi <= t0 * A8_Q1_BAND:
        return '合う'
    return 'はっきりしない'


def q2_word(n):
    if not np.isfinite(n):
        return 'はっきりしない'
    if n <= 0:
        return '向きが逆'
    for edge, w in A8_Q2_BINS:
        if n < edge:
            return w


def analyze_model(cnt, T0, seed=A8_BOOT_SEED):
    """cnt：(17, nD, 10) → 点・選び直しの区間・Q1 の言葉。"""
    nD = cnt.shape[1]
    pt = fit_vh(cnt.sum(1))
    row = dict(point=pt, temps=A8_TU)
    tot = cnt.sum(1)
    row['by_T'] = [dict(T=T, n_freeze=int(tot[k, 1]), n_melt=int(tot[k, 3]), steps_liq=int(tot[k, 0]), steps_cry=int(tot[k, 2]),
                        k_f=float(tot[k, 1] / tot[k, 0]) if tot[k, 0] else None, k_m=float(tot[k, 3] / tot[k, 2]) if tot[k, 2] else None)
                   for k, T in enumerate(A8_TU)]
    if not pt['ok']:
        row.update(Q1_word='はっきりしない', ci=None, boot_fail=1.0)
        return row
    rng = np.random.default_rng(seed)
    keep, fail = {k: [] for k in ('dH_vH', 'Tm', 'dH_cal', 'n')}, 0
    for _ in range(A8_BOOT):
        w = np.bincount(rng.integers(0, nD, nD), minlength=nD).astype(float)
        r = fit_vh(np.tensordot(cnt, w, axes=([1], [0])), temps_use=pt['use_idx'])
        if not r['ok'] or not np.isfinite(r['Tm']):
            fail += 1
            continue
        for k in keep:
            keep[k].append(r[k])
    ff = fail / A8_BOOT
    ci = {k: ([float(np.percentile(v, 2.5, method='lower')), float(np.percentile(v, 97.5, method='higher'))] if len(v) >= 20 else [float('nan')] * 2)
          for k, v in keep.items()}
    row.update(ci=ci, boot_fail=ff)
    row['Q1_word'] = q1_word(ci['Tm'], T0) if ff <= 0.1 and np.isfinite(pt['Tm']) else 'はっきりしない'
    return row


def analyze8(Zs):
    """Zs：{モデル: 読み直しの npz}。→ まとめ。"""
    res = {'models': {}, 'temps': A8_TU}
    for name, _ in A8_MODELS:
        Z = Zs[name]
        cnt = build_counts(Z)
        row = analyze_model(cnt, A8_T0[name])
        row['T0_ref'] = A8_T0[name]
        row['sens_exit'] = {str(e): fit_vh(build_counts(Z, e).sum(1)) for e in A8_SENS_EXIT}
        # 確かめ（記録）：温度 1 の生成では、平均の U と平均の H1 は期待値で等しい
        k1 = A8_T22.index(1.0)
        row['check_T1_U_vs_H1'] = [float(Z['a22_U'][k1].mean()), float(Z['a22_H1'][k1].mean())]
        row['score_check'] = str(Z['check'][0]) if 'check' in Z else None
        res['models'][name] = row
    res['Q1'] = sum(res['models'][m]['Q1_word'] == '合う' for m, _ in A8_MODELS)
    ns = [res['models'][m]['point'].get('n', float('nan')) for m, _ in A8_MODELS]
    ns = [v for v in ns if np.isfinite(v)]
    res['Q2_n_each'] = ns
    res['Q2'] = q2_word(float(np.median(ns))) if len(ns) >= 2 else 'はっきりしない'
    return res


def self_test_a8():
    """作り物：2 つの状態を行き来する鎖を、凍る速さ k_f ＝ exp(a_f − E_f/T)・溶ける速さ k_m で作る。K ＝ k_f/k_m の
    ΔH ＝ E_m − E_f を当てはめで取り戻せるか、T_m が合うか、U の差（熱量計）から n が出るか。"""
    rng = np.random.default_rng(0)
    L, nD, S = 512, 40, 6
    dH_true, Tm_true = 6.0, 0.8                    # ln K ＝ dH/T − dH/Tm
    Uc, Ul = 0.2, 3.2                              # 熱量計の ΔH ＝ 3.0 → n ＝ 2
    Z = {'use': np.arange(nD), 'check': np.array(['{}'])}
    for key, _, temps in A8_SOURCES:
        nT = len(temps)
        toks = np.zeros((nT, 2, nD, L), np.int64); U = np.zeros((nT, 2, nD, L), np.float32)
        for k, T in enumerate(temps):
            lnK = dH_true / T - dH_true / Tm_true
            km = 0.01 * math.exp(-lnK / 2); kf = 0.01 * math.exp(lnK / 2)
            for s in range(2):
                for j in range(nD):
                    st, seq, u = 0, [], []
                    period = list(rng.integers(100, 30000, 7))
                    for t in range(L):
                        if t > 0:
                            st = (1 if rng.random() < kf else 0) if st == 0 else (0 if rng.random() < km else 1)
                        if st == 1:
                            seq.append(period[t % 7]); u.append(Uc)
                        else:
                            seq.append(int(rng.integers(100, 30000))); u.append(Ul)
                    toks[k, s, j] = seq; U[k, s, j] = u
        Z[f'{key}_toks'] = toks; Z[f'{key}_U'] = U; Z[f'{key}_HT'] = U.copy(); Z[f'{key}_H1'] = U.copy()
    row = analyze_model(build_counts(Z), (0.8, 0.78, 0.82))
    p = row['point']
    assert p['ok'] and abs(p['dH_vH'] - dH_true) / dH_true < 0.35, p
    assert abs(p['Tm'] - Tm_true) < 0.06, p
    assert 2.0 < p['dH_cal'] < 3.1 and 1.2 < p['n'] < 3.2, p
    # 言葉
    assert q1_word([0.79, 0.83], (0.8, 0.78, 0.82)) == '合う' and q1_word([0.9, 0.95], (0.8, 0.78, 0.82)) == 'ずれる（高い）'
    assert q1_word([0.6, 0.7], (0.8, 0.78, 0.82)) == 'ずれる（低い）' and q1_word([0.5, 1.1], (0.8, 0.78, 0.82)) == 'はっきりしない'
    assert q2_word(1.1) == '1 くらい（ばらばらに溶ける）' and q2_word(35) == '数十トークン（10〜100、結晶核くらい）' and q2_word(-2) == '向きが逆'
    # 相：はっきりした結晶の区間を拾う
    c = np.zeros(300, np.int8); c[100:200] = 1
    s, t = phases(c)
    assert s[t == 90].item() == 0 and s[t == 150].item() == 1 and s[t == 250].item() == 0
    return 'OK'
