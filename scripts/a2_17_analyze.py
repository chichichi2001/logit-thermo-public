"""A2-17 の分析 v0.2：文章の種類（論文・コード・小説）を変えても、A2-16 と同じ形（場の効きは見かけではない・場の効きが予言より小さい）が出るか。
データ（Drive `A2_17/`、3 種類 × 4 モデル × h ＝ −0.3・0・＋0.3）を集める前に書いた（2026-11-01 07:xx JST）。CPU。
A2-16 の凍結した分析（scripts/a2_16_analyze.py `7c5a60c2…`）を、そのまま種類ごとに当てる（analyze16 に、種類のフォルダと 4 つのモデルを渡す）。
名前空間は scripts/a2_16_run.py の load16ns と同じ。ここで足すのは、4 つのモデルでの決め方と、種類をまたいだ決め方と、記録だけ。

決め方（種類ごと。分母は 4 つ全部。出なかったモデルも「はっきりしない」として数える）：
  GQ1（場の効きは、ものさしの見かけではないか）：A2-16 の HQ1 の言葉（モデルごと、そのまま）が「両方で、…上がる」のモデルが 3 つ以上 → 「見かけではない」。
    「食い違う」が 2 つ以上 → 「ものさしで変わる」。それ以外は「はっきりしない」。
  GQ2（1 トークンあたりのクラペイロン）：A2-16 の HQ3 の言葉（モデルごと、そのまま）のうち、3 つ以上が同じ言葉（「はっきりしない」以外）ならその言葉。
    そうでなければ「モデルで分かれる・はっきりしない」。
種類をまたいだ決め方（3 種類）：
  GQ1：3 種類すべてが「見かけではない」→「文章の種類を越えて、見かけではない」。1 つでも「ものさしで変わる」→「種類によっては、ものさしで変わる」。それ以外は「はっきりしない」。
  GQ2：3 種類すべてが「場の効きが予言より小さい」→「文章の種類を越えて、場の効きが予言より小さい」。
    1 つでも、ほかの決まった言葉（閉じる・大きい・逆向き・区別できない）→「文章の種類で変わる」。それ以外は「はっきりしない」。
記録（判定しない）：種類ごと・モデルごとの R・場の傾き・T_m（h ＝ 0、窓）・n・熱量計の ΔH の分解。T_m が温度の範囲（0.4〜0.7）の外か。
  温度ごとの ∂ln K/∂h を 7 つの温度で足した比（恒等式の予言 ÷ 測った値、M の熱力学の予言 ÷ 測った値）。
  覚えているかの記録：人の書いた続きの −log p の平均（温度 1、仕事 'nll'）。BBC（A2-16 の結果）を並べる。
3 つの腕がそろわないモデルは、分析に入れず「はっきりしない」として数える（分母は 4 のまま）。
"""
import os, json
import numpy as np

A17_NAMES = ['pythia-1.4b', 'smollm2-1.7b', 'qwen2.5-1.5b', 'bloom-1b1']
A17_GEN = ['arxiv', 'code', 'fiction']
A17_MAJ = 3
A17_DECIDED = ('閉じる（2 割の内）', '場の効きが予言より大きい', '逆向き（場で融点が下がる）', '1 より小さい向きだが、ものさしのゆがみと区別できない')


def genre_words(models):
    names = list(models)
    w1 = [models[n]['words']['HQ1'] for n in names]
    n_up = sum(w.startswith('両方で、') for w in w1); n_dis = sum(w == '食い違う' for w in w1)
    g1 = '見かけではない' if n_up >= A17_MAJ else 'ものさしで変わる' if n_dis >= 2 else 'はっきりしない'
    w3 = [models[n]['words']['HQ3'] for n in names]
    g2 = majority(w3, k=A17_MAJ)[0]
    return dict(GQ1=(g1, dict(zip(names, w1))), GQ2=(g2, dict(zip(names, w3))))


def overall(byg):
    g1 = [byg[g]['GQ1'][0] for g in A17_GEN]
    g2 = [byg[g]['GQ2'][0] for g in A17_GEN]
    o1 = ('文章の種類を越えて、見かけではない' if all(w == '見かけではない' for w in g1)
          else '種類によっては、ものさしで変わる' if any(w == 'ものさしで変わる' for w in g1) else 'はっきりしない')
    o2 = ('文章の種類を越えて、場の効きが予言より小さい' if all(w == '場の効きが予言より小さい' for w in g2)
          else '文章の種類で変わる' if any(w in A17_DECIDED for w in g2) else 'はっきりしない')
    return dict(GQ1=o1, GQ2=o2)


def pooled_ratios(rec):
    """記録：7 つの温度で足した ∂ln K/∂h の比（恒等式 ÷ 測った値、M の熱力学 ÷ 測った値）。"""
    bt = rec['by_T']
    me = np.array([b['dlnK_meas'] for b in bt], float)
    ok = np.isfinite(me)
    out = {}
    for key, nm in (('dlnK_identity', 'identity_over_meas'), ('dlnK_M_thermo', 'M_over_meas')):
        v = np.array([b[key] for b in bt], float)
        o = ok & np.isfinite(v)
        out[nm] = float(v[o].sum() / me[o].sum()) if o.any() and me[o].sum() != 0 else float('nan')
    return out


def model_record(r, nll):
    p = r['point']
    rec = {k: p.get(k) for k in ('R', 'slope_w', 'slope_s', 'Tm_h0_w', 'Tm_hm03_w', 'Tm_hp03_w', 'n', 'dm', 'dHcal_h0_w', 'U_pick', 'U_nopick', 'cal_over_split')}
    rec['ci_R'] = r['ci']['R']
    tm = p.get('Tm_h0_w', float('nan'))
    rec['Tm_outside_0.4_0.7'] = bool(np.isfinite(tm) and not (0.4 <= tm <= 0.7))
    rec.update(pooled_ratios(r['records']))
    rec['human_nll_mean'] = float(np.nanmean(nll)) if nll is not None else float('nan')
    return rec


def load_nll(path, genre):
    if not os.path.exists(path):
        return None
    Z = np.load(path)
    assert str(Z['genre']) == genre, (path, str(Z['genre']))
    return np.asarray(Z['human_nll'], float)


def _missing_model():
    return dict(words=dict(HQ1='はっきりしない', HQ3='はっきりしない'), point={}, ci={'R': [float('nan')] * 2}, records={'by_T': []}, missing=True)


def analyze17(root, a16_summary='results/A2_16/a2_16_summary.json', names=A17_NAMES, genres=A17_GEN, boot=G16_BOOT):
    """3 つの腕の npz がそろわないモデルは、分析に入れず「はっきりしない」として数える（分母は 4 のまま）。種類まるごとなくても止まらない。"""
    res = {'genres': {}, 'records': {}, 'missing': {}}
    for g in genres:
        have = [n for n in names if all(os.path.exists(f'{root}/{g}/{n}@{a}.npz') for a, _ in G16_H)]
        res['missing'][g] = [n for n in names if n not in have]
        for n in have:
            Z = np.load(f'{root}/{g}/{n}@h0.npz')
            assert str(Z['genre']) == g, (g, n, str(Z['genre']))
        M = analyze16(f'{root}/{g}', models=have, boot=boot)['models'] if have else {}   # 凍結した A2-16 の分析（verdict16 の 7 モデル用の言葉は使わない）
        models = {n: (M[n] if n in M else _missing_model()) for n in names}
        res['genres'][g] = genre_words(models)
        res['records'][g] = {n: (model_record(models[n], load_nll(f'{root}/{g}/{n}@nll.npz', g)) if n in M else dict(missing=True)) for n in names}
        Rs = [res['records'][g][n].get('R', float('nan')) for n in names]
        res['records'][g]['_median_R'] = float(np.nanmedian(Rs)) if np.isfinite(np.array(Rs, float)).any() else float('nan')
        res['records'][g]['_full'] = M
    res['verdict'] = overall({g: res['genres'][g] for g in A17_GEN}) if list(genres) == A17_GEN else None
    # BBC（A2-16、すでに見た結果）を並べる
    if a16_summary and os.path.exists(a16_summary):
        B = json.load(open(a16_summary))['models']
        res['records']['bbc'] = {n: dict(R=B[n]['point']['R'], ci_R=B[n]['ci']['R'], slope_w=B[n]['point']['slope_w'], Tm_h0_w=B[n]['point']['Tm_h0_w'],
                                         n=B[n]['point']['n'], **pooled_ratios(B[n]['records']),
                                         human_nll_mean=(lambda v: float(np.nanmean(v)) if v is not None else float('nan'))(load_nll(f'{root}/bbc/{n}@nll.npz', 'bbc')))
                                 for n in names if n in B}
    return res


# ---------- 自己テスト ----------
def _write_synth(path, arms_a, h, genre, arm):
    toks, U, pick = arms_a['toks'], arms_a['U'], arms_a['pick']
    z = np.zeros(toks.shape, np.float32)
    np.savez_compressed(path, temps=np.array(G16_TEMPS), h=h, arm=arm, genre=genre, use=np.arange(toks.shape[2]),
                        toks=toks.astype(np.int32), cand=pick, pick=pick, U=U, q0=z, q=z, human_nll=np.full(toks.shape[2], 2.0))


def self_test17(tmp, boot=40):
    """作り物：論文 ＝ 閉じる、コード・小説 ＝ 場の効きが半分。種類ごとの言葉と、またいだ言葉が決まりどおりになるか。言葉の決まりも直接確かめる。"""
    # 言葉の決まり
    M = lambda w1, w3: {f'm{i}': dict(words=dict(HQ1=a, HQ3=b)) for i, (a, b) in enumerate(zip(w1, w3))}
    up, dis, un = '両方で、ものさしの見かけより大きく上がる', '食い違う', 'はっきりしない'
    sm, cl = '場の効きが予言より小さい', '閉じる（2 割の内）'
    assert genre_words(M([up, up, up, un], [sm, sm, sm, cl]))['GQ1'][0] == '見かけではない'
    assert genre_words(M([up, up, dis, dis], [sm, sm, cl, cl]))['GQ1'][0] == 'ものさしで変わる'
    assert genre_words(M([up, up, un, un], [sm, sm, cl, cl]))['GQ2'][0].startswith('モデルで')
    assert genre_words(M([up] * 4, [sm, sm, sm, un]))['GQ2'][0] == sm
    G = lambda a, b: dict(GQ1=(a, {}), GQ2=(b, {}))
    assert overall({g: G('見かけではない', sm) for g in A17_GEN}) == dict(GQ1='文章の種類を越えて、見かけではない', GQ2='文章の種類を越えて、場の効きが予言より小さい')
    assert overall(dict(arxiv=G('見かけではない', sm), code=G('はっきりしない', cl), fiction=G('見かけではない', sm)))['GQ2'] == '文章の種類で変わる'
    assert overall(dict(arxiv=G('見かけではない', sm), code=G('はっきりしない', 'モデルで分かれる・はっきりしない'), fiction=G('見かけではない', sm))) == dict(GQ1='はっきりしない', GQ2='はっきりしない')
    # 作り物で読み込みから判定まで（モデル 2 つ・種類 3 つで、決め方の数 A17_MAJ を 2 にして通す）
    global A17_MAJ
    keep = A17_MAJ
    try:
        A17_MAJ = 2
        names = ['ma', 'mb']
        for gi, (g, sc) in enumerate((('arxiv', 1.0), ('code', 0.5), ('fiction', 0.5))):
            os.makedirs(f'{tmp}/{g}', exist_ok=True)
            for mi, n in enumerate(names):
                arms = synth16(seed=10 * gi + mi + 1, dM_scale=sc, raw=True)
                for a, h in G16_H:
                    _write_synth(f'{tmp}/{g}/{n}@{a}.npz', arms[a], h, g, a)
        res = analyze17(tmp, a16_summary=None, names=names, boot=boot)
        # 1 つのモデルの腕が欠けても止まらず、そのモデルは「はっきりしない」になる
        os.rename(f'{tmp}/code/mb@hp03.npz', f'{tmp}/code/mb@hp03.npz.bak')
        res_m = analyze17(tmp, a16_summary=None, names=names, genres=['code'], boot=boot)
        os.rename(f'{tmp}/code/mb@hp03.npz.bak', f'{tmp}/code/mb@hp03.npz')
        assert res_m['missing']['code'] == ['mb'] and res_m['genres']['code']['GQ2'][1]['mb'] == 'はっきりしない', res_m['genres']
    finally:
        A17_MAJ = keep
    assert res['genres']['code']['GQ2'][0] == sm and res['genres']['fiction']['GQ2'][0] == sm, res['genres']
    assert res['genres']['arxiv']['GQ2'][0] in (cl, '1 より小さい向きだが、ものさしのゆがみと区別できない'), res['genres']['arxiv']
    assert res['verdict']['GQ2'] == '文章の種類で変わる', res['verdict']
    assert all(res['genres'][g]['GQ1'][0] == '見かけではない' for g in A17_GEN), res['genres']
    Rs = {g: [res['records'][g][n]['R'] for n in names] for g in A17_GEN}
    return dict(R=Rs, verdict=res['verdict'], genres={g: {k: v[0] for k, v in res['genres'][g].items()} for g in A17_GEN})
