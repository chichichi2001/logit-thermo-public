"""探索：メイトの手の確率がほぼ 0 だったのは「#」という記号のせいか、手そのもののせいか（2026-10-30 11:1x）。予想なし。GPU（L4）。
scripts/a2_explore_chess.py（4d5c4b4a…）を実行したのと同じ名前空間で実行する（score_moves・load_any・CH_MODELS・CH_THEMES・spearman_np を使う）。

オーナー（2026-10-30 11:09）：（オーナーの承認・発言。原文は非公開の記録にある）
やり方：A2_explore/chess_positions.json の 250 局面で、各合法手の SAN から「+」と「#」を外した文字列の log 確率を取り直す。
外した文字列は、そのあとに + や # が付く場合も含めた「その手の確率」になる（手の SAN は、+ と # を外しても、ほかの合法手の
SAN の頭にはならない。ただし O-O と O-O-O は前と同じく前後の関係）。もとの（記号つきの）log 確率は chess_scores_*.json から読む。
"""
import json
import numpy as np


def strip_san(s):
    return s.rstrip('+#')


def strip_summary(positions, orig, strip):
    """orig・strip：{モデル: [局面ごとの log 確率の並び]} → テーマごとの比べ。"""
    out = {}
    for m in strip:
        row = {}
        for th in CH_THEMES + ['all']:
            idx = [i for i, pz in enumerate(positions) if th == 'all' or pz['theme'] == th]
            r = {}
            for tag, S in (('orig', orig[m]), ('strip', strip[m])):
                pb, t1, rk, rho, mass = [], [], [], [], []
                for i in idx:
                    lp = np.asarray(S[i], float); E = np.asarray(positions[i]['E'], float)
                    p = np.exp(lp - lp.max()); p /= p.sum()
                    best = E <= 0.01
                    pb.append(float(p[best].sum())); t1.append(float(best[int(np.argmax(p))]))
                    order = np.argsort(-p)
                    rk.append(int(np.flatnonzero(best[order])[0]) + 1)             # 最善手（いちばん良いもの）が何番目か
                    rho.append(spearman_np(-lp, E)); mass.append(float(np.exp(lp).sum()))
                r[tag] = dict(p_on_best=float(np.mean(pb)), top1_best=float(np.mean(t1)), best_rank_median=float(np.median(rk)),
                              rho_median=float(np.nanmedian(rho)), legal_mass_mean=float(np.mean(mass)))
            # 記号の付いた手（+ か #）にどれだけ確率が乗っているか（もと → 外した後）
            chk_o, chk_s = [], []
            for i in idx:
                sans = positions[i]['moves_san']
                mark = np.array([s.endswith('+') or s.endswith('#') for s in sans])
                if mark.any() and (~mark).any():
                    for S, acc in ((orig[m], chk_o), (strip[m], chk_s)):
                        lp = np.asarray(S[i], float); p = np.exp(lp - lp.max()); p /= p.sum()
                        acc.append(float(p[mark].sum() / mark.mean()))                 # 一様なら 1
            r['check_ratio'] = dict(orig=float(np.mean(chk_o)) if chk_o else None, strip=float(np.mean(chk_s)) if chk_s else None, n=len(chk_o))
            row[th] = r
        out[m] = row
    return out


def self_test_strip():
    assert strip_san('Qxf7#') == 'Qxf7' and strip_san('Nf3+') == 'Nf3' and strip_san('O-O') == 'O-O' and strip_san('e8=Q+') == 'e8=Q'
    pos = [dict(theme='mateIn1', E=[0.0, 0.5, 0.5], moves_san=['Qxf7#', 'Nf3', 'd4'])]
    orig = {'m': [[-20.0, -1.0, -1.0]]}; strip = {'m': [[-0.5, -1.0, -1.0]]}
    s = strip_summary(pos, orig, strip)['m']['mateIn1']
    assert s['orig']['top1_best'] == 0 and s['strip']['top1_best'] == 1 and s['strip']['best_rank_median'] == 1, s
    assert s['check_ratio']['n'] == 1 and s['check_ratio']['orig'] < 0.01 and s['check_ratio']['strip'] > 1, s
    return 'OK'
