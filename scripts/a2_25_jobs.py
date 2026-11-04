"""A2-25 の仕事の表（Notebook にもこの文字列をそのまま埋め込む）。A2-23 の予言の短い JSON（θ* と格子）から作る。"""
import json, math

A25_K = (2, 4, 8)
A25_THETA_MAX = 3.0


def a25_theta(theta_star, k):
    """λ* ＝ ln(θ* を 4 桁に丸めたもの)、θ_k ＝ e^{k λ*} を 4 桁に丸める。"""
    return round(math.exp(k * math.log(round(float(theta_star), 4))), 4)


def a25_jobs(pred_compact_json):
    """→ [('gen_mult', 名前, 'm{k}', θ_k, 格子, 組)]。θ_k が 3 を超える組は入れない。倍率の小さい順・モデル・組の順。"""
    pm = json.loads(pred_compact_json)['models']
    jobs = []
    for k in A25_K:
        for n in sorted(pm):
            r = pm[n]
            if not r.get('ok'):
                continue
            th = a25_theta(r['theta'], k)
            if th > A25_THETA_MAX:
                continue
            for g in range(3):
                jobs.append(('gen_mult', n, f'm{k}', th, [float(t) for t in r['grid']], g))
    return jobs
