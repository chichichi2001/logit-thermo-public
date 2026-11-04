"""A2-4 の分析（a2_4_analyze.py）を作り物のデータで確かめる。T_0 の動きを決めて作り、判定が決めたとおりに分かれるか。

    python3 scripts/a2_4_synthetic_check.py
"""
import math, os, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ns = {}
exec(open(f'{ROOT}/scripts/a2_2_analyze.py').read().replace("if __name__ == '__main__':", "if False:"), ns)
exec(open(f'{ROOT}/scripts/a2_4_analyze.py').read(), ns)
TEMPS = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5]
nT, S, nD, L, P = len(TEMPS), 2, 40, 512, 64
BASE = {'pythia-1b': 0.852, 'olmo2-1b': 0.801, 'qwen2.5-1.5b': 0.910}


def make(d4, d2, shift_block, shift_placebo, seed):
    rng = np.random.default_rng(seed)
    lt = np.log(np.array(TEMPS))
    for name, T0 in BASE.items():
        use = np.arange(nD)
        prompts = rng.integers(1000, 60000, (nD, P)).astype(np.int32)
        humans = rng.integers(1000, 60000, (nD, L)).astype(np.int32)
        tf = rng.normal(-2, 0.3, (4, nD, nT, L)).astype(np.float32)
        doc = rng.normal(0, 0.6, nD)                                   # 記事ごとの揺れ（つまみで共通）
        slope = 4.0

        def gen_from(T0x, extra_sd):
            f = slope * (lt[:, None] - math.log(T0x)) + doc[None, :] + rng.normal(0, extra_sd, (nT, nD))   # (nT, nD)
            g = tf[0].transpose(1, 0, 2)[:, None, :, :] + f[:, None, :, None]                              # (nT,1,nD,L)
            g = np.repeat(g, S, 1) + rng.normal(0, 0.05, (nT, S, nD, L))
            return g.astype(np.float32)
        out = {}
        for mode, T0x in [('none', T0), ('block', T0 * math.exp(shift_block)), ('placebo', T0 * math.exp(shift_placebo))]:
            g = gen_from(T0x, 0.05)
            toks = rng.integers(1000, 60000, (nT, S, nD, L)).astype(np.int32)
            cand = rng.random((nT, S, nD, L)) < (0.0 if mode == 'block' else 0.1)
            cand_b = rng.random((nT, S, nD, L)) < 0.1 if mode == 'block' else cand
            banned = cand_b if mode == 'block' else (rng.random((nT, S, nD, L)) < 0.03 if mode == 'placebo' else np.zeros_like(cand))
            bt1 = banned & (rng.random((nT, S, nD, L)) < (0.3 if mode == 'block' else 1.0))
            extra = {}
            if mode == 'placebo':
                extra['p_placebo'] = out['block']['g_banned_top1'].reshape(nT, -1).mean(1)
                bt1 = rng.random((nT, S, nD, L)) < extra['p_placebo'][:, None, None, None]
                banned = bt1
            ell_pen = g.copy(); ell_pen[banned] -= 0.5
            arr = dict(tf=tf if mode == 'none' else np.zeros_like(tf), toks=toks, temps=np.array(TEMPS), use=use,
                       prompts=prompts, humans=humans, mode=mode,
                       g_ell=g, g_top1=g, g_ent=g, g_ell2=g, g_ell_pen=ell_pen,
                       g_cand_mass=np.where(cand_b, 0.2, 0).astype(np.float32), g_ban_mass=np.where(banned, 0.2, 0).astype(np.float32),
                       g_cand=cand_b, g_banned=banned, g_banned_top1=bt1, **extra)
            out[mode] = arr
            np.savez_compressed(f'{d4}/{name}@{mode}.npz', **arr)
        # A2-2（同じ T_0、同じ人の文脈）
        g2 = gen_from(T0, 0.05)
        np.savez_compressed(f'{d2}/{name}.npz', tf=tf, gen=np.stack([g2] * 4), use=use)


def run(sb, sp, seed):
    with tempfile.TemporaryDirectory() as d4, tempfile.TemporaryDirectory() as d2:
        make(d4, d2, sb, sp, seed)
        return ns['analyze4'](d4, d2)


cases = [
    ('結晶が支えている（禁止 −15%、偽薬 −2%）', -0.16, -0.02, '4 トークンの完全な写しは T_0 を支えている', '大きく下がる', '写しを止めた分がある'),
    ('動かない（0%、0%）', 0.0, 0.0, 'T_0 は 4 トークンの完全な写しでは決まっていない', '10% 以上は動かない', '偽薬と区別できない'),
    ('偽薬も同じだけ下げる（−15%、−15%）', -0.16, -0.16, 'はっきりしない', '大きく下がる', '偽薬と区別できない'),
    ('少し下がる（−5%、0%）', -0.05, 0.0, None, '少し下がる', None),
]
for label, sb, sp, want_qb1, want_inf, want_diff in cases:
    r = run(sb, sp, 7)
    inf = set(r['Q1_3'].values()); dif = {m['verdict_diff'] for m in r['models'].values()}
    print(label, '→', r['QB1'], '|', r['Q1_3'], '|', dif, '| 確かめ 1', r['check1_all_ok'],
          '| d_inf の点', [round(m['d_inf']['point'], 3) for m in r['models'].values()],
          '| 区間', [[round(x, 3) for x in m['d_inf']['ci']] for m in r['models'].values()])
    assert r['check1_all_ok']
    if want_qb1:
        assert r['QB1'] == want_qb1, (label, r['QB1'])
    assert inf == {want_inf}, (label, inf)
    if want_diff:
        assert dif == {want_diff}, (label, dif)
    for n, c in r['checks'].items():
        assert c['check3_block_seqs_with_copy'] == 0 and c['check4_ok'] and c['check2_ok'], (label, n, c)
print('OK')
