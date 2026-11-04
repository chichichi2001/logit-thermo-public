"""論文の図を描く（CPU）。PRX v0.2：図 1、図 2（関数 fig3、前もっての確かめ）、図 3（関数 fig4、ファイル fig3）、補足 S1・S2。
  python3 scripts/paper_figures.py <A2-19 のフォルダ> <A2-20 第 2 段のフォルダ> paper/figures
図 1 だけ生のデータ（罰なしの腕 c0 のトークン列）を使う。図 2〜4 は results/ の JSON だけから描く。
色：dataviz の既定の色（青 #2a78d6 ＝ 恒等式・測ったもの、橙 #eb6834 ＝ 平衡の読み）。形（丸・四角）でも区別する。
"""
import sys, os, json
import numpy as np
import matplotlib, matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'scripts')
exec(open('scripts/a2_23_run.py').read().split("\nif __name__")[0])

BLUE, ORANGE, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#9a9a94', '#1f1f1d', '#6b6b66'
BAND = (0.8, 1.25)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300})
SMALL = ['pythia-410m', 'pythia-1.4b', 'gpt-neo-1.3b', 'smollm2-1.7b', 'qwen2.5-1.5b', 'gpt2-1.5b', 'bloom-1b1']
BIG = ['qwen2.5-7b', 'mistral-7b-v0.3']
NAME = {'pythia-410m': 'Pythia-410M', 'pythia-1.4b': 'Pythia-1.4B', 'gpt-neo-1.3b': 'GPT-Neo-1.3B', 'smollm2-1.7b': 'SmolLM2-1.7B',
        'qwen2.5-1.5b': 'Qwen2.5-1.5B', 'gpt2-1.5b': 'GPT-2-XL', 'bloom-1b1': 'BLOOM-1.1B', 'qwen2.5-7b': 'Qwen2.5-7B',
        'mistral-7b-v0.3': 'Mistral-7B-v0.3', 'qwen2.5-1.5b-it': 'Qwen2.5-1.5B-Instruct'}
MM = 1 / 25.4
J = lambda p: json.load(open(p))


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)


def fig1(f19, f20, out):
    """(a) 1 本の文の写しの印と状態（A2-24 の R3）。(b) 融点に近い温度 1 つでの 32 歩の窓の写しの割合の分布（A2-24 の XQ15 の記録）。(c) ln K 対 1/T と K ＝ 1。"""
    ns = load23ns()
    H = J('results/A2_18/hstar.json'); H.update(J('results/A2_20/hstar7b.json')['models'])
    A24 = J('results/A2_24/a2_24_summary.json')
    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.42)
    a, b, c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    ex = A24['R3']['example']
    mk = np.array(ex['marks']); st = np.array(ex['states']); t = np.array(ex['steps'])
    a.vlines(np.flatnonzero(mk), 0, 0.35, color=GRAY, lw=0.4)
    a.step(t, 0.55 + 0.35 * st, where='mid', color=BLUE, lw=1.4)
    a.set_yticks([0.175, 0.55, 0.9]); a.set_yticklabels(['repeat marks', 'ordinary', 'repetitive'])
    a.tick_params(axis='y', length=0); a.set_ylim(-0.05, 1.0); a.set_xlim(0, 512)
    a.set_xlabel('Generated token position')
    a.spines['left'].set_visible(False)
    mid = np.arange(33) / 32
    for m in SMALL + BIG:
        h = np.array(A24['XQ15']['records'][m]['hist'])
        b.plot(mid, h, color=BLUE if m == 'pythia-1.4b' else GRAY, lw=1.6 if m == 'pythia-1.4b' else 0.8,
               zorder=3 if m == 'pythia-1.4b' else 2, label=NAME[m] if m == 'pythia-1.4b' else None)
    b.set_ylim(0, None); b.set_xlabel('Fraction of repeats\nin a 32-token window'); b.set_ylabel('Fraction of windows')
    b.axvline(0.5, color=MUTED, lw=0.5, ls=':'); b.axvline(0.25, color=MUTED, lw=0.5, ls=':')
    b.legend(loc='upper center', fontsize=6, handlelength=1.5)
    for m in SMALL + BIG:
        grid = (H[m]['targets']['-0.10'] if 'targets' in H[m] else H[m]['target'])['grid']
        Z = load_groups23(f19 if m in SMALL else f20, m, 'c0', grid, 0.0)
        ev, _ = ns['seq_events'](Z['toks'], Z['pick'], Z['q'])
        lk, *_ = ns['lnK_from'](ev, np.ones(ev.shape[2]))
        x = 1 / np.array(grid)
        col = BLUE if m == 'pythia-1.4b' else GRAY
        c.plot(x, lk, '-', color=col, lw=1.6 if m == 'pythia-1.4b' else 0.8, zorder=3 if m == 'pythia-1.4b' else 2)
        c.plot(x, lk, 'o', ms=2.2, color=col, zorder=3)
    c.axhline(0, color=MUTED, lw=0.6, ls='--')
    c.set_xlabel('1 / T (inverse\nsampling temperature)'); c.set_ylabel('ln K  (entry rate / exit rate)')
    c.text(0.97, 0.06, 'K = 1: crossover\ntemperature T$_\\times$', transform=c.transAxes, ha='right', color=MUTED)
    for ax, l in ((a, 'a'), (b, 'b'), (c, 'c')):
        ax.text(-0.3, 1.04, l, transform=ax.transAxes, fontweight='bold', fontsize=9, color=INK)
    save(fig, out, 'fig1')


def figS2(out):
    """（PRX v0.2 で補足の図 S2 へ。下調べの 40 記事）(a) 測った融点の傾き 対 平衡（クラペイロン）の予言。(b) 温度ごとの ln K の応答：測った ÷ 予言（恒等式・平衡）。融点対 h は補足の図 S1 へ。"""
    S = J('results/A2_16/a2_16_summary.json')['models']
    fig, (b, c) = plt.subplots(1, 2, figsize=(130 * MM, 58 * MM), gridspec_kw=dict(wspace=0.45))
    lim = (0.15, 0.55)
    xs = [S[m]['point']['pred'] for m in SMALL]; ys = [S[m]['point']['slope_w'] for m in SMALL]
    b.plot(lim, lim, color=MUTED, lw=0.6, ls='--'); b.text(0.5, 0.52, 'y = x', color=MUTED, ha='right')
    b.plot(xs, ys, 'o', ms=4, color=INK, mec='white', mew=0.6)
    b.set_xlim(*lim); b.set_ylim(*lim); b.set_aspect('equal')
    b.set_xlabel('Equilibrium prediction, dT$_\\times$/dh'); b.set_ylabel('Measured dT$_\\times$/dh')
    ri, re = [], []
    for m in SMALL:
        for r in S[m]['records']['by_T']:
            ri.append(r['dlnK_meas'] / r['dlnK_identity']); re.append(r['dlnK_meas'] / r['dlnK_M_thermo'])
    rng = np.random.default_rng(0)
    for i, (v, col, mk) in enumerate(((ri, BLUE, 'o'), (re, ORANGE, 's'))):
        c.plot(i + rng.uniform(-0.18, 0.18, len(v)), v, mk, ms=2.6, color=col, mec='white', mew=0.4, alpha=0.9)
        c.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.2)
    c.axhline(1, color=MUTED, lw=0.6, ls='--')
    c.set_xticks([0, 1]); c.set_xticklabels(['Score-function\nidentity', 'Equilibrium\ntwo-state model'])
    c.set_xlim(-0.6, 1.6); c.set_ylabel('Measured ÷ predicted response of ln K')
    for ax, l in ((b, 'a'), (c, 'b')):
        ax.text(-0.38, 1.08, l, transform=ax.transAxes, fontweight='bold', fontsize=9, color=INK)
    save(fig, out, 'figS2')
    return dict(median_identity=float(np.median(ri)), median_equilibrium=float(np.median(re)), n=len(ri))


def figS1(out):
    """補足の図 S1：罰 h ＝ −0.3・0・＋0.3 での融点（小さい 7 本、記事 40 本の区間）。"""
    S = J('results/A2_16/a2_16_summary.json')['models']
    fig, a = plt.subplots(figsize=(80 * MM, 60 * MM))
    for m in SMALL:
        p, ci = S[m]['point'], S[m]['ci']
        ks = ['Tm_hm03_w', 'Tm_h0_w', 'Tm_hp03_w']
        e = np.array([[p[k] - ci[k][0], ci[k][1] - p[k]] for k in ks]).T
        a.errorbar([-0.3, 0.0, 0.3], [p[k] for k in ks], yerr=e, color=GRAY, lw=0.8, marker='o', ms=2.5, capsize=0, elinewidth=0.6)
    # ラベルの重なりをほどく（v0.5：読み手が「重なって読めない」）。端の y を並べ、最小の間隔を空けて、細い線で点とつなぐ。
    ends = sorted(((S[m]['point']['Tm_hp03_w'], m) for m in SMALL))
    lo, hi = a.get_ylim(); gap = (hi - lo) * 0.065
    ys = []
    for y, m in ends:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for (y0, m), y1 in zip(ends, ys):
        a.plot([0.305, 0.34], [y0, y1], color=MUTED, lw=0.4)
        a.text(0.35, y1, NAME[m], fontsize=5, color=MUTED, va='center')
    a.set_xticks([-0.3, 0, 0.3]); a.set_xlim(-0.4, 0.68); a.set_xlabel('Penalty h on repeat candidates'); a.set_ylabel('Crossover temperature T$_\\times$')
    save(fig, out, 'figS1')


def fig3(out, with_eq=False):
    """前もっての確かめ：モデルごとの比と区間。帯 0.8〜1.25。平衡の読みに対する比（ρ_E、A2-26 の全文、同じ罰の腕、区間つき）は橙の四角。PRX v0.2 で図 2 に（以前は 比 ÷ r_C）。v0.5：橙なしを図 2、橙ありを図 S3（with_eq=True）。"""
    A18 = J('results/A2_18/a2_18_summary.json'); A19 = J('results/A2_19/a2_19_summary.json'); A20 = J('results/A2_20/a2_20_summary.json')
    A23 = J('results/A2_23/a2_23_summary.json')
    H = J('results/A2_18/hstar.json'); H.update(J('results/A2_20/hstar7b.json')['models'])
    RE = J('results/A2_26/a2_26_summary.json')['full']['models']
    rc = {m: (RE[m]['point'], RE[m]['ci']) for m in SMALL + BIG}
    groups = [('First test, additive (40 articles)', [], [('Median, h*', A18['verdict']['h1']['median'], A18['verdict']['h1']['ci']),
                                                        ('Median, 2h*', A18['verdict']['h2']['median'], A18['verdict']['h2']['ci'])]),
              ('Replication, additive (120 new articles)', [(m, A19['models'][m]['rho'], A19['models'][m]['ci'], rc[m]) for m in SMALL],
               [('Median', A19['XQ1']['median'], A19['XQ1']['ci'])]),
              ('7B models, additive (120 articles)', [(m, A20['models'][m]['rho'], A20['models'][m]['ci'], rc[m]) for m in BIG],
               [('Mean', A20['XQ7']['mean'], A20['XQ7']['ci'])]),
              ('Multiplicative penalty (120 articles)', [(m, p['rho'], p['ci'], None) for m, p in A23['A']['models'].items()],
               [('Median', A23['A']['point'], A23['A']['ci'])])]
    nrow = sum(len(g[1]) + len(g[2]) + 1 for g in groups)
    fig, ax = plt.subplots(figsize=(120 * MM, nrow * 3.3 * MM + 12 * MM))
    y = 0; ticks, labels = [], []
    for title, rows, agg in groups:
        ax.text(0.205, y, title, fontweight='bold', color=INK, va='center', fontsize=6.5)
        y += 1
        for m, r, ci, req in rows:
            ax.plot(ci, [y, y], color=BLUE, lw=0.9); ax.plot(r, y, 'o', ms=3.2, color=BLUE, mec='white', mew=0.5, zorder=3)
            if with_eq and req is not None:
                ax.plot(req[1], [y + 0.28, y + 0.28], color=ORANGE, lw=0.9)
                ax.plot(req[0], y + 0.28, 's', ms=3.0, color=ORANGE, mec='white', mew=0.5, zorder=3)
            ticks.append(y); labels.append(NAME[m]); y += 1
        for lab, pt, ci in agg:
            ax.plot(ci, [y, y], color=INK, lw=1.4); ax.plot(pt, y, 'D', ms=4, color=INK, zorder=3)
            ticks.append(y); labels.append(lab); y += 1
    ax.axvspan(*BAND, color='#e9e9e4', zorder=0); ax.axvline(1, color=MUTED, lw=0.6, ls='--')
    ax.set_xscale('log'); ax.set_xlim(0.2, 3.0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([0.25, 0.5, 0.8, 1, 1.25, 2]); ax.set_xticklabels(['0.25', '0.5', '0.8', '1', '1.25', '2'])
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6); ax.set_ylim(y - 0.4, -0.8)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Realized ÷ predicted change in ln K (log scale)')
    ax.set_xlabel('Realized ÷ predicted change in ln K (log scale)', labelpad=2)
    if with_eq:
        h1, = ax.plot([], [], 'o', color=BLUE, ms=3.2); h2, = ax.plot([], [], 's', color=ORANGE, ms=3.0)
        ax.legend([h1, h2], ['vs first-order prediction', 'vs equilibrium two-state model'], loc='upper center', bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=6)
        save(fig, out, 'figS3')
    else:
        save(fig, out, 'fig2')



def fig4(f19, f20, out):
    """(a) 候補のある歩を q で 3 つに分けて、歩の割合・揺れ Σ q(1 − q) の割合・恒等式の効きの割合（A2-24 の XQ14、使う温度）。
    揺れの割合は 3 つとも、この図のために c0 の q から出す（A2-24 は迷っている歩の分だけ記録した。値はそれと一致を確かめる）。
    (b) ln K の変化のうち入る速さから来る割合：測った（区間）と恒等式の（A2-22）。"""
    A24 = J('results/A2_24/a2_24_summary.json')['XQ14']['records']
    S22 = J('results/A2_22/a2_22_summary.json'); M = {**S22['small']['models'], **S22['big']['models']}
    H = J('results/A2_18/hstar.json'); H.update(J('results/A2_20/hstar7b.json')['models'])
    S19 = J('results/A2_19/a2_19_summary.json')['models']; S19.update(J('results/A2_20/a2_20_summary.json')['models'])
    order = SMALL + BIG
    bins = [('sure_no', 'q < 0.1'), ('unsure', '0.1 ≤ q ≤ 0.9'), ('sure_copy', 'q > 0.9')]
    V = {}
    for m in order:
        grid = (H[m]['targets']['-0.10'] if 'targets' in H[m] else H[m]['target'])['grid']
        q = load_groups23(f19 if m in SMALL else f20, m, 'c0', grid, 0.0)['q'].astype(float)[np.array(S19[m]['use'], int)]
        v = q * (1 - q); tot = v.sum()
        V[m] = {'sure_no': float(v[q < 0.1].sum() / tot), 'unsure': float(v[(q >= 0.1) & (q <= 0.9)].sum() / tot), 'sure_copy': float(v[q > 0.9].sum() / tot)}
        assert abs(V[m]['unsure'] - A24[m]['var_share_unsure']) < 1e-6, m
    fig, (a, b) = plt.subplots(1, 2, figsize=(170 * MM, 62 * MM), gridspec_kw=dict(wspace=0.38, width_ratios=[1.2, 1]))
    rng = np.random.default_rng(1)
    metrics = [('step_frac_with_candidates', 'steps', GRAY, 'o'), ('var', 'variance q(1−q)', BLUE, 's'), ('share', 'predicted response', INK, 'D')]
    for i, (key, lab) in enumerate(bins):
        for jm, (mk_, mlab, col, mrk) in enumerate(metrics):
            vals = [V[m][key] if mk_ == 'var' else A24[m][mk_][key] for m in order]
            x = i + (jm - 1) * 0.25
            a.plot(x + rng.uniform(-0.05, 0.05, len(vals)), vals, mrk, ms=3, color=col, mfc=(col if mrk != 'D' else 'white'), mew=0.7, alpha=0.9,
                   label=mlab if i == 0 else None)
    a.set_xticks(range(3)); a.set_xticklabels([l for _, l in bins]); a.set_ylim(-0.05, 1.0)
    a.set_ylabel('Share among steps with repeat candidates'); a.axhline(0, color=MUTED, lw=0.5)
    a.legend(loc='upper left', fontsize=6)
    yy = np.arange(len(order))
    for m, y0 in zip(order, yy):
        ci = M[m]['f_entry_ci']
        b.plot(ci, [y0, y0], color=BLUE, lw=0.9); b.plot(M[m]['f_entry'], y0, 'o', ms=3.2, color=BLUE, mec='white', mew=0.5, zorder=3)
        b.plot(M[m]['f_entry_identity'], y0, 'D', ms=3.0, color=INK, mfc='white', mew=0.8, zorder=3)
    b.set_yticks(yy); b.set_yticklabels([NAME[m] for m in order], fontsize=6); b.invert_yaxis(); b.tick_params(axis='y', length=0)
    b.axvline(0.5, color=MUTED, lw=0.6, ls='--'); b.set_xlim(0.2, 0.8)
    b.set_xlabel('Share of the change in ln K\nfrom the entry rate')
    h1, = b.plot([], [], 'o', color=BLUE, ms=3.2); h2, = b.plot([], [], 'D', color=INK, mfc='white', ms=3.0)
    b.legend([h1, h2], ['measured (95% interval)', 'identity'], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6)
    for ax, l, dx in ((a, 'a', -0.24), (b, 'b', -0.45)):
        ax.text(dx, 1.03, l, transform=ax.transAxes, fontweight='bold', fontsize=9, color=INK)
    save(fig, out, 'fig3')
    return V


if __name__ == '__main__':
    f19, f20, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    print('figS2', figS2(out)); figS1(out); fig3(out); fig3(out, with_eq=True); print('fig4 var shares', fig4(f19, f20, out)); fig1(f19, f20, out)
