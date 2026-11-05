"""英語だけの写し（paper/review_copy_v05.md）から REVTeX の原稿（paper/main.tex）を作る。
  python3 scripts/make_revtex.py paper/review_copy_v05.md paper/main.tex [--check]
--check：REVTeX がない環境で、article クラスに置きかえた写しを pdflatex で通して、文法の誤りがないかだけ確かめる。
式：段落だけの式は手で LaTeX に置きかえ（DISPLAY）、文中の記号は下付き・上付きの形を見つけて $…$ にする。
文献：[1904.09751; Glynn 1990] → \\cite{1904.09751,Glynn1990}（鍵は paper/refs.bib と同じ）。
"""
import re, sys, subprocess, os, shutil, tempfile

KEYS = {'Andor et al. 2016': 'AndorEtAl2016', 'Chetrite & Touchette 2015': 'ChetriteTouchette2015', 'Garrahan et al. 2009': 'GarrahanEtAl2009',
        'Glynn 1990': 'Glynn1990', 'Jack & Sollich 2010': 'JackSollich2010', 'Lafferty et al. 2001': 'LaffertyEtAl2001',
        'LatestEval, Li et al. 2024': 'LiEtAl2024', 'Warren & Allen 2012': 'WarrenAllen2012', 'Williams 1992': 'Williams1992',
        'Wiseman & Rush 2016': 'WisemanRush2016', 'Yang & Klein 2021': 'YangKlein2021', 'Zhao et al. 2024': 'ZhaoEtAl2024'}

DISPLAY = {
    "∂ log P_h(x)/∂h |_{h=0} = (N − Q)/T,  with N = Σ_t c_t and Q = Σ_t q_t.":
        r"\begin{equation}\left.\frac{\partial \log P_h(x)}{\partial h}\right|_{h=0}=\frac{N-Q}{T},\qquad N=\sum_t c_t,\quad Q=\sum_t q_t.\end{equation}",
    "∂E[O]/∂h = Cov(O, N)/T − Cov(O, Q)/T.":
        r"\begin{equation}\frac{\partial E[O]}{\partial h}=\frac{\mathrm{Cov}(O,N)}{T}-\frac{\mathrm{Cov}(O,Q)}{T}.\end{equation}",
    "Cov(O, Q) = Cov(O, Σ_t ν_t).":
        r"\begin{equation}\mathrm{Cov}(O,Q)=\mathrm{Cov}\Big(O,\sum_t \nu_t\Big).\end{equation}",
    "ln K = ln(n₀₁/M₀) − ln(n₁₀/M₁).":
        r"\begin{equation}\ln K=\ln\frac{n_{01}}{M_0}-\ln\frac{n_{10}}{M_1}.\end{equation}",
    "∂E[Y]/∂h at h = 0 equals Cov(Y, σ)/T, with σ = Σ_t (c_t − q_t),":
        r"\begin{equation}\left.\frac{\partial E[Y]}{\partial h}\right|_{h=0}=\frac{\mathrm{Cov}(Y,\sigma)}{T},\qquad \sigma=\sum_t (c_t-q_t),\end{equation}",
    "g(T) = ∂ln K/∂h = (d₀₁ − d₀) − (d₁₀ − d₁).":
        r"\begin{equation}g(T)=\frac{\partial \ln K}{\partial h}=(d_{01}-d_0)-(d_{10}-d_1).\end{equation}",
    "log w = Σ_t [ (h/T) c_t − log(1 − q_t + q_t e^{h/T}) ].":
        r"\begin{equation}\log w=\sum_t\Big[\frac{h}{T}c_t-\log\big(1-q_t+q_t e^{h/T}\big)\Big].\end{equation}",
    "∂ log P_β(x)/∂β = −(U − U_comp),  with U = Σ_t U_t(x_t) and U_comp = Σ_t E_t[U_t],":
        r"\begin{equation}\frac{\partial \log P_\beta(x)}{\partial\beta}=-\big(U-U_{\mathrm{comp}}\big),\qquad U=\sum_t U_t(x_t),\quad U_{\mathrm{comp}}=\sum_t E_t[U_t],\end{equation}",
    "g_C(T) = (b/ΔE) Δc_T/T,":
        r"\begin{equation}g_C(T)=\frac{b}{\Delta E}\,\frac{\Delta c_T}{T},\end{equation}",
    "g/g_C = (s_pen/s_temp) κ,  with κ = g_N/g_C^tilt.":
        r"\begin{equation}\frac{g}{g_C}=\frac{s_{\mathrm{pen}}}{s_{\mathrm{temp}}}\,\kappa,\qquad \kappa=\frac{g_N}{g_C^{\mathrm{tilt}}}.\end{equation}",
}

INLINE = {
    "x = (x_1, …, x_L)": r"$x=(x_1,\ldots,x_L)$",
    "P_h(x) = Π_t p_t^h(x_t | x_<t)": r"$P_h(x)=\prod_t p_t^h(x_t\mid x_{<t})$",
    "log p_t^h(x_t)": r"$\log p_t^h(x_t)$",
    "(c_t − q_t)/T": r"$(c_t-q_t)/T$",
    "exp(hN/T)": r"$\exp(hN/T)$",
    "exp(hc_t/T)": r"$\exp(hc_t/T)$",
    "E[exp(hN_{>t}/T) | x_≤t]": r"$E[\exp(hN_{>t}/T)\mid x_{\le t}]$",
    "N_{>t}": r"$N_{>t}$",
    "(h/T) Σ_t ν_t": r"$(h/T)\sum_t \nu_t$",
    "Cov(O, N)/T": r"$\mathrm{Cov}(O,N)/T$",
    "Cov(O, Q)/T": r"$\mathrm{Cov}(O,Q)/T$",
    "E[N | x_≤t] − E[N | x_<t]": r"$E[N\mid x_{\le t}]-E[N\mid x_{<t}]$",
    "N − E[N] = (N − Q) + Σ_t ν_t": r"$N-E[N]=(N-Q)+\sum_t \nu_t$",
    "Q − E[Q] = Σ_t ν_t": r"$Q-E[Q]=\sum_t \nu_t$",
    "N − E[N]": r"$N-E[N]$",
    "c_t − q_t plus ν_t": r"$c_t-q_t$ plus $\nu_t$",
    "exp(−εU)": r"$\exp(-\varepsilon U)$",
    "exp(z/T)": r"$\exp(z/T)$",
    "exp(−U(x)/T)/Z(T)": r"$\exp(-U(x)/T)/Z(T)$",
    "exp(−U(x))": r"$\exp(-U(x))$",
    "(N − Q)/T": r"$(N-Q)/T$",
    "softmax(z/T), where z are": r"$\mathrm{softmax}(z/T)$, where $z$ are",
    "softmax(z/T)": r"$\mathrm{softmax}(z/T)$",
    "where z are": r"where $z$ are",
    "−|z|": r"$-|z|$",
    "A_t = a(x_t) − Σ_v p_t(v) a(v) and a(v) = −|z_v|": r"$A_t=a(x_t)-\sum_v p_t(v)\,a(v)$ and $a(v)=-|z_v|$",
    "S = 4 samples with divisor S − 1": r"$S=4$ samples with divisor $S-1$",
    "With the same a, b and T*": r"With the same $a$, $b$ and $T^*$",
    "λ* = −(a + b/T*)/ḡ_λ and θ* = e^{λ*}": r"$\lambda^*=-(a+b/T^*)/\bar{g}_\lambda$ and $\theta^*=e^{\lambda^*}$",
    "θ_k = e^{kλ*}": r"$\theta_k=e^{k\lambda^*}$",
    "steps k to k + 31": r"steps $k$ to $k+31$",
    "step k + 16": r"step $k+16$",
    "k/32": r"$k/32$",
    "steps d since": r"steps $d$ since",
    "d ≥ 97": r"$d\ge 97$",
    "49 ≤ d ≤ 96": r"$49\le d\le 96$",
    "h[g(T) − ḡ]": r"$h[g(T)-\bar{g}]$",
    "φ_v = z_v/T − max_u z_u/T − ln r ≥ 0": r"$\varphi_v=z_v/T-\max_u z_u/T-\ln r\ge 0$",
    "threshold r, a token v is kept": r"threshold $r$, a token $v$ is kept",
    "min_v |φ_v|": r"$\min_v|\varphi_v|$",
    "units of m steps, with energy difference mΔE": r"units of $m$ steps, with energy difference $m\Delta E$",
    "b = ∂ ln K/∂β = mΔE": r"$b=\partial\ln K/\partial\beta=m\Delta E$",
    "b = mΔE": r"$b=m\Delta E$",
    "mΔE": r"$m\Delta E$",
    "mΔc": r"$m\Delta c$",
    "−log p_t": r"$-\log p_t$",
    "−log p": r"$-\log p$",
    "U_t by −z_t": r"$U_t$ by $-z_t$",
    "−z": r"$-z$",
    "exp(−U)": r"$\exp(-U)$",
    "d_i = Cov(e_i, σ)/(T E[e_i])": r"$d_i=\mathrm{Cov}(e_i,\sigma)/(T\,E[e_i])$",
    "Σ_t log p_t(x_t)": r"$\sum_t \log p_t(x_t)$",
    "∂ log p_t(x_t)/∂h": r"$\partial\log p_t(x_t)/\partial h$",
    "statistic Y": r"statistic $Y$",
    "continuation x": r"continuation $x$",
    "Here a is": r"Here $a$ is",
    "ln K = a + b/T": r"$\ln K=a+b/T$",
    "T_× = −b/a": r"$T_\times=-b/a$",
    "b > 0 and a < 0": r"$b>0$ and $a<0$",
    "a + b/T + h ḡ": r"$a+b/T+h\bar{g}$",
    "a + b/T": r"$a+b/T$",
    "n_u h ḡ": r"$n_u h\bar{g}$",
    "h ḡ": r"$h\bar{g}$",
    "temperatures u": r"temperatures $u$",
    "step-t": r"step-$t$",
    "dT_×/dh": r"$dT_\times/dh$",
    "logit z": r"logit $z$",
    "z/θ if z > 0 and zθ otherwise": r"$z/\theta$ if $z>0$ and $z\theta$ otherwise",
    "The fraction s_pen": r"The fraction $s_{\mathrm{pen}}$",
    "g(T) = ∂ ln K/∂h": r"$g(T)=\partial\ln K/\partial h$",
    "a_t − E_t[a]": r"$a_t-E_t[a]$",
    "statistic O": r"statistic $O$",
    "step t ": r"step $t$ ",
    "step t,": r"step $t$,",
    "s_temp = Σ_u g_C(T_u)/Σ_u g_C^tilt(T_u)": r"$s_{\mathrm{temp}}=\sum_u g_C(T_u)/\sum_u g_C^{\mathrm{tilt}}(T_u)$",
}

GREEK = {'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta', 'ε': r'\varepsilon', 'θ': r'\theta', 'κ': r'\kappa', 'λ': r'\lambda',
         'μ': r'\mu', 'ν': r'\nu', 'π': r'\pi', 'ρ': r'\rho', 'σ': r'\sigma', 'τ': r'\tau', 'φ': r'\varphi', 'Δ': r'\Delta', 'Σ': r'\Sigma'}
WORDSUB = {'pen': r'\mathrm{pen}', 'temp': r'\mathrm{temp}', 'tilt': r'\mathrm{tilt}', 'eq': r'\mathrm{eq}', 'out': r'\mathrm{out}',
           'meas': r'\mathrm{meas}', 'id': r'\mathrm{id}', 'C': 'C', 'N': 'N', 'Q': 'Q', 'U': 'U'}
SUBDIG = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def tex_sym(base):
    out = ''
    for ch in base:
        t = GREEK.get(ch, ch)
        if out and re.search(r'\\[a-zA-Z]+$', out) and ch.isalpha() and ch not in GREEK:
            out += ' '
        out += t
    return out


def inline_math(text):
    """X_y, X_y^z, T_×, h*, ḡ_λ などを $…$ にする（表と式の段落の外だけ）。"""
    def sub(m):
        base, sub_, sup = m.group(1), m.group(2), m.group(3)
        b = tex_sym(base).replace('ḡ', r'\bar{g}')
        s_ = sub_.strip('{}')
        s_ = r'\times' if s_ == '×' else WORDSUB.get(s_, tex_sym(s_))
        r = f'{b}_{{{s_}}}'
        if sup:
            sp = sup[1:].strip('{}')
            r += '^{' + (WORDSUB.get(sp, tex_sym(sp))) + '}'
        return f'${r}$'
    parts = re.split(r'(\$[^$]*\$)', text)
    if len(parts) > 1:
        return ''.join(p if i % 2 else inline_math(p) for i, p in enumerate(parts))
    text = re.sub(r'(?<![\w$\\])([A-Za-zα-ωΔΣḡ]{1,3})_(\{[^}]+\}|[A-Za-z0-9×α-ω]+)(\^(?:\{[^}]+\}|[A-Za-z]+))?(?![\w$])', sub, text)
    text = re.sub(r'(?<![A-Za-z$\\])([hTλθ])\*', lambda m: f'${tex_sym(m.group(1))}^*$', text)
    text = re.sub(r'(?<![\w$\\])k = (\d)', r'$k=\1$', text)
    text = re.sub(r'(?<![\w$\\])Δ([Ec])(?![\w_])', r'$\\Delta \1$', text)
    text = re.sub(r'([nMd])([₀₁]{1,2})', lambda m: f'${m.group(1)}_{{{m.group(2).translate(SUBDIG)}}}$', text)
    return single_letters(text)


def single_letters(text):
    """$…$ の外にある 1 文字の記号（T, K, N, Q, U, P, D, L, h, g, b, m）と ln K・ln θ を斜体の式にする。
    (b) のようなパネルの名前、1b のような番号、語の一部は触らない。"""
    parts = re.split(r'(\$[^$]*\$)', text)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'(?<![\w$\\])ln (K|θ)(?![\w])', lambda m: '$\\ln ' + (r'\theta' if m.group(1) == 'θ' else 'K') + '$', parts[i])
    parts = re.split(r'(\$[^$]*\$)', ''.join(parts))
    for i in range(0, len(parts), 2):
        t = parts[i]
        t = re.sub(r'(?<![\w$\\(\'’-])([TKNQUPDLRYhgbm])(?![\w)\'’-])', r'$\1$', t)
        t = re.sub(r'(?<=\()([TKNQUL])(?=[),])', r'$\1$', t)
        t = re.sub(r'(\b[IVX]+) \$([A-D])\$', r'\1 \2', t)
        t = re.sub(r'(\*\*)\$([A-H])\$\.', r'\1\2.', t)
        t = re.sub(r'(Appendix|Appendices|and) \$([A-H])\$(?=[ .,;:)]|$)', r'\1 \2', t)   # 節の名前 IV D は記号ではない
        parts[i] = t
    return ''.join(parts)


def hand_inline(text):
    """INLINE の決まった式の切れはしを、すでにある $…$ の外だけで置きかえる（長いものから）。"""
    for k in sorted(INLINE, key=len, reverse=True):
        parts = re.split(r'(\$[^$]*\$)', text)
        text = ''.join(p if i % 2 else p.replace(k, INLINE[k]) for i, p in enumerate(parts))
    return text


def table_row(line):
    """パイプ表の 1 行：区切りの行はそのまま、ほかはセルごとに式を $…$ にする。"""
    if re.fullmatch(r'\|[\s:|-]+\|?', line.strip()):
        return line
    cells = line.strip().strip('|').split('|')
    return '| ' + ' | '.join(inline_math(hand_inline(cites(c.strip()))) for c in cells) + ' |'


def _vis(cell):
    """セルの見た目のおおよその文字数（命令と波かっこを除く）。"""
    c = re.sub(r'\\(mathrm|textbf|emph)\{', '', cell)
    c = re.sub(r'\\[a-zA-Z]+', 'x', c)
    return len(re.sub(r'[{}$\\()]', '', c).strip())


def tables(tex):
    r"""pandoc の longtable を、REVTeX で通る table + tabular にする。
    array・calc の書き方（>{…} や \real）は使わない（Overleaf の REVTeX で止まったため）。
    文字が多い表は、列ごとの最長のセルから幅を決めた p 列にし、セルの頭に \tabrr（左寄せ）を置く。
    直前の「Table X | 題」の段落は \caption にする。"""
    lines = tex.split('\n'); out = []; i = 0
    while i < len(lines):
        if not lines[i].startswith('\\begin{longtable}'):
            out.append(lines[i]); i += 1; continue
        j = i
        while not lines[j].startswith('\\end{longtable}'):
            j += 1
        block = '\n'.join(lines[i:j])
        m = re.match(r'\\begin\{longtable\}\[\]\{(@\{\}.*?@\{\})\}\n', block, re.S)
        spec, rest = m.group(1), block[m.end():]
        assert '>{' not in spec and 'real' not in spec, ('pandoc が幅つきの列を出した', spec)
        rest = re.sub(r'\\endfirsthead.*?\\endhead', r'\\endhead', rest, flags=re.S)
        rest = rest.replace('\\noalign{}', '')
        rest = re.sub(r'\\endhead\s*\\bottomrule\s*\\endlastfoot\s*', '', rest)
        rows = [l for l in rest.split('\n') if l.rstrip().endswith('\\\\') and not l.startswith('\\')]
        cells = [[c.strip() for c in l.rstrip()[:-2].split(' & ')] for l in rows]
        n = max(len(r) for r in cells)
        mx = [max(_vis(r[k]) for r in cells if k < len(r)) for k in range(n)]
        if sum(mx) + 3 * n > 118:
            avail, w, free = 110.0, [None] * n, list(range(n))
            while True:
                share = avail / len(free)
                fixed = [k for k in free if mx[k] <= share]
                if not fixed:
                    break
                for k in fixed:
                    w[k] = mx[k]; avail -= mx[k]; free.remove(k)
                if not free:
                    break
            for k in free:
                w[k] = avail / len(free)
            tot = sum(w)
            spec = '@{}' + ''.join('p{%.3f\\dimexpr\\columnwidth-%d\\tabcolsep\\relax}' % (wk / tot, 2 * n) for wk in w) + '@{}'
            new_rows = {l: ' & '.join('\\tabrr ' + c for c in r) + ' \\\\' for l, r in zip(rows, cells)}
            rest = '\n'.join(new_rows.get(l, l) for l in rest.split('\n'))
        cap, label = '', ''
        k = len(out) - 1
        while k >= 0 and out[k].strip() == '':
            k -= 1
        mc = re.match(r'\\textbf\{Table (S?[A-Z]?\d*|[IVX]+) \\textbar\{\} ', out[k]) if k >= 0 else None
        if mc:
            label = mc.group(1)
            cap = '\\caption{\\textbf{' + out[k][mc.end():] + '}'
            del out[k:]
        roman = re.fullmatch(r'[IVX]+', label or 'x') is not None
        out += ['\\begin{table}[htbp]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{6pt}']
        if label and not roman:
            out.append('\\renewcommand{\\thetable}{' + label + '}')
        if cap:
            out += [cap, '\\label{tab' + label + '}']
        out += ['\\begin{tabular}{' + spec + '}', rest.rstrip(), '\\bottomrule', '\\end{tabular}', '\\end{table}', '']
        i = j + 1
    return '\n'.join(out)


def cites(text):
    def sub(m):
        parts = [p.strip() for p in m.group(1).split(';')]
        if not all(re.fullmatch(r'\d{4}\.\d{4,5}', p) or p in KEYS for p in parts):
            return m.group(0)
        return r'\cite{' + ','.join(KEYS.get(p, p) for p in parts) + '}'
    return re.sub(r'\[([^\]\[]+)\]', sub, text)


def figures(md):
    """![figN](figures/figN.png) と、その次の **Fig. N | …** の段落を figure にまとめる（生の LaTeX として pandoc に通す）。"""
    def sub(m):
        path, cap = m.group(1), m.group(2)
        cap = re.sub(r'^\*\*(Fig\. S?\d+) \| ([^*]+)\*\*\s*', r'\\textbf{\2} ', cap)
        cap = re.sub(r'\*\*([a-z])\*\*', r'(\1)', cap)
        cap = cap.replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')
        label = os.path.basename(path).split('.')[0]
        num = '\\renewcommand{\\thefigure}{' + label[3:] + '}\n' if label.startswith('figS') else ''   # 付録の図は S1, S2
        return ('```{=latex}\n\\begin{figure}\n' + num + '\\centering\n\\includegraphics[width=\\linewidth]{' + path.replace('.png', '') +
                '}\n\\caption{' + inline_math(hand_inline(cites(cap))) + '}\n\\label{' + label + '}\n\\end{figure}\n```\n')
    return re.sub(r'!\[[^\]]*\]\(([^)]+)\)\n\n(\*\*Fig\.[^\n]+)\n', sub, md)


def convert(src, dst):
    md = open(src).read()
    title = md.split('\n', 1)[0].lstrip('# ').strip()
    body = md.split('\n', 1)[1]
    ab_i = body.index('## Abstract'); ab_j = body.index('\n## ', ab_i + 5)
    abstract = body[ab_i + len('## Abstract'):ab_j].strip()
    body = body[ab_j:]
    for k, v in DISPLAY.items():
        assert k in body, ('式が見つからない', k)
        body = body.replace(k, '```{=latex}\n' + v + '\n```')
    body = figures(body)
    out = []
    for para in body.split('\n'):
        if para.startswith('|'):
            out.append(table_row(para)); continue
        if para.startswith('```') or para.startswith('\\') or para.startswith('#'):
            out.append(cites(para) if para.startswith('#') is False else hand_inline(para)); continue
        out.append(inline_math(hand_inline(cites(para))))
    body = '\n'.join(out)
    body = re.sub(r'^## ([IVX]+)\. ', '# ', body, flags=re.M)
    body = re.sub(r'^### ([A-D])\. ', '## ', body, flags=re.M)
    body = re.sub(r'^## Appendices\s*$', '\\\\appendix', body, flags=re.M)
    body = re.sub(r'^### Appendix ([A-H]): ', '# ', body, flags=re.M)
    body = re.sub(r'^#### ', '## ', body, flags=re.M)
    body = re.sub(r'^### Data and code availability', '# Data and code availability', body, flags=re.M)
    body = re.sub(r'^## Supplementary figures\s*\n', '', body, flags=re.M)          # 図は浮動体なので空の見出しになる
    body = re.sub(r'^## (F\d+\. .*)$', r'## \1 {-}', body, flags=re.M)               # 付録 F は自前の番号 F1… を使う
    body = re.sub(r'^# Data and code availability\s*$', '# Data and code availability {-}', body, flags=re.M)
    tmp = tempfile.mkdtemp()
    open(f'{tmp}/b.md', 'w').write(body); open(f'{tmp}/a.md', 'w').write(inline_math(hand_inline(cites(abstract))))
    pd = lambda f: subprocess.run(['pandoc', '-f', 'markdown+raw_attribute+tex_math_dollars-auto_identifiers', '-t', 'latex', '--wrap=none', '--columns=100000', f],
                                  check=True, capture_output=True, text=True).stdout
    btex, atex = pd(f'{tmp}/b.md'), pd(f'{tmp}/a.md')
    btex = btex.replace('\\appendix', '\n\\appendix\n')
    btex = tables(btex)
    pre = r"""\documentclass[aps,prx,onecolumn,superscriptaddress,longbibliography]{revtex4-2}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,booktabs,newunicodechar}
\newcommand{\tabrr}{\raggedright\let\\\tabularnewline}
\newunicodechar{−}{\ensuremath{-}}\newunicodechar{×}{\ensuremath{\times}}\newunicodechar{≤}{\ensuremath{\le}}
\newunicodechar{≥}{\ensuremath{\ge}}\newunicodechar{≈}{\ensuremath{\approx}}\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{∂}{\ensuremath{\partial}}\newunicodechar{Σ}{\ensuremath{\Sigma}}\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{κ}{\ensuremath{\kappa}}\newunicodechar{ρ}{\ensuremath{\rho}}\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{λ}{\ensuremath{\lambda}}\newunicodechar{ν}{\ensuremath{\nu}}\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{β}{\ensuremath{\beta}}\newunicodechar{ε}{\ensuremath{\varepsilon}}\newunicodechar{φ}{\ensuremath{\varphi}}
\newunicodechar{μ}{\ensuremath{\mu}}\newunicodechar{α}{\ensuremath{\alpha}}\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{ḡ}{\ensuremath{\bar{g}}}\newunicodechar{…}{\ldots}\newunicodechar{→}{\ensuremath{\to}}
\newunicodechar{Π}{\ensuremath{\Pi}}\newunicodechar{∞}{\ensuremath{\infty}}\newunicodechar{⁻}{\ensuremath{^{-}}}
\newunicodechar{⁹}{\ensuremath{^{9}}}\newunicodechar{¹}{\ensuremath{^{1}}}\newunicodechar{’}{'}\newunicodechar{–}{--}\newunicodechar{—}{---}\newunicodechar{′}{\ensuremath{'}}
\newunicodechar{²}{\ensuremath{^{2}}}\newunicodechar{₀}{\ensuremath{_{0}}}\newunicodechar{₁}{\ensuremath{_{1}}}
\providecommand{\texorpdfstring}[2]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\begin{document}
"""
    head = ('\\title{' + title + '}\n\\author{Chisa Honda}\n\\email{chichichi2001@gmail.com}\n\\affiliation{Independent researcher}\n'
            '\\begin{abstract}\n' + atex.strip() + '\n\\end{abstract}\n\\maketitle\n\n')
    tail = '\n\\bibliography{refs}\n\\end{document}\n'
    open(dst, 'w').write(pre + head + btex + tail)
    return dst


def check(tex):
    """REVTeX のない環境での文法の確かめ：article に置きかえて通す。"""
    d = tempfile.mkdtemp(); base = os.path.dirname(os.path.abspath(tex))
    s = open(tex).read()
    s = s.replace(r'\documentclass[aps,prx,onecolumn,superscriptaddress,longbibliography]{revtex4-2}', r'\documentclass[letterpaper]{article}\usepackage[numbers]{natbib}\usepackage[textwidth=6.75in,textheight=9.2in]{geometry}')
    for c in (r'\email{chichichi2001@gmail.com}', r'\affiliation{Independent researcher}'):
        s = s.replace(c, '')
    s = s.replace(r'\begin{abstract}', r'\maketitle\begin{abstract}').replace(r'\end{abstract}' + '\n' + r'\maketitle', r'\end{abstract}')
    s = s.replace(r'\bibliography{refs}', r'\bibliographystyle{unsrtnat}\bibliography{refs}')
    open(f'{d}/main.tex', 'w').write(s)
    shutil.copytree(f'{base}/figures', f'{d}/figures')
    if os.path.exists(f'{base}/refs.bib'):
        shutil.copy(f'{base}/refs.bib', d)
    r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'], cwd=d, capture_output=True, text=True)
    log = open(f'{d}/main.log', errors='ignore').read() if os.path.exists(f'{d}/main.log') else r.stdout
    errs = [l for l in log.split('\n') if l.startswith('!')]
    return r.returncode, errs[:10], d


if __name__ == '__main__':
    out = convert(sys.argv[1], sys.argv[2])
    if '--check' in sys.argv:
        rc, errs, d = check(out)
        print('latexmk', rc, errs, d)
