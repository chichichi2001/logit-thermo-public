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
    "g_C(T) = (b/ΔE) Δc_T/T,":
        r"\begin{equation}g_C(T)=\frac{b}{\Delta E}\,\frac{\Delta c_T}{T},\end{equation}",
    "g/g_C = (s_pen/s_temp) κ,  with κ = g_N/g_C^tilt.":
        r"\begin{equation}\frac{g}{g_C}=\frac{s_{\mathrm{pen}}}{s_{\mathrm{temp}}}\,\kappa,\qquad \kappa=\frac{g_N}{g_C^{\mathrm{tilt}}}.\end{equation}",
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
            sp = sup.strip('{}')
            r += '^{' + (WORDSUB.get(sp, tex_sym(sp))) + '}'
        return f'${r}$'
    text = re.sub(r'(?<![\w$\\])([A-Za-zα-ωΔΣḡ]{1,3})_(\{[^}]+\}|[A-Za-z0-9×α-ω]+)(\^(?:\{[^}]+\}|[A-Za-z]+))?(?![\w$])', sub, text)
    text = re.sub(r'(?<![A-Za-z$\\])([hTλθ])\*', lambda m: f'${tex_sym(m.group(1))}^*$', text)
    text = text.translate(SUBDIG) if False else text
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
        t = re.sub(r'(?<![\w$\\(\'’-])([TKNQUPDLhgbm])(?![\w)\'’-])', r'$\1$', t)
        parts[i] = t
    return ''.join(parts)


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
        return ('```{=latex}\n\\begin{figure}\n\\centering\n\\includegraphics[width=\\linewidth]{' + path.replace('.png', '') +
                '}\n\\caption{' + inline_math(cites(cap)) + '}\n\\label{' + label + '}\n\\end{figure}\n```\n')
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
        if para.startswith('|') or para.startswith('```') or para.startswith('\\') or para.startswith('#'):
            out.append(cites(para) if para.startswith('#') is False else para); continue
        out.append(inline_math(cites(para)))
    body = '\n'.join(out)
    body = re.sub(r'^## ([IVX]+)\. ', '# ', body, flags=re.M)
    body = re.sub(r'^### ([A-D])\. ', '## ', body, flags=re.M)
    body = re.sub(r'^## Appendices\s*$', '\\\\appendix', body, flags=re.M)
    body = re.sub(r'^### Appendix ([A-H]): ', '# ', body, flags=re.M)
    body = re.sub(r'^#### ', '## ', body, flags=re.M)
    body = re.sub(r'^### Data and code availability', '# Data and code availability', body, flags=re.M)
    tmp = tempfile.mkdtemp()
    open(f'{tmp}/b.md', 'w').write(body); open(f'{tmp}/a.md', 'w').write(inline_math(cites(abstract)))
    pd = lambda f: subprocess.run(['pandoc', '-f', 'markdown+raw_attribute+tex_math_dollars-auto_identifiers', '-t', 'latex', '--wrap=none', f],
                                  check=True, capture_output=True, text=True).stdout
    btex, atex = pd(f'{tmp}/b.md'), pd(f'{tmp}/a.md')
    btex = btex.replace('\\appendix', '\n\\appendix\n')
    pre = r"""\documentclass[aps,prx,onecolumn,superscriptaddress,longbibliography]{revtex4-2}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc,newunicodechar}
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
    s = s.replace(r'\documentclass[aps,prx,onecolumn,superscriptaddress,longbibliography]{revtex4-2}', r'\documentclass{article}\usepackage[numbers]{natbib}')
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
