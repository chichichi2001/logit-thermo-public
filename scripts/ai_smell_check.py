"""論文の英語の本文から「AI が書いた文に見えやすい特徴（臭み）」を数えて、場所つきで書き出す。
  python3 scripts/ai_smell_check.py paper/review_copy_v05.md results/paper_checks/ai_smell.json

定義と出典は docs/AI_SMELL_CHECK.md。ここでは数えるだけで、直さない。
層：L1 単語、L2 文の形、L3 段落・文の運び、L4 中身（スロップ）、L5 Claude らしさ（出典なし・作者の見立て）。
数は 1,000 語あたり。人の論文の基準値は手元にないので、比べるのは「節どうし」と「出典にある目安」だけ。
"""
import re, sys, json, statistics
from collections import Counter

# ---- L1 単語 ----
# Kobak et al. (Sci. Adv. 2025) の 2024 年の過剰な語（本文に名前のある語と、下限の推定に使った語）と、
# Wikipedia:Signs of AI writing の「AI vocabulary」「言いすぎ」「浅い分析」の語。
L1_KOBAK = ['delve', 'delves', 'delving', 'underscore', 'underscores', 'underscoring', 'showcase', 'showcasing', 'showcases',
            'potential', 'findings', 'crucial', 'across', 'additionally', 'comprehensive', 'enhance', 'enhancing', 'enhances',
            'exhibited', 'insights', 'notably', 'particularly', 'within', 'intricate', 'meticulous', 'meticulously', 'pivotal', 'realm']
L1_WIKI = ['tapestry', 'testament', 'landscape', 'vital', 'highlight', 'highlights', 'highlighting', 'foster', 'fostering',
           'ensure', 'ensuring', 'ensures', 'valuable', 'encompass', 'encompassing', 'align', 'aligns', 'resonate', 'boasts',
           'vibrant', 'profound', 'groundbreaking', 'renowned', 'robust', 'seamless', 'seamlessly', 'leverage', 'leverages',
           'utilize', 'utilizes', 'nuanced', 'noteworthy', 'paramount', 'multifaceted', 'holistic', 'garner', 'elucidate',
           'moreover', 'furthermore', 'importantly', 'key', 'novel', 'remarkable', 'remarkably', 'striking', 'strikingly']
L1_PHRASES = ['plays a role', 'play a role', 'a key role', 'serves as', 'stands as', 'it is worth noting', 'it is important to note',
              'shed light', 'sheds light', 'pave the way', 'paves the way', 'in conclusion', 'in summary', 'overall,',
              'in the context of', 'a wide range', 'a variety of', 'in terms of', 'the landscape of', 'valuable insights']

# ---- L2 文の形 ----
ING_OK = {'including', 'according', 'during', 'following', 'using', 'excluding', 'concerning', 'regarding', 'containing',
          'remaining', 'corresponding', 'preceding', 'resulting', 'nothing', 'something', 'anything', 'everything', 'string',
          'sampling', 'labeling', 'smoothing', 'reweighting', 'tilting', 'scoring', 'binning', 'training', 'setting', 'ending',
          'starting', 'matching', 'repeating', 'rounding', 'generating', 'beginning', 'thing', 'king', 'bring', 'spring'}
NOM = re.compile(r'\b[a-z]{3,}(?:tion|tions|ment|ments|ness|ity|ities|ance|ances|ence|ences)\b')
HEDGE = re.compile(r'\b(may|might|could|likely|roughly|approximately|appears?|seems?|suggests?|possibly|perhaps|somewhat|slightly|largely|mainly|mostly)\b', re.I)
# L5 Claude らしさ（作者＝この会話の AI の見立て。出典なし。数えて目で確かめるための候補）
L5 = {'exactly': r'\bexactly\b', 'directly': r'\bdirectly\b', 'precisely': r'\bprecisely\b', 'only (強調)': r'\bonly\b',
      'what X is Y（擬似分裂文）': r'(?:^|[.;] )What [a-z][^.]{0,60}\b(is|are|was|were)\b',
      'this is/that is（言いかえ）': r'\b(In other words|Put differently|That is,|This is (why|where|what|because))',
      'We read this as / we take': r'\bWe (read|take|interpret|call)\b',
      'thus/therefore/so（まとめ）': r'\b(thus|therefore|hence)\b',
      'nevertheless/however': r'\b(nevertheless|however|instead|yet)\b',
      'both ... and': r'\bboth\b[^.]{1,60}\band\b'}


def sections(md):
    """見出しで分ける（付録に入ったかも見出しで決める）。表の行・図の行・空行は本文から外す。"""
    out, name, app = [], 'Title', False
    for line in md.split('\n'):
        if line.startswith('#'):
            name = line.lstrip('#').strip()
            if name.startswith('Appendices') or name.startswith('Appendix'):
                app = True
            continue
        if not line.strip() or line.startswith('|') or line.startswith('!['):
            continue
        out.append((name, app, line.strip()))
    return out


def sentences(text):
    t = re.sub(r'\b(e\.g|i\.e|et al|Fig|Figs|Sec|Eq|Ref|vs|cf|approx)\.', lambda m: m.group(0).replace('.', '§'), text)
    t = re.sub(r'(\d)\.(\d)', r'\1§\2', t)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z(“"])', t)
    return [p.replace('§', '.').strip() for p in parts if p.strip()]


def words(s):
    return re.findall(r"[A-Za-z][A-Za-z'-]*", s)


def main(src, dst):
    md = open(src).read()
    paras = sections(md)
    rows = []
    for name, app, p in paras:
        part = 'abstract' if name == 'Abstract' else ('appendix' if app else 'main')
        is_cap = p.startswith('**Fig.') or p.startswith('**Table')
        for s in sentences(p):
            rows.append({'sec': name, 'part': part, 'caption': is_cap, 's': s, 'w': len(words(s))})
    res = {'source': src, 'parts': {}}
    hits = {}

    def add(key, r, frag):
        hits.setdefault(key, []).append({'sec': r['sec'], 'part': r['part'], 'frag': frag, 'sentence': r['s'][:220]})

    for r in rows:
        low = r['s'].lower()
        for w in words(r['s']):
            lw = w.lower()
            if lw in L1_KOBAK: add('L1 Kobak の語', r, lw)
            if lw in L1_WIKI: add('L1 Wikipedia の語', r, lw)
        for ph in L1_PHRASES:
            if ph in low: add('L1 決まり文句', r, ph)
        for m in re.finditer(r', ([a-z]+ing)\b', r['s']):
            if m.group(1) not in ING_OK: add('L2 文末につく -ing の句', r, m.group(0))
        for m in re.finditer(r'\bnot only\b|\bnot just\b|\bnot merely\b|, not [a-z]+|\bnot [^.,;]{1,40}\bbut\b|\brather than\b|\binstead of\b', r['s']):
            add('L2 否定と対比（not X but Y / rather than）', r, m.group(0))
        for m in re.finditer(r'\b(serves as|stands as|acts as|functions as|represents|constitutes)\b', r['s']):
            add('L2 is の言いかえ', r, m.group(0))
        if '—' in r['s']: add('L2 em ダッシュ', r, '—')
        if not r['caption']:
            for m in re.finditer(r'(?<!\d):(?!\d)', re.sub(r'\*\*[^*]+\*\*', '', r['s'])):
                add('L2 文中のコロン（図表の説明と太字の見出しは除く）', r, ':')
        if ';' in re.sub(r'\[[^\]]*\]|\([^)]*\)', '', r['s']) and not r['caption']: add('L2 セミコロン（文献・括弧・図表の説明の中は除く）', r, ';')
        for m in re.finditer(r'\b[A-Za-z-]+, [A-Za-z -]{1,25}, (?:and|or) [A-Za-z-]+', r['s']):
            add('L2 三つ並べ（A, B and C）', r, m.group(0))
        for m in re.finditer(r'\bfrom [^,.;]{1,30} to [^,.;]{1,30}', r['s']):
            if not re.search(r'\d', m.group(0)): add('L2 from X to Y（数でない）', r, m.group(0))
        if r['w'] > 35 and not r['caption']: add('L3 35 語を超える文', r, f"{r['w']} 語")
        if r['w'] <= 6 and not r['caption']: add('L3 6 語以下の短い文', r, f"{r['w']} 語")
        for m in re.finditer(r'^(Thus|Therefore|Hence|In short|In other words|Put differently|Overall|Taken together|In sum)\b', r['s']):
            add('L3 まとめの文頭', r, m.group(0))
        for m in HEDGE.finditer(r['s']):
            add('L4 ぼかし語', r, m.group(0).lower())
        if r['part'] != 'appendix' and not r['caption']:
            nocite = re.sub(r'\[[^\]]*\]', '', r['s'])                       # [2406.05335; …] の文献は数えない
            nums = re.findall(r'(?<![\w.])[−-]?\d+(?:\.\d+)?%?', nocite)
            nums = [n for n in nums if not re.fullmatch(r'\d', n)]   # 1 桁の整数（番号・個数）は数えない
            if nums: add('L4 本文の数字（1 桁の整数は除く）', r, ' '.join(nums))
        for k, pat in L5.items():
            fl = re.I if k in ('exactly', 'directly', 'precisely', 'thus/therefore/so（まとめ）', 'nevertheless/however', 'both ... and') else 0
            for m in re.finditer(pat, r['s'], flags=fl):
                add('L5 ' + k, r, m.group(0))
    # 部分ごとの語数・文の長さ
    for part in ('abstract', 'main', 'appendix'):
        ws = [r['w'] for r in rows if r['part'] == part and not r['caption']]
        tot = sum(ws)
        nom = sum(len(NOM.findall(r['s'].lower())) for r in rows if r['part'] == part and not r['caption'])
        starts = Counter(words(r['s'])[0] for r in rows if r['part'] == part and not r['caption'] and words(r['s']))
        res['parts'][part] = {'words': tot, 'sentences': len(ws),
                              'mean_len': round(statistics.mean(ws), 1) if ws else None,
                              'sd_len': round(statistics.pstdev(ws), 1) if ws else None,
                              'cv_len': round(statistics.pstdev(ws) / statistics.mean(ws), 2) if ws else None,
                              'nominalizations_per_1000': round(1000 * nom / tot, 1) if tot else None,
                              'top_sentence_starts': starts.most_common(8)}
    summ = {}
    for k, v in hits.items():
        c = Counter(h['part'] for h in v)
        summ[k] = {p: {'n': c.get(p, 0), 'per_1000': round(1000 * c.get(p, 0) / res['parts'][p]['words'], 2)}
                   for p in ('abstract', 'main', 'appendix')}
        summ[k]['top'] = Counter(h['frag'].lower() for h in v).most_common(10)
    res['summary'] = summ
    res['hits'] = hits
    # 本文（main）の節ごとの数字の数
    per_sec = Counter()
    for h in hits.get('L4 本文の数字（1 桁の整数は除く）', []):
        per_sec[h['sec']] += len(h['frag'].split())
    res['numbers_per_section_sentences'] = per_sec.most_common()
    json.dump(res, open(dst, 'w'), ensure_ascii=False, indent=1)
    for k in sorted(summ):
        s = summ[k]
        print(f"{k:42s} 要旨 {s['abstract']['n']:3d}  本文 {s['main']['n']:4d} ({s['main']['per_1000']:5.2f}/千語)  付録 {s['appendix']['n']:4d} ({s['appendix']['per_1000']:5.2f}/千語)  {s['top'][:6]}")
    print(json.dumps(res['parts'], ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
