"""探索 T-scan：温度ごとの崩れ方（結晶・電子不足・具体物・共有結合・気体）。2026-10-29 夕方。確かめには使わない（予想なし、後知恵用）。
CPU のノートブックで、scripts/a2_2_collect.py を実行したのと同じ名前空間で実行する（load_docs・split_doc・TEMPS を使う）。
生成はしない。A2-2・A2-5・A2-6 で保存した生成のトークン（温度 15・サンプル 2・記事 40・512 歩）をデコードして数えるだけ。
文章の中身は表示しない・保存しない（数だけ）。

数えるもの（続き 512 トークンをデコードした英語の文で。単語は [A-Za-z][A-Za-z'-]* で切る）：
  copy4  ：4 トークンの完全な写しの割合（トークン列で。書き出しを含めて、それまでに出た 4 つ組）。結晶。
  pron   ：代名詞・指示語（下の PRON）の、単語に占める割合。ルイス酸（電子不足）。
  proper ：文の頭以外で大文字で始まる単語（I を除く）の割合。ルイス塩基（具体物）。
  bind   ：proper のうち、書き出しかそれより前の続きに同じ綴りで出ていたものの割合。共有結合（登場人物のつながり）。
  gas    ：その記事以外の 39 本の人の記事（全文）に一度も出ない単語（小文字にして）の割合。気体（ぐちゃぐちゃ）。
人の続き（同じ記事の 512 トークン）でも同じように数え、基準の線にする。
"""
import json, re
import numpy as np

TS_RUNS = {
    'A2_2': [('pythia-1b', 'EleutherAI/pythia-1b'), ('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B'), ('qwen2.5-0.5b-inst', 'Qwen/Qwen2.5-0.5B-Instruct'),
             ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'),
             ('smollm2-1.7b-inst', 'HuggingFaceTB/SmolLM2-1.7B-Instruct')],
    'A2_5': [(n, 'EleutherAI/pythia-1b') for n in ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1b', 'pythia-1.4b', 'pythia-2.8b',
                                                   'pythia-1b@step8000', 'pythia-1b@step24000', 'pythia-1b@step64000']],
    'A2_6': [('qwen2.5-0.5b', 'Qwen/Qwen2.5-0.5B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B'), ('qwen2.5-3b', 'Qwen/Qwen2.5-3B'),
             ('smollm2-135m', 'HuggingFaceTB/SmolLM2-135M'), ('smollm2-360m', 'HuggingFaceTB/SmolLM2-360M'),
             ('smollm2-1.7b', 'HuggingFaceTB/SmolLM2-1.7B'), ('qwen3-0.6b', 'Qwen/Qwen3-0.6B-Base'), ('qwen3-1.7b', 'Qwen/Qwen3-1.7B-Base'),
             ('qwen3-4b', 'Qwen/Qwen3-4B-Base')],
}
TS_KEYS = ['copy4', 'pron', 'proper', 'bind', 'gas']
PRON = {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs',
        'this', 'that', 'these', 'those'}
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
SENT_END = re.compile(r'[.!?]["\')\]]*\s*$')


def copy4(prompt, cont):
    seq = [int(t) for t in prompt] + [int(t) for t in cont]
    P, seen, c = len(prompt), set(), 0
    for i in range(3, len(seq)):
        g = tuple(seq[i - 3:i + 1])
        if i >= P and g in seen:
            c += 1
        seen.add(g)
    return c / max(len(cont), 1)


def words_with_pos(text):
    """→ [(単語, 文の頭か)]。文の頭 ＝ 文字列の最初の単語、または直前の単語との間に . ! ? がある単語。"""
    out, last = [], 0
    for m in WORD.finditer(text):
        between = text[last:m.start()]
        start = (not out) or bool(re.search(r'[.!?]', between)) or ('\n' in between)
        out.append((m.group(0), start))
        last = m.end()
    return out


def measures(prompt_ids, cont_ids, prompt_text, cont_text, vocab_other):
    ws = words_with_pos(cont_text)
    n = len(ws)
    prev_words = set(w for w, _ in words_with_pos(prompt_text))
    pron = sum(w.lower() in PRON for w, _ in ws)
    proper_hits, bind_hits = 0, 0
    for w, st in ws:
        if not st and w[0].isupper() and w != 'I':
            proper_hits += 1
            bind_hits += int(w in prev_words)
        prev_words.add(w)
    gas = sum(w.lower() not in vocab_other for w, _ in ws)
    return dict(copy4=copy4(prompt_ids, cont_ids), pron=pron / n if n else np.nan, proper=proper_hits / n if n else np.nan,
                bind=bind_hits / proper_hits if proper_hits else np.nan, gas=gas / n if n else np.nan, n_words=n)


def run_tscan(base, docs):
    from transformers import AutoTokenizer
    doc_words = [set(w.lower() for w in WORD.findall(d['text'])) for d in docs]
    res = {'temps': list(TEMPS), 'runs': {}, 'human': {}}
    tok_cache = {}
    for folder, runs in TS_RUNS.items():
        for name, hf in runs:
            if hf not in tok_cache:
                tok_cache[hf] = AutoTokenizer.from_pretrained(hf)
            tok = tok_cache[hf]
            pairs = [split_doc(tok, d['text']) for d in docs]
            Z = np.load(f'{base}/{folder}/{name}.npz')
            use = [int(u) for u in Z['use']]
            toks = Z['toks']                                     # (nT, S, nD, L)
            assert [float(t) for t in Z['temps']] == list(TEMPS)
            nT, S, nD, L = toks.shape
            vocab_other = {i: set().union(*[doc_words[k] for k in range(len(docs)) if k != i]) for i in use}
            ptxt = {i: tok.decode(pairs[i][0], skip_special_tokens=True) for i in use}
            curve = {k: np.full((nT, S, nD), np.nan) for k in TS_KEYS}
            for kT in range(nT):
                for s in range(S):
                    for j, i in enumerate(use):
                        c = toks[kT, s, j]
                        m = measures(pairs[i][0], c, ptxt[i], tok.decode(c, skip_special_tokens=True), vocab_other[i])
                        for k in TS_KEYS:
                            curve[k][kT, s, j] = m[k]
            hk = hf + '|' + ','.join(map(str, use))
            if hk not in res['human']:
                hm = {k: [] for k in TS_KEYS}
                for i in use:
                    m = measures(pairs[i][0], pairs[i][1], ptxt[i], tok.decode(pairs[i][1], skip_special_tokens=True), vocab_other[i])
                    for k in TS_KEYS:
                        hm[k].append(m[k])
                res['human'][hk] = {k: float(np.nanmean(v)) for k, v in hm.items()}
            res['runs'][f'{folder}/{name}'] = dict(human_key=hk, **{k: np.nanmean(curve[k], axis=(1, 2)).tolist() for k in TS_KEYS},
                                                   **{k + '_doc_sd': np.nanstd(np.nanmean(curve[k], axis=1), axis=1).tolist() for k in TS_KEYS})
    return res


def self_test_tscan():
    ws = words_with_pos('Mr Smith went home. He saw Paris and Smith again.\nThe end')
    assert [w for w, st in ws if st] == ['Mr', 'He', 'The'], ws
    m = measures([1, 2, 3], [1, 2, 3, 4, 1, 2, 3, 4], 'Smith arrived.', 'Then Smith met Jones. It was fun and this was odd.',
                 {'then', 'smith', 'met', 'it', 'was', 'fun', 'and', 'this'})
    # 単語 11：Then Smith met Jones It was fun and this was odd。代名詞 it・this ＝ 2。固有名詞（文頭以外）Smith・Jones ＝ 2、そのうち前に出た Smith ＝ 1
    assert abs(m['pron'] - 2 / 11) < 1e-9 and abs(m['proper'] - 2 / 11) < 1e-9 and m['bind'] == 0.5, m
    assert abs(m['gas'] - 2 / 11) < 1e-9, m                       # jones・odd が辞書にない
    assert abs(m['copy4'] - 1 / 8) < 1e-9, m                      # 続きの 8 歩のうち、それまでに出た 4 つ組（1 2 3 4 の 2 回目）は 1 歩
    return 'OK'
