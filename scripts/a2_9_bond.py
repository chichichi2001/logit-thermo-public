"""A2-9（ゲート docs/A2_9_GATE.md）：文章の結合エネルギー。GPU（L4）で読ませ、そのあと同じノートブックで分析（CPU の計算）。
scripts/a2_2_collect.py（load_model・valid_mask）を実行したのと同じ名前空間で実行する。分析の部分はデータを開く前に固定する。

結合：
- 代名詞の結合（P）：「名前 … Later that day, she」。名前を性別の違う名前に入れ替えて切る。結合エネルギー ＝ 切ったときの
  代名詞の −log p − 切らないときの −log p（ナット）。名前の組は 2 つの向き（女の名前＋she と男の名前＋he）で両方使う。
- 名前の再登場の結合（R）：「Dr. 名字 … The next morning, Dr. 名字」。1 回目の名字を別の名字に入れ替えて切る。2 回目の名字の
  トークンの −log p の和で測る。
- 結合の長さ：あいだに入れる、人の出てこない文（つなぎの文）の数 k ＝ 0・1・2・4・8。
- 2 本の結合の足し算（AB）：「Dr. A and Dr. B … Dr. A … Dr. B」で、A だけ切る・B だけ切る・両方切る（k ＝ 2）。
  2 回目の A と B のトークンの −log p の和で測る。比 ＝ 両方切った増え方 ÷（A だけ ＋ B だけ）。
エネルギーはどれも温度 1 の −log p（使わない番号と EOS を外す。A2-8 の U と同じ）。
"""
import math
import numpy as np

A9_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B'), ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B')]
A9_SEED = 20261113
A9_KS = [0, 1, 2, 4, 8]
A9_K_AB = 2
A9_N_ITEMS = 60
A9_BOOT, A9_BOOT_SEED = 1000, 20261114
A9_Q1_BINS = [(0.4, 'かなり弱まる（4 割未満）'), (0.8, '半分くらいに弱まる（4〜8 割）'), (1.25, 'ほとんど弱まらない（8 割〜1.25 倍）'),
              (float('inf'), '強くなる（1.25 倍以上）')]
A9_Q2_BINS = [(0.8, '足し算より小さい（0.8 未満）'), (1.25, 'ほぼ足し算（0.8〜1.25）'), (float('inf'), '足し算より大きい（1.25 以上）')]

FEMALE = ['Mary', 'Linda', 'Susan', 'Karen', 'Nancy', 'Sarah', 'Emily', 'Laura', 'Helen', 'Anna', 'Rachel', 'Julia', 'Grace',
          'Alice', 'Emma', 'Clara', 'Lucy', 'Diana', 'Rose', 'Ellen']
MALE = ['John', 'Robert', 'Michael', 'David', 'James', 'Thomas', 'Daniel', 'Paul', 'Mark', 'Peter', 'George', 'Edward', 'Henry',
        'Frank', 'Arthur', 'Walter', 'Samuel', 'Oliver', 'Victor', 'Simon']
SURNAMES = ['Thornbury', 'Pemberton', 'Ashcombe', 'Whitlock', 'Calloway', 'Fairbairn', 'Hollinger', 'Marchetti', 'Okonkwo',
            'Lindqvist', 'Bramwell', 'Castellan', 'Dunmore', 'Everleigh', 'Galbraith', 'Halvorsen', 'Ingersoll', 'Kettering',
            'Lockhart', 'Montague', 'Northcott', 'Pellegrini', 'Quennell', 'Radcliffe', 'Sinclair', 'Tremaine', 'Underhill',
            'Vandermeer', 'Wetherby', 'Yarborough', 'Abernathy', 'Blackwood', 'Carrington', 'Delacroix', 'Ellsworth',
            'Fitzwilliam', 'Gainsford', 'Hargreaves', 'Ivanova', 'Jessop']
FILLERS = ['The river ran slowly past the old stone bridge.', 'Rain had fallen on the hills for most of the week.',
           'A few boats were tied up along the narrow harbour wall.', 'The market square was quiet in the early afternoon.',
           'Snow still covered the higher fields above the valley.', 'The road to the coast was lined with tall poplar trees.',
           'An old clock tower stood at the end of the main street.', 'The library had recently been painted a pale green.',
           'Wind moved through the long grass beside the railway line.', 'The bakery window was full of bread and small cakes.',
           'A thin mist hung over the lake until almost noon.', 'The museum was closed for repairs to its glass roof.',
           'Several trucks waited outside the warehouse near the docks.', 'The garden behind the school was full of roses.',
           'Bells from the church could be heard across the town.', 'The price of coffee had risen again that month.',
           'A new bus route now linked the village to the city.', 'The bridge over the canal was being widened.',
           'Leaves had begun to turn red along the park paths.', 'The station clock was running about five minutes slow.',
           'Fresh fish were sold from a stall near the pier.', 'The castle on the hill was open to visitors in summer.',
           'Heavy clouds gathered over the sea in the evening.', 'The old mill had been turned into a small hotel.',
           'A long freight train passed through without stopping.', 'The fountain in the square had been switched off.',
           'Street lamps came on one by one as it grew dark.', 'The ferry crossed the bay twice a day in winter.',
           'A row of bicycles leaned against the town hall railings.', 'The forest beyond the farms was dense and dark.']
VP1 = ['opened the small bakery on the corner.', 'arrived at the station just before nine.', 'signed the contract at the office.',
       'walked along the beach after breakfast.', 'finished painting the kitchen walls.', 'repaired the broken fence by the gate.',
       'bought a ticket for the evening concert.', 'planted tomatoes in the back garden.', 'wrote a long letter to the council.',
       'cleaned the windows of the old shop.']
VP2 = ['walked to the market.', 'checked the morning post.', 'called the bank.', 'made a pot of tea.', 'read the newspaper.']


def _fillers(rng, k):
    return ' '.join(rng.choice(FILLERS, size=k, replace=False).tolist()) if k else ''


def _join(*parts):
    return ' '.join(p for p in parts if p)


def build_items():
    """→ dict(P=[...], R=[...], AB=[...])。各項目は {'texts': {条件: 文}, 'targets': {条件: [(始まりの文字, 終わりの文字), ...]}}。
    同じ項目の中では、入れ替える名前以外の文字はすべて同じ。"""
    rng = np.random.default_rng(A9_SEED)
    P, R, AB = [], [], []
    for i in range(A9_N_ITEMS):
        f, m = FEMALE[i % 20], MALE[(i * 7 + 3) % 20]
        vp = VP1[i % len(VP1)]
        for k in A9_KS:
            fill = _fillers(rng, k)
            for name, pro, other in ((f, 'she', m), (m, 'he', f)):
                d = dict(item=i, k=k, pro=pro, texts={}, targets={})
                for cond, nm in (('intact', name), ('cut', other)):
                    head = _join(f'{nm} {vp}', fill, 'Later that day,')
                    t = head + ' ' + pro
                    d['texts'][cond] = t; d['targets'][cond] = [(len(head), len(t))]
                P.append(d)
        s = rng.choice(len(SURNAMES), size=4, replace=False)
        sa, sb, sc, sd = (SURNAMES[j] for j in s)
        for k in A9_KS:
            fill = _fillers(rng, k)
            d = dict(item=i, k=k, texts={}, targets={})
            for cond, nm in (('intact', sa), ('cut', sc)):
                head = _join(f'Dr. {nm} {vp}', fill, 'The next morning, Dr.')
                t = head + ' ' + sa
                d['texts'][cond] = t; d['targets'][cond] = [(len(head), len(t))]
            R.append(d)
        fill = _fillers(rng, A9_K_AB)
        vp2 = VP2[i % len(VP2)]
        d = dict(item=i, k=A9_K_AB, texts={}, targets={})
        for cond, (x, y) in (('intact', (sa, sb)), ('cutA', (sc, sb)), ('cutB', (sa, sd)), ('cutAB', (sc, sd))):
            h1 = _join(f'Dr. {x} and Dr. {y} met at the station.', fill, 'The next morning, Dr.')
            t1 = h1 + ' ' + sa
            h2 = t1 + ' ' + vp2 + ' After lunch, Dr.'
            t = h2 + ' ' + sb
            d['texts'][cond] = t; d['targets'][cond] = [(len(h1), len(t1)), (len(h2), len(t))]
        AB.append(d)
    return dict(P=P, R=R, AB=AB)


# ---------- 読ませる（GPU） ----------

def target_tokens(tok, text, spans):
    """→ (トークン列, 的のトークンの位置のリスト)。的の文字の区間が、トークンの切れ目にぴったり合うことを確かめる。"""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = list(enc['input_ids']), list(enc['offset_mapping'])
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
    pos = []
    for a, b in spans:
        idx = [j for j, (s, e) in enumerate(off) if s < b and e > a]
        assert idx, ('的のトークンがありません', text[a:b])
        # 区間の頭がトークンの頭と合う（前の語とくっついていない）。空白込みのトークンは、頭が a か a+1
        assert off[idx[0]][0] in (a, a + 1) and off[idx[-1]][1] == b, ('的の区間がトークンの切れ目と合いません', text[a:b], [off[j] for j in idx])
        pos += [j + len(bos) for j in idx]
    return bos + ids, pos


def score_texts(tok, model, valid, texts_spans, device, bs=8):
    """texts_spans：[(文, 区間のリスト)]。→ 各文の的のトークンの −log p₁ の和（ナット、float64）と、的のトークンの数。"""
    import torch
    enc = [target_tokens(tok, t, sp) for t, sp in texts_spans]
    out, ntok = np.zeros(len(enc)), np.zeros(len(enc), int)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for i0 in range(0, len(enc), bs):
            chunk = enc[i0:i0 + bs]
            Lm = max(len(ids) for ids, _ in chunk)
            x = torch.full((len(chunk), Lm), pad, dtype=torch.long, device=device)
            att = torch.zeros((len(chunk), Lm), dtype=torch.long, device=device)
            for b, (ids, _) in enumerate(chunk):
                x[b, :len(ids)] = torch.tensor(ids, device=device); att[b, :len(ids)] = 1
            lg = model(input_ids=x, attention_mask=att).logits.double()
            lg = lg.masked_fill(~valid[None, None, :], float('-inf'))
            lp = torch.log_softmax(lg, dim=-1)
            for b, (ids, pos) in enumerate(chunk):
                p = torch.tensor(pos, device=device)
                y = torch.tensor([ids[j] for j in pos], device=device)
                out[i0 + b] = float(-lp[b, p - 1, y].sum())
                ntok[i0 + b] = len(pos)
    return out, ntok


def run_bond(name, hf_id, items, device):
    tok, model = load_model(hf_id, device)
    valid, _ = valid_mask(tok, model, device)
    res = {}
    for kind, lst in items.items():
        conds = list(lst[0]['texts'])
        flat = [(d['texts'][c], d['targets'][c]) for d in lst for c in conds]
        e, n = score_texts(tok, model, valid, flat, device)
        res[kind] = dict(conds=conds, E=e.reshape(len(lst), len(conds)).tolist(), ntok=n.reshape(len(lst), len(conds)).tolist())
    del model
    return res


def self_test_score9(device):
    """小さなランダムのモデルで：まとめて読ませた値と、1 本ずつ・詰め物なしで読ませた値が合う。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    valid, _ = valid_mask(tok, model, device)
    it = build_items()
    ts = [(d['texts'][c], d['targets'][c]) for d in it['P'][:6] + it['AB'][:3] for c in d['texts']]
    a, _ = score_texts(tok, model, valid, ts, device, bs=8)
    b, _ = score_texts(tok, model, valid, ts, device, bs=1)
    assert np.abs(a - b).max() < 1e-3, np.abs(a - b).max()
    return 'OK'


# ---------- 分析（CPU） ----------

def bond_energies(res):
    """→ P・R：(項目, k) の結合エネルギー（P は 2 つの向きの平均）、AB：(項目, 3) の [A だけ, B だけ, 両方] の増え方。"""
    out = {}
    for kind in ('P', 'R'):
        r = res[kind]
        E = np.array(r['E'])
        ci, cc = r['conds'].index('intact'), r['conds'].index('cut')
        dE = E[:, cc] - E[:, ci]
        per = 2 if kind == 'P' else 1
        out[kind] = dE.reshape(A9_N_ITEMS, len(A9_KS), per).mean(-1)
    r = res['AB']
    E = np.array(r['E'])
    c = {k: r['conds'].index(k) for k in r['conds']}
    out['AB'] = np.stack([E[:, c['cutA']] - E[:, c['intact']], E[:, c['cutB']] - E[:, c['intact']], E[:, c['cutAB']] - E[:, c['intact']]], 1)
    return out


def stats_model(be, rng):
    k0, k8 = A9_KS.index(0), A9_KS.index(8)
    f_ret = lambda X: float(X[:, k8].mean() / X[:, k0].mean()) if X[:, k0].mean() > 0 else float('nan')
    f_add = lambda X: float(X[:, 2].mean() / (X[:, 0].mean() + X[:, 1].mean())) if (X[:, 0].mean() + X[:, 1].mean()) > 0 else float('nan')
    pt = dict(P_by_k=be['P'].mean(0).tolist(), R_by_k=be['R'].mean(0).tolist(), AB=be['AB'].mean(0).tolist(),
              P_ret=f_ret(be['P']), R_ret=f_ret(be['R']), add=f_add(be['AB']))
    bs = {k: [] for k in ('P0', 'P_ret', 'R_ret', 'add')}
    for _ in range(A9_BOOT):
        ii = rng.integers(0, A9_N_ITEMS, A9_N_ITEMS)
        bs['P0'].append(float(be['P'][ii, 0].mean())); bs['P_ret'].append(f_ret(be['P'][ii]))
        bs['R_ret'].append(f_ret(be['R'][ii])); bs['add'].append(f_add(be['AB'][ii]))
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
    pt['ci'] = ci
    pt['P_bond_exists'] = bool(ci['P0'][0] > 0)
    return pt


def bin_word(x, bins):
    if not np.isfinite(x):
        return 'はっきりしない'
    if x < 0:
        return '向きが逆'
    for edge, w in bins:
        if x < edge:
            return w


def analyze9(allres):
    rng = np.random.default_rng(A9_BOOT_SEED)
    out = {'models': {}}
    for name, _ in A9_MODELS:
        be = bond_energies(allres[name])
        out['models'][name] = stats_model(be, rng)
    ms = [out['models'][m] for m, _ in A9_MODELS]
    rets = [m['P_ret'] for m in ms if m['P_bond_exists'] and np.isfinite(m['P_ret'])]
    out['Q1'] = bin_word(float(np.median(rets)), A9_Q1_BINS) if len(rets) >= 2 else 'はっきりしない'
    adds = [m['add'] for m in ms if np.isfinite(m['add'])]
    out['Q2'] = bin_word(float(np.median(adds)), A9_Q2_BINS) if len(adds) >= 2 else 'はっきりしない'
    out['Q1_each'], out['Q2_each'] = rets, adds
    return out


def self_test_a9():
    it = build_items()
    assert len(it['P']) == A9_N_ITEMS * len(A9_KS) * 2 and len(it['R']) == A9_N_ITEMS * len(A9_KS) and len(it['AB']) == A9_N_ITEMS
    for d in it['P'][:40] + it['R'][:40] + it['AB'][:20]:
        ts = list(d['texts'].values())
        assert all(len(t) > 0 for t in ts) and len(set(ts)) == len(ts)
        for c, t in d['texts'].items():
            for a, b in d['targets'][c]:
                assert t[a] == ' ' and t[a + 1:b].strip() == t[a + 1:b]
        # 的の中身はどの条件でも同じ
        tg = [[d['texts'][c][a:b] for a, b in d['targets'][c]] for c in d['texts']]
        assert all(x == tg[0] for x in tg)
    # 人の出てこないつなぎの文
    for s in FILLERS:
        w = set(s.replace('.', '').split())
        assert not (w & {'he', 'she', 'He', 'She', 'his', 'her', 'His', 'Her', 'him', 'man', 'woman'}) and not (w & set(FEMALE + MALE + SURNAMES)), s
    # 作り物の分析：P は距離で半分に、R は変わらず、AB はちょうど足し算
    fake = {}
    rng = np.random.default_rng(0)
    for name, _ in A9_MODELS:
        nP, nR = A9_N_ITEMS * len(A9_KS) * 2, A9_N_ITEMS * len(A9_KS)
        kk = np.tile(np.repeat(np.array(A9_KS), 2), A9_N_ITEMS)
        EP = np.stack([np.full(nP, 3.0), 3.0 + 2.0 * np.exp(-kk / 8 * math.log(2)) + rng.normal(0, 0.2, nP)], 1)
        ER = np.stack([np.full(nR, 1.0), 9.0 + rng.normal(0, 0.2, nR)], 1)
        EA = np.stack([np.zeros(A9_N_ITEMS), 5 + rng.normal(0, .1, A9_N_ITEMS), 6 + rng.normal(0, .1, A9_N_ITEMS), 11 + rng.normal(0, .1, A9_N_ITEMS)], 1)
        fake[name] = dict(P=dict(conds=['intact', 'cut'], E=EP.tolist()), R=dict(conds=['intact', 'cut'], E=ER.tolist()),
                          AB=dict(conds=['intact', 'cutA', 'cutB', 'cutAB'], E=EA.tolist()))
    r = analyze9(fake)
    m = r['models']['pythia-1b']
    assert abs(m['P_ret'] - 0.5) < 0.05 and abs(m['R_ret'] - 1) < 0.05 and abs(m['add'] - 1) < 0.03, m
    assert r['Q1'] == '半分くらいに弱まる（4〜8 割）' and r['Q2'] == 'ほぼ足し算（0.8〜1.25）', r
    return 'OK'
