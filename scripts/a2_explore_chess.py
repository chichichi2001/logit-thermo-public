"""探索：チェスの盤の上でエネルギーを測る下見（2026-10-30 10:0x）。確かめには使わない（予想なし）。GPU（L4）。
scripts/a2_2_collect.py を実行したのと同じ名前空間で実行する（load_model だけ使う。7B は bfloat16 で自前に読む）。

発想（オーナー 2026-10-29 23:32〜10-30 00:00、探索ノート 8.14）：（オーナーの承認・発言。原文は非公開の記録にある）
Stockfish（Stockfish）の最善手を基準にして、そこからの逸脱をエネルギーとして見る。

材料：Lichess のパズル（Hugging Face の Lichess/chess-puzzles、CC0）。テーマ zugzwang・skewer・fork・mateIn1・quietMove を 50 局面ずつ。
  選び方：データの並び順に読み、1000 ≤ Rating ≤ 2000、Popularity ≥ 50、NbPlays ≥ 200 のものを、CH_THEMES の順で最初に当てはまる
  テーマに入れる（テーマの枠が埋まっていたら次のテーマ）。候補はテーマごとに CH_CAND 個まで集め、棋譜が取れたものから 50 個使う。
局面：パズルの FEN（相手の 1 手の前）から、Moves の最初の手（相手の手）を指した局面。その局面で手番の側が指す。
見せ方：棋譜の形（PGN）で、見出しに [SetUp "1"] と [FEN "<相手の手の前の局面>"] を書き、相手の 1 手（SAN、手番号つき）を続ける。
  白番なら最後に「 N.」を足す。（はじめは Lichess から元の対局の棋譜を取り寄せる設計だったが、2026-10-30 の Colab からは 3 回とも
  HTML のページしか返らなかったので、ネットにつながずに作れるこの形に替えた。）
エネルギー：Stockfish（深さ CH_DEPTH、全部の合法手を MultiPV）で各手の評価値 → Lichess の式で勝率（0〜1、手番側から見て）→
  E ＝ 勝率（最善手）− 勝率（その手）。パス：王手でなければ、手番だけ相手に渡した局面の評価値 → E_pass（負なら「パスできれば最善手より良い」）。
モデルの確率：各合法手の SAN の前に空白を付けた文字列の log 確率（トークンの和）。合法手だけで足して 1 にしたものを p、足す前の和を「合法手の質量」。
  （O-O と O-O-O は文字列として前後の関係にあるので、両方合法な局面は記録する。）
"""
import io, json, math, os, re, time
import numpy as np

CH_THEMES = ['zugzwang', 'skewer', 'fork', 'mateIn1', 'quietMove']
CH_N, CH_CAND = 50, 70
CH_RATING, CH_POP, CH_PLAYS = (1000, 2000), 50, 200
CH_DEPTH = 12
CH_MODELS = [('pythia-1b', 'EleutherAI/pythia-1b', 'float32'), ('olmo2-1b', 'allenai/OLMo-2-0425-1B', 'float32'),
             ('qwen2.5-1.5b', 'Qwen/Qwen2.5-1.5B', 'float32'), ('qwen2.5-7b', 'Qwen/Qwen2.5-7B', 'bfloat16')]
CH_HEADER = '[Event "Rated game"]\n[Site "https://lichess.org"]\n[Result "*"]\n\n'
CH_TGRID = np.exp(np.linspace(math.log(1e-3), math.log(10.0), 400))
CH_CHUNK = 4


# ---------- 勝率とエネルギー（numpy だけ） ----------
def win_from_cp(cp):
    """Lichess の式：勝率（0〜1）。cp は手番側から見たセンチポーン。"""
    return 0.5 + 0.5 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def win_from_score(kind, val):
    """kind：'cp' か 'mate'。mate は手番側から見て正なら勝ち（1）、負なら負け（0）。"""
    if kind == 'mate':
        return 1.0 if val > 0 else 0.0
    return win_from_cp(val)


# ---------- 当てはめ（numpy だけ） ----------
def boltzmann(E, T):
    z = -np.asarray(E, float) / T
    z = z - z.max()
    q = np.exp(z)
    return q / q.sum()


def fit_T(Ps, Es):
    """局面ごとの (p, E) の組のリスト → 平均 KL(p ‖ q_T) が最小になる T と、そのときの KL、一様分布との KL。"""
    best = (float('inf'), float('nan'))
    kl_unif = float(np.mean([float(np.sum(p * (np.log(np.clip(p, 1e-300, None)) - math.log(1.0 / len(p))))) for p in Ps]))
    for T in CH_TGRID:
        kl = 0.0
        for p, E in zip(Ps, Es):
            q = boltzmann(E, T)
            kl += float(np.sum(p * (np.log(np.clip(p, 1e-300, None)) - np.log(np.clip(q, 1e-300, None)))))
        kl /= len(Ps)
        if kl < best[0]:
            best = (kl, float(T))
    return dict(T=best[1], KL=best[0], KL_uniform=kl_unif)


def spearman_np(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    # 同じ値の順位は平均（小さい組なので素直に）
    for r, x in ((ra, a), (rb, b)):
        for v in np.unique(x):
            m = x == v
            r[m] = r[m].mean()
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(positions, scores):
    """positions：局面のリスト（moves_san・E・theme・E_pass）、scores：{モデル: [局面ごとの log 確率の並び（moves_san の順）]}。"""
    out = {'n_positions': len(positions), 'themes': CH_THEMES, 'models': {}}
    for m, S in scores.items():
        per_theme = {}
        for th in CH_THEMES + ['all']:
            idx = [i for i, pz in enumerate(positions) if th == 'all' or pz['theme'] == th]
            if not idx:
                continue
            Ps, Es, mass, rho, top1, ent, pbest = [], [], [], [], [], [], []
            for i in idx:
                lp = np.asarray(S[i], float)
                E = np.asarray(positions[i]['E'], float)
                mass.append(float(np.exp(lp).sum()))
                p = np.exp(lp - lp.max()); p /= p.sum()
                Ps.append(p); Es.append(E)
                rho.append(spearman_np(-lp, E))                     # 正なら「確率の低い手ほどエネルギーが高い」
                top1.append(float(E[int(np.argmax(p))] <= 0.01))
                ent.append(float(-(p * np.log(np.clip(p, 1e-300, None))).sum()))
                pbest.append(float(p[E <= 0.01].sum()))
            ft = fit_T(Ps, Es)
            per_theme[th] = dict(n=len(idx), legal_mass_mean=float(np.mean(mass)), legal_mass_median=float(np.median(mass)),
                                 rho_median=float(np.nanmedian(rho)), rho_pos_frac=float(np.nanmean(np.array(rho) > 0)),
                                 top1_best=float(np.mean(top1)), p_on_best=float(np.mean(pbest)), entropy=float(np.mean(ent)),
                                 entropy_uniform=float(np.mean([math.log(len(p)) for p in Ps])), **ft)
        out['models'][m] = per_theme
    # 局面の側の記録（モデルによらない）
    for th in CH_THEMES:
        idx = [i for i, pz in enumerate(positions) if pz['theme'] == th]
        Ep = [positions[i]['E_pass'] for i in idx if positions[i]['E_pass'] is not None]
        gap = [sorted(positions[i]['E'])[1] if len(positions[i]['E']) > 1 else float('nan') for i in idx]
        nb = [int(np.sum(np.asarray(positions[i]['E']) <= 0.01)) for i in idx]
        out.setdefault('positions', {})[th] = dict(n=len(idx), n_legal_mean=float(np.mean([len(positions[i]['E']) for i in idx])),
                                                   gap_median=float(np.nanmedian(gap)), n_near_best_mean=float(np.mean(nb)),
                                                   E_pass_median=(float(np.median(Ep)) if Ep else None),
                                                   pass_better_frac=(float(np.mean(np.array(Ep) < 0)) if Ep else None),
                                                   n_pass=len(Ep))
    return out


# ---------- チェス（python-chess と Stockfish。Colab で使う） ----------
def puzzle_rows(limit_scan=3_000_000):
    from datasets import load_dataset
    ds = load_dataset('Lichess/chess-puzzles', split='train', streaming=True)
    got = {t: [] for t in CH_THEMES}
    for k, r in enumerate(ds):
        if k >= limit_scan or all(len(v) >= CH_CAND for v in got.values()):
            break
        th = r['Themes'].split() if isinstance(r['Themes'], str) else list(r['Themes'])
        if not (CH_RATING[0] <= r['Rating'] <= CH_RATING[1] and r['Popularity'] >= CH_POP and r['NbPlays'] >= CH_PLAYS and r.get('GameId')):
            continue
        for t in CH_THEMES:
            if t in th:
                if len(got[t]) < CH_CAND:
                    got[t].append(dict(PuzzleId=r['PuzzleId'], GameId=r['GameId'], FEN=r['FEN'], Moves=r['Moves'],
                                       Rating=int(r['Rating']), Themes=th, theme=t))
                break                                                # テーマは順番で最初に当てはまる 1 つだけ
    return got


def build_position_fen(pz):
    """パズルの FEN（相手の手の前）と相手の 1 手から、局面とプロンプトを作る。ネットは使わない。"""
    import chess
    b0 = chess.Board(pz['FEN'])
    first = chess.Move.from_uci(pz['Moves'].split()[0])
    if first not in b0.legal_moves:
        return None
    san_first = b0.san(first)
    n = b0.fullmove_number
    prefix = f'{n}. {san_first}' if b0.turn == chess.WHITE else f'{n}... {san_first}'
    b = b0.copy(); b.push(first)
    head = CH_HEADER.replace('[Result "*"]\n', '[Result "*"]\n[SetUp "1"]\n[FEN "' + pz['FEN'] + '"]\n')
    prompt = head + prefix + (f' {b.fullmove_number}.' if b.turn == chess.WHITE else '')
    legal = list(b.legal_moves)
    sans = [b.san(m) for m in legal]
    both_castle = ('O-O' in sans) and ('O-O-O' in sans)
    return dict(PuzzleId=pz['PuzzleId'], theme=pz['theme'], Rating=pz['Rating'], prompt=prompt, fen=b.fen(),
                moves_uci=[m.uci() for m in legal], moves_san=sans, solution_uci=pz['Moves'].split()[1],
                both_castle=both_castle)


def engine_energies(engine, fen, depth=CH_DEPTH):
    """→ ({uci: E}、E_pass か None、最善手の勝率)。E_pass は、王手でなく、手番を渡した局面が正しい局面のときだけ出す。"""
    import chess, chess.engine
    b = chess.Board(fen)
    n = b.legal_moves.count()
    infos = engine.analyse(b, chess.engine.Limit(depth=depth), multipv=n)
    W = {}
    for inf in infos:
        sc = inf['score'].pov(b.turn)
        W[inf['pv'][0].uci()] = win_from_score('mate', sc.mate()) if sc.is_mate() else win_from_score('cp', sc.score())
    wbest = max(W.values())
    Epass = None
    if not b.is_check():
        b2 = b.copy(stack=False)
        b2.turn = not b.turn; b2.ep_square = None                    # 手番だけ相手に渡す（null move は送らない）
        if b2.is_valid():
            inf = engine.analyse(b2, chess.engine.Limit(depth=depth))
            sc = inf['score'].pov(b.turn)
            wpass = win_from_score('mate', sc.mate()) if sc.is_mate() else win_from_score('cp', sc.score())
            Epass = wbest - wpass
    return {u: wbest - w for u, w in W.items()}, Epass, wbest


def collect_positions(engine, notify=print):
    rows = puzzle_rows()
    positions, skipped = [], {'cut': 0, 'engine': 0}
    for t in CH_THEMES:
        k = 0
        for pz in rows[t]:
            if k >= CH_N:
                break
            pos = build_position_fen(pz)
            if pos is None:
                skipped['cut'] += 1; continue
            Ed, Ep, wb = engine_energies(engine, pos['fen'])
            if set(Ed) != set(pos['moves_uci']):
                skipped['engine'] += 1; continue
            pos['E'] = [float(Ed[u]) for u in pos['moves_uci']]
            pos['E_pass'] = None if Ep is None else float(Ep)
            pos['W_best'] = float(wb)
            positions.append(pos); k += 1
        notify(f'局面 {t}：{k} 個（候補 {len(rows[t])}）')
    assert len(positions) >= 100, ('局面がほとんど集まりませんでした', skipped)
    return positions, skipped, {t: len(v) for t, v in rows.items()}


# ---------- モデルの確率（Colab で使う） ----------
def load_any(hf_id, dtype, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=getattr(torch, dtype)).to(device).eval()
    return tok, model


def score_moves(tok, model, prompt, sans, device):
    """→ 各手の log 確率（トークンの和、float64）。並べ方は sans の順。プロンプトと手は別々にトークン化してつなぐ。"""
    import torch
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and getattr(tok, 'add_bos_token', False)) else []
    P = bos + tok(prompt, add_special_tokens=False).input_ids
    C = [tok(' ' + s, add_special_tokens=False).input_ids for s in sans]
    out = np.zeros(len(sans))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    with torch.no_grad():
        for a in range(0, len(C), CH_CHUNK):
            cc = C[a:a + CH_CHUNK]
            L = len(P) + max(len(c) for c in cc)
            ids = torch.full((len(cc), L), pad, dtype=torch.long)
            att = torch.zeros((len(cc), L), dtype=torch.long)
            for j, c in enumerate(cc):
                seq = P + c
                ids[j, :len(seq)] = torch.tensor(seq); att[j, :len(seq)] = 1
            lg = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
            lp = torch.log_softmax(lg, -1)
            for j, c in enumerate(cc):
                pos = torch.arange(len(P) - 1, len(P) - 1 + len(c), device=device)
                out[a + j] = float(lp[j, pos, torch.tensor(c, device=device)].double().sum())
    return out


def self_test_chess_math():
    assert abs(win_from_cp(0) - 0.5) < 1e-12 and win_from_cp(300) > 0.7 and win_from_cp(-300) < 0.3
    assert win_from_score('mate', 3) == 1.0 and win_from_score('mate', -2) == 0.0
    # 当てはめ：q_T そのものを p にすると、その T が戻る
    rng = np.random.default_rng(0)
    Es = [np.sort(rng.uniform(0, 1, 20)) for _ in range(30)]
    for E in Es:
        E -= E.min()
    Ps = [boltzmann(E, 0.05) for E in Es]
    ft = fit_T(Ps, Es)
    assert abs(math.log(ft['T'] / 0.05)) < 0.05 and ft['KL'] < 1e-3, ft
    assert abs(spearman_np([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12 and abs(spearman_np([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    # summarize：一様な確率なら top1 は偶然、rho は nan か小さい
    positions = [dict(theme=CH_THEMES[i % 5], E=list(E), E_pass=(0.1 if i % 2 else None)) for i, E in enumerate(Es)]
    scores = {'boltz': [np.log(boltzmann(E, 0.05)) - 0.3 for E in Es]}
    s = summarize(positions, scores)
    a = s['models']['boltz']['all']
    assert abs(a['legal_mass_mean'] - math.exp(-0.3)) < 1e-9 and a['rho_median'] > 0.99 and abs(math.log(a['T'] / 0.05)) < 0.05, a
    return 'OK'


def self_test_chess_board():
    """python-chess と Stockfish が入っていること、局面とプロンプトの作り方、メイトの局面のエネルギー。"""
    import chess, chess.engine
    # 3... Nf6 の前（黒番、3 手目）の局面を「パズルの FEN」とし、相手の手 Nf6（g8f6）のあとで白が指す（答えは Qxf7#）
    b = chess.Board(); [b.push_san(s) for s in ['e4', 'e5', 'Bc4', 'Nc6', 'Qh5']]
    pz = dict(PuzzleId='t', theme='mateIn1', Rating=1000, FEN=b.fen(), Moves='g8f6 h5f7')
    pos = build_position_fen(pz)
    assert pos is not None and pos['prompt'].endswith('3... Nf6 4.'), pos and pos['prompt']
    assert ('[FEN "' + b.fen() + '"]') in pos['prompt'] and '[SetUp "1"]' in pos['prompt']
    assert 'Qxf7#' in pos['moves_san'] and pos['solution_uci'] == 'h5f7'
    # 白番の前（白が相手）：1. e4 のあと黒が指す → プロンプトは「1. e4」で終わる
    pz2 = dict(PuzzleId='t2', theme='fork', Rating=1000, FEN=chess.STARTING_FEN, Moves='e2e4 e7e5')
    pos2 = build_position_fen(pz2)
    assert pos2['prompt'].endswith('\n\n1. e4') and 'e5' in pos2['moves_san'], pos2['prompt']
    assert build_position_fen(dict(pz, Moves='a1a8 h5f7')) is None                 # 指せない手なら飛ばす
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    try:
        Ed, Ep, wb = engine_energies(eng, pos['fen'], depth=8)
        assert Ed['h5f7'] == 0.0 and wb == 1.0 and all(v > 0 for u, v in Ed.items() if u != 'h5f7'), Ed
        assert set(Ed) == set(pos['moves_uci'])
    finally:
        eng.quit()
    return 'OK'


def self_test_score(device):
    """小さなランダムのモデルで：まとめて（右を埋めて）計算した手の log 確率が、1 本ずつ計算したものと同じ。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_id = 'hf-internal-testing/tiny-random-GPT2LMHeadModel'
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32).to(device).eval()
    prompt = CH_HEADER + '1. e4 e5 2. Nf3 Nc6 3.'
    sans = ['Bb5', 'Bc4', 'd4', 'O-O', 'Nxe5', 'Qe2', 'h3', 'a4', 'c3']
    a = score_moves(tok, model, prompt, sans, device)
    one = np.array([score_moves(tok, model, prompt, [s_], device)[0] for s_ in sans])
    assert np.allclose(a, one, atol=1e-4), (a, one)
    # 手で 1 本だけ計算
    P = tok(prompt, add_special_tokens=False).input_ids; c = tok(' Bb5', add_special_tokens=False).input_ids
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([P + c], device=device)).logits.float(), -1)[0]
    ref = sum(float(lp[len(P) - 1 + k, c[k]]) for k in range(len(c)))
    assert abs(ref - a[0]) < 1e-4, (ref, a[0])
    return 'OK'
