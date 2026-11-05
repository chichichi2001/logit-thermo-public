"""公開用の歩ごとの数を作る（トークンも文も入れない）。オーナーの決め（2026-11-05、選択肢から）：「歩ごとの数だけ公開（おすすめ）」。
  python3 scripts/build_per_step_data.py <作業の写しのフォルダ> data/per_step
入れるもの（続きごと、歩ごと）：候補を選んだか c_t（pick、ビットに詰める）、候補の確率 q_t（float32、罰をかけた腕では罰をかけた分布）、
4 つ組のくり返しの印（marks、ビットに詰める）。続きごと：4 つの数え（普通の歩、入る回、繰り返しの歩、出る回）、N、Q、V ＝ Σ q(1 − q)（float64）。
入れないもの：トークン（toks）、候補の集合（cand）、エネルギー（U）、罰なしの q0。
標準出力にはファイルの名前と大きさだけを出す（中身は出さない）。
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, 'scripts')
exec(open('scripts/a2_31_run.py').read().split("\nif __name__")[0])

SETS = {'gpre': 'design_runs_A2_16', 'a18': 'first_test_A2_18', 'a19': 'replication_A2_19', 'a20s1': 'larger_models_stage1_A2_20',
        'a20s2': 'larger_models_stage2_A2_20', 'a23s2': 'multiplicative_A2_23', 'a25': 'stronger_divisors_A2_25'}
DROP = {'toks', 'cand', 'U', 'q0', 'q', 'pick', 'seconds', 'chunk'}

if __name__ == '__main__':
    ns = load31ns()
    src, dst = sys.argv[1:3]
    man = {}
    for d, name in SETS.items():
        os.makedirs(f'{dst}/{name}', exist_ok=True)
        for f in sorted(x for x in os.listdir(f'{src}/{d}') if x.endswith('.npz')):
            Z = np.load(f'{src}/{d}/{f}')
            toks, pick, q = Z['toks'], Z['pick'], Z['q']
            assert toks.shape[-1] == 512 and not (pick & ~Z['cand']).any()
            marks = ns['marks_of'](toks).astype(bool)
            ev, sig = ns['seq_events'](toks, pick, q)
            qd = q.astype(np.float64)
            out = {k: Z[k] for k in Z.files if k not in DROP}
            out.update(pick=np.packbits(pick, -1), marks=np.packbits(marks, -1), q=q.astype(np.float32), events=ev.astype(np.int16),
                       N=pick.sum(-1).astype(np.int16), Q=qd.sum(-1), V=(qd * (1 - qd)).sum(-1), shape=np.array(pick.shape))
            assert np.allclose(out['N'] - out['Q'], sig, atol=1e-6)
            o = f'{dst}/{name}/{f}'
            np.savez_compressed(o, **out)
            man[f'{name}/{f}'] = dict(bytes=os.path.getsize(o), sha256=hashlib.sha256(open(o, 'rb').read()).hexdigest(), shape=list(pick.shape))
            print(name, f, round(os.path.getsize(o) / 1e6, 2), 'MB', flush=True)
    json.dump(man, open(f'{dst}/MANIFEST.json', 'w'), indent=1)
    print('合計', round(sum(v['bytes'] for v in man.values()) / 1e6, 1), 'MB')
