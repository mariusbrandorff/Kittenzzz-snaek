"""Supplied experiment management; the Q update and learning loop are student work."""
import argparse
import csv
import json
import time
import numpy as np
import settings as cfg
from supplied_environment import SoloBattlesnake
from supplied_visualizer import animate_runs, board_image, training_plot


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def experiment_settings():
    names = ['BOARD_SIZE','MAX_STEPS','MAX_HEALTH','FOOD_REWARD','DEATH_REWARD','STEP_REWARD',
             'STATE_SHAPE','ENCODING_VERSION','ACTIONS']
    return {name: getattr(cfg,name) for name in names}


def save_table(q, path, seed, episode):
    metadata = dict(environment=experiment_settings(),seed=seed,episode=episode,
                    alpha=cfg.ALPHA,gamma=cfg.GAMMA)
    np.savez_compressed(path,q=q,metadata=json.dumps(metadata))


def load_table(path):
    if not path.is_file():
        raise FileNotFoundError('Complete Steps 1–3 and train a Q-table before evaluation.')
    with np.load(path,allow_pickle=False) as archive:
        q, metadata = archive['q'],json.loads(str(archive['metadata']))
    if metadata['environment'] != json.loads(json.dumps(experiment_settings())):
        raise ValueError('Environment or encoding settings changed. Retrain before comparing results.')
    if q.shape != cfg.STATE_SHAPE+(4,) or not np.isfinite(q).all():
        raise ValueError('Invalid Q-table.')
    return q


def play(q, encode, choose, seed, policy_seed=0, keep_frames=False):
    env = SoloBattlesnake()
    observation,_ = env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed)
    frames = [(observation,None)] if keep_frames else []
    total = 0.
    while True:
        action = int(rng.integers(4)) if q is None else choose(q[encode(observation)],0.,rng)
        observation,reward,terminated,truncated,info = env.step(action)
        total += reward
        if keep_frames:frames.append((observation,reward))
        if terminated or truncated:break
    return dict(reward=total,steps=env.turn,apples=info['apples'],
                terminated=int(terminated),truncated=int(truncated),reason=info['reason']),frames


def explore():
    cfg.RESULTS.mkdir(parents=True,exist_ok=True)
    metric,frames = play(None,None,None,seed=7,policy_seed=7,keep_frames=True)
    animate_runs([('Random policy',frames,metric['reason'])],cfg.RESULTS/'step_0_random.gif')
    transitions=[]
    for (before,_),(after,reward) in zip(frames,frames[1:]):
        head,next_head=before['you']['head'],after['you']['head']
        delta=(next_head['x']-head['x'],next_head['y']-head['y'])
        transitions.append(dict(turn=before['turn'],head=str(head),action=cfg.ACTIONS[cfg.DELTAS.index(delta)],
                                reward=reward,next_head=str(next_head),health=after['you']['health']))
    write_csv(cfg.RESULTS/'step_0_transitions.csv',transitions)
    print(json.dumps(metric,indent=2))
    for transition in transitions[:10]:print(transition)
    print('Watch results/step_0_random.gif. Yellow=head, teal=body, red=food/death marker.')


def train_cli(encode, choose, train_episode):
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=cfg.EPISODES)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    args=parser.parse_args()
    if not 1 <= args.episodes <= 100000 or any(s<0 or s>6 for s in args.seeds):
        raise ValueError('Use 1–100000 episodes and training seeds 0–6 to keep training/test seeds disjoint.')
    cfg.MODELS.mkdir(parents=True,exist_ok=True);cfg.RESULTS.mkdir(parents=True,exist_ok=True)
    histories={}
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        q=np.zeros(cfg.STATE_SHAPE+(4,),dtype=np.float64)
        env=SoloBattlesnake()
        snapshots=sorted(set([0,max(1,args.episodes//6),max(1,args.episodes//2),args.episodes]))
        demo_runs=[]
        def snapshot(episode):
            metric,frames=play(q,encode,choose,seed=800000,policy_seed=9,keep_frames=True)
            demo_runs.append((f'After {episode} episodes',frames,metric['reason']))
        snapshot(0)
        started=time.perf_counter();rows=[]
        for episode in range(args.episodes):
            progress=min(1.,episode/max(1,(args.episodes-1)*cfg.DECAY_FRACTION))
            epsilon=cfg.EPSILON_START+(cfg.EPSILON_END-cfg.EPSILON_START)*progress
            metrics=train_episode(env,q,epsilon,rng,seed*100000+episode,cfg.ALPHA,cfg.GAMMA)
            rows.append(dict(episode=episode+1,epsilon=epsilon,**metrics))
            if episode+1 in snapshots:snapshot(episode+1)
        save_table(q,cfg.MODELS/f'q_seed_{seed}.npz',seed,args.episodes)
        write_csv(cfg.RESULTS/f'train_seed_{seed}.csv',rows)
        animate_runs(demo_runs,cfg.RESULTS/f'training_seed_{seed}.gif')
        histories[seed]=rows
        print(f'Seed {seed}: {args.episodes} episodes in {time.perf_counter()-started:.1f}s; '
              f'{np.count_nonzero(np.any(q!=0,axis=-1))}/144 Q-table rows changed.',flush=True)
    training_plot(histories,cfg.RESULTS/'training_curves.png')
    (cfg.RESULTS/'training_manifest.json').write_text(json.dumps(dict(settings=experiment_settings(),
        episodes=args.episodes,seeds=args.seeds,alpha=cfg.ALPHA,gamma=cfg.GAMMA,
        epsilon_start=cfg.EPSILON_START,epsilon_end=cfg.EPSILON_END,decay_fraction=cfg.DECAY_FRACTION,
        demonstration_seed=800000),indent=2))


def evaluate_cli(encode, choose):
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[0])
    parser.add_argument('--games',type=int,default=cfg.EVAL_GAMES)
    args=parser.parse_args()
    if args.games<1:raise ValueError('Use at least one evaluation game.')
    cfg.RESULTS.mkdir(parents=True,exist_ok=True)
    results=[]
    random_runs={}
    for game in range(args.games):
        seed=cfg.EVAL_SEED_START+game
        metric,frames=play(None,None,None,seed,seed,keep_frames=game<3)
        results.append(dict(model='Random',training_seed=-1,game_seed=seed,**metric))
        if game<3:random_runs[game]=(metric,frames)
    for model_seed in args.seeds:
        q=load_table(cfg.MODELS/f'q_seed_{model_seed}.npz')
        original=q.copy()
        for game in range(args.games):
            seed=cfg.EVAL_SEED_START+game
            metric,frames=play(q,encode,choose,seed,seed,keep_frames=game<3)
            results.append(dict(model='Q-learning',training_seed=model_seed,game_seed=seed,**metric))
            if game<3 and model_seed==args.seeds[0]:
                baseline,baseline_frames=random_runs[game]
                animate_runs([('Random policy',baseline_frames,baseline['reason']),
                              ('Q-learning',frames,metric['reason'])],cfg.RESULTS/f'evaluation_game_{game+1}.gif')
        if not np.array_equal(q,original):raise RuntimeError('Evaluation must not update Q.')
    write_csv(cfg.RESULTS/'evaluation_games.csv',results)
    by_run=[]
    for name,seed in sorted(set((r['model'],r['training_seed']) for r in results)):
        group=[r for r in results if r['model']==name and r['training_seed']==seed]
        by_run.append(dict(model=name,training_seed=seed,
            **{key:float(np.mean([r[key] for r in group])) for key in ['reward','steps','apples','truncated']}))
    write_csv(cfg.RESULTS/'evaluation_by_run.csv',by_run)
    print('Model / training seed | Return | Apples | Steps | Time-limit fraction')
    for row in by_run:
        print(f'{row["model"]} / {row["training_seed"]}: {row["reward"]:.2f} | {row["apples"]:.2f} | '
              f'{row["steps"]:.1f} | {row["truncated"]:.1%}')
    lines=['# Your evaluation','',f'{args.games} held-out environment seeds per policy; epsilon=0; no updates.',
        'The same reset seeds are used for every policy; later food layouts can diverge when actions differ.','',
        '| Policy | Training seed | Mean return | Mean apples | Mean steps | At time limit |',
        '|---|---:|---:|---:|---:|---:|']
    for r in by_run:lines.append(f'| {r["model"]} | {r["training_seed"]} | {r["reward"]:.2f} | {r["apples"]:.2f} | {r["steps"]:.1f} | {r["truncated"]:.1%} |')
    lines += ['', 'Training curves include exploration; evaluation results do not. Survival is capped at '
        f'{cfg.MAX_STEPS} turns. A time limit is not a win or a death. The compact state merges different '
        'board situations, so optimal-policy guarantees for a full finite MDP do not apply directly.', '',
        '![Training curves](training_curves.png)', '',
        f'![Training progression](training_seed_{args.seeds[0]}.gif)', '']
    for i in range(1,min(3,args.games)+1):lines += [f'![Held-out game {i}](evaluation_game_{i}.gif)','']
    (cfg.RESULTS/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
