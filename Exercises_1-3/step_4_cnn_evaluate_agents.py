"""Step 4: compare solo survival of the CNN, rule bot, and optionally Exercise 2's MLP."""
import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import threading

import pandas as pd
from werkzeug.serving import make_server
import cnn_config as config
from step_0_cnn_state_attributes import BOARD_WIDTH, BOARD_HEIGHT
from step_3_cnn_neural_agent import CNNAgent, cnn_create_app
from previous_exercise_agent import cnn_load_previous_agent


def cnn_game_result(game, seed, names, starts, ends):
    row = {'Game': game, 'Seed': seed}
    for name in names:
        if name not in starts or name not in ends:
            raise RuntimeError(f'Missing start/end data for {name}; this is not a completed game.')
        if starts[name]['game']['id'] != ends[name]['game']['id']:
            raise RuntimeError(f'Mismatched game callbacks for {name}.')
        row[f'{name} time steps'] = ends[name]['turn']
        row[f'{name} points (apples)'] = ends[name]['you']['length'] - starts[name]['you']['length']
    longest = max(row[f'{name} time steps'] for name in names)
    leaders = [name for name in names if row[f'{name} time steps'] == longest]
    row['Longest survival'] = leaders[0] if len(leaders) == 1 else 'Tie: ' + ', '.join(leaders)
    row['CNN - Rule time steps'] = row['CNN time steps'] - row['Rule-based time steps']
    if 'MLP' in names:
        row['CNN - MLP time steps'] = row['CNN time steps'] - row['MLP time steps']
    return row


def cnn_evaluate(model=None, games=10, seed_start=1000, seconds=120,
                 mlp_model=None, mlp_project=None, output=None):
    if games < 1 or seconds <= 0:
        raise ValueError('Game count and time limit must be positive.')
    engine = Path(config.ENGINE_PATH).resolve()
    if not engine.is_file():
        raise FileNotFoundError(f'Set ENGINE_PATH in cnn_config.py: {engine}')
    cnn = CNNAgent(model)
    rule_based_agent = config.RULE_BASED_AGENT
    used_seeds = set(cnn.recorded_seeds)
    agents = [('CNN', cnn.cnn_move, None, None, None),
              ('Rule-based', rule_based_agent.move, rule_based_agent.start,
               rule_based_agent.end, rule_based_agent.info)]
    if mlp_model:
        project = mlp_project or Path(__file__).resolve().parent.parent/'Exercise 2'
        mlp = cnn_load_previous_agent(project, mlp_model)
        agents.append(('MLP', mlp.move, None, None, None))
        metadata = Path(mlp_model).with_suffix('.json')
        if not metadata.is_file():
            raise FileNotFoundError('Keep the Exercise 2 model .json sidecar to check its recorded seeds.')
        used_seeds.update(json.loads(metadata.read_text(encoding='utf8'))['recorded_seeds'])
    if used_seeds & set(range(seed_start, seed_start + games)):
        raise ValueError('Evaluation seeds overlap recorded seeds. Choose another --seed-start.')

    starts, ends, results = {}, {}, []
    names = [entry[0] for entry in agents]
    path = Path(output if output is not None else config.RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cnn_scored_app(name, move, start, end, info):
        def cnn_on_start(state):
            starts[name] = state
            if start: start(state)

        def cnn_on_end(state):
            ends[name] = state
            if end: end(state)

        return cnn_create_app(move, start=cnn_on_start, end=cnn_on_end, info=info)

    # One HTTP server thread per agent, just like Exercise 2.
    servers, threads = [], []
    try:
        for entry in agents:
            server = make_server('127.0.0.1', 0, cnn_scored_app(*entry))
            servers.append(server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            threads.append(thread)
        for game in range(games):
            starts.clear(); ends.clear()
            seed = seed_start + game
            for name, server in zip(names, servers):
                random.seed(seed)
                print(f'Seed {seed} ({game+1}/{games}): {name} solo game', flush=True)
                command = [str(engine), 'play', '-g', 'solo', '-W', str(BOARD_WIDTH), '-H', str(BOARD_HEIGHT),
                           '--seed', str(seed), '--timeout', '500', '--name', name,
                           '--url', f'http://127.0.0.1:{server.server_port}']
                with path.with_suffix('.engine.log').open('a', encoding='utf8') as log:
                    log.seek(0, 2)
                    offset = log.tell()
                    subprocess.run(command, check=True, timeout=seconds, stdout=log, stderr=log)
                # A failed HTTP move makes the engine choose a fallback, not the agent.
                # Exclude such runs instead of reporting a misleading survival score.
                with path.with_suffix('.engine.log').open(encoding='utf8') as log:
                    log.seek(offset)
                    if 'WARN' in log.read():
                        raise RuntimeError('Simulator warning: this game is not a valid evaluation. See the engine log.')
            results.append(cnn_game_result(game+1, seed, names, starts, ends))
            pd.DataFrame(results).to_csv(path, index=False)
    finally:
        for server, thread in zip(servers, threads):
            server.shutdown()
            thread.join()
        for server in servers:
            server.server_close()

    table = pd.DataFrame(results)
    print('\nIndependent solo games; points = apples eaten.')
    print(table.to_string(index=False))
    print('\nMean survival:')
    for name in names:
        print(f"  {name}: {table[f'{name} time steps'].mean():.1f} steps")
    print(f'Saved {path}')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=config.MODEL_PATH)
    parser.add_argument('--games', type=int, default=10, help='Seeds; each agent plays one solo game per seed')
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--seconds', type=float, default=120)
    parser.add_argument('--mlp-model', help='Optional trained checkpoint from Exercise 2')
    parser.add_argument('--mlp-project', help='Exercise 2 source folder (defaults to the sibling Exercise 2 folder)')
    parser.add_argument('--output', default=config.RESULTS_PATH)
    args = parser.parse_args()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    try:
        cnn_evaluate(args.model, args.games, args.seed_start, args.seconds,
                     args.mlp_model, args.mlp_project, args.output)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Evaluation failed: {error}\n')
