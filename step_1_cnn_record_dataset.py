"""Step 1: record your configured rule bot in regular solo simulator games."""
import argparse
import copy
import logging
from pathlib import Path
import random
import subprocess
import threading

import pandas as pd
from werkzeug.serving import make_server

import cnn_config as config
from step_0_cnn_state_attributes import (
    BOARD_WIDTH, BOARD_HEIGHT, cnn_load_dataframe, cnn_make_training_example,
)
from server import create_app


def cnn_record_games(games=10, data=None, seed=100, seconds=120):
    """Wrap the configured bot's move function; no recording hooks are required."""
    if games < 1 or seconds <= 0:
        raise ValueError('Game count and time limit must be positive.')
    engine = Path(config.ENGINE_PATH).resolve()
    path = Path(data if data is not None else config.DATA_PATH)
    if not engine.is_file():
        raise FileNotFoundError(f'Set ENGINE_PATH in cnn_config.py: {engine}')
    df = cnn_load_dataframe(path) if path.exists() else pd.DataFrame()
    if len(df) and 'seed' in df:
        seed = max(seed, int(df.seed.max()) + 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    recording_seed = seed
    agent = config.RULE_BASED_AGENT

    def cnn_record_move(state):
        snapshot = copy.deepcopy(state)  # Preserve the input even if the bot mutates it.
        decision = agent.move(state)
        rows.append(cnn_make_training_example(snapshot, decision['move'],
                    seed=recording_seed, label_source=agent.__name__))
        return decision

    handlers = {name: getattr(agent, name) for name in ('info', 'start', 'end')}
    handlers['move'] = cnn_record_move
    server = make_server('127.0.0.1', 0, create_app(handlers))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def cnn_save_dataset():
        nonlocal df
        if not rows:
            return
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df = df.drop_duplicates(['game_id', 'turn'], keep='last').reset_index(drop=True)
        temporary = path.with_suffix('.tmp')
        df.to_json(temporary, orient='table', index=False, indent=2)
        temporary.replace(path)
        df.drop(columns=['state', 'tensor', 'channels']).to_csv(path.with_suffix('.csv'), index=False)
        rows.clear()

    try:
        for number in range(games):
            recording_seed = seed + number
            random.seed(recording_seed)
            print(f'Game {number+1}/{games}, seed {recording_seed}', flush=True)
            command = [str(engine), 'play', '-g', 'solo', '-W', str(BOARD_WIDTH), '-H', str(BOARD_HEIGHT),
                       '--seed', str(recording_seed), '--timeout', '500', '--name', 'Rule-based agent',
                       '--url', f'http://127.0.0.1:{server.server_port}']
            with path.with_suffix('.engine.log').open('a', encoding='utf8') as log:
                subprocess.run(command, check=True, timeout=seconds, stdout=log, stderr=log)
            # cnn_save_dataset()
            print(f'Saved {len(df)} decisions to {path}', flush=True)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        cnn_save_dataset()  # Keep recorded decisions even if interrupted mid-game.
    return df


def cnn_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--data', default=config.DATA_PATH)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--seconds', type=float, default=120)
    args = parser.parse_args()
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    try:
        cnn_record_games(args.games, args.data, args.seed, args.seconds)
    except KeyboardInterrupt:
        print('Stopped. Collected examples have been saved.')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f'Recording failed: {error}. See the dataset engine log.\n')


if __name__ == '__main__':
    cnn_main()
