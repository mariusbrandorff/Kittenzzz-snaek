"""Step 0: encode the whole board as a (channels, height, width) tensor.

Recording and live inference use the same function. Add new channels here,
then re-encode the saved states (or record again) and retrain after changes.
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cnn_config as config
from gamestate import GameState, Vector

ACTIONS = ['up', 'down', 'left', 'right']
BOARD_WIDTH = BOARD_HEIGHT = 11
CHANNELS = ['my_head', 'my_neck', 'my_body', 'food', 'health', 'board','enemy_snake_body','enemy_snake_head']
INPUT_SHAPE = (len(CHANNELS), BOARD_HEIGHT, BOARD_WIDTH)
ENCODING_VERSION = 3  # Whole-board view, without opponents or hazards.


def cnn_state_to_attributes(state: GameState):
    """Return float32 input[channel, y, x] in absolute board coordinates.

    Occupancy planes contain 0/1. Health is repeated over the board as health/100.
    Up increases y; use origin='lower' when plotting this top-down board view.
    The full 11x11 board is included, with no crop or head-relative translation.
    The board plane contains 1, distinguishing board cells from zero padding.
    All current body cells, including the head and tail, are marked as occupied.
    """
    board, snake = state.board, state.you
    if (board.height, board.width) != (BOARD_HEIGHT, BOARD_WIDTH):
        raise ValueError(f'This exercise expects a {BOARD_WIDTH}x{BOARD_HEIGHT} board.')
    inputs = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    def cnn_mark(channel: str, positions: list[Vector], score: float = 1):
        for p in positions:
            x, y = p.x, p.y
            if not (0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT):
                raise ValueError('Only encode live board positions inside the board.')
            inputs[CHANNELS.index(channel), y, x] = score

    cnn_mark('my_head', [snake.head])
    cnn_mark('my_neck', [snake.body[1]])
    cnn_mark('my_body', snake.body)
    cnn_mark('food', board.food)
    enemies = [s for s in board.snakes if s.id != snake.id]
    cnn_mark('enemy_snake_head', [e.head for e in enemies])
    cnn_mark('enemy_snake_body', [e.body for e in enemies])
    inputs[CHANNELS.index('health')].fill_(snake.health / 100.0)
    inputs[CHANNELS.index('board')].fill_(1)
    return inputs


def cnn_make_training_example(game_state: GameState, direction, *, label_source, seed=None):
    if direction not in ACTIONS:
        raise ValueError(f'Unknown direction: {direction}')
    return copy.deepcopy({
        # JSON stores nested lists; Step 2 restores a float32 tensor.
        'tensor': cnn_state_to_attributes(game_state).tolist(),
        'channels': CHANNELS.copy(), 'encoding_version': ENCODING_VERSION,
        'game_id': game_state.id, 'turn': game_state.turn,
        'seed': seed, 'direction': direction, 'label_source': label_source,
        'state': game_state,
    })


def cnn_dataframe_to_tensors(df):
    """Stack DataFrame examples into (N,C,H,W), with integer action labels."""
    required = {'tensor', 'channels', 'encoding_version', 'game_id', 'turn', 'direction', 'state'}
    if required - set(df.columns):
        raise ValueError('This is not a CNN dataset. Record a new dataset with Step 1.')
    if df.empty:
        raise ValueError('Record some decisions in Step 1 first.')
    if not df.direction.isin(ACTIONS).all() or df.duplicated(['game_id', 'turn']).any():
        raise ValueError('Invalid direction labels or duplicate game/turn pairs.')
    if not df.channels.apply(lambda names: list(names) == CHANNELS).all() or not (
            df.encoding_version == ENCODING_VERSION).all():
        raise ValueError('Recorded encoding differs from Step 0. Re-encode the saved states or record a new dataset.')
    arrays = np.asarray(df['tensor'].tolist(), dtype=np.float32)
    if arrays.shape != (len(df), *INPUT_SHAPE) or not np.isfinite(arrays).all():
        raise ValueError(f'Expected finite inputs shaped (N, {INPUT_SHAPE}).')
    x = torch.from_numpy(arrays)
    y = torch.tensor(df.direction.map(ACTIONS.index).to_numpy(), dtype=torch.long)
    return x, y


def cnn_load_dataframe(path):
    df = pd.read_json(Path(path), orient='table')
    cnn_dataframe_to_tensors(df)  # Refuse incompatible encodings before appending/training.
    return df


def cnn_reencode_dataframe(df):
    """Reuse the exact states, labels and game split; never call the teacher again."""
    if 'state' not in df:
        raise ValueError('Re-encoding requires the original game state in every row.')
    result = df.copy(deep=True)
    result['tensor'] = [cnn_state_to_attributes(state).tolist() for state in df.state]
    result['channels'] = [CHANNELS.copy() for _ in range(len(df))]
    result['encoding_version'] = ENCODING_VERSION
    cnn_dataframe_to_tensors(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-encode saved states without playing new games.')
    parser.add_argument('--source', default=config.PROJECT_DIR/'data'/'rule_tensor_moves.json')
    parser.add_argument('--output', default=config.DATA_PATH)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        parser.error(f'{path} already exists. Choose a new --output filename.')
    df = cnn_reencode_dataframe(pd.read_json(args.source, orient='table'))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient='table', index=False, indent=2)
    df.drop(columns=['state', 'tensor', 'channels']).to_csv(path.with_suffix('.csv'), index=False)
    print(f'Re-encoded {len(df)} decisions into {path}. Original data is unchanged.')
