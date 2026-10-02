"""Student experiment settings. Change FUTURE to predict the next x actions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORDING = ROOT / 'data' / 'solo_trajectories.json.gz'
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
HISTORY = 8
FUTURE = 3
HIDDEN = 64
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SPLIT_SEED = 42
ACTIONS = ['up', 'down', 'left', 'right']
DELTAS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
FEATURES = ['head_x', 'head_y', 'food_dx', 'food_dy', 'health', 'length',
            'safe_up', 'safe_down', 'safe_left', 'safe_right']

if HISTORY < 1 or not 1 <= FUTURE <= 5:
    raise ValueError('Use HISTORY >= 1 and FUTURE between 1 and 5 (the baseline enumerates paths).')
