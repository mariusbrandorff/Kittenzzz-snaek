"""Small, explicit experiments. Change one setting at a time and retrain."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
RESULTS = ROOT / 'results'
BOARD_SIZE = 7
MAX_STEPS = 250  # External time limit, not a death.
MAX_HEALTH = 100
EPISODES = 6000
ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
DECAY_FRACTION = 0.8
FOOD_REWARD = 1.0
DEATH_REWARD = -1.0
STEP_REWARD = -0.01
EVAL_GAMES = 100
EVAL_SEED_START = 1000000
ACTIONS = ['up', 'down', 'left', 'right']
DELTAS = [(0, 1), (0, -1), (-1, 0), (1, 0)]
# Four safe-direction flags and two three-valued food directions.
STATE_SHAPE = (2, 2, 2, 2, 3, 3)
ENCODING_VERSION = 'safe4_food_direction2_v1'
