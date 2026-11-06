"""Play in the original Battlesnake engine using the DQN trained in dqn_train.py.

This file is complete. After training, start the agent from this folder:
    python dqn_battlesnake.py --model results/dqn_seed0_steps20000/dqn.zip --port 8006
Then start a solo game in a second terminal using the CLI command in the README.
"""
import argparse
from stable_baselines3 import DQN
from dqn_train import prepare_observation
from hisss_tools import MOVES, serve


def choose_move(state, model):
    """Choose one move for the current board. Return a Battlesnake direction string.

    1. The engine sends the board as a dictionary (state). Use prepare_observation
       from the training file to turn it into the seven numbers the network learned from.
    2. Ask the loaded model to predict a move. deterministic=True turns off random
       exploration: DQN picks a direction with the highest predicted Q-value.
       predict also returns a policy state, which this feed-forward agent does not use.
    3. Translate the action ID into the engine's vocabulary, e.g. 1 means "right".
       The server sends that direction back; the engine advances the game.
    """
    action, _ = model.predict(prepare_observation(state), deterministic=True)
    return MOVES[int(action)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/dqn_seed0_steps20000/dqn.zip")
    parser.add_argument("--port", type=int, default=8006)
    args = parser.parse_args()
    # Load once at startup. Every /move request uses these same weights.
    model = DQN.load(args.model, device="cuda")
    # The supplied server calls choose_move with the new board and sends the reply.
    serve(lambda state: choose_move(state, model), args.port)
