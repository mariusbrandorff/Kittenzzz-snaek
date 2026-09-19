"""Shared Battlesnake attributes and training examples, independent of any agent.

Edit state_to_attributes() to select recorded features and INPUT_COLUMNS to
select numeric network inputs. Recording, training and live inference all use
this schema. Other agents can import these helpers without importing a bot or
starting the simulator.
"""
import copy
from pathlib import Path
from gamestate import GameState

import pandas as pd

ACTIONS = ["up", "down", "left", "right"]
DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

INPUT_COLUMNS = (["board_width", "board_height", "you_x", "you_y"])


def state_to_attributes(gamestate: GameState):
    """OPTIONAL TASK: add attributes here; add numeric inputs to INPUT_COLUMNS.

    Step 3 calls this exact function during live play. Safe means inside the
    board and outside all current body cells. Tails are treated as occupied.
    This does not predict future traps or simultaneous head-to-head collisions.
    """
    # The regular server wraps move requests in GameState, while the neural
    # agent's standalone server passes the decoded request dictionary through.
    attributes = {
        "board_width": gamestate.board.width,
        "board_height": gamestate.board.height,
        "you_x": gamestate.you.head.x,
        "you_y": gamestate.you.head.y,
    }
    
    # add further attributes you want to track. For each attribute, also add the
    #  name of the attribute in the list called INPUT_COLUMNS above.


    # add the following two code lines at the bottom of the action selection
    #  function of your rule-based agent right before it returns the next_move of your snake.
    #   ```
    #       if recording_enabled:
    #           record_state(game_state, next_move)
    #   ```
    #  This will ensure that the following files work as expected.

    return attributes


def make_training_example(gamestate : GameState, direction, *, label_source, seed=None):
    """Build an independent state/action snapshot for any agent's dataset.

    The caller owns the rows and decides when to store them. For example:
    rows.append(make_training_example(state, direction, label_source="my_agent"))
    """
    row = {
        **state_to_attributes(gamestate),
        "game_id": gamestate.id,
        "turn": gamestate.turn,
        "seed": seed,
        "direction": direction,
        "label_source": label_source,
        "state": gamestate,
    }
    return copy.deepcopy(row)


def load_dataframe(path):
    """Restore the saved pandas DataFrame, including its nested apple lists."""
    df = pd.read_json(Path(path), orient="table")
    required = set(INPUT_COLUMNS + ["game_id", "turn", "direction", "state"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}. Record a new dataset after changing attributes.")
    if df.empty:
        raise ValueError("The DataFrame is empty. Record some agent decisions in Step 1 first.")
    if not df["direction"].isin(ACTIONS).all():
        raise ValueError("Every label must be up, down, left or right.")
    if df.duplicated(["game_id", "turn"]).any():
        raise ValueError("The dataset contains duplicate game/turn pairs.")
    return df
