# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing as typing
from typing import Dict
from agent import choose_move
from gamestate import GameState
from step_0_state_attributes import make_training_example

# Step 1 enables recording; normal play and Step 4 do not write datasets.
recording_enabled = True
recording_seed = None
recorded_rows = []

def record_state(game_state: GameState, direction: str):
    """Store one example using the shared schema in step_0_state_attributes.py."""
    recorded_rows.append(make_training_example(
        game_state, direction, seed=recording_seed, label_source="rule_based_agent"
    ))

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "kittenzzz snakeuh",  # TODO: Your Battlesnake Username
        "color": "#FC8EAC",  # TODO: Choose color
        "head": "villain",  # TODO: Choose head
        "tail": "weight",  # TODO: Choose tail
    }


# start is called when your Battlesnake begins a game
def start(game_state: GameState):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: GameState):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: GameState) -> Dict:
    print(f"MOVE {game_state.turn}: {game_state}")
    next_move = {"move": choose_move(game_state)}
    
    if recording_enabled:
        record_state(game_state, next_move["move"])
        
    return next_move

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
