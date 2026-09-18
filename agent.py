import random
from typing import Dict, List


from gamestate import GameState, Vector

# Step 1 enables recording; normal play and Step 4 do not write datasets.
recording_enabled = False
recording_seed = None
recorded_rows = []


def record_state(game_state: typing.Dict, direction: str):
    """Store one example using the shared schema in step_0_state_attributes.py."""
    recorded_rows.append(make_training_example(
        game_state, direction, seed=recording_seed, label_source="rule_based_agent"
    ))


def vector_direction(v1: Vector, v2: Vector) -> str:
    if (v2.x < v1.x): return "left"
    if (v2.x > v1.x): return "right"
    if (v2.y < v1.y): return "down"
    if (v2.y > v1.y): return "up"
    return

def collision(v1: Vector, segments: List[Vector], is_move_safe: Dict):
    for segment in segments:
        if (segment.distance(v1) == 1):
            is_move_safe[vector_direction(v1, segment)] = False

    return is_move_safe

def choose_move(game_state: GameState) -> str:

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    my = game_state.you

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = my.head  # Coordinates of your head
    my_neck = my.neck  # Coordinates of your "neck"
    
    # Step 1 - Prevent your Battlesnake from moving out of bounds
    board = game_state.board

    if my_head.y == board.height-1:
        is_move_safe["up"] = False

    if my_head.x == board.width-1:
        is_move_safe["right"] = False

    if my_head.y == 0:
        is_move_safe["down"] = False
        
    if my_head.x == 0:
        is_move_safe["left"] = False

    # Step 2 - Prevent your Battlesnake from colliding with itself
    # my_body = game_state['you']['body']
    is_move_safe = collision(my_head, my.body[1:], is_move_safe)

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    # opponents = game_state['board']['snakes']

    opponents = [
    snake for snake in game_state.board.snakes
    if snake.name != my.name
    ]

    for opponent in opponents:
        is_move_safe = collision(my_head, opponent.body, is_move_safe)

    # Are there any safe moves left? 
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            print(move)
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state.turn}: No safe moves detected! Imma KMS")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    # food = game_state['board']['food']

    print(f"MOVE {game_state.turn}: {next_move}")
    return next_move
