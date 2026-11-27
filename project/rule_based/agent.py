import sys
sys.path.insert(0, '../project')
import gamestate
import math
import random
from dataclasses import dataclass, field
from contextlib import closing

from typing import Literal
from gamestate import GameState
from iAgent import IAgent, InfoReturn

from hisss import BattleSnakeGame
from hisss_tools import advance, rollout_moves, MOVES, from_battlesnake, replay, rollout_moves, seed_native, serve

@dataclass
# Search tree
class Node:
    parent: "Node" = None
    depth: int = 0
    visits: int = 0
    total: float = 0.
    children: dict[int, "Node"] = field(default_factory=dict)  # action -> node

# UCB score function
def score(node: Node, C=1.4):
    total = node.total
    visits = node.visits
    parent_visits = node.parent.visits
    if visits == 0:
        return float("inf")
    return total/visits + C * math.sqrt(math.log(parent_visits)/visits) #  UCB = W/N + C × √(ln(P)/N)

def selection(root: Node, game: BattleSnakeGame, horizon: int, C=1.4) -> tuple[Node, int]:
    node = root
    prefix = 0

    #Stop conditions
    def game_terminal():
        return game.is_terminal()
    def reached_horizon():
        return node.depth >= horizon # As long as we have not reached the horizon we keep going
    def action_has_4_children(): # 4 actions = the four directions
        return len(node.children.keys()) == 4

    while (not game_terminal()) and not reached_horizon() and action_has_4_children():
            if node.visits == 0: break

            # Select the best child (highest UCB score)
            action, child = max(
                node.children.items(),
                key=lambda item: score(item[1], C)
            )
            node = child
            rewards, _, _ = advance(game, action)
            prefix += rewards
        
    return node, prefix

def exploration(node: Node, game: BattleSnakeGame, horizon: int, rng: random.Random) -> tuple[Node, float]:
    if game.is_terminal() or node.depth >= horizon:
        return node, 0

    # Finds every child/action that have not yeat been explored. Picks a random action amongs those 
    untried_actions = [
        a for a in range(0, 4)
        if a not in node.children.keys()
    ]
    action = rng.choice(untried_actions)

    # New child node added to as the parent's children
    new_node = Node()
    new_node.parent = node
    new_node.depth = node.depth + 1
    node.children[action] = new_node

    rewards, _, _ = advance(game, action)

    return new_node, rewards

def simulation(game: BattleSnakeGame, remaining_horizon: int, rng: random.Random) -> int:
    to_return = 0
    while not game.is_terminal() and remaining_horizon > 0:
            eligible_moves = rollout_moves(game)
            action = rng.choice(eligible_moves)
            rewards, _, _ = advance(game, action)
            to_return += rewards
            remaining_horizon = remaining_horizon - 1
    return to_return

def backpropagation(node: Node, value: int):
    current_node = node
    while current_node != None:
        current_node.visits += 1
        current_node.total += value
        current_node = current_node.parent

def search(game: BattleSnakeGame, iterations, rng, horizon=12):
    if iterations < 4 or horizon < 1 or game.is_terminal():
        raise ValueError("Search needs a live board, at least four iterations and a positive horizon.")
    root = Node()
    for _ in range(iterations):
        with closing(game.get_copy()) as trial:
            node, prefix = selection(root, trial, horizon)
            leaf, reward = exploration(node, trial, horizon, rng)
            tail = simulation(trial, horizon-leaf.depth, rng)
            backpropagation(leaf, prefix + reward + tail)
    return max(root.children, key=lambda a: root.children[a].total / root.children[a].visits)

    
def battlesnake_move(state, iterations, rng: random.Random, horizon=12):
    
    with closing(from_battlesnake(state)) as game:
        return MOVES[search(game, iterations, rng, horizon)]
            
class RuleBasedAgent(IAgent):
    def info(self) -> InfoReturn:
        info =  super().info()
        info.color = "#0FF"
        return info
    
    def start(self, gamestate: GameState):
        return super().start(gamestate)

    def move(self, gamestate: GameState) -> Literal['up'] | Literal['right'] | Literal['down'] | Literal['left']:
        return battlesnake_move(
            gamestate.raw_state, 
            iterations=64, 
            rng=random.Random()
        )    

    def end(self, gamestate: GameState):
        return super().end(gamestate)

    def dry(self):
        return super().dry()