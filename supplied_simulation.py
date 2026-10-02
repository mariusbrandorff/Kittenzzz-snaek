"""Supplied solo movement model and uniform complete-path baseline.

Uses current food, tail movement, growth, health and collisions. Unknown food
spawns are not generated. This is a short-horizon approximation to the engine.
"""
from itertools import product
import numpy as np
from settings import ACTIONS, DELTAS


def advance(state, action):
    """Execute the requested action, without repairing illegal moves."""
    if not state['alive']:
        return dict(state)
    body = [tuple(p) for p in state['body']]
    dx, dy = DELTAS[int(action)]
    head = (body[0][0] + dx, body[0][1] + dy)
    moved = [head] + body[:-1]
    food = set(map(tuple, state['food']))
    health = state['health'] - 1
    if head in food:
        moved.append(moved[-1])  # Engine moves first, then duplicates its new tail.
        food.remove(head)
        health = 100
    alive = (0 <= head[0] < state['width'] and 0 <= head[1] < state['height']
             and head not in moved[1:] and health > 0)
    return dict(state, body=moved, food=sorted(food), health=health,
                length=len(moved), alive=alive, turn=state['turn'] + 1)


def simulate(state, actions):
    states = []
    for action in actions:
        state = advance(state, action)
        states.append(state)
    return states


def uniform_paths(state, future):
    """Return equally weighted complete surviving action sequences.

    Equal weight per complete path differs from uniform legal choices per turn.
    In the exceptional case with no surviving path, use all action sequences
    and let simulation show death. Never call the teacher or read future food.
    """
    paths = [([], state)]
    for _ in range(future):
        extended = []
        for actions, current in paths:
            for action in range(len(ACTIONS)):
                next_state = advance(current, action)
                if next_state['alive']:
                    extended.append((actions + [action], next_state))
        paths = extended
    fallback = not paths
    sequences = (np.array(list(product(range(4), repeat=future)), dtype=np.int64)
                 if fallback else np.array([p[0] for p in paths], dtype=np.int64))
    probabilities = np.eye(4)[sequences].mean(axis=0)
    return sequences, probabilities, fallback


def sample_sequences(probabilities, rng, count=100):
    """The simple NN assumes independent future actions given the history."""
    return np.stack([rng.choice(4, count, p=p / p.sum()) for p in probabilities], axis=1)
