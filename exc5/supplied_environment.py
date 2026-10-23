"""Fast SOLO Battlesnake teaching environment, not the official engine.

Core move/health/growth/collision order follows the Battlesnake rules. Teaching
choices: random central-ish start, one uniformly respawned food, no opponents or
hazards, and a time limit. reset/step use a Gymnasium-like API without dependencies.
"""
import copy
import numpy as np
import settings as cfg


def safe_directions(observation):
    """One-step wall/body safety only, in ACTIONS order; no policy or action mask."""
    board, snake = observation['board'], observation['you']
    body = [(p['x'], p['y']) for p in snake['body']]
    flags = []
    for dx, dy in cfg.DELTAS:
        head = (body[0][0]+dx, body[0][1]+dy)
        flags.append(int(0 <= head[0] < board['width'] and 0 <= head[1] < board['height']
                         and head not in body[:-1]))
    return flags


class SoloBattlesnake:
    def __init__(self, size=cfg.BOARD_SIZE, max_steps=cfg.MAX_STEPS):
        if size < 5 or max_steps < 1:
            raise ValueError('Use a board >=5 and a positive step limit.')
        self.size, self.max_steps = size, max_steps
        self.finished = True

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        x, y = self.rng.integers(1, self.size-1, size=2)
        # Stacked initial segments match the engine's initial snake convention.
        self.body = [(int(x), int(y))]*3
        self.food = []
        self.health, self.turn, self.apples = cfg.MAX_HEALTH, 0, 0
        self.alive, self.finished = True, False
        self._spawn_food()
        return self.observation(), dict(apples=0, reason='reset')

    def _spawn_food(self):
        empty = [(x,y) for y in range(self.size) for x in range(self.size) if (x,y) not in self.body]
        self.food = [empty[int(self.rng.integers(len(empty)))]] if empty else []

    def observation(self):
        points = lambda values: [dict(x=x, y=y) for x,y in values]
        snake = dict(id='learner', name='Tabular learner', health=self.health,
                     body=points(self.body), head=dict(x=self.body[0][0], y=self.body[0][1]), length=len(self.body))
        return dict(game=dict(id=f'solo-{self.seed}'), turn=self.turn,
                    board=dict(width=self.size, height=self.size, food=points(self.food),
                               hazards=[], snakes=[copy.deepcopy(snake)] if self.alive else []), you=snake)

    def step(self, action: int | np.integer):
        if self.finished:
            raise RuntimeError('Call reset() after termination or truncation.')
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 4:
            raise ValueError('Action must be an integer in 0..3.')
        dx, dy = cfg.DELTAS[action]
        head = (self.body[0][0]+dx, self.body[0][1]+dy)
        self.body = [head]+self.body[:-1]
        self.health -= 1
        self.turn += 1
        ate = head in self.food
        if ate:
            self.health = cfg.MAX_HEALTH
            self.body.append(self.body[-1])  # Move first; duplicate NEW tail when growing.
            self.apples += 1
            self.food = []
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size):
            reason = 'wall'
        elif head in self.body[1:]:
            reason = 'body'
        elif self.health <= 0:
            reason = 'starvation'
        else:
            reason = 'alive'
        self.alive = reason == 'alive'
        board_full = self.alive and len(set(self.body)) == self.size**2
        terminated = not self.alive or board_full
        truncated = not terminated and self.turn >= self.max_steps
        if self.alive and not self.food and not board_full:
            self._spawn_food()
        self.finished = terminated or truncated
        reward = (cfg.DEATH_REWARD if not self.alive else
                  cfg.FOOD_REWARD if ate else cfg.STEP_REWARD)
        info = dict(apples=self.apples, ate=ate,
                    reason='board_full' if board_full else 'time_limit' if truncated else reason)
        return self.observation(), float(reward), bool(terminated), bool(truncated), info
