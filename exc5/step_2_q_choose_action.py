"""Task: choose among all four actions with an epsilon-greedy policy.

With probability epsilon choose uniformly from 0..3; otherwise select a maximum
Q-value. Break ties randomly using rng. q_values is a length-4 NumPy array.
rng is a supplied np.random.Generator (use it rather than global randomness).
"""
import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def choose_action(q_values: NDArray, epsilon: float, rng: Generator):
    if (rng.random() < epsilon):
        return rng.integers(0, 4)
    else:
        highest_scorers = np.flatnonzero(q_values==q_values.max())
        return rng.choice(highest_scorers)

if __name__ == '__main__':
    values = np.array([0., 2., -1., 2.])
    for epsilon in [1., 0.]:
        rng = np.random.default_rng(5)
        counts = np.bincount([choose_action(values, epsilon, rng) for _ in range(1000)], minlength=4)
        print(f'epsilon={epsilon}: counts for up/down/left/right = {counts}')
    print('Explain how these counts reflect exploration and exploitation.')
