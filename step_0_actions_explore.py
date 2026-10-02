"""Run the supplied exploration and inspect an example game trajectory.

Task: identify one observation, one action, and the resulting next position.
Left plot: one game's head trajectory; dot color indicates turn, not probability.
Right plot: action counts across ALL supplied games, not only the displayed game.
The printed transition table links each action to its resulting next head position.
The supplied dataset is from an agent whose implementation is not provided.
"""
from supplied_pipeline import explore

if __name__ == '__main__':
    explore()
