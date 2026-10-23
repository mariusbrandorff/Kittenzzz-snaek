"""Supplied: evaluate without learning or exploration, compare to random, animate.

Tasks: interpret apples, survival, returns and time limits. Compare to training
with exploration. Relate failure cases to information lost by your state encoding.
"""
if __name__ == '__main__':
    from step_1_q_state import encode_state
    from step_2_q_choose_action import choose_action
    from supplied_experiment import evaluate_cli
    evaluate_cli(encode_state, choose_action)
