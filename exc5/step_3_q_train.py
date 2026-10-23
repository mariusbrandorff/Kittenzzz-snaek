"""Tasks: implement a one-step Q update, then connect the agent/environment loop.

The supplied runner creates a zero-initialized Q-table and manages episodes,
epsilon decay, logging, plots, checkpoint files and videos. You implement learning.
No labelled actions, neural network or pre-trained policy is provided.
"""
import numpy as np
from supplied_environment import SoloBattlesnake
from step_1_q_state import encode_state
from step_2_q_choose_action import choose_action


def update_q(q, state, action, reward, next_state, terminated, alpha, gamma):
    """Modify ONLY q[state][action] in place. Do not make a copy of q or return anything.

    Target = reward at a true terminal state; otherwise include the discounted
    maximum next-state Q-value. A time-limit truncation still bootstraps.
    """
    
    target = reward if terminated else np.abs(reward + gamma * q[next_state].max())
    #new_q = q[state] + alpha * target
    q[state][action] = q[state][action] + alpha * (target - q[state][action])


def train_episode(env: SoloBattlesnake, q, epsilon, rng, seed, alpha, gamma):
    """Use env.reset(seed=seed), then repeatedly call env.step(action).

    step returns (next_observation, reward, terminated, truncated, info).
    
    Encode states, select actions, update Q, and advance to the next observation.
    Stop on terminated OR truncated; pass only terminated to update_q.
    Return dict(reward=episode_reward, steps=steps, apples=info['apples'],
                terminated=terminated, truncated=truncated, reason=info['reason']).
    episode_reward is the undiscounted sum for reporting; gamma is used in updates.
    """
    env.reset(seed = seed)
    steps = []
    rewards = []
    state = encode_state(env.reset(seed = seed)[0]) #Encode states
    while 1:
        action = choose_action(q[state],epsilon,rng) #Select actions
        steppah = env.step(action)
        steps.append(steppah)
        reward = steppah[1]
        rewards.append(reward)
        terminated = steppah[2]
        truncated = steppah[3]
        info = steppah[4]
    
        update_q(q, state, action, reward, encode_state(steppah[0]), terminated, alpha, gamma) #Update Q
        if terminated or truncated:
            return {
                "reward": sum(rewards),
                "steps": len(rewards),
                "apples": info['apples'],
                "terminated": terminated,
                "truncated": truncated,
                "reason": info['reason']
            }
    # # TODO: connect the interaction cycle; update after EVERY transition.



if __name__ == '__main__':
    from supplied_experiment import train_cli
    train_cli(encode_state, choose_action, train_episode)
