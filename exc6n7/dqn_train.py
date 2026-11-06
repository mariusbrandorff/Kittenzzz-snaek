"""Learn which direction to move in solo Battlesnake by playing games in HISSS.

Run: python dqn_train.py --seed 0 --steps 20000
Complete the two marked steps in train(). All other code in this file is supplied.
Deep reinforcement learning is pretty easy to do,
once you figured out the state representation and the rewards.
"""
import argparse
import csv
from math import ceil
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from hisss_tools import DELTAS, TURN_LIMIT, new_game, board, advance, replay


def prepare_observation(state):
    """Convert one Battlesnake board dictionary into the seven inputs of our network.

    # todo: you can replace this logic with something you wish, I just explain this as a minimal version here
    1. Find our head and look one cell up, right, down and left, in that order.
       For each direction, record 1 if the next cell is inside the board and
       clear of our body, otherwise 0. These flags describe immediate danger;
       the agent can still choose any of the four actions.
    2. Find the nearest apple by Manhattan distance. Record how far it is from
       the head in x and y, relative to the board size. If no apple is present,
       using the head as the default gives an offset of (0, 0).
    3. Add the fraction of health remaining and return the seven numbers.
       We reuse this function in dqn_battlesnake.py: the trained network expects
       exactly the same inputs in exactly the same order when playing a real game.
    """
    b, s = state["board"], state["you"]
    x, y = s["head"]["x"], s["head"]["y"]
    body = [(p["x"], p["y"]) for p in s["body"]]
    # HISSS may omit repeated tail cells; the official API includes them.
    # A single tail tip vacates its cell on the next move. A stacked tip stays blocked.
    blocked = body if len(body) < s["length"] else body[:-1]
    safe = [0 <= x+dx < b["width"] and 0 <= y+dy < b["height"] and (x+dx,y+dy) not in blocked
            for dx,dy in DELTAS]
    food = min(b["food"], key=lambda p:(abs(p["x"]-x)+abs(p["y"]-y), p["x"], p["y"]),
               default=s["head"])
    return np.array([*safe, (food["x"]-x)/(b["width"]-1), (food["y"]-y)/(b["height"]-1),
                     s["health"]/100], dtype=np.float32)


class SoloEnv(gym.Env):
    """Stable Baselines3 requires the game to use the Gymnasium framework.
     Here we wrap the efficient version of Battlesnake, called "HISSS"
     such that it complies with Gymnasium's reset and step methods.

    The agent chooses one of four direction IDs and receives seven observation
    values (todo: this value needs to be adjusted when you change the prepare_observations).
    HISSS applies the game rules; this class just forwards information in the right format.
    """
    action_space = gym.spaces.Discrete(4)
    # todo: update the shape (third parameter) to match whatever you return in prepare_observations
    observation_space = gym.spaces.Box(-1., 1., (7,), dtype=np.float32)
    def __init__(self):
        self.game, self.history, self.steps = None, [], 0

    def reset(self, *, seed=None, options=None):
        """Start a new episode and return (first observation, extra info)."""
        super().reset(seed=seed)
        self.close()
        # Draw a new HISSS seed for each game from the seeded Gymnasium generator.
        # Only score and apples restart here; steps and history span all episodes.
        self.game = new_game(self.np_random.integers(2**31-1))
        self.score, self.apples = 0., 0
        return prepare_observation(board(self.game)), {}

    def step(self, action):
        """Apply one move and return (next observation, reward, terminated, truncated, info)."""
        reward, dead, ate = advance(self.game, action)
        self.steps += 1
        self.score += reward
        self.apples += ate
        turns = self.game.turns_played
        capped = not dead and turns >= TURN_LIMIT
        # Both endings finish an episode. Only death rules out future rewards;
        # at the artificial turn limit DQN can still use the final board's Q-value.
        if dead or capped:
            self.history.append((self.steps, self.score, self.apples, turns))
        # After death, seven zeros keep the expected shape; there is no next decision.
        # todo: update the shape (first parameter) to match whatever you return in prepare_observations
        obs = np.zeros(7, np.float32) if dead else prepare_observation(board(self.game))
        return obs, reward, dead, capped, {}

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None


def train(seed, steps, folder):
    """Train one network and return (trained model, history of completed episodes).

    seed selects the random run. steps counts moves across games, not games or
    network updates. folder is where we save the policy and the episode log.

    1. Create one DQN agent. Its MlpPolicy is a feed-forward network: seven
       observations go in and four Q-values come out, one per direction.
    2. Let that same agent play and learn in blocks of up to 1,000 moves.
       Splitting the run into blocks lets us print progress while it trains.
       SB3 stores transitions in a replay buffer, samples earlier experience
       and updates the network. We do not need to write those updates ourselves.
    """
    import torch
    torch.set_num_threads(1)
    env = SoloEnv()
    # The two hidden layers have 32 neurons each. After collecting initial experience,
    # SB3 trains every four moves. Epsilon stays at 0.15 so the snake keeps exploring.
    options = dict(device="cpu", learning_rate=0.001, buffer_size=20000, learning_starts=256,
                   gamma=0.95, train_freq=4, target_update_interval=250,
                   exploration_initial_eps=0.15, exploration_final_eps=0.15,
                   policy_kwargs=dict(net_arch=[32,32]))
    # TODO 1.1: DONE - Replace None with an SB3 DQN. Give it "MlpPolicy", our env and seed,
    #model = DQN("MlpPolicy", env, seed=seed)

    # TODO 2.1: Try to replace DQN with PPO and see if it does better after training
    # and unpack options as keyword arguments. Create it once so it retains experience.
    model = PPO("MlpPolicy", env, seed=seed)
    
    if model is None:
        raise NotImplementedError("Set the model two lines above.")
    try:
        steps_left = steps
        number_of_blocks = ceil(steps / 1000)
        for _ in range(number_of_blocks):
            block_steps = min(1000, steps_left)
            # TODO 1.2: DONE - Replace the exception with a call to the model's learn method.
            # Use block_steps for total_timesteps; the last block may be shorter. Set
            # reset_num_timesteps to False so progress continues across blocks.
            model.learn(total_timesteps=block_steps, reset_num_timesteps=False)
            steps_left -= block_steps

            # Report on the average performance in the last 20 recently completed games
            recent = env.history[-20:]
            if recent:
                means = np.mean(recent, axis=0)
                print(f"{model.num_timesteps} steps trained | {steps_left} steps left | last {len(recent)} episodes: "
                      f"return {means[1]:.2f}, apples {means[2]:.1f}, turns {means[3]:.1f}")
        # The deployment script reads dqn.zip. The CSV keeps one row per finished game.
        folder.mkdir(parents=True, exist_ok=True)
        # TODO 2.2: replace the name of the model to be saved once you added PPO above
        
        model.save(folder/"dqn.zip")
        with (folder/"training.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["training_step","episode_return","apples","turns"])
            writer.writerows(env.history)
        return model, env.history
    finally:
        env.close()


def visualize(model, history, folder, seed):
    """See how training progressed, then watch what the agent learned.

    1. In training.png each dot is a completed game, placed at the training step
       where it ended. The line averages up to 20 games to make the trend visible.
    2. In greedy.gif we play one game. Now we always choose a move with
       the highest predicted Q-value. There are no random exploration moves or
       weight updates, so this replay can look different from the training games
       and should usually perform better.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10,3), layout="constrained")
    for col, (ax, label) in enumerate(zip(axes, ["Return","Apples","Turns"]), start=1):
        if history:
            data = np.array(history)
            w = min(400, len(data))
            ax.scatter(data[:,0], data[:,col], s=3, alpha=0.2)
            ax.plot(data[w-1:,0], np.convolve(data[:,col], np.ones(w)/w, "valid"))
        ax.set(xlabel="Training steps", ylabel=label)
    fig.suptitle("Exploratory training episodes and trailing mean (up to 20 episodes)")
    fig.savefig(folder/"training.png", dpi=130)
    plt.close(fig)
    def greedy(game):
        action, _ = model.predict(prepare_observation(board(game)), deterministic=True)
        return int(action)
    replay(greedy, seed+1000000, folder/"Greedy.gif", "DQN: greedy replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 4 or args.seed < 0:
        parser.error("Use a positive multiple of four steps and a nonnegative seed.")
    # Change --seed to repeat the experiment; each seed and budget gets its own folder.
    folder = Path("results")/f"dqn_seed{args.seed}_steps{args.steps}"
    model, history = train(args.seed, args.steps, folder)
    visualize(model, history, folder, args.seed)
