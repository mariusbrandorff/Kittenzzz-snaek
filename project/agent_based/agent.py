import json
import time
from typing import Any, Literal, SupportsFloat
import gymnasium as gym
from gymnasium import spaces
from hisss import BattleSnakeConfig, BattleSnakeGame
import hisss
from hisss.cpp.lib import CPP_LIB
import numpy as np
from pettingzoo import ParallelEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
import supersuit as ss
import torch

from gamestate import GameState
from iAgent import IAgent, InfoReturn

DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))
MOVES = ("up", "right", "down", "left")

class Agent(IAgent):
    def __init__(self) -> None:
        super().__init__()
        self.model = PPO.load("./battlesnake_ppo.zip", device="auto")

    def info(self) -> InfoReturn:
        info = super().info()
        info.color = "#FF0"
        return info
    
    def start(self, gamestate: GameState):
        return super().start(gamestate)

    def move(self, gamestate: GameState) -> Literal['up'] | Literal['right'] | Literal['down'] | Literal['left']:
        mv, _ = self.model.predict(prepare_observation(gamestate), deterministic=True)
        print(mv)
        return MOVES[int(mv)]

    def end(self, gamestate: GameState):
        return super().end(gamestate)

    def dry(self):
        return super().dry()


    def train(self, steps: int):
        print(f"""
-----------------------
Training Model
Steps: {steps}
-----------------------
""")

        torch.set_num_threads(4)
        start_setup = time.perf_counter()
        env = MultiAgentEnv()
        env = ss.black_death_v3(env)
        vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=1,
            # num_cpus=0,
            base_class="stable_baselines3",
        )

        model = PPO("MlpPolicy", vec_env, batch_size=steps*4, n_steps=steps, verbose=1, device="auto")
        end_setup = time.perf_counter()

        start_learn = time.perf_counter()
        model.learn(total_timesteps=steps)
        end_learn = time.perf_counter()

        model.save("battlesnake_ppo")
        vec_env.close()

        print(f"""
-----------------------
Setup time: {end_setup-start_setup}
Learning time: {end_learn-start_learn}
-----------------------
""")




def prepare_observation(state: GameState):
    board = state.board
    snake = state.you
    body = state.you.body
    
    x, y = snake.head.x, snake.head.y

    blocked = body if len(body) < snake.length else body[:-1]
    safe = [0 <= x+dx < board.width and 0 <= y+dy < board.height and (x+dx,y+dy) not in blocked
            for dx,dy in DELTAS]
    
    food_sorted = sorted(board.food, key=lambda f: snake.head.distance(f))
    # Position relative to head / board size to fit into -1 to 1 float type (a percentage of how far to the board border kinda)
    foods = [food for f in food_sorted[:3] for food in ((f.x-x)/(board.width-1), (f.y-y)/(board.height-1))]
    foods.extend([0.0] * (6 - len(foods)))


    enemies_sorted = sorted(board.snakes, key=lambda e: snake.head.distance(e.head))
    enemies = [
        enemy for e in enemies_sorted[1:]
        for enemy in (
            (e.head.x-x)/(board.width-1), 
            (e.head.y-y)/(board.height-1), 
            e.health / 100, 
            -1 if e.length < snake.length else 1 if e.length > snake.length else 0
        )
    ]
    enemies.extend([0.0] * (12 - len(enemies)))

    return np.array([
        *safe,
        *foods,
        *enemies,
        snake.health/100,
    ], dtype=np.float32)

class MultiAgentEnv(ParallelEnv):
    metadata = {"name": "battlesnake_v0"}

    
    game: BattleSnakeGame | None = None
    history: list[tuple[int, float, int, ]] = []
    steps: int = 0

    h: int
    w: int
    n_players: int

    def __init__(self, w:int = 11, h:int = 11, n_players:int = 4, max_cycles: int = 500) -> None:
        super().__init__()
        self.render_mode = None

        self.w = w
        self.h = h
        self.n_players = n_players
        self.max_cycles = max_cycles

        # Ids
        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self.agents: list[str] = []

        

        # self.observation_spaces = {
        #     agent: spaces.Box(
        #         -1., 1.,
        #         shape=(size,),
        #         dtype=np.float32
        #     ) for agent in self.possible_agents
        # }
        self.action_spaces = {
            agent: spaces.Discrete(4) for agent in self.possible_agents
        }

    def observation_space(self, agent: Any) -> gym.Space:
        # size = 4+3*2+ 3*4 + 1 # 4 (safe moves) + 3*2 (3 closest foods(x,y)) + 3*4 (3 enemies(x,y,health,length(shorter, same or longer))) + 1 (health)
        size = 4+3*2+ 3*4 + 1
        return spaces.Box(
            -1., 1.,
            shape=(size,),
            dtype=np.float32
        )
    
    def action_space(self, agent: Any) -> gym.Space:
        return spaces.Discrete(4)


    def reset(self, seed: int | None = None, options: dict | None = None):
        # super().reset(seed=seed, options=options)
        self.close()

        np_random = np.random.default_rng(seed)
        game_seed = np_random.integers(2**31-1)
        CPP_LIB.lib.set_seed(int(game_seed) % (2**31 - 1))
        
        self.game = BattleSnakeGame(BattleSnakeConfig(
            4, self.n_players, self.w, self.h,
            all_actions_legal=True
        ))
        self.steps = 0
        self.agents = self.possible_agents.copy()

        observations = {
            agent: self._observation_for(self._player_id(agent))
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}
        return observations, infos
        
    
    def step(self, actions: dict) -> tuple[dict, dict[Any, float], dict[Any, bool], dict[Any, bool], dict[Any, dict[str, Any]]]:
        if self.game is None:
            raise RuntimeError("Call reset() before step().")
        
        # Require one action for each currently living snake.
        if set(actions) != set(self.agents):
            raise ValueError(f"Expected actions for {self.agents}, got {list(actions)}")

        for agent, action in actions.items():
            if not self.action_spaces[agent].contains(action):
                raise ValueError(f"Invalid action {action} for {agent}")

        old_agents = self.agents.copy()
        old_gamestates = {
            agent: (self._agent_state(self._player_id(agent)), self.game.is_player_alive(self._player_id(agent)) ) for agent in old_agents
        }

        # TODO 1: Convert {agent_name: action} to the action format expected by
        # BattleSnakeGame, then advance EXACTLY ONE simultaneous game turn.
        # self._apply_actions_and_advance(actions)
        alive_ids = [
        i for i in range(self.n_players)
            if self.game.is_player_alive(i)
        ]
        ordered_actions = tuple(actions[f"player_{i}"] for i in alive_ids)
        _, _, _ = self.game.step(ordered_actions)

        self.steps += 1
        self.agents = [
            agent for agent in self.possible_agents
            if self.game.is_player_alive(self._player_id(agent))
        ]

        # TODO 2: Define individual rewards in _reward_for().
        rewards = {
            agent: float(self._reward_for(self._player_id(agent), old_gamestates[agent][0], self._agent_state(self._player_id(agent)), old_gamestates[agent][1]))
            for agent in old_agents
        }

        game_over = self.game.is_terminal()
        truncated_game = self.steps >= self.max_cycles and not game_over
        
        # Return observations for the agents that acted this turn.
        # Dead agents get a zero observation; alive agents get their next view.
        observations = {
            agent: (
                self._observation_for(self._player_id(agent))
                if agent in self.agents
                else np.zeros(self.observation_space(agent).shape or [], dtype=np.float32)
            )
            for agent in old_agents
        }

        terminations = {
            agent: agent not in self.agents or game_over
            for agent in old_agents
        }

        for terminated in terminations:
            self.history.append((self.steps, rewards[terminated], self.game.turns_played))

        truncations = {agent: truncated_game for agent in old_agents}
        infos = {agent: {} for agent in old_agents}

        if game_over or truncated_game:
            # Keep the observations for the episode that just ended in infos.
            for agent in old_agents:
                infos[agent]["terminal_observation"] = observations[agent]

            # Start the next episode now; SB3 receives these as the next observations.
            reset_observations, _ = self.reset()
            observations = {
                agent: reset_observations[agent]
                for agent in old_agents
            }

        return observations, rewards, terminations, truncations, infos
        
    
    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None
    
    def _player_id(self, agent: str) -> int:
        return int(agent.removeprefix("player_"))
    
    def _agent_state(self, player_id: int):
        if self.game is None:
            raise Exception("Game is None")
        return GameState(
            json.loads(hisss.to_battlesnake_json(self.game, player=player_id))
        )

    def _observation_for(self, player_id: int) -> np.ndarray:
        # Adapt this to return that player's observation.
        # Your current reset code hard-codes player=0.
        state = self._agent_state(player_id)
        obs = np.asarray(prepare_observation(state), dtype=np.float32)
        return obs

    def _reward_for(self, player_id: int, old_game_state: GameState, new_game_state: GameState, was_alive: bool) -> float:
        if self.game is None:
            raise Exception("Game cannot be None")
        # This will go -100 - 100 as an int, so no floating point math.
        score: int = 0
        scores = {
            "food": 30, # Points for eating
            "death": -100,
        }

        if (old_game_state.you.length < new_game_state.you.length):
            # You grew!
            if (old_game_state.you.health):
                eat_mult = max(50/old_game_state.you.health, 2) # Max multiplier is 2
                score += int(scores['food'] * eat_mult)

        if was_alive and not self.game.is_player_alive(player_id):
            score = scores["death"]

        if (score == 0):
            score = 10 # Snake is slowly dying

        return min(score, 100) / 100