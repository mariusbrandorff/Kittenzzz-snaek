# Exercise 6 — DQN and MCTS in solo Battlesnake

Last time you learned about tabular Q-learning. 
This week you will train a neural network via Deep Q Learning
and plan moves by simulating possible futures using MCTS. 
Both agents use native **HISSS** for development and the official 
Battlesnake API to play. 

HISSS is a more efficient version of battlesnakes made for fast training.
I used it such that we can train our agents faster. It is not mandatory 
for your projects but may help a lot.

Install the dependencies with Python 3.12 or 3.13:

```sh
python -m pip install -r requirements.txt
```

## DQN: train, then deploy

There are two DQN files, one for training the agent using HISSS, 
and a second one for applying the agent to Battlesnakes.

In **[dqn_train.py](dqn_train.py)**, read the `prepare_observation` function: 
its input is a Battlesnake board and its output is seven floats. 
The first four indicate immediate wall/body safety in HISSS order 
(**up, right, down, left**); the remaining three give the normalized 
x/y offset to the nearest food and your snake’s health. 
The four network outputs estimate action values. 
A move can still be chosen when its safety flag is zero, but the agent
will quickly learn to avoid unsafe moves.

In the `train` function, complete the two TODOs TODO 1.1 and TODO 1.2: 
construct the SB3 DQN using the supplied settings, 
then continue learning in 1000 training steps each.
We split the training into blocks of 1000 steps, 
to keep track of the performance during the training.
Keep the same model and use `model.learn(reset_num_timesteps=False)` such
that the training continues and not just resets. 

The supplied Gym adapter sends each action to HISSS and 
records completed training episodes. A surviving apple earns +1, 
an ordinary move −0.01 and death −1. An external 250-turn limit truncates 
the episode but is not counted as a death.

```sh
python dqn_train.py --seed 0 --steps 20000
python dqn_train.py --seed 1 --steps 20000
```

Each run writes its own `results/dqn_seed<seed>_steps<steps>/` folder. 
Inspect **training.png** for return, apples and survival during exploratory 
training; **training.csv** has every completed episode.
**greedy.gif** shows one game after training with exploration 
switched off. **dqn.zip** is the saved policy. 
The same seed controls the network, exploration and native HISSS 
episode seeds. Repeating the same arguments replaces that run’s outputs.

**[dqn_battlesnake.py](dqn_battlesnake.py)** loads the checkpoint once. 
Each move request goes through the same `prepare_observation` 
function as training, followed by one greedy prediction.
Use a checkpoint from this exercise: earlier agents may use a 
different action order.

```sh
python dqn_battlesnake.py --model results/dqn_seed0_steps20000/dqn.zip --port 8006
```

Once DQN works, you can try replacing it with PPO in TODO2.1 and TODO 2.2.


## MCTS: one search, two modes

Open **[mcts.py](mcts.py)**. You will write **all four phase functions**. 
Each function contains instructions. 

Here a reminder:

**Step 1 — Selection.** Follow existing children until you reach somewhere to expand or must stop. Return the selected node and the reward collected on the way there. 

**Step 2 — Exploration (expansion).** Try one missing action and add its resulting node to the tree. 
Return the node you reached and just that move's reward.

**Step 3 — Simulation.** Use the remaining moves for a random rollout. The supplied `rollout_moves` helper avoids immediate collisions when possible. Return only the rewards earned during this rollout. 

**Step 4 — Backpropagation.** Record one visit and the complete simulated return at every node on the path, up to and including the root.

Once you have implemented all four functions, compare the same starting seed with different per-move budgets:

```sh
python mcts.py --mode hisss --seed 0 --iterations 16
python mcts.py --mode hisss --seed 0 --iterations 64
```

Each command runs a full HISSS game, reports apples and turns, 
and writes a replay GIF to `results/`. 
In **Battlesnake mode** the search code is identical, 
but the state received from battlesnakes is first turned into a HISSS gamestate
and then hisss is used during the search. As the hisss framework is more efficient,
it allows us to do more search iterations in the same time.

```sh
python mcts.py battlesnake --iterations 64 --port 8006
```

## Play in the original engine

Start **one** of the two servers above. In a separate terminal, run your installed Battlesnake CLI (use its executable path if necessary):

```sh
battlesnake play -g solo -W 7 -H 7 --name Exercise6 --url http://127.0.0.1:8006
```

Use the solo ruleset. If search misses the engine’s move deadline, 
lower `--iterations` in the setup of the MCTS agent.
