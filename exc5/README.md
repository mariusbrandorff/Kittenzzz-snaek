# Exercise 5 — Learn to play Battlesnake with Q-learning

So far, you have learned from actions recorded by another agent. In this exercise,
your agent learns **from scratch by playing the game**: it observes the game's state, chooses a
direction, receives a reward, and updates its own action values. There is no
training dataset, teacher policy, neural network or pre-trained Q-table. Everything
will be learned from the agent's own experiences.

Your goal is to learn to collect apples and avoid dying in a solo Battlesnake game.
Next week, we will replace the small table with a deep reinforcement learning model.

## Learning goals

- Connect the state, action, reward and next state in the interaction loop.
- Choose a small state representation and explain what information it loses.
- Balance exploration and exploitation with an epsilon-greedy policy.
- Implement the one-step Q-learning update and handle terminal states correctly.
- Train from experience and evaluate separately from exploration and learning.
- Relate reward design and state representation to the behavior you observe.

## Setup and workflow

Use Python 3.10 or later. From this folder, install the dependencies:

```sh
python -m pip install -r requirements.txt
python step_0_q_explore.py
```

Step 0 should work without further additions. 
Complete each task before running its next command (or press run in your IDE):

```sh
python step_1_q_state.py
python step_2_q_choose_action.py
python step_3_q_train.py
python step_4_q_evaluate.py
```

Steps 1–3 contain unfinished functions. 
The `supplied_*.py` files provide a faster implementation of the game, 
a renderer to visualize game states, and tools for running experiments and logging the results.
You implement the state representation, action choice, Q update and learning loop.
Generated Q-tables will be stored into `models/`; figures, CSV tables and animations go into
`results/`. These folders will be automatically created.
This exercise has no dependencies to the previous ones, but you may still put it in the same folder.

## The supplied game

Training uses a fast "Python implemmentation" of the Battlesnake game, with the
familiar `observation['board']` and `observation['you']` dictionaries. It implements
the core movement, health, growth, wall and self-collision rules. This is not the
full official engine, but helps us to run the experiments a little quicker.
There is also a much faster framework called HISSS "https://github.com/ymahlau/hisss",
but I found it harder to use for this specific exercise. We come back to it next week.

The default setting for the environment is a 7×7 board, starting length 3, health 100, 
a random starting position, and exactly one apple. 
An eaten apple is replaced uniformly in a free cell. 
The official engine can use different starting layouts and food spawning.
An episode ends on death or filling the board; a 250-turn time limit also stops
a long surviving agent early. This is to keep training quicker while focusing on the early 
game where the snake is not as long. However, it also biases the snake into training mainly for
these situations and it may now know how to react in the late game.


The interface is similar to Gymnasium (a popular reinforcement learning framework,
that we will start using next week). It is based on the agent-environment interface.
The environment is the task to be solved. An environment step is simulated every time 
the agent reports an action. It returns an updated state, a reward and tells the agent
if the task has been terminated.

```python
observation, info = env.reset(seed=7)
next_observation, reward, terminated, truncated, info = env.step(action)
```

`terminated` means a true end such as death. `truncated` means the external time
limit was reached while alive. An episode is stopped on **either** flag.
For the Q-learning target, only a true terminal state removes the future value;
at a time limit, still bootstrap from the final observation before resetting.

The following rewards are provided by the default environment:

| Outcome of the action | Reward |
|---|---:|
| Eat an apple and remain alive | +1.00 |
| Die by collision or starvation | −1.00 |
| Otherwise move without eating | −0.01 |

You can change the trained behavior of the snake by modifying these rewards, e.g.
trying to cutoff the opponent from apples, eating smaller opponents, avoid states of hunger.


## Step 0 — Observe the interaction loop

**File:** `step_0_q_explore.py` — supplied, no implementation required.

Run it and open `results/step_0_random.gif`. The initial random policy chooses
among all four directions equally, so an episode can be very short.

- Yellow is the head; teal cells are the body; red circles are apples.
- A red cross marks death. For a wall collision it is drawn on the border cell
  to remain visible.
- The display shows the turn, health, length and reward from the last action.
- `results/step_0_transitions.csv` links the head position, chosen action, reward,
  next head position and health. The console prints the same transitions.

Choose one transition. Identify the state, action, reward and next state. 


## Step 1 — Build a small state representation

**File:** `step_1_q_state.py`. Complete `encode_state(observation)`.

A table for every possible full board would be too large. Start with this tuple:

```text
(safe_up, safe_down, safe_left, safe_right, food_x, food_y)
```

Use the supplied `safe_directions(observation)` for four binary wall/body safety
flags. They describe only the immediate move, not long-term safety or starvation.
For each food coordinate, use **0 = smaller, 1 = equal, 2 = larger** than the head
coordinate. Select the nearest apple by Manhattan distance, with x then y as
tie-breakers. If there is no food, use `(1, 1)` for its direction.

Return a tuple of six integers. The supplied array is indexed as `q[state]` to
obtain its four action values. Its shape is `(2, 2, 2, 2, 3, 3, 4)`:
at most **144 encoded states and 576 state-action values**, initialized to zero.

Run the file to check the output shape and ranges. Come up with two different boards that
produce the same tuple. Could the best action differ between them?

**Expected outcome:** a compact encoding, plus an explanation of its limitations.
The full board is much closer to a Markov state. This compressed observation loses
body layout, distances and health, so it need not satisfy the Markov property.
Q-learning here is a useful approximation, not a guarantee of an optimal policy.

How much would the number of states grow when we include e.g. the health counter?

## Step 2 — Explore and exploit

**File:** `step_2_q_choose_action.py`. Complete `choose_action(q_values, epsilon, rng)`.

- With probability `epsilon`, choose uniformly among the four actions.
- Otherwise, choose an action with the largest Q-value.
- If several actions tie, choose randomly among them using the supplied generator.

Use `rng` for randomness so experiments can be repeated with a seed. 
The chosen action must be an integer from 0 to 3. So you have to return the index 
of the action with the highest value (or a random index in case of epsilon).

Run the file. With epsilon 1, expect roughly equal counts for all actions. With
epsilon 0, only maximum-value actions should be chosen. Explain why always taking
the first maximum would bias an initially zero table toward one direction.

**Expected outcome:** a behavior policy that explores during training and can be
made greedy for evaluation. Epsilon is unrelated to the learning rate alpha.

## Step 3 — Implement Q-learning and train 

**File:** `step_3_q_train.py`. Complete `update_q` and `train_episode`.

The update from the lecture is:

$$
Q(s,a) \leftarrow Q(s,a)+\alpha\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right].
$$

At a true terminal state, the target is just `r`. 
$$
Q(s,a) \leftarrow Q(s,a)+\alpha r.
$$

Update only the selected state-action entry in place. 

In `train_episode`, we connect everything together into a training loop for our agent:

1. Reset the environment using the provided episode seed and encode the observation.
2. Choose an action using the current state's four Q-values and epsilon.
3. Apply the action to the environment and observe the result.
4. Encode the next observation if the game has not truly terminated.
5. Apply your Q update and accumulate reward and step count.
6. Advance to the next state; repeat until termination or truncation.
7. Return the summary dictionary documented in the function.

Run `python step_3_q_train.py`. The supplied outer loop starts the table at zero,
runs 6,000 episodes, and reduces epsilon from 1.0 to 0.05 over the first 80% of
training. This helps in being more explorative in the beginning and chooses more
conservative actions towards the end. 

The code handles saving and rendering. 
Alpha and gamma can be found and changed in `settings.py`.

Once training completed, inspect:
- `results/training_curves.png`: moving averages of episode reward, apples and
  turns **during exploratory training**.
- `results/training_seed_0.gif`: snapshots of greedy behavior before training,
  after 1,000 episodes, after 3,000 episodes and at the end.
- `results/train_seed_0.csv`: individual episodes, epsilon and outcomes.

The reported episode return is the reward sum. Gamma (the discount factor) is only 
used in learning, it does not impact the evaluation of our agent.

**Expected outcome:** an agent that improves through interaction. Training curves
may be noisy and partly reflect decreasing exploration. The trained agent should be
able to collect several apples before dying. If not, check if any of the previous steps
were faulty.

## Step 4 — Evaluate and explain the behavior

**File:** `step_4_q_evaluate.py` — supplied; interpret its outputs.

Run it after training in step 3 has been completed. 
It sets epsilon to zero such that only the best known move is selected.
It does not further update Q, and evaluates 100 new environment seeds to get a. A random policy uses the same reset seeds. Later food
good picture of the performance.

Open `results/RESULTS.md`, `results/evaluation_by_run.csv`, and the three
`evaluation_game_*.gif` animations. The animations show random and learned behavior
side by side.

Discuss:
1. Compare apples, survival time and reward. Do all three rank behavior identically?
2. Why can evaluation outperform the last training episodes?
3. What might change if gamma (in discounted return) were zero? 
4. Test how different alpha values change the training result. What does alpha control?

**Expected outcome:** a measured comparison with a baseline and an explanation of
failure cases. 

Note: Do not use the same seeds for training and evaluation seeds or the manual tuning of the agent's hyperparameter.
Use separate development seeds for tuning and fresh test seeds for final reporting.


## Optional Step 5 — Use your table in the regular Battlesnake engine

**File:** `step_5_q_serve_optional.py` — supplied API adapter.

Start your trained agent:

```sh
python step_5_q_serve_optional.py --port 8005
```

In another terminal, use the Battlesnake CLI from your earlier exercises or an
installation for your operating system:

(adapt as needed to point to the battlesnake.exe file)
```sh
battlesnake.exe play -g solo -W 7 -H 7 --name Q-learning --url http://127.0.0.1:8005
```

The server calls **your** encoder and action selector with epsilon zero. There
is no training involved, so the agent already needs to be trained. 
Change the CLI board size if you changed the training configuration. 
The official engine uses different initialization and
food spawning algorithms, so performance can differ from the teaching environment. 

## Optional experiments and your semester project

- Change the rewards and retrain from zero. Keep a separate output folder for each
  experiment; changing rewards changes what the agent optimizes and therefore its behavior.
- Compare two epsilon schedules or two values of alpha/gamma.
- Repeat Steps 3 and 4 with `--seeds 0 1 2`. Report variation across training runs,
  not just the best run.
- Add a health bucket or local body information. Update `STATE_SHAPE`, increment
  `ENCODING_VERSION`, and retrain. How quickly does the table grow and how fast 
  does your agent learn to collect apples?
- Adapt the environment/observation to your project or previous state encodings. 
  Define actions, rewards, termination and evaluation before changing the algorithm.
