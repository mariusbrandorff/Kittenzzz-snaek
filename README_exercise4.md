# Exercise 4 — What will the snake do next?

You have already built an agent that chooses a direction. This time you will
predict **another agent's next actions multiple steps into the future** from a short history. 
Your RNN and LSTM will predict the next x actions. Supplied code then moves the snake according to
those predictions and compares the resulting path with the recorded game.

## Learning goals

- Construct a history and its correctly aligned future action labels.
- Understand `(batch, time, features)` inputs and `(batch, future, actions)` outputs.
- Use an RNN's final output to predict several future decisions.
- Replace an RNN with an LSTM and make use of the additional cell state.
- Compare probabilistic predictions and interpret simulated paths, including errors.

## Setup

Install `requirements.txt`, then run the commands below **from this
directory**. The supplied recording contains 80 complete solo games of an
agent that you don't have access to. The dataset is included in
`data/solo_trajectories.json.gz` inside this folder. No other exercise folder,
agent source code or Battlesnake engine is required.

Step 0 runs immediately. Complete each task in Steps 1–3 before running its
command; these files deliberately contain unfinished functions. Step 4 uses the
models you trained. The `supplied_*.py` files provide the encoding, training and
visualization infrastructure; concentrate your implementation on the task files.

```sh
python -m pip install -r requirements.txt
python step_0_actions_explore.py
python step_1_actions_sequences.py
python step_2_actions_train_rnn.py
python step_3_actions_train_lstm.py
python step_4_actions_compare.py
```

## The task in one example

With `HISTORY = 8` and `FUTURE = 3`, a prediction made from state 12 uses:

```text
Observed states:   5, 6, 7, 8, 9, 10, 11, 12
Target actions:   action[12], action[13], action[14]
Resulting states: 13, 14, 15
```

The first target is the decision taken **at** the last observed state.
Future states and future food are not model inputs. The RNN updates its hidden
state while reading the history; the LSTM also updates a cell state. A final
linear layer uses the last recurrent output to predict the future actions.

The supplied encoder produces ten features: head x/y, nearest-food dx/dy, health,
length, and four safe-direction flags. Coordinates and scalar features are
normalized. The game has been played in singleplayer mode; 
there are no opponent features.

The model outputs raw logits of shape `(batch, FUTURE, 4)`, in the order
`up, down, left, right`. A logit can be considered the preference of a direction to walk to.
Cross-entropy will train the four-action classification task
at every future step. At evaluation, a softmax layer/function converts logits into probabilities.
The most probable action at each step gives one predicted sequence, for example
`right, right, up`. The simulator executes that whole sequence from the last
observed state **without replanning** or consulting later recorded observations.

## Step 0 — Explore the recorded games (10 minutes)

**File:** `step_0_actions_explore.py` (supplied; no TODO).

Run it and inspect `results/step_0_recordings.png`:

- **Left — head trajectory of one game:** x and y are board coordinates, with
  `(0, 0)` at the bottom left. Each dot is the head position at an observed turn;
  its color encodes time, from purple (early) to yellow (late), as shown by the
  color bar. Lines connect consecutive head positions. The circle marks the start
  and the red cross marks the last living head position. Cells may be revisited,
  so dots and lines can overlap. This is the path traveled over the whole game,
  not the snake's body at one moment, and the colors are not probabilities.
- **Right — action frequencies across all games:** each bar counts how often
  `up`, `down`, `left` or `right` was recorded in the entire dataset, including
  final actions. These counts are not restricted to the game on the left and
  do not indicate which action is best. Unequal counts can reveal class imbalance.
- **Printed transition table:** the console shows the first eight transitions
  of the displayed game: current turn, head position, chosen action, next turn
  and next head position. The same table is saved as
  `results/step_0_transitions.csv`. Use it to connect a specific action to movement;
  `up` increases y and `right` increases x.

Choose a row in the transition table. Identify its starting head position in the
plot and explain how the action leads to the next position. Then consider why
randomly splitting overlapping windows would leak information between training
and test sets.

The dataset is a pandas table: each row is one observed game state. `game_id`
identifies a game; `turn` gives its time step; `body[0]` is the head and the
remaining body entries are ordered towards the tail. `food` lists apple positions,
`health` and `length` describe the snake, and `direction` is its recorded next
action. The final terminal row has `alive=False` and no next action. The supplied
loader is `read_recording()` in `supplied_pipeline.py`.

**Expected outcome:** you can distinguish an observation, an action and a next
state. Complete games, including duplicate simulator seeds, stay within one split.

## Step 1 — Construct observation/action sequences (15 minutes)

**File:** `step_1_actions_sequences.py`.

Complete `student_make_example` using two array slices. Return the last `history`
observations ending at index `t`, and `future` actions starting at index `t`.
Then run the file to build the dataset. Inspect the printed shapes and first
target sequence. The pipeline handles splitting and staying within game boundaries.

**Expected outcome:** inputs have shape `(N, 8, 10)` and targets `(N, 3)` for the
default settings. We use training, validation and test games separately. Validation
selects the checkpoint; the held-out test set is only for the final comparison.

For simplicity, we keep only examples where the recorded snake remains alive
through all future turns. Excluding dying frames can make the prediction task
easier: results describe surviving situations and do not measure performance
near death or the agent's overall playing ability.

## Step 2 — Predict actions with an RNN

**File:** `step_2_actions_train_rnn.py`.

Define `nn.RNN(input_size, hidden_size, batch_first=True)` and a linear layer
with `future * 4` outputs. In `forward`, take the last recurrent output, pass it
through the linear layer, and reshape to `(batch, future, 4)`. Do not apply softmax
inside the network: the supplied cross-entropy loss takes raw logits.

Run the training script and inspect `results/rnn_learning.png`. Compare training
and validation loss. What happens if you increase `HIDDEN` or train longer?

**Expected outcome:** a short network definition; training and validation curves;
a checkpoint selected using validation loss, without examining test performance.

## Step 3 — Use an LSTM 

**File:** `step_3_actions_train_lstm.py`.

Copy your RNN from step_2 and adapt it to `nn.LSTM`. Its return values are `outputs, (hidden, cell)`.
The final element of `outputs` can still feed the same linear classifier.
Keep the input features, hidden width and training budget the same for comparison.

**Expected outcome:** an LSTM with the same prediction task. Explain what its
cell state is intended to retain. Equal hidden width does not mean equal parameter
count: the LSTM has additional gates and more parameters.

## Step 4 — Simulate, visualize and compare

**File:** `step_4_actions_compare.py` (supplied; no TODO).

After training both models, run it and open the newly generated
`results/RESULTS.md`, the four example PNGs, and their animated
GIFs. Each comparison contains the uniform-path baseline, RNN, LSTM and recorded
future. The figures display action probabilities, a concrete path and a heatmap
of endpoints obtained by sampling action sequences and simulating them.

The GIF animations have two rows. The **top row** executes one concrete action
sequence for each prediction and shows the actual recorded future in the final
column. The **bottom row** shows the probability of the head occupying each cell
at the displayed turn for uniform paths, RNN and LSTM. These heatmaps follow the
same 200 sampled paths throughout the animation, using a shared 0–1 color scale.
They describe the current future turn, not accumulated visits. The probability
of sampled paths having died is shown separately. The recorded-future column
remains a concrete animation; no probability map is added for it.

Discuss:

1. Which recorded actions receive high probability? Is the whole sequence correct?
2. Does the predicted path collide? Can individually plausible actions form an
   impossible sequence when combined?
3. Can different action sequences reach the same final cell?
4. Does the LSTM help in this task? Does this teacher actually need long memory?

**Metrics:** per-action top-1 accuracy, exact-sequence accuracy, action negative
log likelihood (lower is better), simulated survival over the forecast horizon,
and exact final head position. Each test game receives equal weight. These are
prediction metrics, not evidence that the models can play a full game well.

## How the supplied baseline and simulation work

The baseline enumerates all complete surviving paths of length `FUTURE` from the
current state and gives each equal weight. Its action probabilities come from
counting actions along those paths. For a concrete path it selects one with a
fixed random seed; every complete path ties for maximum joint probability.
When no surviving path exists, it falls back to all possible action sequences
and explicitly simulates their deaths. The evaluation records that fallback count.

For RNN/LSTM paths, the default decoder takes each step's argmax. The simple model
predicts all steps from one history, so later actions are **not conditioned on
earlier predicted actions**. Sampling likewise treats these steps as conditionally
independent. This is a simplification; illegal paths are a useful
limitation to observe as we may want to trap our opponent. 

The simulation used for visualizing your networks' results moves the body and tail, 
checks walls and self-collisions, and handles
health and growth after eating currently known food. It cannot know future food
spawns. The recorded-future animation uses the actual recorded states.

Endpoint heatmaps use 200 simulated samples per model. The map sum can be below
one when some sampled sequences die; their death probability is shown separately.

## Optional experiments

- Change `HISTORY` or `FUTURE` in `settings.py`; rebuild Step 1, retrain both
  models, then compare. The exact baseline supports `FUTURE` up to 5 to stay fast.
- Add a feed-forward model using a flattened observation history, or compare to
  a current-state-only model to test whether history actually helps.
- Add an observation feature and explain what extra information it supplies.
- Repeat training and evaluation with `--seeds 0 1 2` on Steps 2, 3 and 4.
  A single seed is sufficient for the required lesson.
- Advanced: condition each future action on previous predicted actions, or apply
  legal-action constraints during decoding. Report these as distinct experiments.
- Adapt all of this towards predicting the opponent's moves in a multiplayer game 
  and add rules to any of your agents for avoiding future opponent positions.
