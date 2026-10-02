"""Step 2: learn the recorded directions with a small PyTorch neural network."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from step_0_state_attributes import ACTIONS, INPUT_COLUMNS, load_dataframe


class DirectionNetwork(nn.Module):
    """Define the architecture here once; training and live play both use it."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(len(INPUT_COLUMNS), 32),
            nn.ReLU(),
            nn.Linear(32, len(ACTIONS)),
        )

    def forward(self, inputs):
        return self.layers(inputs)  # Four scores: up, down, left, right.


def split_dataframe(df, seed=42):
    # Keep whole games (and repeated seeds) together to avoid leaking examples.
    group_column = "seed" if "seed" in df and df.seed.notna().all() else "game_id"
    groups = df[group_column].drop_duplicates().to_numpy().copy()
    if len(groups) < 2:
        raise ValueError("Record at least two different games in Step 1 first.")
    np.random.default_rng(seed).shuffle(groups)
    validation = df[group_column].isin(groups[:max(1, round(len(groups) * .2))])
    return df.loc[~validation], df.loc[validation]


def train(df, output, epochs=150, learning_rate=.003):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists. Choose another --model filename.")
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    training, validation = split_dataframe(df)

    # 1. Turn DataFrame columns into numeric inputs and direction labels.
    x_train = torch.tensor(training[INPUT_COLUMNS].to_numpy(dtype=np.float32))
    y_train = torch.tensor(training.direction.map(ACTIONS.index).to_numpy(), dtype=torch.long)
    x_validation = torch.tensor(validation[INPUT_COLUMNS].to_numpy(dtype=np.float32))
    y_validation = torch.tensor(validation.direction.map(ACTIONS.index).to_numpy(), dtype=torch.long)
    if not torch.isfinite(x_train).all() or not torch.isfinite(x_validation).all():
        raise ValueError("Input attributes must contain finite numbers.")

    # 2. Normalize using training data only; save these values for live play.
    mean, std = x_train.mean(dim=0), x_train.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1
    x_train, x_validation = (x_train - mean) / std, (x_validation - mean) / std

    # 3. Initialize the model defined above
    model = DirectionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()  # Takes scores directly; no softmax needed.

    # 4. Learn from all training examples once per epoch.
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()            # Clear old gradients.
        predictions = model(x_train)           # Predict direction scores.
        loss = loss_function(predictions, y_train)
        loss.backward()                  # Compute gradients.
        optimizer.step()                 # Update weights and biases.
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch}: loss = {loss.item():.3f}")

    # 5. Check agreement on held-out games, then save the model for Step 3.
    model.eval()
    with torch.no_grad():
        accuracy = (model(x_validation).argmax(dim=1) == y_validation).float().mean().item()
    print(f"Validation agreement with the rule-based agent: {accuracy:.1%}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(),
                "input_columns": INPUT_COLUMNS, "actions": ACTIONS,
                "mean": mean.tolist(), "std": std.tolist()}, output)
    # Step 4 uses this seed list to avoid evaluating on recorded games.
    seeds = sorted(int(s) for s in df.seed.dropna().unique()) if "seed" in df else []
    output.with_suffix(".json").write_text(json.dumps({"recorded_seeds": seeds}), encoding="utf8")
    print(f"Saved {output}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/rule_moves.json")
    parser.add_argument("--model", default="models/rb_trained_nn_model.pt")
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    train(load_dataframe(args.data), args.model, epochs=args.epochs)
