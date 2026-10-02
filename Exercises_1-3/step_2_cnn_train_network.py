"""Step 2: train a CNN to imitate the directions recorded by the rule-based bot."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import cnn_config as config
from step_0_cnn_state_attributes import (
    ACTIONS, CHANNELS, INPUT_SHAPE, ENCODING_VERSION, cnn_dataframe_to_tensors, cnn_load_dataframe,
)


class DirectionNetwork(nn.Module):
    """A small LeNet-style classifier for the whole board.

    The input channel count comes from Step 0, so students can start with one
    channel and add others later. Step 3 imports this same architecture.
    """
    def __init__(self):
        super().__init__()
        # Padding keeps the 5x5 filters usable on our small 11x11 board.
        self.features = nn.Sequential(
            nn.Conv2d(len(CHANNELS), 16, kernel_size=5, padding=2),  # Cx11x11 -> 6x11x11
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),                          # -> 6x5x5    # Swaped AvgePool for MaxPool - Avg blurs details where Max only keeps the max value
            nn.Conv2d(16, 16, kernel_size=5, padding=2),           # -> 16x5x5
            # nn.Tanh(),    # Swaped nn.Tanh() for nn.ReLU()
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),                          # -> 16x2x2
        )
        # Two pooling layers halve each spatial dimension twice.
        # TODO: adapt this in case you change the architecture above
        flattened_size = 16 * (INPUT_SHAPE[1] // 4) * (INPUT_SHAPE[2] // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, len(ACTIONS)),
        )

    def forward(self, inputs):
        # Four logits: up, down, left, right. CrossEntropyLoss needs no softmax.
        return self.classifier(self.features(inputs))


def cnn_split_dataframe(df, seed=42):
    # Keep complete games and repeated seeds together, as in Exercise 2.
    group_column = 'seed' if 'seed' in df and df.seed.notna().all() else 'game_id'
    groups = df[group_column].drop_duplicates().to_numpy().copy()
    if len(groups) < 2:
        raise ValueError('Record at least two games with different seeds in Step 1.')
    np.random.default_rng(seed).shuffle(groups)
    held_out = df[group_column].isin(groups[:max(1, round(len(groups) * .2))])
    return df.loc[~held_out], df.loc[held_out]


def cnn_train(df, output=None, epochs=30, learning_rate=.003, batch_size=64):
    output = Path(output if output is not None else config.MODEL_PATH)
    if output.exists():
        raise FileExistsError(f'{output} exists. Choose another --model filename.')
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Epochs, batch size and learning rate must be positive.')
    torch.manual_seed(42)
    torch.set_num_threads(1)
    training, test = cnn_split_dataframe(df)
    x_train, y_train = cnn_dataframe_to_tensors(training)
    x_test, y_test = cnn_dataframe_to_tensors(test)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    print(f'Train: {len(training)} decisions; test: {len(test)}. Input: {tuple(x_train.shape)}')

    # Input values and their scaling are defined by the encoder in Step 0.
    model = DirectionNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        model.eval()
        with torch.inference_mode():
            accuracy = (model(x_test).argmax(1) == y_test).float().mean().item()
        history.append({'epoch': epoch, 'training_loss': total_loss / len(training),
                        'test_accuracy': accuracy})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f'Epoch {epoch}: loss {history[-1]["training_loss"]:.3f}; test agreement {accuracy:.1%}')

    seeds = sorted(int(s) for s in df.seed.dropna().unique()) if 'seed' in df else []
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'weights': model.state_dict(), 'channels': CHANNELS, 'input_shape': INPUT_SHAPE,
                'actions': ACTIONS, 'encoding_version': ENCODING_VERSION, 'recorded_seeds': seeds}, output)
    output.with_suffix('.json').write_text(json.dumps({'recorded_seeds': seeds,
        'train_examples': len(training), 'test_examples': len(test), 'epochs': epochs}), encoding='utf8')
    pd.DataFrame(history).to_csv(output.with_suffix('.history.csv'), index=False)
    print(f'Saved {output}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=config.DATA_PATH)
    parser.add_argument('--model', default=config.MODEL_PATH)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=.003)
    args = parser.parse_args()
    cnn_train(cnn_load_dataframe(args.data), args.model, args.epochs, args.learning_rate, args.batch_size)
