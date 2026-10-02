"""Instructor-provided data preparation, training and plotting infrastructure."""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import settings as cfg
from supplied_simulation import advance


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_recording():
    return pd.read_json(cfg.RECORDING, orient='table')


def encode_observation(row):
    """Ten current-state features; recorded/future actions are never inputs."""
    x, y = row['body'][0]
    w, h = row['width'], row['height']
    food = min(row['food'], key=lambda p: (abs(p[0]-x)+abs(p[1]-y), p[0], p[1])) if row['food'] else (x, y)
    safe = [float(advance(row, a)['alive']) for a in range(4)]
    return np.array([x/(w-1), y/(h-1), (food[0]-x)/(w-1), (food[1]-y)/(h-1),
                     row['health']/100, row['length']/(w*h), *safe], dtype=np.float32)


def explore():
    df = read_recording()
    cfg.RESULTS.mkdir(parents=True, exist_ok=True)
    row = df[df.alive].iloc[0]
    game = df[(df.game_id == row.game_id) & df.alive].sort_values('turn')
    xy = np.array([b[0] for b in game.body])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(xy[:, 0], xy[:, 1], color='#315ab0', alpha=.5)
    dots = axes[0].scatter(xy[:, 0], xy[:, 1], c=game.turn, cmap='viridis', s=15)
    axes[0].scatter(*xy[0], marker='o', s=95, facecolors='none', edgecolors='#111827', label='Start')
    axes[0].scatter(*xy[-1], marker='x', s=65, color='#c73f3f', label='Last living head')
    axes[0].legend(fontsize=8, loc='best')
    axes[0].set(xlim=(-.5, row.width-.5), ylim=(-.5, row.height-.5),
                title=f'Recorded head trajectory · game {row.seed}', xlabel='x', ylabel='y', aspect='equal')
    fig.colorbar(dots, ax=axes[0], label='Turn')
    counts = df.direction.value_counts().reindex(cfg.ACTIONS, fill_value=0)
    axes[1].bar(counts.index, counts, color='#315ab0')
    axes[1].set(title=f'Action counts across all {df.game_id.nunique()} games', ylabel='Count')
    fig.savefig(cfg.RESULTS/'step_0_recordings.png', dpi=150)
    plt.close(fig)
    print(f'{df.game_id.nunique()} games; {len(df):,} states; {len(cfg.FEATURES)} supplied features.')
    print('Features:', cfg.FEATURES)
    print('Example observation:', encode_observation(row))
    print('Next action:', row.direction)
    print('\nLeft: one game; color is time (purple = early, yellow = late). Lines join successive heads.')
    print('The line is the traveled path, not the body at a single turn. Revisited cells overlap.')
    print('Right: counts of the four actions across the complete dataset.')
    transitions = []
    records = game.head(9).to_dict('records')
    for current, following in zip(records, records[1:]):
        transitions.append(dict(turn=current['turn'], head=tuple(current['body'][0]),
            action=current['direction'], next_turn=following['turn'], next_head=tuple(following['body'][0])))
    preview = pd.DataFrame(transitions)
    preview.to_csv(cfg.RESULTS/'step_0_transitions.csv', index=False)
    print('\nFirst transitions of the displayed game (x right, y up):')
    print(preview.to_string(index=False))
    print('\nOpen results/step_0_recordings.png to inspect the figure.')


def prepare(make_example):
    df = read_recording()
    seeds = np.array(sorted(df.seed.unique()))
    if len(seeds) < 10:
        raise ValueError('Use at least 10 recorded games.')
    np.random.default_rng(cfg.SPLIT_SEED).shuffle(seeds)
    a, b = int(.7*len(seeds)), int(.85*len(seeds))
    splits = {int(s): name for name, group in zip(['train', 'validation', 'test'],
              [seeds[:a], seeds[a:b], seeds[b:]]) for s in group}
    arrays = {key: [] for key in ['observations', 'actions', 'split', 'row_index', 'seed', 'turn']}
    for _, game in df.groupby('game_id', sort=False):
        game = game.sort_values('turn')
        rows = game.to_dict('records')
        if rows[-1]['alive'] or list(game.turn) != list(range(len(game))):
            raise ValueError('Expected consecutive complete games starting at turn 0.')
        features = np.array([encode_observation(r) for r in rows[:-1]])
        actions = np.array([cfg.ACTIONS.index(r['direction']) for r in rows[:-1]])
        # Require the state AFTER the final target action to still be alive.
        # At t: history t-H+1..t; actions t..t+F-1; resulting states t+1..t+F.
        for t in range(cfg.HISTORY - 1, len(rows) - 1 - cfg.FUTURE):
            x, y = make_example(features, actions, t, cfg.HISTORY, cfg.FUTURE)
            if x.shape != (cfg.HISTORY, len(cfg.FEATURES)) or y.shape != (cfg.FUTURE,):
                raise ValueError('Check the observation and action slices in Step 1.')
            values = dict(observations=x, actions=y, split=splits[rows[t]['seed']],
                          row_index=game.index[t], seed=rows[t]['seed'], turn=t)
            for key, value in values.items():
                arrays[key].append(value)
    data = {key: np.asarray(value) for key, value in arrays.items()}
    if not len(data['actions']):
        raise ValueError('No windows; reduce HISTORY/FUTURE or record longer games.')
    cfg.DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.DATA/'sequences.npz', **data)
    metadata = dict(history=cfg.HISTORY, future=cfg.FUTURE, features=cfg.FEATURES,
                    actions=cfg.ACTIONS, split_seed=cfg.SPLIT_SEED,
                    recording_sha256=fingerprint(cfg.RECORDING),
                    dataset_sha256=fingerprint(cfg.DATA/'sequences.npz'))
    (cfg.DATA/'metadata.json').write_text(json.dumps(metadata, indent=2))
    summary = pd.DataFrame({'split': data['split'], 'seed': data['seed']}).groupby('split').agg(
        windows=('seed', 'size'), games=('seed', 'nunique'))
    summary.to_csv(cfg.DATA/'split_summary.csv')
    print(summary.to_string())
    print('Inputs:', data['observations'].shape, 'Action labels:', data['actions'].shape)
    print('First target sequence:', [cfg.ACTIONS[a] for a in data['actions'][0]])
    return data


def load_data():
    if not (cfg.DATA/'metadata.json').is_file() or not (cfg.DATA/'sequences.npz').is_file():
        raise FileNotFoundError('Complete and run Step 1 to prepare your observation/action sequences first.')
    metadata = json.loads((cfg.DATA/'metadata.json').read_text())
    expected = dict(history=cfg.HISTORY, future=cfg.FUTURE, features=cfg.FEATURES,
                    actions=cfg.ACTIONS, split_seed=cfg.SPLIT_SEED,
                    recording_sha256=fingerprint(cfg.RECORDING),
                    dataset_sha256=fingerprint(cfg.DATA/'sequences.npz'))
    if metadata != expected:
        raise ValueError('Settings or recordings changed. Rerun Step 1 and then retrain.')
    with np.load(cfg.DATA/'sequences.npz') as archive:
        return {k: archive[k] for k in archive.files}, metadata


def train_cli(kind, model_class):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('Use at least one epoch.')
    data, metadata = load_data()
    torch.set_num_threads(2)
    cfg.MODELS.mkdir(exist_ok=True)
    cfg.RESULTS.mkdir(exist_ok=True)
    summaries = []
    histories = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = model_class(len(cfg.FEATURES), cfg.HIDDEN, cfg.FUTURE)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        loaders = {}
        for split in ['train', 'validation']:
            select = data['split'] == split
            dataset = TensorDataset(torch.from_numpy(data['observations'][select]),
                                    torch.from_numpy(data['actions'][select]))
            loaders[split] = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=split=='train',
                                       generator=torch.Generator().manual_seed(seed))
        best = float('inf')
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            metrics = dict(epoch=epoch)
            for split, batches in loaders.items():
                model.train(split == 'train')
                loss_sum = correct = count = 0
                with torch.set_grad_enabled(split == 'train'):
                    for observations, actions in batches:
                        logits = model(observations)
                        loss = criterion(logits.reshape(-1, 4), actions.reshape(-1))
                        if split == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                        loss_sum += loss.item()*actions.numel()
                        correct += (logits.argmax(-1) == actions).sum().item()
                        count += actions.numel()
                metrics[split+'_loss'] = loss_sum/count
                metrics[split+'_accuracy'] = correct/count
            history.append(metrics)
            if metrics['validation_loss'] < best:
                best, best_epoch = metrics['validation_loss'], epoch
                torch.save(dict(state_dict=model.state_dict(), kind=kind, hidden=cfg.HIDDEN,
                    metadata=metadata, seed=seed, epoch=epoch), cfg.MODELS/f'{kind}_seed_{seed}.pt')
        table = pd.DataFrame(history)
        table.to_csv(cfg.RESULTS/f'{kind}_seed_{seed}_learning.csv', index=False)
        histories.append(table)
        summaries.append(dict(seed=seed, best_epoch=best_epoch, validation_nll=best,
                              parameters=sum(p.numel() for p in model.parameters()),
                              seconds=time.perf_counter()-started))
    pd.DataFrame(summaries).to_csv(cfg.RESULTS/f'{kind}_training.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        for split, color in [('train', '#287ca2'), ('validation', '#bb563d')]:
            values = np.stack([h[split+'_'+metric] for h in histories])
            ax.plot(histories[0].epoch, values.mean(0), color=color, label=split)
            ax.fill_between(histories[0].epoch, values.min(0), values.max(0), color=color, alpha=.15)
        ax.set(xlabel='Epoch', ylabel='Action NLL' if metric=='loss' else 'Action accuracy')
        ax.legend()
    fig.suptitle(f'{kind.upper()} · mean and range across {len(histories)} training seed(s)')
    fig.savefig(cfg.RESULTS/f'{kind}_learning.png', dpi=150)
    plt.close(fig)
    print(pd.DataFrame(summaries).to_string(index=False))
