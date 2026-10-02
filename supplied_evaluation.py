"""Supplied evaluation: action probabilities, simulated paths and uncertainty."""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import settings as cfg
from supplied_pipeline import load_data, read_recording
from supplied_simulation import uniform_paths, simulate


def scores(probabilities, chosen, targets, row, actual_future):
    p = probabilities[np.arange(cfg.FUTURE), targets]
    rollout = simulate(row, chosen)
    return dict(action_accuracy=float((probabilities.argmax(-1) == targets).mean()),
                sequence_accuracy=float(np.array_equal(chosen, targets)),
                action_nll=float(-np.log(p.clip(1e-7, 1)).mean()),
                rollout_survival=float(rollout[-1]['alive']),
                endpoint_accuracy=float(rollout[-1]['alive'] and
                    tuple(rollout[-1]['body'][0]) == tuple(actual_future[-1]['body'][0])))


def predict(kind, model_class, seed, observations, metadata):
    if not (cfg.MODELS/f'{kind}_seed_{seed}.pt').is_file():
        raise FileNotFoundError(f'Train your {kind.upper()} with seed {seed} in Step 2/3 before running the comparison.')
    checkpoint = torch.load(cfg.MODELS/f'{kind}_seed_{seed}.pt', map_location='cpu', weights_only=True)
    if checkpoint['metadata'] != metadata or checkpoint['kind'] != kind:
        raise ValueError('Checkpoint does not match this dataset. Retrain both models.')
    model = model_class(len(cfg.FEATURES), checkpoint['hidden'], cfg.FUTURE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(batch)).softmax(-1).numpy()
                               for batch in np.array_split(observations, max(1, len(observations)//256))])


def main():
    from step_2_actions_train_rnn import ActionRNN
    from step_3_actions_train_lstm import ActionLSTM
    from supplied_visualization import examples
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    torch.set_num_threads(2)
    data, metadata = load_data()
    df = read_recording()
    indices = np.flatnonzero(data['split'] == 'test')
    rows = [df.loc[data['row_index'][i]].to_dict() for i in indices]
    # Explicit turn lookup; never assume that another game's row is the future.
    lookup = {(r['game_id'], r['turn']): r for r in df.to_dict('records')}
    actual = [[lookup[(r['game_id'], r['turn']+h)] for h in range(1, cfg.FUTURE+1)] for r in rows]
    rng = np.random.default_rng(42)
    baseline, paths, chosen, fallback_count = [], [], [], 0
    for row in rows:
        sequences, probabilities, fallback = uniform_paths(row, cfg.FUTURE)
        baseline.append(probabilities)
        paths.append(sequences)
        chosen.append(sequences[rng.integers(len(sequences))])
        fallback_count += fallback
    predictions = {'Uniform paths': np.array(baseline)}
    choices = {'Uniform paths': np.array(chosen)}
    runs = [('Uniform paths', -1, predictions['Uniform paths'], choices['Uniform paths'])]
    for seed in args.seeds:
        for name, kind, cls in [('RNN', 'rnn', ActionRNN), ('LSTM', 'lstm', ActionLSTM)]:
            probabilities = predict(kind, cls, seed, data['observations'][indices], metadata)
            actions = probabilities.argmax(-1)
            runs.append((name, seed, probabilities, actions))
            if seed == args.seeds[0]:
                predictions[name], choices[name] = probabilities, actions
    records = []
    for name, seed, probabilities, actions in runs:
        for j, index in enumerate(indices):
            records.append(dict(model=name, training_seed=seed, game_seed=int(data['seed'][index]),
                turn=int(data['turn'][index]), **scores(probabilities[j], actions[j],
                    data['actions'][index], rows[j], actual[j])))
    table = pd.DataFrame(records)
    metrics = ['action_accuracy', 'sequence_accuracy', 'action_nll', 'rollout_survival', 'endpoint_accuracy']
    by_game = table.groupby(['model', 'training_seed', 'game_seed'])[metrics].mean()
    by_run = by_game.groupby(['model', 'training_seed']).mean()
    summary = by_run.groupby('model').mean()
    cfg.RESULTS.mkdir(exist_ok=True)
    by_game.to_csv(cfg.RESULTS/'test_by_game.csv')
    by_run.to_csv(cfg.RESULTS/'test_by_run.csv')
    summary.to_csv(cfg.RESULTS/'test_summary.csv')
    table.to_csv(cfg.RESULTS/'test_examples.csv', index=False)
    manifest = dict(metadata=metadata, training_seeds=args.seeds, visualization_training_seed=args.seeds[0],
                    test_windows=len(indices), test_games=len(np.unique(data['seed'][indices])),
                    baseline_fallbacks=fallback_count, baseline_random_seed=42)
    (cfg.RESULTS/'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2))
    examples(data, indices, rows, actual, predictions, choices, paths, df, args.seeds[0])
    report(summary, manifest)
    print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'{len(indices)} test windows; {manifest["test_games"]} held-out games; {fallback_count} baseline fallbacks.')


def report(summary, manifest):
    splits = pd.read_csv(cfg.DATA/'split_summary.csv')
    lines = ['# Action prediction — completed experiment', '',
        f'History: {cfg.HISTORY} observations. Prediction: next {cfg.FUTURE} actions.',
        'Inputs contain only observed state features. Targets start at the final observed turn.', '',
        '## Phase results', '', f'0. Explored {read_recording().game_id.nunique()} supplied solo games (see plot below).',
        '1. Created windows; retained only examples whose recorded future survives the whole horizon.']
    for row in splits.itertuples():
        lines.append(f'   - {row.split}: {row.games} games, {row.windows:,} windows.')
    for number, name in [(2, 'rnn'), (3, 'lstm')]:
        train = pd.read_csv(cfg.RESULTS/f'{name}_training.csv')
        lines.append(f'{number}. Trained {name.upper()}: {int(train.parameters.iloc[0]):,} parameters; '
                     f'seeds {train.seed.tolist()}; best validation epochs {train.best_epoch.tolist()}.')
    lines += ['4. Evaluated on untouched test games and simulated the predicted action sequences.', '',
        '| Model | Action accuracy ↑ | Exact sequence ↑ | Action NLL ↓ | Simulated survival ↑ | Correct endpoint ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for name, r in summary.iterrows():
        lines.append(f'| {name} | {r.action_accuracy:.1%} | {r.sequence_accuracy:.1%} | {r.action_nll:.3f} | '
                     f'{r.rollout_survival:.1%} | {r.endpoint_accuracy:.1%} |')
    lines += ['', 'Metrics are averaged within each game, then over games and training seeds. '
        'Action accuracy uses each step\'s marginal argmax. Exact-sequence and rollout metrics use '
        'the displayed decoding rule: NN per-step argmax; one reproducibly sampled uniform complete path '
        '(every baseline path ties for highest joint probability).', '',
        'The simple NN produces separate action distributions from one observed history. It does not '
        'condition later predictions on earlier predicted actions. Consequently, a predicted sequence '
        'can reverse into the body or hit a wall. Collisions are counted and shown, never repaired.', '',
        'Simulation uses currently known food and does not invent future food spawns. Simulated survival '
        'is survival over the short forecast horizon, not whole-game playing strength. Recorded examples '
        'are conditioned on survival; these results do not measure prediction near death.', '',
        'The source agent implementation is not provided. Neither a benefit from long memory nor LSTM '
        'superiority is guaranteed. Compare history lengths or a current-state-only model to investigate this.', '',
        '## Figures', '', '![Recorded games](step_0_recordings.png)', '',
        '![RNN learning](rnn_learning.png)', '', '![LSTM learning](lstm_learning.png)', '',
        f'Figures use training seed {manifest["visualization_training_seed"]}; middle windows from '
        'the four lowest-seed test games, selected without looking at accuracy. Endpoint heatmaps '
        'come from 200 simulated samples per model and use the same 0–1 scale. Sampled death mass '
        'is shown separately; sampled heatmaps are estimates.', '']
    for number in range(1, 5):
        lines += [f'![Example {number}](example_{number}.png)', '',
                  f'![Action rollout {number}](example_{number}.gif)', '']
    (cfg.RESULTS/'RESULTS.md').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    main()
