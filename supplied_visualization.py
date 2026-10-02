"""Instructor-provided figures and animations; no student plotting code needed."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image
import settings as cfg
from supplied_simulation import simulate, sample_sequences

CMAP = LinearSegmentedColormap.from_list('probability', ['#142438', '#268899', '#ffe193'])
NAMES = ['Uniform paths', 'RNN', 'LSTM', 'Recorded future']


def board(ax, row):
    ax.set_facecolor('#142438')
    for x, y in row['body']:
        ax.add_patch(Rectangle((x-.46, y-.46), .92, .92, facecolor='#667b8c', alpha=.7))
    if row['food']:
        food = np.asarray(row['food'])
        ax.scatter(food[:, 0], food[:, 1], s=30, c='#ff6170', zorder=4)
    ax.set(xlim=(-.7, row['width']-.3), ylim=(-.7, row['height']-.3), aspect='equal')
    ax.set_xticks(range(row['width']), minor=True)
    ax.set_yticks(range(row['height']), minor=True)
    ax.grid(which='minor', color='white', alpha=.08)
    ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])


def draw_path(ax, row, actions, trace):
    states = simulate(row, actions)
    board(ax, row)
    ax.plot(trace[:, 0], trace[:, 1], color='#55d3df', lw=2, label='Observed history')
    points = [row['body'][0]]
    for state in states:
        points.append(state['body'][0])
        if not state['alive']:
            break
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], '-o', c='#ffe193', lw=2, ms=5)
    ax.scatter(*points[0], s=75, edgecolor='white', facecolor='#55d3df', zorder=6)
    for step, point in enumerate(points[1:], start=1):
        ax.annotate(str(step), point, xytext=(5, 4), textcoords='offset points', color='white', fontsize=9)
    if not states[-1]['alive']:
        ax.scatter(*points[-1], c='#ff6170', marker='x', s=90, lw=2, clip_on=False, zorder=7)
    return states


def path_probability_maps(row, sequences):
    """Head-position marginals at turns 0..FUTURE from the SAME sampled paths.

    Each frame describes that exact turn, not cumulative visits. Dead paths
    remain in the denominator and contribute to the separate death probability.
    """
    heat = np.zeros((len(sequences[0])+1, row['height'], row['width']))
    deaths = np.zeros(len(heat))
    for actions in sequences:
        for turn, state in enumerate([row] + simulate(row, actions)):
            if state['alive']:
                x, y = state['body'][0]
                heat[turn, y, x] += 1
            else:
                deaths[turn] += 1
    return heat/len(sequences), deaths/len(sequences)


def endpoint_map(row, sequences):
    heat, deaths = path_probability_maps(row, sequences)
    return heat[-1], deaths[-1]


def examples(data, indices, rows, actual, predictions, choices, baseline_paths, df, training_seed):
    seeds = np.unique(data['seed'][indices])[:4]
    for number, seed in enumerate(seeds, start=1):
        locals_ = np.flatnonzero(data['seed'][indices] == seed)
        local = locals_[len(locals_)//2]
        index = indices[local]
        row = rows[local]
        history = df[(df.game_id == row['game_id']) & (df.turn <= row['turn'])].tail(cfg.HISTORY)
        trace = np.asarray([body[0] for body in history.body])
        target = data['actions'][index]
        probabilities = {name: predictions[name][local] for name in NAMES[:-1]}
        probabilities['Recorded future'] = np.eye(4)[target]
        sequences = {name: choices[name][local] for name in NAMES[:-1]}
        sequences['Recorded future'] = target
        rng = np.random.default_rng(900+number)
        samples = {name: sample_sequences(probabilities[name], rng, 200) for name in ['RNN', 'LSTM']}
        all_paths = baseline_paths[local]
        samples['Uniform paths'] = all_paths[rng.integers(len(all_paths), size=200)]
        fig, axes = plt.subplots(3, 4, figsize=(13, 10.5), layout='constrained',
                                 gridspec_kw={'height_ratios': [1, .55, 1]})
        rollouts = {}
        for col, name in enumerate(NAMES):
            rollout = draw_path(axes[0, col], row, sequences[name], trace)
            # Display the actual recorded states rather than simulated future food/body.
            if name == 'Recorded future':
                rollout = actual[local]
            rollouts[name] = rollout
            decoded = ', '.join(cfg.ACTIONS[a] for a in sequences[name])
            axes[0, col].set_title(f'{name}\n{decoded}', fontsize=10)
            image = axes[1, col].imshow(probabilities[name], cmap=CMAP, vmin=0, vmax=1, aspect='auto')
            axes[1, col].set_xticks(range(4), cfg.ACTIONS)
            axes[1, col].set_yticks(range(cfg.FUTURE), [f'Action +{h+1}' for h in range(cfg.FUTURE)])
            for h in range(cfg.FUTURE):
                for a in range(4):
                    axes[1, col].text(a, h, f'{probabilities[name][h,a]:.0%}', ha='center', va='center',
                        fontsize=9, color='#142438' if probabilities[name][h,a]>.6 else 'white')
            if name == 'Recorded future':
                heat = np.zeros((row['height'], row['width']))
                x, y = actual[local][-1]['body'][0]
                heat[y, x] = 1
                death = 0
            else:
                heat, death = endpoint_map(row, samples[name])
            axes[2, col].imshow(heat, origin='lower', cmap=CMAP, vmin=0, vmax=1)
            axes[2, col].set_title(f'Head after {cfg.FUTURE} actions · death {death:.0%}', fontsize=10)
            ax, ay = actual[local][-1]['body'][0]
            axes[2, col].scatter(ax, ay, c='white', marker='x', s=50, lw=2)
            axes[2, col].set_xticks([0,5,10]); axes[2, col].set_yticks([0,5,10])
        fig.suptitle(f'Example {number} · Game {seed}, turn {row["turn"]} · Predict the next {cfg.FUTURE} actions\n'
            'Top: simulated path (cyan = history, yellow = forecast, red × = collision)\n'
            'Middle: action probabilities · Bottom: sampled endpoint probabilities (white × = actual)', fontsize=12)
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.6, label='Probability')
        fig.savefig(cfg.RESULTS/f'example_{number}.png', dpi=140)
        plt.close(fig)
        animate(row, rollouts, samples, number, seed)


def animate(row, rollouts, samples, number, seed):
    maps = {name: path_probability_maps(row, samples[name]) for name in NAMES[:-1]}
    frames = []
    for h in range(cfg.FUTURE+1):
        fig, axes = plt.subplots(2, 4, figsize=(12, 7), layout='constrained')
        for ax, name in zip(axes[0], NAMES):
            state = row if h == 0 else rollouts[name][h-1]
            board(ax, state)
            color = '#ffe193' if state['alive'] else '#ff6170'
            ax.scatter(*state['body'][0], c=color, marker='o' if state['alive'] else 'x', s=90, clip_on=False)
            ax.set_title(name + ('' if state['alive'] else ' · collision/death'), fontsize=11)
        for ax, name in zip(axes[1, :3], NAMES[:-1]):
            heat, deaths = maps[name]
            image = ax.imshow(heat[h], origin='lower', cmap=CMAP, vmin=0, vmax=1)
            ax.set_title(f'Head probability at +{h}\nDeath: {deaths[h]:.0%}', fontsize=11)
            ax.set_xticks([0, 5, 10]); ax.set_yticks([0, 5, 10])
            ax.set_xticks(np.arange(-.5, row['width'], 1), minor=True)
            ax.set_yticks(np.arange(-.5, row['height'], 1), minor=True)
            ax.grid(which='minor', color='white', alpha=.08)
            ax.tick_params(which='minor', length=0)
        # Keep the recorded future as the original concrete animation only.
        key = axes[1, 3]
        key.axis('off')
        colorbar = key.inset_axes([.08, .68, .84, .06])
        fig.colorbar(image, cax=colorbar, orientation='horizontal', ticks=[0, .5, 1],
                     label='Probability of head in cell')
        key.text(.08, .43, f'{len(samples[NAMES[0]])} sampled paths per model\nSame paths across all frames\n\n'
                 'Exact displayed turn;\nnot accumulated visits.\n\nDead paths leave the board;\n'
                 'their probability is shown above.', transform=key.transAxes, va='top', fontsize=10)
        fig.suptitle(f'Game {seed} · starting turn {row["turn"]} · {h}/{cfg.FUTURE} actions executed\n'
                     'Top: one action sequence · Bottom: probabilities across sampled paths\n'
                     'One fixed forecast, simulated step by step; no replanning', fontsize=12)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        plt.close(fig)
    frames[0].save(cfg.RESULTS/f'example_{number}.gif', save_all=True, append_images=frames[1:],
                   duration=[900]+[650]*(len(frames)-2)+[1300], loop=0)
