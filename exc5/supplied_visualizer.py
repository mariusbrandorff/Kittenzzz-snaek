"""Supplied board rendering, greedy-policy rollouts and training curves."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import settings as cfg


def board_image(observation, title, subtitle='', reward=None):
    board, snake = observation['board'], observation['you']
    cell, margin, top = 34, 20, 75
    width = max(350, board['width']*cell+2*margin)
    image = Image.new('RGB', (width, top+board['height']*cell+70), '#edf2f7')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=12)
    draw.text((margin, 12), title, fill='#15243b', font=font)
    draw.text((margin, 36), subtitle, fill='#476076', font=small)
    def rectangle(x, y):
        xx, yy = margin+x*cell, top+(board['height']-1-y)*cell
        return (xx+1, yy+1, xx+cell-1, yy+cell-1)
    for y in range(board['height']):
        for x in range(board['width']):
            draw.rectangle(rectangle(x,y), fill='#15243b')
    for point in snake['body'][1:]:
        if 0 <= point['x'] < board['width'] and 0 <= point['y'] < board['height']:
            draw.rectangle(rectangle(point['x'],point['y']), fill='#358e9d')
    for point in board['food']:
        x0,y0,x1,y1 = rectangle(point['x'],point['y'])
        draw.ellipse((x0+7,y0+7,x1-7,y1-7), fill='#ff6375')
    head = snake['head']
    alive = bool(board['snakes'])
    # Clamp a wall collision marker to the border cell so it stays visible.
    x, y = np.clip(head['x'],0,board['width']-1), np.clip(head['y'],0,board['height']-1)
    box = rectangle(int(x),int(y))
    if alive:
        draw.rectangle(box, fill='#ffe193')
    else:
        draw.line((box[0],box[1],box[2],box[3]), fill='#ff6375', width=4)
        draw.line((box[0],box[3],box[2],box[1]), fill='#ff6375', width=4)
    bottom = top+board['height']*cell+12
    draw.text((margin,bottom), f'Turn {observation["turn"]}  |  Health {snake["health"]}  |  Length {snake["length"]}',
              fill='#15243b', font=small)
    if reward is not None:
        draw.text((margin,bottom+22), f'Last reward: {reward:+.2f}', fill='#476076', font=small)
    return image


def animate_runs(runs, path, limit=160):
    """Each run: (title, list of (observation, reward), ending reason).

    Show every frame for short episodes; for long episodes use evenly spaced
    frames including the final one. Header displays the actual turn number.
    """
    count = max(len(states) for _,states,_ in runs)
    indices = np.unique(np.linspace(0,count-1,min(count,limit),dtype=int))
    frames = []
    for index in indices:
        panels = []
        for title, states, reason in runs:
            observation, reward = states[min(index,len(states)-1)]
            subtitle = reason if index >= len(states)-1 else 'Greedy policy' if title != 'Random policy' else 'All actions equally likely'
            panels.append(board_image(observation, title, subtitle, reward))
        frame = Image.new('RGB', (sum(p.width for p in panels), max(p.height for p in panels)), '#edf2f7')
        offset = 0
        for panel in panels:
            frame.paste(panel, (offset,0)); offset += panel.width
        frames.append(frame)
    durations = [110]*len(frames)
    durations[0], durations[-1] = 600, 1200
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    frames[-1].save(path.with_suffix('.png'))


def training_plot(histories, path):
    fig, axes = plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax, key, label in zip(axes,['reward','apples','steps'],['Episode return','Apples eaten','Time steps']):
        for seed, rows in histories.items():
            values = np.array([r[key] for r in rows])
            window = min(100,len(values))
            smooth = np.convolve(values,np.ones(window)/window,mode='valid')
            ax.plot(np.arange(window,len(values)+1),smooth,label=f'Seed {seed}')
        ax.set(xlabel='Training episode',ylabel=label)
    axes[0].legend()
    fig.suptitle('Training with exploration · 100-episode moving averages (not evaluation)')
    fig.savefig(path,dpi=140);plt.close(fig)
