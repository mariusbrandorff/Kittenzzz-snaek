"""Task: encode one Battlesnake observation as a tuple of six small integers.

Return (safe_up, safe_down, safe_left, safe_right, food_x, food_y).
Use supplied safe_directions() for the first four flags (0 or 1).
For each food coordinate use 0=smaller, 1=equal, 2=larger than the head coordinate.
If there is no food, use (1, 1). With several apples select the nearest by
Manhattan distance, breaking ties by x and then y.

Do not put absolute coordinates, whole body lists or turn numbers in the key.
This will massively increase the number of observable states and make training very hard.
We will try this once we introduced deep reinforcement learning next week.
"""

from supplied_environment import safe_directions


def encode_state(observation):
    directions = safe_directions(observation)

    # DONE: access observation['you']['head'] and observation['board']['food'].
    # Foods
    head = (int(observation['you']['head']['x']), int(observation['you']['head']['y']))
    foods = [(int(f["x"]), int(f["y"])) for f in observation['board']['food']]
    food = (1, 1)
    if (len(foods) != 0):
        if (len(foods) == 1):
            food = foods[0]
        else:
            food = sorted(
                    foods,
                    key=lambda f: abs(head[0] - f[0]) + abs(head[1] - f[1])
            )[0]
        food = (
                0 if food[0] < head[0] else 1 if food[0] == head[0] else 2, # x coordinate
                0 if food[1] < head[1] else 1 if food[1] == head[1] else 2 # y coordinate
            )
    
    # DONE: combine safety flags and relative food direction into a tuple.
    return (directions[0], directions[1], directions[2], directions[3], food[0], food[1])


if __name__ == '__main__':
    from supplied_environment import SoloBattlesnake
    observation, _ = SoloBattlesnake().reset(seed=7)
    key = encode_state(observation)
    print('Encoded observation:', key)
    assert len(key) == 6 and all(0 <= k < n for k,n in zip(key, (2,2,2,2,3,3)))
    print('Maximum: 144 encoded states and 576 state-action values.')
