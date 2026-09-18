import heapq
import random
from typing import Callable, Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np

from gamestate import GameState, Vector
 
def vector_direction(v1: Vector, v2: Vector) -> str:
    if (v2.x < v1.x): return "left"
    if (v2.x > v1.x): return "right"
    if (v2.y < v1.y): return "down"
    if (v2.y > v1.y): return "up"
    return

def collision(v1: Vector, segments: List[Vector], is_move_safe: Dict):
    for segment in segments:
        if (segment.distance(v1) == 1):
            is_move_safe[vector_direction(v1, segment)] = False

    return is_move_safe

def safe_move(game_state: GameState):
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    my = game_state.you

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = my.head  # Coordinates of your head
    my_neck = my.neck  # Coordinates of your "neck"
    
    # Step 1 - Prevent your Battlesnake from moving out of bounds
    board = game_state.board

    if my_head.y == board.height-1:
        is_move_safe["up"] = False

    if my_head.x == board.width-1:
        is_move_safe["right"] = False

    if my_head.y == 0:
        is_move_safe["down"] = False
        
    if my_head.x == 0:
        is_move_safe["left"] = False

    # Step 2 - Prevent your Battlesnake from colliding with itself
    # my_body = game_state['you']['body']
    is_move_safe = collision(my_head, my.body[1:], is_move_safe)

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    # opponents = game_state['board']['snakes']

    opponents = [
    snake for snake in game_state.board.snakes
    if snake.name != my.name
    ]

    for opponent in opponents:
        is_move_safe = collision(my_head, opponent.body, is_move_safe)

    # Are there any safe moves left? 
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            print(move)
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state.turn}: No safe moves detected! Imma KMS")
        return "down"

    # Choose a random move from the safe ones
    return random.choice(safe_moves)

# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
def a_star(start: Vector, goals: List[Vector], danger_map: List[List[float]]):
    width = len(danger_map)
    height = len(danger_map[0])

    # Inside function define smaller
    def h(current: Vector): 
        return 1
    def d(current: Vector, neighbor: Vector):
        return current.distance(neighbor)

    def GetNeighbors(center: Vector) -> List[Vector]:
        x, y = center.x, center.y
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        neighbors = [(x + dx, y + dy) for dx, dy in deltas]
        neighbors = [Vector(nx, ny) for nx, ny in neighbors
                        if 0 <= nx < width and 0 <= ny < height and danger_map[ny][nx] < 1]
        return neighbors

    # We do the long time. We do the a*
    def reconstruct_path(came_from: Dict[Vector, Vector], current) -> List[Vector]:
        total_path: List[Vector] = [current]
        while current in came_from.keys():
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    # The set of discovered nodes that may need to be (re-)expanded.
    # Initially, only the start node is known.
    # This is usually implemented as a min-heap or priority queue rather than a hash-set.
    open_set: List[Vector] = []
    open_set.append(start)
    # heapq.heappush(open_set, (0, start))

    # For node n, came_from[n] is the node immediately preceding it on the cheapest path from the start
    # to n currently known.
    came_from: Dict[Vector, Vector] = dict()

    # For node n, g_score[n] is the currently known cost of the cheapest path from start to n.
    g_score: Dict[Vector, float] = dict()
    g_score[start] = 0

    # For node n, f_score[n] := g_score[n] + h(n). f_score[n] represents our current best guess as to
    # how cheap a path could be from start to finish if it goes through n.
    f_score: Dict[Vector, float] = dict()
    f_score[start] = h(start)

    while open_set:
        # This operation can occur in O(Log(N)) time if open_set is a min-heap or a priority queue
        current = open_set.pop(-1)
        # priority, current = heapq.heappop(open_set)
        # print(priority, current)
        if current in goals:
            path = reconstruct_path(came_from, current)

            # Graphing
            # dangerMapped = np.zeros((width, height), dtype=float)
            # for y in range(height):
            #     for x in range(width):
            #         v = Vector(x, y)
            #         if (v in f_score):
            #             dangerMapped[y][x] = f_score[v]

            # plt.imshow(dangerMapped, cmap="magma", interpolation="nearest")
            # plt.plot(    [vector.x for vector in path],    [vector.y for vector in path],    color="cyan",    linewidth=3,    marker="o",    label="A* path")
            # plt.gca().invert_yaxis()
            # plt.title("2-D Heat Map")
            # plt.xlabel("x-axis")
            # plt.ylabel("y-axis")
            # plt.xticks(range(width))
            # plt.yticks(range(height))
            # plt.colorbar()
            # plt.legend()
            # plt.show()

            return path

        # open_set.remove((priority, current))
        for neighbor in GetNeighbors(current):
            print(neighbor)
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_g_score is the distance from start to the neighbor through current
            tentative_g_score = g_score[current] + d(current, neighbor)
            if tentative_g_score < (50000000000000000 if neighbor not in g_score else g_score[neighbor]): # Hvis g_score ikke har key, så bruger den 1
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h(neighbor)
                if neighbor not in open_set:
                    open_set.append(neighbor)
                    open_set.sort(reverse = True, key=lambda x: f_score[x]) # Fuckass

    # Open set is empty but goal was never reached
    return None

def choose_move(game_state: GameState) -> str:
    # A* go here :)
    me = game_state.you
    dangerMap = game_state.board.MapDanger()

    path = a_star(me.head, game_state.board.food, dangerMap)
    # print([str(x) for x in path])
    # next_moves = []
    # for i in range(1, len(path)-1):
    #     next_moves.append(vector_direction(path[i-1], path[i]))
    # print(next_moves) # Alle moves printet
    # x_values = [v.x for v in path]
    # y_values = [v.y for v in path]
    # plt.plot(x_values, y_values, "o-")
    # plt.axis("equal")
    # plt.grid(True)
    # plt.show()
    if (path != None):
        next_move = vector_direction(path[0], path[1])
    else:
        next_move = safe_move(game_state)

    print(f"MOVE {game_state.turn}: {next_move}")

    return next_move
