import heapq
import random
from typing import Callable, Dict, List, Set


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

# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
def a_star(start: Vector, goals: List[Vector], danger_map: List[List[float]]):

    # Inside function define smaller
    def h(current: Vector): 
        return 1
    def d(current: Vector, neighbor: Vector):
        return current.distance(neighbor)

    def GetNeighbors(center: Vector):
        x, y = center.x, center.y
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        neighbors = [(x + dx, y + dy) for dx, dy in deltas]
        neighbors = [(nx, ny) for nx, ny in neighbors
                        if 0 <= nx < len(danger_map) and 0 <= ny < len(danger_map[0]) and danger_map[nx][ny] < 1]
        return neighbors

    # We do the long time. We do the a*
    def reconstruct_path(came_from: Dict[Vector, Vector], current):
        total_path: Set[Vector] = {current}
        while current in came_from.keys:
            current = came_from[current]
            total_path.add(current)
        return total_path[::-1]

    # The set of discovered nodes that may need to be (re-)expanded.
    # Initially, only the start node is known.
    # This is usually implemented as a min-heap or priority queue rather than a hash-set.
    open_set: Set[Vector] = {}
    heapq.heappush(open_set, (0, start))

    # For node n, came_from[n] is the node immediately preceding it on the cheapest path from the start
    # to n currently known.
    came_from: Dict[Vector, Vector] = []

    # For node n, g_score[n] is the currently known cost of the cheapest path from start to n.
    g_score: Dict[Vector, float] = []
    g_score[start] = 0

    # For node n, f_score[n] := g_score[n] + h(n). f_score[n] represents our current best guess as to
    # how cheap a path could be from start to finish if it goes through n.
    f_score: Dict[Vector, float] = []
    f_score[start] = h(start)

    while open_set:
        # This operation can occur in O(Log(N)) time if open_set is a min-heap or a priority queue
        priority, current = heapq.heappop(open_set)
        if current in goals:
            return reconstruct_path(came_from, current)

        open_set.remove(current)
        for neighbor in GetNeighbors(current, 11, 11):
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_g_score is the distance from start to the neighbor through current
            tentative_g_score = g_score[current] + d(current, neighbor)
            if tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h(neighbor)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # Open set is empty but goal was never reached
    return None

def choose_move(game_state: GameState) -> str:
    # A* go here :)
    me = game_state.you
    dangerMap = game_state.board.MapDanger()




    print(f"MOVE {game_state.turn}: {next_move}")
    return next_move
