from math import sqrt
from typing import Dict, List


class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    """fd stands for From Dictionary"""
    @classmethod
    def fd(self, d: Dict) -> "Vector":
        return Vector(d["x"], d["y"])

    def distance(self, other: "Vector") -> float:
        return sqrt(abs(other.x - self.x)**2 + abs(other.y - self.y)**2)

class SnakeState:
    name: str
    health: int
    body: List[Vector]
    head: Vector
    length: int

    # @property
    # def head(self) -> Vector:
    #     return self.body[0]
    
    @property
    def neck(self) -> Vector:
        return self.body[1]

    def __init__(self, snake: Dict) -> None:
        # Iterate over the body converting into vectors
        self.name = snake["name"]
        self.health = snake["health"]
        self.body = [Vector.fd(vd) for vd in snake["body"]]
        self.head = Vector.fd(snake["head"])
        self.length = snake["length"]
        

class BoardState:
    height: int
    width: int
    food: List[Vector]
    hazards: List[Vector]
    snakes: List[SnakeState]
    size: Vector

    def __init__(self, board: Dict):
        self.height = board["height"]
        self.width = board["width"]
        self.food = [Vector.fd(vd) for vd in board["food"]]
        self.hazard = [Vector.fd(vd) for vd in board["hazards"]]
        self.snakes = [SnakeState(sd) for sd in board["snakes"]]

        
class GameState:
    turn: int
    board: BoardState
    you: SnakeState

    def __init__(self, state: Dict) -> None:
        self.turn = state["turn"]
        self.board = BoardState(state["board"])
        self.you = SnakeState(state["you"])