from math import sqrt
from typing import Dict, List
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, value: "Vector"):
        return self.x == value.x and self.y == value.y # && findes ikke :( gg det skal være 'and'

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    """fd stands for From Dictionary"""
    @classmethod
    def fd(self, d: Dict) -> "Vector":
        return Vector(d["x"], d["y"])

    def distance(self, other: "Vector") -> float:
        return sqrt(abs(other.x - self.x)**2 + abs(other.y - self.y)**2)

class MapType(Enum):
    EMPTY = 1
    SNAKE_HEAD = 2
    SNAKE_BODY = 3
    FOOD = 4
    HAZARD = 5

class SnakeState:
    id: str
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
        self.id = snake["id"]
        self.name = snake["name"]
        self.health = snake["health"]
        self.body = [Vector.fd(vd) for vd in snake["body"]]
        self.head = Vector.fd(snake["head"])
        self.length = snake["length"]
        

class BoardState:
    you: SnakeState
    height: int
    width: int
    food: List[Vector]
    hazards: List[Vector]
    snakes: List[SnakeState]
    size: Vector

    def __init__(self, you: SnakeState, board: Dict):
        self.you = you
        self.height = board["height"]
        self.width = board["width"]
        self.food = [Vector.fd(vd) for vd in board["food"]]
        self.hazards = [Vector.fd(vd) for vd in board["hazards"]]
        self.snakes = [SnakeState(sd) for sd in board["snakes"]]

    def Map(self) -> List[List[MapType]]:
        mapped: List[List[MapType]] = [[MapType.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        for food in self.food:
            mapped[food.y][food.x] = MapType.FOOD
        for hazard in self.hazards:
            mapped[hazard.y][hazard.x] = MapType.HAZARD
        for snake in self.snakes:
            if snake.id != self.you.id:
                mapped[snake.head.y][snake.head.x] = MapType.SNAKE_HEAD
            for segment in snake.body[1:]:
                mapped[segment.y][segment.x] = MapType.SNAKE_BODY
        return mapped

    def MapDanger(self) -> List[List[float]]:
        dangerMapped = np.zeros((self.width, self.height), dtype=float)
        mapped = self.Map()
        for x in range(self.height):
            for y in range(self.width):
                mapType = mapped[y][x]
                if mapType == MapType.SNAKE_HEAD:
                    dangerMapped[y][x] = 1
                elif mapType == MapType.SNAKE_BODY:
                    dangerMapped[y][x] = 1
                elif mapType == MapType.HAZARD:
                    dangerMapped[y][x] = 1
                elif mapType == MapType.FOOD:
                    dangerMapped[y][x] = -0.5
            #TODO: map enemy snake next movepath

        dangerMapped[self.you.head.y][self.you.head.x] = 1.5

        # plt.imshow( dangerMapped , cmap = 'magma' )
        # plt.gca().invert_yaxis()
        # plt.title( "2-D Heat Map" )
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        # plt.colorbar()

        # plt.show()

        return dangerMapped
        
class GameState:
    id: str
    turn: int
    board: BoardState
    you: SnakeState

    def __init__(self, state: Dict) -> None:
        self.id = state["game"]["id"]
        self.turn = state["turn"]
        self.you = SnakeState(state["you"])
        self.board = BoardState(self.you, state["board"])