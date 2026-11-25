from math import sqrt
import random
from typing import Dict, List
from enum import Enum
from PIL import Image, ImageDraw

from hisss import BattleSnakeConfig
from hisss.game.state import BattleSnakeState

class Vector:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, value: object):
        if (isinstance(value, Vector)):
            return self.x == value.x and self.y == value.y # && findes ikke :( gg det skal være 'and'
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    """fd stands for From Dictionary"""
    @classmethod
    def fd(cls, d: Dict) -> "Vector":
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

    def map(self) -> List[List[MapType]]:
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
        
    def draw(self, width: int, height: int):
        image = Image.new("RGB", (width, height), "#BDB")
        draw = ImageDraw.Draw(image)
        
        TILE_WIDTH, TILE_HEIGHT = int(width/self.board.width), int(height/self.board.height)

        def tile_coords(x: int, y: int) -> tuple[int, int]:
            return (
                x*TILE_WIDTH,
                y*TILE_HEIGHT
            )

        for x in range(self.board.width):
            for y in range(self.board.height-1, -1, -1):
                _x, _y = tile_coords(x, y)
                draw.rectangle(
                    (
                        _x, # Start x 0
                        _y, # Start y 0
                        TILE_WIDTH + _x, # End x TILE_WIDTH
                        TILE_HEIGHT + _y # End y TILE_HEIGHT
                    ),
                    outline="#716B6B", width=1)
        
        for food in self.board.food:
            x, y = tile_coords(food.x, food.y)
            draw.circle((x + TILE_WIDTH/2, y + TILE_HEIGHT/2), min(TILE_WIDTH / 3, TILE_HEIGHT / 3), "#FBB")

        for snake in self.board.snakes:
            color = "#FFF"
            if snake.id == self.id:
                color = "#FAD"
            else:
                random.seed(snake.id)
                color = random.randint(0, 16581375) # 255^3
            
            for segment in snake.body:
                x, y = tile_coords(segment.x, segment.y)
                draw.rectangle(
                    (
                        x,
                        y,
                        TILE_WIDTH + x,
                        TILE_HEIGHT + y
                    ),
                    fill=color, width=0
                )

        
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)