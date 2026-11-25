from abc import ABC, abstractmethod
from contextlib import closing
from dataclasses import dataclass
import json
from typing import Literal
from PIL import Image, ImageDraw
import hisss
from hisss.cpp.lib import CPP_LIB

from gamestate import GameState

@dataclass
class InfoReturn:
    apiversion: str = "1"
    author: str = "kittenzzz snakeuh"
    color: str = "#FC8EAC"
    head: str = "villain"
    tail: str = "weight"

    def to_json(self):
        return self.__dict__

class IAgent(ABC):
    @abstractmethod
    def info(self) -> InfoReturn:
        return InfoReturn()

    @abstractmethod
    def start(self, gamestate: GameState):
        pass

    @abstractmethod
    def move(self, gamestate: GameState) -> Literal["up", "right", "down", "left"]:
        return "right"

    @abstractmethod
    def end(self, gamestate: GameState):
        pass

    @abstractmethod
    def dry(self):
        CPP_LIB.lib.set_seed(int("1") % 2**31 - 1) # THIS IS THE SEED, THE SEED IS 1
        IMG_WIDTH = 300
        IMG_HEIGHT = int(IMG_WIDTH * 1.4)

        config = hisss.BattleSnakeConfig(4, 1, w=11, h=11, all_actions_legal=True)
        with closing(hisss.BattleSnakeGame(config)) as game:
            frames: list[Image.Image] = []
            while not game.is_terminal():
                gs = GameState(json.loads(hisss.to_battlesnake_json(game, 0)))
                img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
                draw = ImageDraw.Draw(img)
                
                draw.text((5,5), f"Food: {len(gs.board.food)}")
                draw.text((5, 15), "Snakes:")
                si = 0
                for snake in gs.board.snakes:
                    draw.text((8,25 + si*10), f"{(snake.name)} ({snake.id}): Health: {snake.health}, Length: {snake.length}")

                board_img = gs.draw(IMG_WIDTH, IMG_WIDTH)
                img.paste(board_img, (0, IMG_HEIGHT - IMG_WIDTH))

                frames.append(img)
                action = self.move(gs)
                game.step(tuple([hisss.UP if action == "up" else hisss.DOWN if action == "down" else hisss.LEFT if action == "left" else hisss.RIGHT])) 

            frames[0].save("./test.gif", save_all=True, append_images=frames[1:], loop=0)

            