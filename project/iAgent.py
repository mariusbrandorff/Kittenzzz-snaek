from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

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
        pass

    @abstractmethod
    def end(self, gamestate: GameState):
        pass

    @abstractmethod
    def dry(self):
        pass