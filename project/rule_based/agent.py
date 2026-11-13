from typing import Literal

from gamestate import GameState
from iAgent import IAgent, InfoReturn


class RuleBasedAgent(IAgent):
    def info(self) -> InfoReturn:
        info =  super().info()
        info.color = "#0FF"
        return info
    
    def start(self, gamestate: GameState):
        return super().start(gamestate)

    def move(self, gamestate: GameState) -> Literal['up'] | Literal['right'] | Literal['down'] | Literal['left']:
        return super().move(gamestate)

    def end(self, gamestate: GameState):
        return super().end(gamestate)

    def dry(self):
        return super().dry()