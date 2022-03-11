from random import choice
from utils import Direction

class Agent:
    def __init__(self, game):
        self.game = game
        self.game._setAgent(self)
    
    def _move(self):
        return choice(list(Direction)).value
        