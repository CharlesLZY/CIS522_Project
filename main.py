import argparse

import Game
from Agent import Agent
from BoringAgent import BoringAgent
from SearchAgent import GreedyAgent, AstarAgent

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", type=int, default=16, help="window width")
parser.add_argument("-H", "--height", type=int, default=16, help="window height")
parser.add_argument("-b", "--block_size", type=int, default=20, help="block size")
parser.add_argument("-s", "--speed", type=int, default=50, help="refresh speed")
parser.add_argument("-v", "--verbose", action="store_true", help="print verbose")
parser.add_argument("-seed", type=int, default=0, help="random seed")
parser.add_argument("-a", "--agent", type=str, default="", choices=["", "random", "boring", "greedy", "astar",], help="agent type")

if __name__ == "__main__":

    args = parser.parse_args()

    ### create a game
    game = Game.SnakeGame(Width = args.width, 
                     Height = args.height, 
                     BLOCK_SIZE=args.block_size, 
                     SPEED=args.speed, 
                     VERBOSE=args.verbose,
                     SEED=args.seed)
    
    ### set agent
    if not args.agent:
        game.SPEED = 5
    elif args.agent == "random":
        agent = Agent(game)
    elif args.agent == "boring":
        agent = BoringAgent(game)
    elif args.agent == "greedy":
        agent = GreedyAgent(game)
    elif args.agent == "astar":
        agent = AstarAgent(game)

    while True:
        game._play()