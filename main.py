import argparse

from Game import SnakeGame
from Agent import Agent
from BoringAgent import BoringAgent
from SearchAgent import GreedyAgent, AstarAgent
from QLearningAgent import QLearningAgent

import utils

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", type=int, default=16, help="window width")
parser.add_argument("-H", "--height", type=int, default=16, help="window height")
parser.add_argument("-b", "--block_size", type=int, default=20, help="block size")
parser.add_argument("-s", "--speed", type=int, default=50, help="refresh speed")
parser.add_argument("-v", "--verbose", action="store_true", help="print verbose")
parser.add_argument("-seed", type=int, default=0, help="random seed")
parser.add_argument("-a", "--agent", type=str, default="", choices=["", "random", "boring", "greedy", "astar", "qlearning"], help="agent type")
parser.add_argument("-f", "--file", type=str, help="model file")

if __name__ == "__main__":

    args = parser.parse_args()

    ### create a game
    game = SnakeGame(W = args.width, 
                     H = args.height, 
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
    elif args.agent == "qlearning":
        if args.file:
            agent = QLearningAgent(game, args.file)
        else:
            print("Please specify the model to load. Usage: main.py -a qlearning -f model.pkl")
            exit()

    while True:
        game._play()